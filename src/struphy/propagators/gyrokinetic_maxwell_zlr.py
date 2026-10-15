"""Propagators of the gauge-invariant gyrokinetic Maxwell model in the zero-Larmor-radius (ZLR) limit,
see :class:`~struphy.models.gyrokinetic_maxwell_zlr.GyrokineticMaxwellZLR`.

The field unknowns are :math:`\\mathbf E \\in H(\\textnormal{curl})` and :math:`\\mathbf B \\in H(\\textnormal{div})`
(the model is gauge invariant, hence formulated in the fields rather than in the potentials).
All propagators share the dielectric matrix :func:`dielectric_matrix` (vacuum permittivity plus ZLR polarization).

The propagators work with tensor-product FE coefficients (no polar splines).
"""

import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.kernels.gc_mag_density_0form import gc_mag_density_0form
from struphy.pic.accumulation.kernels.gk_drift_coupling import gk_drift_coupling
from struphy.pic.accumulation.kernels.gk_epar_coupling import gk_epar_coupling
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.pushing.kernels.gk_push_vpar_efield import gk_push_vpar_efield
from struphy.pic.pushing.kernels.push_gc_bxEstar_explicit_multistage import push_gc_bxEstar_explicit_multistage
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


####################
# Dielectric matrix
####################


class _ZLRDielectricWeight:
    r"""Matrix-valued weight :math:`(1 + c\, n_0/|B_0|^2)\, G^{-1} - c\, n_0/|B_0|^2\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top`
    of the dielectric matrix on the logical cube, with :math:`\hat{\mathbf b}_0^v` the contra-variant components
    of the equilibrium unit magnetic field (such that :math:`\hat{\mathbf e}^{1\top} \hat{\mathbf b}^v_0 = \mathbf E \cdot \mathbf b_0`).

    Callable in the sense of :meth:`~struphy.feec.mass.WeightedMassOperators.create_weighted_mass`
    (returns an array of shape ``(n1, n2, n3, 3, 3)``).
    """

    def __init__(self, domain, equil, polarization: float):
        self._domain = domain
        self._equil = equil
        self._polarization = polarization

    def __call__(self, e1, e2, e3):
        ginv = self._domain.metric_inv(e1, e2, e3, change_out_order=True)
        n_over_B2 = self._equil.n0(e1, e2, e3) / self._equil.absB0(e1, e2, e3) ** 2
        bv = xp.array(
            [
                self._equil.unit_bv_1(e1, e2, e3),
                self._equil.unit_bv_2(e1, e2, e3),
                self._equil.unit_bv_3(e1, e2, e3),
            ]
        )
        # bb[..., m, n] = bv[m] * bv[n]
        bb = bv[:, None] * bv[None, :]
        bb = xp.moveaxis(bb, (0, 1), (-2, -1))

        pol = self._polarization * n_over_B2[..., None, None]
        return (1.0 + pol) * ginv - pol * bb


def dielectric_matrix(mass_ops: WeightedMassOperators, polarization: float) -> WeightedMassOperator:
    r"""Dielectric matrix of the ZLR gyrokinetic Maxwell model,

    .. math::

        \mathbb A = \mathbb M^1 + \sum_s \alpha_s^2 \left(\mathbb M^1_{\textnormal{MHDeq}} - \mathbb M^{1,\parallel}_{\textnormal{MHDeq}}\right)
        = \int \vec{\Lambda}^1 \left[ \left(1 + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2}\right) G^{-1} - \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top \right] \vec{\Lambda}^{1\top} \sqrt g\, \textnormal d \boldsymbol \eta\,,

    i.e. the weak form of :math:`\mathbf I + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \Pi_\perp` (vacuum permittivity plus
    the polarization of the second-order Hamiltonian in the ZLR limit), assembled as one
    :class:`~struphy.feec.mass.WeightedMassOperator`.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        The weighted mass operators of the simulation (provide domain and equilibrium).

    polarization : float
        The coefficient :math:`\sum_s \alpha_s^2` of the polarization term (species sum).

    Returns
    -------
    The assembled operator; it is cached in ``mass_ops`` and shared between the propagators.
    """
    cache = mass_ops.__dict__.setdefault("_gk_dielectric_matrices", {})
    key = float(polarization)
    if key not in cache:
        if key == 0.0:
            cache[key] = mass_ops.M1
        else:
            weight = _ZLRDielectricWeight(mass_ops.domain, mass_ops.eq_mhd, key)
            cache[key] = mass_ops.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(weight, "sqrt_g"),
                name="M1_dielectric_zlr",
                assemble=True,
            )
    return cache[key]


def _species_polarization(species: tuple[PICVariable, ...]) -> float:
    r"""The coefficient :math:`\sum_s \alpha_s^2` of the given species."""
    return float(sum(var.species.equation_params.alpha**2 for var in species))


####################################
# Full magnetic field (B_0 + B_tilde)
####################################


class _FullBFieldMixin:
    r"""Tensor-product coefficients of the full magnetic field :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`
    for the kernels: the parallel field :math:`B_\parallel = |B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})`
    (``_B_dot_b``, 0-form), its gradient (``_grad_b_full``, 1-form) and the field itself (``_b2_full``, 2-form).

    The vectors are persistent (the kernels hold their data arrays) and updated in place by :meth:`_update_full_b`.
    If ``b_tilde`` is None they are the equilibrium quantities.
    """

    def _allocate_full_b(self, b_tilde: FEECVariable | None):
        self._b_tilde_var = b_tilde

        self._gradB1 = self.projected_equil.gradB1
        self._absB0 = self.projected_equil.absB0
        self._b2_eq = self.projected_equil.b2

        # the kernels need tensor-product coefficients: transposed extraction operators
        # PolarVector --> Stencil-/BlockVector (identity maps in case of no polar splines)
        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._E1T = self.derham.extraction_ops["1"].transpose()
        self._E2T = self.derham.extraction_ops["2"].transpose()

        if b_tilde is not None:
            self._B_dot_b = self.derham.V0.zeros()
            self._grad_b_full = self.derham.V1.zeros()
            self._b2_full = self.derham.V2.zeros()

            self._PB = getattr(self.basis_ops, "PB")
            self._B_dot_b_pol = self._PB.codomain.zeros()
            self._grad_b_pol = self.derham.grad.codomain.zeros()

            self._update_full_b()
        else:
            self._B_dot_b = self._absB0
            self._grad_b_full = self._gradB1
            self._b2_full = self._b2_eq

    def _update_full_b(self):
        """Parallel magnetic perturbation, its gradient and the perturbation (plus equilibrium) in the tensor-product spaces."""
        if self._b_tilde_var is None:
            return

        b_tilde = self._b_tilde_var.spline.vector

        B_dot_b_pol = self._PB.dot(b_tilde, out=self._B_dot_b_pol)
        B_dot_b = self._E0T.dot(B_dot_b_pol, out=self._B_dot_b)
        B_dot_b += self._absB0
        B_dot_b.update_ghost_regions()

        grad_b_pol = self.derham.grad.dot(B_dot_b_pol, out=self._grad_b_pol)
        grad_b_full = self._E1T.dot(grad_b_pol, out=self._grad_b_full)
        grad_b_full += self._gradB1
        grad_b_full.update_ghost_regions()

        b2_full = self._E2T.dot(b_tilde, out=self._b2_full)
        b2_full += self._b2_eq
        b2_full.update_ghost_regions()


##############
# Propagators
##############


class MaxwellGyroMagnetisation(Propagator):
    r"""FEEC discretization of Maxwell's equations with the ZLR dielectric and the gyrocentre magnetisation:
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \Pi_\perp \right)\frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x
        - \int_{\Omega} \left(\mathbf B + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \,\textrm d v_\parallel \textrm d \mu \right) \cdot \nabla \times \mathbf F \,\textrm d \mathbf x &= 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E &= 0\,.

    The curl term on the right is the curl of the intrinsic magnetisation :math:`-\mu \mathbf b_0` of the gyrocentres.
    In matrix form (:math:`\mathbb A` from :func:`~struphy.propagators.gyrokinetic_maxwell_zlr.dielectric_matrix`):

    .. math::

        \mathbb A \dot{\mathbf e} = \mathbb C^\top \mathbb M^2 \mathbf b + \mathbb C^\top (\mathcal P^b)^\top \sum_s \alpha_s^2 \mathbf m_s\,,
        \qquad \dot{\mathbf b} = - \mathbb C \mathbf e\,,
        \qquad \mathbf m_s = \sum_p w_p \mu_p \vec \Lambda^0(\boldsymbol \eta_p)\,,

    where :math:`\mathcal P^b` is the basis projection ``PB`` (:math:`\mathbf b \mapsto \Pi^0(\mathbf b_0 \cdot \mathbf B)`),
    such that the source is the exact transpose of the map :math:`\mathbf b \mapsto \sum_p w_p \mu_p B_\parallel(\boldsymbol \eta_p)`
    of the particle energy: the system is Hamiltonian and the implicit mid-point rule conserves the energy exactly
    (for fixed markers).

    Time discretization: Crank-Nicolson (implicit mid-point), system size reduction via
    :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (the source enters the right-hand side).
    """

    class Variables:
        """Container for the variables advanced by :class:`MaxwellGyroMagnetisation`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        b : FEECVariable
            Magnetic field (perturbation) in H(div) space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, species: tuple[PICVariable, ...] = ()):
        """
        Parameters
        ----------
        species : tuple[PICVariable]
            The gyrocentre species (``"Particles5D"``) whose magnetisation enters Ampère's law
            and whose :math:`\\alpha_s^2` enter the polarization. Empty means vacuum Maxwell.
        """
        self.variables = self.Variables()
        for var in species:
            assert isinstance(var, PICVariable)
            assert var.space == "Particles5D"
        self.species = tuple(species)

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MaxwellGyroMagnetisation`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # dielectric matrix and the other operators
        self._polarization = _species_polarization(self.species)
        self._A = dielectric_matrix(self.mass_ops, self._polarization)
        M2 = self.mass_ops.M2
        curl = self.derham.curl

        # preconditioner for A
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self._A)

        # block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        self._B = -1 / 2 * curl.T @ M2
        self._C = 1 / 2 * curl

        self._schur_solver = SchurSolver(
            self._A,
            self._B @ self._C,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # magnetisation: accumulated 0-form per species, then C^T PB^T
        self._accums = []
        self._alpha2 = []
        for var in self.species:
            self._accums += [
                AccumulatorVector(
                    var.particles,
                    "H1",
                    gc_mag_density_0form,
                    self.mass_ops,
                    self.domain.args_domain,
                )
            ]
            self._alpha2 += [var.species.equation_params.alpha**2]

        self._PBT = self.basis_ops.PB.T
        self._curlT = curl.T

        # pre-allocate arrays
        self._byn = self._B.codomain.zeros()
        self._m_pol = self._PBT.domain.zeros()
        self._s2 = self._PBT.codomain.zeros()
        self._s1 = self._curlT.codomain.zeros()
        self._e_tmp1 = self.variables.e.spline.vector.space.zeros()
        self._e_tmp2 = self.variables.e.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # B*b^n
        byn = self._B.dot(bn, out=self._byn)

        # magnetisation source: Byn <- B*b^n - s/2 with s = C^T PB^T sum_s alpha_s^2 m_s
        if len(self._accums) > 0:
            self._m_pol *= 0.0
            for accum, alpha2 in zip(self._accums, self._alpha2):
                accum(alpha2)
                self._m_pol += accum.vectors[0]

            s2 = self._PBT.dot(self._m_pol, out=self._s2)
            s1 = self._curlT.dot(s2, out=self._s1)
            s1 *= 0.5
            byn -= s1

        # solve for new e coeffs
        en1, info = self._schur_solver(en, byn, dt, out=self._e_tmp1)

        # new b coeffs
        _e = en.copy(out=self._e_tmp2)
        _e += en1
        bn1 = self._C.dot(_e, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        diffs = self.update_feec_variables(e=en1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for MaxwellGyroMagnetisation: {info['success']}")
            logger.info(f"Iterations for MaxwellGyroMagnetisation: {info['niter']}")
            logger.info(f"Maxdiff e for MaxwellGyroMagnetisation: {diffs['e']}")
            logger.info(f"Maxdiff b for MaxwellGyroMagnetisation: {diffs['b']}")
            logger.info("")


class GyroEparCoupling(Propagator, _FullBFieldMixin):
    r"""Parallel coupling of one gyrocentre species to the electric field:
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`f` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \Pi_\perp \right)\frac{\partial \mathbf{E}}{\partial t} \cdot \mathbf{F}\,\textrm d \mathbf x
        &= - \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int f\, v_\parallel \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf{F} \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x \qquad \forall \, \mathbf{F} \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial f}{\partial t} + \frac{1}{\varepsilon}\, \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf{E}\, \frac{\partial f}{\partial v_\parallel} &= 0\,,

    where :math:`\mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon v_\parallel \nabla \times \mathbf b_0` and
    :math:`B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*`. For each marker, in logical coordinates,

    .. math::

        \mathbb A \dot{\mathbf e} = -\frac{\alpha^2}{\varepsilon} \sum_p w_p v_{\parallel,p}\, \vec \Lambda^1(\boldsymbol \eta_p) \cdot \mathbf c_p\,,
        \qquad \dot v_{\parallel,p} = \frac{1}{\varepsilon}\, \mathbf c_p \cdot \vec \Lambda^{1\top}(\boldsymbol \eta_p)\, \mathbf e\,,
        \qquad \mathbf c_p = \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p)\,,

    with :math:`\mathbf c_p` frozen at :math:`v_{\parallel,p}^n`, such that the system is linear in :math:`(\mathbf e, v_\parallel)`.

    Time discretization: Crank-Nicolson (implicit mid-point), system size reduction via
    :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (same structure as
    :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`).
    The accumulation and push kernels evaluate the identical :math:`\mathbf c_p`, which gives exact energy conservation.
    """

    class Variables:
        """Container for the variables advanced by :class:`GyroEparCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        var : PICVariable
            Gyrocentre particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._var: PICVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def var(self) -> PICVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._var = new

    def __init__(self, b_tilde: FEECVariable = None, polarization: float = 0.0):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hdiv"`` space entering :math:`\\mathbf B^*`.
            If ``None``, only the equilibrium field is used.
        polarization : float
            The coefficient :math:`\\sum_s \\alpha_s^2` (all species) of the dielectric matrix.
        """
        self.variables = self.Variables()
        self.b_tilde = b_tilde
        self.polarization = polarization

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`GyroEparCoupling`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # scaling factors
        alpha = self.variables.var.species.equation_params.alpha
        self._epsilon = self.variables.var.species.equation_params.epsilon
        self._c_mat = -(alpha**2) / (4.0 * self._epsilon**2)
        self._c_vec = alpha**2 / (2.0 * self._epsilon)

        # dielectric matrix
        self._A = dielectric_matrix(self.mass_ops, self.polarization)

        # magnetic field (B_0 + B_tilde)
        self._allocate_full_b(self.b_tilde)
        curl_unit_b2 = self.projected_equil.curl_unit_b2
        curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # accumulator
        particles = self.variables.var.particles
        self._accum = Accumulator(
            particles,
            "Hcurl",
            gk_epar_coupling,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
        )
        self._args_accum = (
            self._epsilon,
            self._b2_full[0]._data,
            self._b2_full[1]._data,
            self._b2_full[2]._data,
            curl_unit_b2[0]._data,
            curl_unit_b2[1]._data,
            curl_unit_b2[2]._data,
            self._B_dot_b._data,
            curl_unit_b_dot_b0._data,
        )

        # buffers
        e_space = self.variables.e.spline.vector.space
        self._e_new = e_space.zeros()
        self._e_mid = e_space.zeros()
        self._byn = e_space.zeros()
        self._e_mid_tp = self.derham.V1.zeros()

        # preconditioner for A
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self._A)

        # Schur solver (BC is replaced at each call by the scaled accumulated matrix)
        self._schur_solver = SchurSolver(
            self._A,
            self._accum.operators[0],
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # pusher of the parallel velocity in the mid-point electric field
        args_kernel = (
            self.derham.args_derham,
            self._epsilon,
            self._b2_full[0]._data,
            self._b2_full[1]._data,
            self._b2_full[2]._data,
            curl_unit_b2[0]._data,
            curl_unit_b2[1]._data,
            curl_unit_b2[2]._data,
            self._B_dot_b._data,
            curl_unit_b_dot_b0._data,
            self._e_mid_tp[0]._data,
            self._e_mid_tp[1]._data,
            self._e_mid_tp[2]._data,
        )

        self._pusher = Pusher(
            particles,
            gk_push_vpar_efield,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector

        # magnetic field and accumulation at v_parallel^n
        self._update_full_b()
        self._accum(*self._args_accum)

        # update Schur solver: BC = -alpha^2/(4 eps^2) * sum_p w_p Lambda c c^T Lambda^T
        self._schur_solver.BC = self._accum.operators[0]
        self._schur_solver.BC *= self._c_mat

        # Byn = alpha^2/(2 eps) * sum_p w_p v_p c_p Lambda
        self._byn *= 0.0
        self._byn += self._accum.vectors[0]
        self._byn *= self._c_vec

        # new e coeffs
        en1, info = self._schur_solver(en, self._byn, dt, out=self._e_new)

        # mid-point e-field in the tensor-product space (the pusher kernel reads the ghost regions)
        e_mid = en.copy(out=self._e_mid)
        e_mid += en1
        e_mid *= 0.5
        e_mid_tp = self._E1T.dot(e_mid, out=self._e_mid_tp)
        e_mid_tp.update_ghost_regions()

        # update parallel velocities
        self._pusher(dt)

        # update weights in case of control variate
        if self.variables.var.species.weights_params.control_variate:
            self.variables.var.particles.update_weights()

        # write new coeffs into self.variables
        max_de = self.update_feec_variables(e=en1)["e"]

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for GyroEparCoupling: {info['success']}")
            logger.info(f"Iterations for GyroEparCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for GyroEparCoupling: {max_de}")
            logger.info("")


class _EquilibriumWithFullB:
    """Proxy of a projected equilibrium whose ``b2`` is a given (persistent) vector holding :math:`\\mathbf B_0 + \\tilde{\\mathbf B}`;
    all other attributes are those of the projected equilibrium."""

    def __init__(self, projected_equil, b2_full):
        self._projected_equil = projected_equil
        self._b2_full = b2_full

    @property
    def b2(self):
        return self._b2_full

    def __getattr__(self, name):
        return getattr(self._projected_equil, name)


class PushGuidingCenterParallelFullB(PushGuidingCenterParallel):
    r"""Same as :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`
    (without electric potential), but with the magnetic perturbation in the full :math:`\mathbf B^*`:

    .. math::

        \mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
        B^*_\parallel = |B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B}) + \varepsilon v_\parallel\, \mathbf b_0 \cdot \nabla \times \mathbf b_0\,,

    which adds the magnetic flutter (streaming along the perturbed field lines) to the parallel motion
    (the parent class uses :math:`\tilde{\mathbf B}` only in :math:`B^*_\parallel` and in :math:`\mathbf E^* = -\varepsilon \mu \nabla B_\parallel`).
    The vector ``b2`` of the projected equilibrium seen by the kernels is replaced by a persistent vector
    holding :math:`\mathbf B_0 + \tilde{\mathbf B}` (tensor-product coefficients), updated before every push.
    """

    def __init__(self, b_tilde: FEECVariable):
        """
        Parameters
        ----------
        b_tilde : FEECVariable
            Magnetic perturbation in ``"Hdiv"`` space.
        """
        assert isinstance(b_tilde, FEECVariable)
        assert b_tilde.space == "Hdiv"
        super().__init__(phi=None, b_tilde=b_tilde)

    @property
    def projected_equil(self):
        equil = super().projected_equil
        if getattr(self, "_b2_full", None) is None:
            return equil
        return _EquilibriumWithFullB(equil, self._b2_full)

    @profile
    def allocate(self):
        self._E2T_full = self.derham.extraction_ops["2"].transpose()
        self._b2_eq_full = super().projected_equil.b2
        self._b2_full = self.derham.V2.zeros()
        self._update_b2_full()

        super().allocate()

    def _update_b2_full(self):
        """Full magnetic field (equilibrium + perturbation) as 2-form in the tensor-product space."""
        b2_full = self._E2T_full.dot(self.b_tilde.spline.vector, out=self._b2_full)
        b2_full += self._b2_eq_full
        b2_full.update_ghost_regions()

    @profile
    def __call__(self, dt):
        self._update_b2_full()
        super().__call__(dt)


class GyroDriftCoupling(Propagator, _FullBFieldMixin):
    r"""Perpendicular (drift) coupling of all gyrocentre species to the electric field:
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`f_s` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_0}{|B_0|^2} \Pi_\perp \right)\frac{\partial \mathbf{E}}{\partial t} \cdot \mathbf{F}\,\textrm d \mathbf x
        &= - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int_{\Omega} \int f_s\, \frac{\mathbf E^* \times \mathbf b_0}{B^*_\parallel} \cdot \mathbf{F} \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x \qquad \forall \, \mathbf{F} \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial f_s}{\partial t} + \frac{\mathbf E^* \times \mathbf b_0}{B^*_\parallel} \cdot \nabla f_s &= 0\,,
        \qquad \mathbf E^* = \mathbf E - \varepsilon_s \mu \nabla B_\parallel\,,

    where :math:`B_\parallel = |B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})`.
    With the markers at :math:`\boldsymbol \eta^n`, the :math:`E \times B` current is an anti-symmetric matrix
    :math:`\mathbb Q = \sum_s \frac{\alpha_s^2}{\varepsilon_s} \sum_p w_p \vec \Lambda^1 \frac{\epsilon_{\mu\nu\lambda} \hat b_{0,\lambda}}{\sqrt g \hat B^*_\parallel} \vec \Lambda^{1\top}`
    (hence exchanges no energy, :math:`\mathbf E \cdot (\mathbf E \times \mathbf b_0) = 0`) and the
    :math:`\nabla B` current a vector :math:`\mathbf j = \sum_s \frac{\alpha_s^2}{\varepsilon_s} \sum_p w_p \dot{\boldsymbol \eta}_{\nabla B,p} \vec \Lambda^1`:

    .. math::

        \left(\mathbb A + \frac{\Delta t}{2} \mathbb Q \right) \mathbf e^{n+1} = \left(\mathbb A - \frac{\Delta t}{2} \mathbb Q \right) \mathbf e^{n} - \Delta t\, \mathbf j\,.

    The markers are then pushed with an explicit Runge-Kutta scheme
    (:func:`~struphy.pic.pushing.kernels.push_gc_bxEstar_explicit_multistage.push_gc_bxEstar_explicit_multistage_kernels.push_gc_bxEstar_explicit_multistage`)
    in the mid-point field :math:`(\mathbf e^n + \mathbf e^{n+1})/2`.
    The energy exchange of the :math:`\nabla B` current with the particle energy :math:`\mu B_\parallel` is consistent to second order.

    All species are handled in one propagator: a single species' :math:`E \times B` current, balanced only by the
    polarization, gives a spurious stiff mode near the cyclotron frequency.
    """

    class Variables:
        """Container for the variables advanced by :class:`GyroDriftCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        """

        def __init__(self):
            self._e: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

    def __init__(
        self,
        species: tuple[PICVariable, ...] = (),
        b_tilde: FEECVariable = None,
        polarization: float = 0.0,
    ):
        """
        Parameters
        ----------
        species : tuple[PICVariable]
            The gyrocentre species (``"Particles5D"``) that are pushed and whose drift currents enter Ampère's law.
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hdiv"`` space entering :math:`B_\\parallel`.
            If ``None``, only the equilibrium field is used.
        polarization : float
            The coefficient :math:`\\sum_s \\alpha_s^2` (all species) of the dielectric matrix.
        """
        self.variables = self.Variables()
        for var in species:
            assert isinstance(var, PICVariable)
            assert var.space == "Particles5D"
        self.species = tuple(species)
        self.b_tilde = b_tilde
        self.polarization = polarization

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`GyroDriftCoupling`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            General iterative solver for the (non-symmetric) linear system.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.

        butcher : ButcherTableau, default=None
            Butcher tableau of the explicit marker push. If ``None``, defaults to ``ButcherTableau(algo="rk2")``.

        mpi_sort : LiteralOptions.OptsMPIsort, default="last"
            MPI sorting policy for particle exchange (the markers are sorted after the full step).
        """

        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None
        mpi_sort: LiteralOptions.OptsMPIsort = "last"

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsGenSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.mpi_sort, LiteralOptions.OptsMPIsort)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.butcher is None:
                self.butcher = ButcherTableau(algo="rk2")

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # dielectric matrix
        self._A = dielectric_matrix(self.mass_ops, self.polarization)

        # magnetic field (B_0 + B_tilde)
        self._allocate_full_b(self.b_tilde)
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # buffers
        e_space = self.variables.e.spline.vector.space
        self._e_new = e_space.zeros()
        self._e_mid = e_space.zeros()
        self._rhs = e_space.zeros()
        self._j = e_space.zeros()
        self._e_mid_tp = self.derham.V1.zeros()

        # accumulators and pushers, one per species
        butcher = self.options.butcher
        self._accums = []
        self._args_accum = []
        self._pushers = []
        for var in self.species:
            particles = var.particles
            alpha = var.species.equation_params.alpha
            epsilon = var.species.equation_params.epsilon

            self._accums += [
                Accumulator(
                    particles,
                    "Hcurl",
                    gk_drift_coupling,
                    self.mass_ops,
                    self.domain.args_domain,
                    add_vector=True,
                    symmetry="asym",
                )
            ]
            self._args_accum += [
                (
                    epsilon,
                    alpha**2 / epsilon,
                    unit_b1[0]._data,
                    unit_b1[1]._data,
                    unit_b1[2]._data,
                    self._grad_b_full[0]._data,
                    self._grad_b_full[1]._data,
                    self._grad_b_full[2]._data,
                    self._B_dot_b._data,
                    curl_unit_b_dot_b0._data,
                )
            ]

            args_kernel = (
                self.derham.args_derham,
                epsilon,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                self._grad_b_full[0]._data,
                self._grad_b_full[1]._data,
                self._grad_b_full[2]._data,
                self._B_dot_b._data,
                curl_unit_b_dot_b0._data,
                self._e_mid_tp[0]._data,
                self._e_mid_tp[1]._data,
                self._e_mid_tp[2]._data,
                True,
                butcher.a_stage,
                butcher.b,
                butcher.c,
            )

            self._pushers += [
                Pusher(
                    particles,
                    push_gc_bxEstar_explicit_multistage,
                    args_kernel,
                    self.domain.args_domain,
                    alpha_in_kernel=1.0,
                    n_stages=butcher.n_stages,
                    mpi_sort=self.options.mpi_sort,
                    pushes_eta=True,
                )
            ]

        # preconditioner and linear solver
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self._A)

        self._A_inv = inverse(
            self._A,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector

        # magnetic field and accumulation at eta^n
        self._update_full_b()
        for accum, args in zip(self._accums, self._args_accum):
            accum(*args)

        # (A + dt/2 Q) e^{n+1} = (A - dt/2 Q) e^n - dt j
        lhs = self._A
        rhs_op = self._A
        self._j *= 0.0
        for accum in self._accums:
            lhs = lhs + dt / 2 * accum.operators[0]
            rhs_op = rhs_op - dt / 2 * accum.operators[0]
            self._j += accum.vectors[0]
        self._j *= dt

        rhs = rhs_op.dot(en, out=self._rhs)
        rhs -= self._j

        self._A_inv.linop = lhs
        with ProfileManager.profile_region(self._solve_region, functions=[self._A_inv.solve]):
            en1 = self._A_inv.solve(rhs, out=self._e_new)
        info = self._A_inv._info

        # mid-point e-field in the tensor-product space (the pusher kernel reads the ghost regions)
        e_mid = en.copy(out=self._e_mid)
        e_mid += en1
        e_mid *= 0.5
        e_mid_tp = self._E1T.dot(e_mid, out=self._e_mid_tp)
        e_mid_tp.update_ghost_regions()

        # push markers
        for var, pusher in zip(self.species, self._pushers):
            pusher(dt)
            if var.species.weights_params.control_variate:
                var.particles.update_weights()

        # write new coeffs into self.variables
        max_de = self.update_feec_variables(e=en1)["e"]

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for GyroDriftCoupling: {info['success']}")
            logger.info(f"Iterations for GyroDriftCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for GyroDriftCoupling: {max_de}")
            logger.info("")
