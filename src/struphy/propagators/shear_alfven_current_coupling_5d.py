import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
)
from struphy.feec.utilities import LocalRotationMatrix
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.kernels.gc_mag_density_0form import gc_mag_density_0form
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenCurrentCoupling5D(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \int \left(\tilde{\mathbf B} - \frac{A_\textnormal{h}}{A_b} \iint f^\text{vol} \mu \mathbf{b}_0\textnormal{d} \mu \textnormal{d} v_\parallel \right) \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf V) \, \textnormal{d} \mathbf{x} \quad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,, \,,
                \\
                &\frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times (\tilde{\mathbf B} \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^{2,n})^{-1} \mathcal {T^2}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^2} (\mathbb M^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            {\mathbb M^{2,n}}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k) \right)
        \end{bmatrix} \,,

    where
    :math:`\mathcal{T}^2 = \hat \Pi^1 \left[\frac{\mathbf B^2}{\sqrt{g}} \times \vec \Lambda^2\right]` with the total
    field :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` (updated in each step) if ``nonlinear``, else the
    equilibrium field :math:`\mathbf B_0`, and
    :math:`\mathbb M^{2,n}` is a :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`\rho_\text{eq}`, the MHD equilibirum density.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenCurrentCoupling5D`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, energetic_ions: PICVariable = None):
        """
        Parameters
        ----------
        energetic_ions : PICVariable, default=None
            Energetic-ion particle distribution (``"Particles5D"`` space) providing
            the current source for the shear Alfvén wave coupling.
        """
        self.variables = self.Variables()
        self.energetic_ions = energetic_ions

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`ShearAlfvenCurrentCoupling5D`.

        Parameters
        ----------
        ep_scale : float, default=None
            Factor of the energetic-particle contributions (accumulation): the mass ratio
            :math:`A_\textnormal{h}/A_\textnormal{b}` of the energetic ions and the MHD bulk if ``None``.
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for velocity variable ``u``.
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver.
        precond : LiteralOptions.OptsShearAlfvenPrecond, default="StiffnessPreconditioner"
            Preconditioner of the linear solve: of the mass-matrix block (``"MassMatrixPreconditioner"``,
            ``"PolarMassMatrixPreconditioner"``) or the Kronecker (fast diagonalization) preconditioner of the whole
            Schur operator (``"StiffnessPreconditioner"``, the default,
            :class:`~struphy.feec.preconditioner.ShearAlfvenSchurPreconditioner`, built from the equilibrium part
            :math:`\mathbb T` of the basis projection operator; with ``nonlinear=True`` it approximates the Schur
            operator with :math:`\mathbb T + \mathbb T^B`, since :math:`|\tilde{\mathbf B}| \ll |\mathbf B_0|`). The
            Kronecker preconditioner needs ``algo="implicit"``, ``solver="pcg"``, ``u_space="Hdiv"``, tensor-product
            splines (no polar splines) and the global commuting projectors; otherwise it falls back (with a warning) to
            ``"MassMatrixPreconditioner"`` (``"PolarMassMatrixPreconditioner"`` with polar splines). ``"MultiGrid"`` is
            not available here.
        exb_correction : bool, default=True
            For ``precond="StiffnessPreconditioner"``: add the auxiliary space correction of the
            :math:`\mathbf E \times \mathbf B` flows, see
            :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`.
        solver_params : SolverParameters, default=None
            Solver controls.
        filter_params : FilterParameters, default=None
            Filter of the accumulated (particle-to-grid) vectors; defaults to ``FilterParameters()`` (no filter).
            The three-point (binomial) filter ``FilterParameters(use_filter="three_point")`` reduces the noise,
            but the propagator then no longer conserves the energy exactly (the filter is applied to the
            accumulated vectors only, not to the fields seen by the particles).
        butcher : ButcherTableau, default=None
            Explicit Runge-Kutta tableau for ``algo="explicit"``.
        nonlinear : bool, default=True
            If ``True``, the basis projection operator :math:`\mathcal T^2` uses the total magnetic field
            :math:`\mathbf B_0 + \tilde{\mathbf B}` (one operator, re-weighted in each step), else the equilibrium
            field :math:`\mathbf B_0` only.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        ep_scale: float = None
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsShearAlfvenPrecond = "StiffnessPreconditioner"
        exb_correction: bool = True
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        butcher: ButcherTableau = None
        nonlinear: bool = True

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsShearAlfvenPrecond)
            assert self.precond != "MultiGrid", 'precond="MultiGrid" is not available for ShearAlfvenCurrentCoupling5D.'
            if self.precond == "StiffnessPreconditioner" and (
                self.algo != "implicit" or self.solver != "pcg" or self.u_space != "Hdiv"
            ):
                logger.warning(
                    'precond="StiffnessPreconditioner" needs algo="implicit", solver="pcg" and u_space="Hdiv"; '
                    'falling back to precond="MassMatrixPreconditioner".'
                )
                self.precond = "MassMatrixPreconditioner"
            assert self.ep_scale is None or isinstance(self.ep_scale, float)
            assert isinstance(self.nonlinear, bool)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

            if self.algo == "explicit" and self.butcher is None:
                self.butcher = ButcherTableau()

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
        # factor of the energetic-particle contributions (default: mass ratio A_h/A_b)
        self._ep_scale = self.options.ep_scale
        if self._ep_scale is None:
            self._ep_scale = self.energetic_ions.species.mass_number / self.variables.u.species.mass_number

        self._u_form = self.derham.space_to_form[self.options.u_space]

        # call operatros
        id_M = "M" + self._u_form + "n"
        id_T = "T" + self._u_form

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)
        M2 = self.mass_ops.M2
        curl = self.derham.curl
        PB = getattr(self.basis_ops, "PB")

        # define Accumulator and arguments
        self._ACC = AccumulatorVector(
            self.energetic_ions.particles,
            "H1",
            gc_mag_density_0form,
            self.mass_ops,
            self.domain.args_domain,
            filter_params=self.options.filter_params,
        )

        # Preconditioner (the Kronecker preconditioner of the Schur operator depends on dt, built in prepare_time_step)
        self._schur_precond_dt = None
        self._precond = self.options.precond
        if self._precond == "StiffnessPreconditioner" and not preconditioner.ShearAlfvenSchurPreconditioner.supported(
            self.derham
        ):
            self._precond = "PolarMassMatrixPreconditioner" if self.derham.polar_splines else "MassMatrixPreconditioner"
            logger.warning(
                'precond="StiffnessPreconditioner" needs tensor-product splines and the global commuting projectors; '
                f'falling back to precond="{self._precond}".'
            )
        if self._precond == "StiffnessPreconditioner":
            # Kronecker operators of the equilibrium part T (built once, independent of dt)
            self._schur_precond = preconditioner.ShearAlfvenSchurPreconditioner(
                self.derham, self.mass_ops, self.basis_ops, _T, id_M, exb_correction=self.options.exb_correction
            )
        if self._precond in (None, "StiffnessPreconditioner"):
            pc = None
        else:
            pc_class = getattr(preconditioner, self._precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        if self.options.nonlinear:
            # one basis projection operator of the total field B0 + b (and its transpose), updated in each step
            self._initialize_projection_operator_Ttot()
            _T = self._Ttot
            _TT = self._TtotT

        else:
            _TT = _T.T

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            self._B = -1 / 2 * _TT @ curl.T @ M2
            self._B2 = -1 / 2 * _TT @ curl.T @ PB.T

            self._C = 1 / 2 * curl @ _T

            # Instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _A,
                _BC,
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )

            # allocate dummy vectors to avoid temporary array allocations
            self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
            self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
            self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

            self._byn = self._B.codomain.zeros()
            self._tmp_acc = self._B2.codomain.zeros()

        else:
            self._info = False

            # define vector field
            A_inv = inverse(
                _A,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            _f1 = A_inv @ _TT @ curl.T @ M2
            _f1_acc = A_inv @ _TT @ curl.T @ PB.T
            _f2 = curl @ _T

            # allocate output of vector field
            out_acc = self.variables.u.spline.vector.space.zeros()
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                _f1_acc.dot(self._ACC.vectors[0], out=out_acc)
                out += out_acc
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                _f2.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.u.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

    def prepare_time_step(self, dt: float):
        """Kronecker preconditioner of the Schur operator for the time step ``dt``: built in the setup, rebuilt when dt
        changes."""
        if self._precond == "StiffnessPreconditioner" and dt != self._schur_precond_dt:
            self._schur_solver.precond = self._schur_precond.build(dt)
            self._schur_precond_dt = dt

    def __call__(self, dt):
        # update time-dependent operator TB
        if self.options.nonlinear:
            self._update_weights_Ttot()

        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # accumulate
        self._ACC(self._ep_scale)

        if self.options.algo == "implicit":
            # preconditioner for the current time step (no-op unless dt changed since the setup)
            self.prepare_time_step(dt)

            # solve for new u coeffs (no tmps created here)
            byn = self._B.dot(bn, out=self._byn)
            b2acc = self._B2.dot(self._ACC.vectors[0], out=self._tmp_acc)
            byn += b2acc

            un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

            # new b coeffs (no tmps created here)
            _u = un.copy(out=self._u_tmp2)
            _u += un1
            bn1 = self._C.dot(_u, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(u=un1, b=bn1)

        else:
            self._ode_solver(0.0, dt)

        # update_weights
        if self.energetic_ions.species.weights_params.control_variate:
            self.energetic_ions.particles.update_weights()

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for ShearAlfvenCurrentCoupling5D: {info['success']}")
                logger.info(f"Iterations for ShearAlfvenCurrentCoupling5D: {info['niter']}")
                logger.info(f"Maxdiff up for ShearAlfvenCurrentCoupling5D: {diffs['u']}")
                logger.info(f"Maxdiff b2 for ShearAlfvenCurrentCoupling5D: {diffs['b']}")
                logger.info("")

    def _initialize_projection_operator_Ttot(self):
        r"""Initialize the BasisProjectionOperator of the total magnetic field (time-varying weight), e.g. for
        ``u_space="Hdiv"``

        .. math::

            \mathcal{T}^{\textnormal{tot}}_{(\mu,ijk),(\nu,mno)} := \hat \Pi^1_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu}
            \frac{B^2_{0,\alpha} + \tilde{B}^2_\alpha}{\sqrt{g}} \Lambda^2_{\nu,mno} \right] \,,

        and its transpose. One operator of the total field (instead of the sum of the equilibrium operator
        :math:`\mathcal T` and of the operator of the perturbation) needs one projection per application.
        """

        P1 = self.derham.P1
        Vh = self.derham.fem_spaces[self._u_form]

        # spline function for the evaluation of the perturbed field
        self._bf = self.derham.create_spline_function("bf", "Hdiv")
        # the weights at the last evaluation points (the 3x3 weights of all blocks of a row share the points)
        self._weight_pts = None
        self._weight_vals = None
        self._b0_cache = []

        op_class = BasisProjectionOperatorLocal if self.derham._with_local_projectors else BasisProjectionOperator
        ops = []
        for transposed in (False, True):
            ops += [
                op_class(
                    P1,
                    Vh,
                    [[None, None, None], [None, None, None], [None, None, None]],
                    transposed=transposed,
                    use_cache=True,
                    polar_shift=True,
                    V_extraction_op=self.derham.extraction_ops[self._u_form],
                    V_boundary_op=self.derham.boundary_ops[self._u_form],
                    P_boundary_op=self.derham.boundary_ops["1"],
                )
            ]
        self._Ttot, self._TtotT = ops

    def _total_field_weights(self, e1, e2, e3):
        """The 3x3 weights of the total-field operator at the points (e1, e2, e3): the rotation matrix of B0 + b (2-form
        components), times the metric factor of the velocity space. Cached for the last points, since all blocks of
        a row are evaluated at the same points."""
        pts = self._weight_pts
        if pts is None or not all(a.shape == b.shape and xp.array_equal(a, b) for a, b in zip(pts, (e1, e2, e3))):
            b0 = self._equilibrium_field(e1, e2, e3)
            b = self._bf(e1, e2, e3, local=True)
            rot = LocalRotationMatrix(*[(lambda *_, c=c: b0[c] + b[c]) for c in range(3)])(e1, e2, e3)
            if self._u_form == "1":
                rot = rot @ self.domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False)
            elif self._u_form == "2":
                rot = rot / abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))[:, :, :, None, None]
            self._weight_pts = (e1.copy(), e2.copy(), e3.copy())
            self._weight_vals = rot
        return self._weight_vals

    def _equilibrium_field(self, e1, e2, e3):
        """The 2-form components of the equilibrium field B0 at the points (e1, e2, e3), cached (the evaluation points
        of the projector are the same in each step)."""
        for pts, vals in self._b0_cache:
            if all(a.shape == b.shape and xp.array_equal(a, b) for a, b in zip(pts, (e1, e2, e3))):
                return vals
        eq = self.basis_ops.weights["eq_mhd"]
        vals = [eq.b2_1(e1, e2, e3), eq.b2_2(e1, e2, e3), eq.b2_3(e1, e2, e3)]
        self._b0_cache += [((e1.copy(), e2.copy(), e3.copy()), vals)]
        return vals

    def _update_weights_Ttot(self):
        """Update the time-dependent weights of the total-field operator and of its transpose (one assembly)."""

        self.variables.b.spline.vector.copy(out=self._bf.vector)
        self._weight_pts = None

        fun = [
            [(lambda e1, e2, e3, m=m, n=n: self._total_field_weights(e1, e2, e3)[:, :, :, m, n]) for n in range(3)]
            for m in range(3)
        ]

        self._Ttot.update_weights(fun)
        if isinstance(self._Ttot, BasisProjectionOperator):
            # the transpose re-uses the DOF matrix just assembled
            self._TtotT.update_weights(fun, dof_mat=self._Ttot.dof_matrix)
        else:
            self._TtotT.update_weights(fun)
        self._weight_pts = None
        self._weight_vals = None
