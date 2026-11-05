import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.propagators.base import Propagator
from struphy.propagators.utils import Magnetization1form, TotalFieldMasses1form
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurrentCoupling5DMagnetization1form(Propagator):
    r"""Force of the magnetization current of drift-kinetic energetic ions on the MHD velocity, with the magnetic
    field in :math:`H(\textnormal{curl})`: the magnetization part of
    :class:`~struphy.propagators.shear_alfven_curl_curl_current_coupling_5d.ShearAlfvenCurlCurlCurrentCoupling5D`, as a
    step of its own (``split_magnetization=True`` of :class:`~struphy.models.LinearMHDDriftkineticCC`), so that the
    shear-Alfvén step is the uncoupled
    :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` such that

    .. math::

        \int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \left[(\nabla \times \mathbf M) \times \mathbf B\right] \cdot \mathbf V \,\textnormal d \mathbf x
        \qquad \forall \,\mathbf V \in H(\textnormal{div})\,,

    with the magnetization of the energetic ions along the equilibrium field, :math:`\mathbf M \in
    H(\textnormal{curl})`,

    .. math::

        \int_\Omega \mathbf M \cdot \mathbf C \,\textnormal d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}}
        \int_\Omega \iint f^\textnormal{vol} \mu \, \mathbf b_0 \cdot \mathbf C \,\textnormal d \mu \,\textnormal d v_\parallel \,\textnormal d \mathbf x
        \qquad \forall \, \mathbf C \in H(\textrm{curl})\,,

    and :math:`\mathbf B = \mathbf B_0` (``nonlinear=False``) or :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`
    (``nonlinear=True``, the magnetic field perturbation ``b_tilde``, not updated). The energetic ions and the
    magnetic field do not change in this step.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u \in \mathbb R^{N_2}` of
    :math:`\tilde{\mathbf U}_h \in V^2_h` and the magnetization :math:`\mathbf m \in \mathbb R^{N_1}` of
    :class:`~struphy.propagators.utils.Magnetization1form`,

    .. math::

        \mathbb M^2_{\rho_0} \dot{\mathbf u} = - \mathbb M^2_{B} \mathbb C \mathbf m\,, \qquad
        \mathbb M^1 \mathbf m = (\mathcal P^{b,1})^\top \mathbf a\,,

    with the density-weighted mass matrix :math:`\mathbb M^2_{\rho_0}`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`), the discrete curl :math:`\mathbb C` and the antisymmetric
    mass matrix :math:`\mathbb M^2_{B}` of :math:`\int (\mathbf B \times \mathbf W) \cdot \mathbf V
    \,\textnormal d \mathbf x` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2B`, or of the total field, see
    :class:`~struphy.propagators.utils.TotalFieldMasses1form`). The noise of the accumulated vector
    :math:`\mathbf a` enters only mass matrix solves, not the curl-curl solve of the shear-Alfvén step.

    In the coupled step of
    :class:`~struphy.propagators.shear_alfven_curl_curl_current_coupling_5d.ShearAlfvenCurlCurlCurrentCoupling5D`, this
    term and the induction equation together conserve the fluid energy plus the energy of the magnetic moments in the
    parallel perturbation, :math:`\mathbf a^\top \mathcal P^{b,1} \mathbf b`. Split off, this step changes the kinetic
    energy of the fluid by :math:`-\Delta t \, \mathbf u^\top \mathbb M^2_B \mathbb C \mathbf m` (at the mid-point)
    without a compensating change of that energy: the split model is not exactly energy conserving (as the
    shear-Alfvén step without projection).

    **Time discretization.** Exact in time, since :math:`\mathbf m` and :math:`\mathbb M^2_B` do not change in the step:

    .. math::

        \mathbf u^{n+1} = \mathbf u^n - \Delta t \, (\mathbb M^2_{\rho_0})^{-1} \mathbb M^2_{B} \mathbb C \mathbf m\,,

    with one ``M1`` solve (``solver_params_M1``) and one ``M2n`` solve (``solver_params_M2n``, both pcg with the mass
    matrix preconditioner) per step.

    Parameters
    ----------
    energetic_ions : PICVariable, default=None
        Energetic-ion particle variable (``"Particles5D"`` space, not updated) providing the magnetization.
    b_tilde : FEECVariable, default=None
        Magnetic field perturbation (``"Hcurl"`` space, not updated), used with ``nonlinear=True``.
    """

    class Variables:
        """Container for variables advanced by :class:`CurrentCoupling5DMagnetization1form`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._u = new

    def __init__(self, energetic_ions: PICVariable = None, b_tilde: FEECVariable = None):
        assert b_tilde is None or b_tilde.space == "Hcurl", "b_tilde must be in Hcurl."
        self.variables = self.Variables()
        self.energetic_ions = energetic_ions
        self.b_tilde = b_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`CurrentCoupling5DMagnetization1form`.

        Parameters
        ----------
        ep_scale : float, default=None
            Factor of the energetic-particle contributions (accumulation): the mass ratio
            :math:`A_\textnormal{h}/A_\textnormal{b}` of the energetic ions and the MHD bulk if ``None``.
        filter_params : FilterParameters, default=None
            Filter of the accumulated vector; defaults to ``FilterParameters()`` (no filter).
        nonlinear : bool, default=False
            If ``True``, the force uses the total magnetic field :math:`\mathbf B_0 + \tilde{\mathbf B}` (mass matrix
            assembled in each step, needs ``b_tilde``), else the equilibrium field :math:`\mathbf B_0`, as the
            shear-Alfvén step
            :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the ``M2n`` solve (``"PolarMassMatrixPreconditioner"`` is used with polar splines).
        solver_params_M1 : SolverParameters, default=None
            Controls of the ``M1`` solve of the magnetization; defaults to ``SolverParameters(tol=1e-14)``.
        solver_params_M2n : SolverParameters, default=None
            Controls of the ``M2n`` solve; defaults to ``SolverParameters(tol=1e-14)``.
        """

        ep_scale: float = None
        filter_params: FilterParameters = None
        nonlinear: bool = False
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params_M1: SolverParameters = None
        solver_params_M2n: SolverParameters = None

        def __post_init__(self):
            assert self.ep_scale is None or isinstance(self.ep_scale, float)
            assert isinstance(self.nonlinear, bool)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            if self.filter_params is None:
                self.filter_params = FilterParameters()

            if self.solver_params_M1 is None:
                self.solver_params_M1 = SolverParameters(tol=1e-14)

            if self.solver_params_M2n is None:
                self.solver_params_M2n = SolverParameters(tol=1e-14)

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
        assert self.energetic_ions is not None, f"{self.__class__.__name__} needs the energetic ions."
        if self.options.nonlinear:
            assert self.b_tilde is not None, f"{self.__class__.__name__} with nonlinear=True needs b_tilde."

        # factor of the energetic-particle contributions (default: mass ratio A_h/A_b)
        self._ep_scale = self.options.ep_scale
        if self._ep_scale is None:
            self._ep_scale = self.energetic_ions.species.mass_number / self.variables.u.species.mass_number
            logger.info(f"{self.__class__.__name__}: ep_scale = A_h/A_b = {self._ep_scale} (mass ratio, default).")

        self._magnetization = Magnetization1form(
            self.energetic_ions.particles,
            self.derham,
            self.domain,
            self.mass_ops,
            self.basis_ops,
            self.options.filter_params,
            self.options.solver_params_M1,
        )

        if self.options.nonlinear:
            self._total_field = TotalFieldMasses1form(
                self.b_tilde, self.derham, self.domain, self.mass_ops, self.basis_ops
            )
            M2B = self._total_field.M2B
        else:
            M2B = self.mass_ops.M2B

        # (M2n)^{-1} M2B C
        M2n = self.mass_ops.M2n
        precond = self.options.precond
        if precond is not None and self.derham.polar_splines:
            precond = "PolarMassMatrixPreconditioner"
        params = self.options.solver_params_M2n
        M2n_inv = inverse(
            M2n,
            "pcg",
            pc=None if precond is None else getattr(preconditioner, precond)(M2n),
            tol=params.tol,
            maxiter=params.maxiter,
            verbose=params.verbose,
            recycle=True,
        )
        self._force = M2n_inv @ M2B @ self.derham.curl

        self._du = self.variables.u.spline.vector.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        if self.options.nonlinear:
            self._total_field.update()

        # magnetization of the (fixed) energetic ions and its force
        m = self._magnetization(self._ep_scale)
        du = self._force.dot(m, out=self._du)

        un = self.variables.u.spline.vector
        u_new = un.copy(out=self._u_new)
        u_new.mul_iadd(-dt, du)
        self.update_feec_variables(u=u_new)
