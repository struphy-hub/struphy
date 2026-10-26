import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.feec.variational_utilities import Hdiv0_transport_operator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.variational_linear_base import VariationalLinearEvolve
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalLinearAlfvenEvolve(VariationalLinearEvolve):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and the magnetic field (shear-Alfvén and
    fast magnetosonic waves) of the linear variational MHD models, with one linear solve per time step: find
    :math:`\mathbf u \in (H^1)^3` and :math:`\mathbf B \in H(\textnormal{div})` (the perturbation of the equilibrium
    :math:`\mathbf B_0`) such that, with the transport operator
    :math:`\mathcal T_{\bar{\mathbf B}}(\mathbf v) = \nabla \times (\bar{\mathbf B} \times \mathbf v)`,

    .. math::

        \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
        - \int_\Omega \left(\mathbf B \cdot \mathcal T_{\mathbf B_0}(\mathbf v)
        + \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v)\right)\textrm d \mathbf x = 0
        \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \qquad
        \partial_t \mathbf B + \mathcal T_{\mathbf B_0}(\mathbf u) = 0 \,.

    The thermodynamic variable is not changed, so this propagator serves the linear models in :math:`p` and in
    :math:`q = \sqrt p` alike; together with
    :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve` it splits the linear
    MHD step into the shear-Alfvén/fast and the sound waves. The second-order term comes from the auxiliary variable
    :math:`\hat{\mathbf B}` (``bt2``), which, if given, is advanced with
    :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`; it enters the conserved energy as
    :math:`\int \mathbf B_0 \cdot \hat{\mathbf B}`.

    Discretization: with :math:`\mathbb T[\bar{\mathbf B}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf B}} \times
    \vec{\boldsymbol\Lambda}^v]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`),

    .. math::

        \begin{align}
        &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \mathbb T[\mathbf B_0]^\top \mathbb M^2 \mathbf b^{n+\frac12}
        - \mathbb T[\mathbf b^n]^\top \mathbb M^2 \mathbf b_0 = 0 \,,
        \\[2mm]
        &\frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} + \mathbb T[\mathbf B_0]\, \mathbf u^{n+\frac12} = 0 \,,
        \qquad
        \frac{\hat{\mathbf b}^{n+1} - \hat{\mathbf b}^n}{\Delta t} + \mathbb T[\mathbf b^n]\, \mathbf u^{n+\frac12} = 0
        \,.
        \end{align}

    The weight :math:`\mathbf b^n` of the second-order term is the same in the momentum equation and in the update of
    :math:`\hat{\mathbf b}`, which makes the step energy conserving; it is first order in time. These are the
    equations of :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve` without the sound waves,
    with ``model="linear_q"`` and ``implicit_weights=False``, but solved with one linear solve
    (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`) instead of a Picard iteration.
    The Schur complement is the shear-Alfvén operator
    :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4} \mathbb T[\mathbf B_0]^\top \mathbb M^2
    \mathbb T[\mathbf B_0]`, constant in time.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalLinearAlfvenEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        b : FEECVariable
            Magnetic-field perturbation in ``"Hdiv"`` space.
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
            assert new.space == "H1vec"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, bt2: FEECVariable = None):
        r"""
        Parameters
        ----------
        bt2 : FEECVariable, default=None
            Auxiliary second-order variable :math:`\hat{\mathbf B}` (``"Hdiv"`` space), advanced with
            :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`. It enters only the energy.
        """
        self.variables = self.Variables()
        self.bt2 = bt2

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalLinearAlfvenEvolve`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver of the Schur system.
        precond : {"MassMatrixPreconditioner", None}, default="MassMatrixPreconditioner"
            Preconditioner of the Schur system (of its velocity block).
        solver_params : SolverParameters, default=None
            Linear-solver controls. The system is solved for a normalized right-hand side, so ``tol`` is relative to
            the norm of the residual at :math:`t^n`.
        info : bool, default=False
            Log the information of the linear solver in every step.
        """

        # specific literals
        OptsPrecond = Literal["MassMatrixPreconditioner", None]
        # propagator options
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: OptsPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        info: bool = False

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, self.OptsPrecond)

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
        self._bt2 = None if self.bt2 is None else self.bt2.spline.vector
        self._info = self.options.info and (self.rank == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._allocate_mass(self.options.precond)

        # transport operators with the equilibrium (constant) and with the perturbation (weight b^n)
        self._T0 = Hdiv0_transport_operator(self.derham)
        self._T0T = self._T0.T
        self._T0.update_coeffs(self.projected_equil.b2)
        self._T0T.update_coeffs(self.projected_equil.b2)
        self._Tb = Hdiv0_transport_operator(self.derham)
        self._TbT = self._Tb.T

        # linear form of the equilibrium field, -M2 b0
        self._M2b0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

        self._allocate_schur_solver(
            self._T0,
            self._T0T @ self.mass_ops.M2,
            -1.0,
            self.derham.V2pol,
            self.options.solver,
            self.options.solver_params,
        )

        # temporaries
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_fu = u.space.zeros()
        self._tmp_fu2 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_fb = b.space.zeros()
        self._tmp_M2b = b.space.zeros()

    @profile
    def __call__(self, dt):
        # WMMnew is assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._refresh_mass_pc()

        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # weight of the second-order term at t^n
        self._Tb.update_coeffs(bn)
        self._TbT.update_coeffs(bn)

        # residual at (u^n, b^n): the force -T0^T M2 b^n - T[b^n]^T M2 b0 and the induction T0 u^n (times dt)
        M2b = self.mass_ops.M2.dot(bn, out=self._tmp_M2b)
        M2b *= -1.0
        f_u = self._T0T.dot(M2b, out=self._tmp_fu)
        f_u += self._TbT.dot(self._M2b0, out=self._tmp_fu2)
        f_u *= dt
        f_b = self._T0.dot(un, out=self._tmp_fb)
        f_b *= dt

        incr = self._solve_increment(dt, f_u, f_b)
        un1 = un.copy(out=self._tmp_un1)
        un1 -= incr[0]
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 -= incr[1]

        # auxiliary second-order variable, with the same weight b^n
        if self._bt2 is not None:
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            b_advection = self._Tb.dot(un12, out=self._tmp_fb)
            b_advection *= dt
            self._bt2 -= b_advection

        self.update_feec_variables(u=un1, b=bn1)
