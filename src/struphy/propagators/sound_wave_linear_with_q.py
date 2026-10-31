import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.density_weighted import VelocityMass
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class SoundWaveLinearWithQ(Propagator):
    r"""Energy conserving sound wave step of the linear models with the variable :math:`q = \sqrt p` (zero-flow
    equilibrium): velocity perturbation and perturbation of :math:`q`, with the density perturbation carried along.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})`, :math:`\tilde q \in L^2` and
    :math:`\tilde \rho \in L^2` such that

    .. math::

        \int_\Omega \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \,\textnormal d \mathbf x
        - 2 \int_\Omega q_0 \tilde q \, \nabla \cdot \mathbf V \,\textnormal d \mathbf x &= 0 \qquad \forall \, \mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        \int_\Omega \frac{\partial \tilde q}{\partial t} \, r \,\textnormal d \mathbf x
        + \frac{\gamma}{2} \int_\Omega q_0 \nabla \cdot \tilde{\mathbf U} \, r \,\textnormal d \mathbf x &= 0 \qquad \forall \, r \in L^2\,,
        \\[2mm]
        \frac{\partial \tilde{\rho}}{\partial t} + \nabla\cdot(\rho_0 \tilde{\mathbf{U}}) &= 0\,,

    with :math:`q_0 = \sqrt{p_0}` and :math:`\gamma = 5/3`. The momentum equation is the weak form of
    :math:`\rho_0 \partial_t \tilde{\mathbf U} = - \nabla \tilde p` with the linearized pressure perturbation
    :math:`\tilde p = 2 q_0 \tilde q` (integration by parts; with free boundary conditions this imposes
    :math:`\tilde q = 0` weakly), the second equation is the :math:`L^2` projection of
    :math:`\partial_t \tilde q = - \frac{\gamma}{2} q_0 \nabla \cdot \tilde{\mathbf U}` and the continuity equation holds
    strongly. The first two equations conserve the energy

    .. math::

        \frac 12 \int_\Omega \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{2}{\gamma} \int_\Omega \tilde q^2 \,\textnormal d \mathbf x\,,

    which is the energy :math:`\frac 12 \int \rho_0 |\tilde{\mathbf U}|^2 + \frac{1}{2\gamma} \int \tilde p^2 / p_0` of
    linear MHD written with :math:`\tilde q`. The remaining term :math:`- \tilde{\mathbf U} \cdot \nabla q_0` of the
    linear equation of :math:`\tilde q` is the step
    :class:`~struphy.propagators.background_q_gradient.BackgroundQGradient`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h` and
    :math:`\mathbf q`, :math:`\mathbf n` of :math:`\tilde q_h, \tilde \rho_h \in V^3_h`,

    .. math::

        \mathbb M^2_{\rho_0} \dot{\mathbf u} = 2\, \mathbb D^\top \mathbb M^3_{q_0} \mathbf q\,, \qquad
        \mathbb M^3 \dot{\mathbf q} = - \frac{\gamma}{2} \, \mathbb M^3_{q_0} \mathbb D \mathbf u\,, \qquad
        \dot{\mathbf n} = - \mathbb D \mathbb Q^2 \mathbf u\,,

    with the discrete divergence :math:`\mathbb D` (:attr:`~struphy.feec.psydac_derham.Derham.div`), the mass matrix
    :math:`\mathbb M^3` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3`), the weighted mass matrices
    :math:`\mathbb M^2_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`) and
    :math:`\mathbb M^3_{q_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3q0`), and the basis projection
    operator :math:`\mathbb Q^2 \approx \hat \Pi^2[\rho_0 \,\cdot\,]`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.Q2`). The force is the transpose of the
    operator of the :math:`\tilde q` equation (up to the factor :math:`4/\gamma`), so the semi-discrete system conserves
    the discrete energy :math:`\frac 12 \mathbf u^\top \mathbb M^2_{\rho_0} \mathbf u + \frac{2}{\gamma} \mathbf q^\top
    \mathbb M^3 \mathbf q`, a positive definite quadratic form, and its spectrum is purely imaginary.

    **Time discretization.** Crank-Nicolson (implicit mid-point), the only option, which conserves the discrete
    energy exactly (up to the solver tolerances); its eigenvalues are on the unit circle. :math:`\mathbf q` is
    eliminated with :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (:math:`x = \mathbf u`,
    :math:`y = \mathbf q`), with the symmetric positive definite Schur complement

    .. math::

        S = \mathbb M^2_{\rho_0} + \frac{\gamma \Delta t^2}{4} \mathbb D^\top \mathbb M^3_{q_0} (\mathbb M^3)^{-1} \mathbb M^3_{q_0} \mathbb D\,,

    solved by ``Options.solver`` with the mass matrix preconditioner ``Options.precond`` of
    :math:`\mathbb M^2_{\rho_0}`, where :math:`(\mathbb M^3)^{-1}` is an inner iterative solve
    (``Options.solver_M3``, ``Options.precond_M3``). Then

    .. math::

        \mathbf q^{n+1} = \mathbf q^n - \frac{\gamma \Delta t}{4} (\mathbb M^3)^{-1} \mathbb M^3_{q_0} \mathbb D (\mathbf u^{n+1} + \mathbf u^n)\,,
        \qquad
        \mathbf n^{n+1} = \mathbf n^n - \frac{\Delta t}{2} \mathbb D \mathbb Q^2 (\mathbf u^{n+1} + \mathbf u^n)\,.

    **Perturbative models** (constructor argument ``rho_field``, the density perturbation :math:`\tilde \rho`): the
    inertia is that of the total density, :math:`\rho_0 \to \rho = \rho_0 + \tilde\rho^n` in the momentum equation,
    i.e. :math:`\mathbb M^2_{\rho_0} \to \mathbb M^2_{\rho}` (:class:`~struphy.feec.density_weighted.VelocityMass`,
    assembled at :math:`t^n`, with the coefficients ``projected_equil.n3`` of :math:`\rho_0`), and the density is not
    advanced (the continuity equation is in :class:`~struphy.propagators.momentum_advection.MomentumAdvection`). The
    step then conserves :math:`\frac 12 \mathbf u^\top \mathbb M^2_\rho \mathbf u + \frac{2}{\gamma} \mathbf q^\top
    \mathbb M^3 \mathbf q` exactly. The preconditioner is still built from :math:`\mathbb M^2_{\rho_0}`.
    """

    class Variables:
        """Container for variables advanced by :class:`SoundWaveLinearWithQ`.

        Attributes
        ----------
        n : FEECVariable
            Density perturbation variable in ``"L2"`` space.
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        q : FEECVariable
            Perturbation of :math:`q = \\sqrt p` in ``"L2"`` space.
        """

        def __init__(self):
            self._n: FEECVariable = None
            self._u: FEECVariable = None
            self._q: FEECVariable = None

        @property
        def n(self) -> FEECVariable:
            return self._n

        @n.setter
        def n(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._n = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._u = new

        @property
        def q(self) -> FEECVariable:
            return self._q

        @q.setter
        def q(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._q = new

    def __init__(self, rho_field: FEECVariable = None):
        r"""
        Parameters
        ----------
        rho_field : FEECVariable, default=None
            Density perturbation in ``"L2"`` space (not updated): the inertia is that of the total density
            :math:`\rho_0 + \tilde \rho` and the density is not advanced (perturbative models). If ``None``, the inertia
            is that of :math:`\rho_0` and the density perturbation (variable ``n``) is advanced (linear models).
        """
        assert rho_field is None or rho_field.space == "L2", "rho_field must be in L2."
        self.variables = self.Variables()
        self.rho_field = rho_field

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`SoundWaveLinearWithQ`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Iterative solver of the (symmetric) Schur complement.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the Schur complement, built from :math:`\\mathbb M^2_{\\rho_0}` (``M2n``).
        solver_params : SolverParameters, default=None
            Controls of the Schur complement solve; defaults to ``SolverParameters()``.
        solver_M3 : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the inner ``M3`` inversion.
        precond_M3 : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the inner ``M3`` inversion.
        solver_params_M3 : SolverParameters, default=None
            Controls of the inner ``M3`` inversion; defaults to ``SolverParameters(tol=1e-14)`` (the outer Krylov
            solver and the energy conservation need an accurate inner inverse).
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        solver_M3: LiteralOptions.OptsSymmSolver = "pcg"
        precond_M3: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params_M3: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.solver_M3, LiteralOptions.OptsSymmSolver)
            check_option(self.precond_M3, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.solver_params_M3 is None:
                self.solver_params_M3 = SolverParameters(tol=1e-14)

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

    @property
    def gamma(self) -> float:
        """Adiabatic index."""
        return 5 / 3

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        self._velocity_mass = VelocityMass(
            self.derham, self.domain, self.mass_ops, self.projected_equil, rho_field=self.rho_field
        )
        _A = self._velocity_mass.M
        _M3 = self.mass_ops.M3
        _M3q0 = self.mass_ops.M3q0
        div = self.derham.div

        # inverse of M3 (inner solve)
        if self.options.precond_M3 is None:
            pc_M3 = None
        else:
            pc_M3 = getattr(preconditioner, self.options.precond_M3)(_M3)
        self._M3_inv = inverse(
            _M3,
            self.options.solver_M3,
            pc=pc_M3,
            tol=self.options.solver_params_M3.tol,
            maxiter=self.options.solver_params_M3.maxiter,
            verbose=self.options.solver_params_M3.verbose,
        )

        # block matrix [[A B], [C I]] (without time step size dt in the off-diagonals)
        self._B = -div.T @ _M3q0
        self._C = self.gamma / 4 * self._M3_inv @ _M3q0 @ div
        self._DQ = div @ self.basis_ops.Q2

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(self.mass_ops.M2n)

        self._schur_solver = SchurSolver(
            _A,
            self._B @ self._C,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
        self._q_tmp1 = self.variables.q.spline.vector.space.zeros()
        self._n_tmp1 = self.variables.n.spline.vector.space.zeros()
        self._byn = self._B.codomain.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        nn = self.variables.n.spline.vector
        un = self.variables.u.spline.vector
        qn = self.variables.q.spline.vector

        # inertia of the total density at t^n (perturbative models)
        self._velocity_mass.update()

        # solve for new u coeffs
        byn = self._B.dot(qn, out=self._byn)
        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new q and n coeffs from the mid-point velocity
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        qn1 = self._C.dot(_u, out=self._q_tmp1)
        qn1 *= -dt
        qn1 += qn

        if self._velocity_mass.weighted:
            diffs = self.update_feec_variables(u=un1, q=qn1)
            diffs["n"] = 0.0
        else:
            nn1 = self._DQ.dot(_u, out=self._n_tmp1)
            nn1 *= -dt / 2
            nn1 += nn
            diffs = self.update_feec_variables(n=nn1, u=un1, q=qn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            info_M3 = self._M3_inv.get_info()
            logger.info(f"Status     for SoundWaveLinearWithQ: {info['success']}")
            logger.info(f"Iterations for SoundWaveLinearWithQ: {info['niter']}")
            logger.info(f"Iterations of the inner M3 solve (last) for SoundWaveLinearWithQ: {info_M3['niter']}")
            logger.info(f"Maxdiff n3 for SoundWaveLinearWithQ: {diffs['n']}")
            logger.info(f"Maxdiff u2 for SoundWaveLinearWithQ: {diffs['u']}")
            logger.info(f"Maxdiff q3 for SoundWaveLinearWithQ: {diffs['q']}")
            logger.info("")
