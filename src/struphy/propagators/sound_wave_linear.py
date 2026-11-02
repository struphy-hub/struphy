import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class SoundWaveLinear(Propagator):
    r"""Energy conserving sound wave step of linear MHD (zero-flow equilibrium): velocity and pressure perturbations,
    with the density perturbation carried along.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})`, :math:`\tilde p \in L^2` and
    :math:`\tilde \rho \in L^2` such that

    .. math::

        \int_\Omega \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \,\textnormal d \mathbf x
        - \int_\Omega \tilde p \, \nabla \cdot \mathbf V \,\textnormal d \mathbf x &= 0 \qquad \forall \, \mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        \int_\Omega \frac{1}{p_0} \frac{\partial \tilde p}{\partial t} \, q \,\textnormal d \mathbf x
        + \gamma \int_\Omega \nabla \cdot \tilde{\mathbf U} \, q \,\textnormal d \mathbf x &= 0 \qquad \forall \, q \in L^2\,,
        \\[2mm]
        \frac{\partial \tilde{\rho}}{\partial t} + \nabla\cdot(\rho_0 \tilde{\mathbf{U}}) &= 0\,,

    with :math:`\gamma = 5/3`. The momentum equation is weak (integration by parts of the pressure gradient; with free
    boundary conditions this imposes :math:`\tilde p = 0` weakly), the pressure equation is the
    :math:`1/p_0`-weighted :math:`L^2` projection of :math:`\partial_t \tilde p = - \gamma p_0 \nabla \cdot \tilde{\mathbf U}`
    and the continuity equation holds strongly. The first two equations conserve the energy

    .. math::

        \frac 12 \int_\Omega \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{1}{2\gamma} \int_\Omega \frac{\tilde p^2}{p_0} \,\textnormal d \mathbf x\,.

    The remaining term :math:`- \tilde{\mathbf U} \cdot \nabla p_0` of the linear pressure equation is the step
    :class:`~struphy.propagators.background_pressure_gradient.BackgroundPressureGradient`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h` and
    :math:`\mathbf p`, :math:`\mathbf n` of :math:`\tilde p_h, \tilde \rho_h \in V^3_h`,

    .. math::

        \mathbb M^2_{\rho_0} \dot{\mathbf u} = \mathbb D^\top \mathbb M^3 \mathbf p\,, \qquad
        \mathbb M^3_{1/p_0} \dot{\mathbf p} = - \gamma \, \mathbb M^3 \mathbb D \mathbf u\,, \qquad
        \dot{\mathbf n} = - \mathbb D \mathbb Q^2 \mathbf u\,,

    with the discrete divergence :math:`\mathbb D` (:attr:`~struphy.feec.psydac_derham.Derham.div`), the mass matrix
    :math:`\mathbb M^3` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3`), the weighted mass matrices
    :math:`\mathbb M^2_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`) and
    :math:`\mathbb M^3_{1/p_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3p_inv`), and the basis projection
    operator :math:`\mathbb Q^2 \approx \hat \Pi^2[\rho_0 \,\cdot\,]`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.Q2`). It conserves the discrete energy
    :math:`\frac 12 \mathbf u^\top \mathbb M^2_{\rho_0} \mathbf u + \frac{1}{2\gamma} \mathbf p^\top \mathbb M^3_{1/p_0} \mathbf p`.

    **Time discretization.** Crank-Nicolson (implicit mid-point), the only option, which conserves the discrete
    energy exactly (up to the solver tolerances). The pressure is eliminated with
    :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (:math:`x = \mathbf u`, :math:`y = \mathbf p`), with the
    symmetric positive definite Schur complement

    .. math::

        S = \mathbb M^2_{\rho_0} + \frac{\gamma \Delta t^2}{4} \mathbb D^\top \mathbb M^3 (\mathbb M^3_{1/p_0})^{-1} \mathbb M^3 \mathbb D\,,

    solved by ``Options.solver`` with the mass matrix preconditioner ``Options.precond`` of
    :math:`\mathbb M^2_{\rho_0}`, where :math:`(\mathbb M^3_{1/p_0})^{-1}` is an inner iterative solve
    (``Options.solver_M3``, ``Options.precond_M3``). Then

    .. math::

        \mathbf p^{n+1} = \mathbf p^n - \frac{\gamma \Delta t}{2} (\mathbb M^3_{1/p_0})^{-1} \mathbb M^3 \mathbb D (\mathbf u^{n+1} + \mathbf u^n)\,,
        \qquad
        \mathbf n^{n+1} = \mathbf n^n - \frac{\Delta t}{2} \mathbb D \mathbb Q^2 (\mathbf u^{n+1} + \mathbf u^n)\,.
    """

    class Variables:
        """Container for variables advanced by :class:`SoundWaveLinear`.

        Attributes
        ----------
        n : FEECVariable
            Density perturbation variable in ``"L2"`` space.
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        p : FEECVariable
            Pressure perturbation variable in ``"L2"`` space.
        """

        def __init__(self):
            self._n: FEECVariable = None
            self._u: FEECVariable = None
            self._p: FEECVariable = None

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
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`SoundWaveLinear`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Iterative solver of the (symmetric) Schur complement.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the Schur complement, built from :math:`\\mathbb M^2_{\\rho_0}` (``M2n``).
        solver_params : SolverParameters, default=None
            Controls of the Schur complement solve; defaults to ``SolverParameters()``.
        solver_M3 : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the inner ``M3p_inv`` inversion.
        precond_M3 : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the inner ``M3p_inv`` inversion.
        solver_params_M3 : SolverParameters, default=None
            Controls of the inner ``M3p_inv`` inversion; defaults to ``SolverParameters(tol=1e-14)`` (the outer Krylov
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

        _A = self.mass_ops.M2n
        _M3 = self.mass_ops.M3
        div = self.derham.div

        # inverse of M3p_inv (inner solve)
        if self.options.precond_M3 is None:
            pc_M3 = None
        else:
            pc_M3 = getattr(preconditioner, self.options.precond_M3)(self.mass_ops.M3p_inv)
        self._M3p_inv_inv = inverse(
            self.mass_ops.M3p_inv,
            self.options.solver_M3,
            pc=pc_M3,
            tol=self.options.solver_params_M3.tol,
            maxiter=self.options.solver_params_M3.maxiter,
            verbose=self.options.solver_params_M3.verbose,
        )

        # block matrix [[A B], [C I]] (without time step size dt in the off-diagonals)
        self._B = -1 / 2 * div.T @ _M3
        self._C = self.gamma / 2 * self._M3p_inv_inv @ _M3 @ div
        self._DQ = div @ self.basis_ops.Q2

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(_A)

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
        self._p_tmp1 = self.variables.p.spline.vector.space.zeros()
        self._n_tmp1 = self.variables.n.spline.vector.space.zeros()
        self._byn = self._B.codomain.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        nn = self.variables.n.spline.vector
        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector

        # solve for new u coeffs
        byn = self._B.dot(pn, out=self._byn)
        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new p and n coeffs from the mid-point velocity
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        # the inner solve is ill-conditioned where p0 is small (e.g. p0 = 0 at a wall): warn if it did not converge
        info_M3 = self._M3p_inv_inv.get_info()
        if not info_M3["success"] and MPI.COMM_WORLD.Get_rank() == 0:
            logger.warning(f"Inner M3p_inv solve of SoundWaveLinear not converged in {info_M3['niter']} iterations.")

        nn1 = self._DQ.dot(_u, out=self._n_tmp1)
        nn1 *= -dt / 2
        nn1 += nn

        diffs = self.update_feec_variables(n=nn1, u=un1, p=pn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for SoundWaveLinear: {info['success']}")
            logger.info(f"Iterations for SoundWaveLinear: {info['niter']}")
            logger.info(f"Iterations of the inner M3p_inv solve (last) for SoundWaveLinear: {info_M3['niter']}")
            logger.info(f"Maxdiff n3 for SoundWaveLinear: {diffs['n']}")
            logger.info(f"Maxdiff u2 for SoundWaveLinear: {diffs['u']}")
            logger.info(f"Maxdiff p3 for SoundWaveLinear: {diffs['p']}")
            logger.info("")
