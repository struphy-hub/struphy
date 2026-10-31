import logging
from dataclasses import dataclass

import cunumpy as xp
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.density_weighted import VelocityMass
from struphy.feec.variational_utilities import Pressure_transport_operator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class SoundWaveNonlinearWithQ(Propagator):
    r"""Energy conserving sound wave step of the terms that are nonlinear in the perturbation of :math:`q = \sqrt p`,
    in the perturbative models with the variable :math:`q` (zero-flow equilibrium, :math:`q = q_0 + \tilde q`).

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` and :math:`\tilde q \in L^2` such that

    .. math::

        \int_\Omega \rho \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \,\textnormal d \mathbf x
        - \frac{2}{\gamma - 1} \int_\Omega \tilde q \, \mathcal S_{\tilde q}(\mathbf V) \,\textnormal d \mathbf x &= 0
        \qquad \forall \, \mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        \frac{\partial \tilde q}{\partial t} + \mathcal S_{\tilde q}(\tilde{\mathbf U}) &= 0\,,

    with the transport operator :math:`\mathcal S_s(\mathbf V) = \nabla \cdot (s \mathbf V) + (\frac{\gamma}{2} - 1) s
    \nabla \cdot \mathbf V`, the total density :math:`\rho = \rho_0 + \tilde \rho` (fixed in this step) and
    :math:`\gamma = 5/3`. Since :math:`\int \tilde q \, \nabla \cdot (\tilde q \mathbf V) = \frac 12 \int \tilde q^2
    \nabla \cdot \mathbf V` (for :math:`\mathbf V \cdot \mathbf n = 0` on the boundary), the force is
    :math:`\int \tilde q^2 \nabla \cdot \mathbf V`, the weak form of :math:`- \nabla \tilde q^2`, and the second
    equation is :math:`\partial_t \tilde q + \tilde{\mathbf U} \cdot \nabla \tilde q + \frac{\gamma}{2} \tilde q \nabla
    \cdot \tilde{\mathbf U} = 0`: the terms quadratic in the perturbations of
    :math:`\rho \partial_t \mathbf U = - \nabla q^2` and :math:`\partial_t q + \mathbf U \cdot \nabla q + \frac{\gamma}{2}
    q \nabla \cdot \mathbf U = 0`. The step conserves the energy

    .. math::

        \frac 12 \int_\Omega \rho |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{1}{\gamma - 1} \int_\Omega \tilde q^2 \,\textnormal d \mathbf x\,.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h` and
    :math:`\mathbf q` of :math:`\tilde q_h \in V^3_h`,

    .. math::

        \mathbb M^2_\rho \dot{\mathbf u} = \frac{2}{\gamma - 1} \mathbb S[\mathbf q]^\top \mathbb M^3 \mathbf q\,, \qquad
        \dot{\mathbf q} = - \mathbb S[\mathbf q] \mathbf u\,,

    with :math:`\mathbb S[\mathbf w] = \mathbb D \hat \Pi^2[\hat w^3 \vec{\boldsymbol\Lambda}^2 / \sqrt g] + (\frac{\gamma}{2} - 1)
    \hat \Pi^3[\hat w^3 \Lambda^3 / \sqrt g] \mathbb D`
    (:class:`~struphy.feec.variational_utilities.Pressure_transport_operator` with the adiabatic index
    :math:`\gamma / 2`), the mass matrix :math:`\mathbb M^3` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3`)
    and the mass matrix :math:`\mathbb M^2_\rho` of the total density
    (:class:`~struphy.feec.density_weighted.VelocityMass`, with the constructor argument ``rho_field``; without it
    :math:`\mathbb M^2_{\rho_0}`). The force is the transpose of the operator of the :math:`\mathbf q` equation, so the
    discrete energy :math:`\frac 12 \mathbf u^\top \mathbb M^2_\rho \mathbf u + \frac{1}{\gamma - 1} \mathbf q^\top \mathbb M^3
    \mathbf q` (a positive definite quadratic form) is conserved for any weight in :math:`\mathbb S`.

    **Time discretization.** Crank-Nicolson in :math:`(\mathbf u, \mathbf q)` with the weight :math:`\mathbf w` of
    :math:`\mathbb S` fixed in the step (``Options.weights``):

    .. math::

        \mathbb M^2_\rho \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t} = \frac{2}{\gamma - 1} \mathbb S[\mathbf w]^\top \mathbb M^3
        \mathbf q^{n+\frac12}\,, \qquad
        \frac{\mathbf q^{n+1} - \mathbf q^n}{\Delta t} = - \mathbb S[\mathbf w] \mathbf u^{n+\frac12}\,,

    which conserves the discrete energy exactly (up to the solver tolerances): with ``"frozen"`` (default)
    :math:`\mathbf w = \mathbf q^n`, one linear solve per step (first order in time); with ``"midpoint"``
    :math:`\mathbf w = \mathbf q^{n+\frac12}`, the implicit midpoint rule (second order), solved by Picard iteration
    (``Options.nonlin_solver``) with one linear solve per iteration. The step is a linear map with the fixed weight,
    the force being the transpose of the :math:`\mathbf q` operator, so it does not damp or amplify the waves. Each linear
    solve eliminates :math:`\mathbf q` with :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, with the symmetric
    positive definite Schur complement

    .. math::

        S = \mathbb M^2_\rho + \frac{\Delta t^2}{2(\gamma - 1)} \mathbb S[\mathbf w]^\top \mathbb M^3 \mathbb S[\mathbf w]\,,

    solved by ``Options.solver`` with the mass matrix preconditioner ``Options.precond`` of
    :math:`\mathbb M^2_{\rho_0}`.
    """

    class Variables:
        """Container for variables advanced by :class:`SoundWaveNonlinearWithQ`.

        Attributes
        ----------
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        q : FEECVariable
            Perturbation of :math:`q = \\sqrt p` in ``"L2"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._q: FEECVariable = None

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
            :math:`\rho_0 + \tilde \rho`. If ``None``, the inertia is that of :math:`\rho_0`.
        """
        assert rho_field is None or rho_field.space == "L2", "rho_field must be in L2."
        self.variables = self.Variables()
        self.rho_field = rho_field

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`SoundWaveNonlinearWithQ`.

        Parameters
        ----------
        weights : LiteralOptions.OptsTimeWeights, default="frozen"
            Weight of the transport operator: :math:`\\mathbf q^n` (``"frozen"``, one linear solve) or
            :math:`\\mathbf q^{n+1/2}` (``"midpoint"``, Picard iteration).
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Iterative solver of the (symmetric) Schur complement.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the Schur complement, built from :math:`\\mathbb M^2_{\\rho_0}` (``M2n``).
        solver_params : SolverParameters, default=None
            Controls of the Schur complement solve; defaults to ``SolverParameters(tol=1e-13)``.
        nonlin_solver : NonlinearSolverParameters, default=None
            Controls of the Picard iteration with ``weights="midpoint"`` (``tol``: relative change of the velocity);
            defaults to ``NonlinearSolverParameters(tol=1e-12)``.
        """

        weights: LiteralOptions.OptsTimeWeights = "frozen"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.weights, LiteralOptions.OptsTimeWeights)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters(tol=1e-13)
            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(tol=1e-12)
            assert self.nonlin_solver.type == "Picard", (
                "SoundWaveNonlinearWithQ: only the Picard iteration is implemented."
            )

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

        # transport operator S[w] and its transpose (weight q at gamma / 2)
        self._S = Pressure_transport_operator(self.derham, self.domain, None, self.gamma / 2, u_space="Hdiv")
        self._ST = Pressure_transport_operator(
            self.derham, self.domain, None, self.gamma / 2, transposed=True, u_space="Hdiv"
        )

        # block matrix [[A B], [C I]] (without time step size dt in the off-diagonals)
        c = 2 / (self.gamma - 1)
        self._B = -c / 2 * self._ST @ self.mass_ops.M3
        self._C = 1 / 2 * self._S

        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(self.mass_ops.M2n)

        self._schur_solver = SchurSolver(
            self._velocity_mass.M,
            self._B @ self._C,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # temporaries
        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
        self._u_prev = self.variables.u.spline.vector.space.zeros()
        self._q_tmp1 = self.variables.q.spline.vector.space.zeros()
        self._q_w = self.variables.q.spline.vector.space.zeros()
        self._byn = self._B.codomain.zeros()

    def _update_weight(self, w):
        self._S.update_coeffs(w)
        self._ST.update_coeffs(w)

    def _solve(self, un, qn, dt):
        """One Crank-Nicolson solve with the current weight: new u and q coeffs (in the temporaries)."""
        byn = self._B.dot(qn, out=self._byn)
        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        _u = un.copy(out=self._u_tmp2)
        _u += un1
        qn1 = self._C.dot(_u, out=self._q_tmp1)
        qn1 *= -dt
        qn1 += qn
        return un1, qn1, info

    @profile
    def __call__(self, dt):
        un = self.variables.u.spline.vector
        qn = self.variables.q.spline.vector

        # inertia of the total density at t^n (perturbative models)
        self._velocity_mass.update()

        self._update_weight(qn)
        un1, qn1, info = self._solve(un, qn, dt)

        it = 0
        if self.options.weights == "midpoint":
            tol = self.options.nonlin_solver.tol
            converged = False
            for it in range(1, self.options.nonlin_solver.maxiter + 1):
                w = qn.copy(out=self._q_w)
                w += qn1
                w *= 0.5
                self._update_weight(w)
                un1.copy(out=self._u_prev)
                un1, qn1, info = self._solve(un, qn, dt)
                self._u_prev -= un1
                change = xp.sqrt(self._u_prev.inner(self._u_prev))
                if change <= tol * max(xp.sqrt(un1.inner(un1)), 1e-300):
                    converged = True
                    break
            if not converged and MPI.COMM_WORLD.Get_rank() == 0:
                logger.warning(f"SoundWaveNonlinearWithQ: Picard iteration not converged in {it} iterations.")

        diffs = self.update_feec_variables(u=un1, q=qn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for SoundWaveNonlinearWithQ: {info['success']}")
            logger.info(f"Iterations for SoundWaveNonlinearWithQ: {info['niter']}")
            logger.info(f"Picard iterations for SoundWaveNonlinearWithQ: {it}")
            logger.info(f"Maxdiff u2 for SoundWaveNonlinearWithQ: {diffs['u']}")
            logger.info(f"Maxdiff q3 for SoundWaveNonlinearWithQ: {diffs['q']}")
            logger.info("")
