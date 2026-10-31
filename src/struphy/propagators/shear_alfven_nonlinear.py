import logging
from dataclasses import dataclass

import cunumpy as xp
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.density_weighted import VelocityMass
from struphy.feec.variational_utilities import Hdiv0_transport_operator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenNonlinear(Propagator):
    r"""Energy conserving shear-Alfvén step of the terms that are nonlinear in the magnetic field perturbation, in
    the perturbative MHD models (zero-flow equilibrium, :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`), with the
    velocity and the magnetic field in :math:`H(\textnormal{div})`.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` and
    :math:`\tilde{\mathbf B} \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \rho \frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x &=
        \int_{\Omega} \tilde{\mathbf{B}} \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf V) \,\textnormal d \mathbf x
        \qquad \forall \,\mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        \frac{\partial \tilde{\mathbf{B}}}{\partial t} &= \nabla\times(\tilde{\mathbf{U}} \times \tilde{\mathbf{B}})\,,

    with the total density :math:`\rho = \rho_0 + \tilde \rho` (fixed in this step): the terms
    :math:`(\nabla \times \tilde{\mathbf B}) \times \tilde{\mathbf B}` of the Lorentz force (weak form) and
    :math:`\nabla \times (\tilde{\mathbf U} \times \tilde{\mathbf B})` of the induction equation, which are quadratic in
    the perturbations. This is the step of
    :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` with the weight
    :math:`\mathbf B_0 \to \tilde{\mathbf B}`; together with it and
    :class:`~struphy.propagators.background_current.BackgroundCurrent` it gives the Lorentz force
    :math:`(\nabla \times \mathbf B) \times \mathbf B - (\nabla \times \mathbf B_0) \times \mathbf B_0` and the induction
    equation :math:`\partial_t \tilde{\mathbf B} = \nabla \times (\tilde{\mathbf U} \times \mathbf B)`. The step conserves
    the energy :math:`\frac 12 \int \rho |\tilde{\mathbf U}|^2 + \frac 12 \int |\tilde{\mathbf B}|^2`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h` and
    :math:`\mathbf b` of :math:`\tilde{\mathbf B}_h \in V^2_h`,

    .. math::

        \mathbb M^2_\rho \dot{\mathbf u} = \mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b\,, \qquad
        \dot{\mathbf b} = - \mathbb T[\mathbf b] \mathbf u\,,

    with :math:`\mathbb T[\mathbf w] = \mathbb C \hat \Pi^1[(\hat{\mathbf w}^2 \times \vec{\boldsymbol\Lambda}^2) / \sqrt g]`
    (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`, as
    :math:`\mathbb C \mathbb T^2` of :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` with
    :math:`\mathbf B_0 \to \mathbf w`), the mass matrix :math:`\mathbb M^2`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M2`) and the mass matrix :math:`\mathbb M^2_\rho` of the total
    density (:class:`~struphy.feec.density_weighted.VelocityMass`, with the constructor argument ``rho_field``; without
    it :math:`\mathbb M^2_{\rho_0}`). The induction equation holds strongly in :math:`V^2_h` (the divergence of
    :math:`\mathbf b` is conserved), and the energy :math:`\frac 12 \mathbf u^\top \mathbb M^2_\rho \mathbf u + \frac 12
    \mathbf b^\top \mathbb M^2 \mathbf b` is conserved for any weight in :math:`\mathbb T`.

    **Time discretization.** Crank-Nicolson in :math:`(\mathbf u, \mathbf b)` with the weight :math:`\mathbf w` of
    :math:`\mathbb T` fixed in the step (``Options.weights``):

    .. math::

        \mathbb M^2_\rho \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t} = \mathbb T[\mathbf w]^\top \mathbb M^2
        \mathbf b^{n+\frac12}\,, \qquad
        \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} = - \mathbb T[\mathbf w] \mathbf u^{n+\frac12}\,,

    which conserves the discrete energy exactly (up to the solver tolerances): with ``"frozen"`` (default)
    :math:`\mathbf w = \mathbf b^n`, one linear solve per step (first order in time); with ``"midpoint"``
    :math:`\mathbf w = \mathbf b^{n+\frac12}`, the implicit midpoint rule (second order), solved by Picard iteration
    (``Options.nonlin_solver``). Each linear solve eliminates :math:`\mathbf b` with
    :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, with the symmetric positive definite Schur complement

    .. math::

        S = \mathbb M^2_\rho + \frac{\Delta t^2}{4} \mathbb T[\mathbf w]^\top \mathbb M^2 \mathbb T[\mathbf w]\,,

    solved by ``Options.solver`` with the mass matrix preconditioner ``Options.precond`` of
    :math:`\mathbb M^2_{\rho_0}`.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenNonlinear`.

        Attributes
        ----------
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        b : FEECVariable
            Magnetic field perturbation in ``"Hdiv"`` space.
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
            assert new.space == "Hdiv"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

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
        """Configuration options for :class:`ShearAlfvenNonlinear`.

        Parameters
        ----------
        weights : LiteralOptions.OptsTimeWeights, default="frozen"
            Weight of the transport operator: :math:`\\mathbf b^n` (``"frozen"``, one linear solve) or
            :math:`\\mathbf b^{n+1/2}` (``"midpoint"``, Picard iteration).
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
                "ShearAlfvenNonlinear: only the Picard iteration is implemented."
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

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        self._velocity_mass = VelocityMass(
            self.derham, self.domain, self.mass_ops, self.projected_equil, rho_field=self.rho_field
        )

        # transport operator T[w] and its transpose (weight: magnetic field perturbation)
        self._S = Hdiv0_transport_operator(self.derham, u_space="Hdiv", phys_domain=self.domain)
        self._ST = Hdiv0_transport_operator(self.derham, transposed=True, u_space="Hdiv", phys_domain=self.domain)

        # block matrix [[A B], [C I]] (without time step size dt in the off-diagonals)
        self._B = -1 / 2 * self._ST @ self.mass_ops.M2
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
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()
        self._b_w = self.variables.b.spline.vector.space.zeros()
        self._byn = self._B.codomain.zeros()

    def _update_weight(self, w):
        self._S.update_coeffs(w)
        self._ST.update_coeffs(w)

    def _solve(self, un, bn, dt):
        """One Crank-Nicolson solve with the current weight: new u and b coeffs (in the temporaries)."""
        byn = self._B.dot(bn, out=self._byn)
        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        _u = un.copy(out=self._u_tmp2)
        _u += un1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn
        return un1, bn1, info

    @profile
    def __call__(self, dt):
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # inertia of the total density at t^n (perturbative models)
        self._velocity_mass.update()

        self._update_weight(bn)
        un1, bn1, info = self._solve(un, bn, dt)

        it = 0
        if self.options.weights == "midpoint":
            tol = self.options.nonlin_solver.tol
            converged = False
            for it in range(1, self.options.nonlin_solver.maxiter + 1):
                w = bn.copy(out=self._b_w)
                w += bn1
                w *= 0.5
                self._update_weight(w)
                un1.copy(out=self._u_prev)
                un1, bn1, info = self._solve(un, bn, dt)
                self._u_prev -= un1
                change = xp.sqrt(self._u_prev.inner(self._u_prev))
                if change <= tol * max(xp.sqrt(un1.inner(un1)), 1e-300):
                    converged = True
                    break
            if not converged and MPI.COMM_WORLD.Get_rank() == 0:
                logger.warning(f"ShearAlfvenNonlinear: Picard iteration not converged in {it} iterations.")

        diffs = self.update_feec_variables(u=un1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for ShearAlfvenNonlinear: {info['success']}")
            logger.info(f"Iterations for ShearAlfvenNonlinear: {info['niter']}")
            logger.info(f"Picard iterations for ShearAlfvenNonlinear: {it}")
            logger.info(f"Maxdiff u2 for ShearAlfvenNonlinear: {diffs['u']}")
            logger.info(f"Maxdiff b2 for ShearAlfvenNonlinear: {diffs['b']}")
            logger.info("")
