import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.density_weighted import HdivDensityOperators
from struphy.feec.variational_utilities import L2_transport_operator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class MomentumAdvection(Propagator):
    r"""Advection of mass and momentum in the perturbative fluid models (zero-flow equilibrium), with the velocity in
    :math:`H(\textnormal{div})`: the density perturbation and the velocity are transported by the velocity, conserving
    the mass and the kinetic energy :math:`\frac 12 \int \rho |\tilde{\mathbf U}|^2` of the total density
    :math:`\rho = \rho_0 + \tilde \rho`.

    **Continuous problem.** Find :math:`\tilde \rho \in L^2` and :math:`\tilde{\mathbf U} \in H(\textnormal{div})` such
    that

    .. math::

        \frac{\partial \tilde \rho}{\partial t} + \nabla \cdot (\rho \tilde{\mathbf U}) &= 0\,,
        \\[2mm]
        \int_\Omega \rho \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \,\textnormal d \mathbf x
        + \int_\Omega \rho \, \mathbf V \cdot (\boldsymbol \omega \times \tilde{\mathbf U}) \,\textnormal d \mathbf x
        - \int_\Omega \nabla \cdot (\rho \mathbf V) \, \frac{|\tilde{\mathbf U}|^2}{2} \,\textnormal d \mathbf x &= 0
        \qquad \forall \, \mathbf V \in H(\textnormal{div})\,,

    with the vorticity :math:`\boldsymbol \omega = \nabla \times \tilde{\mathbf U}` (in the weak sense). This is
    :math:`\rho (\partial_t \tilde{\mathbf U} + \tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U}) = 0` in rotational
    form, :math:`\tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U} = \boldsymbol \omega \times \tilde{\mathbf U} +
    \nabla \frac{|\tilde{\mathbf U}|^2}{2}`, with the last term integrated by parts (the boundary term vanishes for
    :math:`\tilde{\mathbf U} \cdot \mathbf n = 0`). The vorticity term does no work, and the work of the last term is
    the change of the kinetic energy by the transport of the density, such that
    :math:`\frac 12 \int \rho |\tilde{\mathbf U}|^2` and :math:`\int \tilde \rho` are conserved.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf n` of :math:`\tilde \rho_h \in V^3_h`,
    :math:`\boldsymbol \rho = \boldsymbol \rho_0 + \mathbf n` of the total density (:math:`\boldsymbol \rho_0` the
    coefficients of the projected equilibrium density 3-form, ``projected_equil.n3``) and :math:`\mathbf u` of
    :math:`\tilde{\mathbf U}_h \in V^2_h`, the kinetic energy is
    :math:`E = \frac 12 \mathbf u^\top \mathbb M^2_\rho \mathbf u = \frac 12 \boldsymbol \rho^\top k(\mathbf u)`, with the
    density-weighted mass matrix :math:`\mathbb M^2_\rho` and the linear form
    :math:`k(\mathbf u)_{ijk} = \int \Lambda^3_{ijk} |\tilde{\mathbf U}_h|^2 \frac{1}{\sqrt g}\,\textnormal d \mathbf x`
    (:class:`~struphy.feec.density_weighted.HdivDensityOperators`, same quadrature, so that the identity is exact).
    With the transport operator :math:`\mathbb L[\mathbf w] \mathbf u = \mathbb D \hat \Pi^2[\hat w^3 \mathbf u]`
    (:class:`~struphy.feec.variational_utilities.L2_transport_operator`) and the antisymmetric matrix
    :math:`\mathbb W[\mathbf w, \boldsymbol \omega]` of :math:`\int w\, \mathbf v \cdot (\boldsymbol \omega \times
    \mathbf u)` (:class:`~struphy.feec.density_weighted.HdivDensityOperators`), with the weak curl
    :math:`\boldsymbol \omega = (\mathbb M^1)^{-1} \mathbb C^\top \mathbb M^2 \mathbf u`,

    .. math::

        \dot{\mathbf n} = - \mathbb L[\boldsymbol \rho] \mathbf u\,, \qquad
        \mathbb M^2_\rho \dot{\mathbf u} + \mathbb W[\boldsymbol \rho, \boldsymbol \omega] \mathbf u
        - \frac 12 \mathbb L[\boldsymbol \rho]^\top k(\mathbf u) = 0\,,

    which conserves :math:`E`, since :math:`\dot E = \frac 12 \dot{\mathbf n}^\top k(\mathbf u) + \mathbf u^\top
    \mathbb M^2_\rho \dot{\mathbf u}` and :math:`\mathbb W` is antisymmetric.

    **Time discretization.** A discrete gradient of :math:`E` in :math:`(\mathbf n, \mathbf u)`:

    .. math::

        \frac{\mathbf n^{n+1} - \mathbf n^n}{\Delta t} &= - \mathbb L[\mathbf w] \, \mathbf u^{n+\frac12}\,,
        \\[2mm]
        \mathbb M^2_{\bar \rho} \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        + \mathbb W[\mathbf w, \boldsymbol \omega^w] \, \mathbf u^{n+\frac12}
        - \frac 12 \mathbb L[\mathbf w]^\top \frac{k(\mathbf u^n) + k(\mathbf u^{n+1})}{2} &= 0\,,

    with :math:`\bar{\boldsymbol \rho} = \boldsymbol \rho_0 + (\mathbf n^n + \mathbf n^{n+1})/2` and
    :math:`\mathbf u^{n+\frac12} = (\mathbf u^n + \mathbf u^{n+1})/2`. Since
    :math:`E^{n+1} - E^n = \frac 12 (\mathbf n^{n+1} - \mathbf n^n)^\top \frac{k(\mathbf u^n) + k(\mathbf u^{n+1})}{2} +
    (\mathbf u^{n+1} - \mathbf u^n)^\top \mathbb M^2_{\bar \rho} \mathbf u^{n+\frac12}` exactly, the kinetic energy is
    conserved exactly for any weight :math:`\mathbf w` and vorticity :math:`\boldsymbol \omega^w` (up to the tolerances of
    the iteration), and the mass because :math:`\int \mathbb D \,\cdot = 0`. The option ``weights`` sets
    :math:`\mathbf w` and :math:`\boldsymbol \omega^w`: ``"frozen"`` (default) :math:`\boldsymbol \rho_0 + \mathbf n^n`
    and the vorticity of :math:`\mathbf u^n` (first order in time), ``"midpoint"`` :math:`\bar{\boldsymbol \rho}` and
    the vorticity of :math:`\mathbf u^{n+\frac12}` (second order). The kinetic energy is cubic in
    :math:`(\mathbf n, \mathbf u)`, so the system is nonlinear for both options; it is solved by Picard iteration
    (``Options.nonlin_solver``): with the current iterate, :math:`\mathbf n^{n+1}` from the first equation, then
    :math:`\mathbf u^{n+1}` from the linear system with the non-symmetric matrix
    :math:`\mathbb M^2_{\bar \rho} + \frac{\Delta t}{2} \mathbb W` (``Options.solver``, preconditioned with the mass
    matrix preconditioner of :math:`\mathbb M^2_{\rho_0}`).
    """

    class Variables:
        """Container for variables advanced by :class:`MomentumAdvection`.

        Attributes
        ----------
        n : FEECVariable
            Density perturbation variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._n: FEECVariable = None
            self._u: FEECVariable = None

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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MomentumAdvection`.

        Parameters
        ----------
        weights : LiteralOptions.OptsTimeWeights, default="frozen"
            Density weight of the transport and vorticity of the vorticity term: at :math:`t^n` (``"frozen"``) or at
            :math:`t^{n+1/2}` (``"midpoint"``, updated in every Picard iteration).
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            Solver of the (non-symmetric) linear system of the velocity.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the linear system, built from :math:`\\mathbb M^2_{\\rho_0}` (``M2n``).
        solver_params : SolverParameters, default=None
            Controls of the linear solve; defaults to ``SolverParameters(tol=1e-13)``.
        solver_M1 : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the inner ``M1`` inversion of the weak curl.
        solver_params_M1 : SolverParameters, default=None
            Controls of the inner ``M1`` inversion; defaults to ``SolverParameters(tol=1e-14)``.
        nonlin_solver : NonlinearSolverParameters, default=None
            Controls of the Picard iteration (``tol``: relative change of the velocity); defaults to
            ``NonlinearSolverParameters(tol=1e-12)``.
        """

        weights: LiteralOptions.OptsTimeWeights = "frozen"
        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        solver_M1: LiteralOptions.OptsSymmSolver = "pcg"
        solver_params_M1: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.weights, LiteralOptions.OptsTimeWeights)
            check_option(self.solver, LiteralOptions.OptsGenSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.solver_M1, LiteralOptions.OptsSymmSolver)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters(tol=1e-13)
            if self.solver_params_M1 is None:
                self.solver_params_M1 = SolverParameters(tol=1e-14)
            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(tol=1e-12)
            assert self.nonlin_solver.type == "Picard", "MomentumAdvection: only the Picard iteration is implemented."

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
        self._ops = HdivDensityOperators(self.derham, self.domain, self.mass_ops)
        self._L = L2_transport_operator(self.derham, u_space="Hdiv", phys_domain=self.domain)
        self._LT = L2_transport_operator(self.derham, transposed=True, u_space="Hdiv", phys_domain=self.domain)

        # weak curl of the velocity
        M1 = self.mass_ops.M1
        self._M1_inv = inverse(
            M1,
            self.options.solver_M1,
            pc=preconditioner.MassMatrixPreconditioner(M1),
            tol=self.options.solver_params_M1.tol,
            maxiter=self.options.solver_params_M1.maxiter,
        )
        self._curlT_M2 = self.derham.curl.T @ self.mass_ops.M2

        # linear solver of the velocity (the operator is set in __call__)
        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(self.mass_ops.M2n)
        self._solver = inverse(
            self.mass_ops.M2n,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

        # temporaries
        Vu = self.variables.u.spline.vector.space
        Vn = self.variables.n.spline.vector.space
        self._u1 = Vu.zeros()
        self._uh = Vu.zeros()
        self._du = Vu.zeros()
        self._rhs = Vu.zeros()
        self._tmp_u = Vu.zeros()
        self._omega = self._curlT_M2.codomain.zeros()
        self._omega_rhs = self._curlT_M2.codomain.zeros()
        self._n1 = Vn.zeros()
        self._rho_w = Vn.zeros()
        self._rho_bar = Vn.zeros()
        self._k0 = Vn.zeros()
        self._k1 = Vn.zeros()
        self._tmp_n = Vn.zeros()

    def _total_density(self, n_a, n_b, out):
        """rho_0 + (n_a + n_b) / 2 (total density at the midpoint of two perturbations)."""
        n_a.copy(out=out)
        out += n_b
        out *= 0.5
        out += self.projected_equil.n3
        return out

    def _update_weights(self, rho_w, u_w):
        """Transport operators with the density weight rho_w and vorticity matrix with the vorticity of u_w."""
        self._L.update_coeffs(rho_w)
        self._LT.update_coeffs(rho_w)
        self._curlT_M2.dot(u_w, out=self._omega_rhs)
        self._M1_inv.dot(self._omega_rhs, out=self._omega)
        self._ops.update_vorticity(rho_w, self._omega)

    @profile
    def __call__(self, dt):
        nn = self.variables.n.spline.vector
        un = self.variables.u.spline.vector
        midpoint = self.options.weights == "midpoint"
        tol = self.options.nonlin_solver.tol

        k0 = self._ops.kinetic_form(un, out=self._k0)
        if not midpoint:
            self._update_weights(self._total_density(nn, nn, self._rho_w), un)

        u1 = un.copy(out=self._u1)
        n1 = nn.copy(out=self._n1)
        converged = False
        for it in range(self.options.nonlin_solver.maxiter):
            uh = un.copy(out=self._uh)
            uh += u1
            uh *= 0.5
            if midpoint:
                self._update_weights(self._total_density(nn, n1, self._rho_w), uh)

            # continuity equation
            self._L.dot(uh, out=self._tmp_n)
            self._tmp_n *= -dt
            nn.copy(out=n1)
            n1 += self._tmp_n

            # mass matrix of the midpoint density and averaged kinetic form
            self._ops.update_density(self._total_density(nn, n1, self._rho_bar))
            k1 = self._ops.kinetic_form(u1, out=self._k1)
            k1 += k0
            k1 *= 0.5

            # (M + dt/2 W) u^{n+1} = (M - dt/2 W) u^n + dt/2 L^T kbar
            rhs = self._ops.M.dot(un, out=self._rhs)
            self._ops.W.dot(un, out=self._tmp_u)
            self._tmp_u *= -dt / 2
            rhs += self._tmp_u
            self._LT.dot(k1, out=self._tmp_u)
            self._tmp_u *= dt / 2
            rhs += self._tmp_u

            # zero initial guess: a converged iterate as initial guess makes BiCGStab break down (zero residual)
            self._solver.linop = self._ops.M + (dt / 2) * self._ops.W
            self._du *= 0.0
            u_new = self._solver.dot(rhs, out=self._du)

            # Picard update and relative change
            u_new.copy(out=self._tmp_u)
            self._tmp_u -= u1
            change = xp.sqrt(self._tmp_u.inner(self._tmp_u))
            size = xp.sqrt(u_new.inner(u_new))
            u_new.copy(out=u1)
            if change <= tol * max(size, 1e-300):
                converged = True
                break

        # density consistent with the final velocity
        uh = un.copy(out=self._uh)
        uh += u1
        uh *= 0.5
        self._L.dot(uh, out=self._tmp_n)
        self._tmp_n *= -dt
        nn.copy(out=n1)
        n1 += self._tmp_n

        if not converged and MPI.COMM_WORLD.Get_rank() == 0:
            logger.warning(f"MomentumAdvection: Picard iteration not converged in {it + 1} iterations ({change = }).")

        diffs = self.update_feec_variables(n=n1, u=u1)

        if self.options.nonlin_solver.info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Picard iterations for MomentumAdvection: {it + 1}")
            logger.info(f"Maxdiff n3 for MomentumAdvection: {diffs['n']}")
            logger.info(f"Maxdiff u2 for MomentumAdvection: {diffs['u']}")
