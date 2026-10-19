import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.kronecker_block import basis_projection_operator, derivative_operator
from struphy.feec.utilities import LocalRotationMatrix
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions, MultiGridPreconditioner
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenPropagator(Propagator):
    r"""FEEC discretization of the following equations:
    find :math:`\tilde{\mathbf{U}} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\tilde{\mathbf{B}} \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x &= \int_{\Omega} \tilde{\mathbf{B}} \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}
        \\[2mm]
        \frac{\partial \tilde{\mathbf{B}}}{\partial t} &= \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)

    Time discretization:

    - implicit: Crank-Nicolson (implicit mid-point)
    - explicit: explicit RK methods from ButcherTableau

    System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenPropagator`.

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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`ShearAlfvenPropagator`.

        Parameters
        ----------
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the velocity variable.
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by implicit or explicit operators.
        precond : LiteralOptions.OptsShearAlfvenPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the linear solve: of the mass-matrix block (``"MassMatrixPreconditioner"``,
            ``"PolarMassMatrixPreconditioner"``), or of the whole Schur operator
            :math:`\mathbb M^{2,n} + \frac{\Delta t^2}{4} \mathbb T^\top \mathbb C^\top \mathbb M^2 \mathbb C \mathbb T`:
            a geometric multigrid V-cycle (``"MultiGrid"``,
            :class:`~struphy.linear_algebra.multigrid.preconditioner.MultiGridPreconditioner`) or the Kronecker
            (fast diagonalization) preconditioner (``"StiffnessPreconditioner"``,
            :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` with the operator
            :math:`\mathbb C \tilde{\mathbb T}`, where :math:`\tilde{\mathbb T}` is the Kronecker approximation of the
            basis projection operator, :func:`~struphy.feec.kronecker_block.basis_projection_operator`; block Jacobi in
            the velocity components, ``u_space="Hdiv"`` and no polar splines), with the auxiliary space correction of
            the :math:`\mathbf E \times \mathbf B` flows if ``exb_correction`` (see there). Both need the implicit
            algorithm with ``solver="pcg"``. The Schur operator depends on the time step, so these preconditioners are
            built in the setup (:meth:`prepare_time_step`) and updated if the time step changes.
            The multigrid preconditioner is experimental: on the TAE example (HollowTorus, 8x32x4) it needs more PCG
            iterations than the mass matrix preconditioner and is much slower (the smoother does not treat the large
            kernel of the curl term on 2-forms). The Kronecker preconditioner (with ``exb_correction``) needs about 11
            instead of 31 PCG iterations per step there (dt = 0.1), with larger gains for larger time steps.
        exb_correction : bool, default=True
            For ``precond="StiffnessPreconditioner"``: add the auxiliary space correction
            :math:`\mathbb N P_\phi \mathbb N^\top` (:class:`~struphy.feec.preconditioner.AuxiliarySpacePreconditioner`)
            of the :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi`,
            :math:`\mathbb N = \hat \Pi^2[\mathbf B_0 \times \,\cdot\,] \, \mathbb G` with :math:`\phi \in V^0_h`. The block
            Jacobi preconditioner in the velocity components does not reduce the incompressible shear Alfven modes
            with small parallel wave number (their velocity components cancel in the compression), which this
            correction resolves; :math:`P_\phi` is the Kronecker (fast diagonalization) preconditioner of
            :math:`\mathbb N^\top S \mathbb N` on :math:`V^0_h`.
        multigrid : MultiGridOptions, default=None
            Options of the multigrid preconditioner (if ``precond="MultiGrid"``); defaults to
            ``MultiGridOptions()``.
        solver_params : SolverParameters, default=None
            Solver controls; defaults to ``SolverParameters()``.
        butcher : ButcherTableau, default=None
            Explicit Runge-Kutta tableau when ``algo="explicit"``.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsShearAlfvenPrecond = "MassMatrixPreconditioner"
        exb_correction: bool = True
        multigrid: MultiGridOptions = None
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsShearAlfvenPrecond)
            if self.precond in ("MultiGrid", "StiffnessPreconditioner"):
                assert self.algo == "implicit", f'precond="{self.precond}" requires algo="implicit".'
                assert self.solver == "pcg", f'precond="{self.precond}" requires solver="pcg".'
            if self.precond == "StiffnessPreconditioner":
                assert self.u_space == "Hdiv", 'precond="StiffnessPreconditioner" requires u_space="Hdiv".'
            if self.precond == "MultiGrid" and self.multigrid is None:
                self.multigrid = MultiGridOptions()

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

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
        u_space = self.options.u_space

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = "M" + self.derham.space_to_form[u_space] + "n"
        id_T = "T" + self.derham.space_to_form[u_space]

        # call operators
        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)
        _M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner (the preconditioners of the Schur operator depend on dt and are built in __call__)
        self._id_M = id_M
        self._multigrid_dt = None
        self._schur_precond_dt = None
        if self.options.precond == "StiffnessPreconditioner":
            assert not self.derham.polar_splines, 'precond="StiffnessPreconditioner" does not support polar splines.'
            assert not self.derham.with_local_projectors, (
                'precond="StiffnessPreconditioner" needs the global commuting projectors.'
            )
            # Kronecker approximation of the curl of the basis projection (built once, independent of dt)
            self._kron_curl_T = derivative_operator(self.derham, "curl") @ basis_projection_operator(_T, self.derham)
            if self.options.exb_correction:
                # E x B flows B0 x grad(phi): 2-form components eps_ijk b1_j d_k phi (b1: covariant components of B0)
                eq = self.basis_ops.weights["eq_mhd"]
                rot_b1 = LocalRotationMatrix(eq.b1_1, eq.b1_2, eq.b1_3)
                fun = [
                    [(lambda e1, e2, e3, m=m, n=n: rot_b1(e1, e2, e3)[:, :, :, m, n]) for n in range(3)]
                    for m in range(3)
                ]
                Pi_exb = self.basis_ops.create_basis_op(fun, "Hcurl", "Hdiv", assemble=True, name="Pi_ExB")
                grad = self.derham.grad
                self._exb_N = Pi_exb @ grad
                self._exb_NT = grad.T @ Pi_exb.T
                self._kron_exb = basis_projection_operator(Pi_exb, self.derham) @ derivative_operator(
                    self.derham, "grad"
                )
                self._kron_curl_T_exb = self._kron_curl_T @ self._kron_exb
        if self.options.precond in (None, "MultiGrid", "StiffnessPreconditioner"):
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            self._B = -1 / 2 * _T.T @ curl.T @ _M2
            self._C = 1 / 2 * curl @ _T

            # instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _A,
                _BC,
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )

            # pre-allocate arrays
            self._byn = self._B.codomain.zeros()

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
            _f1 = A_inv @ _T.T @ curl.T @ _M2
            _f2 = curl @ _T

            # allocate output of vector field
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                _f2.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.u.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    def prepare_time_step(self, dt: float):
        """Preconditioner of the Schur operator for the time step ``dt`` (multigrid or Kronecker): built in the setup,
        updated or rebuilt when dt changes."""
        if self.options.algo != "implicit":
            return
        # the multigrid preconditioner of the Schur operator A - dt^2 B C: updated (re-using the coarse mass and
        # derivative operators) when dt changes
        if self.options.precond == "MultiGrid" and dt != self._multigrid_dt:
            schur = self._schur_solver.A - dt**2 * self._schur_solver.BC
            if self._schur_solver.precond is None:
                self._schur_solver.precond = MultiGridPreconditioner(
                    schur, self.derham, self.domain, self.options.multigrid, mass_ops=self.mass_ops
                )
            else:
                self._schur_solver.precond.update(schur)
            self._multigrid_dt = dt

        # the Kronecker preconditioner of S = M + dt^2/4 T^T C^T M2 C T (plus the E x B correction), rebuilt when dt
        # changes
        if self.options.precond == "StiffnessPreconditioner" and dt != self._schur_precond_dt:
            mid = (dt**2 / 4) * self.mass_ops.M2
            pc = preconditioner.StiffnessPreconditioner(
                self.mass_ops, self._kron_curl_T, sigma=1.0, mid=mid, stab=self._id_M
            )
            if self.options.exb_correction:
                # N^T S N = N^T M N + dt^2/4 (C T N)^T M2 (C T N) on V0
                pc_phi = preconditioner.StiffnessPreconditioner(
                    self.mass_ops,
                    [self._kron_exb, self._kron_curl_T_exb],
                    sigma=0.0,
                    mid=[self._id_M, mid],
                )
                pc = preconditioner.AuxiliarySpacePreconditioner(pc, [(self._exb_N, self._exb_NT, pc_phi)])
            self._schur_solver.precond = pc
            self._schur_precond_dt = dt

    @profile
    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # preconditioner for the current time step (no-op unless dt changed since the setup)
            self.prepare_time_step(dt)

            # solve for new u coeffs
            byn = self._B.dot(bn, out=self._byn)

            un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

            # new b coeffs
            _u = un.copy(out=self._u_tmp2)
            _u += self._u_tmp1
            bn1 = self._C.dot(_u, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(u=un1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for ShearAlfven: {info['success']}")
                logger.info(f"Iterations for ShearAlfven: {info['niter']}")
                logger.info(f"Maxdiff up for ShearAlfven: {diffs['u']}")
                logger.info(f"Maxdiff b2 for ShearAlfven: {diffs['b']}")
                logger.info("")
