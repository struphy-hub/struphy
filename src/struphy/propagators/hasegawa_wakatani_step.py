from dataclasses import dataclass
from typing import Callable, Literal, get_args
import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

class HasegawaWakataniStep(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`(n, \omega) \in H^1 \times H^1` such that

    .. math::

        &\int_\Omega\frac{\partial n}{\partial t} m \,\textrm d \mathbf x = \int_\Omega C(x, y)(\phi - n) \, m \,\textrm d \mathbf x - \int_\Omega \phi [n, m] \,\textrm d \mathbf x - \kappa \int_\Omega  \partial_y \phi \,m \,\textrm d \mathbf x - \nu \int_\Omega \nabla n \cdot \nabla m \,\textrm d \mathbf x \qquad \forall m \in H^1\,,
        \\[2mm]
        &\int_\Omega\frac{\partial \omega}{\partial t} \psi \,\textrm d \mathbf x = \int_\Omega C(x, y)(\phi - n) \, \psi \,\textrm d \mathbf x - \int_\Omega \phi [\omega, \psi] \,\textrm d \mathbf x - \nu \int_\Omega \nabla \omega \cdot \nabla \psi \,\textrm d \mathbf x \qquad \forall \psi \in H^1\,,

    where  :math:`\phi \in H^1` is a given stream function,
    :math:`C = C(x, y)`, :math:`\kappa` and :math:`\nu` are constants and
    :math:`[a, b] = \partial_x a \partial_y b - \partial_y a \partial_x b`.

    :ref:`time_discret`: explicit Runge-Kutta, see :class:`~struphy.ode.solvers.ODEsolverFEEC`.

    Parameters
    ----------
    n0 : StencilVector
        The density.

    omega0 : StencilVector
        The stream function.

    phi : SplineFuncion
        The potential.

    c_fun : str
        Defines the function c(x,y) in front of (phi - n).

    kappa, nu : float
        Equation parameters.

    algo : str
        See :class:`~struphy.ode.utils.ButcherTableau` for available algorithms.

    M0_solver : dict
        Solver parameters for M0 inversion.
    """

    class Variables:
        """Container for variables advanced by :class:`HasegawaWakataniStep`.

        Attributes
        ----------
        n : FEECVariable
            Density variable in ``"H1"`` space.
        omega : FEECVariable
            Vorticity variable in ``"H1"`` space.
        """
        def __init__(self):
            self._n: FEECVariable = None
            self._omega: FEECVariable = None

        @property
        def n(self) -> FEECVariable:
            return self._n

        @n.setter
        def n(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._n = new

        @property
        def omega(self) -> FEECVariable:
            return self._omega

        @omega.setter
        def omega(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._omega = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        """Configuration options for :class:`HasegawaWakataniStep`.

        Parameters
        ----------
        phi : FEECVariable, default=None
            Stream-function variable in ``"H1"`` space.
            If ``None``, a default ``FEECVariable(space="H1")`` is allocated.

        c_fun : {"const"}, default="const"
            Choice of coupling profile :math:`C(x,y)` used in the model.

        kappa : float, default=1.0
            Constant multiplying the background-gradient drift term.

        nu : float, default=0.01
            Diffusion coefficient in density and vorticity equations.

        butcher : ButcherTableau, default=None
            Butcher tableau for explicit Runge-Kutta integration.
            If ``None``, defaults to ``ButcherTableau()``.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used for ``M0`` inversions.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used with ``M0`` inversions.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """
        # specific literals
        OptsCfun = Literal["const"]
        # propagator options
        phi: FEECVariable = None
        c_fun: OptsCfun = "const"
        kappa: float = 1.0
        nu: float = 0.01
        butcher: ButcherTableau = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.c_fun, self.OptsCfun)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.butcher is None:
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

    @profile
    def allocate(self, verbose: bool = False):
        # default phi
        if self.options.phi is None:
            self.options.phi = FEECVariable(space="H1")
            self.options.phi.allocate(derham=self.derham, domain=self.domain)

        self._phi = self.options.phi.spline
        self._phi.vector[:] = 1.0
        self._phi.vector.update_ghost_regions()

        # default c-function
        if self.options.c_fun == "const":
            c_fun = lambda e1, e2, e3: 0.0 + 0.0 * e1
        else:
            raise NotImplementedError(f"{self.options.c_fun =} is not available.")

        # expose equation parameters
        self._kappa = self.options.kappa
        self._nu = self.options.nu

        # get quadrature grid of V0
        pts = [grid.flatten() for grid in self.derham.V0splines.quad_grid_pts[0]]
        mesh_pts = xp.meshgrid(*pts, indexing="ij")

        # evaluate c(x, y) and metric coeff at local quadrature grid and multiply
        self._weights = c_fun(*mesh_pts)
        self._weights *= self.domain.jacobian_det(*mesh_pts)

        # evaluate phi at local quadrature grid
        self._spans, self._bns, self._bnd = self.derham.prepare_eval_tp_fixed(pts)
        self._phi_at_pts = self._phi.eval_tp_fixed_loc(self._spans, self._bns)

        # Jacobain at quad grid
        self._jac_det = self.domain.jacobian_det(*mesh_pts)
        self._jac_inv = self.domain.jacobian_inv(*mesh_pts, change_out_order=True)
        self._jac_invT = self.domain.jacobian_inv(*mesh_pts, change_out_order=True, transposed=True)

        # grad operator
        grad = self.derham.grad

        # mass operators
        M0 = self.mass_ops.M0
        M1 = self.mass_ops.M1
        M0c = self.mass_ops.create_weighted_mass(
            "H1",
            "H1",
            name="M0c",
            weights=[[self._weights]],
            assemble=True,
        )

        self._M1hw_weights = []
        for m in range(3):
            self._M1hw_weights += [[None, None, None]]

        self._phi_5d = xp.zeros((*self._phi_at_pts.shape, 3, 3), dtype=float)
        self._tmp_5d = xp.zeros((*self._phi_at_pts.shape, 3, 3), dtype=float)
        self._tmp_5dT = xp.zeros((3, 3, *self._phi_at_pts.shape), dtype=float)
        self._phi_5d[:, :, :, 0, 1] = self._phi_at_pts * self._jac_det
        self._phi_5d[:, :, :, 1, 0] = -self._phi_at_pts * self._jac_det
        self._tmp_5d[:] = self._jac_inv @ self._phi_5d @ self._jac_invT
        self._tmp_5dT[:] = xp.transpose(self._tmp_5d, axes=(3, 4, 0, 1, 2))

        self._M1hw_weights[0][1] = self._tmp_5dT[0, 1, :, :, :]
        self._M1hw_weights[1][0] = self._tmp_5dT[1, 0, :, :, :]

        # self._self._M1hw_weights = ["DFinv", [self._phi_5d], "DFinvT"]
        self._M1hw = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            name="M1hw",
            weights=self._M1hw_weights,
            assemble=True,
        )

        # inverse M0 mass matrix
        solver = self.options.solver
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M0)
        # solver_params = deepcopy(M0_solver)  # need a copy to pop, otherwise testing fails
        # solver_params.pop("type")
        self._info = self.options.solver_params.info
        M0_inv = inverse(
            M0,
            solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )

        # basis projection operator
        df_12 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[0, 1, :, :, :]
        df_22 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[1, 1, :, :, :]
        df_32 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[2, 1, :, :, :]
        fun = [[df_12, df_22, df_32]]
        # fun = [[None, lambda e1, e2, e3: 1.0 + 0.0 * e1, None]]
        self._BPO = self.basis_ops.create_basis_op(
            fun,
            "Hcurl",
            "H1",
            name="dy_phi",
            assemble=True,
        )
        # logger.info(f"{self._BPO._dof_mat.blocks = }")

        # pre-allocated helper arrays
        n0 = self.variables.n.spline.vector
        omega0 = self.variables.omega.spline.vector

        self._tmp1 = n0.space.zeros()
        tmp2 = n0.space.zeros()
        self._tmp3 = n0.space.zeros()
        tmp4 = n0.space.zeros()
        tmp5 = n0.space.zeros()

        # rhs-callables for explicit ode solve
        terms1_n = -M0c + grad.T @ self._M1hw @ grad - self.options.nu * grad.T @ M1 @ grad
        terms1_phi = M0c
        terms1_phi_strong = -self.options.kappa * self._BPO @ grad

        terms2_omega = grad.T @ self._M1hw @ grad - self.options.nu * grad.T @ M1 @ grad
        terms2_n = -M0c
        terms2_phi = M0c

        out1 = n0.space.zeros()
        out2 = omega0.space.zeros()

        def f1(t, n, omega, out=out1):
            terms1_n.dot(n, out=self._tmp1)
            terms1_phi.dot(self._phi.vector, out=tmp2)
            self._tmp1 += tmp2
            M0_inv.dot(self._tmp1, out=out)
            terms1_phi_strong.dot(self._phi.vector, out=tmp2)
            out += tmp2
            out.update_ghost_regions()
            return out

        def f2(t, n, omega, out=out2):
            terms2_omega.dot(omega, out=self._tmp3)
            terms2_n.dot(n, out=tmp4)
            terms2_phi.dot(self._phi.vector, out=tmp5)
            self._tmp3 += tmp4
            self._tmp3 += tmp5
            M0_inv.dot(self._tmp3, out=out)
            out.update_ghost_regions()
            return out

        vector_field = {n0: f1, omega0: f2}
        self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

    def __call__(self, dt):
        # update time-dependent mass operator
        self._phi.eval_tp_fixed_loc(self._spans, self._bns, out=self._phi_at_pts)

        self._phi_5d[:, :, :, 0, 1] = self._phi_at_pts * self._jac_det
        self._phi_5d[:, :, :, 1, 0] = -self._phi_at_pts * self._jac_det
        self._tmp_5d[:] = self._jac_inv @ self._phi_5d @ self._jac_invT
        self._tmp_5dT[:] = xp.transpose(self._tmp_5d, axes=(3, 4, 0, 1, 2))

        self._M1hw_weights[0][1] = self._tmp_5dT[0, 1, :, :, :]
        self._M1hw_weights[1][0] = self._tmp_5dT[1, 0, :, :, :]

        self._M1hw.assemble(
            weights=self._M1hw_weights,
        )

        # solve with RK
        self._ode_solver(0.0, dt)
