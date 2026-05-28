import logging
from dataclasses import dataclass
from typing import Callable, Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVector
from line_profiler import profile

from struphy.feec.mass import L2Projector, WeightedMassOperator
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")

class PressureWave(Propagator):
    r"""
    Weak discretization of the curl-curl problem.

    Find :math:`\mathbf E \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \nabla \times \mathbf F \cdot \nabla \mathbf E\,\textrm d \mathbf x - \sigma \int_\Omega \mathbf F \cdot \mathbf E\,\textrm d \mathbf x = \sum_i \int_\Omega \mathbf F \cdot \mathbf J _i\,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,

    where :math:`\mathbf J _i:\Omega \to \mathbb R^3` is a real-valued vector field and
    :math:`\sigma \in \mathbb R`
    is a scalar.
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \mathbb C^\top \cdot \mathbb M^2 \cdot \mathbb C - \sigma \mathbb M^1 \right)\, \boldsymbol e^{n+1} =\sum_i \mathbb P^1 \cdot \boldsymbol J _i\,,

    where :math:`\mathbb M^1` and :math:`\mathbb M^2` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>` and :math:`\mathbb P^1`
    is the projector into the space :math:`V ^1 _h`.

    """

    class Variables:
        """Container for variables advanced by :class:`ImplicitDiffusion`.

        Attributes
        ----------
        rho : FEECVariable
            Scalar solution field in ``"H1"`` space.
        
        u : FEECVariable
            Vector-valued solution in ``"Hcurl"`` space.
        """

        def __init__(self):
            self._rho: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def rho(self) -> FEECVariable:
            return self._rho

        @rho.setter
        def rho(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._rho = new
        
        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._u = new

    def __init__(self):
        self.variables = self.Variables()
    
    @dataclass
    class Options:
        """Configuration options for :class:`PressureWave`.

        Parameters
        ----------
        omega: float, default=1.0
            Coefficient multiplying the mass matrices on the left-hand sides.
            Represents the oscillation frequency of the wave.
        
        mass: float, default=1.0
            Coefficient representing the mass of the particle species.

        Z: int, default=1
            Coefficient representing the charge of the particle species.
        
        E: FEECVariable in ``"Hcurl"``
            Source term on the right-hand side(s).
        
        rhobar: FEECVariable in ``"H1"``
            Field representing the density of the fluid species.
            Acts as weight in the mass matrix for ``u`` and ``E``.
        
        theta: FEECVariable in ``"H1"``
            Field representing the temperature of the fluid species.
            Acts as weight in the construction of the gradient matrix for ``rho``.

        pressure_gradient: {"chain_rule_splittng", "projection"}, default="chain_rule_splitting"
            Technique for discretizing the gradient of the product of ``rho`` and ``thetabar``.
        
        solve_type: {"rhosin_ucos", "rhocos_usin"}, default="rhosin_ucos"
            If ``"rhocos_usin"`` is chosen, ``u`` will be multiplied by -1
            after being calculated. Represents which pair of wave coefficients to solve for.
        
        rho0 : StencilVector, default=None
            Initial rho guess for the iterative linear solver.
            
        u0 : StencilVector, default=None
            Initial u guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Name of the preconditioner configuration.
            Currently this class sets ``pc=None`` internally, so this option is
            reserved for compatibility and future extensions.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.
        """

        # Literals
        PressureGradient = Literal["chain_rule_splitting", "projection"]
        SolveType = Literal["rhosin_ucos", "rhocos_usin"]
        # options
        omega: float = 1.0
        mass: float = 1.0
        Z: float = 1.0
        E: FEECVariable = None
        rhobar: FEECVariable = None
        theta: FEECVariable = None
        pressure_gradient: PressureGradient = "projection"
        solve_type: SolveType = "rhosin_ucos"
        rho0: StencilVector = None
        u0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.pressure_gradient, self.PressureGradient)
            check_option(self.solve_type, self.SolveType)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

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
    
    @profile
    def allocate(self, verbose: bool = False):
        # always stabilize
        if xp.abs(self.options.omega) < 1e-14:
            self.options.omega = 1e-14
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.info(f"Running Pressure Wave solve with {self.options.omega =}")
            
        self._omega = self.options.omega

        if xp.abs(self.options.mass) < 1e-6:
            self.options.mass = 1e-6
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.info(f"Running Pressure Wave solve with {self.options.mass =}")
            
        # model parameters
        
        self._mass = self.options.mass

        self._Z = self.options.Z

        # model variables

        rho = self.variables.rho.spline.vector

        u = self.variables.u.spline.vector

        # model sources

        if self.options.E is not None:
            assert self.options.E.space == "Hcurl"
            E = self.options.E.spline.vector
            E.update_ghost_regions()
        
        else:
            E = u.space.zeros()


        # density and density weight operator

        self._M1rho = None
        self._u_system_matrix = None

        if self.options.rhobar is None:
            self._M1rho = self.mass_ops.M1
            if self.options.solve_type == "rhosin_ucos":
                self._u_system_matrix = self._omega * self._M1rho
            
            if self.options.solve_type == "rhocos_usin":
                self._u_system_matrix = - self._omega * self._M1rho

        else:
            assert self.options.rhobar.space == "H1"
            rhobar = self.options.rhobar.spline

            self._M1rho = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(
                "Ginv",
                "sqrt_g",
                rhobar,
            ),
            name = "M1rho",
            assemble = True,
            )

            if self.options.solve_type == "rhosin_ucos":
                self._u_system_matrix = self._omega * self._M1rho
            
            if self.options.solve_type == "rhocos_usin":
                self._u_system_matrix = - self._omega * self._M1rho
        

        # temperature and temperature gradient operators

        if self.options.theta is None:
            self._Btheta = self.mass_ops.M1 @ self.derham.grad / self._mass

        else:
            assert self.options.theta.space == "H1"
            theta = self.options.theta.spline

            if self.options.pressure_gradient == "projection":

                P0theta = self.basis_ops.create_basis_op(
                    [[theta]],
                    "H1",
                    "H1",
                    assemble = True,
                    name = "P0theta",
                )

                self._Btheta = self.mass_ops.M1 @ self.derham.grad @ P0theta / self._mass
                
                # grad_theta_vector = self.derham.grad.dot(theta.vector)
                # grad_theta_vector.update_ghost_regions()
                # grad_theta_feec = FEECVariable(space="Hcurl")
                # grad_theta_feec.allocate(derham=self.derham, domain=self.domain)

                # grad_theta = grad_theta_feec.spline
                # grad_theta.vector = grad_theta_vector
                # grad_theta.vector.update_ghost_regions()

                # M1theta = self.mass_ops.create_weighted_mass(
                #     "Hcurl",
                #     "Hcurl",
                #     weights=(
                #         "Ginv",
                #         "sqrt_g",
                #         theta,
                #     ),
                #     name = "M1theta",
                #     assemble = True,
                # )

                # Atheta = self.mass_ops.create_weighted_mass(
                #     "H1",
                #     "Hcurl",
                #     weights=(
                #         "Ginv",
                #         "sqrt_g",
                #         grad_theta,
                #     ),
                #     name = "Atheta",
                #     assemble = True,
                # )

                # self._Btheta = (M1theta @ self.derham.grad + Atheta) / self._mass


        # rhobar and theta are taken as SplineFunction callables to build the operators
        

        # initial guess and solver params
        self._rho0 = self.options.rho0
        self._u0 = self.options.u0
        self._info = self.options.solver_params.info

        self._rho_system_matrix = (self._omega * self._omega) * self.mass_ops.M0 - self.derham.grad.T @ self._Btheta

        
        # preconditioner and solver for Ax=b
        if self.options.precond is None:
            pc = None
        else:
            # TODO: waiting for multigrid preconditioner
            pc = None


        # preparation of the solvers

        self._rho_solver = inverse(
            self._rho_system_matrix,
            # self.options.solver,
            "gmres",
            # pc=pc,
            x0=self.rho0,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )

        self._u_solver = inverse(
            self._u_system_matrix,
            #self.options.solver,
            "gmres",
            #pc=pc,
            x0=self.u0,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )
        

        # definition of the source term
        
        self._E_source = - self._Z * self._M1rho.dot(E) / self._mass
        self._E_source.update_ghost_regions()

        self._u_rhs = u.space.zeros()
        self._u_tmp = u.space.zeros()
        self._rho_tmp = rho.space.zeros()

    @property
    def rho0(self):
        """
        feectools.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First rho guess of the iterative solver.
        """
        return self.options.rho0

    @rho0.setter
    def rho0(self, value: StencilVector):
        """In-place setter for StencilVector/PolarVector. First rho guess of the iterative solver."""
        assert value.space == self.derham.V0
        assert value.space.symbolic_space == "H1", (
            f"Right-hand side must be in H1, but is in {value.space.symbolic_space}."
        )

        if self.options.rho0 is None:
            self.options.rho0 = value
        else:
            self.options.rho0[:] = value[:]
    
    @property
    def u0(self):
        """
        feectools.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First u guess of the iterative solver.
        """
        return self.options.u0

    @u0.setter
    def u0(self, value: StencilVector):
        """In-place setter for StencilVector/PolarVector. First u guess of the iterative solver."""
        assert value.space == self.derham.V1
        assert value.space.symbolic_space == "Hcurl", (
            f"Right-hand side must be in Hcurl, but is in {value.space.symbolic_space}."
        )

        if self.options.u0 is None:
            self.options.u0 = value
        else:
            self.options.u0[:] = value[:]
    
    @profile
    def __call__(self, dt):
        # compute rho
        self._rho_rhs = self.derham.grad.T.dot(self._E_source)
        self._rho_rhs.update_ghost_regions()

        rho_out = self._rho_solver.solve(self._rho_rhs, out=self._rho_tmp)
        rho_info = self._rho_solver._info

        if self._info:
            logger.info(rho_info)

        # compute u_rhs
        self._u_rhs = self._Btheta.dot(rho_out) + self._E_source
        self._u_rhs.update_ghost_regions()

        u_out = self._u_solver.solve(self._u_rhs, out=self._u_tmp)
        u_info = self._u_solver._info

        if self._info:
            logger.info(u_info)
        
        self.update_feec_variables(rho=rho_out, u=u_out)


