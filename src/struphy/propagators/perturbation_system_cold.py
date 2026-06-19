import logging
from dataclasses import dataclass
from typing import Callable, get_args
from warnings import warn

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.utilities import LocalRotationMatrix
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ColdPlasmaPerturbation(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in H(\textnormal{div})`, :math:`\mathbf u_e \in H(\textnormal{div})` and  :math:`\mathbf \phi \in L^2` such that

    .. math::

        \int_{\Omega} \partial_t  \mathbf{u}\cdot \mathbf{v} \, \textrm d\mathbf{x} &=  \int_{\Omega}  \phi \nabla \! \cdot \! \mathbf{v} \, \textrm d\mathbf{x}  + \int_{\Omega}  \mathbf{u}\! \times \! \mathbf{B}_0 \cdot \mathbf{v} \, \textrm d\mathbf{x} + \nu \int_{\Omega} \nabla \mathbf{u}\! : \! \nabla \mathbf{v} \, \textrm d\mathbf{x} + \int_{\Omega} f \mathbf{v} \, \textrm d\mathbf{x} \qquad \forall \, \mathbf{v} \in H(\textrm{div}) \,.
        \\[2mm]
        0 &= - \int_{\Omega} \phi \nabla \! \cdot \! \mathbf{v_e} \, \textrm d\mathbf{x} - \int_{\Omega} \mathbf{u_e} \! \times \! \mathbf{B}_0 \cdot \mathbf{v_e} \, \textrm d\mathbf{x}  + \nu_e \int_{\Omega} \nabla \mathbf{u_e}  \!: \! \nabla \mathbf{v_e} \, \textrm d\mathbf{x} + \int_{\Omega} f_e \mathbf{v_e} \, \textrm d\mathbf{x} \qquad \forall \ \mathbf{v_e} \in H(\textrm{div}) \,.
        \\[2mm]
        0 &= \int_{\Omega} \psi \nabla \cdot (\mathbf{u}-\mathbf{u_e}) \, \textrm d\mathbf{x} \qquad \forall \, \psi \in L^2 \,.

    :ref:`time_discret`: fully implicit.
    """

    # =========================================================================
    ### State variables (electron density rhosin and rhocos, electron velocity usin and ucos, electric field Esin and Ecos, magnetic field Bsin and Bcos)
    # =========================================================================

    class Variables:
        """Container for variables advanced by :class:`ColdPlasmaPerturbation`.

        Attributes
        ----------
        rhosin : FEECVariable or None
            Sine component of electron density variable in ``"H1"`` space.
        rhocos : FEECVariable or None
            Cosine component of electron density variable in ``"H1"`` space.
        usin : FEECVariable or None
            Sine component of the electron velocity variable in ``"Hcurl"`` space.
        ucos : FEECVariable or None
            Cosine component of the electron velocity variable in ``"Hcurl"`` space.
        Esin : FEECVariable or None
            Sine component of the electric field variable in ``"Hcurl"`` space.
        Ecos : FEECVariable or None
            Cosine component of the electric field variable in ``"Hcurl"`` space.
        Bsin : FEECVariable or None
            Sine component of the magnetic field variable in ``"Hdiv"`` space.
        Bcos : FEECVariable or None
            Cosine component of the magnetic field variable in ``"Hdiv"`` space.
        """

        def __init__(self) -> None:
            self._rhosin: FEECVariable | None = None
            self._rhocos: FEECVariable | None = None
            self._usin: FEECVariable | None = None
            self._ucos: FEECVariable | None = None
            self._Esin: FEECVariable | None = None
            self._Ecos: FEECVariable | None = None
            self._Bsin: FEECVariable | None = None
            self._Bcos: FEECVariable | None = None

        @property
        def rhosin(self) -> FEECVariable | None:
            return self._rhosin

        @rhosin.setter
        def rhosin(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._rhosin = new

        @property
        def rhocos(self) -> FEECVariable | None:
            return self._rhocos

        @rhocos.setter
        def rhocos(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._rhocos = new

        @property
        def usin(self) -> FEECVariable | None:
            return self._usin

        @usin.setter
        def usin(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._usin = new

        @property
        def ucos(self) -> FEECVariable | None:
            return self._ucos

        @ucos.setter
        def ucos(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._ucos = new

        @property
        def Esin(self) -> FEECVariable | None:
            return self._Esin

        @Esin.setter
        def Esin(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._Esin = new

        @property
        def Ecos(self) -> FEECVariable | None:
            return self._Ecos

        @Ecos.setter
        def Ecos(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._Ecos = new
        
        @property
        def Bsin(self) -> FEECVariable | None:
            return self._Bsin

        @Bsin.setter
        def Bsin(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._Bsin = new

        @property
        def Bcos(self) -> FEECVariable | None:
            return self._Bcos

        @ucos.setter
        def Bcos(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._Bcos = new


    def __init__(self):
        self.variables = self.Variables()

    # =========================================================================
    ### Options
    # =========================================================================

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`ColdPlasmaPerturbation`.

        Parameters
        ----------
        J : FEECVariable in ``"Hcurl"`` or list
            Cosine component of the source term.
        omega : float, default=1.0
            Source term oscillation frequency.
        curlcurl_lambda : float, default=1.0
            Coefficient in the curl-curl operator.
        mass : float, default=1.0
            Electron mass in relative unis.
        mu : Callable or float, default=1.0
            Electron viscosity coefficient.
        nu : Callable or float, default=1.0
            Electron-Ion collision frequency.
        rhobar : FEECVariable in ``"H1"`` or Callable or float, default=1.0
            Average electron mass density.
        theta : FEECVariable in ``"H1"`` or Callable or float, default=1.0
            Average electron temperature.
        Ebar : FEECVariable in ``"Hcurl"`` or list
            Average electrostatic field.
        Esin0 : StencilVector, default=None
            Initial Esin guess for the iterative linear solver.
        Ecos0 : StencilVector, default=None
            Initial Ecos guess for the iterative linear solver.
        solver : LiteralOptions.OptsGenSolver, default="gmres"
            Linear/saddle-point solver used for the global system.
        solver_params : SolverParameters or None, default=None
            Solver controls.
        """

        J: FEECVariable | list
        omega: float = 1.0
        curlcurl_lambda: float = 1.0
        mass: float = 1.0
        mu: Callable | float = 1.0
        nu: Callable | float = 1.0
        rhobar: FEECVariable | Callable | float = 1.0
        theta: FEECVariable | Callable | float = 1.0
        Ebar: FEECVariable | list
        
        Esin0: FEECVariable | StencilVector = None
        Ecos0: FEECVariable | StencilVector = None

        solver: LiteralOptions.OptsGenSolver = "gmres"
        solver_params: SolverParameters | None = None

        def __post_init__(self):
            # input format correctness
            assert self.J is not None
            if (not isinstance(self.J, (FEECVariable, list))):
                raise TypeError(f"J must be either a Hcurl FEECVariable or list of Callables, got {type(self.J)}")
            if isinstance(self.J, FEECVariable):
                assert self.J.space == "Hcurl"
            if isinstance(self.J,list):
                assert len(self.J) == 3
                for ji in self.J:
                    assert isinstance(ji, Callable)
            
            if (self.rhobar is not None) and (not isinstance(self.rhobar, (FEECVariable, Callable, float))):
                raise TypeError(f"rhobar must be either a H1 FEECVariable or a Callable or a float, got {type(self.rhobar)}")
            if isinstance(rhobar, FEECVariable):
                assert rhobar.space == "H1"
            
            if (self.theta is not None) and (not isinstance(self.theta, (FEECVariable, Callable, float))):
                raise TypeError(f"theta must be either a H1 FEECVariable or a Callable or a float, got {type(self.theta)}")
            if isinstance(self.theta, FEECVariable):
                assert self.theta.space == "H1"
            
            assert self.Ebar is not None
            if (not isinstance(self.Ebar,(FEECVariable, list))):
                raise TypeError(f"Ebar must be either a Hcurl FEECVariable or list of Callables, got {type(self.Ebar)}")
            if isinstance(self.Ebar, FEECVariable):
                assert self.Ebar.space == "Hcurl"
            if isinstance(self.Ebar,list):
                assert len(self.Ebar) == 3
                for ei in self.Ebar:
                    assert isinstance(ei, Callable)
            
            # --- physical parameter sanity checks ---
            if self.omega <= 0:
                raise ValueError(f"omega must be positive, got {self.omega}")
            if self.curlcurl_lambda <= 0:
                raise ValueError(f"curlcurl_lambda must be positive, got {self.curlcurl_lambda}")
            if self.mass <= 0:
                raise ValueError(f"mass must be positive, got {self.mass}")
            if isinstance(mu, float) and self.mu < 0:
                raise ValueError(f"mu must be non-negative, got {self.mu}")
            if isinstance(nu, float) and self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")

            # --- E initial guess correct space check ---
            if isinstance(self.Esin0, FEECVariable):
                assert self.Esin0.space == "Hcurl"
            if isinstance(self.Ecos0, FEECVariable):
                assert self.Ecos0.space == "Hcurl"

            check_option(self.solver, LiteralOptions.OptsGenSolver, LiteralOptions.OptsSaddlePointSolver)
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        assert hasattr(self, "_options"), "Options not set."
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    # =========================================================================
    ### Allocate
    # =========================================================================

    def allocate(self):

        # ---- source term vector (for RHS assembly) ---------------------------

        self._j: StencilVector

        if isinstance(self._options.J,FEECVariable):
            self._j = self._options.J.spline.vector
        else:
            self._j = self.derham.P1(self._options.J) # works if J is a list of Callables

        # ---- unconstrained operators (for RHS assembly) ----------------------

        self._M0 = self.mass_ops.M0
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2
        self._grad = self.derham.grad
        self._curl = self.derham.curl
        self._div = self.derham.div

        self._M0inv = inverse(
            self._M0,
            "pcg",
            pc="MassMatrixPreconditioner",
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose = False,
            recycle = self._options.solver_params.recycle,
        )

        self._M1mu: WeightedMassOperators
        self._M2mu: WeightedMassOperators
        self._M3mu: WeightedMassOperators

        if isinstance(self._options.mu, float):
            self._M1mu = self._options.mu * self._M1
            self._M2mu = self._options.mu * self._M2
            self._M3mu = self._options.mu * self.mass_ops.M3
        else:
            self._M1mu = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(
                "Ginv",
                "sqrt_g",
                self._options.mu,
            ),
            name = "M1mu",
            assemble = True,
            )

            self._M2mu = self.mass_ops.create_weighted_mass(
            "Hdiv",
            "Hdiv",
            weights=(
                "G",
                "1/sqrt_g",
                self._options.mu,
            ),
            name = "M2mu",
            assemble = True,
            )

            self._M3mu = self.mass_ops.create_weighted_mass(
            "L2",
            "L2",
            weights=(
                "1/sqrt_g",
                self._options.mu,
            ),
            name = "M3mu",
            assemble = True,
            )
        

        self._M1rho: WeightedMassOperators
        self._M1xrhoB: WeightedMassOperators

        rot_B = LocalRotationMatrix(
            self.eq_mhd.b2_1,
            self.eq_mhd.b2_2,
            self.eq_mhd.b2_3,
        )

        rhoB1: Callable
        rhoB2: Callable
        rhoB3: Callable
        
        if isinstance(self._options.rhobar, float):
            self._M1rho = self._options.rhobar * self.mass_ops.M1
            self._M1xrhoB = self.mass_ops.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    rot_B,
                    "Ginv",
                    "sqrt_g",
                    ),
                    name = "M1_xrhoB",
                    assemble = True,
            )
            self._M1xrhoB *= self._options.rhobar
            
        if isinstance(self._options.rhobar, Callable):
            self._M1rho = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(
                "Ginv",
                "sqrt_g",
                self._options.rhobar,
            ),
            name = "M1rho",
            assemble = True,
            )

            self._M1xrhoB = self.mass_ops.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    rot_B,
                    self._options.rhobar,
                    "Ginv",
                    "sqrt_g",
                    ),
                    name = "M1_xrhoB",
                    assemble = True,
            )
        
        if isinstance(self._options.rhobar,FEECVariable):
            self._M1rho = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(
                "Ginv",
                "sqrt_g",
                self._options.rhobar.spline,
            ),
            name = "M1rho",
            assemble = True,
            )
            
            self._M1xrhoB = self.mass_ops.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    rot_B,
                    self._options.rhobar.spline,
                    "Ginv",
                    "sqrt_g",
                    ),
                    name = "M1_xrhoB",
                    assemble = True,
            )
        
        self._M1rho_inv = inverse(
            self._M1rho,
            "pcg",
            pc="MassMatrixPreconditioner",
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose=False,
            recycle=self._options.solver_params.recycle,
        )
        

        self._M1nurho: WeightedMassOperators

        if isinstance(self._options.nu, float):
            self._M1nurho = self._options.nu * self._M1rho

        if isinstance(self._options.nu, Callable):
            nurho: Callable
            if isinstance(self._options.rhobar, float):
                nurho = lambda *etas: self._options.nu(*etas) * self._options.rhobar
            if isinstance(self._options.rhobar,Callable):
                nurho = lambda *etas: self._options.nu(*etas) * self._options.rhobar(*etas)
            if isinstance(self._options.rhobar,FEECVariable):
                nurho = lambda *etas: self._options.nu(*etas) * self._options.rhobar.spline(*etas)

            self._M1nurho = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            weights=(
                "Ginv",
                "sqrt_g",
                nurho,
            ),
            name = "M1nurho",
            assemble = True,
            )


        self._P00theta: BasisProjectionOperators

        if isinstance(self._options.theta, float):
            self._P00theta = self._options.theta * IdentityOperator(self.derham.V0)
        
        if isinstance(self._options.theta, Callable):
            self._P00theta = self.basis_ops.create_basis_op(
                    [[self._options.theta]],
                    "H1",
                    "H1",
                    assemble = True,
                    name = "P00theta",
                )
        
        if isinstance(self._options.theta, FEECVariable):
            self._P00theta = self.basis_ops.create_basis_op(
                    [[self._options.theta.spline]],
                    "H1",
                    "H1",
                    assemble = True,
                    name = "P00theta",
                )
        
        self._P01Ebar: BasisProjectionOperators

        if isinstance(self._options.Ebar, list):
            self._P01Ebar = self.basis_ops.create_basis_op(
                [[self._options.Ebar[0]],[self._options.Ebar[1]],[self._options.Ebar[2]]],
                "H1",
                "Hcurl",
                assemble = True,
                name = "P01Ebar",
            )
        
        if isinstance(self._options.Ebar, FEECVariable):
            Ebar1 = lambda *etas: self._options.Ebar.spline(etas)[0]
            Ebar2 = lambda *etas: self._options.Ebar.spline(etas)[1]
            Ebar3 = lambda *etas: self._options.Ebar.spline(etas)[2]

            self._P01Ebar = self.basis_ops.create_basis_op(
                [[Ebar1],[Ebar2],[Ebar3]],
                "H1",
                "Hcurl",
                assemble = True,
                name = "P01Ebar",
            )


        self._P12 = self.basis_ops.U1


        ones = lambda *etas: 1.0 + 0 * etas
        zeroes = lambda *etas: 0 * etas

        self._O1 = self.basis_ops.create_basis_op(
            [[ones, zeroes, zeroes]],
            "Hcurl",
            "H1",
            assemble = True,
            name = "O1",
        )
        self._O2 = self.basis_ops.create_basis_op(
            [[zeroes, ones, zeroes]],
            "Hcurl",
            "H1",
            assemble = True,
            name = "O1",
        )
        self._O3 = self.basis_ops.create_basis_op(
            [[zeroes, zeroes, ones]],
            "Hcurl",
            "H1",
            assemble = True,
            name = "O1",
        )


        self._Acurlcurl = self._curl.T @ self._M2 @ self._curl - self._options.curlcurl_lambda * self._M1

        self._divPi = - self._curl.T @ self._M2mu @ self._curl \
                    - 2/3 * self._P12.T @ self._div.T @ self._M3mu @ self._div @ self._P12 \
                        + 2 * self._O1.T @ self._grad.T @ self._M1mu @ self._grad @ self._O1 \
                        + 2 * self._O2.T @ self._grad.T @ self._M1mu @ self._grad @ self._O2 \
                        + 2 * self._O3.T @ self._grad.T @ self._M1mu @ self._grad @ self._O3
        
        # self._A = self._M1rho + self._M1 @ (self._grad @ self._P0theta + self._P01Ebar) @ self._M0inv @ self._grad.T @ self._M1rho \
        #                         / (self._options.mass * self._options.omega * self._options.omega)

        # self._B = (self._divPi - self._M1xrhoB / self._options.mass + self._M1nurho) / self._options.omega

        # ---- block saddle-point system ----------------------------------------

        self._block_V0 = BlockVectorSpace(self.derham.V0, self.derham.V0)
        self._block_V1 = BlockVectorSpace(self.derham.V1, self.derham.V1)
        self._block_V2 = BlockVectorSpace(self.derham.V2, self.derham.V2)

        self._block_source = BlockVector(self._block_V1, blocks=[self._M1.dot(self._j), None])

        self._block_M = BlockLinearOperator(
            self._block_V0, self._block_V0, blocks=[[self._M0, None], [None, self._M0]]
        )

        self._block_Divergence = BlockLinearOperator(
            self._block_V1, self._block_V0, blocks=[[None, - self._grad.T @ self._M1rho], [self._grad.T @ self._M1rho, None]]
        )

        self._block_Acurlcurl = BlockLinearOperator(
            self.block_V, self._block_V1, blocks=[[self._Acurlcurl, None], [None, self._Acurlcurl]]
        )

        self._block_B = BlockLinearOperator(
            self._block_V1, self._block_V1, blocks=[[None, self._M1rho / self._options.mass], [- self._M1rho / self._options.mass, None]]
        )

        self._block_P = BlockLinearOperator(
            self._block_V0, self._block_V1, 
            blocks=[[None, self._M1 @ (self._grad @ self._P0theta + self._P01Ebar) / self._options.mass], 
            [self._M1 @ (self._grad @ self._P0theta + self._P01Ebar) / self._options.mass, None]]
        )

        self._block_Q = BlockLinearOperator(
            self._block_V1, self._block_V1,
            blocks=[[self._options.omega * self._M1rho, self._divPi - self._M1xrhoB / self._options.mass + self._M1nurho], 
            [self._divPi - self._M1xrhoB / self._options.mass + self._M1nurho, - self._options.omega * self._M1rho]]
        )

        self._block_R = BlockLinearOperator(
            self._block_V1, self._block_V1, blocks=[[None, self._M1rho / self._options.mass], [self._M1rho / self._options.mass, None]]
        )

        self._block_Minv = inverse(
            self._block_M,
            "pcg",
            pc="MassMatrixPreconditioner",
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose = False,
            recycle = self._options.solver_params.recycle,
        )

        self._block_rhomatrix = - self._block_Minv @ self._block_Divergence / self._options.omega

        self._block_umatrix = - self._block_P @ self._block_rhomatrix - self._block_Q

        self._block_umatrix_inv = inverse(
            self._block_umatrix,
            "gmres",
            pc=None,
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose = False,
            recycle = self._options.solver_params.recycle,
        )

        self._block_Ematrix = self._block_Acurlcurl / self._options.omega + self._block_B @ self._block_umatrix_inv @ self._block_R

        self._block_Ematrix_inv = inverse(
            self._block_Ematrix,
            "gmres",
            pc=None,
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose = False,
            recycle = self._options.solver_params.recycle,
        )

        # self._coupled_equations_matrix = BlockLinearOperator(
        #     self._block_domain, self._block_codomain, blocks=[[self._A, -self._B], [self._B, self._A]]
        # )

        # M1rhoinv_j = self._M1rho_inv.solve(self._j)
        # A_j = self._A.dot(M1rhoinv_j)
        # minusB_j = - self.B.dot(M1rhoinv_j)

        # self._calEsin0: StencilVector = None
        # self._calEcos0: StencilVector = None

        # # --- copy current state ---
        # Esin0 = self.variables.Esin.spline.vector
        # Ecos0 = self.variables.Ecos.spline.vector

        # self._calEsin0 = self._M1rho_inv.dot(self._options.mass * self._Acurlcurl.dot(Esin0))
        # self._calEcos0 = self._M1rho_inv.dot(self._options.mass * self._Acurlcurl.dot(Ecos0))

        # self._calE0 = BlockVector(self._block_domain, blocks=[self._calEsin0, self._calEcos0])

        # self._coupled_equations_matrix_inverse = inverse(
        #     self._coupled_equations_matrix,
        #     solver="gmres",
        #     x0=self._calE0,
        #     tol=self._options.solver_params.tol,
        #     maxiter=self._options.solver_params.maxiter,
        #     verbose=False,
        #     recycle=True,
        # )

        # # --- build inverses of the curl-curl matrices with good initial guesses ---
        # self._Acurlcurl_inv_sin = inverse(
        #     self._Acurlcurl,
        #     solver="pcg",
        #     x0=Esin0,
        #     tol=self._options.solver_params.tol,
        #     maxiter=self._options.solver_params.maxiter,
        #     verbose=False,
        #     recycle=True,
        # )

        # self._Acurlcurl_inv_cos = inverse(
        #     self._Acurlcurl,
        #     solver="pcg",
        #     x0=Ecos0,
        #     tol=self._options.solver_params.tol,
        #     maxiter=self._options.solver_params.maxiter,
        #     verbose=False,
        #     recycle=True,
        # )


    # =========================================================================
    ### Equation solve
    # =========================================================================

    def __call__(self, dt):

        # --- calculate auxilliary vectors ---
        _calE = self._coupled_equations_matrix_inverse.solve(self._calE_RHS)

        _calEsin = _calE[0]
        _calEcos = _calE[1]

        _m_curlcurlEsin = self._M1rho.dot(_calEsin)
        _m_curlcurlEcos = self._M1rho.dot(_calEcos)

        _M1rho_usin = _m_curlcurlEcos.copy()
        _M1rho_ucos = self._options.mass * self._j - _m_curlcurlEsin

        # --- calculate solutions ---
        self._Esin.vector = self._Acurlcurl_inv_sin.solve(_m_curlcurlEsin / self._options.mass)
        self._Ecos.vector = self._Acurlcurl_inv_cos.solve(_m_curlcurlEcos / self._options.mass)

        self._Bsin.vector = - self._curl.dot(self._Ecos.vector) / self._options.omega
        self._Bcos.vector = self._curl.dot(self._Esin.vector) / self._options.omega

        self._usin.vector = self._M1rho_inv.solve(_M1rho_usin)
        self._ucos.vector = self._M1rho_inv.solve(_M1rho_ucos)

        self._rhosin.vector = self._M0inv.solve(self._grad.T.dot(_M1rho_ucos)) / self._options.omega
        self._rhocos.vector = - self._M0inv.solve(self._grad.T.dot(_M1rho_usin)) / self._options.omega

        # --- update FEEC variables ---
        self.update_feec_variables(
            rhosin=self._rhosin.vector, rhocos=self._rhocos.vector,
            usin=self._usin.vector, ucos=self._ucos.vector,
            Esin=self._Esin.vector, Ecos=self._Ecos.vector,
            Bsin=self._Bsin.vector, Bcos=self._Bcos.vector)

