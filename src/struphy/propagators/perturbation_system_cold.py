import logging
from dataclasses import dataclass
from typing import Callable, get_args
from warnings import warn

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator, ZeroOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.preconditioner import MassMatrixPreconditioner
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
    r""":ref:`FEEC <gempic>` discretization of a linearized cold plasma fluid system perturbed by a source of frequency :math:`\omega`.
    The state variables are the first-order components of the oscillations in electron density, electron velocity, electric field and magnetic field.
    Their oscillation-average counterparts (making up the plasma bulk) are passed as parameters of the solver.
    Each variable is split into a real (cosine) and imaginary (sine) part to represent the complex quantity fully.
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

        @Bcos.setter
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
        c0 : float, default=1.0
            First coefficient in the curl-curl operator.
        c1 : float, default=1.0
            Second coefficient in the curl-curl operator.
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
        solver : LiteralOptions.OptsGenSolver, default="gmres"
            Linear/saddle-point solver used for the global system.
        solver_params : SolverParameters or None, default=None
            Solver controls.
        """

        J: FEECVariable | list = None
        omega: float = 1.0
        c0: float = 1.0
        c1: float = 1.0
        mass: float = 1.0
        mu: Callable | float = 1.0
        nu: Callable | float = 1.0
        rhobar: FEECVariable | Callable | float = 1.0
        theta: FEECVariable | Callable | float = 1.0
        Ebar: FEECVariable | list = None

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
            if isinstance(self.rhobar, FEECVariable):
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
            if not isinstance(self.omega, float):
                raise TypeError(f"omega must be a float, recieved {type(self.omega)}")
            if self.omega <= 0:
                raise ValueError(f"omega must be positive, got {self.omega}")
            if not isinstance(self.c0, float):
                raise TypeError(f"c0 must be a float, recieved {type(self.c0)}")
            if self.c0 <= 0:
                raise ValueError(f"c0 must be positive, got {self.c0}")
            if not isinstance(self.c1, float):
                raise TypeError(f"c1 must be a float, recieved {type(self.c1)}")
            if self.c1 <= 0:
                raise ValueError(f"c1 must be positive, got {self.c1}")
            if not isinstance(self.mass, float):
                raise TypeError(f"mass must be a float, recieved {type(self.mass)}")
            if self.mass <= 0:
                raise ValueError(f"mass must be positive, got {self.mass}")
            if isinstance(self.mu, float) and self.mu < 0:
                raise ValueError(f"mu must be non-negative, got {self.mu}")
            if isinstance(self.nu, float) and self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")

            check_option(self.solver, LiteralOptions.OptsGenSolver, LiteralOptions.OptsSaddlePointSolver, LiteralOptions.OptsDirectSolver)
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

        self._j: StencilVector = None

        if isinstance(self._options.J,FEECVariable):
            self._j = self._options.J.spline.vector
        else:
            self._j = self.derham.P1(self._options.J) # works if J is a list of Callables
        
        zeroes = lambda x,y,z: 0. * (x+y+z)
        self._zerovectorV1 = self.derham.P1([zeroes,zeroes,zeroes]) # need this for RHS assembly
        self._zerovectorV2 = self.derham.P2([zeroes,zeroes,zeroes])
        self._zerofield = self.derham.P0(zeroes)

        # ---- unconstrained operators (for RHS assembly) ----------------------

        self._M0 = self.mass_ops.M0
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2
        self._grad = self.derham.grad
        self._curl = self.derham.curl
        self._div = self.derham.div

        _M0preconditioner = MassMatrixPreconditioner(self._M0)
        
        self._M0inv = inverse(
            self._M0,
            "pcg",
            pc=_M0preconditioner,
            tol=1e-12,
            maxiter=self._options.solver_params.maxiter,
            verbose = False,
            recycle = self._options.solver_params.recycle,
        )

        _M1preconditioner = MassMatrixPreconditioner(self.mass_ops.M1)
            
        self._M1inv = inverse(
            self._M1,
            "pcg",
            pc=_M1preconditioner,
            tol=1e-12,
            maxiter=self._options.solver_params.maxiter,
            verbose=False,
            recycle=self._options.solver_params.recycle,
        )

        self._M1mu: WeightedMassOperators = None
        self._M2mu: WeightedMassOperators = None
        self._M3mu: WeightedMassOperators = None

        if isinstance(self._options.mu, float):
            if self._options.mu == 0.:
                self._M1mu = ZeroOperator(self.derham.V1,self.derham.V1)
                self._M2mu = ZeroOperator(self.derham.V2,self.derham.V2)
                self._M3mu = ZeroOperator(self.derham.V3,self.derham.V3)
            else:
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
        

        self._M1rho: WeightedMassOperators = None
        self._M1xrhoB: WeightedMassOperators = None

        rot_B = LocalRotationMatrix(
            self.projected_equil.equil.b1_1,
            self.projected_equil.equil.b1_2,
            self.projected_equil.equil.b1_3,
        )
        
        if isinstance(self._options.rhobar, float):
            if self._options.rhobar == 0.:
                self._M1rho = ZeroOperator(self.derham.V1,self.derham.V1)
                self._M1xrhoB = ZeroOperator(self.derham.V1,self.derham.V1)
            else:
                rhoarray = lambda e1,e2,e3: self._options.rhobar + 0.*(e1+e2+e3)
                self._M1rho = self.mass_ops.create_weighted_mass(
                    "Hcurl",
                    "Hcurl",
                    weights=(
                        "Ginv",
                        "sqrt_g",
                        rhoarray,
                    ),
                    name = "M1rho",
                    assemble = True,
                )
                self._M1xrhoB = self.mass_ops.create_weighted_mass(
                    "Hcurl",
                    "Hcurl",
                    weights=(
                        rot_B,
                        rhoarray,
                    ),
                    name = "M1_xrhoB",
                    assemble = True,
                )
            
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
                    rot_B,
                    self._options.rhobar,
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
                    rot_B,
                    self._options.rhobar.spline,
                ),
                name = "M1_xrhoB",
                assemble = True,
            )
        
        # if not isinstance(self._options.rhobar,float):
        #     _M1rhopreconditioner = MassMatrixPreconditioner(self._M1rho)
            
        #     self._M1rho_inv = inverse(
        #         self._M1rho,
        #         "pcg",
        #         pc=_M1rhopreconditioner,
        #         tol=1e-12,
        #         maxiter=self._options.solver_params.maxiter,
        #         verbose=False,
        #         recycle=self._options.solver_params.recycle,
        #     )

        self._M1nurho: WeightedMassOperators = None

        assert isinstance(self._options.nu, (Callable, float))

        self._M1nurho = ZeroOperator(self.derham.V1,self.derham.V1)

        if self._options.rhobar != 0.:
            if isinstance(self._options.nu, float):
                if self._options.nu == 0.:
                    self._M1nurho = ZeroOperator(self.derham.V1,self.derham.V1)
                else:
                    self._M1nurho = self._options.nu * self._M1rho

            if isinstance(self._options.nu, Callable):
                nurho: Callable = None
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


        self._P00theta: BasisProjectionOperators = None

        if isinstance(self._options.theta, float):
            if self._options.theta == 0.:
                self._P00theta = ZeroOperator(self.derham.V0,self.derham.V0)
            else:
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
        
        self._P01Ebar: BasisProjectionOperators = None

        if isinstance(self._options.Ebar, list):
            self._P01Ebar = self.basis_ops.create_basis_op(
                [[self._options.Ebar[0]],[self._options.Ebar[1]],[self._options.Ebar[2]]],
                "H1",
                "Hcurl",
                assemble = True,
                name = "P01Ebar",
            )
        
        if isinstance(self._options.Ebar, FEECVariable):
            Ebar1 = lambda *etas: self._options.Ebar.spline(*etas)[0]
            Ebar2 = lambda *etas: self._options.Ebar.spline(*etas)[1]
            Ebar3 = lambda *etas: self._options.Ebar.spline(*etas)[2]

            self._P01Ebar = self.basis_ops.create_basis_op(
                [[Ebar1],[Ebar2],[Ebar3]],
                "H1",
                "Hcurl",
                assemble = True,
                name = "P01Ebar",
            )


        ones = lambda e1, e2, e3: 1.0 + 0.*(e1 + e2 + e3)
        zeroes = lambda e1, e2, e3: 0.*(e1 + e2 + e3)

        # self._P12 = self.basis_ops.create_basis_op(
        #     [[ones, zeroes, zeroes],
        #     [zeroes, ones, zeroes],
        #     [zeroes, zeroes, ones]],
        #     "Hcurl",
        #     "Hdiv",
        #     assemble = True,
        #     name = "P12",
        # )


        # self._O1 = self.basis_ops.create_basis_op(
        #     [[ones, zeroes, zeroes]],
        #     "Hcurl",
        #     "H1",
        #     assemble = True,
        #     name = "O1",
        # )
        # self._O2 = self.basis_ops.create_basis_op(
        #     [[zeroes, ones, zeroes]],
        #     "Hcurl",
        #     "H1",
        #     assemble = True,
        #     name = "O2",
        # )
        # self._O3 = self.basis_ops.create_basis_op(
        #     [[zeroes, zeroes, ones]],
        #     "Hcurl",
        #     "H1",
        #     assemble = True,
        #     name = "O3",
        # )


        self._Acurlcurl = self._options.c0 * self._curl.T @ self._M2 @ self._curl - self._options.c1 * self._M1

        # self._divPi = - self._curl.T @ self._M2mu @ self._curl \
        #             - 2/3 * self._P12.T @ self._div.T @ self._M3mu @ self._div @ self._P12 \
        #                 + 2 * self._O1.T @ self._grad.T @ self._M1mu @ self._grad @ self._O1 \
        #                 + 2 * self._O2.T @ self._grad.T @ self._M1mu @ self._grad @ self._O2 \
        #                 + 2 * self._O3.T @ self._grad.T @ self._M1mu @ self._grad @ self._O3
        self._divPi = - self._curl.T @ self._M2mu @ self._curl + 4/3 * self._M1mu @ self._grad @ self._M0inv @ self._grad.T @ self._M1

        # ---- block Schur solve system ----------------------------------------

        self._V0squared = BlockVectorSpace(self.derham.V0, self.derham.V0)
        self._V1squared = BlockVectorSpace(self.derham.V1, self.derham.V1)
        self._V2squared = BlockVectorSpace(self.derham.V2, self.derham.V2)

        self._space_block1 = BlockVectorSpace(self._V0squared, self._V1squared)
        self._space_block2 = BlockVectorSpace(self._V1squared, self._V2squared)

        self._source = BlockVector(self._V1squared, blocks=[self._zerovectorV1, self._options.omega * self._M1.dot(self._j)])
        self._zerovectorV2squared = BlockVector(self._V2squared, blocks=[self._zerovectorV2, self._zerovectorV2])
        self._block_source = BlockVector(self._space_block2, blocks=[self._source, self._zerovectorV2squared])

        self._block_M0 = BlockLinearOperator(
            self._V0squared, self._V0squared, blocks=[[None, - self._M0], [self._M0, None]]
        )

        self._block_M1 = BlockLinearOperator(
            self._V1squared, self._V1squared, blocks=[[None, - self._M1], [self._M1, None]]
        )

        self._block_Divergence = BlockLinearOperator(
            self._V1squared, self._V0squared, blocks=[[self._grad.T @ self._M1rho, None], [None, self._grad.T @ self._M1rho]]
        )

        self._block_curl = BlockLinearOperator(
            self._V1squared, self._V2squared, blocks=[[self._curl, None], [None, self._curl]]
        )

        self._block_curlV2 = BlockLinearOperator(
            self._V2squared, self._V1squared, blocks=[[self._curl.T @ self._M2, None], [None, self._curl.T @ self._M2]]
        )

        self._block_ImV2 = BlockLinearOperator(
            self._V2squared, self._V2squared, blocks=[[None, - IdentityOperator(self.derham.V2)], [IdentityOperator(self.derham.V2), None]]
        )

        self._block_Acurlcurl = BlockLinearOperator(
            self._V1squared, self._V1squared, blocks=[[self._Acurlcurl, None], [None, self._Acurlcurl]]
        )

        self._block_M1rho = BlockLinearOperator(
            self._V1squared, self._V1squared, blocks=[[self._M1rho, None], [None, self._M1rho]]
        )

        self._block_iM1rho = BlockLinearOperator(
            self._V1squared, self._V1squared, blocks=[[None, - self._M1rho], [self._M1rho, None]]
        )

        self._block_P = BlockLinearOperator(
            self._V0squared, self._V1squared, 
            blocks=[[self._M1 @ (self._grad @ self._P00theta + self._P01Ebar) / self._options.mass, None], 
            [None, self._M1 @ (self._grad @ self._P00theta + self._P01Ebar) / self._options.mass]]
        )

        self._block_Q = BlockLinearOperator(
            self._V1squared, self._V1squared,
            blocks=[[self._divPi - (self._M1xrhoB / self._options.mass) + self._M1nurho, self._options.omega * self._M1rho], 
            [- self._options.omega * self._M1rho, self._divPi - (self._M1xrhoB / self._options.mass) + self._M1nurho]]
        )

        # constru
        self._block_A = BlockLinearOperator(
            self._space_block1, self._space_block1,
            blocks=[[self._options.omega * self._block_M0, self._block_Divergence], [self._block_P, self._block_Q]]
        )

        self._block_B = BlockLinearOperator(
            self._space_block2, self._space_block1,
            blocks=[[None, None], [self._block_M1rho / self._options.mass, None]]
        )

        self._block_D = BlockLinearOperator(
            self._space_block1, self._space_block2,
            blocks=[[None, self._options.omega * self._block_iM1rho / self.options.mass], [None, None]]
        )

        self._block_C = BlockLinearOperator(
            self._space_block2, self._space_block2,
            blocks=[[self._block_Acurlcurl, None],
            [self._block_curl, - self._options.omega * self._block_ImV2]]
        )

        self._block_Cinv = inverse(
            self._block_C,
            "gmres",
            x0=None,
            tol=1e-10,
            maxiter=self._options.solver_params.maxiter,
            verbose = True,
            recycle = self._options.solver_params.recycle,
        )

        self._block_A_schur = self._block_A - self._block_B @ self._block_Cinv @ self._block_D

        self._block_A_schur_inv = inverse(
            self._block_A_schur,
            "gmres",
            x0=None,
            tol=self._options.solver_params.tol,
            maxiter=self._options.solver_params.maxiter,
            verbose = True,
            recycle = self._options.solver_params.recycle,
        )


    # =========================================================================
    ### Equation solve
    # =========================================================================

    def __call__(self, dt):

        tmp_z = self._block_Cinv.solve(self._block_source)

        rhs_x1 = - self._block_B.dot(tmp_z)

        tmp_x1 = self._block_A_schur_inv.solve(rhs_x1)

        tmp_x2 = tmp_z - self._block_Cinv.solve(self._block_D.dot(tmp_x1))

        comp_rho = tmp_x1[0]

        comp_u = tmp_x1[1]

        comp_E = tmp_x2[0]

        comp_B = tmp_x2[1]

        # --- update FEEC variables ---
        self.update_feec_variables(
            rhosin=comp_rho[1], rhocos=comp_rho[0],
            usin=comp_u[1], ucos=comp_u[0],
            Esin=comp_E[1], Ecos=comp_E[0],
            Bsin=comp_B[1], Bcos=comp_B[0])

