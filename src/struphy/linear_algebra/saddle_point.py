import numpy as np
import scipy as sc
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from psydac.linalg.basic import LinearOperator, Vector, ScaledLinearOperator, SumLinearOperator, ComposedLinearOperator
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.solvers import inverse
from psydac.linalg.direct_solvers import SparseSolver

from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.utilities import create_equal_random_arrays

from scipy.sparse.linalg import svds
from struphy.feec import linear_operators
from scipy.sparse import diags

from struphy.feec.linear_operators import LinOpWithTransp

from struphy.geometry.base import Domain
from struphy.feec.psydac_derham import Derham

from psydac.linalg.basic import IdentityOperator

class SaddlePointSolver:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        x^{n+1} = A^{-1} \left[ f - B^{\top} y_n \\right] \,,\\
        R = B x^{n+1} \,,\\
        y^{n+1} = y_n - \rho  R \,.


    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        rho: float,
        solver_name: str,
        tol=1e-6,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
        self._tol = tol
        self._max_iter = max_iter
        self._solver_params = solver_params

        # Allocate memory for matrices used in solving the system
        self._rhs = F.copy()
        self._R = B.codomain.zeros()
        self._p2 = B.codomain.zeros()
        self._p1 = B.codomain.zeros()
        self._a2 = B.codomain.zeros()
        self._alpha = 0
        self._beta = 0

        # Counter
        self._iterationssolverA = 0

        # initialize solver with matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._solverA = inverse(A, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # self._x0 = self._solverA._options["x0"]
        # print(f"{self._x0 =}")

        # self._solver_params["x0"] = self._x0

        # self._solverArecycle = inverse(A, solver_name, tol=tol,
        #                         maxiter=max_iter, **self._solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()

        # List to store residual norms
        self._residual_norms = []

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        assert self._F.space == self._A.domain

        # Initialize P to zero or given initial guess
        P = P_init if P_init is not None else self._P
        # Step 1: Compute velocity U by solving A U = -Bᵀ P + F
        self._rhs *= 0
        self._rhs -= self._B.transpose().dot(P)
        self._rhs += self._F
        U = self._solverA.dot(self._rhs, out=self._U)
        self._iterationssolverA += self._solverA._info["niter"]
        print(f"{self._iterationssolverA =}")

        # self._x0 = self._solverA._options["x0"]
        # print(f"{self._x0 =}")
        # self._solver_params["x0"] = self._x0

        # Step 2: Compute residual R = BU
        R = self._B.dot(self._U, out=self._R)
        residual = self._R.copy()
        self._p2 = residual

        for iteration in range(self._max_iter):
            residual_norm = np.linalg.norm(self._R.toarray())
            print(f"{self._iterationssolverA =}")
            print(f"residual norm = {residual_norm}")
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol * 10000:
                print(f"{self._iterationssolverA =}")
                print(f"{iteration =}")
                return self._U, self._P, self._solverA._info, self._residual_norms

            self._p1 = self._solverA.dot(self._B.transpose().dot(self._p2))
            # self._x0 = self._solverArecycle._options["x0"]
            # print(f"{self._x0 =}")
            # self._solver_params["x0"] = self._x0
            self._iterationssolverA += self._solverA._info["niter"]
            self._a2 = self._B.dot(self._p1)
            self._alpha = self._p2.dot(self._R) / (self._p2.dot(self._a2))

            # Step 3: Update velocity u <- u - alpha * p1  and potential p <- p + alpha * p2
            self._P += self._alpha * self._p2
            self._R -= self._alpha * self._a2
            self._U -= self._alpha * self._p1

            self._beta = self._R.dot(self._a2) / (self._p2.dot(self._a2))
            self._p2 = self._R - self._beta * self._p2

        # Return with info if maximum iterations reached
        return self._U, self._P, self._solverA._info, self._residual_norms


class SaddlePointSolverTest:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        A11: SumLinearOperator,
        A22: SumLinearOperator,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)
        assert isinstance(A11, SumLinearOperator) or isinstance(A11, LinearOperator) or isinstance(A11, ComposedLinearOperator)
        assert isinstance(A22, SumLinearOperator) or isinstance(A22, LinearOperator) or isinstance(A22, ComposedLinearOperator)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._A11 = A11
        self._A22 = A22
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()

        # Allocate memory for matrices used in solving the Schur system
        self._rhs1 = F[0].copy()
        self._rhs2 = F[1].copy()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._solverA11 = inverse(self._A[0,0], solver_name, tol=tol, maxiter=max_iter, **solver_params)
        self._solverA22 = inverse(self._A[1,1], solver_name, tol=tol, maxiter=max_iter, **solver_params)
        
        self._solverAinv = inverse(self._A11,'cg',tol=tol, maxiter=max_iter,**solver_params)
        
        self._solverAeinv = inverse(self._A22,'cg',tol=tol, maxiter=max_iter,**solver_params)
        
        _blocksinv = [[self._solverAinv, None], [None, self._solverAeinv]]
        self._Ainv = BlockLinearOperator(self._A.domain, self._A.codomain, blocks=_blocksinv)


        # if solver_params.get('pc'):
        #     solver_params.pop('pc')
        
        print(f'{self._Ainv.shape =}')
        print(f'{self._B.shape =}')
        print(f'{self._Ainv.domain =}')
        print(f'{self._B.domain =}')
        print(f'{self._A.shape =}')
        print(f'{self._A.domain =}')
        print(f'{self._F[0].shape =}')
        print(f'{self._F[1].shape =}')

        self._schurcomplement = self._B @ self._Ainv @ self._BT
        self._solverschurcomplement = inverse(
            self._schurcomplement, solver_name, tol=tol, maxiter=max_iter, **solver_params
        )

        print(f"Solver schur done")
        # Solution vectors
        self._P = self._B.codomain.zeros()
        self._U = self._A[0,0].codomain.zeros()
        self._Ue = self._A[1,1].codomain.zeros()

        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schurcomplement = 0  # Iterations for _solverschur

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        assert isinstance(self._F, BlockVector)
        assert self._F.space == self._A.domain

        # #use setter to update lhs matrix
        # self._solverA.linop = self._A
        # self._solverschur.linop = self._schur

        self._Matrixproduct = self._solverschurcomplement @ self._B @ self._Ainv #Here self._solverA???
        # Step 1: Compute potential P by solving B A^-1 Bᵀ  P = B A^-1 F
        self._P = self._Matrixproduct.dot(self._F, out=self._P)

        # Track iterations for _solverA (used during the Schur complement solve)

        print(f"Solved P with solverschur and solverA")

        # Step 2a: Compute velocity by solving A11 U  = F1 - B1ᵀ P
        self._rhs1 *= 0
        self._rhs1 -= self._BT[0,0].dot(self._P)
        self._rhs1 += self._F[0]
        
        self._U = self._solverA11.dot(self._rhs1, out=self._U)
        
        # Step 2b: Compute velocity by solving A22 Ue  = F2 - B2ᵀ P
        self._rhs2 *= 0
        self._rhs2 -= self._BT[1,0].dot(self._P)
        self._rhs2 += self._F[1]
        
        self._Ue = self._solverA22.dot(self._rhs2, out=self._Ue)

        

        # Return with info if maximum iterations reached
        return self._U, self._Ue, self._P, self._solverA22._info


class SaddlePointSolverNoCG:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        x^{n+1} = A^{-1} \left[ f - B^{\top} y_n \\right] \,,\\
        R = B x^{n+1} \,,\\
        y^{n+1} = y_n - \rho  R \,.


    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: LinearOperator,
        B: LinearOperator,
        F: Vector,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.codomain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter

        # Allocate memory for matrices used in solving the system
        self._rhs = F.copy()
        self._R = B.codomain.zeros()

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()

        # initialize solver with matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._solverA = inverse(A, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # List to store residual norms
        self._residual_norms = []

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        assert self._F.space == self._A.domain

        # Initialize P to zero or given initial guess
        self._P = P_init if P_init is not None else self._P
        for iteration in range(self._max_iter):
            # Step 1: Compute velocity U by solving A U = -Bᵀ P + F
            self._rhs *= 0
            self._rhs -= self._B.transpose().dot(self._P)
            self._rhs += self._F
            U = self._solverA.dot(self._rhs, out=self._U)

            # Step 2: Compute residual R = BU (divergence of U)
            R = self._B.dot(self._U, out=self._R)
            residual = self._R.copy()
            residual_norm = np.linalg.norm(self._R.toarray())
            print(f"{residual_norm =}")
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                return self._U, self._P, self._solverA._info, self._residual_norms

            self._P += self._rho * self._R

        # Return with info if maximum iterations reached
        return self._U, self._P, self._solverA._info, self._residual_norms


class SaddlePointSolverInexactUzawa():
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        A11: SumLinearOperator,
        A22: SumLinearOperator,
        masspc,
        rho: float,
        derham: Derham,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)
        assert isinstance(A11, SumLinearOperator) or isinstance(A11, LinearOperator) or isinstance(A11, ComposedLinearOperator)
        assert isinstance(A22, SumLinearOperator) or isinstance(A22, LinearOperator) or isinstance(A22, ComposedLinearOperator)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._A11 = A11
        self._A22 = A22
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()
        self.derham = derham
        self._masspc = masspc
        
        self._method_to_solve = 'DirectNPInverse' #  'GMRES', 'SparseSolver', 'ScipySparse', 'InexactNPInverse', 'DirectNPInverse'
        self._preconditioner = True
        spectralanalysis = False
        self._aslinop = False
        
        if self._method_to_solve == 'SparseSolver':
            spectralanalysis = False
        
        # Numpy arrays
        if self._method_to_solve == 'SparseSolver' or self._method_to_solve == 'ScipySparse':
            self._Anp = self._A[0,0].toarray_struphy(is_sparse=True)
            self._Aenp = self._A[1,1].toarray_struphy(is_sparse=True)
            self._B1np = self._B[0,0].toarray_struphy(is_sparse=True)
            self._B2np = self._B[0,1].toarray_struphy(is_sparse=True)
            self._F1np = self._F[0].toarray()
            self._F2np = self._F[1].toarray()
        elif self._method_to_solve == 'DirectNPInverse' or self._method_to_solve == 'InexactNPInverse':
            self._Anp = self._A[0,0].toarray_struphy()
            self._Aenp = self._A[1,1].toarray_struphy()
            self._B1np = self._B[0,0].toarray_struphy()
            self._B2np = self._B[0,1].toarray_struphy()
            self._F1np = self._F[0].toarray()
            self._F2np = self._F[1].toarray()
            if self._preconditioner == True:
                self._A11np = self._A11.toarray_struphy()
                self._A22np = self._A22.toarray_struphy()
        elif self._method_to_solve == 'GMRES':
            pass
        else:
            print(f'Method to solve not defined.')
        
        print(f'Arrays initialized')
        
        #  ### Toarrayofwholesystem
        # self._block_domainA = BlockVectorSpace(self._A11.domain, self._A22.domain)
        # self._block_codomainA = self._block_domainA
        # _blocksA = [[self._A11, None], [None, self._A22]]
        # _A = BlockLinearOperator(self._block_domainA, self._block_codomainA, blocks=_blocksA)
        # _anp = _A.toarray_struphy()
        

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")
            
         ### Solver inverse
        if self._method_to_solve == 'ScipySparse':
            self._Anpinv = sc.sparse.linalg.inv(self._Anp)
            self._Aenpinv = sc.sparse.linalg.inv(self._Aenp)
            self._Precnp = self._B1np@self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T
        elif self._method_to_solve == 'GMRES':   
            self._solverA = inverse(A, solver_name, tol=tol, maxiter=max_iter, **solver_params)
            self._solverAinv = inverse(self._A[0,0],'gmres',tol=tol, maxiter=max_iter,**solver_params)
            self._solverAeinv = inverse(self._A[1,1],'gmres',tol=tol, maxiter=max_iter,**solver_params)
            self._Prec = self._B[0,0]@self._solverAinv @ self._BT[0,0] + self._B[0,1] @ self._solverAeinv @ self._BT[1,0]
            if self._preconditioner == True:
                self._solverAinvpre = inverse(self._A11,'pcg',tol=tol, pc = self._masspc, maxiter=max_iter,**solver_params)
                self._solverAeinvpre = inverse(self._A22,'pcg',tol=tol, pc = self._masspc, maxiter=max_iter,**solver_params)
                #self._solverAeinvpre = inverse(self._A[1,1],'gmres',tol=tol, maxiter=max_iter,**solver_params)
                self._Prec = self._B[0,0]@self._solverAinvpre @ self._BT[0,0] + self._B[0,1] @ self._solverAeinvpre @ self._BT[1,0]
        elif self._method_to_solve == 'DirectNPInverse':
            if self._preconditioner == False:
                self._Anpinv = np.linalg.inv(self._Anp)
                self._Aenpinv = np.linalg.inv(self._Aenp)
            if self._preconditioner == True:
                self._A11npinv = np.linalg.inv(self._A11np)
                self._A22npinv = np.linalg.inv(self._A22np)
                #self._Precnp = self._B1np@self._A11npinv @ self._B1np.T + self._B2np @ self._A22npinv @ self._B2np.T
                self._Anpinv = np.linalg.inv(self._A11npinv@self._Anp)
                self._Aenpinv = np.linalg.inv(self._A22npinv@self._Aenp)
            self._Precnp = self._B1np@self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T
        elif self._method_to_solve == 'SparseSolver':
            self._directA11 = SparseSolver(self._Anp)
            self._directA22 = SparseSolver(self._Aenp)
            self._directA11inv_np = self._directA11.solve(np.identity(self._A11.shape[0]))
            self._directA22inv_np = self._directA22.solve(np.identity(self._A22.shape[0]))
            self._Precnp = self._B1np@self._directA11inv_np @ self._B1np.T + self._B2np @ self._directA22inv_np @ self._B2np.T
        elif self._method_to_solve == 'InexactNPInverse':
            self._A11npinv = np.linalg.inv(self._A11np)
            self._A22npinv = np.linalg.inv(self._A22np)
            self._Precnp = self._B1np@self._A11npinv @ self._B1np.T + self._B2np @ self._A22npinv @ self._B2np.T
                   
        print(f'Solvers initialized')
             
        #A and Ae preconditioned inverse
        
        # _A11np = self._A11.toarray_struphy()
        # self._A11inv = np.linalg.inv(_A11np)
        # _A22np = self._A22.toarray_struphy()
        # self._A22inv = np.linalg.inv(_A22np)
              
        print(f'Inverses initialized')
        
        
        ### Diagonalization
        
        # _Asparse = self._A[0,0].toarray_struphy(is_sparse=True)
        # _Aesparse = self._A[1,1].toarray_struphy(is_sparse=True)
        # _Adiag = _Asparse.diagonal()
        # _Aediag = _Aesparse.diagonal()
        # _Adiag = self._Anp.diagonal()
        # _Aediag = self._Aenp.diagonal()
        
        # epsilon = 1e-10
        # _diagA11 = 1.0 / (_Adiag + epsilon)
        # _diagA22 = 1.5 / (_Aediag + epsilon)
        # self._diagA11inv=diags(_diagA11)
        # self._diagA22inv=diags(_diagA22)
        
        # _diagA11 = np.diag(self._Anp, k=0)
        # _diagA22 = np.diag(self._Aenp, k=0)
        # _diagA11_1 = np.diag(self._Anp, k=1)
        # _diagA22_1 = np.diag(self._Aenp, k=1)
        # _diagA11_2 = np.diag(self._Anp, k=2)
        # _diagA22_2 = np.diag(self._Aenp, k=2)
        # _off_diag_1_zeroed = _diagA11_1.copy()
        # _off_diag_2_zeroed = _diagA11_2.copy()
        # _off_diag_A221_zeroed = _diagA22_1.copy()
        # _off_diag_A222_zeroed = _diagA22_2.copy()
        # for i in range(len(_diagA11_1)):
        #     if (i+1) % 3 == 0:
        #         _off_diag_1_zeroed[i] = 0.
        # for i in range(len(_diagA11_2)):
        #     if i % 3 == 0:
        #         pass
        #     else:
        #         _off_diag_2_zeroed[i] = 0.
        # for i in range(len(_diagA22_1)):
        #     if (i+1) % 3 == 0:
        #         _off_diag_A221_zeroed[i] = 0.
        # for i in range(len(_diagA22_2)):
        #     if i % 3 == 0:
        #         pass
        #     else:
        #         _off_diag_A222_zeroed[i] = 0.
                
        # _diagA11_1 = _off_diag_1_zeroed
        # _diagA22_1 = _off_diag_A221_zeroed
        # _diagA11_2 = _off_diag_2_zeroed
        # _diagA22_2 = _off_diag_A222_zeroed
            
        
        # self._diagonalA11 = diags(diagonals=[_diagA11_2, _diagA11_1, _diagA11, _diagA11_1, _diagA11_2],
        #                           offsets=[2, 1, 0, -1, -2],
        #                         format='csr')
        # self._diagonalA22 = diags(diagonals=[_diagA22_2, _diagA22_1, _diagA22, _diagA22_1, _diagA22_2],
        #                           offsets=[2, 1, 0, -1, -2],
        #                         format='csr')
        
        # self._Anpinv = np.linalg.inv(self._diagA11inv@self._Anp)
        # self._Aenpinv = np.linalg.inv(self._diagA22inv@self._Aenp)
        # self._Precnp = self._B1np@self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T
               
        
        # self._A11soldiag = SparseSolver(self._diagonalA11)
        # self._A22soldiag = SparseSolver(self._diagonalA22)
        # self._diagA11inv = self._A11soldiag.solve(np.identity(self._A11.shape[0]))
        # self._diagA22inv = self._A22soldiag.solve(np.identity(self._A22.shape[0]))
        
        
        # self._directA11inv = ArrayAsLinearOperator(self._directA11inv_np, self.derham, self._A11.domain, self._A11.domain)
        # self._directA22inv = ArrayAsLinearOperator(self._directA22inv_np, self.derham, self._A22.domain, self._A22.domain)
        
       
        ### Spectral analysis
        if spectralanalysis == True:
            # A11 before
            # eigvalsA11, vecsA11 = np.linalg.eig(self._Anp)
            # eigvalA11inv = np.diag(1.0 / (eigvalsA11))
            # self._PA11diag = vecsA11@eigvalA11inv@np.linalg.inv(vecsA11)
            if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                eigvalsA11_before, eigvecs_before = np.linalg.eig(self._Anp)    #self._PA11diag)#@
            elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                eigvalsA11_before, eigvecs_before = sc.sparse.linalg.eigs(self._Anp)
            maxbeforeA11 = max(eigvalsA11_before)
            maxbeforeA11_abs = np.max(np.abs(eigvalsA11_before))
            minbeforeA11_abs = np.min(np.abs(eigvalsA11_before))
            minbeforeA11 = min(eigvalsA11_before)
            specA11_bef = maxbeforeA11/minbeforeA11
            specA11_bef_abs = maxbeforeA11_abs/minbeforeA11_abs
            print(f'{maxbeforeA11 = }')
            print(f'{maxbeforeA11_abs = }')
            print(f'{minbeforeA11_abs = }')
            print(f'{minbeforeA11 = }')
            print(f'{specA11_bef = }')
            print(f'{specA11_bef_abs = }')
            
            # A22 before
            # eigvalsA22, vecsA22 = np.linalg.eig(self._Aenp)
            # eigvalA22inv = np.diag(1.0 / (eigvalsA22))
            # self._PA22diag = vecsA22@eigvalA22inv@np.linalg.pinv(vecsA22)
            if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                eigvalsA22_before, eigvecs_before = np.linalg.eig(self._Aenp)   #self._PA22diag)#@
            elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                eigvalsA22_before, eigvecs_before = sc.sparse.linalg.eigs(self._Aenp)
            maxbeforeA22 = max(eigvalsA22_before)
            maxbeforeA22_abs = np.max(np.abs(eigvalsA22_before))
            minbeforeA22_abs = np.min(np.abs(eigvalsA22_before))
            minbeforeA22 = min(eigvalsA22_before)
            specA22_bef = maxbeforeA22/minbeforeA22
            specA22_bef_abs = maxbeforeA22_abs/minbeforeA22_abs
            print(f'{maxbeforeA22 = }')
            print(f'{maxbeforeA22_abs = }')
            print(f'{minbeforeA22_abs = }')
            print(f'{minbeforeA22 = }')
            print(f'{specA22_bef = }')
            print(f'{specA22_bef_abs = }')
            
            #Update with preconditioner
            # self._Anpinv = np.linalg.inv(self._PA11diag@self._Anp)
            # self._Aenpinv = np.linalg.inv(self._PA22diag@self._Aenp)
            #self._Precnp = self._B1np@ self._PA22diag@ self._B1np.T + self._B2np @ self._PA22diag@ self._B2np.T
            #self._Precnp = self._B1np@self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T
            #self._Precnp = self._B1np@self._A11diaginv @ self._B1np.T + self._B2np @ self._A22diaginv @ self._B2np.T
            
                
            # # A11 after diagonalization
            # eigvalsA11_after, eigvecs_after = np.linalg.eig(self._A11diaginv@self._A[0,0].toarray_struphy())     ### Implement this
            # maxafterA11 = max(eigvalsA11_after)
            # minafterA11 = min(eigvalsA11_after)
            # maxafterA11_abs = np.max(np.abs(eigvalsA11_after))
            # minafterA11_abs = np.min(np.abs(eigvalsA11_after))
            # specA11_aft = maxafterA11/minafterA11
            # specA11_aft_abs = maxafterA11_abs/minafterA11_abs
            # print(f'{maxafterA11 = }')
            # print(f'{maxafterA11_abs = }')
            # print(f'{minafterA11_abs = }')
            # print(f'{minafterA11 = }')
            # print(f'{specA11_aft = }')
            # print(f'{specA11_aft_abs = }')
            
            
            # # A22 after diagonalization
            # eigvalsA22_after_diag, eigvecs_after = np.linalg.eig(self._A22diaginv@self._A[1,1].toarray_struphy())     ### Implement this
            # maxafterA22_diag = max(eigvalsA22_after_diag)
            # minafterA22_diag = min(eigvalsA22_after_diag)
            # maxafterA22_abs_diag = np.max(np.abs(eigvalsA22_after_diag))
            # minafterA22_abs_diag = np.min(np.abs(eigvalsA22_after_diag))
            # specA22_aft_diag = maxafterA22_diag/minafterA22_diag
            # specA22_aft_abs_diag = maxafterA22_abs_diag/minafterA22_abs_diag
            # print(f'{maxafterA22_diag = }')
            # print(f'{maxafterA22_abs_diag = }')
            # print(f'{minafterA22_abs_diag = }')
            # print(f'{minafterA22_diag = }')
            # print(f'{specA22_aft_diag = }')
            # print(f'{specA22_aft_abs_diag = }')
            
            if self._preconditioner == True:
                # A11 after preconditioning with its inverse
                if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                    eigvalsA11_after_prec, eigvecs_after = np.linalg.eig(self._A11npinv@self._A[0,0].toarray_struphy())     ### Implement this
                elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                    eigvalsA11_after_prec, eigvecs_after = sc.sparse.linalg.eigs(self._A11npinv@self._A[0,0].toarray_struphy())
                maxafterA11_prec = max(eigvalsA11_after_prec)
                minafterA11_prec = min(eigvalsA11_after_prec)
                maxafterA11_abs_prec = np.max(np.abs(eigvalsA11_after_prec))
                minafterA11_abs_prec = np.min(np.abs(eigvalsA11_after_prec))
                specA11_aft_prec = maxafterA11_prec/minafterA11_prec
                specA11_aft_abs_prec = maxafterA11_abs_prec/minafterA11_abs_prec
                print(f'{maxafterA11_prec = }')
                print(f'{maxafterA11_abs_prec = }')
                print(f'{minafterA11_abs_prec = }')
                print(f'{minafterA11_prec = }')
                print(f'{specA11_aft_prec = }')
                print(f'{specA11_aft_abs_prec = }')
                
                # A22 after preconditioning with its inverse
                if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                    eigvalsA22_after_prec, eigvecs_after = np.linalg.eig(self._A22npinv@self._A[1,1].toarray_struphy())     ### Implement this
                elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                    eigvalsA22_after_prec, eigvecs_after = sc.sparse.linalg.eigs(self._A22npinv@self._A[1,1].toarray_struphy()) 
                maxafterA22_prec = max(eigvalsA22_after_prec)
                minafterA22_prec = min(eigvalsA22_after_prec)
                maxafterA22_abs_prec = np.max(np.abs(eigvalsA22_after_prec))
                minafterA22_abs_prec = np.min(np.abs(eigvalsA22_after_prec))
                specA22_aft_prec = maxafterA22_prec/minafterA22_prec
                specA22_aft_abs_prec = maxafterA22_abs_prec/minafterA22_abs_prec
                print(f'{maxafterA22_prec = }')
                print(f'{maxafterA22_abs_prec = }')
                print(f'{minafterA22_abs_prec = }')
                print(f'{minafterA22_prec = }')
                print(f'{specA22_aft_prec = }')
                print(f'{specA22_aft_abs_prec = }')
        
        if self._aslinop==True:
            self._Ainv_op = ArrayAsLinearOperator(self._Anpinv, self.derham, self._A11.domain, self._A11.codomain)
            self._Aeinv_op = ArrayAsLinearOperator(self._Aenpinv, self.derham, self._A22.domain, self._A22.codomain)
            if self._preconditioner == True:
                self._A11inv_op = ArrayAsLinearOperator(self._A11npinv, self.derham, self._A11.domain, self._A11.codomain)
                self._A22inv_op = ArrayAsLinearOperator(self._A22npinv, self.derham, self._A22.domain, self._A22.codomain)
              
                
            self._A11inv_op.domain = self._A11._domain
            self._A11inv_op.codomain = self._A11._codomain
            self._A22inv_op.domain = self._A22._domain
            self._A22inv_op.codomain = self._A22._codomain
        
            self._Prec = self._B[0,0]@self._A11inv_op @ self._BT[0,0] + self._B[0,1] @ self._A22inv_op @ self._BT[1,0]
                
        print(f'Inverses initialized as linear operators')
        
        
        # Solution vectors
        if self._method_to_solve == 'GMRES' or self._aslinop == True:
            self._P = B.codomain.zeros()
            self._U = A[0,0].codomain.zeros()
            self._Ue = A[1,1].codomain.zeros()
            # Allocate memory for matrices used in solving the system
            self._rhs0 = F[0].copy()
            self._rhs1 = F[1].copy()
            self._R = B.codomain.zeros()
         
        else:
            # Solution vectors numpy
            self._Pnp = np.zeros(self._B1np.shape[0])
            self._Unp = np.zeros(self._Anp.shape[1])
            self._Uenp = np.zeros(self._Aenp.shape[1])
            # Allocate memory for matrices used in solving the system
            self._rhs0np = self._F1np.copy()
            self._rhs1np = self._F2np.copy()
            self._Rnp = np.zeros(self._B.shape[1])

        # List to store residual norms
        self._residual_norms = []
        self._stepsize = 0.
        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_solverAinv = 0  
        self._iterations_solverAeinv = 0 
        self._iterations_solverAinvpre = 0 
        self._iterations_solverAeinvpre = 0 
        


    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F
    
    @property
    def A11(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A11
    
    @property
    def A22(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A22

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f
        
    @A.setter
    def A11(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A11 = a
        
    @A.setter
    def A22(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A22 = a

    def __call__(self, U_init=None, Ue_init=None, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        # Initialize P to zero or given initial guess
        if self._method_to_solve == 'GMRES' or self._aslinop == True:
            self._P = P_init if P_init is not None else self._P
            self._U = U_init if U_init is not None else self._U
            self._Ue = Ue_init if U_init is not None else self._Ue
         
        else:
            self._Pnp = P_init.toarray() if P_init is not None else self._Pnp
            self._Unp = U_init.toarray() if U_init is not None else self._Unp
            self._Uenp = Ue_init.toarray() if U_init is not None else self._Uenp
            

        for iteration in range(self._max_iter):
            
            # ### Psydac
            if self._aslinop == True: 
                # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
                self._rhs0 *= 0
                self._rhs0 -= self._B[0,0].transpose().dot(self._P)
                self._rhs0 -= self.A[0,0].dot(self._U)
                self._rhs0 += self._F[0]
                if self._preconditioner == True:
                    self._U += self._Ainv_op.dot(self._A11inv_op@self._rhs0)
                else: 
                    self._U += self._Ainv_op.dot(self._rhs0)
                R1 = self._B[0,0].dot(self._U)
                #print(f'{np.linalg.norm(R1.toarray()) = }')
                
                self._rhs1 *= 0
                self._rhs1 -= self._B[0,1].transpose().dot(self._P)
                self._rhs1 -= self.A[1,1].dot(self._Ue)
                self._rhs1 += self._F[1]
                
                if self._preconditioner == True:
                    self._Ue = self._Aeinv_op.dot(self._A22inv_op @ self._rhs1) + self._Ue 
                else:
                    self._Ue = self._Aeinv_op.dot(self._rhs1) + self._Ue
                
                R2 = self._B[0,1].dot(self._Ue)
                #print(f'{np.linalg.norm(R2.toarray()) = }')
                
                # Step 2: Compute residual R = BU (divergence of U)
                R = self._B[0,0].dot(self._U) + self._B[0,1].dot(self._Ue)
                # R = self._B[0,1].dot(self._Ue)
                # Rx = self._F - self._A.dot(self._U) - self._B.transpose().dot(self._P)
                # residual = self._R.copy()
                residual_norm = np.linalg.norm(R.toarray())
                #print(f"{residual_norm =}")
                self._residual_norms.append(residual_norm)  # Store residual norm
                # Check for convergence based on residual norm
                if residual_norm < self._tol:
                    print(f'{iteration =}')
                    return self._U, self._Ue, self._P, iteration, self._residual_norms
                
                alpha = (R.dot(R))/(R.dot(self._Prec.dot(R)))
                
                self._stepsize = 0.5*self._stepsize + 0.5* alpha

                #self._P += alpha * R
                self._P += alpha * R
            
            
            elif self._method_to_solve == 'GMRES': 
                if self._preconditioner == True:
                    self._solverAinv.linop = self._solverAinvpre @ self._A[0,0]
                    self._solverAeinv.linop = self._solverAeinvpre @ self._A[0,0]
                # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
                self._rhs0 *= 0
                self._rhs0 -= self._B[0,0].transpose().dot(self._P)
                self._rhs0 -= self.A[0,0].dot(self._U)
                self._rhs0 += self._F[0]
                if self._preconditioner == True:
                    self._U += self._solverAinv.dot(self._solverAinvpre@self._rhs0)
                    self._iterations_solverAinvpre += self._solverAinvpre._info["niter"]
                    self._iterations_solverAinv += self._solverAinv._info["niter"]
                    #print(f'{self._iterations_solverAinv = }')
                    #print(f'{self._iterations_solverAinvpre = }')
                else: 
                    self._U += self._solverAinv.dot(self._rhs0)
                    self._iterations_solverAinv += self._solverAinv._info["niter"]
                    print(f'{self._iterations_solverAinv = }')
                R1 = self._B[0,0].dot(self._U)
                #print(f'{np.linalg.norm(R1.toarray()) = }')
                
                self._rhs1 *= 0
                self._rhs1 -= self._B[0,1].transpose().dot(self._P)
                self._rhs1 -= self.A[1,1].dot(self._Ue)
                self._rhs1 += self._F[1]
                
                if self._preconditioner == True:
                    self._Ue = self._solverAeinv.dot(self._solverAeinvpre @ self._rhs1) + self._Ue 
                    self._iterations_solverAeinvpre += self._solverAeinvpre._info["niter"]
                    self._iterations_solverAeinv += self._solverAeinv._info["niter"]
                    #print(f'{self._iterations_solverAeinv = }')
                    #print(f'{self._iterations_solverAeinvpre = }')
                else:
                    self._Ue = self._solverAeinv.dot(self._rhs1) + self._Ue
                    self._iterations_solverAeinv += self._solverAeinv._info["niter"]
                    print(f'{self._iterations_solverAeinv = }')
                
                R2 = self._B[0,1].dot(self._Ue)
                #print(f'{np.linalg.norm(R2.toarray()) = }')
                
                # Step 2: Compute residual R = BU (divergence of U)
                R = self._B[0,0].dot(self._U) + self._B[0,1].dot(self._Ue)
                # R = self._B[0,1].dot(self._Ue)
                # Rx = self._F - self._A.dot(self._U) - self._B.transpose().dot(self._P)
                # residual = self._R.copy()
                residual_norm = np.linalg.norm(R.toarray())
                #print(f"{residual_norm =}")
                self._residual_norms.append(residual_norm)  # Store residual norm
                # Check for convergence based on residual norm
                if residual_norm < self._tol:
                    print(f'{self._iterations_solverAinvpre =}')
                    print(f'{self._iterations_solverAeinvpre =}')
                    print(f'{iteration =}')
                    return self._U, self._Ue, self._P, self._solverAinv._info, self._residual_norms
                
                alpha = (R.dot(R))/(R.dot(self._Prec.dot(R)))
                
                self._stepsize = 0.5*self._stepsize + 0.5* alpha

                #self._P += alpha * R
                self._P += alpha * R
            
            
            else:    ### Numpy
                # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
                self._rhs0np *= 0
                self._rhs0np -= self._B1np.transpose().dot(self._Pnp)
                self._rhs0np -= self._Anp.dot(self._Unp)
                self._rhs0np += self._F1np
                #self._Unp += self._PA11diag.dot(self._rhs0np)    #diag inverse 
                #self._Unp += self._Anpinv.dot(self._PA11diag@self._rhs0np)    #np.linalg.inv() with preconditioner generated via diagonalization
                if self._method_to_solve == 'DirectNPInverse':
                    if self._preconditioner == False:
                        self._Unp += self._Anpinv.dot(self._rhs0np)    #np.linalg.inv() without preconditioner
                    elif self._preconditioner == True:
                        self._Unp += self._Anpinv.dot(self._A11npinv@self._rhs0np)    #np.linalg.inv() with preconditioner from input
                elif self._method_to_solve == 'SparseSolver':
                    self._Unp += self._directA11inv_np.dot(self._rhs0np)    #sparse solver
                elif self._method_to_solve == 'InexactNPInverse':
                    self._Unp += self._A11npinv.dot(self._rhs0np)    #A11 inv
                #self._Unp += self._Anpinv.dot(self._diagA11inv@self._rhs0np)    #3x3 diag inv as preconditioner
                
                
                
                #self._iterations_solverAinvpre += self._solverAinv._info["niter"]
                #print(f"{self._iterations_solverAinvpre =}")
                
                R1 = self._B1np.dot(self._Unp)
                #print(f'{np.linalg.norm(R1) = }')
                
                self._rhs1np *= 0
                self._rhs1np -= self._B2np.transpose().dot(self._Pnp)
                self._rhs1np -= self._Aenp.dot(self._Uenp)
                self._rhs1np += self._F2np
                #self._Uenp += (self._PA22diag.real).dot(self._rhs1np)  #diag inv 
                #self._Uenp += self._Aenpinv.dot(self._PA22diag@self._rhs1np)  #np.linalg.inv() with preconditioner generated via diagonalization
                if self._method_to_solve == 'DirectNPInverse':
                    if self._preconditioner == False:
                        self._Uenp += self._Aenpinv.dot(self._rhs1np)  #np.linalg.inv() without preconditioner
                    elif self._preconditioner == True:
                        self._Uenp += self._Aenpinv.dot(self._A22npinv@self._rhs1np)    #np.linalg.inv() with preconditioner from input
                elif self._method_to_solve == 'SparseSolver':
                    self._Uenp += self._directA22inv_np.dot(self._rhs1np)   # sparse solver
                elif self._method_to_solve == 'InexactNPInverse':
                    self._Uenp += self._A22npinv.dot(self._rhs1np)    #A11 inv
                #self._Uenp += self._Aenpinv.dot(self._diagA11inv@self._rhs1np)    #3x3 diag inv 
                
                
                
                #self._iterations_solverAeinv += self._solverAeinv._info["niter"]
                #print(f"{self._iterations_solverAeinv =}")
                
                R2 = self._B2np.dot(self._Uenp)
                #print(f'{np.linalg.norm(R2) = }')
                # Step 2: Compute residual R = BU (divergence of U)
                R = self._B1np.dot(self._Unp) + self._B2np.dot(self._Uenp)
                # R = self._B[0,1].dot(self._Ue)
                # Rx = self._F - self._A.dot(self._U) - self._B.transpose().dot(self._P)
                # residual = self._R.copy()
                residual_norm = np.linalg.norm(R)
                #print(f"{residual_norm =}")
                self._residual_norms.append(residual_norm)  # Store residual norm
                # Check for convergence based on residual norm
                if residual_norm < self._tol:
                    # print(f'{self._iterations_solverAinvpre =}')
                    # print(f'{self._iterations_solverAeinv =}')
                    print(f'{iteration =}')
                    self._Ulinop = ArrayAsLinearOperator(self._Unp, self.derham, self._A11.domain, self._A11.codomain)
                    self._Uelinop = ArrayAsLinearOperator(self._Uenp, self.derham, self._A11.domain, self._A11.codomain)
                    self._Plinop = ArrayAsLinearOperator(self._Pnp, self.derham, self._A11.domain, self._A11.codomain)
                    #return self._Ulinop, self._Uelinop, self._Plinop, self._solverAinv._info, self._residual_norms
                    if self._method_to_solve == 'GMRES':
                        print(f'{self._iterations_solverAinvpre =}')
                        print(f'{self._iterations_solverAeinvpre =}')
                        return self._Unp, self._Uenp, self._Pnp, self._solverAinv._info, self._residual_norms
                    else:
                        return self._Unp, self._Uenp, self._Pnp, iteration, self._residual_norms
                
                alpha = (R.dot(R))/(R.dot(self._Precnp.dot(R)))
                #alpha = (R.dot(R))/(R.dot(self._Precsparsenp.dot(R)))
                
                self._stepsize = 0.5*self._stepsize + 0.5* alpha

                #self._P += alpha * R
                self._Pnp += alpha.real * R.real

        # Return with info if maximum iterations reached
        print(f'{iteration =}')
        #return self._Ulinop, self._Uelinop, self._Plinop, self._solverAinv._info, self._residual_norms
        if self._method_to_solve == 'GMRES':
            return self._U, self._U, self._P, self._solverAinv._info, self._residual_norms
        else:
            self._Ulinop = ArrayAsLinearOperator(self._Unp, self.derham, self._A11.domain, self._A11.codomain)
            self._Uelinop = ArrayAsLinearOperator(self._Uenp, self.derham, self._A11.domain, self._A11.codomain)
            self._Plinop = ArrayAsLinearOperator(self._Pnp, self.derham, self._A11.domain, self._A11.codomain)
            return self._Unp, self._Uenp, self._Pnp, iteration, self._residual_norms
            

class SaddlePointSolverArrowHurwicz:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()

        # Allocate memory for matrices used in solving the Schur system
        self._rhs = F.copy()
        self._R = B.codomain.zeros()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        self._solverA = inverse(A, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        self._alpha = 0.0001
        self._beta = 0.0001

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()

        # List to store residual norms
        self._residual_norms = []

        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schur = 0  # Iterations for _solverschur

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        # Initialize P to zero or given initial guess
        self._P = P_init if P_init is not None else self._P

        for iteration in range(self._max_iter):
            # Step 1: Compute velocity U by solving A U = -Bᵀ P + F
            self._rhs *= 0
            self._rhs -= self._B.transpose().dot(self._P)
            self._rhs -= self.A.dot(self._U)
            self._rhs += self._F
            self._U = self._U + self._alpha * self._rhs

            # Step 2: Compute residual R = BU (divergence of U)
            R = self._B.dot(self._U, out=self._R)
            residual = self._R.copy()
            residual_norm = np.linalg.norm(self._R.toarray())
            print(f"{residual_norm =}")
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                return self._U, self._P, self._solverA._info, self._residual_norms

            self._P += self._beta * self._R

        # Return with info if maximum iterations reached
        return self._U, self._P, self._solverA._info, self._residual_norms


class SaddlePointSolverGMRESsolution:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        rho: float,
        x: BlockVector,
        y: BlockVector,
        solver_name: str,
        massmatrix: WeightedMassOperator,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()
        self._x = x
        x1 = x[0]
        x2 = x[1]

        print(f"{x.shape =}")
        print(f"{np.max(abs(x[0][0].toarray())) =}")
        print(f"{x[1].shape =}")

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        block_domainM = BlockVectorSpace(self._A.domain, self._B.transpose().domain)
        block_codomainM = block_domainM
        blocks = [[self._A, self._B.transpose()], [self._B, None]]
        self._M = BlockLinearOperator(block_domainM, block_codomainM, blocks=blocks)
        self._RHS = BlockVector(block_domainM, blocks=[self._F, B.codomain.zeros()])

        print(f"{block_domainM =}")
        print(f"{self._F.shape =}")
        print(f"{B.codomain =}")

        x0 = BlockVector(block_domainM, blocks=[0.22 * x, 0.22 * y])

        # Mpc = MassMatrixPreconditioner(massmatrix)
        # Mpcblock = [[Mpc, None], [None, Mpc]]
        # MassMatrixBlock = BlockLinearOperator(A.domain, A.domain, blocks = Mpcblock )
        # # solver_massmatrix = inverse(MassMatrixBlock, solver_name, tol=tol,
        # #                         maxiter=max_iter, **solver_params)
        # approxschurcomplement = self._B @ MassMatrixBlock @self._BT
        # pcblocks = [[MassMatrixBlock, None], [None, approxschurcomplement]]
        # self._pc= BlockLinearOperator(block_domainM, block_codomainM, blocks=pcblocks)

        # self._solver_pc = inverse(self._pc, solver = 'cg', tol=tol,
        #                         maxiter=max_iter, **solver_params)

        # self._M = self._solver_pc@self._M
        # self._RHS = self._solver_pc@self._RHS

        # Allocate memory for matrices used in solving the Schur system
        self._rhs = self._F.copy()
        self._R = self._B.codomain.zeros()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        self._solverM = inverse(self._M, solver_name, tol=tol, maxiter=max_iter, x0=x0, **solver_params)

        self._alpha = 0.0001
        self._beta = 0.0001

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()

        # List to store residual norms
        self._residual_norms = []

        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schur = 0  # Iterations for _solverschur

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        # Initialize P to zero or given initial guess
        self._sol = self._solverM.dot(self._RHS)

        self._U = self._sol[0]
        self._P = self._sol[1]
        print(f"{self._sol =}")
        print(f"{self._U =}")
        print(f"{self._P =}")

        return self._U, self._P, self._solverM._info


class SaddlePointSolverGMRES:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        F: BlockVector,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        # Allocate memory for matrices used in solving the Schur system
        self._rhs = self._F.copy()
        self._R = self._B.codomain.zeros()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        self._block_domainM = BlockVectorSpace(self._A.domain, self._B.transpose().domain)
        self._block_codomainM = self._block_domainM
        self._blocks = [[self._A, None], [None, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)

        self._solverM = inverse(self._M, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()

        # List to store residual norms
        self._residual_norms = []

        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schur = 0  # Iterations for _solverschur

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, dt):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """
        self._M *= 0.0
        self._blocks = [[self._A, self._B.transpose()], [self._B, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)
        self._RHS = BlockVector(self._block_domainM, blocks=[self._F, self._B.codomain.zeros()])
        
        # use setter to update lhs matrix
        self._solverM.linop = self._M

        # Initialize P to zero or given initial guess
        self._sol = self._solverM.dot(self._RHS)
        self._U = self._sol[0]
        self._P = self._sol[1]
        print(f'{self._solverM._info["niter"] =}')
        return self._U, self._P, self._solverM._info

class SaddlePointSolverGMRESwithPC:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        y^{n+1} = \left[ B A^{-1} B^{\top}\\right]^{-1} B A^{-1} f \,,\\
        x^{n+1} = A^{-1} \left[ f - B^{\top} y^{n+1} \\right] \,.
        

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].
        
    A11: LinearOperator
        Preconditioner for upper left block from A. Not inverted
        
    A11: LinearOperator
        Preconditioner for lower right block from A. Not inverted.

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A: BlockLinearOperator,
        B: BlockLinearOperator,
        A11: SumLinearOperator,
        A22: SumLinearOperator,
        F: BlockVector,
        precdt,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        derham = None, 
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)
        assert isinstance(rho, float)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._A11 = A11
        self._A22 = A22
        self._B = B
        self._F = F
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter
        self._BT = B.transpose()
        self._derham = derham
        self._pc = precdt,

        if solver_params["pc"] is None:
            solver_params.pop("pc")

        # Allocate memory for matrices used in solving the Schur system
        self._rhs = self._F.copy()
        self._R = self._B.codomain.zeros()

        # initialize solver with dummy matrix A
        self._solver_name = solver_name
        self._block_domainM = BlockVectorSpace(self._A.domain, self._B.transpose().domain)
        self._block_codomainM = self._block_domainM
        
        ### Inverses for Preconditioner
        #A11 and A22 direct inverse with sparse solver
        # self._A11sparse = SparseSolver(self._A11.toarray_struphy(is_sparse=True))
        # self._A22sparse = SparseSolver(self._A22.toarray_struphy(is_sparse=True))
        # self._directA11inv_np = self._A11sparse.solve(np.identity(self._A11.shape[0]))
        # self._directA22inv_np = self._A22sparse.solve(np.identity(self._A22.shape[0]))
        # self._directA11inv = ArrayAsLinearOperator(self._directA11inv_np, self._derham, self._A11.domain, self._A11.domain)
        # self._directA22inv = ArrayAsLinearOperator(self._directA22inv_np, self._derham, self._A22.domain, self._A22.domain)
        # self._P = self._B[0,0]@self._directA11inv @ self._B[0,0].T + self._B[0,1] @ self._directA22inv @ self._B[0,1].T
        #_blocksinv = [[self._directA11inv, None], [None, self._directA22inv]]
        # _Ainv = BlockLinearOperator(self._A.domain, self._A.codomain, blocks=_blocksinv)
        
        
        # A11 and A22 conjugate gradient inverse
        self._solverA = inverse(self._A11, solver = 'pcg', pc=precdt, tol=tol, maxiter=max_iter, **solver_params)
        self._solverAe = inverse(self._A22, solver = 'pcg', pc=precdt, tol=tol, maxiter=max_iter, **solver_params)
        
        # Inverse of system
        self._blocks = [[self._A, None], [None, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)

        self._solverM = inverse(self._M, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._Utot = A.codomain.zeros()
        self._U = A[0,0].codomain.zeros()
        self._Ue = A[1,1].codomain.zeros()

        # List to store residual norms
        self._residual_norms = []

        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schur = 0  # Iterations for _solverschur

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A
    
    @property
    def A11(self):
        """Upper left block from A."""
        return self._A11
    
    @property
    def A22(self):
        """Lower right block from A."""
        return self._A22
    
    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a
        
    @A11.setter
    def A11(self, a):
        """Upper left block from A."""
        self._A11 = a

    @A22.setter
    def A22(self, a):
        """Lower right block from A."""
        self._A22 = a
    
    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, dt, U_init=None, Ue_init=None, P_init=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """
        
        self._P = P_init if P_init is not None else self._P
        self._U = U_init if U_init is not None else self._U
        self._Ue = Ue_init if U_init is not None else self._Ue
        
        self._M *= 0.0
        self._blocks = [[self._A, self._B.transpose()], [self._B, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)
        self._RHS = BlockVector(self._block_domainM, blocks=[self._F, self._B.codomain.zeros()])
        
        self._blockU = BlockVector(self._A.domain, blocks = [self._U, self._Ue])
        self._solblocks = [self._blockU, self._P]
        self._solverA._options["x0"]= self._U
        self._solverAe._options["x0"] = self._Ue
        self._solverM._options["x0"] = BlockVector(self._block_domainM, blocks=self._solblocks)
        
        ### Preconditioner
        _blocksinv = [[self._solverA, None], [None, self._solverAe]]
        _Ainv = BlockLinearOperator(self._A.domain, self._A.codomain, blocks=_blocksinv)
        self._Pre = IdentityOperator(self._B.codomain)#self._B @ _Ainv @ self._B.T # 
        _blocksPrecadded = [[_Ainv, None], [None, self._Pre]]
        self._Prec = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=_blocksPrecadded)        
        
        
        self._M = self._Prec@self._M
        self._RHS = self._Prec.dot(self._RHS)

        # use setter to update lhs matrix
        self._solverM.linop = self._M

        # Initialize P to zero or given initial guess
        self._sol = self._solverM.dot(self._RHS)
        self._Utot = self._sol[0]
        self._P = self._sol[1]
        print(f'{self._solverM._info=}')
        print(f'{self._solverA._info=}')
        print(f'{self._solverAe._info=}')
        return self._Utot, self._P, self._solverM._info


class SaddlePointSolverNoCGPaper:
    """Solves for math:`\left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)` in the block system

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            f \cr 0
        } \\right)

    using the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f`. The solution is given by

    .. math::

        x^{n+1} = A^{-1} \left[ f - B^{\top} y_n \\right] \,,\\
        R = B x^{n+1} \,,\\
        y^{n+1} = y_n - \rho  R \,.


    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    f : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    rho : float
        Descent parameter for the Uzawa iteration.

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    """

    def __init__(
        self,
        A11: ScaledLinearOperator,
        A22: ScaledLinearOperator,
        B1: ScaledLinearOperator,
        B2: ScaledLinearOperator,
        F1: Vector,
        F2: Vector,
        Pinit: Vector,
        rho: float,
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A11, LinearOperator)
        assert isinstance(A22, LinearOperator)
        assert isinstance(B1, LinearOperator)
        assert isinstance(B2, LinearOperator)
        assert isinstance(F1, Vector)
        assert isinstance(F2, Vector)
        assert isinstance(rho, float)

        assert A11.codomain == B1.domain
        assert A22.codomain == B2.domain

        # linear operators
        self._A11 = A11
        self._A22 = A22
        self._B1 = B1
        self._B2 = B2
        self._F1 = F1
        self._F2 = F2
        self._Pinit = Pinit
        self._rho = rho
        self._tol = tol
        self._max_iter = max_iter

        # Allocate memory for matrices used in solving the system
        self._rhs1 = self._F1.copy()
        self._rhs2 = self._F2.copy()
        self._rhs3 = B1.codomain.zeros()
        self._R = B1.codomain.zeros()
        self._Rf = B1.codomain.zeros()

        # Solution vectors
        self._P = B1.codomain.zeros()
        self._U = self._A11.codomain.zeros()
        self._Ue = self._A22.codomain.zeros()
        self._uf = self._A11.codomain.zeros()
        self._uef = self._A22.codomain.zeros()

        # initialize solver with matrix A
        self._solver_name = solver_name

        if solver_params["pc"] is None:
            solver_params.pop("pc")
       
        self._solverAe = inverse(self._A22, solver_name, tol=tol, maxiter=max_iter, **solver_params)
        self._solverA = inverse(self._A11, solver_name, tol=tol, maxiter=max_iter, **solver_params)
        self._solverB = inverse(self._B1, solver_name, tol=tol, maxiter=max_iter, **solver_params)
        # List to store residual norms
        self._residual_norms = []

    @property
    def A(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._A

    @property
    def B(self):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._B

    @property
    def F(self):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        return self._F

    @A.setter
    def A(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._A = a

    @B.setter
    def B(self, b):
        """Lower left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._B = b

    @F.setter
    def F(self, f):
        """Right hand side vector of the upper block of [A :math: `B^{\top}`]."""
        self._F = f

    def __call__(self, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        P_init : Vector, optional
            Initial guess for the potential. If None, initializes to zero.

        Returns
        -------
        U : Vector
            Solution vector for the velocity.

        P : Vector
            Solution vector for the potential.

        info : dict
            Convergence information.
        """

        assert self._F1.space == self._A11.domain
        assert self._F2.space == self._A22.domain

        # use setter to update lhs matrix
        self._solverA.linop = self._A11
        self._solverAe.linop = self._A22

        # Initialize P to zero or given initial guess
        self._P = self._Pinit if self._Pinit is not None else self._P
        for iteration in range(self._max_iter):
            # # Step 1a: Compute velocity U by solving A U = -Bᵀ P
            # self._rhs1 *= 0
            # self._rhs1 -= self._B1.transpose().dot(self._P)
            # self._U = self._solverA.dot(self._rhs1)
            # # Step 1b: Compute velocity Ue by solving Ae Ue = Bᵀ P
            # self._rhs2 *= 0
            # self._rhs2 -= self._B2.transpose().dot(self._P)
            # self._Ue = self._solverAe.dot(self._rhs2)

            # # Step 2: Compute residual R = -B(U-Ue) (divergence of U)
            # self._R = -(self._B1.dot(self._U)+self._B2.dot(self._Ue))
            # residual_norm = np.linalg.norm(self._R.toarray())
            # print(f"{residual_norm =}")
            
            # Step3 3a: uf: uf=A^-1 (f)
            self._rhs1 *= 0
            self._rhs1 += self._F1
            self._uf = self._solverA.dot(self._rhs1)
            
            # Step3 3b: ufe: ufe=Ae^-1 (fe)
            self._rhs2 *= 0
            self._rhs2 += self._F2
            self._uef = self._solverAe.dot(self._rhs2)
            
            
            self._rhs3 *= 0
            self._solverB.linop = self._B1.dot(self._uf) - self._B2.dot(self._uef)
            self._P = self._solverB.dot(self._rhs3)
            
            # Step 4: Compute residual R = -B((uf+U)-(ufe+Ue)) (divergence of U)
            self._Rf = self._B1.dot(self._uf) + self._B2.dot(self._uef)
            rnorm = np.linalg.norm(self._Rf.toarray())
            self._residual_norms.append(rnorm)  # Store residual norm
            
            self._U = self._uf - self._B1.transpose().dot(self._P)
            self._Ue = self._uf - self._B2.transpose().dot(self._P)
            
            print(f'residual : {rnorm}')
            # Check for convergence based on residual norm
            if rnorm < 1e-5:#self._tol:
                return self._uf, self._uef, self._P, self._solverA._info, self._residual_norms
            

        # Return with info if maximum iterations reached
        return self._U, self._Ue, self._P, self._solverA._info, self._residual_norms


class ArrayAsLinearOperator(LinearOperator):
    def __init__(self, array, derham, domain, codomain):
        self.array = array
        self._derham = derham
        self._domain = domain
        self._codomain = codomain
        super().__init__()
        
    @property    
    def transpose(self, conjugate=False):
        return super().transpose(conjugate)
    
    @property
    def toarray(self):
        return super().toarray()
    
    @property
    def tosparse(self):
        return super().tosparse()
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    def dot(self, v, out=None):
        return super().dot(v, out)

    @domain.setter
    def domain(self, new_domain):
        self._domain = new_domain
        
    @codomain.setter
    def codomain(self, new_codomain):
        self._codomain = new_codomain
        