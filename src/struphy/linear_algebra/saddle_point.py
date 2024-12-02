from psydac.linalg.basic import Vector, LinearOperator
from psydac.linalg.block import BlockLinearOperator, BlockVectorSpace, BlockVector
from psydac.linalg.solvers import inverse
import numpy as np
from scipy.linalg import null_space


class SaddlePointSolver:
    '''Solves for math:`\left( \matrix{
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
    '''

    def __init__(self,
                 A: BlockLinearOperator,
                 B: BlockLinearOperator,
                 F: BlockVector,
                 rho: float,
                 solver_name: str,
                 tol=1e-6,
                 max_iter=1000,
                 **solver_params):

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

        # Allocate memory for matrices used in solving the system
        self._rhs = F.copy()
        self._R = B.codomain.zeros()
        self._p2 = B.codomain.zeros()
        self._p1 = B.codomain.zeros()
        self._a2 = B.codomain.zeros()
        self._alpha = 0
        self._beta = 0
        
        #Counter
        self._iterationssolverA = 0

        # initialize solver with matrix A
        self._solver_name = solver_name

        if solver_params['pc'] is None:
            solver_params.pop('pc')

        self._solverA = inverse(A, solver_name, tol=tol,
                                maxiter=max_iter, **solver_params)
        

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()
        
        # List to store residual norms
        self._residual_norms = []

    @property
    def A(self):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._A

    @property
    def B(self):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._B

    @property
    def F(self):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        return self._F

    @A.setter
    def A(self, a):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._A = a

    @B.setter
    def B(self, b):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._B = b

    @F.setter
    def F(self, f):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        self._F = f

    def __call__(self, P_init=None, out=None):
        '''
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
        '''

        assert self._F.space == self._A.domain

        # Initialize P to zero or given initial guess
        P = P_init if P_init is not None else self._P
        # Step 1: Compute velocity U by solving A U = -Bᵀ P + F
        self._rhs *= 0
        self._rhs -= self._B.transpose().dot(P)
        self._rhs += self._F
        U = self._solverA.dot(self._rhs, out=self._U)
        self._iterationssolverA += self._solverA._info['niter']

        # Step 2: Compute residual R = BU 
        R = self._B.dot(self._U, out=self._R)
        residual = self._R.copy()
        self._p2 = residual

        for iteration in range(self._max_iter):
            residual_norm = np.linalg.norm(self._R.toarray())
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                print(f"{self._iterationssolverA =}")
                print(f"{iteration =}")
                return self._U, self._P, self._solverA._info, self._residual_norms

            self._p1 = self._solverA.dot(self._B.transpose().dot(self._p2))
            self._iterationssolverA += self._solverA._info['niter']
            self._a2 = self._B.dot(self._p1)
            self._alpha = self._p2.dot(self._R)/(self._p2.dot(self._a2))

            # Step 3: Update velocity u <- u - alpha * p1  and potential p <- p + alpha * p2
            self._P += self._alpha*self._p2
            self._R -= self._alpha*self._a2
            self._U -= self._alpha*self._p1

            self._beta = self._R.dot(self._a2) / (self._p2.dot(self._a2))
            self._p2 = self._R - self._beta*self._p2

        # Return with info if maximum iterations reached
        return self._U, self._P, self._solverA._info, self._residual_norms


class SaddlePointSolverTest:
    '''Solves for math:`\left( \matrix{
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
    '''

    def __init__(self,
                 A: BlockLinearOperator,
                 B: BlockLinearOperator,
                 F: BlockVector,
                 rho: float,
                 solver_name: str,
                 tol=1e-8,
                 max_iter=1000,
                 **solver_params):

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

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params['pc'] is None:
            solver_params.pop('pc')

        self._solverA = inverse(A, solver_name, tol=tol,
                                maxiter=max_iter, **solver_params)
        
        print(f"SolverA done")
        
        if solver_params.get('pc'): 
            solver_params.pop('pc')
       
        self._schur = self._B @ self._solverA @ self._BT
        self._solverschur = inverse(
            self._schur, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        print(f"Solver schur done")
        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()
        
        # Initialize counters
        self._iterations_solverA = 0  # Total iterations for _solverA
        self._iterations_schur = 0    # Iterations for _solverschur

    @property
    def A(self):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._A

    @property
    def B(self):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._B

    @property
    def F(self):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        return self._F

    @A.setter
    def A(self, a):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._A = a

    @B.setter
    def B(self, b):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._B = b

    @F.setter
    def F(self, f):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        self._F = f

    def __call__(self, P_init=None, out=None):
        '''
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
        '''

        assert isinstance(self._F, BlockVector)
        assert self._F.space == self._A.domain
        
        # #use setter to update lhs matrix
        # self._solverA.linop = self._A
        # self._solverschur.linop = self._schur

        Matrixproduct=self._solverschur @ self._B @ self._solverA
        # Step 1: Compute potential P by solving B A^-1 Bᵀ  P = B A^-1 F
        P = Matrixproduct.dot(self._F)
        
        # Track iterations for _solverschur
        self._iterations_schur += self._solverschur._info['niter']

        # Track iterations for _solverA (used during the Schur complement solve)
        
        print(f"Solved P with solverschur and solverA")

        # Step 2: Compute velocity by solving A U  = F - Bᵀ P
        self._rhs *= 0
        self._rhs -= self._BT.dot(P)
        self._rhs += self._F

        U = self._solverA.dot(self._rhs)
        self._iterations_solverA += self._solverA._info['niter']
        

        # Return with info if maximum iterations reached
        print(f"{self._iterations_schur =}")
        print(f"{self._iterations_solverA =}")
        return U, P, self._solverA._info
    
class SaddlePointSolverNoCG:
    '''Solves for math:`\left( \matrix{
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
    '''

    def __init__(self,
                 A: LinearOperator,
                 B: LinearOperator,
                 F: Vector,
                 rho: float,
                 solver_name: str,
                 tol=1e-8,
                 max_iter=1000,
                 **solver_params):

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
        self._Ainverse = A.copy()
        self._rhs = F.copy()
        self._R = B.codomain.zeros()

        # initialize solver with matrix A
        self._solver_name = solver_name

        if solver_params['pc'] is None:
            solver_params.pop('pc')

        self._solverA = inverse(A, solver_name, tol=tol,
                                maxiter=max_iter, **solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()
        
        # List to store residual norms
        self._residual_norms = []

    @property
    def A(self):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._A

    @property
    def B(self):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        return self._B

    @property
    def F(self):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        return self._F

    @A.setter
    def A(self, a):
        """ Upper left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._A = a

    @B.setter
    def B(self, b):
        """ Lower left block from [[A :math: `B^{\top}`], [B 0]].
        """
        self._B = b

    @F.setter
    def F(self, f):
        """ Right hand side vector of the upper block of [A :math: `B^{\top}`].
        """
        self._F = f

    def __call__(self, P_init=None, out=None):
        '''
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
        '''

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
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                return self._U, self._P, self._solverA._info, self._residual_norms


            self._P += self._rho*self._R

        # Return with info if maximum iterations reached
        return self._U, self._P, self._solverA._info, self._residual_norms