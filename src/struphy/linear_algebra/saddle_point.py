import numpy as np
import scipy as sc
from psydac.linalg.basic import LinearOperator, Vector, SumLinearOperator, ComposedLinearOperator
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.solvers import inverse
from psydac.linalg.direct_solvers import SparseSolver

from psydac.linalg.basic import IdentityOperator


class SaddlePointSolverUzawaNumpy():
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
        
    The algorithm works for 
    
    Parameters
    ----------
    A : list
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].
        The entries on the diagonals of block A are given as list of np.ndarray or sc.sparse.csr_matrix.

    B : list
        Lower left block from [[A :math: `B^{\top}`], [B 0]].
        All entries of block B are given as list of np.ndarray or sc.sparse.csr_matrix.

    F : list
        Right hand side of the upper block from [A :math: `B^{\top}`B]. Given as list of np.ndarray or sc.sparse.csr_matrix.
    
    Apre : list
        The non-inverted preconditioner for entries on the diagonals of block A are given as list of np.ndarray or sc.sparse.csr_matrix.

    method_to_solve : str
        Method for the inverses. Choose from 'DirectNPInverse', 'ScipySparse', 'InexactNPInverse' ,'SparseSolver'.

    preconditioner : bool
        Wheter to use preconditioners given in Apre or not.
        
    spectralanalysis : str
        Do the spectralanalyis for the matrices in A and if preconditioner given, compare them to the preconditioned matrices. 
        
    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.
    """

    def __init__(
        self,
        A: list,
        B: list,
        F: list,
        Apre: list,
        method_to_solve: str,
        preconditioner: bool,
        spectralanalysis: bool,
        tol=1e-8,
        max_iter=1000,
    ):
        assert isinstance(A, list)
        assert isinstance(B, list)
        assert isinstance(F, list)
        assert isinstance(Apre, list) 
        for i in A: assert isinstance(i,np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in B: assert isinstance(i,np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in F: assert isinstance(i,np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in Apre: assert isinstance(i,np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        assert method_to_solve in ('SparseSolver', 'ScipySparse', 'InexactNPInverse', 'DirectNPInverse')
        assert isinstance(preconditioner, bool)
        assert isinstance(spectralanalysis, bool)
        assert A[0].shape[0] == B[0].shape[1]
        assert A[0].shape[1] == B[0].shape[1]
        assert A[1].shape[0] == B[1].shape[1]
        assert A[1].shape[1] == B[1].shape[1]

        # linear operators
        self._A = A
        self._Apre = Apre
        self._B = B
        self._F = F
        self._tol = tol
        self._max_iter = max_iter
        self._method_to_solve = method_to_solve #  'SparseSolver', 'ScipySparse', 'InexactNPInverse', 'DirectNPInverse'
        self._preconditioner = preconditioner
        
        if self._method_to_solve == 'SparseSolver':
            spectralanalysis = False
            
        if self._method_to_solve in ('InexactNPInverse','SparseSolver'):
            self._preconditioner = False
        
        self._Anp = self._A[0]
        self._Aenp = self._A[1]
        self._B1np = self._B[0]
        self._B2np = self._B[1]
        self._F1np = self._F[0]
        self._F2np = self._F[1]
        if self._preconditioner == True or self._method_to_solve == 'InexactNPInverse':
            self._A11np = self._Apre[0]
            self._A22np = self._Apre[1]
        
        
        print(f'Arrays initialized')
            
         ### Solver inverse
        if self._method_to_solve == 'ScipySparse':
            if self._preconditioner == False:
                self._Anpinv = sc.sparse.linalg.inv(self._Anp)
                self._Aenpinv = sc.sparse.linalg.inv(self._Aenp)
            elif self._preconditioner == True:
                self._A11npinv = sc.sparse.linalg.inv(self._A11np)
                self._A22npinv = sc.sparse.linalg.inv(self._A22np)
                self._Anpinv = sc.sparse.linalg.inv(self._A11npinv@self._Anp)
                self._Aenpinv = sc.sparse.linalg.inv(self._A22npinv@self._Aenp)
        elif self._method_to_solve == 'DirectNPInverse':
            if self._preconditioner == False:
                self._Anpinv = np.linalg.inv(self._Anp)
                self._Aenpinv = np.linalg.inv(self._Aenp)
            elif self._preconditioner == True:
                self._A11npinv = np.linalg.inv(self._A11np)
                self._A22npinv = np.linalg.inv(self._A22np)
                self._Anpinv = np.linalg.inv(self._A11npinv@self._Anp)
                self._Aenpinv = np.linalg.inv(self._A22npinv@self._Aenp)
        elif self._method_to_solve == 'SparseSolver':
            if self._preconditioner == False:
                self._directAnp = SparseSolver(self._Anp)
                self._directAenp = SparseSolver(self._Aenp)
                self._Anpinv = self._directAnp.solve(np.identity(self._A[0].shape[0]))
                self._Aenpinv = self._directAenp.solve(np.identity(self._A[1].shape[0])) 
            elif self._preconditioner == True:
                self._directA11np = SparseSolver(self._A11np)
                self._directA22np = SparseSolver(self._A22np)
                self._A11npinv = self._directA11np.solve(np.identity(self._A[0].shape[0]))
                self._A22npinv = self._directA22np.solve(np.identity(self._A[1].shape[0]))
                self._directAnp = SparseSolver(self._A11npinv@self._Anp)
                self._directAenp = SparseSolver(self._A22npinv@self._Aenp)
                self._Anpinv = self._directAnp.solve(np.identity(self._A[0].shape[0]))
                self._Aenpinv = self._directAenp.solve(np.identity(self._A[1].shape[0]))
        elif self._method_to_solve == 'InexactNPInverse':
            self._Anpinv = np.linalg.inv(self._A11np)
            self._Aenpinv = np.linalg.inv(self._A22np)
                       
        self._Precnp = self._B1np@self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T
        
        ### Spectral analysis
        if spectralanalysis == True:
            # A11 before
            if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                eigvalsA11_before, eigvecs_before = np.linalg.eig(self._A[0])    #self._PA11diag)#@
            elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                eigvalsA11_before, eigvecs_before = np.linalg.eig(self._A[0].toarray())
            maxbeforeA11 = max(eigvalsA11_before)
            maxbeforeA11_abs = np.max(np.abs(eigvalsA11_before))
            minbeforeA11_abs = np.min(np.abs(eigvalsA11_before))
            minbeforeA11 = min(eigvalsA11_before)
            specA11_bef = maxbeforeA11/minbeforeA11
            specA11_bef_abs = maxbeforeA11_abs/minbeforeA11_abs
            # print(f'{maxbeforeA11 = }')
            # print(f'{maxbeforeA11_abs = }')
            # print(f'{minbeforeA11_abs = }')
            # print(f'{minbeforeA11 = }')
            # print(f'{specA11_bef = }')
            print(f'{specA11_bef_abs = }')
            
            # A22 before
            if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                eigvalsA22_before, eigvecs_before = np.linalg.eig(self._A[1])   #self._PA22diag)#@
            elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                eigvalsA22_before, eigvecs_before = np.linalg.eig(self._A[1].toarray())
            maxbeforeA22 = max(eigvalsA22_before)
            maxbeforeA22_abs = np.max(np.abs(eigvalsA22_before))
            minbeforeA22_abs = np.min(np.abs(eigvalsA22_before))
            minbeforeA22 = min(eigvalsA22_before)
            specA22_bef = maxbeforeA22/minbeforeA22
            specA22_bef_abs = maxbeforeA22_abs/minbeforeA22_abs
            # print(f'{maxbeforeA22 = }')
            # print(f'{maxbeforeA22_abs = }')
            # print(f'{minbeforeA22_abs = }')
            # print(f'{minbeforeA22 = }')
            # print(f'{specA22_bef = }')
            print(f'{specA22_bef_abs = }')
            

            
            if self._preconditioner == True:
                # A11 after preconditioning with its inverse
                if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                    eigvalsA11_after_prec, eigvecs_after = np.linalg.eig(self._A11npinv@self._A[0])     ### Implement this
                elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                    eigvalsA11_after_prec, eigvecs_after = np.linalg.eig((self._A11npinv@self._A[0]).toarray())
                maxafterA11_prec = max(eigvalsA11_after_prec)
                minafterA11_prec = min(eigvalsA11_after_prec)
                maxafterA11_abs_prec = np.max(np.abs(eigvalsA11_after_prec))
                minafterA11_abs_prec = np.min(np.abs(eigvalsA11_after_prec))
                specA11_aft_prec = maxafterA11_prec/minafterA11_prec
                specA11_aft_abs_prec = maxafterA11_abs_prec/minafterA11_abs_prec
                # print(f'{maxafterA11_prec = }')
                # print(f'{maxafterA11_abs_prec = }')
                # print(f'{minafterA11_abs_prec = }')
                # print(f'{minafterA11_prec = }')
                # print(f'{specA11_aft_prec = }')
                print(f'{specA11_aft_abs_prec = }')
                
                # A22 after preconditioning with its inverse
                if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                    eigvalsA22_after_prec, eigvecs_after = np.linalg.eig(self._A22npinv@self._A[1])     ### Implement this
                elif self._method_to_solve in ('SparseSolver', 'ScipySparse'):
                    eigvalsA22_after_prec, eigvecs_after = np.linalg.eig((self._A22npinv@self._A[1]).toarray())
                maxafterA22_prec = max(eigvalsA22_after_prec)
                minafterA22_prec = min(eigvalsA22_after_prec)
                maxafterA22_abs_prec = np.max(np.abs(eigvalsA22_after_prec))
                minafterA22_abs_prec = np.min(np.abs(eigvalsA22_after_prec))
                specA22_aft_prec = maxafterA22_prec/minafterA22_prec
                specA22_aft_abs_prec = maxafterA22_abs_prec/minafterA22_abs_prec
                # print(f'{maxafterA22_prec = }')
                # print(f'{maxafterA22_abs_prec = }')
                # print(f'{minafterA22_abs_prec = }')
                # print(f'{minafterA22_prec = }')
                # print(f'{specA22_aft_prec = }')
                print(f'{specA22_aft_abs_prec = }')
        
        print(f'Inverses initialized as linear operators')
        
        
       
        # Solution vectors numpy
        self._Pnp = np.zeros(self._B1np.shape[0])
        self._Unp = np.zeros(self._Anp.shape[1])
        self._Uenp = np.zeros(self._Aenp.shape[1])
        # Allocate memory for matrices used in solving the system
        self._rhs0np = self._F1np.copy()
        self._rhs1np = self._F2np.copy()
        self._Rnp = np.zeros(self._B[0].shape[1]+self._B[1].shape[1])

        # List to store residual norms
        self._residual_norms = []
        self._stepsize = 0.


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
    def Apre(self):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        return self._Apre

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
    def Apre(self, a):
        """Upper left block from [[A :math: `B^{\top}`], [B 0]]."""
        self._Apre = a
        

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
        info={}
        
        # Initialize P to zero or given initial guess
        if isinstance(U_init, np.ndarray) or isinstance (U_init, sc.sparse.csr.csr_matrix):
            self._P = P_init if P_init is not None else self._P
            self._U = U_init if U_init is not None else self._U
            self._Ue = Ue_init if U_init is not None else self._Ue
         
        else:
            self._Pnp = P_init.toarray() if P_init is not None else self._Pnp
            self._Unp = U_init.toarray() if U_init is not None else self._Unp
            self._Uenp = Ue_init.toarray() if U_init is not None else self._Uenp
            

        for iteration in range(self._max_iter):
            # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
            self._rhs0np *= 0
            self._rhs0np -= self._B1np.transpose().dot(self._Pnp)
            self._rhs0np -= self._Anp.dot(self._Unp)
            self._rhs0np += self._F1np
            if self._preconditioner == False:
                self._Unp += self._Anpinv.dot(self._rhs0np)    
            elif self._preconditioner == True:
                self._Unp += self._Anpinv.dot(self._A11npinv@self._rhs0np)   
            
            
            R1 = self._B1np.dot(self._Unp)
            #print(f'{np.linalg.norm(R1) = }')
            
            self._rhs1np *= 0
            self._rhs1np -= self._B2np.transpose().dot(self._Pnp)
            self._rhs1np -= self._Aenp.dot(self._Uenp)
            self._rhs1np += self._F2np
            if self._preconditioner == False:
                self._Uenp += self._Aenpinv.dot(self._rhs1np)  
            elif self._preconditioner == True:
                self._Uenp += self._Aenpinv.dot(self._A22npinv@self._rhs1np)    
            
            R2 = self._B2np.dot(self._Uenp)
            #print(f'{np.linalg.norm(R2) = }')
            
            # Step 2: Compute residual R = BU (divergence of U)
            R = R1+R2 #self._B1np.dot(self._Unp) + self._B2np.dot(self._Uenp)
            residual_norm = np.linalg.norm(R)
            #print(f"{residual_norm =}")
            self._residual_norms.append(residual_norm)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                info["success"] = True
                info["niter"] = iteration+1
                # print(f'{max(self._Unp) = }')
                # print(f'{max(self._Uenp) = }')
                # print(f'{max(self._Pnp) = }')
                # print(f'{min(self._Unp) = }')
                # print(f'{min(self._Uenp) = }')
                # print(f'{min(self._Pnp) = }')
                # TestRest1 = self._F1np-self._Anp.dot(self._Unp)-self._B1np.T.dot(self._Pnp)
                # print(f'{max(TestRest1) =}')
                # TestRest2 = self._F2np-self._Aenp.dot(self._Uenp)-self._B2np.T.dot(self._Pnp)
                # print(f'{max(TestRest2) =}')
                # TestRest3 = self._B1np.dot(self._Unp)+self._B2np.dot(self._Uenp)
                # print(f'{max(TestRest3) =}')
                return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms
                
            
            alpha = (R.dot(R))/(R.dot(self._Precnp.dot(R)))
            #alpha = (R.dot(R))/(R.dot(self._Precsparsenp.dot(R)))
            self._stepsize = 0.5*self._stepsize + 0.5* alpha
            #self._P += alpha * R
            self._Pnp += alpha.real * R.real

        # Return with info if maximum iterations reached
        info["success"] = False
        info["niter"] = iteration+1
        return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms
            

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
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._B = B
        self._F = F
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
        self._blocks = [[self._A, self._B.T], [self._B, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)

        self._solverM = inverse(self._M, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._U = A.codomain.zeros()
        self._Utmp = F.copy()*0

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

    def __call__(self, U_init=None, Ue_init=None, P_init=None):
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
        self._P1 = P_init if P_init is not None else self._P
        self._U1 = U_init if U_init is not None else self._Utmp[0]
        self._U2 = Ue_init if U_init is not None else self._Utmp[1]
        
        self._blockU = BlockVector(self._A.domain, blocks = [self._U1, self._U2])
        self._solblocks = [self._blockU, self._P1]
        x0 = BlockVector(self._block_domainM, blocks=self._solblocks)
        self._solverM._options["x0"] = x0
        
        self._M *= 0.0
        self._blocks = [[self._A, self._B.T], [self._B, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)
        self._RHS = BlockVector(self._block_domainM, blocks=[self._F, self._B.codomain.zeros()])
        
        
        # use setter to update lhs matrix
        self._solverM.linop = self._M
        TestRest1 = self._F[0]-self._A[0,0].dot(self._U1)-self._B[0,0].T.dot(self._P1)
        print(f'{max(TestRest1.toarray()) =}')
        TestRest2 = self._F[1]-self._A[1,1].dot(self._U2)-self._B[0,1].T.dot(self._P1)
        print(f'{max(TestRest2.toarray()) =}')
        TestRest3 = self._B[0,0].dot(self._U1)+self._B[0,1].dot(self._U2)
        print(f'{max(TestRest3.toarray()) =}')
        
        TestM = self._RHS - self._M.dot(x0)
        print(f'{max(TestM.toarray()) =}')
        
        # Initialize P to zero or given initial guess
        self._sol = self._solverM.dot(self._RHS)
        self._U = self._sol[0]
        self._P = self._sol[1]
        # TestRest1 = self._F[0]-self._A[0,0].dot(self._U[0])-self._B[0,0].T.dot(self._P)
        # print(f'{max(TestRest1.toarray()) =}')
        # TestRest2 = self._F[1]-self._A[1,1].dot(self._U[1])-self._B[0,1].T.dot(self._P)
        # print(f'{max(TestRest2.toarray()) =}')
        # TestRest3 = self._B[0,0].dot(self._U[0])+self._B[0,1].dot(self._U[1])
        # print(f'{max(TestRest3.toarray()) =}')
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

        