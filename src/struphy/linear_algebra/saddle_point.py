import numpy as np
import scipy as sc
from psydac.linalg.basic import LinearOperator, Vector, SumLinearOperator, ComposedLinearOperator
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.solvers import inverse
from psydac.linalg.direct_solvers import SparseSolver

from psydac.linalg.basic import IdentityOperator


class SaddlePointSolverUzawaNumpy:
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
        for i in A:
            assert isinstance(i, np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in B:
            assert isinstance(i, np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in F:
            assert isinstance(i, np.ndarray) or isinstance(i, sc.sparse.csr_matrix)
        for i in Apre:
            assert isinstance(i, np.ndarray) or isinstance(
                i, sc.sparse.csr_matrix) or isinstance(i, sc.sparse.csr_array)
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
        self._method_to_solve = method_to_solve  # 'SparseSolver', 'ScipySparse', 'InexactNPInverse', 'DirectNPInverse'
        self._preconditioner = preconditioner

        # if self._method_to_solve == 'SparseSolver':
        #     spectralanalysis = False

        if self._method_to_solve in ('InexactNPInverse', 'SparseSolver'):
            self._preconditioner = False

        self._Anp = self._A[0]
        self._Aenp = self._A[1]
        self._B1np = self._B[0]
        self._B2np = self._B[1]
        # if self._preconditioner == True or self._method_to_solve == 'InexactNPInverse':
        #     self._A11np = self._Apre[0]
        #     self._A22np = self._Apre[1]

        # Instanciate inverses
        self._setup_inverses()

        print(f'Arrays initialized')

        # Spectral analysis
        if spectralanalysis == True:
            # A11 before
            if self._method_to_solve in ('DirectNPInverse', 'InexactNPInverse'):
                eigvalsA11_before, eigvecs_before = np.linalg.eig(self._A[0])  # self._PA11diag)#@
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
                eigvalsA22_before, eigvecs_before = np.linalg.eig(self._A[1])  # self._PA22diag)#@
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
                    eigvalsA11_after_prec, eigvecs_after = np.linalg.eig(self._A11npinv@self._A[0])  # Implement this
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
                    eigvalsA22_after_prec, eigvecs_after = np.linalg.eig(self._A22npinv@self._A[1])  # Implement this
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
        self._Unp = np.zeros(self._A[0].shape[1])
        self._Uenp = np.zeros(self._A[1].shape[1])
        # Allocate memory for matrices used in solving the system
        self._rhs0np = self._F[0].copy()
        self._rhs1np = self._F[1].copy()
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
        need_update = True
        A0_old, A1_old = self._A
        A0_new, A1_new = a
        if self._method_to_solve in ("ScipySparse", "SparseSolver"):
            same_A0 = (A0_old != A0_new).nnz == 0
            same_A1 = (A1_old != A1_new).nnz == 0
        else:
            same_A0 = np.allclose(A0_old, A0_new, atol=1e-10)
            same_A1 = np.allclose(A1_old, A1_new, atol=1e-10)
        if same_A0 and same_A1:
            need_update = False
        self._A = a
        self._Anp = self._A[0]
        self._Aenp = self._A[1]
        if need_update:
            self._setup_inverses()

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
        need_update = True
        A0_old, A1_old = self._Apre
        A0_new, A1_new = a
        if self._method_to_solve in ("ScipySparse", "SparseSolver"):
            same_A0 = (A0_old != A0_new).nnz == 0
            same_A1 = (A1_old != A1_new).nnz == 0
        else:
            same_A0 = np.allclose(A0_old, A0_new, atol=1e-10)
            same_A1 = np.allclose(A1_old, A1_new, atol=1e-10)
        if same_A0 and same_A1:
            need_update = False
        self._Apre = a
        if need_update:
            self._setup_inverses()

    def __call__(self, U_init=None, Ue_init=None, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        U_init : Vector, np.ndarray or sc.sparse.csr.csr_matrix, optional
            Initial guess for the velocity of the ions. If None, initializes to zero.

        Ue_init : Vector, np.ndarray or sc.sparse.csr.csr_matrix, optional
            Initial guess for the velocity of the electrons. If None, initializes to zero.

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
        info = {}

        # Initialize P to zero or given initial guess
        if isinstance(U_init, np.ndarray) or isinstance(U_init, sc.sparse.csr.csr_matrix):
            self._Pnp = P_init if P_init is not None else self._P
            self._Unp = U_init if U_init is not None else self._U
            self._Uenp = Ue_init if U_init is not None else self._Ue

        else:
            self._Pnp = P_init.toarray() if P_init is not None else self._Pnp
            self._Unp = U_init.toarray() if U_init is not None else self._Unp
            self._Uenp = Ue_init.toarray() if U_init is not None else self._Uenp

        # print(f'Is incoming a solution in solver?')
        # TestRest1 = self._F[0]-self._A[0].dot(self._Unp)-self._B[0].T.dot(self._Pnp)
        # print(f'{max(TestRest1) =}')
        # TestRest2 = self._F[1]-self._A[1].dot(self._Uenp)-self._B[1].T.dot(self._Pnp)
        # print(f'{max(TestRest2) =}')
        # TestRest3 = self._B[0].dot(self._Unp)+self._B[1].dot(self._Uenp)
        # print(f'{max(TestRest3) =}')

        for iteration in range(self._max_iter):
            # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
            self._rhs0np *= 0
            self._rhs0np -= self._B1np.transpose().dot(self._Pnp)
            self._rhs0np -= self._Anp.dot(self._Unp)
            self._rhs0np += self._F[0]
            if self._preconditioner == False:
                self._Unp += self._Anpinv.dot(self._rhs0np)
            elif self._preconditioner == True:
                self._Unp += self._Anpinv.dot(self._A11npinv@self._rhs0np)

            R1 = self._B1np.dot(self._Unp)
            #print(f'{np.linalg.norm(R1) = }')

            self._rhs1np *= 0
            self._rhs1np -= self._B2np.transpose().dot(self._Pnp)
            self._rhs1np -= self._Aenp.dot(self._Uenp)
            self._rhs1np += self._F[1]
            if self._preconditioner == False:
                self._Uenp += self._Aenpinv.dot(self._rhs1np)
            elif self._preconditioner == True:
                self._Uenp += self._Aenpinv.dot(self._A22npinv@self._rhs1np)

            R2 = self._B2np.dot(self._Uenp)
            #print(f'{np.linalg.norm(R2) = }')

            # Step 2: Compute residual R = BU (divergence of U)
            R = R1+R2  # self._B1np.dot(self._Unp) + self._B2np.dot(self._Uenp)
            residual_norm = np.linalg.norm(R)
            residual_normR1 = np.linalg.norm(R)
            #print(f"{residual_norm =}")
            self._residual_norms.append(residual_normR1)  # Store residual norm
            # Check for convergence based on residual norm
            if residual_norm < self._tol:
                info["success"] = True
                info["niter"] = iteration+1
                # print(f'Is outgoing a solution in solver?')
                # TestRest1 = self._F[0]-self._A[0].dot(self._Unp)-self._B[0].T.dot(self._Pnp)
                # print(f'{max(TestRest1) =}')
                # TestRest2 = self._F[1]-self._A[1].dot(self._Uenp)-self._B[1].T.dot(self._Pnp)
                # print(f'{max(TestRest2) =}')
                # TestRest3 = self._B[0].dot(self._Unp)+self._B[1].dot(self._Uenp)
                # print(f'{max(TestRest3) =}')
                return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms

            alpha = (R.dot(R))/(R.dot(self._Precnp.dot(R)))
            #alpha = (R.dot(R))/(R.dot(self._Precsparsenp.dot(R)))
            self._stepsize = 0.5*self._stepsize + 0.5 * alpha
            #self._P += alpha * R
            self._Pnp += alpha.real * R.real

        # Return with info if maximum iterations reached
        info["success"] = False
        info["niter"] = iteration+1
        return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms

    def _setup_inverses(self):

        A0 = self._A[0]
        A1 = self._A[1]

        # === Preconditioner inverses, if used
        if self._preconditioner:
            A11 = self._Apre[0]
            A22 = self._Apre[1]

            if hasattr(self, "_A11npinv") and self._is_inverse_still_valid(self._A11npinv, A11, "A11 pre"):
                pass
            else:
                self._A11npinv = self._compute_inverse(A11, which="A11 pre")

            if hasattr(self, "_A22npinv") and self._is_inverse_still_valid(self._A22npinv, A22, "A22 pre"):
                pass
            else:
                self._A22npinv = self._compute_inverse(A22, which="A22 pre")

            # === Inverse for A[0] if preconditioned
            if hasattr(self, "_Anpinv") and self._is_inverse_still_valid(self._Anpinv, A0, "A[0]", pre=self._A11npinv):
                pass
            else:
                self._Anpinv = self._compute_inverse(self._A11npinv@A0, which="A[0]")

            # === Inverse for A[1]
            if hasattr(self, "_Aenpinv") and self._is_inverse_still_valid(self._Aenpinv, A1, "A[1]", pre=self._A22npinv):
                pass
            else:
                self._Aenpinv = self._compute_inverse(self._A22npinv@A1, which="A[1]")

        else:  # No preconditioning:
            # === Inverse for A[0]
            if hasattr(self, "_Anpinv") and self._is_inverse_still_valid(self._Anpinv, A0, "A[0]"):
                pass
            else:
                self._Anpinv = self._compute_inverse(A0, which="A[0]")

            # === Inverse for A[1]
            if hasattr(self, "_Aenpinv") and self._is_inverse_still_valid(self._Aenpinv, A1, "A[1]"):
                pass
            else:
                self._Aenpinv = self._compute_inverse(A1, which="A[1]")

        # Precompute Schur complement
        self._Precnp = self._B1np @ self._Anpinv @ self._B1np.T + self._B2np @ self._Aenpinv @ self._B2np.T

    def _is_inverse_still_valid(self, inv, mat, name="", pre=None):
        #try:
        if pre is not None:
            test_mat = pre @ mat
        else:
            test_mat = mat
        I_approx = inv @ test_mat

        if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            I_exact = np.eye(test_mat.shape[0])
            if not np.allclose(I_approx, I_exact, atol=1e-6):
                diff = I_approx - I_exact
                max_abs = np.abs(diff).max()
                print(f"{name} inverse is NOT valid anymore. Max diff: {max_abs:.2e}")
                return False
            print(f"{name} inverse is still valid.")
            return True
        elif self._method_to_solve == "ScipySparse":
            I_exact = sc.sparse.identity(I_approx.shape[0], format=I_approx.format)
            diff = (I_approx - I_exact).tocoo()
            max_abs = np.abs(diff.data).max() if diff.nnz > 0 else 0.0

            if max_abs > 1e-6:
                print(f"{name} inverse is NOT valid anymore.")
                print(f"Max absolute difference: {max_abs:.2e}")
                print(f"Number of differing entries: {diff.nnz}")
                return False
            print(f"{name} inverse is still valid.")
            return True

    def _compute_inverse(self, mat, which="matrix"):
        print(f"Computing inverse for {which} using method {self._method_to_solve}")
        if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            return np.linalg.inv(mat)
        elif self._method_to_solve == "ScipySparse":
            return sc.sparse.linalg.inv(mat)
        elif self._method_to_solve == "SparseSolver":
            solver = SparseSolver(mat)
            return solver.solve(np.eye(mat.shape[0]))
        else:
            raise ValueError(f"Unknown solver method {self._method_to_solve}")


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

    using on of the solvers given in [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47).
    Prefered solver is GMRES.


    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A :math: `B^{\top}`B], [B 0]].

    B : LinearOperator
        Lower left block from [[A :math: `B^{\top}`], [B 0]].

    F : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

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
        U_init : Vector, optional
            Initial guess for the velocity of the ions. If None, initializes to zero.

        Ue_init : Vector, optional
            Initial guess for the velocity of the electrons. If None, initializes to zero.

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

        self._blockU = BlockVector(self._A.domain, blocks=[self._U1, self._U2])
        self._solblocks = [self._blockU, self._P1]
        x0 = BlockVector(self._block_domainM, blocks=self._solblocks)
        self._solverM._options["x0"] = x0

        self._M *= 0.0
        self._blocks = [[self._A, self._B.T], [self._B, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)
        self._RHS = BlockVector(self._block_domainM, blocks=[self._F, self._B.codomain.zeros()])

        # use setter to update lhs matrix
        self._solverM.linop = self._M

        # Initialize P to zero or given initial guess
        self._sol = self._solverM.dot(self._RHS)
        self._U = self._sol[0]
        self._P = self._sol[1]

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

    The system is left-hand-side preconditioned and solved using on of the solvers given in [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47).
    The left hand side preconditioner inverse is calculated with a preconditioned conjugate gradient.


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

    F : Linear Vector
        Right hand side vector of the upper block from [A :math: `B^{\top}`B].

    precdt: MassMatrixPreconditioner
        Preconditioner for the pcg needed for the inverse of the preconditioner.

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
        solver_name: str,
        tol=1e-8,
        max_iter=1000,
        derham=None,
        **solver_params,
    ):
        assert isinstance(A, BlockLinearOperator) or isinstance(A, LinearOperator)
        assert isinstance(B, BlockLinearOperator) or isinstance(B, LinearOperator)
        assert isinstance(F, BlockVector) or isinstance(F, Vector)

        assert A.domain == B.domain

        # linear operators
        self._A = A
        self._A11 = A11
        self._A22 = A22
        self._B = B
        self._F = F
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
        self._solverA = inverse(self._A11, solver='pcg', pc=precdt, tol=tol, maxiter=max_iter, **solver_params)
        self._solverAe = inverse(self._A22, solver='pcg', pc=precdt, tol=tol, maxiter=max_iter, **solver_params)

        # Inverse of system
        self._blocks = [[self._A, None], [None, None]]
        self._M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)

        self._solverM = inverse(self._M, solver_name, tol=tol, maxiter=max_iter, **solver_params)

        # Solution vectors
        self._P = B.codomain.zeros()
        self._Utot = A.codomain.zeros()
        self._U = A[0, 0].codomain.zeros()
        self._Ue = A[1, 1].codomain.zeros()

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
        U_init : Vector, optional
            Initial guess for the velocity of the ions. If None, initializes to zero.

        Ue_init : Vector, optional
            Initial guess for the velocity of the electrons. If None, initializes to zero.

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

        self._blockU = BlockVector(self._A.domain, blocks=[self._U, self._Ue])
        self._solblocks = [self._blockU, self._P]
        self._solverA._options["x0"] = self._U
        self._solverAe._options["x0"] = self._Ue
        self._solverM._options["x0"] = BlockVector(self._block_domainM, blocks=self._solblocks)

        # Preconditioner
        _blocksinv = [[self._solverA, None], [None, self._solverAe]]
        _Ainv = BlockLinearOperator(self._A.domain, self._A.codomain, blocks=_blocksinv)
        self._Pre = IdentityOperator(self._B.codomain)  # self._B @ _Ainv @ self._B.T #
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
