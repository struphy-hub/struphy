from typing import Union

import cunumpy as xp
import scipy as sc
from psydac.linalg.basic import LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.solvers import inverse

from struphy.linear_algebra.tests.test_saddlepoint_massmatrices import _plot_residual_norms


class SaddlePointSolver:
    r"""Solves for :math:`(x, y)` in the saddle point problem

    .. math::

        \left( \matrix{
            A &  B^{\top} \cr
            B & 0
        } \right)
        \left( \matrix{
            x \cr y
        } \right)
        =
        \left( \matrix{
            f \cr 0
        } \right)

    using either the Uzawa iteration :math:`BA^{-1}B^{\top} y = BA^{-1} f` or using on of the solvers given in :mod:`psydac.linalg.solvers`. The prefered solver is GMRES.
    The decission which variant to use is given by the type of A. If A is of type list of xp.ndarrays or sc.sparse.csr_matrices, then this class uses the Uzawa algorithm.
    If A is of type LinearOperator or BlockLinearOperator, a solver is used for the inverse.
    Using the Uzawa algorithm, solution is given by:

    .. math::

        y = \left[ B A^{-1} B^{\top}\right]^{-1} B A^{-1} f \,, \qquad
        x = A^{-1} \left[ f - B^{\top} y \right] \,.

    Parameters
    ----------
    A : list, LinearOperator or BlockLinearOperator
        Upper left block.
        Either the entries on the diagonals of block A are given as list of xp.ndarray or sc.sparse.csr_matrix.
        Alternative: Give whole matrice A as LinearOperator or BlockLinearOperator.
        list: Uzawa algorithm is used.
        LinearOperator: A solver given in :mod:`psydac.linalg.solvers` is used. Specified by solver_name.
        BlockLinearOperator: A solver given in :mod:`psydac.linalg.solvers` is used. Specified by solver_name.

    B : list, LinearOperator or BlockLinearOperator
        Lower left block.
        Uzwaw Algorithm: All entries of block B are given either as list of xp.ndarray or sc.sparse.csr_matrix.
        Solver: Give whole B as LinearOperator or BlocklinearOperator

    F : list
        Right hand side of the upper block.
        Uzawa: Given as list of xp.ndarray or sc.sparse.csr_matrix.
        Solver: Given as LinearOperator or BlockLinearOperator

    Apre : list
        The non-inverted preconditioner for entries on the diagonals of block A are given as list of xp.ndarray or sc.sparse.csr_matrix. Only required for the Uzawa algorithm.

    method_to_solve : str
        Method for the inverses. Choose from 'DirectNPInverse', 'ScipySparse', 'InexactNPInverse' ,'SparseSolver'. Only required for the Uzawa algorithm.

    preconditioner : bool
        Wheter to use preconditioners given in Apre or not. Only required for the Uzawa algorithm.

    spectralanalysis : bool
        Do the spectralanalyis for the matrices in A and if preconditioner given, compare them to the preconditioned matrices. Only possible if A is given as list.

    dimension : str
        Which of the predefined manufactured solutions to use ('1D', '2D' or 'Restelli')

    tol : float
        Convergence tolerance for the potential residual.

    max_iter : int
        Maximum number of iterations allowed.
    """

    def __init__(
        self,
        A: Union[list, LinearOperator, BlockLinearOperator],
        B: Union[list, LinearOperator, BlockLinearOperator],
        F: Union[list, Vector, BlockVector],
        Apre: list = None,
        method_to_solve: str = "DirectNPInverse",
        preconditioner: bool = False,
        spectralanalysis: bool = False,
        dimension: str = "2D",
        solver_name: str = "GMRES",
        tol: float = 1e-8,
        max_iter: int = 1000,
        **solver_params,
    ):
        assert type(A) == type(B)
        if isinstance(A, list):
            self._variant = "Uzawa"
            for i in A:
                assert isinstance(i, xp.ndarray) or isinstance(i, sc.sparse.csr_matrix)
            for i in B:
                assert isinstance(i, xp.ndarray) or isinstance(i, sc.sparse.csr_matrix)
            for i in F:
                assert isinstance(i, xp.ndarray) or isinstance(i, sc.sparse.csr_matrix)
            for i in Apre:
                assert (
                    isinstance(i, xp.ndarray)
                    or isinstance(i, sc.sparse.csr_matrix)
                    or isinstance(i, sc.sparse.csr_array)
                )
            assert method_to_solve in ("SparseSolver", "ScipySparse", "InexactNPInverse", "DirectNPInverse")
            assert A[0].shape[0] == B[0].shape[1]
            assert A[0].shape[1] == B[0].shape[1]
            assert A[1].shape[0] == B[1].shape[1]
            assert A[1].shape[1] == B[1].shape[1]

            self._method_to_solve = (
                method_to_solve  # 'SparseSolver', 'ScipySparse', 'InexactNPInverse', 'DirectNPInverse'
            )
            self._preconditioner = preconditioner

        elif isinstance(A, LinearOperator) or isinstance(A, BlockLinearOperator):
            self._variant = "Inverse_Solver"
            assert A.domain == B.domain
            assert A.codomain == B.domain
            self._solver_name = solver_name
            if solver_params["pc"] is None:
                solver_params.pop("pc")

        # operators
        self._A = A
        self._Apre = Apre
        self._B = B
        self._F = F
        self._tol = tol
        self._max_iter = max_iter
        self._spectralanalysis = spectralanalysis
        self._dimension = dimension
        self._verbose = solver_params["verbose"]

        if self._variant == "Inverse_Solver":
            self._BT = B.transpose()

            # initialize solver with dummy matrix A
            self._block_domainM = BlockVectorSpace(self._A.domain, self._B.transpose().domain)
            self._block_codomainM = self._block_domainM
            self._blocks = [[self._A, self._B.T], [self._B, None]]
            _Minit = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=self._blocks)
            self._solverMinv = inverse(_Minit, solver_name, tol=tol, maxiter=max_iter, **solver_params)

            # Solution vectors
            self._P = B.codomain.zeros()
            self._U = A.codomain.zeros()
            self._Utmp = F.copy() * 0
            # Allocate memory for call
            self._rhstemp = BlockVector(self._block_domainM, blocks=[A.codomain.zeros(), self._B.codomain.zeros()])

        elif self._variant == "Uzawa":
            if self._method_to_solve in ("InexactNPInverse", "SparseSolver"):
                self._preconditioner = False

            self._Anp = self._A[0]
            self._Aenp = self._A[1]
            self._B1np = self._B[0]
            self._B2np = self._B[1]

            # Instanciate inverses
            self._setup_inverses()

            # Solution vectors numpy
            self._Pnp = xp.zeros(self._B1np.shape[0])
            self._Unp = xp.zeros(self._A[0].shape[1])
            self._Uenp = xp.zeros(self._A[1].shape[1])
            # Allocate memory for matrices used in solving the system
            self._rhs0np = self._F[0].copy()
            self._rhs1np = self._F[1].copy()

            # List to store residual norms
            self._residual_norms = []
            self._stepsize = 0.0

    @property
    def A(self):
        """Upper left block."""
        return self._A

    @A.setter
    def A(self, a):
        if self._variant == "Uzawa":
            need_update = True
            A0_old, A1_old = self._A
            A0_new, A1_new = a
            if self._method_to_solve in ("ScipySparse", "SparseSolver"):
                same_A0 = (A0_old != A0_new).nnz == 0
                same_A1 = (A1_old != A1_new).nnz == 0
            else:
                same_A0 = xp.allclose(A0_old, A0_new, atol=1e-10)
                same_A1 = xp.allclose(A1_old, A1_new, atol=1e-10)
            if same_A0 and same_A1:
                need_update = False
            self._A = a
            self._Anp = self._A[0]
            self._Aenp = self._A[1]
            if need_update:
                self._setup_inverses()
        elif self._variant == "Inverse_Solver":
            self._A = a

    @property
    def B(self):
        """Lower left block."""
        return self._B

    @B.setter
    def B(self, b):
        self._B = b

    @property
    def F(self):
        """Right hand side vector."""
        return self._F

    @F.setter
    def F(self, f):
        self._F = f

    @property
    def Apre(self):
        """Preconditioner for upper left block A."""
        return self._Apre

    @Apre.setter
    def Apre(self, a):
        if self._variant == "Uzawa":
            need_update = True
            A0_old, A1_old = self._Apre
            A0_new, A1_new = a
            if self._method_to_solve in ("ScipySparse", "SparseSolver"):
                same_A0 = (A0_old != A0_new).nnz == 0
                same_A1 = (A1_old != A1_new).nnz == 0
            else:
                same_A0 = xp.allclose(A0_old, A0_new, atol=1e-10)
                same_A1 = xp.allclose(A1_old, A1_new, atol=1e-10)
            if same_A0 and same_A1:
                need_update = False
            self._Apre = a
            if need_update:
                self._setup_inverses()
        elif self._variant == "Inverse_Solver":
            self._Apre = a

    def __call__(self, U_init=None, Ue_init=None, P_init=None, out=None):
        """
        Solves the saddle-point problem using the Uzawa algorithm.

        Parameters
        ----------
        U_init : Vector, xp.ndarray or sc.sparse.csr.csr_matrix, optional
            Initial guess for the velocity of the ions. If None, initializes to zero. Types xp.ndarray and sc.sparse.csr.csr_matrix can only be given if system should be solved with Uzawa algorithm.

        Ue_init : Vector, xp.ndarray or sc.sparse.csr.csr_matrix, optional
            Initial guess for the velocity of the electrons. If None, initializes to zero. Types xp.ndarray and sc.sparse.csr.csr_matrix can only be given if system should be solved with Uzawa algorithm.

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
        if self._variant == "Inverse_Solver":
            self._P1 = P_init if P_init is not None else self._P
            self._U1 = U_init if U_init is not None else self._Utmp[0]
            self._U2 = Ue_init if Ue_init is not None else self._Utmp[1]

            _blocksM = [[self._A, self._B.T], [self._B, None]]
            _M = BlockLinearOperator(self._block_domainM, self._block_codomainM, blocks=_blocksM)
            _RHS = BlockVector(self._block_domainM, blocks=[self._F, self._B.codomain.zeros()])

            self._blockU = BlockVector(self._A.domain, blocks=[self._U1, self._U2])
            self._solblocks = [self._blockU, self._P1]
            # comment out the next two lines if working with lifting and GMRES
            x0 = BlockVector(self._block_domainM, blocks=self._solblocks)
            self._solverMinv._options["x0"] = x0

            # use setter to update lhs matrix
            self._solverMinv.linop = _M

            # Initialize P to zero or given initial guess
            self._sol = self._solverMinv.dot(_RHS, out=self._rhstemp)
            self._U = self._sol[0]
            self._P = self._sol[1]

            return self._U, self._P, self._solverMinv._info

        elif self._variant == "Uzawa":
            info = {}

            if self._spectralanalysis:
                self._spectralresult = self._spectral_analysis()
            else:
                self._spectralresult = []

            # Initialize P to zero or given initial guess
            if isinstance(U_init, xp.ndarray) or isinstance(U_init, sc.sparse.csr.csr_matrix):
                self._Pnp = P_init if P_init is not None else self._P
                self._Unp = U_init if U_init is not None else self._U
                self._Uenp = Ue_init if U_init is not None else self._Ue

            else:
                self._Pnp = P_init.toarray() if P_init is not None else self._Pnp
                self._Unp = U_init.toarray() if U_init is not None else self._Unp
                self._Uenp = Ue_init.toarray() if U_init is not None else self._Uenp

            if self._verbose:
                print("Uzawa solver:")
                print("+---------+---------------------+")
                print("+ Iter. # | L2-norm of residual |")
                print("+---------+---------------------+")
                template = "| {:7d} | {:19.2e} |"

            for iteration in range(self._max_iter):
                # Step 1: Compute velocity U by solving A U = -Bᵀ P + F -A Un
                self._rhs0np *= 0
                self._rhs0np -= self._B1np.transpose().dot(self._Pnp)
                self._rhs0np -= self._Anp.dot(self._Unp)
                self._rhs0np += self._F[0]
                if not self._preconditioner:
                    self._Unp += self._Anpinv.dot(self._rhs0np)
                elif self._preconditioner:
                    self._Unp += self._Anpinv.dot(self._A11npinv @ self._rhs0np)

                R1 = self._B1np.dot(self._Unp)

                self._rhs1np *= 0
                self._rhs1np -= self._B2np.transpose().dot(self._Pnp)
                self._rhs1np -= self._Aenp.dot(self._Uenp)
                self._rhs1np += self._F[1]
                if not self._preconditioner:
                    self._Uenp += self._Aenpinv.dot(self._rhs1np)
                elif self._preconditioner:
                    self._Uenp += self._Aenpinv.dot(self._A22npinv @ self._rhs1np)

                R2 = self._B2np.dot(self._Uenp)

                # Step 2: Compute residual R = BU (divergence of U)
                R = R1 + R2  # self._B1np.dot(self._Unp) + self._B2np.dot(self._Uenp)
                residual_norm = xp.linalg.norm(R)
                residual_normR1 = xp.linalg.norm(R)
                self._residual_norms.append(residual_normR1)  # Store residual norm
                # Check for convergence based on residual norm
                if residual_norm < self._tol:
                    if self._verbose:
                        print(template.format(iteration + 1, residual_norm))
                        print("+---------+---------------------+")
                    info["success"] = True
                    info["niter"] = iteration + 1
                    if self._verbose:
                        _plot_residual_norms(self._residual_norms)
                    return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms, self._spectralresult

                # Steepest gradient
                alpha = (R.dot(R)) / (R.dot(self._Precnp.dot(R)))
                # Minimal residual
                # alpha = ((self._Precnp.dot(R)).dot(R)) / ((self._Precnp.dot(R)).dot(self._Precnp.dot(R)))
                self._Pnp += alpha.real * R.real

                if self._verbose:
                    print(template.format(iteration + 1, residual_norm))

            if self._verbose:
                print("+---------+---------------------+")

            # Return with info if maximum iterations reached
            info["success"] = False
            info["niter"] = iteration + 1
            if self._verbose:
                _plot_residual_norms(self._residual_norms)
            return self._Unp, self._Uenp, self._Pnp, info, self._residual_norms, self._spectralresult

    def _setup_inverses(self):
        A0 = self._A[0]
        A1 = self._A[1]

        # === Preconditioner inverses, if used
        if self._preconditioner:
            A11_pre = self._Apre[0]
            A22_pre = self._Apre[1]

            if hasattr(self, "_A11npinv") and self._is_inverse_still_valid(self._A11npinv, A11_pre, "A11 pre"):
                pass
            else:
                self._A11npinv = self._compute_inverse(A11_pre, which="A11 pre")

            if hasattr(self, "_A22npinv") and self._is_inverse_still_valid(self._A22npinv, A22_pre, "A22 pre"):
                pass
            else:
                self._A22npinv = self._compute_inverse(A22_pre, which="A22 pre")

            # === Inverse for A[0] if preconditioned
            if hasattr(self, "_Anpinv") and self._is_inverse_still_valid(self._Anpinv, A0, "A[0]", pre=self._A11npinv):
                pass
            else:
                self._Anpinv = self._compute_inverse(self._A11npinv @ A0, which="A[0]")

            # === Inverse for A[1]
            if hasattr(self, "_Aenpinv") and self._is_inverse_still_valid(
                self._Aenpinv,
                A1,
                "A[1]",
                pre=self._A22npinv,
            ):
                pass
            else:
                self._Aenpinv = self._compute_inverse(self._A22npinv @ A1, which="A[1]")

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
        # try:
        if pre is not None:
            test_mat = pre @ mat
        else:
            test_mat = mat
        I_approx = inv @ test_mat

        if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            I_exact = xp.eye(test_mat.shape[0])
            if not xp.allclose(I_approx, I_exact, atol=1e-6):
                diff = I_approx - I_exact
                max_abs = xp.abs(diff).max()
                print(f"{name} inverse is NOT valid anymore. Max diff: {max_abs:.2e}")
                return False
            print(f"{name} inverse is still valid.")
            return True
        elif self._method_to_solve == "ScipySparse":
            I_exact = sc.sparse.identity(I_approx.shape[0], format=I_approx.format)
            diff = (I_approx - I_exact).tocoo()
            max_abs = xp.abs(diff.data).max() if diff.nnz > 0 else 0.0

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
            return xp.linalg.inv(mat)
        elif self._method_to_solve == "ScipySparse":
            return sc.sparse.linalg.inv(mat)
        elif self._method_to_solve == "SparseSolver":
            solver = SparseSolver(mat)
            return solver.solve(xp.eye(mat.shape[0]))
        else:
            raise ValueError(f"Unknown solver method {self._method_to_solve}")

    def _spectral_analysis(self):
        # Spectral analysis
        # A11 before
        if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            eigvalsA11_before, eigvecs_before = xp.linalg.eig(self._A[0])
            condA11_before = xp.linalg.cond(self._A[0])
        elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
            eigvalsA11_before, eigvecs_before = xp.linalg.eig(self._A[0].toarray())
            condA11_before = xp.linalg.cond(self._A[0].toarray())
        maxbeforeA11 = max(eigvalsA11_before)
        maxbeforeA11_abs = xp.max(xp.abs(eigvalsA11_before))
        minbeforeA11_abs = xp.min(xp.abs(eigvalsA11_before))
        minbeforeA11 = min(eigvalsA11_before)
        specA11_bef = maxbeforeA11 / minbeforeA11
        specA11_bef_abs = maxbeforeA11_abs / minbeforeA11_abs
        # print(f'{maxbeforeA11 = }')
        # print(f'{maxbeforeA11_abs = }')
        # print(f'{minbeforeA11_abs = }')
        # print(f'{minbeforeA11 = }')
        # print(f'{specA11_bef = }')
        print(f"{specA11_bef_abs =}")

        # A22 before
        if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
            eigvalsA22_before, eigvecs_before = xp.linalg.eig(self._A[1])
            condA22_before = xp.linalg.cond(self._A[1])
        elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
            eigvalsA22_before, eigvecs_before = xp.linalg.eig(self._A[1].toarray())
            condA22_before = xp.linalg.cond(self._A[1].toarray())
        maxbeforeA22 = max(eigvalsA22_before)
        maxbeforeA22_abs = xp.max(xp.abs(eigvalsA22_before))
        minbeforeA22_abs = xp.min(xp.abs(eigvalsA22_before))
        minbeforeA22 = min(eigvalsA22_before)
        specA22_bef = maxbeforeA22 / minbeforeA22
        specA22_bef_abs = maxbeforeA22_abs / minbeforeA22_abs
        # print(f'{maxbeforeA22 = }')
        # print(f'{maxbeforeA22_abs = }')
        # print(f'{minbeforeA22_abs = }')
        # print(f'{minbeforeA22 = }')
        # print(f'{specA22_bef = }')
        print(f"{specA22_bef_abs =}")
        print(f"{condA22_before =}")

        if self._preconditioner:
            # A11 after preconditioning with its inverse
            if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                eigvalsA11_after_prec, eigvecs_after = xp.linalg.eig(self._A11npinv @ self._A[0])  # Implement this
            elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                eigvalsA11_after_prec, eigvecs_after = xp.linalg.eig((self._A11npinv @ self._A[0]).toarray())
            maxafterA11_prec = max(eigvalsA11_after_prec)
            minafterA11_prec = min(eigvalsA11_after_prec)
            maxafterA11_abs_prec = xp.max(xp.abs(eigvalsA11_after_prec))
            minafterA11_abs_prec = xp.min(xp.abs(eigvalsA11_after_prec))
            specA11_aft_prec = maxafterA11_prec / minafterA11_prec
            specA11_aft_abs_prec = maxafterA11_abs_prec / minafterA11_abs_prec
            # print(f'{maxafterA11_prec = }')
            # print(f'{maxafterA11_abs_prec = }')
            # print(f'{minafterA11_abs_prec = }')
            # print(f'{minafterA11_prec = }')
            # print(f'{specA11_aft_prec = }')
            print(f"{specA11_aft_abs_prec =}")

            # A22 after preconditioning with its inverse
            if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                eigvalsA22_after_prec, eigvecs_after = xp.linalg.eig(self._A22npinv @ self._A[1])  # Implement this
                condA22_after = xp.linalg.cond(self._A22npinv @ self._A[1])
            elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                eigvalsA22_after_prec, eigvecs_after = xp.linalg.eig((self._A22npinv @ self._A[1]).toarray())
                condA22_after = xp.linalg.cond((self._A22npinv @ self._A[1]).toarray())
            maxafterA22_prec = max(eigvalsA22_after_prec)
            minafterA22_prec = min(eigvalsA22_after_prec)
            maxafterA22_abs_prec = xp.max(xp.abs(eigvalsA22_after_prec))
            minafterA22_abs_prec = xp.min(xp.abs(eigvalsA22_after_prec))
            specA22_aft_prec = maxafterA22_prec / minafterA22_prec
            specA22_aft_abs_prec = maxafterA22_abs_prec / minafterA22_abs_prec
            # print(f'{maxafterA22_prec = }')
            # print(f'{maxafterA22_abs_prec = }')
            # print(f'{minafterA22_abs_prec = }')
            # print(f'{minafterA22_prec = }')
            # print(f'{specA22_aft_prec = }')
            print(f"{specA22_aft_abs_prec =}")

            return condA22_before, specA22_bef_abs, condA11_before, condA22_after, specA22_aft_abs_prec

        else:
            return condA22_before, specA22_bef_abs, condA11_before
