import cunumpy as xp
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.ddm.cart import CartDecomposition, DomainDecomposition
from psydac.fem.tensor import TensorFemSpace
from psydac.linalg.basic import ComposedLinearOperator, LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator
from psydac.linalg.direct_solvers import BandedSolver, SparseSolver
from psydac.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix
from psydac.linalg.stencil import StencilMatrix, StencilVectorSpace
from scipy import sparse
from scipy.linalg import solve_circulant

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator


class MassMatrixPreconditioner(LinearOperator):
    """
    Preconditioner for inverting 3d weighted mass matrices.

    The mass matrix is approximated by a Kronecker product of 1d mass matrices
    in each direction with correct boundary conditions (block diagonal in case of vector-valued spaces).
    In this process, the 3d weight function is appoximated by a 1d counterpart in the dim_reduce direction
    (default 1st direction) at the fixed point (0.5) in the other directions. The inversion is then
    performed with a Kronecker solver.

    Parameters
    ----------
    mass_operator : struphy.feec.mass.WeightedMassOperator
        The weighted mass operator for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.

    dim_reduce : int
        Along which axis to take the approximate value of the weight
    """

    def __init__(self, mass_operator, apply_bc=True, dim_reduce=0):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square mass matrices can be inverted!"

        self._mass_operator = mass_operator
        self._femspace = mass_operator.domain_femspace
        self._space = mass_operator.domain
        self._dtype = mass_operator.dtype
        self._codomain = mass_operator.codomain
        self._domain = mass_operator.domain
        self._apply_bc = apply_bc

        # 3d Kronecker stencil matrices and solvers
        solverblocks = []
        matrixblocks = []

        # collect TensorFemSpaces in a tuple
        if isinstance(self._femspace, TensorFemSpace):
            femspaces = (self._femspace,)
        else:
            femspaces = self._femspace.spaces

        n_comps = len(femspaces)
        n_dims = self._femspace.ldim

        assert n_dims == 3  # other dims not yet implemented
        assert dim_reduce < n_dims

        # get boundary conditions list from BoundaryOperator in ComposedLinearOperator M0 of mass operator
        if apply_bc and isinstance(mass_operator.M0, ComposedLinearOperator):
            if isinstance(mass_operator.M0.multiplicants[-1], BoundaryOperator):
                bc = mass_operator.M0.multiplicants[-1].bc
            else:
                apply_bc = False
                bc = None
        else:
            apply_bc = False
            bc = None

        # loop over components
        for c in range(n_comps):
            # 1d mass matrices and solvers
            solvercells = []
            matrixcells = []

            # loop over spatial directions
            for d in range(n_dims):
                # weight function only along in first direction
                if d == dim_reduce:
                    # pts = [0.5] * (n_dims - 1)
                    loc_weights = mass_operator.weights[c][c]
                    if callable(loc_weights):

                        def fun(e):
                            # make input in meshgrid format to be able to use it with general functions
                            s = e.shape[0]
                            newshape = tuple([1 if i != d else s for i in range(n_dims)])
                            f = e.reshape(newshape)
                            return xp.atleast_1d(
                                loc_weights(
                                    *[xp.array(xp.full_like(f, 0.5)) if i != d else xp.array(f) for i in range(n_dims)],
                                ).squeeze(),
                            )
                    elif isinstance(loc_weights, xp.ndarray):
                        s = loc_weights.shape
                        if d == 0:
                            fun = loc_weights[:, s[1] // 2, s[2] // 2]
                        elif d == 1:
                            fun = loc_weights[s[0] // 2, :, s[2] // 2]
                        elif d == 2:
                            fun = loc_weights[s[0] // 2, s[1] // 2, :]
                    elif loc_weights is None:
                        fun = lambda e: xp.ones(e.size, dtype=float)
                    else:
                        raise TypeError(
                            "weights needs to be callable, xp.ndarray or None but is{}".format(type(loc_weights)),
                        )
                    fun = [[fun]]
                else:
                    fun = [[lambda e: xp.ones(e.size, dtype=float)]]

                # get 1D FEM space (serial, not distributed) and quadrature order
                femspace_1d = femspaces[c].spaces[d]
                qu_order_1d = [mass_operator.derham.nquads[d]]

                # assemble 1d weighted mass matrix
                domain_decompos_1d = DomainDecomposition(
                    [femspace_1d.ncells],
                    [femspace_1d.periodic],
                )
                femspace_1d_tensor = TensorFemSpace(domain_decompos_1d, femspace_1d)
                # femspace_1d_tensor.nquads = [qu_order_1d] # TODO: This should not be here!

                M = WeightedMassOperator(
                    mass_operator.derham,
                    femspace_1d_tensor,
                    femspace_1d_tensor,
                    weights_info=fun,
                    nquads=qu_order_1d,
                )
                M.assemble(verbose=False)
                M = M.matrix

                # apply boundary conditions
                if apply_bc:
                    if mass_operator._domain_symbolic_name not in ("H1H1H1", "H1vec"):
                        if femspace_1d.basis == "B":
                            if bc[d][0]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=-1,
                                    order=0,
                                    identity=True,
                                )
                            if bc[d][1]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=+1,
                                    order=0,
                                    identity=True,
                                )
                    else:
                        if c == d:
                            if bc[d][0]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=-1,
                                    order=0,
                                    identity=True,
                                )
                            if bc[d][1]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=+1,
                                    order=0,
                                    identity=True,
                                )

                M_arr = M.toarray()

                # create 1d solver for mass matrix
                if is_circulant(M_arr):
                    solvercells += [FFTSolver(M_arr)]
                else:
                    solvercells += [SparseSolver(M.tosparse())]

                # === NOTE: for KroneckerStencilMatrix being built correctly, 1d matrices must be local to process! ===
                periodic = femspaces[c].coeff_space.periods[d]

                n = femspaces[c].coeff_space.npts[d]
                p = femspaces[c].coeff_space.pads[d]
                s = femspaces[c].coeff_space.starts[d]
                e = femspaces[c].coeff_space.ends[d]

                cart_decomp_1d = CartDecomposition(
                    domain_decompos_1d,
                    [n],
                    [[s]],
                    [[e]],
                    [p],
                    [1],
                )

                V_local = StencilVectorSpace(cart_decomp_1d)

                M_local = StencilMatrix(V_local, V_local)

                row_indices, col_indices = xp.nonzero(M_arr)

                for row_i, col_i in zip(row_indices, col_indices):
                    # only consider row indices on process
                    if row_i in range(V_local.starts[0], V_local.ends[0] + 1):
                        row_i_loc = row_i - s

                        M_local._data[
                            row_i_loc + p,
                            (col_i + p - row_i) % M_arr.shape[1],
                        ] = M_arr[row_i, col_i]

                # check if stencil matrix was built correctly
                assert xp.allclose(M_local.toarray()[s : e + 1], M_arr[s : e + 1])

                matrixcells += [M_local.copy()]
                # =======================================================================================================

            if isinstance(self._femspace, TensorFemSpace):
                matrixblocks += [
                    KroneckerStencilMatrix(
                        self._femspace.coeff_space,
                        self._femspace.coeff_space,
                        *matrixcells,
                    ),
                ]
                solverblocks += [
                    KroneckerLinearSolver(
                        self._femspace.coeff_space,
                        self._femspace.coeff_space,
                        solvercells,
                    ),
                ]
            else:
                matrixblocks += [
                    KroneckerStencilMatrix(
                        self._femspace.coeff_space[c],
                        self._femspace.coeff_space[c],
                        *matrixcells,
                    ),
                ]
                solverblocks += [
                    KroneckerLinearSolver(
                        self._femspace.coeff_space[c],
                        self._femspace.coeff_space[c],
                        solvercells,
                    ),
                ]

        # build final matrix and solver
        if isinstance(self._femspace, TensorFemSpace):
            self._matrix = matrixblocks[0]
            self._solver = solverblocks[0]
        else:
            blocks = [
                [matrixblocks[0], None, None],
                [None, matrixblocks[1], None],
                [None, None, matrixblocks[2]],
            ]

            self._matrix = BlockLinearOperator(
                self._femspace.coeff_space,
                self._femspace.coeff_space,
                blocks=blocks,
            )

            sblocks = [
                [solverblocks[0], None, None],
                [None, solverblocks[1], None],
                [None, None, solverblocks[2]],
            ]

            self._solver = BlockLinearOperator(
                self._femspace.coeff_space,
                self._femspace.coeff_space,
                blocks=sblocks,
            )

        # save mass operator to be inverted (needed in solve method)
        if apply_bc:
            self._M = mass_operator.M0
        else:
            self._M = mass_operator.M

        self._is_composed = isinstance(self._M, ComposedLinearOperator)

        # temporary vectors for dot product
        if self._is_composed:
            tmp_vectors = []
            for op in self._M.multiplicants[1:]:
                tmp_vectors.append(op.codomain.zeros())

            self._tmp_vectors = tuple(tmp_vectors)
        else:
            self._tmp_vector = self._M.codomain.zeros()

    @property
    def space(self):
        """Stencil-/BlockVectorSpace or PolarDerhamSpace."""
        return self._space

    @property
    def matrix(self):
        """Approximation of the input mass matrix as KroneckerStencilMatrix."""
        return self._matrix

    @property
    def solver(self):
        """KroneckerLinearSolver or BlockDiagonalSolver for exactly inverting the approximate mass matrix self.matrix."""
        return self._solver

    @property
    def domain(self):
        """The domain of the linear operator - an element of Vectorspace"""
        return self._space

    @property
    def codomain(self):
        """The codomain of the linear operator - an element of Vectorspace"""
        return self._codomain

    @property
    def domain(self):
        """The domain of the linear operator - an element of Vectorspace"""
        return self._domain

    @property
    def dtype(self):
        return self._dtype

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return MassMatrixPreconditioner(self._mass_operator.transpose(), self._apply_bc)

    def solve(self, rhs, out=None):
        """
        Computes (B * E * M^(-1) * E^T * B^T) * rhs as an approximation for an inverse mass matrix.

        Parameters
        ----------
        rhs : psydac.linalg.basic.Vector
            The right-hand side vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The result of (B * E * M^(-1) * E^T * B^T) * rhs.
        """

        assert isinstance(rhs, Vector)
        assert rhs.space == self._space

        # successive dot products with all but last operator
        if self._is_composed:
            x = rhs
            for i in range(len(self._tmp_vectors)):
                y = self._tmp_vectors[-1 - i]
                A = self._M.multiplicants[-1 - i]
                if isinstance(A, (StencilMatrix, BlockLinearOperator)):
                    self.solver.dot(x, out=y)
                else:
                    A.dot(x, out=y)
                x = y

            # last operator
            A = self._M.multiplicants[0]
            if out is None:
                out = A.dot(x)
            else:
                assert isinstance(out, Vector)
                assert out.space == self._space
                A.dot(x, out=out)

        else:
            if out is None:
                out = self._tmp_vector.copy()
            self.solver.dot(rhs, out=out)

        return out

    def dot(self, v, out=None):
        """Apply linear operator to Vector v. Result is written to Vector out, if provided."""

        assert isinstance(v, Vector)
        assert v.space == self.domain

        # newly created output vector
        if out is None:
            out = self.solve(v)

        # in-place dot-product (result is written to out)
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            self.solve(v, out=out)

        return out


class MassMatrixDiagonalPreconditioner(LinearOperator):
    r"""
    Preconditioner for inverting 3d weighted mass matrices. The mass matrix is approximated by

    .. math::
        D^{1/2} * \hat D^{-1/2} * \hat M * \hat D^{-1/2} * D^{1/2}

    Where $D$ is the diagonal of the matrix to invert, :math:`\hat M` is the mass matrix on the logical domain
    that is a Kronecker product (fastly inverted) and :math:`\hat D^{-1/2}` is the diagonal of :math:`\hat M`.

    Notes
    -----

    Reference: `G. Loli, G. Sangalli, M. Tani, "Easy and efficient preconditioning of the isogeometric mass matrix", Comp. Math. Appl., Vol. 116, 2022 <https://www.sciencedirect.com/science/article/pii/S0898122120304715?via%3Dihub>`_

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        The weighted mass operator for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.
    """

    def __init__(self, mass_operator, apply_bc=True):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square mass matrices can be inverted!"

        self._mass_operator = mass_operator
        self._femspace = mass_operator.domain_femspace
        self._space = mass_operator.domain
        self._dtype = mass_operator.dtype
        self._codomain = mass_operator.codomain
        self._domain = mass_operator.domain
        self._apply_bc = apply_bc

        # 3d Kronecker stencil matrices and solvers
        solverblocks = []
        matrixblocks = []

        # collect TensorFemSpaces in a tuple
        if isinstance(self._femspace, TensorFemSpace):
            femspaces = (self._femspace,)
        else:
            femspaces = self._femspace.spaces

        n_comps = len(femspaces)
        n_dims = self._femspace.ldim

        assert n_dims == 3  # other dims not yet implemented

        # get boundary conditions list from BoundaryOperator in ComposedLinearOperator M0 of mass operator
        if apply_bc and isinstance(mass_operator.M0, ComposedLinearOperator):
            if isinstance(mass_operator.M0.multiplicants[-1], BoundaryOperator):
                bc = mass_operator.M0.multiplicants[-1].bc
            else:
                apply_bc = False
                bc = None
        else:
            apply_bc = False
            bc = None

        # loop over components
        for c in range(n_comps):
            # 1d mass matrices and solvers
            solvercells = []
            matrixcells = []

            # loop over spatial directions
            for d in range(n_dims):
                fun = [[lambda e: xp.ones(e.size, dtype=float)]]

                # get 1D FEM space (serial, not distributed) and quadrature order
                femspace_1d = femspaces[c].spaces[d]
                qu_order_1d = [self._mass_operator.derham.nquads[d]]
                # assemble 1d weighted mass matrix
                domain_decompos_1d = DomainDecomposition(
                    [femspace_1d.ncells],
                    [femspace_1d.periodic],
                )
                femspace_1d_tensor = TensorFemSpace(domain_decompos_1d, femspace_1d)
                # femspace_1d_tensor.nquads = [qu_order_1d]
                # femspace_1d_tensor.nquads = self._mass_operator.derham.nquads

                M = WeightedMassOperator(
                    self._mass_operator.derham,
                    femspace_1d_tensor,
                    femspace_1d_tensor,
                    weights_info=fun,
                    nquads=qu_order_1d,
                )
                M.assemble(verbose=False)
                M = M.matrix

                # apply boundary conditions
                if apply_bc:
                    if mass_operator._domain_symbolic_name not in ("H1H1H1", "H1vec"):
                        if femspace_1d.basis == "B":
                            if bc[d][0]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=-1,
                                    order=0,
                                    identity=True,
                                )
                            if bc[d][1]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=+1,
                                    order=0,
                                    identity=True,
                                )
                    else:
                        if c == d:
                            if bc[d][0]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=-1,
                                    order=0,
                                    identity=True,
                                )
                            if bc[d][1]:
                                apply_essential_bc_stencil(
                                    M,
                                    axis=0,
                                    ext=+1,
                                    order=0,
                                    identity=True,
                                )

                M_arr = M.toarray()

                # create 1d solver for mass matrix
                if is_circulant(M_arr):
                    solvercells += [FFTSolver(M_arr)]
                else:
                    solvercells += [SparseSolver(M.tosparse())]

                # === NOTE: for KroneckerStencilMatrix being built correctly, 1d matrices must be local to process! ===
                periodic = femspaces[c].coeff_space.periods[d]

                n = femspaces[c].coeff_space.npts[d]
                p = femspaces[c].coeff_space.pads[d]
                s = femspaces[c].coeff_space.starts[d]
                e = femspaces[c].coeff_space.ends[d]

                cart_decomp_1d = CartDecomposition(
                    domain_decompos_1d,
                    [n],
                    [[s]],
                    [[e]],
                    [p],
                    [1],
                )

                V_local = StencilVectorSpace(cart_decomp_1d)

                M_local = StencilMatrix(V_local, V_local)

                row_indices, col_indices = xp.nonzero(M_arr)

                for row_i, col_i in zip(row_indices, col_indices):
                    # only consider row indices on process
                    if row_i in range(V_local.starts[0], V_local.ends[0] + 1):
                        row_i_loc = row_i - s

                        M_local._data[
                            row_i_loc + p,
                            (col_i + p - row_i) % M_arr.shape[1],
                        ] = M_arr[row_i, col_i]

                # check if stencil matrix was built correctly
                assert xp.allclose(M_local.toarray()[s : e + 1], M_arr[s : e + 1])

                matrixcells += [M_local.copy()]
                # =======================================================================================================

            if isinstance(self._femspace, TensorFemSpace):
                matrixblocks += [
                    KroneckerStencilMatrix(
                        self._femspace.coeff_space,
                        self._femspace.coeff_space,
                        *matrixcells,
                    ),
                ]
                solverblocks += [
                    KroneckerLinearSolver(
                        self._femspace.coeff_space,
                        self._femspace.coeff_space,
                        solvercells,
                    ),
                ]
            else:
                matrixblocks += [
                    KroneckerStencilMatrix(
                        self._femspace.coeff_space[c],
                        self._femspace.coeff_space[c],
                        *matrixcells,
                    ),
                ]
                solverblocks += [
                    KroneckerLinearSolver(
                        self._femspace.coeff_space[c],
                        self._femspace.coeff_space[c],
                        solvercells,
                    ),
                ]

        # build final matrix and solver
        if isinstance(self._femspace, TensorFemSpace):
            self._matrix = matrixblocks[0]
            self._solver = solverblocks[0]
        else:
            blocks = [
                [matrixblocks[0], None, None],
                [None, matrixblocks[1], None],
                [None, None, matrixblocks[2]],
            ]

            self._matrix = BlockLinearOperator(
                self._femspace.coeff_space,
                self._femspace.coeff_space,
                blocks=blocks,
            )

            sblocks = [
                [solverblocks[0], None, None],
                [None, solverblocks[1], None],
                [None, None, solverblocks[2]],
            ]
            self._solver = BlockLinearOperator(
                self._femspace.coeff_space,
                self._femspace.coeff_space,
                blocks=sblocks,
            )

        # save mass operator to be inverted (needed in solve method)
        if apply_bc:
            self._M = mass_operator.M0
        else:
            self._M = mass_operator.M

        self._is_composed = isinstance(self._M, ComposedLinearOperator)

        # temporary vectors for dot product
        if self._is_composed:
            tmp_vectors = []
            for op in self._M.multiplicants[1:]:
                tmp_vectors.append(op.codomain.zeros())

            self._tmp_vectors = tuple(tmp_vectors)
        else:
            self._tmp_vector = self._M.codomain.zeros()

        # Need to assemble the logical mass matrix to extract the coefficients
        fun = [
            [lambda e1, e2, e3: xp.ones_like(e1, dtype=float) if i == j else None for j in range(3)] for i in range(3)
        ]
        log_M = WeightedMassOperator(
            self._mass_operator.derham,
            self._femspace,
            self._femspace,
            weights_info=fun,
        )
        log_M.assemble(verbose=False)
        self._logM_srqt_diag = log_M.matrix.diagonal(sqrt=True)
        self._M_invsrqt_diag = self._mass_operator.matrix.diagonal(inverse=True, sqrt=True)

        self._tmp_vector_no_bc = [self._mass_operator.matrix.codomain.zeros() for i in range(2)]

    @property
    def space(self):
        """Stencil-/BlockVectorSpace or PolarDerhamSpace."""
        return self._space

    @property
    def matrix(self):
        """Mass matrix on the logical domain as KroneckerStencilMatrix."""
        return self._matrix

    @property
    def solver(self):
        """KroneckerLinearSolver or BlockDiagonalSolver for exactly inverting the approximate mass matrix self.matrix."""
        return self._solver

    @property
    def domain(self):
        """The domain of the linear operator - an element of Vectorspace"""
        return self._space

    @property
    def codomain(self):
        """The codomain of the linear operator - an element of Vectorspace"""
        return self._codomain

    @property
    def domain(self):
        """The domain of the linear operator - an element of Vectorspace"""
        return self._domain

    @property
    def dtype(self):
        return self._dtype

    def update_mass_operator(self, mass_operator):
        """Update the mass operator to enable recycling the preconditioner"""
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square mass matrices can be inverted!"
        assert mass_operator.domain == self.domain, "Update needs to have the same domain and codomain"

        if self._is_composed:
            if self._apply_bc:
                assert isinstance(mass_operator.M0, ComposedLinearOperator)
            else:
                assert isinstance(mass_operator.M, ComposedLinearOperator)

        self._mass_operator = mass_operator

        if self._apply_bc:
            self._M = mass_operator.M0
        else:
            self._M = mass_operator.M
        self._M_invsrqt_diag = self._mass_operator.matrix.diagonal(inverse=True, sqrt=True, out=self._M_invsrqt_diag)

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return MassMatrixPreconditioner(self._mass_operator.transpose(), self._apply_bc)

    def _solve_no_bc(self, rhs, out):
        r"""
        Computes M^(-1) * rhs as an approximation for an inverse mass matrix.
        With $M = D^{1/2} * \hat D^{-1/2} * \hat M * \hat D^{-1/2} * D^{1/2}$
        Should only be called by the solve method that will handle the bcs.

        Parameters
        ----------
        rhs : psydac.linalg.basic.Vector
            The right-hand side vector.

        out : psydac.linalg.basic.Vector
            The output vector will be written into this vector in-place.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The result of M^(-1) * rhs.
        """

        assert isinstance(rhs, Vector)
        assert rhs.space == self._mass_operator.matrix.domain

        # M^-1 ~ D^{-1/2} \hat D^{1/2} \hat M ^{-1} \hat D^{1/2} D^{-1/2}
        Dmr = self._M_invsrqt_diag.dot(rhs, out=self._tmp_vector_no_bc[0])
        DhDmr = self._logM_srqt_diag.dot(Dmr, out=self._tmp_vector_no_bc[1])
        invMr = self.solver.dot(DhDmr, out=self._tmp_vector_no_bc[0])
        DhiMr = self._logM_srqt_diag.dot(invMr, out=self._tmp_vector_no_bc[1])
        out = self._M_invsrqt_diag.dot(DhiMr, out=out)

        return out

    def solve(self, rhs, out=None):
        r"""
        Computes :math:`(B * E * M^{-1} * E^T * B^T) * rhs` as an approximation for an inverse mass matrix,
        with :math:`M = D^{1/2} * \hat D^{-1/2} * \hat M * \hat D^{-1/2} * D^{1/2}`.

        Parameters
        ----------
        rhs : Vector
            The right-hand side vector.

        out : Vector, optional
            If given, the output vector will be written into this vector in-place.

        Returns
        -------
        out : Vector
            The result of :math:`(B * E * M^{-1} * E^T * B^T) * rhs`.
        """

        assert isinstance(rhs, Vector)
        assert rhs.space == self._space

        # successive dot products with all but last operator
        if self._is_composed:
            x = rhs
            for i in range(len(self._tmp_vectors)):
                y = self._tmp_vectors[-1 - i]
                A = self._M.multiplicants[-1 - i]
                if isinstance(A, (StencilMatrix, BlockLinearOperator)):
                    self._solve_no_bc(x, out=y)
                else:
                    A.dot(x, out=y)
                x = y

            # last operator
            A = self._M.multiplicants[0]
            if out is None:
                out = A.dot(x)
            else:
                assert isinstance(out, Vector)
                assert out.space == self._space
                A.dot(x, out=out)

        else:
            if out is None:
                out = self._tmp_vector.copy()
            self._solve_no_bc(rhs, out=out)

        return out

    def dot(self, v, out=None):
        """Apply linear operator to Vector v. Result is written to Vector out, if provided."""

        assert isinstance(v, Vector)
        assert v.space == self.domain

        # newly created output vector
        if out is None:
            out = self.solve(v)

        # in-place dot-product (result is written to out)
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            self.solve(v, out=out)

        return out


class FFTSolver(BandedSolver):
    """
    Solve the equation Ax = b for x, assuming A is a circulant matrix.
    b can contain multiple right-hand sides (RHS) and is of shape (#RHS, N).

    Parameters
    ----------
    circmat : xp.ndarray
        Generic circulant matrix.
    """

    def __init__(self, circmat):
        assert isinstance(circmat, xp.ndarray)
        assert is_circulant(circmat)

        self._space = xp.ndarray
        self._column = circmat[:, 0]

    # --------------------------------------
    # Abstract interface
    # --------------------------------------
    @property
    def space(self):
        return self._space

    # ...
    def solve(self, rhs, out=None, transposed=False):
        """
        Solves for the given right-hand side.

        Parameters
        ----------
        rhs : xp.ndarray
            The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order,
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside
            a right-hand side.

        out : xp.ndarray, optional
            Output vector. If given, it has to have the same shape and datatype as rhs.

        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """

        assert rhs.T.shape[0] == self._column.size

        if out is None:
            out = solve_circulant(self._column, rhs.T).T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            try:
                out[:] = solve_circulant(self._column, rhs.T).T
            except xp.linalg.LinAlgError:
                eps = 1e-4
                print(f"Stabilizing singular preconditioning FFTSolver with {eps =}:")
                self._column[0] *= 1.0 + eps
                out[:] = solve_circulant(self._column, rhs.T).T

        return out


def is_circulant(mat):
    """
    Returns true if a matrix is circulant.

    Parameters
    ----------
    mat : array[float]
        The matrix that is checked to be circulant.

    Returns
    -------
    circulant : bool
        Whether the matrix is circulant (=True) or not (=False).
    """

    assert isinstance(mat, xp.ndarray)
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]

    if mat.shape[0] > 1:
        for i in range(mat.shape[0] - 1):
            circulant = xp.allclose(mat[i, :], xp.roll(mat[i + 1, :], -1))
            if not circulant:
                return circulant
    else:
        circulant = True

    return circulant
