import logging

import cunumpy as xp
import numpy as np
from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import ComposedLinearOperator, LinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator
from feectools.linalg.direct_solvers import BandedSolver, SparseSolver
from feectools.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix
from feectools.linalg.stencil import StencilDiagonalMatrix, StencilMatrix, StencilVectorSpace
from line_profiler import profile
from scipy import sparse

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator

logger = logging.getLogger("struphy")

_diag_apply_kernel = None


def _apply_squared_diagonal_gpu(diagonal, rhs, out, interior):
    """Apply ``diagonal**2 * rhs`` in one CuPy elementwise kernel."""
    global _diag_apply_kernel
    if _diag_apply_kernel is None:
        import cupy as cp

        _diag_apply_kernel = cp.ElementwiseKernel(
            "T d, T x",
            "T y",
            "y = d * d * x",
            "struphy_apply_squared_diagonal",
        )
    _diag_apply_kernel(diagonal, rhs[interior], out[interior])


def _apply_squared_diagonal_operator_gpu(diagonal, rhs, out):
    """Apply a scalar or block diagonal operator with one kernel per block."""
    if isinstance(diagonal, StencilDiagonalMatrix):
        V = rhs.space
        interior = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
        # Index through StencilVector so global starts/ends are translated to
        # the local padded storage correctly.
        _apply_squared_diagonal_gpu(diagonal._data, rhs, out, interior)
        out.ghost_regions_in_sync = False
        return

    if isinstance(diagonal, BlockLinearOperator):
        for component, (rhs_block, out_block) in enumerate(zip(rhs.blocks, out.blocks)):
            _apply_squared_diagonal_operator_gpu(
                diagonal[component, component], rhs_block, out_block
            )
        out.ghost_regions_in_sync = False
        return

    raise TypeError(f"Unsupported GPU diagonal operator {type(diagonal)}.")


def _iter_diagonal_blocks(operator):
    """Yield StencilDiagonalMatrix blocks from a scalar or block diagonal."""

    if isinstance(operator, StencilDiagonalMatrix):
        yield operator
        return

    if isinstance(operator, BlockLinearOperator):
        for component in range(3):
            block = operator[component, component]

            if not isinstance(block, StencilDiagonalMatrix):
                raise TypeError(f"Expected StencilDiagonalMatrix in diagonal block {component}, got {type(block)}.")

            yield block
        return

    raise TypeError(f"Expected StencilDiagonalMatrix or BlockLinearOperator, got {type(operator)}.")


def _combine_diagonals(left, right, out, *, alpha=1.0):
    """Set out = left + alpha * right."""

    left_blocks = tuple(_iter_diagonal_blocks(left))
    right_blocks = tuple(_iter_diagonal_blocks(right))
    out_blocks = tuple(_iter_diagonal_blocks(out))

    if not (len(left_blocks) == len(right_blocks) == len(out_blocks)):
        raise ValueError("Incompatible diagonal block structures.")

    for left_block, right_block, out_block in zip(
        left_blocks,
        right_blocks,
        out_blocks,
    ):
        out_block._data[:] = left_block._data
        out_block._data[:] += alpha * right_block._data


def _inverse_sqrt_diagonal_inplace(operator):
    """Replace every diagonal entry d by 1/sqrt(d)."""

    for block in _iter_diagonal_blocks(operator):
        if bool(xp.any(block._data <= 0.0)):
            raise ValueError("The kinetic-metric diagonal must be strictly positive.")

        xp.sqrt(block._data, out=block._data)
        xp.divide(1.0, block._data, out=block._data)


def _multiply_diagonals(left, right, out):
    """Set out = left * right pointwise."""

    left_blocks = tuple(_iter_diagonal_blocks(left))
    right_blocks = tuple(_iter_diagonal_blocks(right))
    out_blocks = tuple(_iter_diagonal_blocks(out))

    if not (len(left_blocks) == len(right_blocks) == len(out_blocks)):
        raise ValueError("Incompatible diagonal block structures.")

    for left_block, right_block, out_block in zip(
        left_blocks,
        right_blocks,
        out_blocks,
    ):
        xp.multiply(
            left_block._data,
            right_block._data,
            out=out_block._data,
        )


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

        # define subcomm to gather 1d weight info along dim_reduce
        derham = mass_operator.derham
        logger.debug(f"{derham.num_elements = }, {derham.bcs = }, {derham.degree = }")
        comm = derham.comm
        if not isinstance(comm, (MockComm, type(None))):
            rank = comm.Get_rank()
            dom_arr = derham.domain_array
            selected_ranks = []
            left = 0.0
            right = 1.0
            for i, arr in enumerate(dom_arr):
                left_i = arr[3 * dim_reduce]
                right_i = arr[3 * dim_reduce + 1]
                if left_i != left or right_i != right:
                    selected_ranks.append(i)
                    left = left_i
                    right = right_i

            logger.debug(f"Selected ranks for gathering 1d weight info in dimension {dim_reduce}: {selected_ranks}")
            logger.debug(f"{dom_arr = }")
            color = 0 if rank in selected_ranks else MPI.UNDEFINED
            subcomm = comm.Split(color=color, key=rank)

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
                        logger.debug(f"{loc_weights.shape = } for component {c} and direction {d}.")
                        npts = derham.num_elements[d] * derham.nquads[d]
                        fun = xp.zeros(npts, dtype=float)
                        if d == 0:
                            local_fun = loc_weights[:, s[1] // 2, s[2] // 2]
                        elif d == 1:
                            local_fun = loc_weights[s[0] // 2, :, s[2] // 2]
                        elif d == 2:
                            local_fun = loc_weights[s[0] // 2, s[1] // 2, :]
                        local_fun = xp.ascontiguousarray(local_fun)
                        logger.debug(
                            f"{fun.size = } for component {c} and direction {d} before gathering on all processes."
                        )
                        if (
                            local_fun.size < npts
                        ):  # this branch is only entered if comm exists (and thus subcomm has been initialized)
                            if subcomm != MPI.COMM_NULL:
                                subcomm.Allgather(local_fun, fun)
                                """gathered = subcomm.gather(local_fun, root=selected_ranks[0])
                                if rank == selected_ranks[0]:
                                    if gathered is None:
                                        raise RuntimeError("MPI gather failed to return data on root rank")
                                    fun[:] = xp.concatenate(gathered)
                                    assert fun.size == npts, (
                                        f"Gathered weight size {fun.size} does not match expected {npts}"
                                    )"""
                            comm.Bcast(fun, root=selected_ranks[0])
                        else:
                            fun[:] = local_fun
                        logger.debug(
                            f"{fun.shape = } for component {c} and direction {d} after gathering on all processes."
                        )
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
                if femspaces[c].spaces[d].basis == "B":
                    femspace_1d_tensor = mass_operator.derham.H1_1d_serial[d]
                else:
                    femspace_1d_tensor = mass_operator.derham.L2_1d_serial[d]

                domain_decompos_1d = femspace_1d_tensor.domain_decomposition
                qu_order_1d = (mass_operator.derham.nquads[d],)

                M = WeightedMassOperator(
                    mass_operator.derham,
                    femspace_1d_tensor,
                    femspace_1d_tensor,
                    weights_info=fun,
                    nquads=qu_order_1d,
                )
                M.assemble()
                M = M.matrix

                # apply boundary conditions
                if apply_bc:
                    if mass_operator._domain_symbolic_name not in ("H1H1H1", "H1vec"):
                        if femspaces[c].spaces[d].basis == "B":
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

                row_indices, col_indices = np.nonzero(M_arr)  # M_arr is always a host array (StencilMatrix.toarray())

                for row_i, col_i in zip(row_indices, col_indices):
                    # only consider row indices on process
                    if row_i in range(V_local.starts[0], V_local.ends[0] + 1):
                        row_i_loc = row_i - s

                        M_local._data[
                            row_i_loc + p,
                            (col_i + p - row_i) % M_arr.shape[1],
                        ] = M_arr[row_i, col_i]

                # check if stencil matrix was built correctly
                assert np.allclose(M_local.toarray()[s : e + 1], M_arr[s : e + 1])  # both sides are host arrays

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

    @profile
    def solve(self, rhs, out=None):
        """
        Computes (B * E * M^(-1) * E^T * B^T) * rhs as an approximation for an inverse mass matrix.

        Parameters
        ----------
        rhs : feectools.linalg.basic.Vector
            The right-hand side vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.

        Returns
        -------
        out : feectools.linalg.basic.Vector
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
                if femspaces[c].spaces[d].basis == "B":
                    femspace_1d_tensor = mass_operator.derham.H1_1d_serial[d]
                else:
                    femspace_1d_tensor = mass_operator.derham.L2_1d_serial[d]

                domain_decompos_1d = femspace_1d_tensor.domain_decomposition
                qu_order_1d = (mass_operator.derham.nquads[d],)

                M = WeightedMassOperator(
                    self._mass_operator.derham,
                    femspace_1d_tensor,
                    femspace_1d_tensor,
                    weights_info=fun,
                    nquads=qu_order_1d,
                )
                M.assemble()
                M = M.matrix

                # apply boundary conditions
                if apply_bc:
                    if mass_operator._domain_symbolic_name not in ("H1H1H1", "H1vec"):
                        if femspaces[c].spaces[d].basis == "B":
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

                row_indices, col_indices = np.nonzero(M_arr)  # M_arr is always a host array (StencilMatrix.toarray())

                for row_i, col_i in zip(row_indices, col_indices):
                    # only consider row indices on process
                    if row_i in range(V_local.starts[0], V_local.ends[0] + 1):
                        row_i_loc = row_i - s

                        M_local._data[
                            row_i_loc + p,
                            (col_i + p - row_i) % M_arr.shape[1],
                        ] = M_arr[row_i, col_i]

                # check if stencil matrix was built correctly
                assert np.allclose(M_local.toarray()[s : e + 1], M_arr[s : e + 1])  # both sides are host arrays

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
        log_M.assemble()
        self._logM_srqt_diag = log_M.matrix.diagonal(sqrt=True)
        self._M_invsrqt_diag = self._mass_operator.matrix.diagonal(inverse=True, sqrt=True)
        self._scaled_diag = self._mass_operator.matrix.diagonal()
        self._update_scaled_diagonal()

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
    def dtype(self):
        return self._dtype

    def update_mass_operator(self, mass_operator):
        """Update the mass operator while recycling the preconditioner."""

        self._set_mass_operator_references(mass_operator)

        self._mass_operator.matrix.diagonal(
            inverse=True,
            sqrt=True,
            out=self._M_invsrqt_diag,
        )

        self._update_scaled_diagonal()

    def _update_scaled_diagonal(self):
        """
        Update

            S = Dhat^(1/2) D^(-1/2).
        """
        _multiply_diagonals(
            self._logM_srqt_diag,
            self._M_invsrqt_diag,
            self._scaled_diag,
        )

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return self

    @profile
    def _solve_no_bc(self, rhs, out):
        r"""
        Apply

            S * Mhat^(-1) * S,

        where S = Dhat^(1/2) D^(-1/2).
        """
        assert isinstance(rhs, Vector)
        assert rhs.space == self._mass_operator.matrix.domain

        scaled_rhs = self._scaled_diag.dot(
            rhs,
            out=self._tmp_vector_no_bc[0],
        )

        solved = self.solver.dot(
            scaled_rhs,
            out=self._tmp_vector_no_bc[1],
        )

        self._scaled_diag.dot(
            solved,
            out=out,
        )

        return out

    def _solve_diagonal_no_bc(self, rhs, out):
        r"""
        Apply the inverse mass diagonal

            D_M^{-1} rhs

        without applying the logical Kronecker mass solve.

        This is substantially cheaper than `_solve_no_bc` and is intended
        for repeated applications inside auxiliary preconditioners.
        """
        assert isinstance(rhs, Vector)
        assert rhs.space == self._mass_operator.matrix.domain

        # _M_invsrqt_diag represents D_M^{-1/2}.  On CuPy, combine the two
        # diagonal applications into one elementwise kernel (D^-1 rhs).
        first_data = rhs.blocks[0]._data if hasattr(rhs, "blocks") else rhs._data
        if xp.is_gpu(first_data) and isinstance(
            self._M_invsrqt_diag, (StencilDiagonalMatrix, BlockLinearOperator)
        ):
            _apply_squared_diagonal_operator_gpu(self._M_invsrqt_diag, rhs, out)
            return out

        tmp = self._M_invsrqt_diag.dot(
            rhs,
            out=self._tmp_vector_no_bc[0],
        )

        self._M_invsrqt_diag.dot(
            tmp,
            out=out,
        )

        return out

    @profile
    def _solve_with_core(self, rhs, out, core_solver):
        """
        Apply a supplied tensor-space inverse while preserving the existing
        boundary/extraction handling.

        Parameters
        ----------
        rhs : Vector
            Input vector in the public operator domain.

        out : Vector or None
            Output vector.

        core_solver : callable
            Function with signature core_solver(rhs, out).
        """
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space

        if self._is_composed:
            x = rhs

            for i in range(len(self._tmp_vectors)):
                y = self._tmp_vectors[-1 - i]
                A = self._M.multiplicants[-1 - i]

                if isinstance(
                    A,
                    (StencilMatrix, BlockLinearOperator),
                ):
                    core_solver(x, out=y)
                else:
                    A.dot(x, out=y)

                x = y

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

            core_solver(rhs, out=out)

        return out

    @profile
    def solve(self, rhs, out=None):
        """
        Apply the full diagonal-scaled Kronecker mass preconditioner.
        """
        return self._solve_with_core(
            rhs,
            out,
            self._solve_no_bc,
        )

    def solve_diagonal(self, rhs, out=None):
        r"""
        Apply the cheap mass Jacobi inverse

            D_M^{-1}

        with the same extraction and boundary handling as `solve`.
        """
        return self._solve_with_core(
            rhs,
            out,
            self._solve_diagonal_no_bc,
        )

    def _set_mass_operator_references(self, mass_operator):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain
        assert mass_operator.domain == self.domain

        if self._is_composed:
            if self._apply_bc:
                assert isinstance(
                    mass_operator.M0,
                    ComposedLinearOperator,
                )
            else:
                assert isinstance(
                    mass_operator.M,
                    ComposedLinearOperator,
                )

        self._mass_operator = mass_operator

        if self._apply_bc:
            self._M = mass_operator.M0
        else:
            self._M = mass_operator.M

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


class H1vecKineticMetricPreconditioner(
    MassMatrixDiagonalPreconditioner,
):
    r"""
    Preconditioner for the regularized H1vec kinetic metric

    .. math::

        A_\rho = M_\rho + \beta K_\rho,

    where

    .. math::

        v^\top K_\rho u
        =
        \int_\Omega
        \rho\,
        \operatorname{div}(v)\,
        \operatorname{div}(u)
        \,\mathrm dx.

    The coefficient ``beta`` is ``kinetic_metric.alpha``. In
    ``VariationalDensityEvolve`` this is already

    .. math::

        \beta = 2\alpha_{\mathrm{divdiv}}.

    The approximation is

    .. math::

        P^{-1}
        =
        D_A^{-1/2}
        \widehat D^{1/2}
        \widehat M^{-1}
        \widehat D^{1/2}
        D_A^{-1/2},

    where

    .. math::

        D_A = \operatorname{diag}(M_\rho + \beta K_\rho),

    and ``hat(M)`` is the logical tensor-product mass matrix used by
    :class:`MassMatrixDiagonalPreconditioner`.

    Parameters
    ----------
    kinetic_metric
        An H1vecKineticMetric-like operator providing ``mass_operator``,
        ``divdiv_operator`` and ``alpha``.

    apply_bc : bool
        Whether extraction and boundary operators are included when
        applying the preconditioner.
    """

    def __init__(
        self,
        kinetic_metric,
        apply_bc=True,
    ):
        self._validate_metric(kinetic_metric)

        self._kinetic_metric = kinetic_metric
        self._divdiv_operator = kinetic_metric.divdiv_operator
        self._metric_alpha = kinetic_metric.alpha

        mass_operator = kinetic_metric.mass_operator

        # Construct the logical mass solver, extraction/boundary
        # temporaries and diagonal scaling infrastructure.
        super().__init__(
            mass_operator,
            apply_bc=apply_bc,
        )

        self._initialize_metric_diagonal_matrix()
        self._update_metric_diagonal()

    @staticmethod
    def _validate_metric(kinetic_metric):
        required_attributes = (
            "mass_operator",
            "divdiv_operator",
            "alpha",
            "domain",
            "codomain",
        )

        for attribute in required_attributes:
            if not hasattr(kinetic_metric, attribute):
                raise TypeError(f"kinetic_metric must provide the attribute '{attribute}'.")

        mass_operator = kinetic_metric.mass_operator
        divdiv_operator = kinetic_metric.divdiv_operator

        if not isinstance(
            mass_operator,
            WeightedMassOperator,
        ):
            raise TypeError("kinetic_metric.mass_operator must be a WeightedMassOperator.")

        if kinetic_metric.domain != kinetic_metric.codomain:
            raise ValueError("The kinetic metric must be square.")

        if mass_operator.domain != divdiv_operator.domain:
            raise ValueError("The mass and div-div operators must have the same domain.")

        if mass_operator.codomain != divdiv_operator.codomain:
            raise ValueError("The mass and div-div operators must have the same codomain.")

        if kinetic_metric.alpha < 0.0:
            raise ValueError(f"The div-div coefficient must be non-negative, got {kinetic_metric.alpha}.")

        mass_matrix = mass_operator.matrix
        divdiv_matrix = divdiv_operator.matrix

        if not isinstance(
            mass_matrix,
            BlockLinearOperator,
        ):
            raise TypeError("The H1vec mass matrix must be a BlockLinearOperator.")

        if not isinstance(
            divdiv_matrix,
            BlockLinearOperator,
        ):
            raise TypeError("The H1vec div-div matrix must be a BlockLinearOperator.")

        if mass_matrix.domain != divdiv_matrix.domain:
            raise ValueError("The tensor mass and div-div matrices have incompatible domains.")

        if mass_matrix.codomain != divdiv_matrix.codomain:
            raise ValueError("The tensor mass and div-div matrices have incompatible codomains.")

    def _initialize_metric_diagonal_matrix(self):
        """
        Allocate diagonal-only storage for

            diag(M_rho), diag(K_rho), diag(M_rho + beta K_rho).
        """
        mass_matrix = self._mass_operator.matrix
        divdiv_matrix = self._divdiv_operator.matrix

        self._mass_diagonal = mass_matrix.diagonal()
        self._divdiv_diagonal = divdiv_matrix.diagonal()
        self._metric_diagonal = mass_matrix.diagonal()

    def _update_metric_diagonal(self):
        """
        Update diag(M_rho + beta K_rho) without copying full stencil blocks.
        """
        self._mass_operator.matrix.diagonal(
            out=self._mass_diagonal,
        )

        self._divdiv_operator.matrix.diagonal(
            out=self._divdiv_diagonal,
        )

        _combine_diagonals(
            self._mass_diagonal,
            self._divdiv_diagonal,
            self._metric_diagonal,
            alpha=self._metric_alpha,
        )

        _inverse_sqrt_diagonal_inplace(
            self._metric_diagonal,
        )

        # The parent solve expects this to contain D_A^{-1/2}.
        self._M_invsrqt_diag = self._metric_diagonal

        self._update_scaled_diagonal()

    @property
    def kinetic_metric(self):
        """Regularized kinetic metric being preconditioned."""
        return self._kinetic_metric

    @property
    def divdiv_operator(self):
        """Density-weighted H1vec div-div operator."""
        return self._divdiv_operator

    @property
    def metric_alpha(self):
        """
        Coefficient multiplying the div-div operator.

        This is already ``2 * alpha_divdiv``.
        """
        return self._metric_alpha

    @property
    def metric_diagonal_matrix(self):
        """
        Diagonal operator containing

            diag(M_rho + metric_alpha * K_rho)^(-1/2).
        """
        return self._metric_diagonal

    @profile
    def update_metric(
        self,
        kinetic_metric=None,
    ):
        """
        Update the preconditioner after the density-dependent mass and
        div-div operators have been reassembled.

        Parameters
        ----------
        kinetic_metric : optional
            Updated kinetic metric. If omitted, the existing metric
            reference is reused.
        """
        if kinetic_metric is not None:
            self._validate_metric(kinetic_metric)

            if kinetic_metric.domain != self.domain:
                raise ValueError("The updated kinetic metric has an incompatible domain.")

            if kinetic_metric.codomain != self.codomain:
                raise ValueError("The updated kinetic metric has an incompatible codomain.")

            self._kinetic_metric = kinetic_metric
            self._divdiv_operator = kinetic_metric.divdiv_operator
            self._metric_alpha = kinetic_metric.alpha

        mass_operator = self._kinetic_metric.mass_operator

        # This updates the parent references to M/M0 and temporarily
        # computes the mass-only inverse square-root diagonal.
        self._set_mass_operator_references(mass_operator)
        self._update_metric_diagonal()

    def transpose(self, conjugate=False):
        """The preconditioner is symmetric."""
        return self

    @profile
    def solve(self, rhs, out=None):
        """Apply the kinetic-metric preconditioner.

        The logical Kronecker solve used by the inherited implementation is
        efficient on the host, but on the current CuPy backend it launches a
        long sequence of small device operations for every outer Krylov
        iteration.  For device vectors use the already available diagonal
        mass core instead; it is fully device-resident and preserves the same
        extraction and boundary handling.  The CPU path retains the stronger
        Kronecker preconditioner.
        """
        first_data = rhs.blocks[0]._data if hasattr(rhs, "blocks") else rhs._data
        if xp.is_gpu(first_data):
            return self._solve_with_core(
                rhs,
                out,
                self._solve_diagonal_no_bc,
            )

        return super().solve(rhs, out=out)


class H1vecKineticMetricWoodburyPreconditioner(
    LinearOperator,
):
    r"""
    Schur-Woodbury preconditioner for

        A = M_rho + beta K_rho,

    with

        K_rho = R.T R,
        R = sqrt(W_rho) Q.

    The preconditioner is

        P^-1 =
            Z
            - beta Z R.T
              (I + beta R Z R.T)^-1
              R Z,

    where Z is a mass-matrix approximate inverse.

    The auxiliary quadrature-space system is solved privately with CG.
    """

    def __init__(
        self,
        kinetic_metric,
        *,
        auxiliary_nsteps=2,
        spectral_safety=1.5,
        spectral_iterations=8,
        mass_inverse_kind="diagonal",
    ):
        self._validate_metric(kinetic_metric)

        if auxiliary_nsteps < 1:
            raise ValueError("auxiliary_nsteps must be at least one.")

        if spectral_iterations < 1:
            raise ValueError("spectral_iterations must be at least one.")

        if spectral_safety <= 1.0:
            raise ValueError("spectral_safety should be greater than one.")

        self._kinetic_metric = kinetic_metric
        self._mass_operator = kinetic_metric.mass_operator
        self._divdiv_operator = kinetic_metric.divdiv_operator
        self._beta = kinetic_metric.alpha

        self._domain = kinetic_metric.domain
        self._codomain = kinetic_metric.codomain
        self._dtype = kinetic_metric.dtype

        self._auxiliary_nsteps = auxiliary_nsteps
        self._spectral_safety = spectral_safety
        self._spectral_iterations = spectral_iterations

        # ------------------------------------------------------------
        # Base mass approximate inverse Z
        # ------------------------------------------------------------
        if mass_inverse_kind not in (
            "diagonal",
            "kronecker",
        ):
            raise ValueError(f"mass_inverse_kind must be either 'diagonal' or 'kronecker', got {mass_inverse_kind!r}.")

        self._mass_inverse_kind = mass_inverse_kind
        # ------------------------------------------------------------
        # Velocity-space temporaries
        # ------------------------------------------------------------

        # Main Woodbury application.
        self._z = self.domain.zeros()
        self._rt_y = self.domain.zeros()
        self._z_rt_y = self.domain.zeros()

        # Application of S = R Z R.T.
        self._aux_velocity_1 = self.domain.zeros()
        self._aux_velocity_2 = self.domain.zeros()

        # ------------------------------------------------------------
        # Quadrature-space temporaries
        # ------------------------------------------------------------

        shape = self._divdiv_operator.quadrature_shape

        # Main Woodbury application.
        self._q = xp.zeros(shape, dtype=float)
        self._y = xp.zeros(shape, dtype=float)

        # Application of S and H.
        self._aux_RZR = xp.zeros(shape, dtype=float)
        self._aux_Hy = xp.zeros(shape, dtype=float)
        self._aux_residual = xp.zeros(shape, dtype=float)
        # Search direction used by the Chebyshev semi-iteration.
        self._aux_direction = xp.zeros(shape, dtype=float)

        # Spectral estimation.
        self._spectral_x = xp.zeros(shape, dtype=float)
        self._spectral_Sx = xp.zeros(shape, dtype=float)

        self._estimated_lambda_S = 0.0
        self._estimated_upper_H = 1.0
        self._mass_inverse = MassMatrixDiagonalPreconditioner(
            self._mass_operator,
        )

        self._estimate_auxiliary_spectrum()

        self._estimated_lambda_S = 0.0
        self._estimated_upper_H = 1.0
        self._estimated_tau = 0.0

    @staticmethod
    def _validate_metric(kinetic_metric):
        required = (
            "mass_operator",
            "divdiv_operator",
            "alpha",
            "domain",
            "codomain",
            "dtype",
        )

        for name in required:
            if not hasattr(kinetic_metric, name):
                raise TypeError(f"kinetic_metric must provide attribute '{name}'.")

        if kinetic_metric.domain != kinetic_metric.codomain:
            raise ValueError("The kinetic metric must be square.")

        if kinetic_metric.alpha < 0.0:
            raise ValueError("The kinetic-metric coefficient must be non-negative.")

        divdiv = kinetic_metric.divdiv_operator

        for name in (
            "apply_R",
            "apply_RT",
            "quadrature_shape",
        ):
            if not hasattr(divdiv, name):
                raise TypeError(f"The div-div operator must provide '{name}'.")

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def kinetic_metric(self):
        return self._kinetic_metric

    @property
    def mass_inverse(self):
        return self._mass_inverse

    @property
    def mass_inverse_kind(self):
        """Mass inverse used in the Woodbury factorization."""
        return self.mass_inverse_kind

    def _apply_Z(self, rhs, out):
        """
        Apply the selected base inverse Z.

        In diagonal mode:

            Z = diag(M_rho)^{-1}.

        In Kronecker mode:

            Z = the full diagonal-scaled logical mass preconditioner.
        """
        if self._mass_inverse_kind == "diagonal":
            return self._mass_inverse.solve_diagonal(
                rhs,
                out=out,
            )

        return self._mass_inverse.dot(
            rhs,
            out=out,
        )

    @property
    def auxiliary_iterations(self):
        """Fixed number of auxiliary Chebyshev iterations."""
        return self._auxiliary_nsteps

    @property
    def estimated_regularization_strength(self):
        """Estimated beta * lambda_max(R Z R.T)."""
        return self._estimated_tau

    @property
    def auxiliary_residual(self):
        return self._auxiliary_residual

    def _global_dot(self, x, y):
        """
        Global Euclidean scalar product on the distributed quadrature
        arrays.
        """
        value = float(xp.sum(x * y))

        comm = self._divdiv_operator.derham.comm

        if comm is not None and not isinstance(comm, MockComm):
            value = comm.allreduce(
                value,
                op=MPI.SUM,
            )

        return value

    @profile
    def _apply_S(self, x, out):
        """
        Apply S = R Z R.T.
        """
        self._divdiv_operator.apply_RT(
            x,
            out=self._aux_velocity_1,
        )

        self._apply_Z(
            self._aux_velocity_1,
            out=self._aux_velocity_2,
        )

        self._divdiv_operator.apply_R(
            self._aux_velocity_2,
            out=out,
        )

        return out

    def _estimate_auxiliary_spectrum(self):
        r"""
        Estimate a safe spectral interval for

            H = I + beta S,
            S = R Z R.T.

        Since S is positive semidefinite, the lower bound for H is one.
        A power iteration is used to estimate lambda_max(S), followed by
        a safety factor.
        """
        x = self._spectral_x
        Sx = self._spectral_Sx

        x[:] = 1.0

        norm_sq = self._global_dot(x, x)

        if norm_sq <= 0.0:
            raise RuntimeError("Cannot initialize the Woodbury spectral estimate.")

        x /= norm_sq**0.5

        eigenvalue = 0.0

        for _ in range(self._spectral_iterations):
            self._apply_S(x, Sx)

            eigenvalue = self._global_dot(x, Sx)
            norm_sq = self._global_dot(Sx, Sx)

            if norm_sq <= 0.0:
                eigenvalue = 0.0
                break

            x[:] = Sx
            x /= norm_sq**0.5

        eigenvalue = max(float(eigenvalue), 0.0)

        # Safe estimate of lambda_max(S).
        lambda_upper_S = self._spectral_safety * eigenvalue

        # Spectral bounds for H = I + beta S.
        self._auxiliary_lambda_min = 1.0
        self._auxiliary_lambda_max = 1.0 + self._beta * lambda_upper_S

        self._estimated_lambda_S = eigenvalue
        self._estimated_upper_H = self._auxiliary_lambda_max

        tau = self._beta * eigenvalue
        safe_tau = self._beta * lambda_upper_S

        logger.info(
            "Woodbury-Chebyshev auxiliary parameters: "
            f"beta={self._beta:.6e}, "
            f"lambda_S_est={eigenvalue:.6e}, "
            f"beta_lambda_S={tau:.6e}, "
            f"safe_beta_lambda_S={safe_tau:.6e}, "
            f"lambda_H_min={self._auxiliary_lambda_min:.6e}, "
            f"lambda_H_max={self._auxiliary_lambda_max:.6e}, "
            f"steps={self._auxiliary_nsteps}."
        )

    def _apply_auxiliary_operator(self, y, out):
        r"""
        Apply

            H y = y + beta R Z R.T y.
        """
        if self._beta == 0.0:
            out[:] = y
            return out

        self._divdiv_operator.apply_RT(
            y,
            out=self._aux_velocity_1,
        )

        self._apply_Z(
            self._aux_velocity_1,
            out=self._aux_velocity_2,
        )

        self._divdiv_operator.apply_R(
            self._aux_velocity_2,
            out=self._aux_RZR,
        )

        out[:] = y
        out += self._beta * self._aux_RZR

        return out

    @profile
    def _solve_auxiliary(self, rhs, out):
        r"""
        Apply a fixed number of Chebyshev semi-iterations to

            H y = rhs,
            H = I + beta R Z R.T,

        starting from y_0 = 0.

        The spectral interval of H is assumed to be contained in

            [self._auxiliary_lambda_min,
             self._auxiliary_lambda_max].

        The fixed iteration count makes the cost deterministic and avoids
        global scalar products inside the auxiliary solve.
        """
        if self._beta == 0.0:
            out[:] = rhs
            return out

        lambda_min = self._auxiliary_lambda_min
        lambda_max = self._auxiliary_lambda_max

        if lambda_min <= 0.0:
            raise ValueError("The Chebyshev lower spectral bound must be positive.")

        if lambda_max < lambda_min:
            raise ValueError("The Chebyshev upper spectral bound is smaller than the lower bound.")

        # If the interval has collapsed, H is effectively a multiple of
        # the identity.
        if lambda_max == lambda_min:
            out[:] = rhs
            out /= lambda_min
            return out

        theta = 0.5 * (lambda_max + lambda_min)
        delta = 0.5 * (lambda_max - lambda_min)
        sigma = theta / delta

        residual = self._aux_residual
        direction = self._aux_direction
        Hy = self._aux_Hy

        # Initial guess y_0 = 0, hence r_0 = rhs.
        residual[:] = rhs

        # First Chebyshev update:
        #
        #     d_0 = r_0 / theta,
        #     y_1 = y_0 + d_0.
        direction[:] = residual
        direction /= theta

        out[:] = direction

        # rho_0 = 1 / sigma = delta / theta.
        rho = 1.0 / sigma

        # Remaining fixed Chebyshev iterations.
        for _ in range(1, self._auxiliary_nsteps):
            # Recompute the residual:
            #
            #     r = rhs - H out.
            #
            # Recomputing it is slightly more expensive than a recurrence,
            # but is more robust against accumulated roundoff.
            self._apply_auxiliary_operator(
                out,
                out=Hy,
            )

            residual[:] = rhs
            residual -= Hy

            rho_new = 1.0 / (2.0 * sigma - rho)

            beta_cheb = rho_new * rho
            alpha_cheb = 2.0 * rho_new / delta

            # d_k = beta_k d_{k-1} + alpha_k r_k.
            direction *= beta_cheb
            direction += alpha_cheb * residual

            out += direction

            rho = rho_new

        return out

    def update_metric(self, kinetic_metric=None, *, update_spectrum=False):
        """
        Update density-dependent mass and div-div references.

        The kinetic metric itself is assumed to have already been
        reassembled by H1vecKineticMetric.update_weight().
        """
        if kinetic_metric is not None:
            self._validate_metric(kinetic_metric)

            if kinetic_metric.domain != self.domain:
                raise ValueError("Updated kinetic metric has incompatible domain.")

            if kinetic_metric.codomain != self.codomain:
                raise ValueError("Updated kinetic metric has incompatible codomain.")

            self._kinetic_metric = kinetic_metric
            self._mass_operator = kinetic_metric.mass_operator
            self._divdiv_operator = kinetic_metric.divdiv_operator
            self._beta = kinetic_metric.alpha

        self._mass_inverse.update_mass_operator(
            self._mass_operator,
        )

        if update_spectrum:
            self._estimate_auxiliary_spectrum()

    @profile
    def dot(self, rhs, out=None):
        assert isinstance(rhs, Vector)
        assert rhs.space == self.domain

        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain

        if self._beta == 0.0:
            return self._mass_inverse.dot(rhs, out=out)

        # z = Z rhs.
        self._apply_Z(
            rhs,
            out=self._z,
        )

        # q = R z.
        self._divdiv_operator.apply_R(
            self._z,
            out=self._q,
        )

        # Approximately solve
        #
        #     (I + beta R Z R.T) y = q
        #
        # with fixed-step Chebyshev.
        self._solve_auxiliary(
            self._q,
            out=self._y,
        )

        # rt_y = R.T y.
        self._divdiv_operator.apply_RT(
            self._y,
            out=self._rt_y,
        )

        # z_rt_y = Z R.T y.
        self._apply_Z(
            self._rt_y,
            out=self._z_rt_y,
        )

        # out = Z rhs - beta Z R.T y.
        self._z.copy(out=out)

        self._z_rt_y *= self._beta
        out -= self._z_rt_y

        return out

    def solve(self, rhs, out=None):
        return self.dot(rhs, out=out)

    def transpose(self, conjugate=False):
        return self

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()


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
        # circmat comes from StencilMatrix.toarray(), which always returns a
        # host (NumPy) array. Store it in the active backend so the cached FFT
        # solve below also works with device-resident right-hand sides.
        assert isinstance(circmat, np.ndarray)
        assert is_circulant(circmat)

        self._space = xp.ndarray
        self._column = xp.asarray(circmat[:, 0]).copy()

        if not xp.isrealobj(self._column):
            raise TypeError("FFTSolver currently supports real circulant matrices only.")

        self._build_inverse_spectrum()

    # --------------------------------------
    # Abstract interface
    # --------------------------------------
    @property
    def space(self):
        return self._space

    def _build_inverse_spectrum(self):
        """Precompute the inverse real-FFT spectrum of the matrix."""

        spectrum = xp.fft.rfft(self._column)

        scale = max(
            float(xp.max(xp.abs(spectrum))),
            1.0,
        )
        threshold = 100.0 * xp.finfo(float).eps * scale

        if xp.any(xp.abs(spectrum) <= threshold):
            eps = 1.0e-4
            logger.info(
                "Stabilizing singular preconditioning FFTSolver with eps=%s.",
                eps,
            )

            self._column[0] *= 1.0 + eps
            spectrum = xp.fft.rfft(self._column)

            if xp.any(xp.abs(spectrum) <= threshold):
                raise xp.linalg.LinAlgError(
                    "The circulant preconditioning matrix remains singular after stabilization."
                )

        self._inverse_spectrum = 1.0 / spectrum

        # For a real circulant matrix, the transpose has the conjugate
        # Fourier spectrum.
        self._inverse_spectrum_transposed = xp.conjugate(
            self._inverse_spectrum,
        )

    @profile
    def solve(self, rhs, out=None, transposed=False):
        """
        Solve a circulant system using a cached Fourier spectrum.
        """
        assert rhs.shape[-1] == self._column.size

        if out is None:
            out = xp.empty_like(rhs)
        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

        rhs_array = xp.asarray(rhs)

        rhs_fourier = xp.fft.rfft(
            rhs_array,
            axis=-1,
        )

        if transposed:
            rhs_fourier *= self._inverse_spectrum_transposed
        else:
            rhs_fourier *= self._inverse_spectrum

        result = xp.fft.irfft(
            rhs_fourier,
            n=self._column.size,
            axis=-1,
        )

        out[:] = result
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

    # mat is always a host (NumPy) array in practice: the only callers pass
    # StencilMatrix.toarray() output, which feectools always returns on the host.
    assert isinstance(mat, np.ndarray)
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]

    if mat.shape[0] > 1:
        for i in range(mat.shape[0] - 1):
            circulant = np.allclose(mat[i, :], np.roll(mat[i + 1, :], -1))
            if not circulant:
                return circulant
    else:
        circulant = True

    return circulant
