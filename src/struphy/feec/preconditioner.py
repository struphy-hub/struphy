from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import cunumpy as xp
from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import ComposedLinearOperator, LinearOperator, Vector, VectorSpace
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.direct_solvers import BandedSolver, SparseSolver
from feectools.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix
from feectools.linalg.stencil import StencilMatrix, StencilVectorSpace
from line_profiler import profile
from maybempi import MPI, SerialComm
from scipy import sparse

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator

logger = logging.getLogger("struphy")


class MassMatrixPreconditioner(LinearOperator):
    r"""
    Preconditioner for inverting 3d weighted mass matrices.

    The mass matrix is approximated by a Kronecker product of 1d mass matrices
    (block diagonal for vector-valued spaces), which is inverted exactly with a
    :class:`~feectools.linalg.kron.KroneckerLinearSolver`:

    * In the direction ``dim_reduce``, the 1d mass matrix carries a 1d weight:
      the diagonal block ``(c, c)`` of the 3d weight, evaluated at the mid point
      (0.5) of the other two directions.
    * In the other directions, the 1d mass matrices are unweighted.
    * Essential boundary conditions of the mass operator are imposed on the 1d
      matrices (identity rows).

    The preconditioner applies :math:`B E \tilde M^{-1} E^T B^T`, where
    :math:`B E \dots E^T B^T` is the composition of the mass operator (if any)
    and :math:`\tilde M` is the Kronecker approximation (see :meth:`solve`).

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        The weighted mass operator for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.

    dim_reduce : int
        Axis along which the weight is kept (in the other axes it is taken at 0.5).
    """

    def __init__(self, mass_operator: WeightedMassOperator, apply_bc: bool = True, dim_reduce: int = 0):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square mass matrices can be inverted!"

        self._mass_operator = mass_operator
        self._femspace = mass_operator.domain_femspace
        self._space = mass_operator.domain
        self._dtype = mass_operator.dtype
        self._codomain = mass_operator.codomain
        self._domain = mass_operator.domain
        self._apply_bc = apply_bc
        self._dim_reduce = dim_reduce

        n_dims = self._femspace.ldim
        assert n_dims == 3  # other dims not yet implemented
        assert dim_reduce < n_dims

        # boundary conditions are only imposed if the mass operator has a BoundaryOperator
        bc = _boundary_conditions(mass_operator, apply_bc)
        apply_bc = bc is not None

        derham = mass_operator.derham
        logger.debug(f"{derham.num_elements = }, {derham.bcs = }, {derham.degree = }")

        # MPI setup for gathering array-valued weights along dim_reduce
        gather = _WeightGather.setup(derham, dim_reduce)

        def weight_1d(c: int, d: int) -> Callable | xp.ndarray:
            if d == dim_reduce:
                return _reduced_weight_1d(mass_operator.weights[c][c], c, d, derham, gather)
            return _ones_1d

        # Kronecker approximation of the mass matrix and its exact inverse
        self._matrix, self._solver = _kronecker_approximation(mass_operator, bc, weight_1d)

        # mass operator to be inverted (with boundary operators B, E if apply_bc), needed in solve
        self._M, self._is_composed, tmp = _operator_to_invert(mass_operator, apply_bc)
        if self._is_composed:
            self._tmp_vectors = tmp
        else:
            self._tmp_vector = tmp

    @property
    def space(self) -> VectorSpace:
        """Stencil-/BlockVectorSpace or PolarDerhamSpace."""
        return self._space

    @property
    def matrix(self) -> KroneckerStencilMatrix | BlockLinearOperator:
        """Approximation of the input mass matrix as KroneckerStencilMatrix (block diagonal for vector-valued spaces)."""
        return self._matrix

    @property
    def solver(self) -> KroneckerLinearSolver | BlockLinearOperator:
        """KroneckerLinearSolver (block diagonal for vector-valued spaces) for exactly inverting the approximate mass matrix self.matrix."""
        return self._solver

    @property
    def domain(self) -> VectorSpace:
        """The domain of the linear operator - an element of Vectorspace"""
        return self._space

    @property
    def codomain(self) -> VectorSpace:
        """The codomain of the linear operator - an element of Vectorspace"""
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def transpose(self, conjugate: bool = False) -> "MassMatrixPreconditioner":
        """
        Returns the transposed operator.
        """
        return MassMatrixPreconditioner(self._mass_operator.transpose(), self._apply_bc, self._dim_reduce)

    @profile
    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """
        Computes (B * E * M^(-1) * E^T * B^T) * rhs as an approximation for an inverse mass matrix.

        The operators of the composition are applied from right to left; the mass
        matrix itself is replaced by the Kronecker solver.

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

        if not self._is_composed:
            if out is None:
                out = self._tmp_vector.copy()
            self.solver.dot(rhs, out=out)
            return out

        # successive dot products with all but the first (left-most) operator
        x = rhs
        for A, y in zip(reversed(self._M.multiplicants[1:]), reversed(self._tmp_vectors)):
            if isinstance(A, (StencilMatrix, BlockLinearOperator)):
                # the mass matrix: apply the approximate inverse instead
                self.solver.dot(x, out=y)
            else:
                A.dot(x, out=y)
            x = y

        # first operator
        A = self._M.multiplicants[0]
        if out is None:
            out = A.dot(x)
        else:
            assert isinstance(out, Vector)
            assert out.space == self._space
            A.dot(x, out=out)

        return out

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
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

        n_dims = self._femspace.ldim
        assert n_dims == 3  # other dims not yet implemented

        # boundary conditions are only imposed if the mass operator has a BoundaryOperator
        bc = _boundary_conditions(mass_operator, apply_bc)
        apply_bc = bc is not None

        # mass matrix on the logical domain (unit weights) as Kronecker product, and its exact inverse
        self._matrix, self._solver = _kronecker_approximation(mass_operator, bc, lambda c, d: _ones_1d)

        # mass operator to be inverted (with boundary operators B, E if apply_bc), needed in solve
        self._M, self._is_composed, tmp = _operator_to_invert(mass_operator, apply_bc)
        if self._is_composed:
            self._tmp_vectors = tmp
        else:
            self._tmp_vector = tmp

        # Need to assemble the logical mass matrix to extract the coefficients
        fun = [
            [(lambda e1, e2, e3: xp.ones_like(e1, dtype=float)) if i == j else None for j in range(3)] for i in range(3)
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

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return MassMatrixDiagonalPreconditioner(self._mass_operator.transpose(), self._apply_bc)

    def _solve_no_bc(self, rhs, out):
        r"""
        Computes M^(-1) * rhs as an approximation for an inverse mass matrix.
        With $M = D^{1/2} * \hat D^{-1/2} * \hat M * \hat D^{-1/2} * D^{1/2}$
        Should only be called by the solve method that will handle the bcs.

        Parameters
        ----------
        rhs : feectools.linalg.basic.Vector
            The right-hand side vector.

        out : feectools.linalg.basic.Vector
            The output vector will be written into this vector in-place.

        Returns
        -------
        out : feectools.linalg.basic.Vector
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

    @profile
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


# --------------------------------------------------------------------------------------
# Helper functions for building Kronecker approximations of mass matrices
# --------------------------------------------------------------------------------------
def _ones_1d(e: xp.ndarray) -> xp.ndarray:
    """Unit weight for 1d mass matrices."""
    return xp.ones(e.size, dtype=float)


def _boundary_conditions(mass_operator: WeightedMassOperator, apply_bc: bool) -> list | None:
    """
    Boundary conditions of the mass operator, taken from the BoundaryOperator
    at the right end of the composition ``mass_operator.M0``.

    Returns
    -------
    list | None
        ``bc[d] = (left, right)`` flags of essential boundary conditions per direction,
        or None if ``apply_bc`` is False or ``M0`` has no BoundaryOperator.
    """
    if not apply_bc or not isinstance(mass_operator.M0, ComposedLinearOperator):
        return None
    last = mass_operator.M0.multiplicants[-1]
    if not isinstance(last, BoundaryOperator):
        return None
    return last.bc


def _operator_to_invert(
    mass_operator: WeightedMassOperator, apply_bc: bool
) -> tuple[LinearOperator, bool, tuple[Vector, ...] | Vector]:
    """
    The operator approximately inverted by the preconditioners: ``M0`` (with boundary
    operators) if ``apply_bc``, else ``M``.

    Returns
    -------
    M : LinearOperator
        The operator.

    is_composed : bool
        Whether ``M`` is a ComposedLinearOperator.

    tmp : tuple[Vector, ...] | Vector
        Temporary vectors for ``solve``: the codomains of all but the first factor of
        ``M`` if it is composed, else one vector in the codomain of ``M``.
    """
    M = mass_operator.M0 if apply_bc else mass_operator.M
    is_composed = isinstance(M, ComposedLinearOperator)
    if is_composed:
        tmp = tuple(op.codomain.zeros() for op in M.multiplicants[1:])
    else:
        tmp = M.codomain.zeros()
    return M, is_composed, tmp


@dataclass
class _WeightGather:
    """
    MPI setup for gathering a 1d cut of an array-valued weight along ``dim_reduce``.

    In the directions other than ``dim_reduce``, the weight is taken at the global mid
    point; the ranks owning the mid element there (exactly one rank per slab along
    ``dim_reduce``) form ``subcomm``. ``root`` is the lowest of these ranks in ``comm``.
    """

    comm: object
    subcomm: object
    root: int

    @classmethod
    def setup(cls, derham, dim_reduce: int) -> "_WeightGather | None":
        """Collective on ``derham.comm``; returns None in serial runs."""
        comm = derham.comm
        if isinstance(comm, (SerialComm, type(None))):
            return None

        dom_dec = derham.domain_decomposition
        rank = comm.Get_rank()
        is_selected = all(
            dom_dec.starts[i] <= derham.num_elements[i] // 2 <= dom_dec.ends[i]
            for i in range(len(derham.num_elements))
            if i != dim_reduce
        )
        color = 0 if is_selected else MPI.UNDEFINED
        subcomm = comm.Split(color=color, key=rank)
        root = comm.allreduce(rank if is_selected else comm.Get_size(), op=MPI.MIN)
        logger.debug(f"Rank {rank} selected for gathering 1d weight info in dimension {dim_reduce}: {is_selected}")
        return cls(comm, subcomm, root)


def _reduced_weight_1d(weight, c: int, d: int, derham, gather: _WeightGather | None) -> Callable | xp.ndarray:
    """
    1d weight along direction ``d``: the 3d weight at the mid point (0.5) of the other directions.

    Parameters
    ----------
    weight : callable | xp.ndarray | None
        Block ``(c, c)`` of the weights of the 3d mass operator: a function of the three
        logical coordinates, its values at the local quadrature points, or None (unit weight).

    c, d : int
        Component and direction (only used for logging and for the callable case).

    derham : Derham
        Discrete de Rham sequence of the mass operator.

    gather : _WeightGather | None
        MPI setup from ``_WeightGather.setup``; None in serial runs.

    Returns
    -------
    Callable | xp.ndarray
        A function of the 1d quadrature points, or the weight at all global 1d
        quadrature points (collective on ``gather.comm`` in parallel runs).
    """
    n_dims = 3

    if weight is None:
        return _ones_1d

    if callable(weight):

        def fun(e):
            # evaluate the 3d weight on the "meshgrid" (0.5, ..., e, ..., 0.5)
            s = e.shape[0]
            newshape = tuple([1 if i != d else s for i in range(n_dims)])
            f = e.reshape(newshape)
            return xp.atleast_1d(
                weight(
                    *[xp.array(xp.full_like(f, 0.5)) if i != d else xp.array(f) for i in range(n_dims)],
                ).squeeze(),
            )

        return fun

    if isinstance(weight, xp.ndarray):
        s = weight.shape
        logger.debug(f"{weight.shape = } for component {c} and direction {d}.")
        dom_dec = derham.domain_decomposition
        npts = derham.num_elements[d] * derham.nquads[d]
        fun = xp.zeros(npts, dtype=float)

        # local index of the global mid quadrature point in the other directions
        # (clipped on non-selected ranks, which receive the gathered weight below)
        mid = [0] * n_dims
        for i in range(n_dims):
            if i != d:
                nq_i = s[i] // dom_dec.local_ncells[i]
                mid_i = (derham.num_elements[i] * nq_i) // 2 - dom_dec.starts[i] * nq_i
                mid[i] = min(max(mid_i, 0), s[i] - 1)
        cut = tuple(slice(None) if i == d else mid[i] for i in range(n_dims))
        local_fun = xp.ascontiguousarray(weight[cut], dtype=float)

        logger.debug(f"{fun.size = } for component {c} and direction {d} before gathering on all processes.")
        if gather is not None:
            # local sizes differ if num_elements[d] is not divisible by the number of processes
            if gather.subcomm != MPI.COMM_NULL:
                counts = gather.subcomm.allgather(local_fun.size)
                displs = [sum(counts[:j]) for j in range(len(counts))]
                gather.subcomm.Allgatherv(local_fun, [fun, counts, displs, MPI.DOUBLE])
            gather.comm.Bcast(fun, root=gather.root)
        else:
            fun[:] = local_fun
        logger.debug(f"{fun.shape = } for component {c} and direction {d} after gathering on all processes.")
        return fun

    raise TypeError(f"weights needs to be callable, xp.ndarray or None but is {type(weight)}")


def _has_essential_bc(mass_operator: WeightedMassOperator, basis: str, c: int, d: int) -> bool:
    """
    Whether the boundary conditions of direction ``d`` apply to the 1d matrix of component ``c``.

    For H1 vector spaces (``H1H1H1``, ``H1vec``) this is the case if ``c == d``; for the other
    spaces, in all directions with B-splines (``basis == "B"``).
    """
    if mass_operator._domain_symbolic_name in ("H1H1H1", "H1vec"):
        return c == d
    return basis == "B"


def _mass_matrix_1d(
    mass_operator: WeightedMassOperator, basis: str, d: int, weight: Callable | xp.ndarray
) -> tuple[StencilMatrix, DomainDecomposition]:
    """
    Assemble the 1d mass matrix in direction ``d`` on the serial (not distributed) 1d space.

    Parameters
    ----------
    basis : str
        ``"B"`` for B-splines (H1), else M-splines (L2).

    weight : Callable | xp.ndarray
        1d weight, see ``_reduced_weight_1d``.

    Returns
    -------
    M : StencilMatrix
        The 1d mass matrix.

    domain_decomposition : DomainDecomposition
        Domain decomposition of the serial 1d space.
    """
    derham = mass_operator.derham
    femspace_1d = derham.H1_1d_serial[d] if basis == "B" else derham.L2_1d_serial[d]

    M = WeightedMassOperator(
        derham,
        femspace_1d,
        femspace_1d,
        weights_info=[[weight]],
        nquads=(derham.nquads[d],),
    )
    M.assemble()
    return M.matrix, femspace_1d.domain_decomposition


def _apply_bc_1d(M: StencilMatrix, bc_d: tuple[bool, bool]) -> None:
    """Impose essential boundary conditions (identity rows) at the left/right end of the 1d matrix M (in place)."""
    for ext, is_essential in zip((-1, +1), bc_d):
        if is_essential:
            apply_essential_bc_stencil(M, axis=0, ext=ext, order=0, identity=True)


def _solver_1d(M: StencilMatrix, M_arr: xp.ndarray) -> FFTSolver | SparseSolver:
    """Direct solver for the 1d matrix M (dense copy M_arr): FFT if circulant, else sparse LU."""
    if is_circulant(M_arr):
        return FFTSolver(M_arr)
    return SparseSolver(M.tosparse())


def _process_local_matrix_1d(
    M_arr: xp.ndarray, coeff_space: StencilVectorSpace, d: int, domain_decomposition: DomainDecomposition
) -> StencilMatrix:
    """
    Process-local 1d factor of a KroneckerStencilMatrix on ``coeff_space``.

    The factor lives on a 1d space without communicator that owns the same rows
    (``starts[d]`` to ``ends[d]``) as ``coeff_space`` in direction ``d`` on this process,
    as required by KroneckerStencilMatrix.

    Parameters
    ----------
    M_arr : xp.ndarray
        The global 1d matrix as dense array.

    coeff_space : StencilVectorSpace
        The (distributed) 3d coefficient space.

    d : int
        Direction.

    domain_decomposition : DomainDecomposition
        Domain decomposition of the serial 1d space.
    """
    n = coeff_space.npts[d]
    p = coeff_space.pads[d]
    s = coeff_space.starts[d]
    e = coeff_space.ends[d]

    cart_1d = CartDecomposition(domain_decomposition, [n], [[s]], [[e]], [p], [1])
    V_local = StencilVectorSpace(cart_1d)
    M_local = StencilMatrix(V_local, V_local)

    # copy the rows owned by this process: entry (i, j) is stored at row i - s + p, diagonal (j - i + p) mod n
    rows, cols = xp.nonzero(M_arr)
    on_process = (rows >= s) & (rows <= e)
    rows, cols = rows[on_process], cols[on_process]
    M_local._data[rows - s + p, (cols + p - rows) % M_arr.shape[1]] = M_arr[rows, cols]

    # check if stencil matrix was built correctly
    assert xp.allclose(M_local.toarray()[s : e + 1], M_arr[s : e + 1])

    return M_local


def _kronecker_approximation(
    mass_operator: WeightedMassOperator,
    bc: list | None,
    weight_1d: Callable[[int, int], Callable | xp.ndarray],
) -> tuple[KroneckerStencilMatrix | BlockLinearOperator, KroneckerLinearSolver | BlockLinearOperator]:
    """
    Approximate the mass matrix by Kronecker products of 1d mass matrices
    (block diagonal for vector-valued spaces) and build their exact inverse.

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        The mass operator to approximate.

    bc : list | None
        Boundary conditions from ``_boundary_conditions``; None for no boundary conditions.

    weight_1d : callable
        ``weight_1d(c, d)`` returns the 1d weight of component ``c`` in direction ``d``.

    Returns
    -------
    matrix : KroneckerStencilMatrix | BlockLinearOperator
        The approximate mass matrix.

    solver : KroneckerLinearSolver | BlockLinearOperator
        Its exact inverse.
    """
    femspace = mass_operator.domain_femspace
    is_scalar = isinstance(femspace, TensorFemSpace)
    femspaces = (femspace,) if is_scalar else femspace.spaces
    n_dims = femspace.ldim

    matrixblocks = []
    solverblocks = []
    for c, femspace_c in enumerate(femspaces):
        coeff_space = femspace.coeff_space if is_scalar else femspace.coeff_space[c]

        # 1d mass matrices (process-local) and solvers in each direction
        matrixcells = []
        solvercells = []
        for d in range(n_dims):
            basis = femspace_c.spaces[d].basis
            M, domain_decomposition = _mass_matrix_1d(mass_operator, basis, d, weight_1d(c, d))

            if bc is not None and _has_essential_bc(mass_operator, basis, c, d):
                _apply_bc_1d(M, bc[d])

            M_arr = M.toarray()
            solvercells.append(_solver_1d(M, M_arr))
            matrixcells.append(_process_local_matrix_1d(M_arr, femspace_c.coeff_space, d, domain_decomposition))

        matrixblocks.append(KroneckerStencilMatrix(coeff_space, coeff_space, *matrixcells))
        solverblocks.append(KroneckerLinearSolver(coeff_space, coeff_space, solvercells))

    if is_scalar:
        return matrixblocks[0], solverblocks[0]
    return (
        _block_diagonal(femspace.coeff_space, matrixblocks),
        _block_diagonal(femspace.coeff_space, solverblocks),
    )


def _block_diagonal(space: BlockVectorSpace, blocks: list[LinearOperator]) -> BlockLinearOperator:
    """Block-diagonal operator on ``space`` with the given diagonal blocks."""
    n = len(blocks)
    return BlockLinearOperator(
        space,
        space,
        blocks=[[blocks[i] if i == j else None for j in range(n)] for i in range(n)],
    )


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

    @profile
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

        from scipy.linalg import solve_circulant

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
                logger.info(f"Stabilizing singular preconditioning FFTSolver with {eps =}:")
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
