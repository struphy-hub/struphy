from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from functools import reduce

import cunumpy as xp
import numpy as np
from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.basic import (
    ComposedLinearOperator,
    IdentityOperator,
    LinearOperator,
    LinearSolver,
    ScaledLinearOperator,
    SumLinearOperator,
    Vector,
    VectorSpace,
)
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.direct_solvers import BandedSolver, SparseSolver
from feectools.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix, KroneckerSumSolver
from feectools.linalg.stencil import StencilDiagonalMatrix, StencilMatrix, StencilVector, StencilVectorSpace
from line_profiler import profile
from maybempi import MPI, SerialComm
from scipy import sparse
from scope_profiler import ProfileManager

from struphy.feec.kronecker_block import KroneckerBlockOperator
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedAverageProjection, WeightedMassOperator, WeightedMassOperators
from struphy.polar.basic import PolarDerhamSpace
from struphy.polar.kron_2dx1d import (
    ILU_DROP_TOL,
    ILU_FILL_FACTOR,
    Kron2Dx1DGroup,
    KronSum2Dx1DGroup,
    PolarKron2Dx1DSolver,
    PolarKronSum2Dx1DSolver,
    bc_masks,
    component_groups,
    poloidal_average_basis,
    poloidal_extraction_matrix,
    poloidal_mass_matrix,
    reduced_weight_2d,
    spline_bases,
    tp_space,
)

logger = logging.getLogger("struphy")


class KroneckerPreconditioner(LinearOperator):
    r"""
    Base class for preconditioners built from Kronecker approximations.

    The operator to be preconditioned has the form :math:`A = B E \, A_c \, E^T B^T`, where
    :math:`A_c` acts on the tensor-product coefficient space and :math:`B E` are the
    boundary and extraction operators of a mass operator (the composition ``M0`` or ``M``
    of ``mass_operator``). The preconditioner applies

    .. math::

        P = B E \, S \tilde A_c^{-1} S \, E^T B^T \,,

    where :math:`\tilde A_c` (``matrix``) approximates :math:`A_c` and can be inverted exactly
    (``solver``), e.g. as a Kronecker product. With ``diagonal_scaling``, the diagonal operator
    :math:`S = \hat D^{1/2} D^{-1/2}` with :math:`D = \mathrm{diag}(A_c)` and
    :math:`\hat D = \mathrm{diag}(\tilde A_c)` corrects the approximation pointwise
    (Loli, Sangalli, Tani, Comp. Math. Appl. 116, 2022); otherwise :math:`S = I`.

    Subclasses implement ``_build_approximation`` and ``transpose``, and ``_core_diagonal``
    for diagonal scaling.

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        Mass operator whose composition defines the spaces and :math:`B E`; its core
        (``mass_operator.matrix``) is replaced by :math:`S \tilde A_c^{-1} S`.

    apply_bc : bool
        Whether to include boundary operators.

    diagonal_scaling : bool
        Whether to correct the approximation with the diagonal scaling :math:`S`.
    """

    def __init__(
        self,
        mass_operator: WeightedMassOperator,
        apply_bc: bool = True,
        diagonal_scaling: bool = False,
    ):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square operators can be inverted!"

        self._mass_operator = mass_operator
        self._femspace = mass_operator.domain_femspace
        self._space = mass_operator.domain
        self._dtype = mass_operator.dtype
        self._codomain = mass_operator.codomain
        self._domain = mass_operator.domain
        self._apply_bc = apply_bc
        self._diagonal_scaling = diagonal_scaling

        assert self._femspace.ldim == 3  # other dims not yet implemented

        # boundary conditions are only imposed if the mass operator has a BoundaryOperator
        self._bc = _boundary_conditions(mass_operator, apply_bc)

        # approximation of the core operator and its exact inverse
        with ProfileManager.profile_region(f"build_approximation", functions=[self._build_approximation]):
            self._matrix, self._solver = self._build_approximation(self._bc)

        # diagonal scaling S (a sequence of diagonal operators, applied in this order before the
        # solver and in reverse order after it)
        with ProfileManager.profile_region(f"build_scaling", functions=[self._build_scaling]):
            self._scaling = self._build_scaling() if diagonal_scaling else ()
            self._tmp_core = tuple(self._matrix.codomain.zeros() for _ in range(2)) if self._scaling else ()

        # operator to be inverted (with boundary operators B, E if apply_bc), needed in solve
        self._M, self._mass_index, self._tmp_vectors = _operator_to_invert(mass_operator, self._bc is not None)

    # --------------------------------------
    # To be implemented by subclasses
    # --------------------------------------
    def _build_approximation(self, bc: list | None) -> tuple[LinearOperator, LinearOperator]:
        """
        Approximation of the core operator and its exact inverse.

        Parameters
        ----------
        bc : list | None
            Boundary conditions from ``_boundary_conditions``; None for no boundary conditions.

        Returns
        -------
        matrix : LinearOperator
            The approximation :math:`\\tilde A_c`.

        solver : LinearOperator
            Its exact inverse.
        """
        raise NotImplementedError

    def _core_diagonal(self) -> LinearOperator:
        """Diagonal :math:`D` of the core operator as diagonal operator (for diagonal scaling)."""
        raise NotImplementedError(f"{type(self).__name__} does not support diagonal scaling.")

    def _scaling_matrix(self) -> LinearOperator:
        """The approximation whose diagonal :math:`\\hat D` enters the diagonal scaling (default: ``matrix``)."""
        return self._matrix

    def transpose(self, conjugate: bool = False) -> KroneckerPreconditioner:
        raise NotImplementedError

    # --------------------------------------
    # Common interface
    # --------------------------------------
    @property
    def space(self) -> VectorSpace:
        """Stencil-/BlockVectorSpace or PolarDerhamSpace."""
        return self._space

    @property
    def matrix(self) -> LinearOperator:
        """Approximation of the core operator (Kronecker products, block diagonal for vector-valued spaces)."""
        return self._matrix

    @property
    def solver(self) -> LinearOperator:
        """Exact inverse of the approximation self.matrix."""
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

    @property
    def diagonal_scaling(self) -> bool:
        """Whether the approximation is corrected by diagonal scaling."""
        return self._diagonal_scaling

    def _build_scaling(self) -> tuple[LinearOperator, ...]:
        """Diagonal scaling :math:`S = \\hat D^{1/2} D^{-1/2}` as one diagonal operator."""
        with ProfileManager.profile_region(f"local_diagonal of core", functions=[_local_diagonal, self._core_diagonal]):
            D = _local_diagonal(self._core_diagonal())
        D_hat = _local_diagonal(self._scaling_matrix())
        S = [xp.where(d > 0, xp.sqrt(xp.abs(dh) / xp.where(d > 0, d, 1.0)), 1.0) for d, dh in zip(D, D_hat)]
        return (_diagonal_operator(self._matrix.domain, S),)

    def _apply_inverse(self, x: Vector, out: Vector) -> Vector:
        """Apply :math:`S \\tilde A_c^{-1} S` (the replacement of the core operator) to x."""
        if not self._scaling:
            return self._solver.dot(x, out=out)

        buf = self._tmp_core
        k = 0
        y = x
        for op in self._scaling:
            y = op.dot(y, out=buf[k])
            k = 1 - k
        y = self._solver.dot(y, out=buf[k])
        k = 1 - k
        for op in reversed(self._scaling[1:]):
            y = op.dot(y, out=buf[k])
            k = 1 - k
        return self._scaling[0].dot(y, out=out)

    @profile
    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """
        Computes :math:`B E \\, S \\tilde A_c^{-1} S \\, E^T B^T` rhs, an approximation of :math:`A^{-1}` rhs.

        The operators of the composition are applied from right to left; the core operator
        is replaced by :math:`S \\tilde A_c^{-1} S`.

        Parameters
        ----------
        rhs : feectools.linalg.basic.Vector
            The right-hand side vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.

        Returns
        -------
        out : feectools.linalg.basic.Vector
            The result.
        """
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._space

        return _apply_composed(self._M, self._mass_index, self._apply_inverse, self._tmp_vectors, rhs, out)

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        """Apply linear operator to Vector v. Result is written to Vector out, if provided."""
        assert isinstance(v, Vector)
        assert v.space == self.domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
        return self.solve(v, out=out)


class MassMatrixPreconditioner(KroneckerPreconditioner):
    r"""
    Preconditioner for inverting 3d weighted mass matrices.

    The mass matrix is approximated by a Kronecker product of 1d mass matrices
    (block diagonal for vector-valued spaces), which is inverted exactly with a
    :class:`~feectools.linalg.kron.KroneckerLinearSolver`:

    * In the direction ``dim_reduce``, the 1d mass matrix carries a 1d weight
      obtained from the diagonal block ``(c, c)`` of the 3d weight: its value at
      the mid point (0.5) of the other two directions, or its mean over them
      (see ``weight_reduction``). With ``dim_reduce=None``, no direction is weighted
      (mass matrix on the logical cube).
    * In the other directions, the 1d mass matrices are unweighted.
    * Essential boundary conditions of the mass operator are imposed on the 1d
      matrices (identity rows).

    The preconditioner applies :math:`B E \tilde M^{-1} E^T B^T`, where
    :math:`B E \dots E^T B^T` is the composition of the mass operator (if any)
    and :math:`\tilde M` is the Kronecker approximation (see :class:`KroneckerPreconditioner`).

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        The weighted mass operator for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.

    dim_reduce : int | None
        Axis along which the weight is kept (default 0); None for unit weights in all directions.
        With an axis, the approximation is exact for weights (and mappings) that vary only along
        that axis, e.g. radial profiles in eta1. With None, the Kronecker factors do not depend
        on the weights, so ``update_mass_operator`` only recomputes the diagonal scaling: use it
        for weights that change often (e.g. a density updated in every time step).

    weight_reduction : str
        How the weight is reduced in the other axes: ``"midpoint"`` (value at 0.5, default)
        or ``"average"`` (mean over the axes). Which one gives the better preconditioner
        depends on the weight.

    diagonal_scaling : bool
        Whether to correct the approximation with the diagonals of the mass matrix and of
        its approximation (see :class:`KroneckerPreconditioner`; default True).

    Notes
    -----
    The diagonal scaling follows `G. Loli, G. Sangalli, M. Tani, "Easy and efficient preconditioning
    of the isogeometric mass matrix", Comp. Math. Appl., Vol. 116, 2022
    <https://www.sciencedirect.com/science/article/pii/S0898122120304715?via%3Dihub>`_.
    """

    def __init__(
        self,
        mass_operator: WeightedMassOperator,
        apply_bc: bool = True,
        dim_reduce: int | None = 0,
        weight_reduction: str = "midpoint",
        diagonal_scaling: bool = True,
    ):
        assert dim_reduce is None or dim_reduce < 3
        assert weight_reduction in ("midpoint", "average"), f"Unknown weight_reduction {weight_reduction!r}."
        self._dim_reduce = dim_reduce
        self._weight_reduction = weight_reduction

        super().__init__(mass_operator, apply_bc=apply_bc, diagonal_scaling=diagonal_scaling)

    def _build_approximation(self, bc):
        mass_operator = self._mass_operator
        derham = mass_operator.derham
        logger.debug(f"{derham.num_elements = }, {derham.bcs = }, {derham.degree = }")

        def weight_1d(c: int, d: int) -> Callable | xp.ndarray:
            if d == self._dim_reduce:
                return _reduced_weight_1d(mass_operator.weights[c][c], c, d, derham, self._weight_reduction)
            return _ones_1d

        return _kronecker_approximation(mass_operator, bc, weight_1d)

    def _core_diagonal(self):
        # diagonal of the mass matrix, including the band entries that wrap onto the main diagonal
        # (StencilMatrix.diagonal() keeps only the entry at offset 0)
        M = self._mass_operator.matrix
        return _diagonal_operator(M.domain, _local_diagonal(M))

    @property
    def dim_reduce(self) -> int | None:
        """Axis along which the weight is kept (None: unit weights in all directions)."""
        return self._dim_reduce

    @property
    def weight_reduction(self) -> str:
        """How the weight is reduced in the directions other than ``dim_reduce``: ``"midpoint"`` or ``"average"``."""
        return self._weight_reduction

    def update_mass_operator(self, mass_operator: WeightedMassOperator) -> None:
        """
        Update the mass operator (same spaces and structure, e.g. new weights) to recycle the preconditioner.

        The Kronecker approximation is rebuilt only if it depends on the weights (``dim_reduce`` is
        not None); the diagonal scaling is updated.
        """
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square mass matrices can be inverted!"
        assert mass_operator.domain == self.domain, "Update needs to have the same domain and codomain"

        self._mass_operator = mass_operator

        # the composition has the same structure as before, so the temporary vectors can be reused
        M, mass_index, _ = _operator_to_invert(mass_operator, self._bc is not None)
        assert mass_index == self._mass_index, "The updated mass operator must have the same structure."
        self._M = M

        if self._dim_reduce is not None:
            self._matrix, self._solver = self._build_approximation(self._bc)
        if self._diagonal_scaling:
            self._scaling = self._build_scaling()

    def transpose(self, conjugate: bool = False) -> MassMatrixPreconditioner:
        """
        Returns the transposed operator.
        """
        return MassMatrixPreconditioner(
            self._mass_operator.transpose(),
            self._apply_bc,
            self._dim_reduce,
            self._weight_reduction,
            self._diagonal_scaling,
        )


class PolarMassMatrixPreconditioner(LinearOperator):
    r"""
    Preconditioner for inverting weighted mass matrices on polar spline spaces, exploiting the
    2D x 1D Kronecker structure of the polar extraction (see :mod:`struphy.polar.kron_2dx1d`):

    .. math::

        \mathbb B \mathbb E \mathbb M \mathbb E^T \mathbb B^T \approx \bigoplus_G
        \left(\mathbb E_\textnormal{pol} \mathbb M^G_{12}[\bar w] \mathbb E_\textnormal{pol}^T\right) \otimes \mathbb M^G_3 \,,

    with boundary conditions imposed on both factors. Here, :math:`G` are groups of vector components
    (same spline basis in :math:`\eta_3`, coupled at the pole), :math:`\mathbb M^G_{12}[\bar w]` is the
    (block) mass matrix of the poloidal plane with the weights of the 3d mass operator reduced in
    :math:`\eta_3` (see :func:`~struphy.polar.kron_2dx1d.reduced_weight_2d`), and :math:`\mathbb M^G_3`
    is the unweighted 1d mass matrix in :math:`\eta_3`. Couplings between different groups are neglected.

    The approximation is exact for weights that do not depend on :math:`\eta_3` and have no couplings
    between the groups (e.g. a cylinder or an axisymmetric torus with axisymmetric weights); with an exact
    poloidal solver, an iterative solver then converges in one iteration.

    Parameters
    ----------
    mass_operator : WeightedMassOperator
        The weighted mass operator on a polar spline space for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.

    poloidal_solver : str
        Solver for the poloidal matrices: ``"schur"`` (block elimination of the pole block), ``"lu"``
        (sparse LU) or ``"ilu"`` (incomplete LU).

    weight_reduction : str
        How the weights are reduced in eta_3: ``"midpoint"`` (value at 0.5) or ``"average"`` (mean).

    schur_outer : str
        Factorization of the outer block in the Schur solver: ``"lu"`` (exact) or ``"ilu"``.

    drop_tol, fill_factor : float
        Parameters of the incomplete LU factorizations (``scipy.sparse.linalg.spilu``).
    """

    def __init__(
        self,
        mass_operator: WeightedMassOperator,
        apply_bc: bool = True,
        poloidal_solver: str = "schur",
        weight_reduction: str = "midpoint",
        schur_outer: str = "lu",
        drop_tol: float = ILU_DROP_TOL,
        fill_factor: float = ILU_FILL_FACTOR,
    ):
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, "Only square operators can be inverted!"
        if not isinstance(mass_operator.domain, PolarDerhamSpace):
            raise ValueError(
                "PolarMassMatrixPreconditioner needs a mass operator on polar splines, use MassMatrixPreconditioner instead."
            )
        assert weight_reduction in ("midpoint", "average"), f"Unknown weight_reduction {weight_reduction!r}."

        self._mass_operator = mass_operator
        self._space = mass_operator.domain
        self._apply_bc = apply_bc
        self._poloidal_solver = poloidal_solver
        self._weight_reduction = weight_reduction
        self._solver_kwargs = {"schur_outer": schur_outer, "drop_tol": drop_tol, "fill_factor": fill_factor}

        # boundary conditions are only imposed if the mass operator has a BoundaryOperator
        self._bc = _boundary_conditions(mass_operator, apply_bc)
        self._tmp = self._space.zeros() if self._bc is not None else None

        self._solver = self._build_solver()

    def _build_solver(self) -> PolarKron2Dx1DSolver:
        groups = _polar_mass_groups(self._mass_operator, self._weight_reduction)
        if self._bc is not None:
            bases = spline_bases(self._mass_operator.domain_femspace)
            groups = [g.masked(*bc_masks(self._space, g.comps, bases, self._bc), True, self._space) for g in groups]
        return PolarKron2Dx1DSolver(self._space, groups, self._poloidal_solver, **self._solver_kwargs)

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return self._mass_operator.dtype

    @property
    def space(self):
        """PolarDerhamSpace."""
        return self._space

    @property
    def solver(self) -> PolarKron2Dx1DSolver:
        """The 2D x 1D Kronecker solver."""
        return self._solver

    @property
    def exact(self) -> bool:
        """Whether the poloidal solves are exact (incomplete LU factorizations are not)."""
        return self._solver.exact

    def update_mass_operator(self, mass_operator: WeightedMassOperator) -> None:
        """Update the mass operator (same spaces and structure, e.g. new weights) and rebuild the factorizations."""
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == self.domain, "Update needs to have the same domain and codomain"
        self._mass_operator = mass_operator
        self._solver = self._build_solver()

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Applies the approximate inverse to rhs."""
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space
        if self._bc is not None:
            self._mass_operator._V_boundary_op.dot(rhs, out=self._tmp)
            rhs = self._tmp
        return self._solver.dot(rhs, out=out)

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        return self.solve(v, out=out)

    def transpose(self, conjugate: bool = False) -> PolarMassMatrixPreconditioner:
        return PolarMassMatrixPreconditioner(
            self._mass_operator.transpose(),
            self._apply_bc,
            self._poloidal_solver,
            self._weight_reduction,
            **self._solver_kwargs,
        )


def _polar_mass_groups(mass_operator: WeightedMassOperator, reduction: str) -> list[Kron2Dx1DGroup]:
    """Factors (E_pol M_12[w] E_pol^T, M_3) of the 2D x 1D Kronecker approximation of a polar mass matrix."""
    derham = mass_operator.derham
    space = mass_operator.domain
    bases = spline_bases(mass_operator.domain_femspace)
    E_op = derham.extraction_ops[space.space_id]
    weights = mass_operator.weights
    symmetry = mass_operator.symmetry

    groups = []
    for comps in component_groups(space):
        E = poloidal_extraction_matrix(space, comps, E_op.blocks_ten_to_pol, None)

        # poloidal mass matrix (blocks of the group); the weight reduction is collective
        blocks = [[None] * len(comps) for _ in comps]
        for a, c in enumerate(comps):
            for b, c2 in enumerate(comps):
                w2 = reduced_weight_2d(weights[c][c2], derham, reduction)
                if w2 is not None:
                    blocks[a][b] = poloidal_mass_matrix(derham, tuple(bases[c][:2]), tuple(bases[c2][:2]), w2)
        for a, c in enumerate(comps):
            for b, c2 in enumerate(comps):
                if blocks[a][b] is None:
                    if symmetry in ("symm", "asym") and blocks[b][a] is not None:
                        blocks[a][b] = (1.0 if symmetry == "symm" else -1.0) * blocks[b][a].T
                    else:
                        shape_r = _poloidal_size(space, c)
                        shape_c = _poloidal_size(space, c2)
                        blocks[a][b] = sparse.csr_matrix((shape_r, shape_c))
        M12 = sparse.bmat(blocks, format="csr")
        A_pol = (E @ M12 @ E.T).tocsr()

        # unweighted 1d mass matrix in eta_3
        A_3 = xp.to_numpy(_mass_matrix_1d(mass_operator, bases[comps[0]][2], 2, _ones_1d)[0].toarray())

        groups.append(Kron2Dx1DGroup(comps, A_pol, A_3))
    return groups


def _poloidal_size(space: PolarDerhamSpace, c: int) -> int:
    """Number of tensor-product coefficients of component c in the poloidal plane."""
    n1, n2 = tp_space(space, c).npts[:2]
    return n1 * n2


class StiffnessPreconditioner(KroneckerPreconditioner):
    r"""
    Preconditioner for the stabilized stiffness operators

    .. math::

        A = \mathbb d^T \mathbb M_{k+1} \mathbb d + \sigma \, \mathbb M_k \,,
        \qquad \mathbb d \in \{\mathbb G, \mathbb C, \mathbb D\} \ (k = 0, 1, 2)\,,

    i.e. :math:`\mathbb G^T \mathbb M_1 \mathbb G + \sigma \mathbb M_0`,
    :math:`\mathbb C^T \mathbb M_2 \mathbb C + \sigma \mathbb M_1` and
    :math:`\mathbb D^T \mathbb M_3 \mathbb D + \sigma \mathbb M_2`.

    The matrix in the middle (``mid``, default :math:`\mathbb M_{k+1}`) and the stabilization
    (``stab``, default :math:`\mathbb M_k`) can be other weighted mass operators with the same
    domain and codomain (or sums and multiples of them), e.g. ``mid="M1gyro"`` and
    ``stab="M0ad_withT"``; ``stab="Id"`` is the identity. For the gradient, ``stab_average``
    replaces the stabilization :math:`\mathbb S` by :math:`\mathbb S (\mathbb I - \mathbb A)`, where
    :math:`\mathbb A` is the :math:`\mathbb S`-weighted average over the given directions
    (:class:`~struphy.feec.mass.WeightedAverageProjection`), as in the adiabatic gyrokinetic Poisson
    equation.

    With unit weights (the operator on the logical cube), the mass matrices are
    block-diagonal Kronecker products of 1d mass matrices :math:`M_d`, and each diagonal block of
    :math:`A` is a sum of Kronecker products,

    .. math::

        A_{cc} \approx \sum_{d \in N_c} M_1 \otimes \dots \otimes S_d \otimes \dots \otimes M_3
            + \sigma \, M_1 \otimes M_2 \otimes M_3 \,,

    with the 1d stiffness matrices :math:`S_d = \mathbb d_d^T M^D_d \mathbb d_d` in the directions
    :math:`N_c` in which component c is differentiated (its B-spline directions). These blocks
    are inverted exactly by fast diagonalization
    (:class:`~feectools.linalg.kron.KroneckerSumSolver`). For the gradient this is the exact
    inverse on the logical cube. For curl and div, the off-diagonal blocks are neglected
    (block Jacobi, :math:`P_{BJ}`).

    Block Jacobi overestimates :math:`A` on the kernel of :math:`\mathbb d` (gradients for
    curl, curls for div), where :math:`A = \sigma \mathbb M_k`. With ``kernel_correction``
    (default, needs :math:`\sigma > 0`), the preconditioner is

    .. math::

        P = P_{BJ} + \sigma^{-1} \, \mathbb d_- \, P_- \, \mathbb d_-^T \,,

    where :math:`\mathbb d_-` is the previous derivative (:math:`\mathbb G` for curl,
    :math:`\mathbb C` for div) and :math:`P_-` approximates the inverse of
    :math:`\mathbb d_-^T \mathbb M_k \mathbb d_-` (the grad preconditioner for curl, the curl
    block Jacobi for div). On the range of :math:`\mathbb d_-^T` the kernel of
    :math:`\mathbb d_-` does not matter, since :math:`\mathbb G^T \mathbb C^T = 0`.

    The geometry is included with ``weights="average"`` (default) and by diagonal scaling
    (default, see :class:`KroneckerPreconditioner`). For the weights, fast diagonalization needs the same 1d mass matrix in
    each direction for all terms, so the weights :math:`w` of the mass matrices are approximated
    by one common separable shape :math:`\prod_d \phi_d(\eta_d)` (mean 1; :math:`\phi_d` is the mean
    over the other directions, averaged over the diagonal blocks of :math:`\mathbb M_{k+1}`), times
    the mean of the weight of each term (the block of :math:`\mathbb M_{k+1}` for the stiffness
    term of each direction, the block of :math:`\mathbb M_k` for the mass term). This is exact
    e.g. for constant but anisotropic weights. A block of ``mid`` with zero mean (e.g. the
    parallel direction of a perpendicular operator) gives no stiffness term in that direction; its
    1d stiffness still defines the eigenbasis of the fast diagonalization.

    The averaged stabilization :math:`\mathbb S (\mathbb I - \mathbb A)` is approximated by
    :math:`\bar s \, (M_1 \otimes M_2 \otimes M_3 - F_1 \otimes F_2 \otimes F_3)` with
    :math:`F_d = c_d c_d^T / (\mathbb 1^T c_d)`, :math:`c_d = M_d \mathbb 1`, in the averaged
    directions (:math:`F_d = M_d` otherwise). In directions with :math:`S_d \mathbb 1 = 0`
    (periodic or without essential boundary conditions) the constants are an eigenvector of the
    fast diagonalization and this term is exact in its eigenbasis (it removes the averaged modes).
    The identity (``stab="Id"``) is replaced by its diagonal in that eigenbasis; its average is
    Euclidean, i.e. :math:`M_d` is replaced by the identity and :math:`c_d = \mathbb 1`.

    **General operators.** Instead of a derivative, ``derivative`` can be a
    :class:`~struphy.feec.kronecker_block.KroneckerBlockOperator` :math:`\mathbb L` (blocks that are sums of
    Kronecker products of 1d matrices), e.g. :math:`\mathbb L = \mathbb C \, \tilde{\mathbb T}` with the Kronecker
    approximation :math:`\tilde{\mathbb T}` of a basis projection operator
    (:func:`~struphy.feec.kronecker_block.basis_projection_operator`). The preconditioner then approximates

    .. math::

        A = \mathbb L^T \mathbb M_{\textrm{mid}} \mathbb L + \sigma \, \mathbb S \,,

    with ``mid`` (default: the mass operator of the codomain of :math:`\mathbb L`) and ``stab`` (default: of its
    domain). The weights of ``mid`` and ``stab`` are approximated by separable shapes (separately), times the
    mean of each block; the off-diagonal blocks of ``mid`` are neglected. Each diagonal block of :math:`A` is then a
    sum of Kronecker products :math:`\sum_t c_t F_{t,1} \otimes F_{t,2} \otimes F_{t,3} + \sigma_c M_1 \otimes M_2
    \otimes M_3` (block Jacobi in the components of the domain). It is inverted in the basis
    :math:`U_1 \otimes U_2 \otimes U_3` of the generalized eigenproblems :math:`\hat S_d U_d = M_d U_d \Lambda_d` of
    the stabilization :math:`M_d` and a summary stiffness
    :math:`\hat S_d = \sum_t |c_t| F_{t,d} \prod_{e \neq d} \textrm{tr}(M_e^{-1} F_{t,e}) / n_e` (over the
    symmetric terms), where every term is replaced by its diagonal in that basis. This is the diagonal of
    the Kronecker approximation of :math:`A` in the eigenbasis, hence positive. It is exact if the eigenbasis
    diagonalizes all terms, e.g. for derivatives as above, or for circulant 1d factors (periodic directions with
    constant weights); otherwise, e.g. for the mixed terms of a field-aligned operator, it is an approximation.
    ``diagonal_scaling``, ``kernel_correction`` and ``stab_average`` are not supported (the diagonal of the exact
    operator is not available, since basis projection operators contain inverse inter-/histopolation matrices).

    With polar splines (``mass_ops.derham.polar_splines``), the constructor returns a
    :class:`PolarStiffnessPreconditioner` instead (``grad`` and ``curl`` only; ``diagonal_scaling``
    and ``weights`` are ignored there, and its own options such as ``poloidal_solver`` can be passed
    as keyword arguments).

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        The mass operators :math:`\mathbb M_k`.

    derivative : str | KroneckerBlockOperator | list[KroneckerBlockOperator]
        ``"grad"``, ``"curl"`` or ``"div"``, or a general operator :math:`\mathbb L` (see above), or a list of general
        operators :math:`\mathbb L_i` with the same domain, for :math:`A = \sum_i \mathbb L_i^T \mathbb M_{\textrm{mid},i}
        \mathbb L_i + \sigma \mathbb S` (then ``mid`` is a list as well, or one entry for all).

    sigma : float
        Coefficient of the mass term.

    apply_bc : bool
        Whether to include boundary operators.

    diagonal_scaling : bool
        Whether to correct the approximation with the diagonals of :math:`A` and of its
        approximation.

    kernel_correction : bool
        For curl and div: whether to add the correction on the kernel of the derivative (needs
        ``sigma > 0``). Ignored for grad.

    weights : str
        ``"average"`` (separable approximation of the weights, see above; default) or
        ``"unit"`` (operator on the logical cube).

    mid : str | LinearOperator | None | list
        Matrix in the middle (a list for a list of general operators): the name of a mass operator of ``mass_ops`` (e.g. ``"M1gyro"``,
        ``"M1perp"``) or the operator itself (a WeightedMassOperator, or a sum or multiple of such),
        with the domain and codomain of :math:`\mathbb M_{k+1}`. Default :math:`\mathbb M_{k+1}`.

    stab : str | LinearOperator | None
        Stabilization: the name of a mass operator of ``mass_ops`` (e.g. ``"M0ad"``,
        ``"M0ad_withT"``), the operator itself (as for ``mid``), with the domain and codomain of
        :math:`\mathbb M_k`, or ``"Id"`` for the identity. Default :math:`\mathbb M_k`.

    stab_average : str | None
        Gradient only: the directions averaged in the stabilization :math:`\mathbb S (\mathbb I - \mathbb A)`,
        e.g. ``"eta3"``, ``"eta2 eta3"`` or ``"eta1 eta2 eta3"`` (global mean); None (default) for no averaging.
    """

    _FORMS = {"grad": 0, "curl": 1, "div": 2}

    def __new__(cls, mass_ops: WeightedMassOperators | None = None, *args, **kwargs):
        # polar splines: the 2D x 1D Kronecker version
        if cls is StiffnessPreconditioner and mass_ops is not None and mass_ops.derham.polar_splines:
            return _polar_stiffness_preconditioner(mass_ops, *args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        derivative: str = "grad",
        sigma: float = 0.0,
        apply_bc: bool = True,
        diagonal_scaling: bool = True,
        kernel_correction: bool = True,
        weights: str = "average",
        mid: str | LinearOperator | None = None,
        stab: str | LinearOperator | None = None,
        stab_average: str | None = None,
    ):
        derham = mass_ops.derham
        ops = list(derivative) if isinstance(derivative, (list, tuple)) else [derivative]
        self._general = all(isinstance(op, KroneckerBlockOperator) for op in ops)
        if self._general:
            with ProfileManager.profile_region("StiffnessPreconditioner: general L_i"):
                # general operators L_i: no diagonal scaling, kernel correction or averaged stabilization
                if stab_average is not None:
                    raise NotImplementedError("stab_average is not supported for a general operator.")
                diagonal_scaling = False
                kernel_correction = False
                assert len({op.domain_space for op in ops}) == 1, "The operators L_i must have the same domain."
                mids = list(mid) if isinstance(mid, (list, tuple)) else [mid] * len(ops)
                assert len(mids) == len(ops), "Give one mid per operator L_i."
                form_dom = derham.space_to_form[ops[0].domain_space]
                form_mid = derham.space_to_form[ops[0].codomain_space]
                # pairs (L_i, terms of mid_i)
                self._general_pairs = [
                    (
                        op,
                        _mass_terms(
                            _resolve_mass_operator(
                                mass_ops, m, getattr(mass_ops, f"M{derham.space_to_form[op.codomain_space]}")
                            )
                        ),
                    )
                    for op, m in zip(ops, mids)
                ]
                for op, mid_terms in self._general_pairs:
                    ref = getattr(mass_ops, f"M{derham.space_to_form[op.codomain_space]}")
                    for _, t in mid_terms:
                        assert t.domain == ref.domain and t.codomain == ref.codomain, (
                            f"Each mid must have the domain and codomain of the mass operator of the codomain "
                            f"({op.codomain_space}) of its operator."
                        )
                mid = mids[0]
        elif len(ops) > 1 or not isinstance(derivative, str):
            raise TypeError("derivative must be 'grad', 'curl', 'div', a KroneckerBlockOperator or a list of these.")
        else:
            assert derivative in self._FORMS, f"derivative must be one of {tuple(self._FORMS)}, got {derivative!r}."
            k = self._FORMS[derivative]
            form_dom, form_mid = str(k), str(k + 1)
        assert sigma >= 0.0
        assert weights in ("unit", "average"), f"weights must be 'unit' or 'average', got {weights!r}."
        kernel_correction = kernel_correction and derivative != "grad"
        if kernel_correction and sigma == 0.0:
            raise ValueError("The kernel correction needs sigma > 0 (or kernel_correction=False).")

        if derham.polar_splines:
            raise NotImplementedError("Use PolarStiffnessPreconditioner for polar splines.")

        self._mass_ops = mass_ops
        self._derivative = derivative
        self._sigma = sigma
        self._weights = weights

        # matrix in the middle and stabilization, as sums of weighted mass operators
        with ProfileManager.profile_region("StiffnessPreconditioner: resolve mass operators"):
            mass_mid_default = getattr(mass_ops, f"M{form_mid}")
            self._mid = mid
            self._mid_terms = _mass_terms(_resolve_mass_operator(mass_ops, mid, mass_mid_default))
            self._stab = stab
            self._stab_is_identity = isinstance(stab, str) and stab == "Id"
            mass_k = getattr(mass_ops, f"M{form_dom}")
            self._stab_terms = (
                None if self._stab_is_identity else _mass_terms(_resolve_mass_operator(mass_ops, stab, mass_k))
            )
        for terms, ref, name in ((self._mid_terms, mass_mid_default, "mid"), (self._stab_terms or [], mass_k, "stab")):
            for _, op in terms:
                assert op.domain == ref.domain and op.codomain == ref.codomain, (
                    f"{name} must have the domain and codomain of M{form_mid if name == 'mid' else form_dom}."
                )

        # averaged directions of the stabilization (gradient only)
        self._stab_average = stab_average
        self._average_dirs = _parse_directions(stab_average)
        if self._average_dirs and derivative != "grad":
            raise NotImplementedError("stab_average is only implemented for derivative='grad'.")

        # tensor-product derivative without boundary operators (no polar splines), or the general operator
        self._d = derivative if self._general else getattr(derham, f"{derivative}_bcfree")

        with ProfileManager.profile_region("KroneckerPreconditioner init", functions=[KroneckerPreconditioner.__init__]):
            super().__init__(mass_k, apply_bc=apply_bc, diagonal_scaling=diagonal_scaling)

        # correction on the kernel of the derivative: sigma^{-1} d_prev P_prev d_prev^T,
        # where A = sigma * stab; P_prev approximates the inverse of d_prev^T stab d_prev
        self._kernel_correction = None
        if kernel_correction:
            with ProfileManager.profile_region("StiffnessPreconditioner: kernel correction"):
                prev = "grad" if derivative == "curl" else "curl"
                d_prev = getattr(derham, prev)
                P_prev = StiffnessPreconditioner(
                    mass_ops,
                    prev,
                    apply_bc=apply_bc,
                    diagonal_scaling=diagonal_scaling,
                    kernel_correction=False,
                    weights=weights,
                    mid=None if self._stab_is_identity else stab,
                )
                self._kernel_correction = (1.0 / sigma) * (d_prev @ P_prev @ d_prev.T)
                self._tmp_kernel = self.codomain.zeros()

    @property
    def derivative(self) -> str | KroneckerBlockOperator:
        """``"grad"``, ``"curl"`` or ``"div"``, or the general operator."""
        return self._derivative

    @property
    def sigma(self) -> float:
        """Coefficient of the mass term."""
        return self._sigma

    @property
    def weights(self) -> str:
        """``"unit"`` or ``"average"``."""
        return self._weights

    @property
    def kernel_correction(self) -> LinearOperator | None:
        """The correction on the kernel of the derivative, or None."""
        return self._kernel_correction

    @property
    def mid(self) -> str | LinearOperator | None:
        """The matrix in the middle as given (None for the default :math:`\\mathbb M_{k+1}`)."""
        return self._mid

    @property
    def stab(self) -> str | LinearOperator | None:
        """The stabilization as given (None for the default :math:`\\mathbb M_k`, ``"Id"`` for the identity)."""
        return self._stab

    @property
    def stab_average(self) -> str | None:
        """The directions averaged in the stabilization, or None."""
        return self._stab_average

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Apply the preconditioner (block Jacobi part plus kernel correction, if any) to rhs."""
        out = super().solve(rhs, out=out)
        if self._kernel_correction is not None:
            out += self._kernel_correction.dot(rhs, out=self._tmp_kernel)
        return out

    @property
    def core_operator(self) -> LinearOperator:
        """The core operator :math:`\\mathbb d^T \\mathbb M_{\\textrm{mid}} \\mathbb d + \\sigma \\mathbb S (\\mathbb I - \\mathbb A)`
        on the tensor-product spaces (without the average if ``stab_average`` is None)."""
        return self._core(with_average=True)

    def _core(self, with_average: bool) -> LinearOperator:
        if self._general:
            raise NotImplementedError("The core operator of a general (Kronecker) operator is not available.")
        A = self._d.T @ _terms_matrix(self._mid_terms) @ self._d
        if self._sigma != 0.0:
            space = self._mass_operator.matrix.domain
            S = IdentityOperator(space) if self._stab_is_identity else _terms_matrix(self._stab_terms)
            if self._average_dirs and with_average:
                avg = WeightedAverageProjection(self._mass_ops.derham, S, self._average_dirs)
                S = S @ (IdentityOperator(space) - avg)
            A = A + self._sigma * S
        return A

    def _build_approximation(self, bc):
        if self._general:
            return self._build_general_approximation(bc)

        femspace = self._femspace
        is_scalar = isinstance(femspace, TensorFemSpace)
        comps = (femspace,) if is_scalar else femspace.spaces

        # directions in which each component is differentiated, and the codomain block
        stiff_dirs = _derivative_directions(self._d)

        # weights: common shape per direction and means per block (unit or separable average)
        derham = self._mass_operator.derham
        n_mid = len(self._mid_terms[0][1].weights)
        n_dom = len(self._mass_operator.weights)
        if self._weights == "average":
            shapes, mid_means = _separable_weight(_block_weights(self._mid_terms, n_mid), derham)
            if self._stab_is_identity:
                dom_means = [1.0] * n_dom
            else:
                _, dom_means = _separable_weight(_block_weights(self._stab_terms, n_dom), derham)
        else:
            shapes, mid_means, dom_means = [_ones_1d] * 3, [1.0] * n_mid, [1.0] * n_dom

        matrixblocks = []
        solverblocks = []
        scalingblocks = []
        for c, comp in enumerate(comps):
            coeff_space = femspace.coeff_space if is_scalar else femspace.coeff_space[c]

            stiffness, coeffs, mass, local_mass, local_stiffness = [], [], [], [], []
            identity, average = [], []
            for d in range(3):
                basis = comp.spaces[d].basis
                M_d, domain_decomposition = _dense_mass_1d(self._mass_operator, basis, d, shapes[d])

                # unscaled 1d stiffness (defines the eigenbasis also for a zero coefficient)
                S_d = None
                if d in stiff_dirs[c]:
                    assert basis == "B", "Only B-spline directions are differentiated."
                    M_D, _ = _dense_mass_1d(self._mass_operator, "M", d, shapes[d])
                    D_d = _dense_derivative_1d(self._d, stiff_dirs[c][d], c)
                    assert D_d.shape == (M_D.shape[0], M_d.shape[0])
                    S_d = D_d.T @ M_D @ D_d
                coeffs.append(mid_means[stiff_dirs[c][d]] if S_d is not None else 1.0)

                if bc is not None and basis == "B":
                    M_d = _apply_bc_dense(M_d, bc[d])
                    S_d = None if S_d is None else _apply_bc_dense(S_d, bc[d])

                # 1d factors of the stabilization terms (None for M_d); the average is weighted with
                # M_d, or Euclidean (1 1^T / n) for the identity stabilization
                identity.append(np.eye(M_d.shape[0]))
                if d in self._average_dirs:
                    c_d = np.ones(M_d.shape[0]) if self._stab_is_identity else M_d @ np.ones(M_d.shape[0])
                    average.append(np.outer(c_d, c_d) / c_d.sum())
                else:
                    average.append(None)

                stiffness.append(S_d)
                mass.append(M_d)
                local_mass.append(_process_local_matrix_1d(M_d, comp.coeff_space, d, domain_decomposition))
                local_stiffness.append(
                    None
                    if S_d is None
                    else _process_local_matrix_1d(coeffs[d] * S_d, comp.coeff_space, d, domain_decomposition)
                )

            # stabilization: sigma_c * (M_1 x M_2 x M_3 - F_1 x F_2 x F_3), or the identity
            sigma_c = self._sigma * dom_means[c]
            terms = []
            if self._stab_is_identity and sigma_c != 0.0:
                terms.append((sigma_c, identity))
            if self._average_dirs and sigma_c != 0.0:
                factors = (
                    [average[d] if average[d] is not None else identity[d] for d in range(3)]
                    if self._stab_is_identity
                    else average
                )
                terms.append((-sigma_c, factors))
            sigma_mass = 0.0 if self._stab_is_identity else sigma_c

            solverblocks.append(
                KroneckerSumSolver(
                    coeff_space, stiffness, mass, sigma=sigma_mass, stiffness_coeffs=coeffs, mass_terms=terms
                )
            )
            dense_terms = [
                (coeff, [mass[d] if F is None else F for d, F in enumerate(factors)]) for coeff, factors in terms
            ]
            matrixblocks.append(_kronecker_sum(coeff_space, local_stiffness, local_mass, sigma_mass, dense_terms))
            # the average term is the last one; it is left out in the diagonal scaling
            n_scaling = len(dense_terms) - (1 if self._average_dirs and sigma_c != 0.0 else 0)
            scalingblocks.append(
                _kronecker_sum(coeff_space, local_stiffness, local_mass, sigma_mass, dense_terms[:n_scaling])
            )

        if is_scalar:
            self._matrix_without_average = scalingblocks[0]
            return matrixblocks[0], solverblocks[0]
        self._matrix_without_average = _block_diagonal(femspace.coeff_space, scalingblocks)
        return (
            _block_diagonal(femspace.coeff_space, matrixblocks),
            _block_diagonal(femspace.coeff_space, solverblocks),
        )

    def _build_general_approximation(self, bc):
        """Block Jacobi approximation of L^T M_mid L + sigma S for a general KroneckerBlockOperator L (see the class
        docstring): the Kronecker terms of each diagonal block, inverted approximately by fast diagonalization."""
        femspace = self._femspace
        is_scalar = isinstance(femspace, TensorFemSpace)
        comps = (femspace,) if is_scalar else femspace.spaces
        derham = self._mass_operator.derham
        n_dom = len(self._mass_operator.weights)

        # separable weights of stab (shapes per direction, means per block); unit weights keep the coefficients of the
        # terms (e.g. of a scaled mass operator)
        if self._stab_is_identity:
            stab_shapes, dom_means = [_ones_1d] * 3, [1.0] * n_dom
        elif self._weights == "average":
            stab_shapes, dom_means = _separable_weight(_block_weights(self._stab_terms, n_dom), derham)
        else:
            stab_shapes, dom_means = [_ones_1d] * 3, _unit_means(self._stab_terms, n_dom)

        # per operator L_i: its blocks, the means of the blocks of mid_i and their 1d mass matrices (with the essential
        # boundary conditions of mid_i as zero rows and columns)
        pairs = []
        for L, mid_terms in self._general_pairs:
            assert L.domain_shapes == [tuple(V.nbasis for V in comp.spaces) for comp in comps]
            mid_op = mid_terms[0][1]
            n_mid = len(mid_op.weights)
            if self._weights == "average":
                mid_shapes, mid_means = _separable_weight(_block_weights(mid_terms, n_mid), derham)
            else:
                mid_shapes, mid_means = [_ones_1d] * 3, _unit_means(mid_terms, n_mid)
            mid_femspace = mid_op.domain_femspace
            mid_comps = (mid_femspace,) if isinstance(mid_femspace, TensorFemSpace) else mid_femspace.spaces
            mid_bc = _boundary_conditions(mid_op, bc is not None)
            mid_1d = []
            for r, comp in enumerate(mid_comps):
                mats = []
                for d in range(3):
                    basis = comp.spaces[d].basis
                    M_d, _ = _dense_mass_1d(mid_op, basis, d, mid_shapes[d])
                    if mid_bc is not None and _has_essential_bc(mid_op, basis, r, d):
                        M_d = _zero_bc_dense(M_d, mid_bc[d])
                    mats.append(M_d)
                mid_1d.append(mats)
            pairs.append((L, mid_means, mid_1d))

        matrixblocks = []
        solverblocks = []
        for c, comp in enumerate(comps):
            coeff_space = femspace.coeff_space if is_scalar else femspace.coeff_space[c]

            # stabilization (defines the eigenbasis together with the summary stiffness)
            mass, local_mass, has_bc = [], [], []
            for d in range(3):
                basis = comp.spaces[d].basis
                M_d, domain_decomposition = _dense_mass_1d(self._mass_operator, basis, d, stab_shapes[d])
                if self._stab_is_identity:
                    M_d = np.eye(M_d.shape[0])
                essential = bc is not None and _has_essential_bc(self._mass_operator, basis, c, d)
                has_bc.append(essential)
                if essential:
                    M_d = _apply_bc_dense(M_d, bc[d])
                mass.append(M_d)
                local_mass.append(_process_local_matrix_1d(M_d, comp.coeff_space, d, domain_decomposition))

            # terms of the diagonal block: sum_i sum_r sum_{a, b} c_a c_b mean_{i,r} F_a^T M_{mid_i,r} F_b
            terms, symmetric = [], []
            for L, mid_means, mid_1d in pairs:
                for r in range(len(mid_means)):
                    block = L.blocks.get((r, c), [])
                    for i, (ca, Fa) in enumerate(block):
                        for j, (cb, Fb) in enumerate(block):
                            coeff = mid_means[r] * ca * cb
                            if coeff == 0.0:
                                continue
                            factors = []
                            for d in range(3):
                                M_rd = mid_1d[r][d]
                                left = M_rd if Fa[d] is None else Fa[d].T @ M_rd
                                F = left if Fb[d] is None else left @ Fb[d]
                                if has_bc[d]:
                                    F = _zero_bc_dense(F, bc[d])
                                factors.append(F)
                            terms.append((coeff, factors))
                            symmetric.append(i == j)

            # summary stiffness per direction (from the symmetric terms), defining the eigenbasis
            stiffness = []
            for d in range(3):
                S_d = np.zeros_like(mass[d])
                for (coeff, factors), sym in zip(terms, symmetric):
                    if sym:
                        scale = abs(coeff)
                        for e in range(3):
                            if e != d:
                                scale *= np.trace(np.linalg.solve(mass[e], factors[e])) / mass[e].shape[0]
                        S_d += scale * factors[d]
                S_d = 0.5 * (S_d + S_d.T)
                stiffness.append(S_d if np.any(S_d) else None)

            sigma_c = self._sigma * dom_means[c]
            assert terms or sigma_c != 0.0, f"The diagonal block {c} of the operator vanishes."
            solverblocks.append(
                KroneckerSumSolver(
                    coeff_space, stiffness, mass, sigma=sigma_c, stiffness_coeffs=[0.0] * 3, mass_terms=terms
                )
            )
            matrixblocks.append(_kronecker_sum(coeff_space, [None] * 3, local_mass, sigma_c, terms))

        if is_scalar:
            return matrixblocks[0], solverblocks[0]
        return (
            _block_diagonal(femspace.coeff_space, matrixblocks),
            _block_diagonal(femspace.coeff_space, solverblocks),
        )

    def _core_diagonal(self):
        # without the average: it couples all degrees of freedom along the averaged directions,
        # which probing cannot resolve; its diagonal is small (O(1/n)) and left out on both sides
        # (see _scaling_matrix), which keeps the preconditioner exact on the logical cube
        space = self._matrix.domain
        return _diagonal_operator(space, _probe_diagonal(self._core(with_average=False), space))

    def _scaling_matrix(self) -> LinearOperator:
        return self._matrix_without_average

    def transpose(self, conjugate: bool = False) -> StiffnessPreconditioner:
        """The operator is symmetric, so is the preconditioner."""
        return self


def _polar_stiffness_preconditioner(
    mass_ops: WeightedMassOperators,
    derivative: str = "grad",
    sigma: float = 0.0,
    apply_bc: bool = True,
    diagonal_scaling: bool = True,
    kernel_correction: bool = True,
    weights: str = "average",
    mid: str | LinearOperator | None = None,
    stab: str | LinearOperator | None = None,
    stab_average: str | None = None,
    **polar_kwargs,
) -> PolarStiffnessPreconditioner:
    """The PolarStiffnessPreconditioner for the arguments of StiffnessPreconditioner (plus its own options)."""
    return PolarStiffnessPreconditioner(
        mass_ops,
        derivative,
        sigma,
        apply_bc=apply_bc,
        kernel_correction=kernel_correction,
        mid=mid,
        stab=stab,
        stab_average=stab_average,
        **polar_kwargs,
    )


class PolarStiffnessPreconditioner(LinearOperator):
    r"""
    Preconditioner for the stabilized stiffness operators :math:`A = \mathbb d^T \mathbb M_{k+1} \mathbb d + \sigma \mathbb M_k`
    (:math:`\mathbb d \in \{\mathbb G, \mathbb C\}`, see :class:`StiffnessPreconditioner`) on polar spline spaces,
    exploiting the 2D x 1D Kronecker structure of the polar extraction (see :mod:`struphy.polar.kron_2dx1d`).

    With the polar extraction :math:`\mathbb E` (:math:`\mathbb E_{k+1}^T \mathbb d = \mathbb d_\textnormal{tp} \mathbb E_k^T`),
    :math:`A = \mathbb E_k (\mathbb d_\textnormal{tp}^T \mathbb M^\textnormal{tp}_{k+1} \mathbb d_\textnormal{tp} + \sigma \mathbb M^\textnormal{tp}_k) \mathbb E_k^T`.
    The blocks of the tensor-product derivative :math:`\mathbb d_\textnormal{tp}` differentiate either in the
    poloidal plane (identity in :math:`\eta_3`) or in :math:`\eta_3` (identity in the poloidal plane). With the weights of
    the mass matrices reduced in :math:`\eta_3` (see :func:`~struphy.polar.kron_2dx1d.reduced_weight_2d`), each group :math:`G`
    of components (see :func:`~struphy.polar.kron_2dx1d.component_groups`) is approximated by

    .. math::

        A_G \approx \left(\mathbb K_\textnormal{pol} + \sigma \mathbb M_\textnormal{pol}\right) \otimes \mathbb M_3
            + \mathbb W_\textnormal{pol} \otimes \mathbb S_3 \,,

    where :math:`\mathbb K_\textnormal{pol}` collects the terms with poloidal derivatives on both sides, :math:`\mathbb W_\textnormal{pol}`
    those with :math:`\eta_3` derivatives on both sides (:math:`\mathbb S_3 = \mathbb D_3^T \mathbb M^D_3 \mathbb D_3`), and
    :math:`\mathbb M_\textnormal{pol}` is the poloidal stabilization (all with the polar extraction applied). It is inverted by
    fast diagonalization in :math:`\eta_3` and one poloidal solve per :math:`\eta_3` mode (:class:`~struphy.polar.kron_2dx1d.PolarKronSum2Dx1DSolver`).
    Neglected are the mixed terms (poloidal derivative on one side, :math:`\eta_3` derivative on the other), which vanish if the
    weights do not couple the poloidal plane and :math:`\eta_3` (e.g. a cylinder or an axisymmetric torus without straight field
    line coordinates), and the couplings between the groups. For the gradient, the preconditioner is then the exact inverse
    of :math:`A` for weights that do not depend on :math:`\eta_3`. For the curl, the groups :math:`(E_1, E_2)` and :math:`E_3` are
    decoupled (block Jacobi) and, with ``kernel_correction``, the correction :math:`\sigma^{-1} \mathbb G P_\textnormal{grad} \mathbb G^T`
    on the kernel of the curl is added (see :class:`StiffnessPreconditioner`). The identity stabilization (``stab="Id"``) is
    approximated by its diagonal in the eigenbasis of the fast diagonalization.

    For the gradient, ``stab_average`` replaces the stabilization :math:`\mathbb S` by :math:`\mathbb S (\mathbb I - \Pi_{\mathbb S})`
    (:class:`~struphy.feec.mass.PolarWeightedAverageProjection`, averages over :math:`\eta_2` and/or :math:`\eta_3`). With
    :math:`\mathbb S \approx \mathbb S_\textnormal{pol} \otimes \mathbb M_3`, the average over :math:`\eta_3` gives the 1d factor
    :math:`F_3 = c_3 c_3^\top / (\mathbb 1^\top c_3)`, :math:`c_3 = \mathbb M_3 \mathbb 1`, which acts only on the constant mode of the
    fast diagonalization (exact for periodic :math:`\eta_3`), and the average over :math:`\eta_2` the low-rank poloidal factor
    :math:`\mathbb S_\textnormal{pol} \mathbb B (\mathbb B^\top \mathbb S_\textnormal{pol} \mathbb B)^{-1} \mathbb B^\top \mathbb S_\textnormal{pol}`, which is
    included in the poloidal solves by the Woodbury identity. With weights that do not depend on :math:`\eta_3`, the
    preconditioner thus stays exact (except for ``stab="Id"``, whose average is Euclidean).

    The divergence is not implemented yet.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        The mass operators :math:`\mathbb M_k` (on polar splines).

    derivative : str
        ``"grad"`` or ``"curl"``.

    sigma : float
        Coefficient of the mass term.

    apply_bc : bool
        Whether to include boundary operators.

    kernel_correction : bool
        For curl: whether to add the correction on the kernel of the curl (needs ``sigma > 0``).

    mid, stab : str | LinearOperator | None
        Matrix in the middle and stabilization, see :class:`StiffnessPreconditioner`.

    stab_average : str | None
        Gradient only: the directions averaged in the stabilization, ``"eta3"``, ``"eta2"`` or ``"eta2 eta3"``
        (see :class:`StiffnessPreconditioner`); None (default) for no averaging.

    poloidal_solver : str
        Solver for the poloidal matrices: ``"schur"`` (block elimination of the pole block), ``"lu"`` (sparse LU)
        or ``"ilu"`` (incomplete LU). Incomplete LU makes the preconditioner non-symmetric; for curl with small
        ``sigma`` the poloidal matrices are nearly singular on the gradients, and PCG may fail. Use an exact
        solver there.

    weight_reduction : str
        How the weights are reduced in eta_3: ``"midpoint"`` (value at 0.5) or ``"average"`` (mean).

    schur_outer : str
        Factorization of the outer block in the Schur solver: ``"lu"`` (exact) or ``"ilu"`` (same caveat as
        ``poloidal_solver="ilu"``).

    drop_tol, fill_factor : float
        Parameters of the incomplete LU factorizations (``scipy.sparse.linalg.spilu``).
    """

    _FORMS = {"grad": 0, "curl": 1}

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        derivative: str = "grad",
        sigma: float = 0.0,
        apply_bc: bool = True,
        kernel_correction: bool = True,
        mid: str | LinearOperator | None = None,
        stab: str | LinearOperator | None = None,
        stab_average: str | None = None,
        poloidal_solver: str = "schur",
        weight_reduction: str = "midpoint",
        schur_outer: str = "lu",
        drop_tol: float = ILU_DROP_TOL,
        fill_factor: float = ILU_FILL_FACTOR,
    ):
        if derivative == "div":
            raise NotImplementedError("PolarStiffnessPreconditioner is not implemented for derivative='div' yet.")
        assert derivative in self._FORMS, f"derivative must be one of {tuple(self._FORMS)}, got {derivative!r}."
        assert sigma >= 0.0
        assert weight_reduction in ("midpoint", "average"), f"Unknown weight_reduction {weight_reduction!r}."
        kernel_correction = kernel_correction and derivative != "grad"
        if kernel_correction and sigma == 0.0:
            raise ValueError("The kernel correction needs sigma > 0 (or kernel_correction=False).")

        derham = mass_ops.derham
        if not derham.polar_splines:
            raise ValueError("PolarStiffnessPreconditioner needs polar splines, use StiffnessPreconditioner instead.")

        k = self._FORMS[derivative]
        self._mass_ops = mass_ops
        self._derivative = derivative
        self._sigma = sigma
        self._mass_k = getattr(mass_ops, f"M{k}")
        self._space = self._mass_k.domain
        self._poloidal_solver = poloidal_solver
        self._solver_kwargs = {"schur_outer": schur_outer, "drop_tol": drop_tol, "fill_factor": fill_factor}

        # matrix in the middle and stabilization, as sums of weighted mass operators
        mass_mid_default = getattr(mass_ops, f"M{k + 1}")
        mid_terms = _mass_terms(_resolve_mass_operator(mass_ops, mid, mass_mid_default))
        stab_is_identity = isinstance(stab, str) and stab == "Id"
        stab_terms = None if stab_is_identity else _mass_terms(_resolve_mass_operator(mass_ops, stab, self._mass_k))
        for terms, ref, name in ((mid_terms, mass_mid_default, "mid"), (stab_terms or [], self._mass_k, "stab")):
            for _, op in terms:
                assert op.domain == ref.domain and op.codomain == ref.codomain, (
                    f"{name} must have the domain and codomain of M{k + 1 if name == 'mid' else k}."
                )

        # averaged directions of the stabilization (gradient only)
        self._stab_average = stab_average
        average_dirs = _parse_directions(stab_average)
        if average_dirs and derivative != "grad":
            raise NotImplementedError("stab_average is only implemented for derivative='grad'.")
        if 0 in average_dirs:
            raise NotImplementedError("Averages over eta1 (across the pole) are not implemented for polar splines.")

        groups = _polar_stiffness_groups(
            mass_ops, derivative, sigma, mid_terms, stab_terms, stab_is_identity, weight_reduction, average_dirs
        )

        # boundary conditions are only imposed if the mass operator has a BoundaryOperator
        self._bc = _boundary_conditions(self._mass_k, apply_bc)
        if self._bc is not None:
            bases = spline_bases(self._mass_k.domain_femspace)
            for g in groups:
                g.free_pol, g.free_3 = bc_masks(self._space, g.comps, bases, self._bc)
        self._tmp = self._space.zeros() if self._bc is not None else None

        self._solver = PolarKronSum2Dx1DSolver(self._space, groups, poloidal_solver, **self._solver_kwargs)

        # correction on the kernel of the curl: sigma^{-1} G P_grad G^T
        self._kernel_correction = None
        if kernel_correction:
            P_prev = PolarStiffnessPreconditioner(
                mass_ops,
                "grad",
                apply_bc=apply_bc,
                kernel_correction=False,
                mid=None if stab_is_identity else stab,
                poloidal_solver=poloidal_solver,
                weight_reduction=weight_reduction,
                **self._solver_kwargs,
            )
            d_prev = derham.grad
            self._kernel_correction = (1.0 / sigma) * (d_prev @ P_prev @ d_prev.T)
            self._tmp_kernel = self._space.zeros()

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return self._mass_k.dtype

    @property
    def derivative(self) -> str:
        """``"grad"`` or ``"curl"``."""
        return self._derivative

    @property
    def sigma(self) -> float:
        """Coefficient of the mass term."""
        return self._sigma

    @property
    def stab_average(self) -> str | None:
        """The directions averaged in the stabilization, or None."""
        return self._stab_average

    @property
    def solver(self) -> PolarKronSum2Dx1DSolver:
        """The 2D x 1D fast-diagonalization solver (without the kernel correction)."""
        return self._solver

    @property
    def kernel_correction(self) -> LinearOperator | None:
        """The correction on the kernel of the curl, or None."""
        return self._kernel_correction

    @property
    def exact(self) -> bool:
        """Whether the poloidal solves are exact (incomplete LU factorizations are not)."""
        return self._solver.exact

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Apply the preconditioner (block Jacobi part plus kernel correction, if any) to rhs."""
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space
        # kernel correction first, in case out is rhs
        if self._kernel_correction is not None:
            self._kernel_correction.dot(rhs, out=self._tmp_kernel)
        b = rhs
        if self._bc is not None:
            self._mass_k._V_boundary_op.dot(rhs, out=self._tmp)
            b = self._tmp
        out = self._solver.dot(b, out=out)
        if self._kernel_correction is not None:
            out += self._tmp_kernel
        return out

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        return self.solve(v, out=out)

    def transpose(self, conjugate: bool = False) -> PolarStiffnessPreconditioner:
        """The operator is symmetric, so is the preconditioner."""
        return self


def _poloidal_mass_block(
    terms: list[tuple[float, WeightedMassOperator]], derham, bases: list, r: int, r2: int, reduction: str
) -> sparse.csr_matrix | None:
    """Block (r, r2) of the poloidal mass matrix of a sum of mass operators, with the weights reduced in eta_3
    (on the tensor-product coefficients; None if the block vanishes). Collective for array weights."""
    out = None
    for coeff, op in terms:
        w, w_t = op.weights[r][r2], op.weights[r2][r]
        if w is not None:
            B = poloidal_mass_matrix(
                derham, tuple(bases[r][:2]), tuple(bases[r2][:2]), reduced_weight_2d(w, derham, reduction)
            )
        elif op.symmetry in ("symm", "asym") and w_t is not None:
            B_t = poloidal_mass_matrix(
                derham, tuple(bases[r2][:2]), tuple(bases[r][:2]), reduced_weight_2d(w_t, derham, reduction)
            )
            B = (1.0 if op.symmetry == "symm" else -1.0) * B_t.T.tocsr()
        else:
            continue
        out = coeff * B if out is None else out + coeff * B
    return out


def _polar_stiffness_groups(
    mass_ops: WeightedMassOperators,
    derivative: str,
    sigma: float,
    mid_terms: list,
    stab_terms: list | None,
    stab_is_identity: bool,
    reduction: str,
    average_dirs: tuple[int, ...] = (),
) -> list[KronSum2Dx1DGroup]:
    """Factors of the 2D x 1D approximation of the polar stiffness operator, see :class:`PolarStiffnessPreconditioner`
    (without boundary conditions)."""
    derham = mass_ops.derham
    k = PolarStiffnessPreconditioner._FORMS[derivative]
    mass_k = getattr(mass_ops, f"M{k}")
    space = mass_k.domain
    bases = spline_bases(mass_k.domain_femspace)
    bases_mid = spline_bases(getattr(mass_ops, f"M{k + 1}").domain_femspace)
    E_op = derham.extraction_ops[space.space_id]

    # blocks (r, c) of the tensor-product derivative: direction and 1d derivative matrix (with sign)
    d_tp = getattr(derham, f"{derivative}_tp")
    blocks = {}
    for c, dirs in enumerate(_derivative_directions(d_tp)):
        for diffdir, r in dirs.items():
            blocks[(r, c)] = (diffdir, _dense_derivative_1d(d_tp, r, c))

    # 1d stiffness in eta_3; the derivatives in eta_3 of all blocks are equal up to the sign
    D3 = next(D for diffdir, D in blocks.values() if diffdir == 2)
    S_3 = D3.T @ _dense_mass_1d(mass_k, "M", 2, _ones_1d)[0] @ D3

    def poloidal_factor(diffdir: int, D: np.ndarray, c: int) -> sparse.csr_matrix:
        n1, n2 = tp_space(space, c).npts[:2]
        if diffdir == 0:
            return sparse.kron(sparse.csr_matrix(D), sparse.eye(n2), format="csr")
        if diffdir == 1:
            return sparse.kron(sparse.eye(n1), sparse.csr_matrix(D), format="csr")
        if np.allclose(D, D3):
            return sparse.eye(n1 * n2, format="csr")
        assert np.allclose(D, -D3), "The derivatives in eta_3 must be equal up to the sign."
        return -sparse.eye(n1 * n2, format="csr")

    # poloidal mass blocks of the matrix in the middle (computed once, in the same order on all processes)
    mid_blocks = {}

    def mid_block(r: int, r2: int):
        if (r, r2) not in mid_blocks:
            mid_blocks[(r, r2)] = _poloidal_mass_block(mid_terms, derham, bases_mid, r, r2, reduction)
        return mid_blocks[(r, r2)]

    groups = []
    for comps in component_groups(space):
        n = len(comps)
        terms = {"M": [[None] * n for _ in comps], "S": [[None] * n for _ in comps]}
        for a, c in enumerate(comps):
            for b, c2 in enumerate(comps):
                for (r, cc), (dd, D) in blocks.items():
                    if cc != c:
                        continue
                    for (r2, cc2), (dd2, D2) in blocks.items():
                        # mixed terms (poloidal and eta_3 derivative) are neglected
                        if cc2 != c2 or (dd == 2) != (dd2 == 2):
                            continue
                        M_rr = mid_block(r, r2)
                        if M_rr is None:
                            continue
                        term = poloidal_factor(dd, D, c).T @ M_rr @ poloidal_factor(dd2, D2, c2)
                        t = "S" if dd == 2 else "M"
                        terms[t][a][b] = term if terms[t][a][b] is None else terms[t][a][b] + term
                if sigma != 0.0 and not stab_is_identity:
                    S_ab = _poloidal_mass_block(stab_terms, derham, bases, c, c2, reduction)
                    if S_ab is not None:
                        S_ab = sigma * S_ab
                        terms["M"][a][b] = S_ab if terms["M"][a][b] is None else terms["M"][a][b] + S_ab

        E = poloidal_extraction_matrix(space, comps, E_op.blocks_ten_to_pol, None)
        A_pol = {}
        for t, blk in terms.items():
            if all(B is None for row in blk for B in row):
                continue
            for a, c in enumerate(comps):
                for b, c2 in enumerate(comps):
                    if blk[a][b] is None:
                        blk[a][b] = sparse.csr_matrix((_poloidal_size(space, c), _poloidal_size(space, c2)))
            A_pol[t] = (E @ sparse.bmat(blk, format="csr") @ E.T).tocsr()

        # unweighted 1d mass matrix in eta_3
        M_3 = _dense_mass_1d(mass_k, bases[comps[0]][2], 2, _ones_1d)[0]
        B_3 = {}
        if sigma != 0.0 and stab_is_identity:
            A_pol["I"] = sigma * sparse.eye(E.shape[0], format="csr")
            B_3["I"] = np.eye(M_3.shape[0])

        # averaged stabilization: -sigma S Pi_S (gradient only, one group)
        low_rank = []
        if sigma != 0.0 and average_dirs:
            if stab_is_identity:
                S_pol = sparse.eye(E.shape[0], format="csr")
            else:
                S_pol = (E @ _poloidal_mass_block(stab_terms, derham, bases, 0, 0, reduction) @ E.T).tocsr()
            if 2 in average_dirs:
                c_3 = np.ones(M_3.shape[0]) if stab_is_identity else M_3 @ np.ones(M_3.shape[0])
                B_3["F"] = np.outer(c_3, c_3) / c_3.sum()
                key_3 = "F"
            else:
                key_3 = "I" if stab_is_identity else "M"
            if 1 in average_dirs:
                B_pol = poloidal_average_basis(space, E_op)
                low_rank.append((-sigma, S_pol, B_pol, key_3))
            else:
                A_pol["F"] = -sigma * S_pol

        groups.append(KronSum2Dx1DGroup(comps, A_pol, M_3, S_3 if "S" in A_pol else None, B_3, low_rank))
    return groups


class HiptmairXuPreconditioner(LinearOperator):
    r"""
    Auxiliary space preconditioner of Hiptmair and Xu (SIAM J. Numer. Anal. 45, 2007) for the curl-curl operator

    .. math::

        A = \mathbb C^T \mathbb M_2 \mathbb C + \sigma \, \mathbb M_1 \,, \qquad \sigma > 0\,.

    The kernel of the curl (the gradients) and the smooth fields are not reduced by point or block smoothers.
    They are corrected in the auxiliary spaces :math:`V^0_h` (scalar potentials) and
    :math:`(V^0_h)^3` (vector-valued :math:`H^1` splines, ``"H1vec"``):

    .. math::

        P = S + \sigma^{-1} \, \mathbb G \, P_0 \, \mathbb G^T + \Pi \, P_v \, \Pi^T \,,

    where

    * :math:`S` is the smoother, either the Jacobi preconditioner :math:`\textrm{diag}(A)^{-1}` (``"jacobi"``)
      or the block Jacobi part of :class:`StiffnessPreconditioner` (``"block_jacobi"``, default,
      fast diagonalization of the diagonal blocks of :math:`A`),
    * :math:`P_0` approximates the inverse of :math:`\mathbb G^T \mathbb M_1 \mathbb G`
      (:class:`StiffnessPreconditioner` for the gradient with ``mid=stab``),
      since :math:`A \mathbb G = \sigma \mathbb M_1 \mathbb G`,
    * :math:`\Pi: (V^0_h)^3 \to V^1_h` is the commuting projector :math:`\hat \Pi^1` applied to vector-valued
      :math:`H^1` splines (:class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` with identity weight),
    * :math:`P_v` approximates the inverse of the vector Laplacian :math:`\nabla^T \nabla + \sigma` on
      :math:`(V^0_h)^3`, which bounds :math:`\Pi^T A \Pi` from above: the scalar :class:`StiffnessPreconditioner`
      for the gradient with coefficient :math:`\sigma`, applied to each component.

    With essential boundary conditions, all components of the auxiliary space :math:`(V^0_h)^3` vanish on the
    Dirichlet boundary, which is the stable regular decomposition :math:`H_0(\textnormal{curl}) = H^1_0(\Omega)^3 + \nabla H^1_0(\Omega)`.
    The number of PCG iterations is then bounded independently of the mesh size and of :math:`\sigma`.

    Polar splines are not supported (as for :class:`StiffnessPreconditioner`).

    Parameters
    ----------
    A : LinearOperator
        The curl-curl operator on ``derham.coeff_spaces["1"]``; only needed for ``smoother="jacobi"``.

    mass_ops : WeightedMassOperators
        The mass operators.

    sigma : float
        Coefficient of the mass term (must be positive).

    smoother : str
        ``"block_jacobi"`` (default) or ``"jacobi"``.

    mid : str | LinearOperator | None
        Matrix in the middle of the curl-curl term (default :math:`\mathbb M_2`), see :class:`StiffnessPreconditioner`.

    stab : str | LinearOperator | None
        Stabilization (default :math:`\mathbb M_1`, ``"Id"`` for the identity), see :class:`StiffnessPreconditioner`.

    aux_weight : float
        Factor in front of the vector auxiliary space correction :math:`\Pi P_v \Pi^T` (0 switches it off).

    apply_bc, diagonal_scaling, weights :
        Passed to the Kronecker preconditioners (see :class:`StiffnessPreconditioner`).
    """

    _SMOOTHERS = ("block_jacobi", "jacobi")

    def __init__(
        self,
        A: LinearOperator | None,
        mass_ops: WeightedMassOperators,
        sigma: float,
        smoother: str = "block_jacobi",
        mid: str | LinearOperator | None = None,
        stab: str | LinearOperator | None = None,
        aux_weight: float = 1.0,
        apply_bc: bool = True,
        diagonal_scaling: bool = True,
        weights: str = "average",
    ):
        # avoid circular imports
        from struphy.feec.basis_projection_ops import BasisProjectionOperators
        from struphy.linear_algebra.multigrid.smoothers import DiagonalComputer, inverse_diagonal

        assert smoother in self._SMOOTHERS, f"smoother must be one of {self._SMOOTHERS}, got {smoother!r}."
        if sigma <= 0.0:
            raise ValueError(f"HiptmairXuPreconditioner needs sigma > 0, got {sigma =}.")

        derham = mass_ops.derham
        if derham.polar_splines:
            raise NotImplementedError("HiptmairXuPreconditioner does not support polar splines yet.")

        self._mass_ops = mass_ops
        self._sigma = sigma
        self._smoother = smoother
        self._aux_weight = aux_weight
        self._space = derham.coeff_spaces["1"]
        kron_kwargs = dict(apply_bc=apply_bc, diagonal_scaling=diagonal_scaling, weights=weights)

        # smoother and gradient correction
        if smoother == "block_jacobi":
            with ProfileManager.profile_region("block_jacobi smoother", functions=[StiffnessPreconditioner.__init__]):
            # block Jacobi plus the correction sigma^{-1} G P_0 G^T on the gradients
                self._S = StiffnessPreconditioner(
                    mass_ops, "curl", sigma=sigma, mid=mid, stab=stab, kernel_correction=True, **kron_kwargs
                )
                self._grad_correction = None
        else:
            with ProfileManager.profile_region("jacobi smoother", functions=[StiffnessPreconditioner.__init__]):
                assert A is not None, "smoother='jacobi' needs the operator A."
                assert A.domain is self._space and A.codomain is self._space
                self._S = inverse_diagonal(DiagonalComputer(derham.degree)(A))
                P_0 = StiffnessPreconditioner(mass_ops, "grad", mid=None if stab == "Id" else stab, **kron_kwargs)
                self._grad_correction = (1.0 / sigma) * (derham.grad @ P_0 @ derham.grad.T)

        # vector auxiliary space: Pi P_v Pi^T
        self._Pi = None
        if aux_weight != 0.0:
            with ProfileManager.profile_region("auxiliary_space correction"):
                identity = [[(lambda e1, e2, e3: 1.0 + 0.0 * e1) if m == n else None for n in range(3)] for m in range(3)]
                basis_ops = BasisProjectionOperators(derham, mass_ops.domain)
                self._Pi = basis_ops.create_basis_op(identity, "H1vec", "Hcurl", assemble=True, name="Pi_v1")
                self._PiT = self._Pi.T
                self._P_v = StiffnessPreconditioner(mass_ops, "grad", sigma=sigma, **kron_kwargs)
                self._aux = derham.coeff_spaces["v"]
                self._tmp_v = (self._aux.zeros(), self._aux.zeros())
                self._tmp_0 = (self._P_v.domain.zeros(), self._P_v.codomain.zeros())

        self._tmp = self._space.zeros()

    @property
    def domain(self) -> VectorSpace:
        return self._space

    @property
    def codomain(self) -> VectorSpace:
        return self._space

    @property
    def dtype(self):
        return self._space.dtype

    @property
    def sigma(self) -> float:
        """Coefficient of the mass term."""
        return self._sigma

    @property
    def smoother(self) -> str:
        """``"block_jacobi"`` or ``"jacobi"``."""
        return self._smoother

    @property
    def aux_weight(self) -> float:
        """Factor in front of the vector auxiliary space correction."""
        return self._aux_weight

    def _apply_P_v(self, z: Vector, out: Vector) -> Vector:
        """Scalar preconditioner applied to each component of ``z`` in ``(V^0_h)^3``."""
        x0, y0 = self._tmp_0
        for zc, oc in zip(z.blocks, out.blocks):
            _copy_owned(zc, x0)
            self._P_v.dot(x0, out=y0)
            _copy_owned(y0, oc)
        return out

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        """Apply the preconditioner to ``v``."""
        assert v.space is self._space
        if out is None:
            out = self._space.zeros()

        self._S.dot(v, out=out)
        if self._grad_correction is not None:
            out += self._grad_correction.dot(v, out=self._tmp)
        if self._Pi is not None:
            z, w = self._tmp_v
            self._PiT.dot(v, out=z)
            self._apply_P_v(z, w)
            self._Pi.dot(w, out=self._tmp)
            if self._aux_weight != 1.0:
                self._tmp *= self._aux_weight
            out += self._tmp
        return out

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Same as ``dot`` (approximate inverse of :math:`A`)."""
        return self.dot(rhs, out=out)

    def transpose(self, conjugate: bool = False) -> HiptmairXuPreconditioner:
        """The operator is symmetric, so is the preconditioner."""
        return self


class AuxiliarySpacePreconditioner(LinearOperator):
    r"""
    Additive auxiliary space preconditioner

    .. math::

        P = P_0 + \sum_i \mathbb N_i \, P_i \, \mathbb N_i^T \,,

    where :math:`P_0` is a preconditioner of :math:`A` (e.g. a smoother, or a block Jacobi
    :class:`StiffnessPreconditioner`) and :math:`P_i` approximates the inverse of :math:`\mathbb N_i^T A \mathbb N_i` on an
    auxiliary space, mapped into the space of :math:`A` by :math:`\mathbb N_i`. The auxiliary spaces correct the modes
    that :math:`P_0` does not reduce, e.g. the gradients in :class:`HiptmairXuPreconditioner`.

    Parameters
    ----------
    P0 : LinearOperator
        The preconditioner on the space of :math:`A`.

    corrections : list
        Per auxiliary space, the triple ``(N, NT, P)``: the map :math:`\mathbb N_i`, its transpose and :math:`P_i`.
    """

    def __init__(self, P0: LinearOperator, corrections: list[tuple[LinearOperator, LinearOperator, LinearOperator]]):
        assert P0.domain == P0.codomain
        self._P0 = P0
        self._space = P0.domain
        self._corrections = []
        for N, NT, P in corrections:
            assert N.codomain == self._space and NT.domain == self._space
            assert P.domain == NT.codomain and P.codomain == N.domain
            self._corrections.append((N, NT, P, NT.codomain.zeros(), P.codomain.zeros()))
        self._tmp = self._space.zeros()

    @property
    def domain(self) -> VectorSpace:
        return self._space

    @property
    def codomain(self) -> VectorSpace:
        return self._space

    @property
    def dtype(self):
        return self._space.dtype

    @property
    def P0(self) -> LinearOperator:
        """The preconditioner on the space of :math:`A`."""
        return self._P0

    @property
    def corrections(self) -> list[tuple[LinearOperator, LinearOperator, LinearOperator]]:
        """The triples ``(N, NT, P)`` of the auxiliary spaces."""
        return [c[:3] for c in self._corrections]

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        """Apply the preconditioner to ``v``."""
        assert v.space == self._space
        if out is None:
            out = self._space.zeros()
        assert out is not v, "In-place application is not supported."
        self._P0.dot(v, out=out)
        for N, NT, P, z, w in self._corrections:
            NT.dot(v, out=z)
            P.dot(z, out=w)
            out += N.dot(w, out=self._tmp)
        return out

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Same as ``dot`` (approximate inverse of :math:`A`)."""
        return self.dot(rhs, out=out)

    def transpose(self, conjugate: bool = False) -> AuxiliarySpacePreconditioner:
        """Symmetric if :math:`P_0` and the :math:`P_i` are."""
        return self


def _copy_owned(src: StencilVector, dst: StencilVector) -> None:
    """Copy the owned coefficients of ``src`` into ``dst`` (stencil vectors of equal shape on different spaces)."""
    assert tuple(src.space.starts) == tuple(dst.space.starts) and tuple(src.space.ends) == tuple(dst.space.ends)
    dst._data[...] = src._data
    dst.ghost_regions_in_sync = False


# --------------------------------------------------------------------------------------
# Helper functions for building Kronecker approximations of mass matrices
# --------------------------------------------------------------------------------------
def _ones_1d(e: xp.ndarray) -> xp.ndarray:
    """Unit weight for 1d mass matrices."""
    return xp.ones(e.size, dtype=float)


def _local_diagonal(op: LinearOperator) -> list[xp.ndarray]:
    """
    Local (rows owned by this process) diagonal of ``op``, one array per block of its domain.

    Supported are StencilMatrix, StencilDiagonalMatrix, KroneckerStencilMatrix (1d factors),
    _DenseKroneckerProduct, sums, scalings and BlockLinearOperators of these (missing diagonal
    blocks give zeros).
    """
    if isinstance(op, BlockLinearOperator):
        diags = []
        for i, V in enumerate(op.domain.spaces):
            if op[i, i] is None:
                # zero block (e.g. a mass matrix block with zero weight)
                diags.append(xp.zeros(tuple(e - s + 1 for s, e in zip(V.starts, V.ends)), dtype=float))
            else:
                diags.extend(_local_diagonal(op[i, i]))
        return diags
    if isinstance(op, StencilDiagonalMatrix):
        return [op._data]
    if isinstance(op, StencilMatrix):
        return [_stencil_local_diagonal(op)]
    if isinstance(op, KroneckerStencilMatrix):
        # outer product of the diagonals of the (process-local) factors on the local rows
        diags = []
        for A, s, e in zip(op.mats, op.codomain.starts, op.codomain.ends):
            assert A.domain.ndim == 1, "Only 1d factors are supported."
            # the factor may own more rows than the codomain on this process
            off = A.codomain.pads[0] * A.codomain.shifts[0] + s - A.codomain.starts[0]
            diags.append(A._data[off : off + e - s + 1, A.pads[0]])
        # outer product by broadcasting (works on every backend, unlike ufunc.outer)
        out = diags[0]
        for d in diags[1:]:
            out = out[..., None] * d
        return [out]
    if isinstance(op, _DenseKroneckerProduct):
        return [op.local_diagonal()]
    if isinstance(op, SumLinearOperator):
        parts = [_local_diagonal(a) for a in op.addends]
        return [sum(blocks) for blocks in zip(*parts)]
    if isinstance(op, ScaledLinearOperator):
        return [op.scalar * a for a in _local_diagonal(op.operator)]
    raise NotImplementedError(f"Diagonal of {type(op).__name__} is not supported.")


def _stencil_local_diagonal(M: StencilMatrix) -> xp.ndarray:
    """
    Local diagonal of a StencilMatrix, including the band entries that wrap onto the main diagonal.

    In a periodic direction with fewer points than the band width (``2p + 1 > n``, e.g. a single
    element), the band entries at the offsets ``k = m n`` (``m != 0``) are on the main diagonal too;
    :meth:`StencilMatrix.diagonal` returns only the entry at ``k = 0``.
    """
    V = M.codomain
    assert all(m == 1 for m in V.shifts) and all(m == 1 for m in M.domain.shifts), "Only shifts 1 are supported."
    offsets = [
        [k for k in range(-p, p + 1) if k % n == 0] if periodic else [0]
        for n, p, periodic in zip(V.npts, M.pads, V.periods)
    ]
    diag = M.diagonal()._data
    if all(o == [0] for o in offsets):
        return diag
    idx = M._get_diagonal_indices()
    rows, cols = idx[: V.ndim], idx[V.ndim :]
    for ks in itertools.product(*offsets):
        if any(ks):
            diag = diag + M._data[rows + tuple(j + k for j, k in zip(cols, ks))]
    return diag


def _diagonal_operator(space: VectorSpace, diags: list[xp.ndarray]) -> LinearOperator:
    """Diagonal operator on ``space`` (Stencil- or BlockVectorSpace) with the given local diagonals."""
    if isinstance(space, BlockVectorSpace):
        n = len(space.spaces)
        blocks = [
            [StencilDiagonalMatrix(V, V, diags[i]) if i == j else None for j in range(n)]
            for i, V in enumerate(space.spaces)
        ]
        return BlockLinearOperator(space, space, blocks=blocks)
    return StencilDiagonalMatrix(space, space, diags[0])


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
    last = mass_operator.M0.multiplicands[-1]
    if not isinstance(last, BoundaryOperator):
        return None
    return last.bc


def _operator_to_invert(
    mass_operator: WeightedMassOperator, apply_bc: bool
) -> tuple[ComposedLinearOperator | LinearOperator, int | None, tuple[Vector, ...]]:
    """
    The operator approximately inverted by the preconditioners: ``M0`` (with boundary
    operators) if ``apply_bc``, else ``M``.

    Returns
    -------
    M : ComposedLinearOperator | LinearOperator
        The operator.

    mass_index : int | None
        Position of the mass matrix ``mass_operator.matrix`` in ``M.multiplicands``, or
        None if ``M`` is not composed (then ``M`` is the mass matrix itself).

    tmp_vectors : tuple[Vector, ...]
        Temporary vectors for ``_apply_composed``: the codomains of all but the first
        factor of ``M`` (empty if ``M`` is not composed).
    """
    M = mass_operator.M0 if apply_bc else mass_operator.M
    if not isinstance(M, ComposedLinearOperator):
        return M, None, ()

    mass_index = next((k for k, op in enumerate(M.multiplicands) if op is mass_operator.matrix), None)
    assert mass_index is not None, "The mass matrix was not found in the composed mass operator."
    tmp_vectors = tuple(op.codomain.zeros() for op in M.multiplicands[1:])
    return M, mass_index, tmp_vectors


def _apply_composed(
    M: LinearOperator,
    mass_index: int | None,
    apply_inverse: Callable[[Vector, Vector], Vector],
    tmp_vectors: tuple[Vector, ...],
    rhs: Vector,
    out: Vector | None,
) -> Vector:
    """
    Apply the operator ``M`` (see ``_operator_to_invert``) to ``rhs``, with the mass matrix
    replaced by ``apply_inverse(x, out)`` (an approximate inverse).

    The factors of a composition are applied from right to left; the result is written
    to ``out`` (allocated if None).
    """
    if mass_index is None:
        if out is None:
            out = M.codomain.zeros()
        return apply_inverse(rhs, out)

    ops = M.multiplicands
    if out is None:
        out = ops[0].codomain.zeros()

    x = rhs
    for k in range(len(ops) - 1, -1, -1):
        y = out if k == 0 else tmp_vectors[k - 1]
        if k == mass_index:
            apply_inverse(x, y)
        else:
            ops[k].dot(x, out=y)
        x = y
    return out


# number of Gauss-Legendre points per direction for averaging callable weights
_N_AVG_CALLABLE = 16


def _reduced_weight_1d(weight, c: int, d: int, derham, reduction: str = "midpoint") -> Callable | xp.ndarray:
    r"""
    1d weight along direction ``d``, obtained from the 3d weight by reducing the other two directions
    :math:`i, j \neq d`:

    * ``"midpoint"``: the weight at the mid point, :math:`\bar w(\eta_d) = w(\eta_d; \eta_i = \eta_j = 0.5)`,
    * ``"average"``: the mean, :math:`\bar w(\eta_d) = \int_0^1 \int_0^1 w \, \textnormal d\eta_i \, \textnormal d\eta_j`.

    Parameters
    ----------
    weight : callable | xp.ndarray | None
        Block ``(c, c)`` of the weights of the 3d mass operator: a function of the three
        logical coordinates, its values at the local quadrature points, or None (unit weight).

    c, d : int
        Component and direction (``c`` is only used for logging).

    derham : Derham
        Discrete de Rham sequence of the mass operator.

    reduction : str
        ``"midpoint"`` or ``"average"``.

    Returns
    -------
    Callable | xp.ndarray
        For a callable weight, a function of the 1d points (for ``"average"``, the mean is
        approximated with Gauss-Legendre quadrature on ``_N_AVG_CALLABLE`` points per direction).
        For an array weight, the reduced weight at all global 1d quadrature points; this is
        collective on ``derham.comm``. For the mid point (array weight) it is the value at the
        global mid quadrature point; for the mean, Gauss quadrature of the derham is used.
    """
    assert reduction in ("midpoint", "average"), f"Unknown weight reduction {reduction!r}."
    n_dims = 3
    i, j = (k for k in range(n_dims) if k != d)

    if weight is None:
        return _ones_1d

    if callable(weight):
        if reduction == "midpoint":

            def fun(e):
                # evaluate the 3d weight on the "meshgrid" (0.5, ..., e, ..., 0.5)
                s = e.shape[0]
                f = e.reshape(tuple([1 if k != d else s for k in range(n_dims)]))
                return xp.atleast_1d(
                    weight(
                        *[xp.array(xp.full_like(f, 0.5)) if k != d else xp.array(f) for k in range(n_dims)]
                    ).squeeze(),
                )

            return fun

        # Gauss-Legendre points and weights on [0, 1]
        nodes, wts = np.polynomial.legendre.leggauss(_N_AVG_CALLABLE)
        pts = xp.asarray((nodes + 1.0) / 2.0)
        wts = xp.asarray(wts / 2.0)

        def fun(e):
            # integrate over direction i on the grid (e, pts) for each point in direction j
            e_d, e_i = xp.meshgrid(xp.ravel(e), pts, indexing="ij")
            total = xp.zeros(e_d.shape[0], dtype=float)
            args = [None] * n_dims
            args[d] = e_d
            args[i] = e_i
            for eta_j, w_j in zip(pts, wts):
                args[j] = xp.full_like(e_d, eta_j)
                total += w_j * (xp.reshape(weight(*args), e_d.shape) @ wts)
            return total

        return fun

    if isinstance(weight, xp.ndarray):
        logger.debug(f"{weight.shape = } for component {c} and direction {d}.")
        dom_dec = derham.domain_decomposition
        nq_d = derham.nquads[d]
        start = dom_dec.starts[d] * nq_d
        stop = start + weight.shape[d]

        # every process writes its contribution at its rows along d; the sum over all processes
        # gives the reduced weight (row 0) and the normalization (row 1)
        sums = xp.zeros((2, derham.num_elements[d] * nq_d), dtype=float)

        if reduction == "midpoint":
            # only the process owning the global mid point in directions i and j contributes
            # (exactly one process per slab along d)
            mid = {}
            for k in (i, j):
                nq_k = weight.shape[k] // dom_dec.local_ncells[k]
                mid[k] = (derham.num_elements[k] * nq_k) // 2 - dom_dec.starts[k] * nq_k
            if all(0 <= mid[k] < weight.shape[k] for k in (i, j)):
                cut = tuple(slice(None) if k == d else mid[k] for k in range(n_dims))
                sums[0, start:stop] = weight[cut]
                sums[1, start:stop] = 1.0
        else:
            # Gauss weights at the local quadrature points (same grid as the array weight)
            qwts = [xp.ravel(w) for w in derham.spline_attributes["H1"].quad_grid_wts[0]]
            assert weight.shape == tuple(w.size for w in qwts), (
                f"Array weight of shape {weight.shape} does not match the local quadrature grid."
            )
            weighted = weight * qwts[i].reshape([-1 if k == i else 1 for k in range(n_dims)])
            weighted = weighted * qwts[j].reshape([-1 if k == j else 1 for k in range(n_dims)])
            sums[0, start:stop] = weighted.sum(axis=(i, j))
            sums[1, start:stop] = qwts[i].sum() * qwts[j].sum()

        comm = derham.comm
        if not isinstance(comm, (SerialComm, type(None))):
            local_sums = sums.copy()
            comm.Allreduce(local_sums, sums, op=MPI.SUM)

        return sums[0] / sums[1]

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


def _solver_1d(M: StencilMatrix, M_arr: np.ndarray) -> FFTSolver | SparseSolver:
    """
    Direct solver for the 1d matrix M (dense host copy M_arr): FFT if circulant, else sparse LU.

    The solver is a host solver on every backend, as expected by KroneckerLinearSolver: on the
    CuPy backend, the latter solves device data with the dense inverse it builds from this solver
    once, so that applying the preconditioner makes no host copies.
    """
    if is_circulant(M_arr):
        return FFTSolver(M_arr)
    return SparseSolver(M.tosparse())


def _process_local_matrix_1d(
    M_arr: np.ndarray, coeff_space: StencilVectorSpace, d: int, domain_decomposition: DomainDecomposition
) -> StencilMatrix:
    """
    Process-local 1d factor of a KroneckerStencilMatrix on ``coeff_space``.

    The factor lives on a 1d space without communicator that owns the same rows
    (``starts[d]`` to ``ends[d]``) as ``coeff_space`` in direction ``d`` on this process,
    as required by KroneckerStencilMatrix. The band is filled on the host and copied to the
    data of the factor (active backend) in one step.

    Parameters
    ----------
    M_arr : np.ndarray
        The global 1d matrix as dense array (a device array is copied to the host).

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
    M_arr = xp.to_numpy(M_arr)

    # copy the rows owned by this process: entry (i, j) is stored at row i - s + p and diagonal
    # index k + p with the offset k = j - i; in periodic directions k is taken nearest to 0
    # (in [-n/2, n/2)), so that the main diagonal is at k = 0 also if 2p + 1 > n
    rows, cols = np.nonzero(M_arr)
    on_process = (rows >= s) & (rows <= e)
    rows, cols = rows[on_process], cols[on_process]
    k = cols - rows
    if coeff_space.periods[d]:
        k = (k + n // 2) % n - n // 2
    assert np.all(np.abs(k) <= p), "Entries outside the band of the stencil matrix."
    data = np.zeros(M_local._data.shape, dtype=M_local._data.dtype)
    data[rows - s + p, k + p] = M_arr[rows, cols]
    M_local._data[...] = xp.asarray(data)

    # check if stencil matrix was built correctly
    assert np.allclose(xp.to_numpy(M_local.toarray())[s : e + 1], M_arr[s : e + 1])

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

            # the dense 1d matrix and the 1d solver are host setup data on every backend
            M_arr = xp.to_numpy(M.toarray())
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


# --------------------------------------------------------------------------------------
# Helper functions for stiffness preconditioners
# --------------------------------------------------------------------------------------
def _derivative_directions(derivative: LinearOperator) -> list[dict[int, int]]:
    """
    For each component c of the domain of a (tensor-product) derivative, the directions in
    which it is differentiated, mapped to the codomain block of that derivative.

    The blocks ``(r, c)`` of ``derivative`` (a BlockLinearOperator, or a single operator) are
    DirectionalDerivativeOperators, possibly negated.
    """
    if isinstance(derivative, BlockLinearOperator):
        nrows, ncols = derivative.n_block_rows, derivative.n_block_cols
        blocks = {(r, c): derivative[r, c] for r in range(nrows) for c in range(ncols) if derivative[r, c] is not None}
    else:
        ncols = 1
        blocks = {(0, 0): derivative}

    directions = [{} for _ in range(ncols)]
    for (r, c), op in blocks.items():
        assert isinstance(op, DirectionalDerivativeOperator), f"Unexpected block {type(op).__name__} in derivative."
        assert op.diffdir not in directions[c], "Component differentiated twice in the same direction."
        directions[c][op.diffdir] = r
    return directions


def _separable_weight(weights: list, derham) -> tuple[list[xp.ndarray], list[float]]:
    r"""
    Common separable shape of 3d weights, :math:`w_i \approx \bar w_i \prod_d \phi_d(\eta_d)`.

    Parameters
    ----------
    weights : list
        Per weight :math:`w_i`: a 3d weight (callable, array at the local quadrature points, or
        None for 1), or a list of ``(coefficient, weight)`` terms (an empty list is a zero weight).

    derham : Derham
        Discrete de Rham sequence (quadrature grid).

    Returns
    -------
    shapes : list[xp.ndarray]
        :math:`\phi_d` at the global 1d quadrature points: the mean over the other directions,
        divided by the mean, averaged over the weights with non-zero mean (mean 1).

    means : list[float]
        The means :math:`\bar w_i` of the weights over the logical cube (0 for weights whose
        mean vanishes relative to the largest one).

    Collective on ``derham.comm`` for array weights.
    """
    grids = [derham.H1_1d_serial[d].get_assembly_grids(derham.nquads[d])[0] for d in range(3)]
    pts = [xp.asarray(np.ravel(g.points)) for g in grids]
    wts = [xp.asarray(np.ravel(g.weights)) for g in grids]

    profiles, means = [], []
    for w in weights:
        terms = w if isinstance(w, list) else [(1.0, w)]
        prof = []
        for d in range(3):
            p_d = xp.zeros(pts[d].size, dtype=float)
            for coeff, w_t in terms:
                p = _reduced_weight_1d(w_t, 0, d, derham, "average")
                p_d = p_d + coeff * xp.asarray(p(pts[d]) if callable(p) else p, dtype=float)
            prof.append(p_d)
        profiles.append(prof)
        means.append(float(xp.sum(prof[0] * wts[0]) / xp.sum(wts[0])))

    largest = max(abs(m) for m in means)
    assert largest > 0.0, "The weights must not all vanish."
    means = [0.0 if abs(m) <= 1e-12 * largest else m for m in means]
    assert all(m >= 0.0 for m in means), f"The weights must have non-negative means, got {means}."

    nonzero = [(prof, m) for prof, m in zip(profiles, means) if m > 0.0]
    shapes = [sum(prof[d] / m for prof, m in nonzero) / len(nonzero) for d in range(3)]
    return shapes, means


def _dense_mass_1d(
    mass_operator: WeightedMassOperator, basis: str, d: int, weight: Callable | xp.ndarray
) -> tuple[np.ndarray, DomainDecomposition]:
    """Global 1d mass matrix in direction d (B- or M-splines) as dense host array, see ``_mass_matrix_1d``."""
    M, domain_decomposition = _mass_matrix_1d(mass_operator, basis, d, weight)
    return np.asarray(xp.to_numpy(M.toarray()), dtype=float), domain_decomposition


def _dense_derivative_1d(derivative: LinearOperator, r: int, c: int) -> np.ndarray:
    """
    Global 1d derivative matrix (B- to M-spline coefficients) of the block ``(r, c)`` of a
    tensor-product derivative, as dense host array, taken from the 1d factor of the block itself
    (:meth:`DirectionalDerivativeOperator.tokronstencil`), so that it matches the 3d operator
    (e.g. it vanishes in a periodic direction with a single element).
    """
    op = derivative[r, c] if isinstance(derivative, BlockLinearOperator) else derivative
    assert isinstance(op, DirectionalDerivativeOperator) and not op.transposed
    D = op.tokronstencil().mats[op.diffdir].toarray()
    # the sign of a negated block is already included in its 1d factor
    return np.asarray(xp.to_numpy(D), dtype=float)


def _apply_bc_dense(A: np.ndarray, bc_d: tuple[bool, bool]) -> np.ndarray:
    """Impose essential boundary conditions on a dense 1d matrix: zero row and column, 1 on the diagonal."""
    A = A.copy()
    for i, is_essential in zip((0, -1), bc_d):
        if is_essential:
            A[i, :] = 0.0
            A[:, i] = 0.0
            A[i, i] = 1.0
    return A


def _zero_bc_dense(A: np.ndarray, bc_d: tuple[bool, bool]) -> np.ndarray:
    """Impose essential boundary conditions on a dense 1d matrix by zero rows and columns."""
    A = A.copy()
    for i, is_essential in zip((0, -1), bc_d):
        if is_essential:
            A[i, :] = 0.0
            A[:, i] = 0.0
    return A


def _kronecker_sum(
    space: StencilVectorSpace,
    stiffness: list[StencilMatrix | None],
    mass: list[StencilMatrix],
    sigma: float,
    mass_terms: list[tuple[float, list[StencilMatrix | None]]] | None = None,
) -> LinearOperator:
    """
    The operator :math:`\\sum_d M_1 \\otimes \\dots \\otimes S_d \\otimes \\dots \\otimes M_3 + \\sigma M_1 \\otimes M_2 \\otimes M_3
    + \\sum_t c_t F_{t,1} \\otimes F_{t,2} \\otimes F_{t,3}` as sum of KroneckerStencilMatrix (process-local
    1d factors; directions with ``S_d = None`` have no term; factors ``F = None`` stand for ``M_d``).
    """
    terms = []
    for d, S_d in enumerate(stiffness):
        if S_d is not None:
            factors = [S_d.copy() if e == d else M_e.copy() for e, M_e in enumerate(mass)]
            terms.append(KroneckerStencilMatrix(space, space, *factors))
    if sigma != 0.0:
        terms.append(KroneckerStencilMatrix(space, space, *[M_e.copy() for M_e in mass]) * sigma)
    for coeff, factors in mass_terms or []:
        terms.append(coeff * _DenseKroneckerProduct(space, factors))
    assert terms, "The operator has no terms."
    return reduce(lambda a, b: a + b, terms)


class _DenseKroneckerProduct(LinearOperator):
    """
    Kronecker product :math:`F_1 \\otimes F_2 \\otimes F_3` of dense global 1d matrices on a
    StencilVectorSpace (e.g. the rank-one factors of an average), applied with a
    KroneckerLinearSolver (also along distributed axes).
    """

    class _DenseApply(LinearSolver):
        """'Solver' for KroneckerLinearSolver that applies a dense matrix to each right-hand side."""

        def __init__(self, mat):
            self._mat = mat

        @property
        def space(self):
            return xp.ndarray

        def transpose(self):
            return _DenseKroneckerProduct._DenseApply(self._mat.T)

        def solve(self, rhs, out=None):
            # rows of rhs are the vectors: out_i = mat @ rhs_i
            result = rhs @ self._mat.T
            if out is None:
                return result
            out[...] = result
            return out

    def __init__(self, space: StencilVectorSpace, factors: list[np.ndarray]):
        assert all(F.shape == (n, n) for F, n in zip(factors, space.npts))
        self._space = space
        self._factors = [np.asarray(F, dtype=float) for F in factors]
        self._kron = KroneckerLinearSolver(
            space, space, [_DenseKroneckerProduct._DenseApply(xp.asarray(F)) for F in self._factors]
        )

    @property
    def domain(self) -> StencilVectorSpace:
        return self._space

    @property
    def codomain(self) -> StencilVectorSpace:
        return self._space

    @property
    def dtype(self):
        return self._space.dtype

    def local_diagonal(self) -> xp.ndarray:
        """Diagonal on the rows owned by this process."""
        diags = [
            xp.asarray(np.diag(F)[s : e + 1]) for F, s, e in zip(self._factors, self._space.starts, self._space.ends)
        ]
        out = diags[0]
        for d in diags[1:]:
            out = xp.multiply.outer(out, d)
        return out

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        assert v.space == self._space
        return self._kron.solve(v, out=out)

    def transpose(self, conjugate: bool = False) -> _DenseKroneckerProduct:
        return _DenseKroneckerProduct(self._space, [F.T for F in self._factors])

    def tosparse(self):
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError


def _resolve_mass_operator(mass_ops: WeightedMassOperators, spec, default: LinearOperator) -> LinearOperator:
    """The mass operator given by ``spec``: None (``default``), a name in ``mass_ops`` or an operator."""
    if spec is None:
        return default
    if isinstance(spec, str):
        if not hasattr(mass_ops, spec):
            raise ValueError(f"WeightedMassOperators has no mass operator {spec!r}.")
        return getattr(mass_ops, spec)
    assert isinstance(spec, LinearOperator), f"Expected a name or a LinearOperator, got {type(spec).__name__}."
    return spec


def _mass_terms(op: LinearOperator) -> list[tuple[float, WeightedMassOperator]]:
    """``op`` (a WeightedMassOperator, or a sum or multiple of such) as list of ``(coefficient, operator)``."""
    if isinstance(op, WeightedMassOperator):
        return [(1.0, op)]
    if isinstance(op, ScaledLinearOperator):
        return [(op.scalar * c, t) for c, t in _mass_terms(op.operator)]
    if isinstance(op, SumLinearOperator):
        return [term for a in op.addends for term in _mass_terms(a)]
    raise TypeError(f"Expected a WeightedMassOperator or a sum/multiple of such, got {type(op).__name__}.")


def _terms_matrix(terms: list[tuple[float, WeightedMassOperator]]) -> LinearOperator:
    """Sum of the tensor-product matrices (without boundary operators) of the terms."""
    return reduce(lambda a, b: a + b, [t.matrix if c == 1.0 else c * t.matrix for c, t in terms])


def _block_weights(terms: list[tuple[float, WeightedMassOperator]], n: int) -> list[list[tuple[float, object]]]:
    """Diagonal blocks of the weights of a sum of mass operators: per block, the ``(coefficient, weight)`` terms
    (a stored weight None is a zero block and is left out)."""
    return [[(c, t.weights[r][r]) for c, t in terms if t.weights[r][r] is not None] for r in range(n)]


def _unit_means(terms: list[tuple[float, WeightedMassOperator]], n: int) -> list[float]:
    """Means of the diagonal blocks of a sum of mass operators with unit weights: the sum of the coefficients of the
    terms with a non-zero block."""
    return [sum(c for c, t in terms if t.weights[r][r] is not None) for r in range(n)]


def _parse_directions(spec: str | None) -> tuple[int, ...]:
    """``"eta2 eta3"`` (or ``"eta2,eta3"``) -> ``(1, 2)``; None or ``""`` -> ``()``."""
    if not spec:
        return ()
    dirs = []
    for token in spec.replace(",", " ").split():
        if token not in ("eta1", "eta2", "eta3"):
            raise ValueError(f"Unknown direction {token!r}, expected 'eta1', 'eta2' or 'eta3'.")
        dirs.append(int(token[-1]) - 1)
    return tuple(sorted(set(dirs)))


def _probe_diagonal(op: LinearOperator, space: VectorSpace) -> list[xp.ndarray]:
    """
    Local diagonal of ``op`` (one array per block of ``space``) by probing with colored unit vectors.

    In each direction, the degrees of freedom are colored with a stride larger than the coupling
    width of ``op`` (assumed at most ``pads + 1``); applying ``op`` to the sum of the unit vectors
    of one color gives the diagonal entries of that color. In periodic directions, the stride
    divides the number of points (so that no two dofs of one color couple across the boundary).
    This needs ``prod(stride)`` applications of ``op`` per block; it is collective.
    """
    is_block = isinstance(space, BlockVectorSpace)
    spaces = space.spaces if is_block else (space,)
    x = space.zeros()
    y = space.zeros()
    xs = x.blocks if is_block else (x,)
    ys = y.blocks if is_block else (y,)

    diags = []
    for c, V in enumerate(spaces):
        local = tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(V.pads, V.shifts, V.starts, V.ends))
        shape = tuple(e - s + 1 for s, e in zip(V.starts, V.ends))

        strides = []
        for n, p, periodic in zip(V.npts, V.pads, V.periods):
            st = min(p + 2, n)
            while periodic and n % st:
                st += 1
            strides.append(st)
        indices = [xp.arange(s, e + 1) for s, e in zip(V.starts, V.ends)]

        diag = xp.zeros(shape, dtype=float)
        for color in itertools.product(*(range(st) for st in strides)):
            mask = xp.ones(shape, dtype=bool)
            for d, (g, st, r) in enumerate(zip(indices, strides, color)):
                mask = mask & (g % st == r).reshape([-1 if e == d else 1 for e in range(len(shape))])
            for xb in xs:
                xb._data[:] = 0.0
            xs[c]._data[local] = mask
            x.ghost_regions_in_sync = False
            op.dot(x, out=y)
            diag[mask] = ys[c]._data[local][mask]
        diags.append(diag)
    return diags


class FFTSolver(BandedSolver):
    """
    Solve the equation Ax = b for x, assuming A is a circulant matrix.
    b can contain multiple right-hand sides (RHS) and is of shape (#RHS, N).

    A host (NumPy) solver on every backend: the 1d matrices of Kronecker solvers are small
    setup data. On the CuPy backend, KroneckerLinearSolver solves device data with the dense
    inverse it builds from this solver (by solving for the identity on the host), so ``solve``
    is only called with host arrays.

    Parameters
    ----------
    circmat : np.ndarray
        Generic circulant matrix (a device array is copied to the host).
    """

    def __init__(self, circmat):
        circmat = xp.to_numpy(circmat)
        assert is_circulant(circmat)

        from scipy.linalg import solve_circulant  # deferred: scipy.linalg is slow to import

        self._space = np.ndarray
        # copy: circmat is still used by the caller (e.g. for the process-local stencil matrix)
        self._column = np.array(circmat[:, 0], copy=True)

        # stabilize a singular matrix once, here, so that all solves use the same matrix
        try:
            solve_circulant(self._column, np.ones_like(self._column))
        except np.linalg.LinAlgError:
            eps = 1e-4
            logger.info(f"Stabilizing singular preconditioning FFTSolver with {eps =}:")
            self._column[0] *= 1.0 + eps

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
        rhs : np.ndarray
            The right-hand sides to solve for (host array). The vectors are assumed to be given in C-contiguous order,
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside
            a right-hand side.

        out : np.ndarray, optional
            Output vector. If given, it has to have the same shape and datatype as rhs.

        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """

        from scipy.linalg import solve_circulant

        assert isinstance(rhs, np.ndarray), f"FFTSolver solves on the host, got {type(rhs).__name__}."
        assert rhs.T.shape[0] == self._column.size

        if out is None:
            return solve_circulant(self._column, rhs.T).T

        assert out.shape == rhs.shape
        assert out.dtype == rhs.dtype
        out[:] = solve_circulant(self._column, rhs.T).T
        return out


def is_circulant(mat):
    """
    Returns true if a matrix is circulant.

    Parameters
    ----------
    mat : array[float]
        The matrix that is checked to be circulant (host or device array; checked on the host).

    Returns
    -------
    circulant : bool
        Whether the matrix is circulant (=True) or not (=False).
    """

    mat = xp.to_numpy(mat)
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]

    # circulant: every row is the previous one shifted by one, i.e. mat[i, j] = mat[0, (j - i) % n]
    n = mat.shape[0]
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return bool(np.allclose(mat, mat[0][idx]))
