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
    LinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    Vector,
    VectorSpace,
)
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.direct_solvers import BandedSolver, SparseSolver
from feectools.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix, KroneckerSumSolver
from feectools.linalg.stencil import StencilDiagonalMatrix, StencilMatrix, StencilVectorSpace
from line_profiler import profile
from maybempi import MPI, SerialComm
from scipy import sparse

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators

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
        self._matrix, self._solver = self._build_approximation(self._bc)

        # diagonal scaling S (a sequence of diagonal operators, applied in this order before the
        # solver and in reverse order after it)
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
        D = _local_diagonal(self._core_diagonal())
        D_hat = _local_diagonal(self._matrix)
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
        Axis along which the weight is kept; None for unit weights in all directions.

    weight_reduction : str
        How the weight is reduced in the other axes: ``"midpoint"`` (value at 0.5, default)
        or ``"average"`` (mean over the axes). Which one gives the better preconditioner
        depends on the weight.

    diagonal_scaling : bool
        Whether to correct the approximation with the diagonals of the mass matrix and of
        its approximation (see :class:`KroneckerPreconditioner`).
    """

    def __init__(
        self,
        mass_operator: WeightedMassOperator,
        apply_bc: bool = True,
        dim_reduce: int | None = 0,
        weight_reduction: str = "midpoint",
        diagonal_scaling: bool = False,
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
        return self._mass_operator.matrix.diagonal()

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


class MassMatrixDiagonalPreconditioner(MassMatrixPreconditioner):
    r"""
    Preconditioner for inverting 3d weighted mass matrices. The mass matrix is approximated by

    .. math::
        D^{1/2} * \hat D^{-1/2} * \hat M * \hat D^{-1/2} * D^{1/2}

    Where $D$ is the diagonal of the matrix to invert, :math:`\hat M` is the mass matrix on the logical domain
    that is a Kronecker product (fastly inverted) and :math:`\hat D^{-1/2}` is the diagonal of :math:`\hat M`.

    This is ``MassMatrixPreconditioner(mass_operator, apply_bc, dim_reduce=None, diagonal_scaling=True)``;
    the class is kept for its name (e.g. in solver options).

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

    def __init__(self, mass_operator: WeightedMassOperator, apply_bc: bool = True):
        super().__init__(mass_operator, apply_bc=apply_bc, dim_reduce=None, diagonal_scaling=True)

    def transpose(self, conjugate: bool = False) -> MassMatrixDiagonalPreconditioner:
        """
        Returns the transposed operator.
        """
        return MassMatrixDiagonalPreconditioner(self._mass_operator.transpose(), self._apply_bc)


class StiffnessPreconditioner(KroneckerPreconditioner):
    r"""
    Preconditioner for the stabilized stiffness operators

    .. math::

        A = \mathbb d^T \mathbb M_{k+1} \mathbb d + \sigma \, \mathbb M_k \,,
        \qquad \mathbb d \in \{\mathbb G, \mathbb C, \mathbb D\} \ (k = 0, 1, 2)\,,

    i.e. :math:`\mathbb G^T \mathbb M_1 \mathbb G + \sigma \mathbb M_0`,
    :math:`\mathbb C^T \mathbb M_2 \mathbb C + \sigma \mathbb M_1` and
    :math:`\mathbb D^T \mathbb M_3 \mathbb D + \sigma \mathbb M_2`.

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
    e.g. for constant but anisotropic weights.

    Polar splines are not supported yet.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        The mass operators :math:`\mathbb M_k`.

    derivative : str
        ``"grad"``, ``"curl"`` or ``"div"``.

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
    """

    _FORMS = {"grad": 0, "curl": 1, "div": 2}

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        derivative: str = "grad",
        sigma: float = 0.0,
        apply_bc: bool = True,
        diagonal_scaling: bool = True,
        kernel_correction: bool = True,
        weights: str = "average",
    ):
        assert derivative in self._FORMS, f"derivative must be one of {tuple(self._FORMS)}, got {derivative!r}."
        assert sigma >= 0.0
        assert weights in ("unit", "average"), f"weights must be 'unit' or 'average', got {weights!r}."
        kernel_correction = kernel_correction and derivative != "grad"
        if kernel_correction and sigma == 0.0:
            raise ValueError("The kernel correction needs sigma > 0 (or kernel_correction=False).")

        derham = mass_ops.derham
        if derham.polar_splines:
            raise NotImplementedError("StiffnessPreconditioner does not support polar splines yet.")

        k = self._FORMS[derivative]
        self._mass_ops = mass_ops
        self._derivative = derivative
        self._sigma = sigma
        self._weights = weights
        self._mass_mid = getattr(mass_ops, f"M{k + 1}")
        # tensor-product derivative without boundary operators (no polar splines)
        self._d = getattr(derham, f"{derivative}_bcfree")

        super().__init__(getattr(mass_ops, f"M{k}"), apply_bc=apply_bc, diagonal_scaling=diagonal_scaling)

        # correction on the kernel of the derivative: sigma^{-1} d_prev P_prev d_prev^T
        self._kernel_correction = None
        if kernel_correction:
            prev = "grad" if derivative == "curl" else "curl"
            d_prev = getattr(derham, prev)
            P_prev = StiffnessPreconditioner(
                mass_ops,
                prev,
                apply_bc=apply_bc,
                diagonal_scaling=diagonal_scaling,
                kernel_correction=False,
                weights=weights,
            )
            self._kernel_correction = (1.0 / sigma) * (d_prev @ P_prev @ d_prev.T)
            self._tmp_kernel = self.codomain.zeros()

    @property
    def derivative(self) -> str:
        """``"grad"``, ``"curl"`` or ``"div"``."""
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

    def solve(self, rhs: Vector, out: Vector | None = None) -> Vector:
        """Apply the preconditioner (block Jacobi part plus kernel correction, if any) to rhs."""
        out = super().solve(rhs, out=out)
        if self._kernel_correction is not None:
            out += self._kernel_correction.dot(rhs, out=self._tmp_kernel)
        return out

    @property
    def core_operator(self) -> LinearOperator:
        """The core operator :math:`\\mathbb d^T \\mathbb M_{k+1} \\mathbb d + \\sigma \\mathbb M_k` on the tensor-product spaces."""
        A = self._d.T @ self._mass_mid.matrix @ self._d
        if self._sigma != 0.0:
            A = A + self._sigma * self._mass_operator.matrix
        return A

    def _build_approximation(self, bc):
        femspace = self._femspace
        is_scalar = isinstance(femspace, TensorFemSpace)
        comps = (femspace,) if is_scalar else femspace.spaces

        # directions in which each component is differentiated, and the codomain block
        stiff_dirs = _derivative_directions(self._d)

        # weights: common shape per direction and means per block (unit or separable average)
        n_mid = len(self._mass_mid.weights)
        n_dom = len(self._mass_operator.weights)
        if self._weights == "average":
            mid_weights = [self._mass_mid.weights[r][r] for r in range(n_mid)]
            shapes, mid_means = _separable_weight(mid_weights, self._mass_operator.derham)
            _, dom_means = _separable_weight(
                [self._mass_operator.weights[c][c] for c in range(n_dom)], self._mass_operator.derham
            )
        else:
            shapes, mid_means, dom_means = [_ones_1d] * 3, [1.0] * n_mid, [1.0] * n_dom

        matrixblocks = []
        solverblocks = []
        for c, comp in enumerate(comps):
            coeff_space = femspace.coeff_space if is_scalar else femspace.coeff_space[c]

            stiffness, mass, local_mass, local_stiffness = [], [], [], []
            for d in range(3):
                basis = comp.spaces[d].basis
                M_d, domain_decomposition = _dense_mass_1d(self._mass_operator, basis, d, shapes[d])

                S_d = None
                if d in stiff_dirs[c]:
                    assert basis == "B", "Only B-spline directions are differentiated."
                    M_D, _ = _dense_mass_1d(self._mass_operator, "M", d, shapes[d])
                    D_d = _difference_matrix_1d(M_d.shape[0], M_D.shape[0])
                    S_d = mid_means[stiff_dirs[c][d]] * (D_d.T @ M_D @ D_d)

                if bc is not None and basis == "B":
                    M_d = _apply_bc_dense(M_d, bc[d])
                    S_d = None if S_d is None else _apply_bc_dense(S_d, bc[d])

                stiffness.append(S_d)
                mass.append(M_d)
                local_mass.append(_process_local_matrix_1d(xp.asarray(M_d), comp.coeff_space, d, domain_decomposition))
                local_stiffness.append(
                    None
                    if S_d is None
                    else _process_local_matrix_1d(xp.asarray(S_d), comp.coeff_space, d, domain_decomposition)
                )

            sigma_c = self._sigma * dom_means[c]
            solverblocks.append(KroneckerSumSolver(coeff_space, stiffness, mass, sigma=sigma_c))
            matrixblocks.append(_kronecker_sum(coeff_space, local_stiffness, local_mass, sigma_c))

        if is_scalar:
            return matrixblocks[0], solverblocks[0]
        return (
            _block_diagonal(femspace.coeff_space, matrixblocks),
            _block_diagonal(femspace.coeff_space, solverblocks),
        )

    def _core_diagonal(self):
        space = self._matrix.domain
        return _diagonal_operator(space, _probe_diagonal(self.core_operator, space))

    def transpose(self, conjugate: bool = False) -> StiffnessPreconditioner:
        """The operator is symmetric, so is the preconditioner."""
        return self


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
    sums, scalings and BlockLinearOperators of these (missing diagonal blocks give zeros).
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
        return [op.diagonal()._data]
    if isinstance(op, KroneckerStencilMatrix):
        # outer product of the diagonals of the (process-local) factors on the local rows
        diags = []
        for A, s, e in zip(op.mats, op.codomain.starts, op.codomain.ends):
            assert A.domain.ndim == 1, "Only 1d factors are supported."
            off = A.codomain.pads[0] * A.codomain.shifts[0]
            diags.append(A._data[off : off + e - s + 1, A.pads[0]])
        out = diags[0]
        for d in diags[1:]:
            out = xp.multiply.outer(out, d)
        return [out]
    if isinstance(op, SumLinearOperator):
        parts = [_local_diagonal(a) for a in op.addends]
        return [sum(blocks) for blocks in zip(*parts)]
    if isinstance(op, ScaledLinearOperator):
        return [op.scalar * a for a in _local_diagonal(op.operator)]
    raise NotImplementedError(f"Diagonal of {type(op).__name__} is not supported.")


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
        3d weights (callable, array at the local quadrature points, or None for 1).

    derham : Derham
        Discrete de Rham sequence (quadrature grid).

    Returns
    -------
    shapes : list[xp.ndarray]
        :math:`\phi_d` at the global 1d quadrature points: the mean over the other directions,
        divided by the mean, averaged over the weights (mean 1).

    means : list[float]
        The means :math:`\bar w_i` of the weights over the logical cube.

    Collective on ``derham.comm`` for array weights.
    """
    grids = [derham.H1_1d_serial[d].get_assembly_grids(derham.nquads[d])[0] for d in range(3)]
    pts = [xp.asarray(np.ravel(g.points)) for g in grids]
    wts = [xp.asarray(np.ravel(g.weights)) for g in grids]

    profiles, means = [], []
    for w in weights:
        prof = []
        for d in range(3):
            p = _reduced_weight_1d(w, 0, d, derham, "average")
            prof.append(xp.asarray(p(pts[d]) if callable(p) else p, dtype=float))
        mean = float(xp.sum(prof[0] * wts[0]) / xp.sum(wts[0]))
        assert mean > 0.0, "The weights must have a positive mean."
        profiles.append(prof)
        means.append(mean)

    shapes = [sum(prof[d] / mean for prof, mean in zip(profiles, means)) / len(weights) for d in range(3)]
    return shapes, means


def _dense_mass_1d(
    mass_operator: WeightedMassOperator, basis: str, d: int, weight: Callable | xp.ndarray
) -> tuple[np.ndarray, DomainDecomposition]:
    """Global 1d mass matrix in direction d (B- or M-splines) as dense host array, see ``_mass_matrix_1d``."""
    M, domain_decomposition = _mass_matrix_1d(mass_operator, basis, d, weight)
    return np.asarray(xp.to_numpy(M.toarray()), dtype=float), domain_decomposition


def _difference_matrix_1d(n_B: int, n_M: int) -> np.ndarray:
    """
    1d derivative matrix from B-spline to M-spline coefficients, :math:`(\\mathbb d c)_r = c_{r+1} - c_r`
    (periodic if ``n_M == n_B``, else ``n_M == n_B - 1``).
    """
    assert n_M in (n_B, n_B - 1)
    D = np.zeros((n_M, n_B))
    for r in range(n_M):
        D[r, r] = -1.0
        D[r, (r + 1) % n_B] = 1.0
    return D


def _apply_bc_dense(A: np.ndarray, bc_d: tuple[bool, bool]) -> np.ndarray:
    """Impose essential boundary conditions on a dense 1d matrix: zero row and column, 1 on the diagonal."""
    A = A.copy()
    for i, is_essential in zip((0, -1), bc_d):
        if is_essential:
            A[i, :] = 0.0
            A[:, i] = 0.0
            A[i, i] = 1.0
    return A


def _kronecker_sum(
    space: StencilVectorSpace,
    stiffness: list[StencilMatrix | None],
    mass: list[StencilMatrix],
    sigma: float,
) -> LinearOperator:
    """
    The operator :math:`\\sum_d M_1 \\otimes \\dots \\otimes S_d \\otimes \\dots \\otimes M_3 + \\sigma M_1 \\otimes M_2 \\otimes M_3`
    as sum of KroneckerStencilMatrix (process-local 1d factors; directions with ``S_d = None`` have no term).
    """
    terms = []
    for d, S_d in enumerate(stiffness):
        if S_d is not None:
            factors = [S_d.copy() if e == d else M_e.copy() for e, M_e in enumerate(mass)]
            terms.append(KroneckerStencilMatrix(space, space, *factors))
    if sigma != 0.0:
        terms.append(KroneckerStencilMatrix(space, space, *[M_e.copy() for M_e in mass]) * sigma)
    assert terms, "The operator has no terms."
    return reduce(lambda a, b: a + b, terms)


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

    Parameters
    ----------
    circmat : xp.ndarray
        Generic circulant matrix.
    """

    def __init__(self, circmat):
        assert isinstance(circmat, xp.ndarray)
        assert is_circulant(circmat)

        from scipy.linalg import solve_circulant  # deferred: scipy.linalg is slow to import

        self._space = xp.ndarray
        # copy: circmat is still used by the caller (e.g. for the process-local stencil matrix)
        self._column = xp.array(circmat[:, 0], copy=True)

        # stabilize a singular matrix once, here, so that all solves use the same matrix
        try:
            solve_circulant(self._column, xp.ones_like(self._column))
        except xp.linalg.LinAlgError:
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
        The matrix that is checked to be circulant.

    Returns
    -------
    circulant : bool
        Whether the matrix is circulant (=True) or not (=False).
    """

    assert isinstance(mat, xp.ndarray)
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]

    # circulant: every row is the previous one shifted by one, i.e. mat[i, j] = mat[0, (j - i) % n]
    n = mat.shape[0]
    idx = (xp.arange(n)[None, :] - xp.arange(n)[:, None]) % n
    return bool(xp.allclose(mat, mat[0][idx]))
