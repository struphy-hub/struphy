r"""Block operators whose blocks are sums of Kronecker products of dense global 1d matrices.

They approximate the tensor-product operators of the discrete de Rham sequence (derivatives, basis projection
operators) in the stiffness preconditioners (:class:`~struphy.feec.preconditioner.StiffnessPreconditioner`):
derivatives are exactly of this form, basis projection operators after a separable approximation of their
weights.
"""

from __future__ import annotations

from collections.abc import Callable

import cunumpy as xp
import numpy as np
from feectools.core.bsplines import collocation_matrix, histopolation_matrix
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.fem.splines import SplineSpace
from feectools.fem.tensor import TensorFemSpace
from feectools.linalg.block import BlockLinearOperator

from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.psydac_derham import get_pts_and_wts

# a term c * F_1 x F_2 x F_3; a factor None is the identity
Term = tuple[float, tuple[np.ndarray | None, ...]]

# number of Gauss-Legendre points per direction for averaging callable weights
_N_AVG = 16


class KroneckerBlockOperator:
    r"""
    Block operator :math:`L` from a vector-valued (or scalar) tensor-product spline space to another one, whose
    blocks are sums of Kronecker products of dense global 1d matrices,

    .. math::

        L_{rc} = \sum_t c_t \, F_{t,1} \otimes F_{t,2} \otimes F_{t,3} \,,

    with :math:`F_{t,d}` of shape ``(codomain_shapes[r][d], domain_shapes[c][d])``. A factor None stands for the
    identity (equal sizes). Composition (``@``), transposition (``.T``) and scaling (``*``) give operators of the
    same kind.

    Parameters
    ----------
    blocks : dict
        ``(r, c) -> list of (c_t, (F_1, F_2, F_3))``; missing blocks are zero.

    domain_shapes, codomain_shapes : list[tuple[int, int, int]]
        Number of coefficients per direction of each component of the domain and codomain.

    domain_space, codomain_space : str
        Symbolic names of the domain and codomain (``"H1"``, ``"Hcurl"``, ``"Hdiv"``, ``"L2"`` or ``"H1vec"``).
    """

    def __init__(
        self,
        blocks: dict[tuple[int, int], list[Term]],
        domain_shapes: list[tuple[int, ...]],
        codomain_shapes: list[tuple[int, ...]],
        domain_space: str,
        codomain_space: str,
    ):
        self._domain_shapes = [tuple(s) for s in domain_shapes]
        self._codomain_shapes = [tuple(s) for s in codomain_shapes]
        self._domain_space = domain_space
        self._codomain_space = codomain_space
        self._blocks = {}
        for (r, c), terms in blocks.items():
            terms = [(float(coeff), tuple(factors)) for coeff, factors in terms if coeff != 0.0]
            for _, factors in terms:
                assert len(factors) == 3
                for d, F in enumerate(factors):
                    shape = (self._codomain_shapes[r][d], self._domain_shapes[c][d])
                    if F is None:
                        assert shape[0] == shape[1], (
                            f"Identity factor of block {(r, c)} in direction {d} is not square."
                        )
                    else:
                        assert F.shape == shape, f"Factor of block {(r, c)} in direction {d} has shape {F.shape}."
            if terms:
                self._blocks[(r, c)] = terms

    @property
    def blocks(self) -> dict[tuple[int, int], list[Term]]:
        """``(r, c) -> list of (c_t, (F_1, F_2, F_3))`` (non-zero blocks only)."""
        return self._blocks

    @property
    def domain_shapes(self) -> list[tuple[int, ...]]:
        return self._domain_shapes

    @property
    def codomain_shapes(self) -> list[tuple[int, ...]]:
        return self._codomain_shapes

    @property
    def domain_space(self) -> str:
        return self._domain_space

    @property
    def codomain_space(self) -> str:
        return self._codomain_space

    @property
    def T(self) -> KroneckerBlockOperator:
        """The transposed operator."""
        blocks = {
            (c, r): [(coeff, tuple(None if F is None else F.T for F in factors)) for coeff, factors in terms]
            for (r, c), terms in self._blocks.items()
        }
        return KroneckerBlockOperator(
            blocks, self._codomain_shapes, self._domain_shapes, self._codomain_space, self._domain_space
        )

    def __mul__(self, a: float) -> KroneckerBlockOperator:
        blocks = {key: [(a * coeff, factors) for coeff, factors in terms] for key, terms in self._blocks.items()}
        return KroneckerBlockOperator(
            blocks, self._domain_shapes, self._codomain_shapes, self._domain_space, self._codomain_space
        )

    __rmul__ = __mul__

    def __matmul__(self, other: KroneckerBlockOperator) -> KroneckerBlockOperator:
        """Composition ``self @ other``."""
        assert isinstance(other, KroneckerBlockOperator)
        assert self._domain_shapes == other._codomain_shapes, "Domain of the left factor != codomain of the right one."
        blocks = {}
        for (r, k), left in self._blocks.items():
            for (k2, c), right in other._blocks.items():
                if k2 != k:
                    continue
                terms = blocks.setdefault((r, c), [])
                for a, F in left:
                    for b, G in right:
                        terms.append((a * b, tuple(_matmul(f, g) for f, g in zip(F, G))))
        return KroneckerBlockOperator(
            blocks, other._domain_shapes, self._codomain_shapes, other._domain_space, self._codomain_space
        )

    def toarray(self) -> np.ndarray:
        """Dense global matrix (components stacked, C-ordered tensor-product index; for tests on small grids)."""
        rows = np.cumsum([0] + [int(np.prod(s)) for s in self._codomain_shapes])
        cols = np.cumsum([0] + [int(np.prod(s)) for s in self._domain_shapes])
        A = np.zeros((rows[-1], cols[-1]))
        for (r, c), terms in self._blocks.items():
            for coeff, factors in terms:
                mats = [np.eye(n) if F is None else F for F, n in zip(factors, self._domain_shapes[c])]
                A[rows[r] : rows[r + 1], cols[c] : cols[c + 1]] += coeff * np.kron(np.kron(mats[0], mats[1]), mats[2])
        return A


def _matmul(F: np.ndarray | None, G: np.ndarray | None) -> np.ndarray | None:
    """Product of two 1d factors (None is the identity)."""
    if F is None:
        return G
    if G is None:
        return F
    return F @ G


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def _component_spaces(femspace) -> list[TensorFemSpace]:
    """The scalar tensor-product spaces of the components of a (vector) FEM space."""
    return [femspace] if isinstance(femspace, TensorFemSpace) else list(femspace.spaces)


def _shapes(femspace) -> list[tuple[int, ...]]:
    return [tuple(V.nbasis for V in comp.spaces) for comp in _component_spaces(femspace)]


def derivative_operator(derham, derivative: str) -> KroneckerBlockOperator:
    """
    The tensor-product derivative ``"grad"``, ``"curl"`` or ``"div"`` of the de Rham sequence (without boundary
    operators) as KroneckerBlockOperator (exact).
    """
    forms = {"grad": ("H1", "Hcurl"), "curl": ("Hcurl", "Hdiv"), "div": ("Hdiv", "L2")}
    assert derivative in forms, f"derivative must be one of {tuple(forms)}, got {derivative!r}."
    dom, cod = forms[derivative]
    op = getattr(derham, f"{derivative}_bcfree")
    if isinstance(op, BlockLinearOperator):
        items = [((r, c), op[r, c]) for r in range(op.n_block_rows) for c in range(op.n_block_cols)]
    else:
        items = [((0, 0), op)]

    blocks = {}
    for (r, c), D in items:
        if D is None:
            continue
        assert isinstance(D, DirectionalDerivativeOperator) and not D.transposed
        # the 1d factor of the block itself (includes the sign of a negated block)
        D_1d = np.asarray(xp.to_numpy(D.tokronstencil().mats[D.diffdir].toarray()), dtype=float)
        factors = [None, None, None]
        factors[D.diffdir] = D_1d
        blocks[(r, c)] = [(1.0, tuple(factors))]

    return KroneckerBlockOperator(blocks, _shapes(derham.fem_spaces[dom]), _shapes(derham.fem_spaces[cod]), dom, cod)


def basis_projection_operator(op, derham, rtol: float = 1e-10) -> KroneckerBlockOperator:
    r"""
    Kronecker approximation of a (tensor-product, not transposed) basis projection operator
    :math:`\hat \Pi^\beta[A_{\mu\nu} \Lambda^\alpha_\nu]`
    (:class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator`).

    Each weight block is approximated by its mean times a separable shape,
    :math:`A_{\mu\nu} \approx \bar A_{\mu\nu} \prod_d \phi^{\mu\nu}_d(\eta_d)`, where :math:`\phi_d` is the mean over the
    other directions divided by :math:`\bar A_{\mu\nu}`; blocks whose mean vanishes (relative to the largest one, up to
    ``rtol``) are dropped. The 1d factors are the 1d projections of the weighted 1d basis functions,
    :math:`\hat \Pi_d[\phi_d \Lambda_d]`, with the inter-/histopolation of the 3d projector (Greville points,
    histopolation on the extended Greville grid with ``derham.nquads_proj`` Gauss points per cell, split at the breaks
    for even degree) and its essential boundary
    conditions (the boundary degrees of freedom and coefficients are zero). This is exact for constant weights.
    """
    from struphy.feec.basis_projection_ops import BasisProjectionOperator

    assert isinstance(op, BasisProjectionOperator)
    assert not op.transposed, "Approximate the operator itself and transpose the result."
    P, V = op._P, op._V
    assert not P.is_polar, "Polar splines are not supported."
    dom, cod = op._domain_symbolic_name, op._codomain_symbolic_name
    dom_comps = _component_spaces(V)
    cod_comps = _component_spaces(P.space)
    src_bc = _essential_bc(op._V_boundary_op, dom, len(dom_comps))
    tgt_bc = _essential_bc(op._P_boundary_op, cod, len(cod_comps))

    # separable approximation of the weight blocks
    approx = {}
    for r, row in enumerate(op._weights):
        for c, w in enumerate(row):
            if w is not None:
                approx[(r, c)] = separable_weight(w)
    largest = max((abs(m) for m, _ in approx.values()), default=0.0)
    assert largest > 0.0, "All weights of the basis projection operator vanish."

    blocks = {}
    for (r, c), (mean, shapes) in approx.items():
        if abs(mean) <= rtol * largest:
            continue
        factors = tuple(
            _projection_1d(
                cod_comps[r].spaces[d],
                dom_comps[c].spaces[d],
                shapes[d],
                derham.nquads_proj[d],
                tgt_bc[r][d],
                src_bc[c][d],
            )
            for d in range(3)
        )
        blocks[(r, c)] = [(mean, factors)]

    return KroneckerBlockOperator(blocks, _shapes(V), _shapes(P.space), dom, cod)


def separable_weight(fun: Callable) -> tuple[float, list[Callable]]:
    r"""
    Separable approximation :math:`w \approx \bar w \prod_d \phi_d(\eta_d)` of a 3d weight on the logical cube.

    Parameters
    ----------
    fun : callable
        The weight, evaluated on 3d meshgrids (``indexing="ij"``) of the logical coordinates.

    Returns
    -------
    mean : float
        The mean :math:`\bar w` (Gauss-Legendre quadrature with ``_N_AVG`` points per direction).

    shapes : list[callable]
        :math:`\phi_d`, the mean over the other two directions divided by :math:`\bar w` (functions of 1d point
        arrays; 1 if the mean vanishes).
    """
    nodes, wts = np.polynomial.legendre.leggauss(_N_AVG)
    nodes, wts = (nodes + 1.0) / 2.0, wts / 2.0

    def profile(d: int, x: np.ndarray) -> np.ndarray:
        # mean over the other directions at the points x along d
        axes = [np.asarray(x, dtype=float).ravel() if e == d else nodes for e in range(3)]
        E = np.meshgrid(*axes, indexing="ij")
        W = np.asarray(xp.to_numpy(fun(*[xp.asarray(e) for e in E])), dtype=float).reshape(E[0].shape)
        for e in (2, 1, 0):
            if e != d:
                W = np.tensordot(W, wts, axes=([e], [0]))
        return W

    mean = float(profile(0, nodes) @ wts)
    if mean == 0.0:
        return 0.0, [lambda x: np.ones(np.size(x))] * 3
    return mean, [lambda x, d=d: profile(d, x) / mean for d in range(3)]


def _essential_bc(boundary_op, space: str, ncomps: int) -> list[list[tuple[bool, bool]]]:
    """Per component and direction, the (left, right) flags of essential boundary conditions of a boundary operator
    (identity: none). They apply in the B-spline directions (H1vec: component c in direction c)."""
    no_bc = [[(False, False)] * 3 for _ in range(ncomps)]
    if not isinstance(boundary_op, BoundaryOperator):
        return no_bc
    bc = boundary_op.bc
    if space == "H1vec":
        return [[tuple(bc[d]) if d == c else (False, False) for d in range(3)] for c in range(ncomps)]
    # B-spline directions: H1 all; Hcurl comp c all but c; Hdiv comp c only c; L2 none
    b_dirs = {
        "H1": lambda c, d: True,
        "Hcurl": lambda c, d: d != c,
        "Hdiv": lambda c, d: d == c,
        "L2": lambda c, d: False,
    }[space]
    return [[tuple(bc[d]) if b_dirs(c, d) else (False, False) for d in range(3)] for c in range(ncomps)]


def _projection_1d(
    target: SplineSpace,
    source: SplineSpace,
    shape: Callable,
    nquad: int,
    target_bc: tuple[bool, bool],
    source_bc: tuple[bool, bool],
) -> np.ndarray | None:
    r"""
    1d projection :math:`\hat \Pi[\phi \Lambda_j]` of the weighted source basis functions into the target space:
    interpolation at the Greville points for B-splines, histopolation on the extended Greville grid for M-splines
    (as in the 3d commuting projectors). With essential boundary conditions, the boundary degrees of freedom of the
    target (before the inversion), the boundary coefficients of the target and the boundary basis functions of the
    source are set to zero. Returns None for the identity (unit weight, same space, no boundary conditions).
    """
    if target.basis == "B":
        x = np.asarray(target.greville, dtype=float)
        dofs = _values(source, x) * shape(x)[:, None]
        imat = _values(target, x)
    else:
        # the points of BasisProjectionOperator: Greville cells split at the breaks for even degree (exact integration
        # of the piecewise polynomials); the integrals over the sub-intervals (subs = 1) add up to one degree of freedom
        quad_x, quad_w, subs = get_pts_and_wts(target, 0, target.nbasis - 1, n_quad=nquad)
        n_int, nq = quad_x.shape
        vals = _values(source, quad_x.ravel()).reshape(n_int, nq, -1)
        integrals = np.einsum("iq,iq,iqj->ij", quad_w, shape(quad_x.ravel()).reshape(n_int, nq), vals)
        dof_index = np.cumsum(1 - subs) - 1
        dofs = np.zeros((target.nbasis, integrals.shape[1]))
        np.add.at(dofs, dof_index, integrals)
        imat = np.asarray(
            xp.to_numpy(
                histopolation_matrix(
                    target.knots,
                    target.degree,
                    target.periodic,
                    target.basis,
                    target.ext_greville,
                    multiplicity=target.multiplicity,
                )
            ),
            dtype=float,
        )

    # B I^{-1} B^T applied to the dofs (as BasisProjectionOperator with apply_bc=True)
    for i, essential in zip((0, -1), target_bc):
        if essential:
            dofs[i, :] = 0.0
    F = np.linalg.solve(imat, dofs)
    for i, essential in zip((0, -1), target_bc):
        if essential:
            F[i, :] = 0.0
    for j, essential in zip((0, -1), source_bc):
        if essential:
            F[:, j] = 0.0

    if F.shape[0] == F.shape[1] and np.allclose(F, np.eye(F.shape[0]), rtol=0.0, atol=1e-13):
        return None
    return F


def _values(space: SplineSpace, x: np.ndarray) -> np.ndarray:
    """Values of the basis functions of a 1d spline space at the points x, shape ``(x.size, nbasis)``."""
    if space.nbasis == 1 and x.size == 1:
        return np.ones((1, 1))
    return np.asarray(
        xp.to_numpy(
            collocation_matrix(
                space.knots, space.degree, space.periodic, space.basis, x, multiplicity=space.multiplicity
            )
        ),
        dtype=float,
    )
