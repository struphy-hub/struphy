r"""Re-discretization of a linear operator on a coarser Derham.

A (composite) operator on the fine level is walked as an expression tree. Composite nodes (sums,
compositions, scalings, powers, block operators) are rebuilt from their coarsened children, scalars are
kept. Leaves are re-created on the coarse Derham:

* :class:`IdentityOperator`, :class:`ZeroOperator`, :class:`BoundaryOperator`: on the corresponding coarse spaces,
* :class:`DirectionalDerivativeOperator` (blocks of ``derham.grad``, ``curl``, ``div`` and their transposes),
* :class:`WeightedMassOperator` and :class:`BasisProjectionOperator`: from their ``to_dict()`` recipe,
* :class:`WeightedAverageProjection`: from its coarsened weighting operator.

Further leaf types can be supported with :func:`register_coarsening`.
"""

from collections.abc import Callable

from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.linalg.basic import (
    ComposedLinearOperator,
    IdentityOperator,
    LinearOperator,
    PowerLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    VectorSpace,
    ZeroOperator,
)
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace

from struphy.feec.basis_projection_ops import BasisProjectionOperator
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedAverageProjection, WeightedMassOperator, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain

_REGISTRY: dict[type, Callable[[LinearOperator, "OperatorCoarsener"], LinearOperator]] = {}


def register_coarsening(cls: type):
    """Decorator registering ``fun(op, coarsener) -> LinearOperator`` as the coarsening rule for leaves of type ``cls``."""

    def decorator(fun):
        _REGISTRY[cls] = fun
        return fun

    return decorator


class OperatorCoarsener:
    r"""Maps linear operators on the coefficient spaces of ``fine`` to the corresponding operators on ``coarse``.

    Coarse leaves are cached (keyed by the fine leaf object), hence calling the coarsener again on an
    operator that differs only in its scalars or composition (e.g. ``sigma * M0 + grad.T @ M1 @ grad``
    with a new ``sigma``) re-assembles nothing.

    Parameters
    ----------
    fine, coarse : Derham
        Fine and coarse level (same options, nested grids).

    domain : Domain
        Mapping used for re-assembling mass matrices on the coarse level.

    matrix_free : bool
        Whether coarse mass matrices are matrix-free.
    """

    def __init__(self, fine: Derham, coarse: Derham, domain: Domain, *, matrix_free: bool = False):
        self._fine = fine
        self._coarse = coarse
        self._mass_ops = WeightedMassOperators(coarse, domain, matrix_free=matrix_free)

        # fine -> coarse coefficient spaces (also components of block spaces)
        self._spaces: dict[int, tuple[VectorSpace, VectorSpace]] = {}
        for form in ("0", "1", "2", "3", "v"):
            Vf, Vc = fine.coeff_spaces[form], coarse.coeff_spaces[form]
            self._spaces[id(Vf)] = (Vf, Vc)
            if isinstance(Vf, BlockVectorSpace):
                for vf, vc in zip(Vf.spaces, Vc.spaces):
                    self._spaces.setdefault(id(vf), (vf, vc))

        self._cache: dict[int, tuple[LinearOperator, LinearOperator]] = {}

    @property
    def fine(self) -> Derham:
        return self._fine

    @property
    def coarse(self) -> Derham:
        return self._coarse

    @property
    def mass_ops(self) -> WeightedMassOperators:
        """Mass operators of the coarse level."""
        return self._mass_ops

    def space(self, V: VectorSpace) -> VectorSpace:
        """Coarse counterpart of the fine coefficient space ``V``."""
        try:
            return self._spaces[id(V)][1]
        except KeyError:
            raise ValueError(f"{V} is not a coefficient space of the fine Derham.") from None

    def __call__(self, A: LinearOperator) -> LinearOperator:
        """Return the coarse-level version of the fine-level operator ``A``."""
        if isinstance(A, ScaledLinearOperator):
            B = self(A.operator)
            return ScaledLinearOperator(B.domain, B.codomain, A.scalar, B)

        if isinstance(A, SumLinearOperator):
            addends = [self(a) for a in A.addends]
            return SumLinearOperator(self.space(A.domain), self.space(A.codomain), *addends)

        if isinstance(A, ComposedLinearOperator):
            factors = [self(a) for a in A.multiplicands]
            return ComposedLinearOperator(self.space(A.domain), self.space(A.codomain), *factors)

        if isinstance(A, PowerLinearOperator):
            return PowerLinearOperator(self.space(A.domain), self.space(A.codomain), self(A.operator), A.factorial)

        if isinstance(A, BlockLinearOperator):
            blocks = {ij: self(A[ij]) for ij in A.nonzero_block_indices}
            return BlockLinearOperator(self.space(A.domain), self.space(A.codomain), blocks=blocks)

        if isinstance(A, IdentityOperator):
            return IdentityOperator(self.space(A.domain), self.space(A.codomain))

        if isinstance(A, ZeroOperator):
            return ZeroOperator(self.space(A.domain), self.space(A.codomain))

        # leaves: cached
        cached = self._cache.get(id(A))
        if cached is not None and cached[0] is A:
            return cached[1]

        for cls in type(A).__mro__:
            if cls in _REGISTRY:
                B = _REGISTRY[cls](A, self)
                break
        else:
            raise NotImplementedError(
                f"Cannot re-discretize an operator of type {type(A).__name__} on a coarse grid; "
                "use struphy operators or register a rule with register_coarsening."
            )

        assert B.domain is self.space(A.domain) and B.codomain is self.space(A.codomain), (
            f"Coarsening of {type(A).__name__} gave wrong (co)domain."
        )
        self._cache[id(A)] = (A, B)
        return B


@register_coarsening(DirectionalDerivativeOperator)
def _coarsen_derivative(A: DirectionalDerivativeOperator, c: OperatorCoarsener) -> LinearOperator:
    return DirectionalDerivativeOperator(
        c.space(A._spaceV),
        c.space(A._spaceW),
        A._diffdir,
        negative=A._negative,
        transposed=A._transposed,
    )


@register_coarsening(BoundaryOperator)
def _coarsen_boundary(A: BoundaryOperator, c: OperatorCoarsener) -> LinearOperator:
    return BoundaryOperator(c.space(A.domain), A._space_id, A.bc)


@register_coarsening(WeightedMassOperator)
def _coarsen_mass(A: WeightedMassOperator, c: OperatorCoarsener) -> LinearOperator:
    return WeightedMassOperator.from_dict(A.to_dict(), c.mass_ops)


@register_coarsening(BasisProjectionOperator)
def _coarsen_basis_projection(A: BasisProjectionOperator, c: OperatorCoarsener) -> LinearOperator:
    return BasisProjectionOperator.from_dict(A.to_dict(), c.coarse)


@register_coarsening(WeightedAverageProjection)
def _coarsen_average(A: WeightedAverageProjection, c: OperatorCoarsener) -> LinearOperator:
    return WeightedAverageProjection(c.coarse, c(A._S), A._directions)
