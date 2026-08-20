"""Memory footprint of FEEC objects (coefficient spaces, vectors and matrices).

The functions in this module never allocate any of the (potentially large) arrays they measure:
they either read the metadata of an existing object or the ``nbytes`` of an already allocated one.
They are used to estimate the memory usage of a simulation before allocating it, see
:meth:`struphy.simulation.sim.Simulation.estimate_mem`.
"""

import logging

from feectools.linalg.block import BlockLinearOperator

logger = logging.getLogger("struphy")


def vector_nbytes(vector) -> int:
    """Actual local (per-MPI-rank) memory footprint, in bytes, of an *allocated*
    :class:`~feectools.linalg.stencil.StencilVector` or
    :class:`~feectools.linalg.block.BlockVector`, including the ghost/padding regions."""
    if hasattr(vector, "_data"):
        return int(vector._data.nbytes)
    if hasattr(vector, "blocks"):
        return sum(vector_nbytes(block) for block in vector.blocks)
    logger.debug(f"Cannot determine the memory footprint of a {type(vector).__name__}, counting 0 bytes.")
    return 0


def coeff_space_nbytes(space, float_size: int = 8) -> int:
    """Local (per-MPI-rank) memory footprint, in bytes, of a coefficient space
    (:class:`~feectools.linalg.stencil.StencilVectorSpace` or
    :class:`~feectools.linalg.block.BlockVectorSpace`), computed from its (metadata-only)
    local array ``shape`` -- no array is allocated."""
    if hasattr(space, "spaces"):
        return sum(coeff_space_nbytes(s, float_size=float_size) for s in space.spaces)
    nbytes = float_size
    for n in space.shape:
        nbytes *= n
    return int(nbytes)


def linop_nbytes(op, _seen: set = None) -> int:
    """Local (per-MPI-rank) memory footprint, in bytes, of the matrices stored in an
    (already allocated) linear operator.

    Composite operators (compositions, sums, scalings, block operators, polar operators)
    are traversed recursively; operators appearing more than once in the tree (for example
    ``curl.T @ M2 @ curl``) are counted once. Operators that do not store a matrix
    (identity, matrix-free, boundary/extraction masks) contribute zero.
    """
    if _seen is None:
        _seen = set()

    if op is None or id(op) in _seen:
        return 0
    _seen.add(id(op))

    # composite operators: recurse into the children
    if isinstance(op, BlockLinearOperator):
        return sum(linop_nbytes(block, _seen) for row in op.blocks for block in row)

    for attr in ("multiplicants", "addends", "mats"):
        if hasattr(op, attr):
            return sum(linop_nbytes(child, _seen) for child in getattr(op, attr))

    for attr in ("operator", "tp_operator"):
        if hasattr(op, attr):
            return linop_nbytes(getattr(op, attr), _seen)

    # leaf operator: has its own data array (StencilMatrix, StencilDiagonalMatrix, ...)
    return int(getattr(op, "nbytes", 0))
