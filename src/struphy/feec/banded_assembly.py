"""Assemble banded FEEC operators into scipy sparse matrices by probing."""

import itertools

import numpy as np
import scipy.sparse as sp
from feectools.linalg.utilities import array_to_psydac


def assemble_banded_operator(operator, space, bandwidth=None):
    """Sparse matrix of a linear operator on a (serial) 3D ``StencilVectorSpace``.

    Entry ``(i, j)`` is assumed to vanish unless ``|i_d - j_d| <= bandwidth[d]`` in every
    direction (modulo ``n_d`` in periodic directions). This holds for products of
    mass matrices and derivatives, e.g. ``grad.T @ M1 @ grad`` with ``bandwidth = pads``.
    Columns whose indices differ by more than ``2 * bandwidth`` in some direction do
    not share rows, so they can be probed together: only ``prod(2 b_d + 1)``
    applications of ``operator`` are needed.

    Parameters
    ----------
    operator : feectools.linalg.basic.LinearOperator
        Operator from ``space`` to ``space``.

    space : feectools.linalg.stencil.StencilVectorSpace
        Serial coefficient space; the flat ordering is that of ``Vector.toarray()``.

    bandwidth : tuple[int], optional
        Band half-width in each direction; defaults to ``space.pads``.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    shape = tuple(int(n) for n in space.npts)
    bandwidth = tuple(space.pads) if bandwidth is None else tuple(bandwidth)
    periodic = tuple(space.periods)

    strides = []
    for n, b, per in zip(shape, bandwidth, periodic):
        s = 2 * b + 1
        if per:
            # colors must tile the periodic direction evenly
            s = next((d for d in range(s, n + 1) if n % d == 0), n)
        strides.append(min(s, n))

    index = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    rows, cols, vals = [], [], []
    for color in itertools.product(*[range(s) for s in strides]):
        probe = np.ones(shape, dtype=bool)
        for d in range(3):
            probe &= index[d] % strides[d] == color[d]
        result = operator.dot(array_to_psydac(probe.ravel().astype(float), space)).toarray()
        nonzero = np.nonzero(result)[0]
        row_index = np.unravel_index(nonzero, shape)
        col_index = []
        for d in range(3):
            i, s, b, n = row_index[d], strides[d], bandwidth[d], shape[d]
            if s == n:
                j = np.full_like(i, color[d])  # every index is its own color
            else:
                j = i - b + np.mod(color[d] - (i - b), s)
                if periodic[d]:
                    j = np.mod(j, n)
            col_index.append(j)
        rows.append(nonzero)
        cols.append(np.ravel_multi_index(col_index, shape))
        vals.append(result[nonzero])

    size = int(np.prod(shape))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size))
