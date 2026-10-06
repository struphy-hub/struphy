"""Parity-test arguments of ``kernel_pullpush_pic``: pull-backs, push-forwards and transformations at 129 markers.

Every ``kind_fun`` of pull (0, 1, 10, 11, 12), push (the same) and tran (0, 1, 10-21) is run once; the mapping cycles
through all analytic mappings, so each mapping is used two or three times, and each spline mapping gets three
further transforms. Every third case passes ``a`` without
holes (one row per inside marker), and ``remove_outside`` alternates.
"""

import cunumpy as xp
import numpy as np

from struphy.geometry.base import inside_logical_cube
from struphy.pic.tests.kernel_test_args import (
    N_ANALYTIC_DOMAINS,
    N_GEOMETRY_DOMAINS,
    geometry_domain,
    logical_markers,
)

RTOL = 1e-10
ATOL = 1e-10

PULLPUSH_KINDS = (0, 1, 10, 11, 12)
TRAN_KINDS = (0, 1, *range(10, 22))
TRANSFORMS = (
    tuple((0, k) for k in PULLPUSH_KINDS) + tuple((1, k) for k in PULLPUSH_KINDS) + tuple((2, k) for k in TRAN_KINDS)
)

# (index into geometry_domain(), kind_transform, kind_fun, a_has_holes, remove_outside); the analytic mappings
# cycle through all transforms, then each spline mapping gets three of them
CASES = tuple(
    (n % N_ANALYTIC_DOMAINS, kind_transform, kind_fun, n % 3 != 2, n % 2 == 0)
    for n, (kind_transform, kind_fun) in enumerate(TRANSFORMS)
) + tuple(
    (N_ANALYTIC_DOMAINS + n // 3, kind_transform, kind_fun, n % 3 != 2, n % 2 == 1)
    for n, (kind_transform, kind_fun) in enumerate(
        TRANSFORMS[(7 * j + 3 * s) % len(TRANSFORMS)]
        for s in range(N_GEOMETRY_DOMAINS - N_ANALYTIC_DOMAINS)
        for j in range(3)
    )
)


def N_THREADS(args):
    """One thread per marker row (args[1]); the first array, a, has fewer rows when it has no holes."""
    return args[1].shape[0]


def make_args(backend, seed):
    domain, kind_transform, kind_fun, a_has_holes, remove_outside = CASES[seed]
    markers = logical_markers()
    rng = np.random.default_rng(9)
    n_comp = 1 if kind_fun < 10 else 3
    a = rng.normal(size=(markers.shape[0], n_comp))
    if not a_has_holes:
        a = a[inside_logical_cube(markers)]
    out = rng.random((markers.shape[0], 3))
    return (
        xp.asarray(a),
        xp.asarray(markers),
        kind_transform,
        kind_fun,
        geometry_domain(domain).args_domain,
        xp.asarray(out),
        remove_outside,
    )
