"""Parity-test arguments of ``kernel_pullpush``: pull-backs, push-forwards and transformations on a 7 x 5 x 4 grid.

Every ``kind_fun`` of pull (0, 1, 10, 11, 12), push (the same) and tran (0, 1, 10-21) is run once; the mapping cycles
through all analytic mappings, so each mapping is used two or three times, and full and sparse meshgrids alternate.
"""

import cunumpy as xp
import numpy as np

from struphy.pic.tests.kernel_test_args import N_ANALYTIC_DOMAINS, analytic_domains, evaluation_grid

RTOL = 1e-10
ATOL = 1e-10

PULLPUSH_KINDS = (0, 1, 10, 11, 12)
TRAN_KINDS = (0, 1, *range(10, 22))
TRANSFORMS = (
    tuple((0, k) for k in PULLPUSH_KINDS) + tuple((1, k) for k in PULLPUSH_KINDS) + tuple((2, k) for k in TRAN_KINDS)
)

# (index into analytic_domains(), kind_transform, kind_fun, is_sparse_meshgrid)
CASES = tuple(
    ((3 * n) % N_ANALYTIC_DOMAINS, kind_transform, kind_fun, n % 2 == 1)
    for n, (kind_transform, kind_fun) in enumerate(TRANSFORMS)
)


def N_THREADS(args):
    """One thread per grid point: the first three axes of out (the last argument, shape (n1, n2, n3, 3))."""
    return args[-1].size // 3


def make_args(backend, seed):
    domain, kind_transform, kind_fun, is_sparse_meshgrid = CASES[seed]
    eta1, eta2, eta3 = evaluation_grid(sparse=is_sparse_meshgrid)
    rng = np.random.default_rng(10)
    n_comp = 1 if kind_fun < 10 else 3
    return (
        xp.asarray(rng.normal(size=(7, 5, 4, n_comp))),
        eta1,
        eta2,
        eta3,
        kind_transform,
        kind_fun,
        analytic_domains()[domain].args_domain,
        is_sparse_meshgrid,
        xp.asarray(rng.random((7, 5, 4, 3))),
    )
