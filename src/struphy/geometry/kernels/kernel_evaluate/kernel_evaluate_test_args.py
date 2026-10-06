"""Parity-test arguments of ``kernel_evaluate``: metric coefficients on a 7 x 5 x 4 grid, for every analytic mapping.

Each mapping is evaluated on a full and on a sparse meshgrid; the metric coefficient (``kind_coeff`` -1 to 5) and
``avoid_round_off`` change from case to case, so every coefficient is evaluated for several mappings.
"""

import cunumpy as xp
import numpy as np

from struphy.pic.tests.kernel_test_args import N_ANALYTIC_DOMAINS, analytic_domains, evaluation_grid

RTOL = 1e-10
ATOL = 1e-10

KIND_COEFFS = (-1, 0, 1, 2, 3, 4, 5)

# (index into analytic_domains(), kind_coeff, is_sparse_meshgrid, avoid_round_off)
CASES = tuple(
    (domain, KIND_COEFFS[(2 * domain + sparse) % len(KIND_COEFFS)], bool(sparse), (domain + sparse) % 2 == 0)
    for domain in range(N_ANALYTIC_DOMAINS)
    for sparse in (0, 1)
)


def N_THREADS(args):
    """One thread per grid point: the first three axes of mat_f (args[5], shape (n1, n2, n3, 3, 3))."""
    return args[5].size // 9


def make_args(backend, seed):
    domain, kind_coeff, is_sparse_meshgrid, avoid_round_off = CASES[seed]
    eta1, eta2, eta3 = evaluation_grid(sparse=is_sparse_meshgrid)
    mat_f = np.random.default_rng(8).random((7, 5, 4, 3, 3))
    return (
        eta1,
        eta2,
        eta3,
        kind_coeff,
        analytic_domains()[domain].args_domain,
        xp.asarray(mat_f),
        is_sparse_meshgrid,
        avoid_round_off,
    )
