"""Parity-test arguments of ``kernel_evaluate_pic``: metric coefficients at 129 markers, for every analytic mapping.

Every mapping is evaluated for the mapping F, det(DF), DF^(-1) and G^(-1) (which runs DF and G); the identity, DF
and G are evaluated for a diagonal and two non-diagonal mappings. ``remove_outside`` and ``avoid_round_off``
alternate, so both the compacted and the uncompacted output (holes and outside markers) are compared.
"""

import cunumpy as xp
import numpy as np

from struphy.pic.tests.kernel_test_args import N_ANALYTIC_DOMAINS, analytic_domains, logical_markers

RTOL = 1e-10
ATOL = 1e-10

# (index into analytic_domains(), kind_coeff, remove_outside, avoid_round_off)
CASES = tuple(
    (domain, kind_coeff, (domain + n) % 2 == 0, n % 2 == 1)
    for domain in range(N_ANALYTIC_DOMAINS)
    for n, kind_coeff in enumerate((0, 2, 3, 5))
) + tuple((domain, kind_coeff, kind_coeff == 1, True) for domain in (1, 2, 6) for kind_coeff in (-1, 1, 4))


def make_args(backend, seed):
    domain, kind_coeff, remove_outside, avoid_round_off = CASES[seed]
    markers = logical_markers()
    mat_f = np.random.default_rng(6).random((markers.shape[0], 3, 3))
    return (
        xp.asarray(markers),
        kind_coeff,
        analytic_domains()[domain].args_domain,
        xp.asarray(mat_f),
        remove_outside,
        avoid_round_off,
    )
