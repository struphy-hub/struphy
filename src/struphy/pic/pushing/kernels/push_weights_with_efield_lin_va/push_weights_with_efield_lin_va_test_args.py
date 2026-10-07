"""Parity-test arguments of ``push_weights_with_efield_lin_va``: random fields and f0 values."""

import cunumpy as xp
import numpy as np

from struphy.pic.tests.kernel_test_args import (
    BOUNDARY_CONDITIONS,
    N_MARKERS,
    derham_arguments,
    marker_arguments,
    spline_coefficients,
)

CASES = BOUNDARY_CONDITIONS
RTOL = 1e-13
ATOL = 1e-14


def make_args(backend, seed):
    f0_values = xp.asarray(np.random.default_rng(13).random(N_MARKERS))
    return (
        0.2,
        0,
        *marker_arguments(CASES[seed]),
        derham_arguments(),
        *spline_coefficients(),
        f0_values,
        1.3,
        0.8,
    )
