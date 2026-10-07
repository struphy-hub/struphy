"""Parity-test arguments of ``charge_density_0form``: accumulation of the marker weights into a 0-form vector."""

import cunumpy as xp

from struphy.pic.tests.kernel_test_args import derham_arguments, marker_arguments

CASES = ((0, 0, 0), (2, 0, 1))
# atomics add in another order than the serial loop
RTOL = 1e-12
ATOL = 1e-13


def make_args(backend, seed):
    args_markers, args_domain = marker_arguments(CASES[seed])
    return (args_markers, derham_arguments(), args_domain, xp.zeros((18, 20, 16)))
