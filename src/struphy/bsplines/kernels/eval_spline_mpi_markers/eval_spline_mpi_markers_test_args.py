"""Parity-test arguments of ``eval_spline_mpi_markers``: spline values at 129 markers, one of them flagged."""

import cunumpy as xp
import numpy as np

from struphy.pic.tests.kernel_test_args import spline_evaluation_arguments

CASES = ((0, 0, 0), (1, 0, 1))


def make_args(backend, seed):
    markers = np.random.default_rng(5).random((129, 3))
    markers[0, 0] = -1.0  # not on the process domain: skipped
    return (xp.asarray(markers), *spline_evaluation_arguments(CASES[seed]), xp.zeros(129))
