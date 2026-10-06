"""Parity-test arguments of ``eval_spline_mpi_matrix``: spline values at a 3d array of points, one point flagged."""

import cunumpy as xp

from struphy.pic.tests.kernel_test_args import evaluation_grid, spline_evaluation_arguments

CASES = ((0, 0, 0), (1, 0, 1))


def N_THREADS(args):
    """One thread per entry of the output values (the last argument)."""
    return args[-1].size


def make_args(backend, seed):
    return (*evaluation_grid(sparse=False), *spline_evaluation_arguments(CASES[seed]), xp.zeros((7, 5, 4)))
