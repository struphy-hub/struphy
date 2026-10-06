"""Parity-test arguments of ``reflect``: reverse one logical velocity component of 10 markers in a Cuboid."""

import cunumpy as xp
import numpy as np

from struphy.geometry.domains import Cuboid

CASES = (0, 1, 2)  # the axis


def N_THREADS(args):
    """One thread per entry of outside_inds."""
    return args[2].size


def make_args(backend, seed):
    markers = np.random.default_rng(32).random((140, 8))
    outside_inds = np.arange(139, 129, -1, dtype=np.int64)
    return (xp.asarray(markers), Cuboid(r1=2.0, r2=3.0, r3=4.0).args_domain, xp.asarray(outside_inds), CASES[seed])
