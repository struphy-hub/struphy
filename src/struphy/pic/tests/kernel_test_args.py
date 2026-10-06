"""Argument builders shared by the ``<name>_test_args.py`` modules of the kernel folders.

Each ``<name>_test_args.py`` defines ``make_args(backend, seed)`` for cunumpy's parity tests
(:func:`cunumpy.kernel_testing.check_parity`) and ``CASES``: the test cases, selected by ``seed``. The builders
create random host data and convert it with ``xp.asarray``, so the arguments land on the active backend and are
the same on both. Argument objects are the pyccel classes on the NumPy backend and their CUDA versions on
the CuPy backend, as created by the owners.
"""

import cunumpy as xp
import numpy as np

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaMarkerArguments
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, MarkerArguments
from struphy.ode.utils import ButcherTableau

N_MARKERS = 129  # not a multiple of the block size
N_COLS = 25
MARKER_INDICES = (3, 6, 7, 8, 14, 17, 18, 4)  # vdim, weight_idx, ..., mu_idx
BOUNDARY_CONDITIONS = ((0, 0, 0), (1, 1, 1), (2, 0, 1))  # periodic, reflect, mixed with remove


def marker_arguments(bc):
    """Markers with a hole (row 0) and a boundary particle (row 1), in a Cuboid."""
    rng = np.random.default_rng(7)
    markers = rng.random((N_MARKERS, N_COLS))
    markers[:, 3:6] = rng.uniform(-2, 2, (N_MARKERS, 3))
    markers[:, 8:14] = markers[:, :6]
    markers[:, 18:21] = 0.0
    markers[0, 8] = -1.0
    markers[1, -1] = -2.0
    valid = np.ones(N_MARKERS, dtype=bool)
    valid[:2] = False
    args_class = CudaMarkerArguments if xp.get_backend() == "cupy" else MarkerArguments
    args_markers = args_class(
        xp.asarray(markers),
        xp.asarray(valid),
        N_MARKERS,
        *MARKER_INDICES,
        xp.asarray(bc, dtype=np.int64),
    )
    return args_markers, Cuboid().args_domain


def butcher_arguments(method):
    butcher = ButcherTableau(method)
    return xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages


def derham_arguments():
    """Splines of degrees 2, 3, 1 on 8 cells, starting at index 0."""
    degree = np.array([2, 3, 1], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 9), np.ones(p)] for p in degree]
    args_class = CudaDerhamArguments if xp.get_backend() == "cupy" else DerhamArguments
    return args_class(xp.asarray(degree), *(xp.asarray(t) for t in knots), xp.zeros(3, dtype=np.int64))


def spline_coefficients(n=3, seed=11):
    """Random coefficients, large enough for every span of :func:`derham_arguments`."""
    rng = np.random.default_rng(seed)
    return tuple(xp.asarray(rng.normal(size=(18, 20, 16))) for _ in range(n))
