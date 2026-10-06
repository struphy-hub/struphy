"""Argument builders for the kernel tests (pyccel/CUDA parity, CPU emulation, device helpers).

The builders create random host data and convert it with ``xp.asarray``, so the arguments land on the active
backend and are the same on both. Argument objects are the pyccel classes on the NumPy backend and their CUDA
versions on the CuPy backend, as created by the owners. The parity cases of every CUDA kernel, built from these, are
in :mod:`struphy.pic.tests.cuda_parity_cases`.
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


def spline_evaluation_arguments(kind):
    """``_data, kind, pn, tn1, tn2, tn3, starts`` of the spline evaluation kernels, for degrees 2, 3, 1 on 8 cells.

    The coefficients cover every span of the knots with start indices 1 on all axes.
    """
    rng = np.random.default_rng(34)
    degree = np.array([2, 3, 1], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 9), np.ones(p)] for p in degree]
    return (
        xp.asarray(rng.normal(size=(16, 18, 20))),
        xp.asarray(kind, dtype=np.int64),
        xp.asarray(degree),
        *(xp.asarray(t) for t in knots),
        xp.ones(3, dtype=np.int64),
    )


def evaluation_grid(sparse):
    """Evaluation points on a 7 x 5 x 4 grid (sparse: shapes (7, 1, 1), (1, 5, 1), (1, 1, 4)); one is flagged -1."""
    axes = [np.linspace(0.3, 0.7, n) for n in (7, 5, 4)]
    coords = [axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]]
    # copies: pyccel rejects the zero strides of views and broadcasts
    coords = [(c if sparse else np.broadcast_to(c, (7, 5, 4))).copy() for c in coords]
    coords[1][0, 0, 0] = -1.0
    return tuple(xp.asarray(c) for c in coords)


def analytic_domains():
    """One domain per analytic mapping (``kind_map`` 10-12, 20-22, 30-32), HollowTorus in both angle parametrizations.

    The parameters differ from the defaults and every Jacobian except Cuboid's and Orthogonal's is non-diagonal, so a
    transposed or mixed-up ``DF`` changes the results. Created on the active backend (``args_domain`` is the pyccel
    class on NumPy and the CUDA class on CuPy).
    """
    from struphy.geometry.domains import (
        Colella,
        HollowCylinder,
        HollowTorus,
        Orthogonal,
        PoweredEllipticCylinder,
        ShafranovDshapedCylinder,
        ShafranovShiftCylinder,
        ShafranovSqrtCylinder,
    )

    return (
        Cuboid(l1=-1.0, r1=2.0, l2=0.5, r2=3.0, l3=-2.0, r3=4.0),
        Orthogonal(Lx=2.0, Ly=3.0, alpha=0.05, Lz=4.0),
        Colella(Lx=2.0, Ly=3.0, alpha=0.05, Lz=4.0),
        HollowCylinder(a1=0.2, a2=1.0, Lz=4.0, poc=2),
        PoweredEllipticCylinder(rx=1.0, ry=2.0, Lz=3.0, s=0.5),
        HollowTorus(a1=0.1, a2=1.0, R0=3.0, sfl=False, pol_period=2, tor_period=3),
        HollowTorus(a1=0.1, a2=1.0, R0=3.0, sfl=True, pol_period=1, tor_period=3),
        ShafranovShiftCylinder(rx=1.0, ry=1.5, Lz=4.0, delta=0.1),
        ShafranovSqrtCylinder(rx=1.5, ry=1.0, Lz=4.0, delta=0.1),
        ShafranovDshapedCylinder(
            R0=2.0, Lz=3.0, delta_x=0.1, delta_y=0.05, delta_gs=0.33, epsilon_gs=0.32, kappa_gs=1.7
        ),
    )


N_ANALYTIC_DOMAINS = 10


def logical_markers(n=N_MARKERS, seed=5):
    """Logical points in marker format, shape (n, 7), away from the poles (eta1 >= 0.1).

    Row 0 is a hole (-1), row 1 is outside in eta1 and row 2 in eta3, so ``remove_outside`` changes the result.
    """
    rng = np.random.default_rng(seed)
    markers = rng.random((n, 7))
    markers[:, 0] = rng.uniform(0.1, 0.95, n)
    markers[0, :] = -1.0
    markers[1, 0] = 1.2
    markers[2, 2] = -0.3
    return markers
