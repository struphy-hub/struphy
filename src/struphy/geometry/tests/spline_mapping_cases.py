"""Shared cases for the device spline mappings (``geometry/spline_mappings_kernels.cuh``) against pyccel.

Used by the GPU device-helper test (``pic/tests/test_device_helpers.py``) and by its CPU emulation
(``pic/tests/test_cuda_emulation.py``). The device function :data:`PROTOTYPE` evaluates the mapping and its
Jacobian of a spline-mapped domain at one point, as ``evaluation_kernels.f``/``df`` do for ``kind_map`` 0-2,
and returns one of the twelve entries (``F`` then ``DF`` row-major); ``cunumpy.kernel_testing.device_function_kernel``
wraps it in a kernel with one thread per (point, entry).
"""

import cunumpy as xp
import numpy as np

SOURCE = r"""
#include "struphy/geometry/spline_mappings_kernels.cuh"
__device__ double spline_mapping_entry(double eta1, double eta2, double eta3, int entry, const DomainArgs& args_domain) {
    double f_out[3], df_out[9];
    switch (args_domain.kind_map) {
        case 0:
            struphy_cuda::spline_mappings_kernels::spline_3d(eta1, eta2, eta3, args_domain.degree, args_domain.ind1, args_domain.ind2,
                                    args_domain.ind3, args_domain, f_out);
            struphy_cuda::spline_mappings_kernels::spline_3d_df(eta1, eta2, eta3, args_domain.degree, args_domain.ind1, args_domain.ind2,
                                       args_domain.ind3, args_domain, df_out);
            break;
        case 1:
            struphy_cuda::spline_mappings_kernels::spline_2d_straight(eta1, eta2, eta3, args_domain.degree, args_domain.ind1,
                                             args_domain.ind2, args_domain, args_domain.params[0], f_out);
            struphy_cuda::spline_mappings_kernels::spline_2d_straight_df(eta1, eta2, args_domain.degree, args_domain.ind1, args_domain.ind2,
                                                args_domain, args_domain.params[0], df_out);
            break;
        case 2:
            struphy_cuda::spline_mappings_kernels::spline_2d_torus(eta1, eta2, eta3, args_domain.degree, args_domain.ind1, args_domain.ind2,
                                          args_domain, args_domain.params[0], f_out);
            struphy_cuda::spline_mappings_kernels::spline_2d_torus_df(eta1, eta2, eta3, args_domain.degree, args_domain.ind1,
                                             args_domain.ind2, args_domain, args_domain.params[0], df_out);
            break;
        default:
            asm("trap;");
    }
    return entry < 3 ? f_out[entry] : df_out[entry - 3];
}
"""
PROTOTYPE = (
    "double spline_mapping_entry(double eta1, double eta2, double eta3, int entry, const DomainArgs& args_domain)"
)

# name -> constructor of a spline-mapped domain (on the active backend)
DOMAINS = {
    "Spline": lambda: _spline_3d(),
    "IGAPolarCylinder": lambda: _domains().IGAPolarCylinder(num_elements=(6, 8), degree=(2, 3), a=1.3, Lz=2.0),
    "IGAPolarCylinder_p1": lambda: _domains().IGAPolarCylinder(num_elements=(4, 5), degree=(1, 1)),
    "IGAPolarTorus": lambda: _domains().IGAPolarTorus(num_elements=(5, 7), degree=(3, 2), sfl=True, tor_period=3),
    "PoloidalSplineStraight": lambda: _poloidal_without_pole(),
}


def _domains():
    from struphy import domains

    return domains


def _spline_3d():
    from struphy.pic.tests.kernel_test_args import twisted_torus_spline

    return twisted_torus_spline()


def _poloidal_without_pole():
    """A straight 2d spline mapping (kind_map 1) of an annulus: no pole, so the eta1 == 0 branches do not apply."""
    from struphy.geometry.base import PoloidalSplineStraight, interp_mapping

    num_elements, degree, spl_kind = (5, 6), (2, 2), (False, True)

    def X(e1, e2):
        return (0.5 + e1) * np.cos(2 * np.pi * e2)

    def Y(e1, e2):
        return (0.5 + e1) * np.sin(2 * np.pi * e2)

    cx, cy = interp_mapping(num_elements, degree, spl_kind, X, Y)
    return PoloidalSplineStraight(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, Lz=3.0)


def points(seed=0, n=61):
    """Evaluation points (n + 8, 3): random interior points plus the pole eta1 = 0, the boundaries and knots."""
    rng = np.random.default_rng(seed)
    special = np.array(
        [
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 0.0],
            [0.0, 0.81, 0.25],
            [1.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.125, 0.2],
            [1e-12, 0.4, 0.9],
        ]
    )
    return np.vstack([special, rng.uniform(0.0, 1.0, (n, 3))])


def expected(args_domain, etas):
    """F and DF at each point from the pyccel spline mappings, as (len(etas), 12): F, then DF row-major."""
    from struphy.geometry import spline_mappings_kernels as pyccel

    a = args_domain
    out = np.empty((len(etas), 12))
    f_out, df_out = np.empty(3), np.empty((3, 3))
    for i, (eta1, eta2, eta3) in enumerate(etas):
        if a.kind_map == 0:
            pyccel.spline_3d(eta1, eta2, eta3, a.degree, a.ind1, a.ind2, a.ind3, a, f_out)
            pyccel.spline_3d_df(eta1, eta2, eta3, a.degree, a.ind1, a.ind2, a.ind3, a, df_out)
        elif a.kind_map == 1:
            pyccel.spline_2d_straight(eta1, eta2, eta3, a.degree, a.ind1, a.ind2, a, a.params[0], f_out)
            pyccel.spline_2d_straight_df(eta1, eta2, a.degree, a.ind1, a.ind2, a, a.params[0], df_out)
        else:
            pyccel.spline_2d_torus(eta1, eta2, eta3, a.degree, a.ind1, a.ind2, a, a.params[0], f_out)
            pyccel.spline_2d_torus_df(eta1, eta2, eta3, a.degree, a.ind1, a.ind2, a, a.params[0], df_out)
        out[i, :3], out[i, 3:] = f_out, df_out.ravel()
    return out


def flat_inputs(etas):
    """Per-thread inputs of the wrapper kernel: one thread per (point, entry)."""
    n = len(etas)
    eta1, eta2, eta3 = (np.repeat(etas[:, k], 12) for k in range(3))
    entry = np.tile(np.arange(12, dtype=np.int32), n)
    return eta1, eta2, eta3, entry


def make_kernel():
    """The wrapper kernel (a :class:`cunumpy.kernels.CudaKernel`) around :data:`PROTOTYPE`."""
    from cunumpy.kernel_testing import device_function_kernel

    from struphy.utils.cuda_arguments import CUDA_OPTIONS

    return device_function_kernel(SOURCE, PROTOTYPE, **CUDA_OPTIONS)


def host_domain(name):
    """The domain `name` on the NumPy backend."""
    with xp.use_backend("numpy"):
        return DOMAINS[name]()


DER_SPLINES_SOURCE = r"""
#include "struphy/bsplines/bsplines_kernels.cuh"
extern "C" __global__ void evaluate_der_splines(const double* t, int nt, int p, const double* x, double* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int span = struphy_cuda::bsplines_kernels::find_span(t, nt, p, x[i]);
    double* row = out + i * (3 * p + 3);
    struphy_cuda::bsplines_kernels::b_splines_slim(t, p, x[i], span, row);
    struphy_cuda::bsplines_kernels::b_der_splines_slim(t, p, x[i], span, row + p + 1, row + 2 * p + 2);
}
"""


def der_splines_case(degree):
    """Knots, points and the pyccel reference (b_splines_slim, then b_der_splines_slim) for DER_SPLINES_SOURCE."""
    from struphy.bsplines import bsplines_kernels as splines

    knots = np.r_[np.zeros(degree), np.linspace(0, 1, 11) ** 1.5, np.ones(degree)]
    pts = np.r_[0.0, np.linspace(0, 1, 129), 1.0, 0.5**1.5]
    expected = np.empty((len(pts), 3 * degree + 3))
    for i, x in enumerate(pts):
        span = splines.find_span(knots, degree, x)
        splines.b_splines_slim(knots, degree, x, span, expected[i, : degree + 1])
        splines.b_der_splines_slim(
            knots, degree, x, span, expected[i, degree + 1 : 2 * degree + 2], expected[i, 2 * degree + 2 :]
        )
    return knots, pts, expected
