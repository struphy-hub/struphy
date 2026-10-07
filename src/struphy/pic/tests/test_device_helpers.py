"""Elementwise parity checks for the shared CUDA device helpers."""

import cunumpy
import numpy as np
import pytest
from cunumpy.kernel_testing import device_function_kernel, requires_cupy
from cunumpy.kernels import CudaKernel

from struphy.bsplines import bsplines_kernels as splines
from struphy.geometry.tests import spline_mapping_cases
from struphy.linear_algebra import linalg_kernels as linalg
from struphy.pic.tests.kernel_test_args import N_GEOMETRY_DOMAINS
from struphy.utils.cuda_arguments import CUDA_OPTIONS


@requires_cupy
@pytest.mark.parametrize("degree", range(1, 9))
def test_spline_helpers(degree):
    import cupy as cp

    knots = np.r_[np.zeros(degree), np.linspace(0, 1, 11), np.ones(degree)]
    points = np.r_[-0.01, 0.0, np.linspace(0, 1, 129), 1.0, 1.01]
    source = r"""
#include "struphy/bsplines/bsplines_kernels.cuh"
extern "C" __global__ void evaluate(const double* t, int nt, int p, const double* x, double* out, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    int span=struphy_cuda::bsplines_kernels::find_span(t,nt,p,x[i]);
    double* row=out+i*(3*p+3);
    row[0]=span;
    struphy_cuda::bsplines_kernels::basis_funs(t,p,x[i],span,row+1);
    struphy_cuda::bsplines_kernels::b_d_splines_slim(t,p,x[i],span,row+p+2,row+2*p+3);
}
"""
    out = cp.empty((len(points), 3 * degree + 3))
    CudaKernel(source, "evaluate", **CUDA_OPTIONS)(
        cp.asarray(knots), len(knots), degree, cp.asarray(points), out, len(points), n_threads=len(points)
    )
    expected = np.empty(out.shape)
    for i, x in enumerate(points):
        span = splines.find_span(knots, degree, x)
        expected[i, 0] = span
        splines.basis_funs(knots, degree, x, span, np.empty(degree), np.empty(degree), expected[i, 1 : degree + 2])
        splines.b_d_splines_slim(
            knots, degree, x, span, expected[i, degree + 2 : 2 * degree + 3], expected[i, 2 * degree + 3 :]
        )
    np.testing.assert_allclose(out.get(), expected, rtol=1e-13, atol=1e-13)


@requires_cupy
def test_matrix_helpers():
    import cupy as cp

    rng = np.random.default_rng(4)
    matrices = rng.normal(size=(129, 3, 3)) + 4 * np.eye(3)
    vectors = rng.normal(size=(129, 3))
    source = r"""
#include "struphy/linear_algebra/linalg_kernels.cuh"
extern "C" __global__ void evaluate(const double* a, const double* v, double* inv, double* out, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    struphy_cuda::linalg_kernels::matrix_inv(a+9*i,inv+9*i);
    struphy_cuda::linalg_kernels::matrix_vector(inv+9*i,v+3*i,out+3*i);
}
"""
    inv, out = cp.empty(matrices.shape), cp.empty(vectors.shape)
    CudaKernel(source, "evaluate", **CUDA_OPTIONS)(
        cp.asarray(matrices), cp.asarray(vectors), inv, out, len(vectors), n_threads=len(vectors)
    )
    expected_inv, expected_out = np.empty_like(matrices), np.empty_like(vectors)
    for i in range(len(vectors)):
        linalg.matrix_inv(matrices[i], expected_inv[i])
        linalg.matrix_vector(expected_inv[i], vectors[i], expected_out[i])
    np.testing.assert_allclose(inv.get(), expected_inv, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(out.get(), expected_out, rtol=1e-13, atol=1e-14)


@requires_cupy
@pytest.mark.parametrize("bc", [(0, 0, 0), (1, 1, 1), (2, 0, 1), (3, 1, 0)])
@pytest.mark.parametrize("newton", [False, True])
def test_boundary_helpers(bc, newton):
    from struphy.pic.pushing.pusher_utilities_kernels import apply_kinetic_bc_marker
    from struphy.pic.tests.test_kernel_backends import make_arguments

    source = r"""
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"
extern "C" __global__ void evaluate(MarkerArgs m, DomainArgs d, int newton) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<m.n_markers) struphy_cuda::pusher_utilities_kernels::apply_kinetic_bc_marker(i,m,d,newton);
}
"""
    results = []
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            m, d = make_arguments(129)
            m.bc_type[:] = cunumpy.asarray(bc)
            m.markers[:, :3] = cunumpy.asarray(np.random.default_rng(2).uniform(-0.5, 1.5, (129, 3)))
            if backend == "numpy":
                for i in range(129):
                    apply_kinetic_bc_marker(i, m, d, newton)
            else:
                CudaKernel(source, "evaluate", **CUDA_OPTIONS)(m, d, int(newton), n_threads=129)
            results.append(cunumpy.to_numpy(m.markers).copy())
    np.testing.assert_allclose(*results, rtol=1e-13, atol=1e-14)


@requires_cupy
@pytest.mark.parametrize("degree", range(1, 9))
def test_span_with_cunumpy_wrapper(degree):
    import cupy as cp

    knots = np.r_[np.zeros(degree), np.linspace(0, 1, 11), np.ones(degree)]
    points = np.r_[-0.01, np.linspace(0, 1, 129), 1.01]
    kernel = device_function_kernel(
        '#include "struphy/bsplines/bsplines_kernels.cuh"\n'
        "__device__ int span_at(const double* t, int nt, int p, double x) {"
        "return struphy_cuda::bsplines_kernels::find_span(t, nt, p, x);}",
        "int span_at(const double* t, int nt, int p, double x)",
        **CUDA_OPTIONS,
    )
    out = cp.empty(len(points), dtype=cp.int32)
    kernel(
        cp.asarray(knots),
        cp.full(len(points), len(knots), dtype=cp.int32),
        cp.full(len(points), degree, dtype=cp.int32),
        cp.asarray(points),
        out,
        len(points),
        n_threads=len(points),
    )
    np.testing.assert_array_equal(out.get(), [splines.find_span(knots, degree, x) for x in points])


@requires_cupy
def test_cuboid_helpers():
    """The CUDA mapping signatures and Jacobian dispatcher mirror Pyccel."""
    import cupy as cp

    from struphy.geometry.domains.cuboid.cuboid_kernels import cuboid, cuboid_df

    source = r"""
#include "struphy/geometry/evaluation_kernels.cuh"
extern "C" __global__ void evaluate(const double* eta, const double* params, double* out) {
    struphy_cuda::cuboid_kernels::cuboid(eta[0], eta[1], eta[2], params[0], params[1], params[2],
                        params[3], params[4], params[5], out);
    struphy_cuda::cuboid_kernels::cuboid_df(params[0], params[1], params[2],
                           params[3], params[4], params[5], out + 3);
    DomainArgs args = {};
    args.kind_map = 10;
    args.params = const_cast<double*>(params);
    struphy_cuda::evaluation_kernels::df(eta[0], eta[1], eta[2], args, out + 12);
}
"""
    eta = np.array([0.2, 0.4, 0.8])
    params = np.array([-2.0, 3.0, 1.0, 7.0, -5.0, -1.0])
    out = cp.empty(21)
    CudaKernel(source, "evaluate", **CUDA_OPTIONS)(cp.asarray(eta), cp.asarray(params), out, n_threads=1)
    f_out = np.empty(3)
    df_out = np.empty((3, 3))
    cuboid(*eta, *params, f_out)
    cuboid_df(*params, df_out)
    np.testing.assert_allclose(out.get(), np.r_[f_out, df_out.ravel(), df_out.ravel()], rtol=1e-13)


@requires_cupy
def test_get_spans_helper():
    """Named scratch fields retain the Pyccel spans and spline values per thread."""
    import cupy as cp

    from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments

    pn = np.array([1, 3, 8], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 11), np.ones(p)] for p in pn]
    points = np.random.default_rng(3).uniform(-0.01, 1.01, (129, 3))
    source = r"""
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
extern "C" __global__ void evaluate(const double* eta, DerhamArgs args_derham, double* out, int n) {
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= n) return;
    struphy_cuda::evaluation_kernels_3d::SplineScratch scratch;
    struphy_cuda::evaluation_kernels_3d::get_spans(eta[3*ip], eta[3*ip+1], eta[3*ip+2], args_derham, scratch);
    const int spans[3] = {scratch.span1, scratch.span2, scratch.span3};
    const double* bn[3] = {scratch.bn1, scratch.bn2, scratch.bn3};
    const double* bd[3] = {scratch.bd1, scratch.bd2, scratch.bd3};
    // Each axis occupies 18 columns: span, up to 9 B-values, up to 8 D-values.
    for (int axis = 0; axis < 3; ++axis) {
        double* row = out + ip*54 + axis*18;
        row[0] = spans[axis];
        for (int j = 0; j <= args_derham.pn[axis]; ++j) row[1+j] = bn[axis][j];
        for (int j = 0; j < args_derham.pn[axis]; ++j) row[10+j] = bd[axis][j];
    }
}
"""
    args_derham = CudaDerhamArguments(cp.asarray(pn), *(cp.asarray(tn) for tn in knots), cp.zeros(3, dtype=cp.int64))
    out = cp.zeros((len(points), 3, 18))
    CudaKernel(source, "evaluate", **CUDA_OPTIONS)(
        cp.asarray(points), args_derham, out, len(points), n_threads=len(points)
    )
    expected = np.zeros(out.shape)
    for ip, eta in enumerate(points):
        for axis, (tn, p) in enumerate(zip(knots, pn)):
            span = splines.find_span(tn, p, eta[axis])
            expected[ip, axis, 0] = span
            splines.b_d_splines_slim(
                tn, p, eta[axis], span, expected[ip, axis, 1 : p + 2], expected[ip, axis, 10 : 10 + p]
            )
    np.testing.assert_allclose(out.get(), expected, rtol=1e-13, atol=1e-13)


@requires_cupy
@pytest.mark.parametrize("degree", range(1, 9))
def test_der_spline_helpers(degree):
    """b_splines_slim and b_der_splines_slim on the device agree with pyccel."""
    import cupy as cp

    from struphy.geometry.tests.spline_mapping_cases import DER_SPLINES_SOURCE, der_splines_case

    knots, pts, expected = der_splines_case(degree)
    out = cp.zeros(expected.shape)
    CudaKernel(DER_SPLINES_SOURCE, "evaluate_der_splines", **CUDA_OPTIONS)(
        cp.asarray(knots), len(knots), degree, cp.asarray(pts), out, len(pts), n_threads=len(pts)
    )
    np.testing.assert_allclose(out.get(), expected, rtol=1e-13, atol=1e-13)


@requires_cupy
@pytest.mark.parametrize("name", list(spline_mapping_cases.DOMAINS))
def test_spline_mapping_helpers(name):
    """spline_3d(_df), spline_2d_straight(_df) and spline_2d_torus(_df) on the device agree with pyccel.

    The domain is created on the CuPy backend, so the device reads the CudaDomainArguments of the domain itself.
    """
    import cupy as cp

    from struphy.kernel_arguments.pusher_args_cuda import CudaDomainArguments

    etas = spline_mapping_cases.points()
    expected = spline_mapping_cases.expected(spline_mapping_cases.host_domain(name).args_domain, etas)
    with cunumpy.use_backend("cupy"):
        args_domain = spline_mapping_cases.DOMAINS[name]().args_domain
    assert type(args_domain) is CudaDomainArguments
    out = cp.zeros(expected.size)
    spline_mapping_cases.make_kernel()(
        *(cp.asarray(a) for a in spline_mapping_cases.flat_inputs(etas)),
        args_domain,
        out,
        out.size,
        n_threads=out.size,
    )
    np.testing.assert_allclose(out.get().reshape(expected.shape), expected, rtol=1e-12, atol=1e-12)


# CUDA-only wrappers that return one entry of a helper's output, so that device_function_kernel can call the
# helpers elementwise (its pointer arguments are shared by all threads)
GEOMETRY_WRAPPERS = r"""
#include "struphy/geometry/transform_kernels.cuh"
__device__ double metric_entry(double eta1, double eta2, double eta3, int kind_coeff, int entry,
                               bool avoid_round_off, const DomainArgs& args) {
    double tmp1[9], tmp2[9], tmp3[9], out[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (kind_coeff == 0) struphy_cuda::evaluation_kernels::f(eta1, eta2, eta3, args, out);
    else if (kind_coeff == 1) struphy_cuda::evaluation_kernels::df(eta1, eta2, eta3, args, out);
    else if (kind_coeff == 2) out[0] = struphy_cuda::evaluation_kernels::det_df(eta1, eta2, eta3, args, tmp1);
    else if (kind_coeff == 3) struphy_cuda::evaluation_kernels::df_inv(eta1, eta2, eta3, args, tmp1, avoid_round_off, out);
    else if (kind_coeff == 4) struphy_cuda::evaluation_kernels::g(eta1, eta2, eta3, args, tmp1, tmp2, avoid_round_off, out);
    else struphy_cuda::evaluation_kernels::g_inv(eta1, eta2, eta3, args, tmp1, tmp2, tmp3, avoid_round_off, out);
    return out[entry];
}
__device__ double transform_entry(double a0, double a1, double a2, double eta1, double eta2, double eta3,
                                  int kind_transform, int kind_fun, int entry, const DomainArgs& args_domain) {
    double a[3] = {a0, a1, a2}, out[3] = {0, 0, 0};
    if (kind_transform == 0) struphy_cuda::transform_kernels::pull(a, eta1, eta2, eta3, kind_fun, args_domain, out);
    else if (kind_transform == 1) struphy_cuda::transform_kernels::push(a, eta1, eta2, eta3, kind_fun, args_domain, out);
    else struphy_cuda::transform_kernels::tran(a, eta1, eta2, eta3, kind_fun, args_domain, out);
    return out[entry];
}
"""


def metric_helper_case(domain_index, avoid_round_off):
    """Wrapper kernel, per-thread host inputs and pyccel reference for f, df, det_df, df_inv, g and g_inv.

    Domain `domain_index` of :func:`~struphy.pic.tests.kernel_test_args.geometry_domain` (analytic and spline
    mappings). Used by :func:`test_metric_helpers` (GPU) and by its CPU emulation in ``test_cuda_emulation.py``.
    """
    from struphy.geometry import evaluation_kernels as geometry
    from struphy.pic.tests.kernel_test_args import geometry_domain, logical_markers

    kernel = device_function_kernel(
        GEOMETRY_WRAPPERS,
        "double metric_entry(double eta1, double eta2, double eta3, int kind_coeff, int entry, "
        "bool avoid_round_off, const DomainArgs& args)",
        **CUDA_OPTIONS,
    )
    points = logical_markers()[3:, :3]  # inside the logical cube
    rows = [(p, kind, entry) for p in points for kind in range(6) for entry in range(9)]
    eta = np.array([r[0] for r in rows])
    kinds = np.array([r[1] for r in rows], dtype=np.int32)
    entries = np.array([r[2] for r in rows], dtype=np.int32)

    with cunumpy.use_backend("numpy"):
        args = geometry_domain(domain_index).args_domain
    expected = np.empty(len(rows))
    tmp1, tmp2, tmp3 = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    for i, (p, kind, entry) in enumerate(rows):
        out = np.zeros((3, 3))
        if kind == 0:
            f_out = np.zeros(3)
            geometry.f(*p, args, f_out)
            out.flat[:3] = f_out
        elif kind == 1:
            geometry.df(*p, args, out)
        elif kind == 2:
            out[0, 0] = geometry.det_df(*p, args, tmp1)
        elif kind == 3:
            geometry.df_inv(*p, args, tmp1, avoid_round_off, out)
        elif kind == 4:
            geometry.g(*p, args, tmp1, tmp2, avoid_round_off, out)
        else:
            geometry.g_inv(*p, args, tmp1, tmp2, tmp3, avoid_round_off, out)
        expected[i] = out.flat[entry]

    inputs = (*(eta[:, k].copy() for k in range(3)), kinds, entries, np.full(len(rows), avoid_round_off))
    return kernel, inputs, args, expected


def transform_helper_case(domain_index):
    """Wrapper kernel, per-thread host inputs and pyccel reference for pull, push and tran (every kind_fun).

    Used by :func:`test_transform_helpers` (GPU) and by its CPU emulation in ``test_cuda_emulation.py``.
    """
    from struphy.geometry import transform_kernels as transforms
    from struphy.pic.tests.kernel_test_args import geometry_domain, logical_markers

    kernel = device_function_kernel(
        GEOMETRY_WRAPPERS,
        "double transform_entry(double a0, double a1, double a2, double eta1, double eta2, double eta3, "
        "int kind_transform, int kind_fun, int entry, const DomainArgs& args_domain)",
        **CUDA_OPTIONS,
    )
    kinds = [(0, k) for k in (0, 1, 10, 11, 12)] + [(1, k) for k in (0, 1, 10, 11, 12)]
    kinds += [(2, k) for k in (0, 1, *range(10, 22))]
    points = logical_markers()[3:20, :3]
    values = np.random.default_rng(12).normal(size=(len(points), 3))
    rows = [(i, t, k, e) for i in range(len(points)) for t, k in kinds for e in range(3 if k >= 10 else 1)]

    with cunumpy.use_backend("numpy"):
        args = geometry_domain(domain_index).args_domain
    expected = np.empty(len(rows))
    helper = {0: transforms.pull, 1: transforms.push, 2: transforms.tran}
    for n, (i, t, k, e) in enumerate(rows):
        a = values[i].copy() if k >= 10 else values[i, :1].copy()
        out = np.zeros(3 if k >= 10 else 1)
        helper[t](a, *points[i], k, args, out)
        expected[n] = out[e]

    columns = np.array(rows)
    inputs = (
        *(values[columns[:, 0], c] for c in range(3)),
        *(points[columns[:, 0], c] for c in range(3)),
        *(columns[:, c].astype(np.int32) for c in (1, 2, 3)),
    )
    return kernel, inputs, args, expected


def _run_on_device(kernel, inputs, domain_index, n):
    import cupy as cp

    from struphy.pic.tests.kernel_test_args import geometry_domain

    with cunumpy.use_backend("cupy"):
        device_args = geometry_domain(domain_index).args_domain
        result = cp.empty(n)
        kernel(*(cp.asarray(a) for a in inputs), device_args, result, n, n_threads=n)
    return result.get()


@requires_cupy
@pytest.mark.parametrize("domain_index", range(N_GEOMETRY_DOMAINS))
@pytest.mark.parametrize("avoid_round_off", [False, True])
def test_metric_helpers(domain_index, avoid_round_off):
    """f, df, det_df, df_inv, g and g_inv of every analytic mapping and four spline mappings agree with pyccel."""
    kernel, inputs, _, expected = metric_helper_case(domain_index, avoid_round_off)
    result = _run_on_device(kernel, inputs, domain_index, len(expected))
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


@requires_cupy
@pytest.mark.parametrize("domain_index", range(N_GEOMETRY_DOMAINS))
def test_transform_helpers(domain_index):
    """pull, push and tran of every analytic mapping and four spline mappings agree with pyccel, for every kind_fun."""
    kernel, inputs, _, expected = transform_helper_case(domain_index)
    result = _run_on_device(kernel, inputs, domain_index, len(expected))
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
