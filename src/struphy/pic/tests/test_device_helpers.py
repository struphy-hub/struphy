"""Elementwise parity checks for the shared CUDA device helpers."""

import cunumpy
import numpy as np
import pytest

from struphy.bsplines import bsplines_kernels as splines
from struphy.linear_algebra import linalg_kernels as linalg
from struphy.utils.kernel_backends import CudaKernel

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")


@requires_cupy
@pytest.mark.parametrize("degree", range(1, 9))
def test_spline_helpers(degree):
    import cupy as cp

    knots = np.r_[np.zeros(degree), np.linspace(0, 1, 11), np.ones(degree)]
    points = np.r_[-0.01, 0., np.linspace(0, 1, 129), 1., 1.01]
    source = r'''
#include "struphy/bsplines/bsplines_kernels.cuh"
extern "C" __global__ void evaluate(const double* t, int nt, int p, const double* x, double* out, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    int span=struphy_cuda::find_span(t,nt,p,x[i]);
    double* row=out+i*(3*p+3);
    row[0]=span;
    struphy_cuda::basis_funs(t,p,x[i],span,row+1);
    struphy_cuda::b_d_splines_slim(t,p,x[i],span,row+p+2,row+2*p+3);
}
'''
    out = cp.empty((len(points), 3 * degree + 3))
    CudaKernel(source, "evaluate")(cp.asarray(knots), len(knots), degree, cp.asarray(points), out, len(points), n_threads=len(points))
    expected = np.empty(out.shape)
    for i, x in enumerate(points):
        span = splines.find_span(knots, degree, x)
        expected[i, 0] = span
        splines.basis_funs(knots, degree, x, span, np.empty(degree), np.empty(degree), expected[i, 1:degree+2])
        splines.b_d_splines_slim(knots, degree, x, span, expected[i, degree+2:2*degree+3], expected[i, 2*degree+3:])
    np.testing.assert_allclose(out.get(), expected, rtol=1e-13, atol=1e-13)


@requires_cupy
def test_matrix_helpers():
    import cupy as cp

    rng = np.random.default_rng(4)
    matrices = rng.normal(size=(129, 3, 3)) + 4 * np.eye(3)
    vectors = rng.normal(size=(129, 3))
    source = r'''
#include "struphy/linear_algebra/linalg_kernels.cuh"
extern "C" __global__ void evaluate(const double* a, const double* v, double* inv, double* out, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    struphy_cuda::matrix_inv(a+9*i,inv+9*i);
    struphy_cuda::matrix_vector(inv+9*i,v+3*i,out+3*i);
}
'''
    inv, out = cp.empty(matrices.shape), cp.empty(vectors.shape)
    CudaKernel(source, "evaluate")(cp.asarray(matrices), cp.asarray(vectors), inv, out, len(vectors), n_threads=len(vectors))
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

    source = r'''
#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"
extern "C" __global__ void evaluate(MarkerArgs m, DomainArgs d, int newton) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<m.n_markers) struphy_cuda::apply_kinetic_bc_marker(i,m,d,newton);
}
'''
    results = []
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            m, d = make_arguments(129)
            m.bc_type[:] = cunumpy.asarray(bc)
            m.markers[:, :3] = cunumpy.asarray(np.random.default_rng(2).uniform(-.5, 1.5, (129, 3)))
            if backend == "numpy":
                for i in range(129):
                    apply_kinetic_bc_marker(i, m, d, newton)
            else:
                CudaKernel(source, "evaluate")(m, d, int(newton), n_threads=129)
            results.append(cunumpy.to_numpy(m.markers).copy())
    np.testing.assert_allclose(*results, rtol=1e-13, atol=1e-14)
