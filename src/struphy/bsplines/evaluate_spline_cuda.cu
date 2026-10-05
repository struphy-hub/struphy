#include "struphy/bsplines/evaluation_kernels_3d.cuh"

// Shared point evaluation. The public kernels below take the same arguments,
// in the same order, as evaluation_kernels_3d.py. Array views carry dimensions
// and strides without extra CUDA-only parameters or coordinate copies.
__device__ inline double evaluate_point(double e1, double e2, double e3,
                                        Array3D<double> data, const long long* kind,
                                        long long* pn, Array1D<double> tn1,
                                        Array1D<double> tn2, Array1D<double> tn3,
                                        long long* starts) {
    struphy_cuda::SplineScratch scratch;
    scratch.span1 = struphy_cuda::find_span(tn1.data, tn1.shape[0], pn[0], e1, tn1.strides[0]);
    scratch.span2 = struphy_cuda::find_span(tn2.data, tn2.shape[0], pn[1], e2, tn2.strides[0]);
    scratch.span3 = struphy_cuda::find_span(tn3.data, tn3.shape[0], pn[2], e3, tn3.strides[0]);
    struphy_cuda::b_d_splines_slim(tn1.data, pn[0], e1, scratch.span1, scratch.bn1, scratch.bd1, tn1.strides[0]);
    struphy_cuda::b_d_splines_slim(tn2.data, pn[1], e2, scratch.span2, scratch.bn2, scratch.bd2, tn2.strides[0]);
    struphy_cuda::b_d_splines_slim(tn3.data, pn[2], e3, scratch.span3, scratch.bn3, scratch.bd3, tn3.strides[0]);
    const double* b1 = kind[0] == 0 ? scratch.bn1 : scratch.bd1;
    const double* b2 = kind[1] == 0 ? scratch.bn2 : scratch.bd2;
    const double* b3 = kind[2] == 0 ? scratch.bn3 : scratch.bd3;
    return struphy_cuda::eval_spline_mpi_kernel(
        pn[0] - kind[0], pn[1] - kind[1], pn[2] - kind[2], b1, b2, b3,
        scratch.span1, scratch.span2, scratch.span3, data, starts);
}

extern "C" __global__ void eval_spline_mpi_markers(
    Array2D<double> markers, Array3D<double> _data, const long long* kind,
    long long* pn, Array1D<double> tn1, Array1D<double> tn2, Array1D<double> tn3,
    long long* starts, Array1D<double> values) {
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= markers.shape[0] || markers(ip, 0) == -1.) return;
    values(ip) = evaluate_point(markers(ip, 0), markers(ip, 1), markers(ip, 2),
                                _data, kind, pn, tn1, tn2, tn3, starts);
}

extern "C" __global__ void eval_spline_mpi_matrix(
    Array3D<double> eta1, Array3D<double> eta2, Array3D<double> eta3,
    Array3D<double> _data, const long long* kind, long long* pn,
    Array1D<double> tn1, Array1D<double> tn2, Array1D<double> tn3,
    long long* starts, Array3D<double> values) {
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= values.shape[0] * values.shape[1] * values.shape[2]) return;
    long long k = ip % values.shape[2];
    long long j = (ip / values.shape[2]) % values.shape[1];
    long long i = ip / (values.shape[1] * values.shape[2]);
    double e1 = eta1(i, j, k), e2 = eta2(i, j, k), e3 = eta3(i, j, k);
    if (e1 == -1. || e2 == -1. || e3 == -1.) return;
    values(i, j, k) = evaluate_point(e1, e2, e3, _data, kind, pn, tn1, tn2, tn3, starts);
}

extern "C" __global__ void eval_spline_mpi_sparse_meshgrid(
    Array3D<double> eta1, Array3D<double> eta2, Array3D<double> eta3,
    Array3D<double> _data, const long long* kind, long long* pn,
    Array1D<double> tn1, Array1D<double> tn2, Array1D<double> tn3,
    long long* starts, Array3D<double> values) {
    long long ip = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= values.shape[0] * values.shape[1] * values.shape[2]) return;
    long long k = ip % values.shape[2];
    long long j = (ip / values.shape[2]) % values.shape[1];
    long long i = ip / (values.shape[1] * values.shape[2]);
    double e1 = eta1(i, 0, 0), e2 = eta2(0, j, 0), e3 = eta3(0, 0, k);
    if (e1 == -1. || e2 == -1. || e3 == -1.) return;
    values(i, j, k) = evaluate_point(e1, e2, e3, _data, kind, pn, tn1, tn2, tn3, starts);
}
