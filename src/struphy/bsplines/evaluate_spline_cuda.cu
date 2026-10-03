#include "struphy/bsplines/evaluation_kernels_3d.cuh"
extern "C" __global__ void evaluate_spline(DerhamArgs a, Array3D<double> coeff,
    const double* x, const double* y, const double* z, int k0, int k1, int k2, double* out, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    if(x[i]==-1. || y[i]==-1. || z[i]==-1.) {out[i]=0.; return;}
    struphy_cuda::SplineScratch s;
    struphy_cuda::get_spans(x[i],y[i],z[i],a,s);
    out[i]=struphy_cuda::eval_spline(a,s,coeff,k0,k1,k2);
}
