// Device versions of the helpers in particle_to_mat_kernels.py (one marker at a time).
#pragma once
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/pic/accumulation/filler_kernels.cuh"
namespace struphy_cuda {
// vec += fill * N(eta) for the 0-form basis
__device__ inline void vec_fill_b_v0(const DerhamArgs& args_derham, double eta1, double eta2, double eta3, Array3D<double> vec, double fill) {
    SplineScratch s;
    get_spans(eta1,eta2,eta3,args_derham,s);
    fill_vec(args_derham.pn[0],args_derham.pn[1],args_derham.pn[2],s.bn[0],s.bn[1],s.bn[2],
             s.spans[0],s.spans[1],s.spans[2],args_derham.starts,vec,fill);
}
}
