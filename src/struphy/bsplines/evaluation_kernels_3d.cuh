#pragma once
#include "struphy/kernel_arguments/pusher_args.cuh"
#include "struphy/bsplines/bsplines_kernels.cuh"
namespace struphy_cuda {
struct SplineScratch {
    int spans[3];
    double bn[3][MAX_SPLINE_DEGREE+1], bd[3][MAX_SPLINE_DEGREE];
};
__device__ inline void get_spans(double x, double y, double z, const DerhamArgs& a, SplineScratch& s) {
    const double* knots[3]={a.tn1,a.tn2,a.tn3};
    int lengths[3]={a.nt1,a.nt2,a.nt3};
    double eta[3]={x,y,z};
    for(int i=0;i<3;++i) {
        s.spans[i]=find_span(knots[i],lengths[i],a.pn[i],eta[i]);
        b_d_splines_slim(knots[i],a.pn[i],eta[i],s.spans[i],s.bn[i],s.bd[i]);
    }
}
}
