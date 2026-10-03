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

#include "cunumpy/array_view.cuh"
namespace struphy_cuda {
__device__ inline double eval_spline(const DerhamArgs& a, const SplineScratch& s, Array3D<double> c, int k0, int k1, int k2) {
    int kind[3]={k0,k1,k2};
    const double* basis[3];
    for(int j=0;j<3;++j) basis[j]=kind[j]?s.bd[j]:s.bn[j];
    double out=0.;
    for(int i=0;i<=a.pn[0]-k0;++i)
        for(int j=0;j<=a.pn[1]-k1;++j)
            for(int k=0;k<=a.pn[2]-k2;++k)
                out+=c(s.spans[0]+i-a.starts[0],s.spans[1]+j-a.starts[1],s.spans[2]+k-a.starts[2])*basis[0][i]*basis[1][j]*basis[2][k];
    return out;
}
__device__ inline void eval_form(const DerhamArgs& a, const SplineScratch& s, Array3D<double> c0, Array3D<double> c1, Array3D<double> c2, int form, double* out) {
    Array3D<double> c[3]={c0,c1,c2};
    for(int j=0;j<3;++j) {
        int k[3];
        for(int axis=0;axis<3;++axis) k[axis]=form==1?(axis==j):(axis!=j);
        out[j]=eval_spline(a,s,c[j],k[0],k[1],k[2]);
    }
}
}
