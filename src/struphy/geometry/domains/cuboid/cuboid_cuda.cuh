#pragma once
namespace struphy_cuda {
__device__ inline void cuboid(double x, double y, double z, const double* p, double* out) {
    out[0]=p[0]+(p[1]-p[0])*x; out[1]=p[2]+(p[3]-p[2])*y; out[2]=p[4]+(p[5]-p[4])*z;
}
__device__ inline void cuboid_df(const double* p, double* out) {
    for(int i=0;i<9;++i) out[i]=0.;
    for(int i=0;i<3;++i) out[4*i]=p[2*i+1]-p[2*i];
}
}
