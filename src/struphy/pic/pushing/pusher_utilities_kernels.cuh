#pragma once
#include <math.h>
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
namespace struphy_cuda {
__device__ inline void reflect_velocity(int ip, const MarkerArgs& m, const DomainArgs& d, int axis) {
    double jac[9], inv[9], v[3], logical[3], out[3];
    df(MARKER(m,ip,0),MARKER(m,ip,1),MARKER(m,ip,2),d,jac);
    matrix_inv(jac,inv);
    for(int j=0;j<3;++j) v[j]=MARKER(m,ip,3+j);
    matrix_vector(inv,v,logical); logical[axis]*=-1.; matrix_vector(jac,logical,out);
    for(int j=0;j<3;++j) MARKER(m,ip,3+j)=out[j];
}
__device__ inline void apply_kinetic_bc_marker(int ip, const MarkerArgs& m, const DomainArgs& d, bool newton) {
    for(int axis=0;axis<3;++axis) if(m.bc_type[axis]==2 && (MARKER(m,ip,axis)>1. || MARKER(m,ip,axis)<0.)) {
        for(int j=0;j<m.n_cols-1;++j) MARKER(m,ip,j)=-1.;
        return;
    }
    bool reflected[3]={false,false,false};
    for(int axis=0;axis<3;++axis) {
        double& x=MARKER(m,ip,axis);
        if(m.bc_type[axis]==0) {
            double shift=x>1.?1.:(x<0.?-1.:0.);
            if(shift!=0.) x-=floor(x);
            if(newton) MARKER(m,ip,m.first_shift_idx+axis)+=shift;
            else MARKER(m,ip,m.first_shift_idx+axis)=shift;
        }
        if(m.bc_type[axis]==1) {
            if(x>1.) {x=2.-x; reflected[axis]=true;}
            else if(x<0.) {x=-x; reflected[axis]=true;}
        }
    }
    for(int axis=0;axis<3;++axis) if(reflected[axis]) {
        MARKER(m,ip,m.first_init_idx)=-1.; reflect_velocity(ip,m,d,axis);
    }
}
}
