#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"
extern "C" __global__
void push_eta_stage(double dt, int stage, MarkerArgs m, DomainArgs d,
                    const double* a, const double* b, const double* c, int n_stages) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=m.n_markers) return;
    if(MARKER(m,ip,m.first_init_idx)==-1. || MARKER(m,ip,m.n_cols-1)==-2.) return;
    double jac[9], inv[9], v[3], k[3];
    struphy_cuda::df(MARKER(m,ip,0),MARKER(m,ip,1),MARKER(m,ip,2),d,jac);
    struphy_cuda::matrix_inv(jac,inv);
    for(int j=0;j<3;++j) v[j]=MARKER(m,ip,3+j);
    struphy_cuda::matrix_vector(inv,v,k);
    double last=stage==n_stages-1?1.:0.;
    for(int j=0;j<3;++j) {
        MARKER(m,ip,m.first_free_idx+j)+=dt*b[stage]*k[j];
        MARKER(m,ip,j)=MARKER(m,ip,m.first_init_idx+j)+dt*a[stage]*k[j]+last*MARKER(m,ip,m.first_free_idx+j);
    }
    struphy_cuda::apply_kinetic_bc_marker(ip,m,d,false);
}
