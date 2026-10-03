#include "struphy/pic/pushing/pusher_utilities_kernels.cuh"
extern "C" __global__
void push_eta_stage(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain,
                    const double* a, const double* b, const double* c, int n_stages) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=args_markers.n_markers) return;
    if(MARKER(args_markers,ip,args_markers.first_init_idx)==-1. || MARKER(args_markers,ip,args_markers.n_cols-1)==-2.) return;
    double jac[9], inv[9], v[3], k[3];
    struphy_cuda::df(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_domain,jac);
    struphy_cuda::matrix_inv(jac,inv);
    for(int j=0;j<3;++j) v[j]=MARKER(args_markers,ip,3+j);
    struphy_cuda::matrix_vector(inv,v,k);
    double last=stage==n_stages-1?1.:0.;
    for(int j=0;j<3;++j) {
        MARKER(args_markers,ip,args_markers.first_free_idx+j)+=dt*b[stage]*k[j];
        MARKER(args_markers,ip,j)=MARKER(args_markers,ip,args_markers.first_init_idx+j)+dt*a[stage]*k[j]+last*MARKER(args_markers,ip,args_markers.first_free_idx+j);
    }
    struphy_cuda::apply_kinetic_bc_marker(ip,args_markers,args_domain,false);
}
