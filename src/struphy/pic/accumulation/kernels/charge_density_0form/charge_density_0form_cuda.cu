#include "struphy/pic/accumulation/particle_to_mat_kernels.cuh"
extern "C" __global__ void charge_density_0form(MarkerArgs args_markers, DerhamArgs args_derham, DomainArgs args_domain, Array3D<double> vec) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=args_markers.n_markers) return;
    // only do something if particle is a "true" particle (i.e. not a hole)
    if(MARKER(args_markers,ip,0)==-1.) return;
    struphy_cuda::vec_fill_b_v0(args_derham,MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),
                                vec,MARKER(args_markers,ip,args_markers.weight_idx));
}
