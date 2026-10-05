#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
extern "C" __global__ void push_v_with_efield(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham,
    Array3D<double> e1_1, Array3D<double> e1_2, Array3D<double> e1_3, double const_factor) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=args_markers.n_markers) return;
    if(!args_markers.valid_mks[ip]) return;
    double jac[9], inv[9], e_form[3];
    struphy_cuda::SplineScratch s;
    struphy_cuda::df(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_domain,jac);
    struphy_cuda::matrix_inv(jac,inv);
    struphy_cuda::get_spans(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_derham,s);
    struphy_cuda::eval_form(args_derham,s,e1_1,e1_2,e1_3,1,e_form);
    // Cartesian field DF^{-T} e
    for(int j=0;j<3;++j) {
        double e_cart=inv[j]*e_form[0]+inv[3+j]*e_form[1]+inv[6+j]*e_form[2];
        MARKER(args_markers,ip,3+j)+=dt*const_factor*e_cart;
    }
}
