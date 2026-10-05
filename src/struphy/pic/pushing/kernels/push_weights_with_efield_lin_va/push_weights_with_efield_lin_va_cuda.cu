#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
extern "C" __global__ void push_weights_with_efield_lin_va(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham,
    Array3D<double> e1_1, Array3D<double> e1_2, Array3D<double> e1_3, const double* f0_values, double kappa, double vth) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=args_markers.n_markers) return;
    if(MARKER(args_markers,ip,0)==-1. || MARKER(args_markers,ip,args_markers.n_cols-1)==-2.) return;
    double jac[9], inv[9], v[3], inv_v[3], e_vec[3];
    for(int j=0;j<3;++j) v[j]=MARKER(args_markers,ip,3+j);
    struphy_cuda::SplineScratch s;
    struphy_cuda::get_spans(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_derham,s);
    struphy_cuda::df(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_domain,jac);
    struphy_cuda::matrix_inv(jac,inv);
    struphy_cuda::matrix_vector(inv,v,inv_v);
    struphy_cuda::eval_form(args_derham,s,e1_1,e1_2,e1_3,1,e_vec);
    // w_{n+1} = w_n + kappa * dt / (2 * N * s_0 * v_th^2) * f_0 * (DF^{-1} v_p) . (e_{n+1} + e_n)
    double update=(inv_v[0]*e_vec[0]+inv_v[1]*e_vec[1]+inv_v[2]*e_vec[2])*f0_values[ip]*kappa*dt
        /(2*args_markers.Np*MARKER(args_markers,ip,7)*(vth*vth));
    MARKER(args_markers,ip,6)+=update;
}
