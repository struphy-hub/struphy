#include <math.h>
#include "struphy/bsplines/evaluation_kernels_3d.cuh"
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
extern "C" __global__ void push_vxb_implicit(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain, DerhamArgs args_derham,
    Array3D<double> b2_1, Array3D<double> b2_2, Array3D<double> b2_3) {
    int ip=blockDim.x*blockIdx.x+threadIdx.x;
    if(ip>=args_markers.n_markers) return;
    if(MARKER(args_markers,ip,args_markers.first_init_idx)==-1.) return;
    double jac[9], field[3], v[3];
    struphy_cuda::SplineScratch s;
    struphy_cuda::get_spans(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_derham,s);
    struphy_cuda::df(MARKER(args_markers,ip,0),MARKER(args_markers,ip,1),MARKER(args_markers,ip,2),args_domain,jac);
    for(int j=0;j<3;++j) v[j]=MARKER(args_markers,ip,3+j);
    struphy_cuda::eval_form(args_derham,s,b2_1,b2_2,b2_3,2,field);
    double b[3]; struphy_cuda::matrix_vector(jac,field,b);
    double det=jac[0]*(jac[4]*jac[8]-jac[5]*jac[7])-jac[1]*(jac[3]*jac[8]-jac[5]*jac[6])+jac[2]*(jac[3]*jac[7]-jac[4]*jac[6]);
    for(int j=0;j<3;++j) b[j]/=det;
    double cross[9]={0.,b[2],-b[1],-b[2],0.,b[0],b[1],-b[0],0.};
    double lhs[9],rhs[9],inv[9],vec[3],out[3];
    for(int j=0;j<9;++j) {double id=(j%4==0)?1.:0.; lhs[j]=id-.5*dt*cross[j]; rhs[j]=id+.5*dt*cross[j];}
    struphy_cuda::matrix_inv(lhs,inv); struphy_cuda::matrix_vector(rhs,v,vec); struphy_cuda::matrix_vector(inv,vec,out);
    for(int j=0;j<3;++j) MARKER(args_markers,ip,3+j)=out[j];
}
