#pragma once
namespace struphy_cuda {
__device__ inline void matrix_vector(const double* a, const double* b, double* c) {
    for (int i=0; i<3; ++i) c[i]=a[3*i]*b[0]+a[3*i+1]*b[1]+a[3*i+2]*b[2];
}
__device__ inline void matrix_inv(const double* a, double* b) {
    double det=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
    b[0]=(a[4]*a[8]-a[5]*a[7])/det; b[1]=(a[2]*a[7]-a[1]*a[8])/det;
    b[2]=(a[1]*a[5]-a[2]*a[4])/det; b[3]=(a[5]*a[6]-a[3]*a[8])/det;
    b[4]=(a[0]*a[8]-a[2]*a[6])/det; b[5]=(a[2]*a[3]-a[0]*a[5])/det;
    b[6]=(a[3]*a[7]-a[4]*a[6])/det; b[7]=(a[1]*a[6]-a[0]*a[7])/det;
    b[8]=(a[0]*a[4]-a[1]*a[3])/det;
}
}
