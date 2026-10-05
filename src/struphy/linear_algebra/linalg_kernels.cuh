#pragma once
namespace struphy_cuda {
/**
 * Multiply a 3x3 matrix by a vector, as in linalg_kernels.matrix_vector.
 *
 * @param a Input matrix stored as nine row-major entries.
 * @param b Input vector with three entries.
 * @param c Output vector with three entries; must not overlap a or b.
 */
__device__ inline void matrix_vector(const double* a, const double* b, double* c) {
    for (int i=0; i<3; ++i) c[i]=a[3*i]*b[0]+a[3*i+1]*b[1]+a[3*i+2]*b[2];
}
/**
 * Invert a 3x3 matrix, as in linalg_kernels.matrix_inv.
 *
 * @param a Non-singular input matrix stored as nine row-major entries.
 * @param b Output inverse with nine row-major entries; must not overlap a.
 *
 * det_a is the determinant of a. No singularity check is performed.
 */
__device__ inline void matrix_inv(const double* a, double* b) {
    double det_a=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
    b[0]=(a[4]*a[8]-a[5]*a[7])/det_a; b[1]=(a[2]*a[7]-a[1]*a[8])/det_a;
    b[2]=(a[1]*a[5]-a[2]*a[4])/det_a; b[3]=(a[5]*a[6]-a[3]*a[8])/det_a;
    b[4]=(a[0]*a[8]-a[2]*a[6])/det_a; b[5]=(a[2]*a[3]-a[0]*a[5])/det_a;
    b[6]=(a[3]*a[7]-a[4]*a[6])/det_a; b[7]=(a[1]*a[6]-a[0]*a[7])/det_a;
    b[8]=(a[0]*a[4]-a[1]*a[3])/det_a;
}
}
