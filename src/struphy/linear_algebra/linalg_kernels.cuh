#pragma once
namespace struphy_cuda::linalg_kernels {
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
 * Product of two 3x3 matrices, as in linalg_kernels.matrix_matrix.
 *
 * @param a First input matrix stored as nine row-major entries.
 * @param b Second input matrix stored as nine row-major entries.
 * @param c Output a.dot(b) with nine row-major entries; must not overlap a or b.
 */
__device__ inline void matrix_matrix(const double* a, const double* b, double* c) {
    for (int i = 0; i < 9; ++i) c[i] = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
}
/**
 * Determinant of a 3x3 matrix, as in linalg_kernels.det.
 *
 * @param a Input matrix stored as nine row-major entries.
 * @return det_a, the determinant of a, computed as plus - minus like the pyccel kernel.
 */
__device__ inline double det(const double* a) {
    double plus = a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7];
    double minus = a[6] * a[4] * a[2] + a[7] * a[5] * a[0] + a[8] * a[3] * a[1];
    double det_a = plus - minus;
    return det_a;
}
/**
 * Invert a 3x3 matrix whose determinant is known, as in linalg_kernels.matrix_inv_with_det.
 *
 * @param a Non-singular input matrix stored as nine row-major entries.
 * @param det_a The determinant of a (pull() passes 1.0 to get the adjugate).
 * @param b Output inverse with nine row-major entries; must not overlap a.
 */
__device__ inline void matrix_inv_with_det(const double* a, double det_a, double* b) {
    b[0] = (a[4] * a[8] - a[7] * a[5]) / det_a;
    b[1] = (a[7] * a[2] - a[1] * a[8]) / det_a;
    b[2] = (a[1] * a[5] - a[4] * a[2]) / det_a;

    b[3] = (a[5] * a[6] - a[8] * a[3]) / det_a;
    b[4] = (a[8] * a[0] - a[2] * a[6]) / det_a;
    b[5] = (a[2] * a[3] - a[5] * a[0]) / det_a;

    b[6] = (a[3] * a[7] - a[6] * a[4]) / det_a;
    b[7] = (a[6] * a[1] - a[0] * a[7]) / det_a;
    b[8] = (a[0] * a[4] - a[3] * a[1]) / det_a;
}
/**
 * Invert a 3x3 matrix, as in linalg_kernels.matrix_inv.
 *
 * @param a Non-singular input matrix stored as nine row-major entries.
 * @param b Output inverse with nine row-major entries; must not overlap a.
 *
 * det_a is the determinant of a, computed with det() as in pyccel. No singularity check is performed.
 */
__device__ inline void matrix_inv(const double* a, double* b) {
    double det_a = det(a);
    matrix_inv_with_det(a, det_a, b);
}
/**
 * Scalar (dot) product of two vectors of length 3, as in linalg_kernels.scalar_dot.
 *
 * @param a First input vector with three entries.
 * @param b Second input vector with three entries.
 * @return value, the scalar product of a and b.
 */
__device__ inline double scalar_dot(const double* a, const double* b) {
    double value = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return value;
}
/**
 * Vector (cross) product of two vectors of length 3, as in linalg_kernels.cross.
 *
 * @param a First input vector with three entries.
 * @param b Second input vector with three entries.
 * @param c Output vector a x b with three entries; must not overlap a or b.
 */
__device__ inline void cross(const double* a, const double* b, double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
/**
 * Transpose of a 3x3 matrix, as in linalg_kernels.transpose.
 *
 * @param a Input matrix stored as nine row-major entries.
 * @param b Output transpose with nine row-major entries; must not overlap a.
 */
__device__ inline void transpose(const double* a, double* b) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) b[3 * i + j] = a[3 * j + i];
}
/**
 * Outer product of two vectors of length 3, as in linalg_kernels.outer.
 *
 * @param a First input vector with three entries.
 * @param b Second input vector with three entries.
 * @param c Output matrix c_ij = a_i * b_j with nine row-major entries; must not overlap a or b.
 */
__device__ inline void outer(const double* a, const double* b, double* c) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) c[3 * i + j] = a[i] * b[j];
}
}  // namespace struphy_cuda::linalg_kernels
