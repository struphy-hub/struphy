#pragma once
// Device versions of geometry/transform_kernels.py: pull-backs, push-forwards and transformations between scalar
// fields, vector fields and differential p-forms at one logical point. Matrices are stored as nine row-major entries
// (m[3 * i + j] is pyccel's m[i, j]); dfmat1..dfmat3, vec1 and vec2 are the stack arrays of the pyccel helpers.
// Vector-valued kinds (kind_fun >= 10) read and write three entries of a and out; scalar kinds only a[0], out[0].
#include "struphy/geometry/evaluation_kernels.cuh"
#include "struphy/linear_algebra/linalg_kernels.cuh"
namespace struphy_cuda {
/**
 * Pull-back of a Cartesian scalar/vector field to a differential p-form, as in transform_kernels.pull.
 *
 * @param a Value of the scalar field a[0], or the Cartesian components (a[0], a[1], a[2]).
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param kind_fun Which pull-back: 0 (0-form), 1 (3-form), 10 (1-form), 11 (2-form), 12 (vector).
 * @param args_domain Mapping arguments.
 * @param out Output values (one or three entries); other kind_fun values leave it unchanged.
 */
__device__ inline void pull(const double* a, double eta1, double eta2, double eta3, int kind_fun,
                            const DomainArgs& args_domain, double* out) {
    double dfmat1[9], dfmat2[9];
    double detdf = 0.0;

    // evaluate Jacobian matrix and its determinant
    if (kind_fun > 0) {
        df(eta1, eta2, eta3, args_domain, dfmat1);
        detdf = det(dfmat1);
    }

    // 0-form
    if (kind_fun == 0) {
        out[0] = a[0];
    }
    // 3-form
    else if (kind_fun == 1) {
        out[0] = a[0] * fabs(detdf);
    }
    // 1-form
    else if (kind_fun == 10) {
        transpose(dfmat1, dfmat2);
        matrix_vector(dfmat2, a, out);
    }
    // 2-form
    else if (kind_fun == 11) {
        matrix_inv_with_det(dfmat1, 1.0, dfmat2);
        matrix_vector(dfmat2, a, out);

        if (detdf < 0.0)
            for (int i = 0; i < 3; ++i) out[i] = -out[i];
    }
    // vector
    else if (kind_fun == 12) {
        matrix_inv(dfmat1, dfmat2);
        matrix_vector(dfmat2, a, out);
    }
}

/**
 * Push-forward of a differential p-form to a Cartesian scalar/vector field, as in transform_kernels.push.
 *
 * @param a Value of the scalar p-form a[0], or its components (a[0], a[1], a[2]).
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param kind_fun Which push-forward: 0 (0-form), 1 (3-form), 10 (1-form), 11 (2-form), 12 (vector).
 * @param args_domain Mapping arguments.
 * @param out Output values (one or three entries); other kind_fun values leave it unchanged.
 */
__device__ inline void push(const double* a, double eta1, double eta2, double eta3, int kind_fun,
                            const DomainArgs& args_domain, double* out) {
    double dfmat1[9], dfmat2[9], dfmat3[9];
    double detdf = 0.0;

    // evaluate Jacobian matrix and its determinant
    if (kind_fun > 0) {
        df(eta1, eta2, eta3, args_domain, dfmat1);
        detdf = det(dfmat1);
    }

    // 0-form
    if (kind_fun == 0) {
        out[0] = a[0];
    }
    // 3-form
    else if (kind_fun == 1) {
        out[0] = a[0] / fabs(detdf);
    }
    // 1-form
    else if (kind_fun == 10) {
        matrix_inv_with_det(dfmat1, detdf, dfmat2);
        transpose(dfmat2, dfmat3);
        matrix_vector(dfmat3, a, out);
    }
    // 2-form
    else if (kind_fun == 11) {
        matrix_vector(dfmat1, a, out);
        for (int i = 0; i < 3; ++i) out[i] = out[i] / fabs(detdf);
    }
    // vector
    else if (kind_fun == 12) {
        matrix_vector(dfmat1, a, out);
    }
}

/**
 * Euclidean norm of column `column` of a 3x3 row-major matrix, sqrt(m[0, j]^2 + m[1, j]^2 + m[2, j]^2).
 *
 * CUDA-only helper for the expression that transform_kernels.tran repeats for the normalized-vector kinds.
 *
 * @param dfmat1 Input 3x3 matrix (DF) stored as nine row-major entries.
 * @param column Column index j (0, 1 or 2).
 * @return The norm h_j = ||DF[:, j]||.
 */
__device__ inline double column_norm(const double* dfmat1, int column) {
    return sqrt(pow(dfmat1[column], 2) + pow(dfmat1[3 + column], 2) + pow(dfmat1[6 + column], 2));
}

/**
 * Transformation between differential p-forms and/or vector fields, as in transform_kernels.tran.
 *
 * @param a Value of the scalar function a[0], or the components (a[0], a[1], a[2]).
 * @param eta1 Logical coordinate along the first axis.
 * @param eta2 Logical coordinate along the second axis.
 * @param eta3 Logical coordinate along the third axis.
 * @param kind_fun Which transformation (Domain.dict_transformations["tran"]): 0 0_to_3, 1 3_to_0, 10 1_to_2,
 *        11 2_to_1, 12 norm_to_v, 13 norm_to_1, 14 norm_to_2, 15 v_to_1, 16 v_to_2, 17 1_to_v, 18 2_to_v,
 *        19 1_to_norm, 20 2_to_norm, 21 v_to_norm.
 * @param args_domain Mapping arguments.
 * @param out Output values (one or three entries); other kind_fun values leave it unchanged.
 */
__device__ inline void tran(const double* a, double eta1, double eta2, double eta3, int kind_fun,
                            const DomainArgs& args_domain, double* out) {
    double dfmat1[9], dfmat2[9], dfmat3[9];
    double vec1[3], vec2[3];

    // evaluate Jacobian matrix and its determinant
    df(eta1, eta2, eta3, args_domain, dfmat1);
    double detdf = det(dfmat1);

    // 0-form to 3-form
    if (kind_fun == 0) {
        out[0] = a[0] * fabs(detdf);
    }
    // 3-form to 0-form
    else if (kind_fun == 1) {
        out[0] = a[0] / fabs(detdf);
    }
    // 1-form to 2-form (a^2 = G^(-1) * a^1 * |det(DF)|)
    else if (kind_fun == 10) {
        matrix_inv_with_det(dfmat1, detdf, dfmat2);
        transpose(dfmat2, dfmat3);
        matrix_vector(dfmat3, a, vec1);
        matrix_vector(dfmat2, vec1, out);
        for (int i = 0; i < 3; ++i) out[i] = out[i] * fabs(detdf);
    }
    // 2-form to 1-form (a^1 = G * a^2 / |det(DF)|)
    else if (kind_fun == 11) {
        transpose(dfmat1, dfmat2);
        matrix_vector(dfmat1, a, vec1);
        matrix_vector(dfmat2, vec1, out);
        for (int i = 0; i < 3; ++i) out[i] = out[i] / fabs(detdf);
    }
    // norm vector to vector
    else if (kind_fun == 12) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] / column_norm(dfmat1, i);
    }
    // norm vector to 1-form (a^1 = G * a)
    else if (kind_fun == 13) {
        for (int i = 0; i < 3; ++i) vec1[i] = a[i] / column_norm(dfmat1, i);
        transpose(dfmat1, dfmat2);
        matrix_vector(dfmat1, vec1, vec2);
        matrix_vector(dfmat2, vec2, out);
    }
    // norm vector to 2-form (a^2 = |det(DF)| * a)
    else if (kind_fun == 14) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] / column_norm(dfmat1, i);
        for (int i = 0; i < 3; ++i) out[i] = out[i] * fabs(detdf);
    }
    // vector to 1-form (a^1 = G * a)
    else if (kind_fun == 15) {
        transpose(dfmat1, dfmat2);
        matrix_vector(dfmat1, a, vec1);
        matrix_vector(dfmat2, vec1, out);
    }
    // vector to 2-form (a^2 = |det(DF)| * a)
    else if (kind_fun == 16) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] * fabs(detdf);
    }
    // 1-form to vector (a = G^(-1) * a^1)
    else if (kind_fun == 17) {
        matrix_inv_with_det(dfmat1, detdf, dfmat2);
        transpose(dfmat2, dfmat3);
        matrix_vector(dfmat3, a, vec1);
        matrix_vector(dfmat2, vec1, out);
    }
    // 2-form to vector (a = a^2 / |det(DF)|)
    else if (kind_fun == 18) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] / fabs(detdf);
    }
    // 1-form to normalized vector
    else if (kind_fun == 19) {
        matrix_inv_with_det(dfmat1, detdf, dfmat2);
        transpose(dfmat2, dfmat3);
        matrix_vector(dfmat3, a, vec1);
        matrix_vector(dfmat2, vec1, out);
        for (int i = 0; i < 3; ++i) out[i] = out[i] * column_norm(dfmat1, i);
    }
    // 2-form to normalized vector
    else if (kind_fun == 20) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] / fabs(detdf);
        for (int i = 0; i < 3; ++i) out[i] = out[i] * column_norm(dfmat1, i);
    }
    // Cartesian vector to normalized vector
    else if (kind_fun == 21) {
        for (int i = 0; i < 3; ++i) out[i] = a[i] * column_norm(dfmat1, i);
    }
}
}
