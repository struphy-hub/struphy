__device__ void outer_dev(const double* a, const double* b, double* c)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            c[3*i+j] = a[i] * b[j];
}

// Port of filler_kernels.fill_mat_vec: fills one matrix block (banded
// storage, j = pad + jl - il) and, along the shared (i1,i2,i3) row loop,
// also fills the corresponding vector block.
__device__ void fill_mat_vec_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat, int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec, int vn2, int vn3,
    double filling_vec)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];

                atomicAdd(&vec[(size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3], b3 * filling_vec);

                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat[idx], b6);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat: matrix-only block fill (off-diagonal
// blocks, no associated vector).
__device__ void fill_mat_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat, int d2, int d3, int d4, int d5, int d6,
    double filling_mat)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1] * filling_mat;
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1];
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat[idx], b6);
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__
void linear_vlasov_ampere_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const double* f0_values,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    double* mat11, double* mat12, double* mat13,
    double* mat22, double* mat23, double* mat33,
    double* vec1, double* vec2, double* vec3,
    const int m11_d2, const int m11_d3, const int m11_d4, const int m11_d5, const int m11_d6,
    const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6,
    const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6,
    const int m22_d2, const int m22_d3, const int m22_d4, const int m22_d5, const int m22_d6,
    const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6,
    const int m33_d2, const int m33_d3, const int m33_d4, const int m33_d5, const int m33_d6,
    const int v1_n2, const int v1_n3,
    const int v2_n2, const int v2_n3,
    const int v3_n2, const int v3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};
    const double weight = row[6];
    const double s0 = row[7];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    double df_inv[9], df_inv_v[3];
    matrix_inv_dev(dfm, df_inv);
    matvec_dev(df_inv, v, df_inv_v);

    double filling_m[9];
    outer_dev(df_inv_v, df_inv_v, filling_m);
    const double fm_scale = f0_values[ip] / s0;
    for (int k = 0; k < 9; k++) filling_m[k] *= fm_scale;

    double filling_v[3];
    filling_v[0] = weight * df_inv_v[0];
    filling_v[1] = weight * df_inv_v[1];
    filling_v[2] = weight * df_inv_v[2];

    const double fill11 = filling_m[0], fill12 = filling_m[1], fill13 = filling_m[2];
    const double fill22 = filling_m[4], fill23 = filling_m[5], fill33 = filling_m[8];

    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    fill_mat_vec_dev(pd1,p2,p3, pd1,p2,p3, bd1,bn2,bn3, bd1,bn2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11,
        vec1, v1_n2,v1_n3, filling_v[0]);

    fill_mat_vec_dev(p1,pd2,p3, p1,pd2,p3, bn1,bd2,bn3, bn1,bd2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22,
        vec2, v2_n2,v2_n3, filling_v[1]);

    fill_mat_vec_dev(p1,p2,pd3, p1,p2,pd3, bn1,bn2,bd3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33,
        vec3, v3_n2,v3_n3, filling_v[2]);

    fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12);

    fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13);

    fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
        span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23);
}

