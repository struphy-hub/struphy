extern "C" __global__
void pc_lin_mhd_6d_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double ep_scale,
    double* mat11_11, double* mat12_11, double* mat13_11, double* mat22_11, double* mat23_11, double* mat33_11, double* mat11_12, double* mat12_12, double* mat13_12, double* mat22_12, double* mat23_12, double* mat33_12, double* mat11_22, double* mat12_22, double* mat13_22, double* mat22_22, double* mat23_22, double* mat33_22,
    double* vec1_1, double* vec2_1, double* vec3_1, double* vec1_2, double* vec2_2, double* vec3_2,
    const int m11_d2, const int m11_d3, const int m11_d4, const int m11_d5, const int m11_d6, const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6, const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6, const int m22_d2, const int m22_d3, const int m22_d4, const int m22_d5, const int m22_d6, const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6, const int m33_d2, const int m33_d3, const int m33_d4, const int m33_d5, const int m33_d6,
    const int v1_n2, const int v1_n3, const int v2_n2, const int v2_n3, const int v3_n2, const int v3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v[3] = {row[3], row[4], row[5]};
    const double weight = row[6];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    double df_inv[9];
    matrix_inv_dev(dfm, df_inv);
    double g_inv[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += df_inv[3*i+k] * df_inv[3*j+k];
            g_inv[3*i+j] = s;
        }
    double tmp_v[3];
    matvec_dev(df_inv, v, tmp_v);

    const double fill11 = weight * g_inv[0] * ep_scale;
    const double fill12 = weight * g_inv[1] * ep_scale;
    const double fill13 = weight * g_inv[2] * ep_scale;
    const double fill22 = weight * g_inv[4] * ep_scale;
    const double fill23 = weight * g_inv[5] * ep_scale;
    const double fill33 = weight * g_inv[8] * ep_scale;
    const double fill1 = weight * tmp_v[0] * ep_scale;
    const double fill2 = weight * tmp_v[1] * ep_scale;
    const double fill3 = weight * tmp_v[2] * ep_scale;
    const double vx = v[0], vy = v[1];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);
    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    fill_mat_vec_pressure_dev(pd1,p2,p3, pd1,p2,p3, bd1,bn2,bn3, bd1,bn2,bn3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat11_11, mat11_12, mat11_22, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, fill11,
        vec1_1, vec1_2, v1_n2,v1_n3, fill1, vx,vy);

    fill_mat_pressure_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat12_11, mat12_12, mat12_22, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, fill12, vx,vy);

    fill_mat_pressure_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat13_11, mat13_12, mat13_22, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, fill13, vx,vy);

    fill_mat_vec_pressure_dev(p1,pd2,p3, p1,pd2,p3, bn1,bd2,bn3, bn1,bd2,bn3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat22_11, mat22_12, mat22_22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, fill22,
        vec2_1, vec2_2, v2_n2,v2_n3, fill2, vx,vy);

    fill_mat_pressure_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat23_11, mat23_12, mat23_22, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, fill23, vx,vy);

    fill_mat_vec_pressure_dev(p1,p2,pd3, p1,p2,pd3, bn1,bn2,bd3, bn1,bn2,bd3, span1,span2,span3, start0,start1,start2, p1,p2,p3,
        mat33_11, mat33_12, mat33_22, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, fill33,
        vec3_1, vec3_2, v3_n2,v3_n3, fill3, vx,vy);

}
