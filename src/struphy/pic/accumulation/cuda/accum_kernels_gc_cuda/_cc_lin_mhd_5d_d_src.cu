extern "C" __global__
void cc_lin_mhd_5d_D_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const double epsilon, const double ep_scale,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int bb1_n2, const int bb1_n3,
    const double* b2_2, const int bb2_n2, const int bb2_n3,
    const double* b2_3, const int bb3_n2, const int bb3_n3,
    const double* nb11, const int nb1_n2, const int nb1_n3,
    const double* nb12, const int nb2_n2, const int nb2_n3,
    const double* nb13, const int nb3_n2, const int nb3_n3,
    const double* cnb1, const int cb1_n2, const int cb1_n3,
    const double* cnb2, const int cb2_n2, const int cb2_n3,
    const double* cnb3, const int cb3_n2, const int cb3_n3,
    const int basis_u,
    double* mat12, double* mat13, double* mat23,
    const int m12_d2, const int m12_d3, const int m12_d4, const int m12_d5, const int m12_d6,
    const int m13_d2, const int m13_d3, const int m13_d4, const int m13_d5, const int m13_d6,
    const int m23_d2, const int m23_d3, const int m23_d4, const int m23_d5, const int m23_d6)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];
    const double weight = row[5];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b2_1,bb1_n2,bb1_n3, b2_2,bb2_n2,bb2_n3, b2_3,bb3_n2,bb3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb11,nb1_n2,nb1_n3, nb12,nb2_n2,nb2_n3, nb13,nb3_n2,nb3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,cb1_n2,cb1_n3, cnb2,cb2_n2,cb2_n3, cnb3,cb3_n2,cb3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + epsilon * v * curl_norm_b[k];

    const double b_para = dot3_dev(norm_b1, b);
    const double b_star_para = dot3_dev(norm_b1, b_star);
    const double density_const = 1.0 - b_para / b_star_para;

    const double pref = -weight * density_const * ep_scale / epsilon;
    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    double f12, f13, f23;

    if (basis_u == 0) {
        f12 = pref * b_prod[1];
        f13 = pref * b_prod[2];
        f23 = pref * b_prod[5];

        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);

    } else if (basis_u == 1) {
        double df_inv[9], g_inv[9];
        matrix_inv_dev(dfm, df_inv);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double sacc = 0.0;
                for (int k = 0; k < 3; k++) sacc += df_inv[3*i+k] * df_inv[3*j+k];
                g_inv[3*i+j] = sacc;
            }
        double tmp1[9], tmp2[9];
        matmat_dev(g_inv, b_prod, tmp1);
        matmat_dev(tmp1, g_inv, tmp2);

        f12 = pref * tmp2[1];
        f13 = pref * tmp2[2];
        f23 = pref * tmp2[5];

        fill_mat_dev(pd1,p2,p3, p1,pd2,p3, bd1,bn2,bn3, bn1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(pd1,p2,p3, p1,p2,pd3, bd1,bn2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(p1,pd2,p3, p1,p2,pd3, bn1,bd2,bn3, bn1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);

    } else if (basis_u == 2) {
        const double det2 = det_df * det_df;
        f12 = pref * b_prod[1] / det2;
        f13 = pref * b_prod[2] / det2;
        f23 = pref * b_prod[5] / det2;

        fill_mat_dev(p1,pd2,pd3, pd1,p2,pd3, bn1,bd2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(p1,pd2,pd3, pd1,pd2,p3, bn1,bd2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(pd1,p2,pd3, pd1,pd2,p3, bd1,bn2,bd3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);
    }
}

