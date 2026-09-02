extern "C" __global__
void cc_lin_mhd_5d_curlb_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int kind_map, const double* params,
    const double epsilon, const double ep_scale,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b_1, const int b1_n2, const int b1_n3,
    const double* b_2, const int b2_n2, const int b2_n3,
    const double* b_3, const int b3_n2, const int b3_n3,
    const double* nb1, const int n1_n2, const int n1_n3,
    const double* nb2, const int n2_n2, const int n2_n3,
    const double* nb3, const int n3_n2, const int n3_n3,
    const double* cnb1, const int c1_n2, const int c1_n3,
    const double* cnb2, const int c2_n2, const int c2_n3,
    const double* cnb3, const int c3_n2, const int c3_n3,
    const int basis_u,
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
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double weight = row[5];
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double bfull_star[3];
    for (int k = 0; k < 3; k++) bfull_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, bfull_star);

    // tmp = curl_norm_b (x) curl_norm_b
    double tmp[9];
    outer_dev(curl_norm_b, curl_norm_b, tmp);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};
    double b_prod_neg[9];
    for (int k = 0; k < 9; k++) b_prod_neg[k] = -b_prod[k];

    double tmp1[9], tmp_m[9], tmp_v[3];
    matmat_dev(b_prod, tmp, tmp1);
    matmat_dev(tmp1, b_prod_neg, tmp_m);
    matvec_dev(b_prod, curl_norm_b, tmp_v);

    double fm[9], fv[3];
    if (basis_u == 0) {
        const double sm = weight * v * v / (abs_b_star_para * abs_b_star_para) * ep_scale;
        const double sv = weight * v * v / abs_b_star_para * ep_scale;
        for (int k = 0; k < 9; k++) fm[k] = tmp_m[k] * sm;
        for (int k = 0; k < 3; k++) fv[k] = tmp_v[k] * sv;
    } else {
        const double sm = weight * v * v / (abs_b_star_para * abs_b_star_para)
                        / (det_df * det_df) * ep_scale;
        const double sv = weight * v * v / abs_b_star_para / det_df * ep_scale;
        for (int k = 0; k < 9; k++) fm[k] = tmp_m[k] * sm;
        for (int k = 0; k < 3; k++) fv[k] = tmp_v[k] * sv;
    }

    const double f11 = fm[0], f12 = fm[1], f13 = fm[2];
    const double f22 = fm[4], f23 = fm[5], f33 = fm[8];
    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;

    if (basis_u == 0) {
        // V0vec: every block N-N-N
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, f11, vec1, v1_n2,v1_n3, fv[0]);
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, f22, vec2, v2_n2,v2_n3, fv[1]);
        fill_mat_vec_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, f33, vec3, v3_n2,v3_n3, fv[2]);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat12, m12_d2,m12_d3,m12_d4,m12_d5,m12_d6, f12);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat13, m13_d2,m13_d3,m13_d4,m13_d5,m13_d6, f13);
        fill_mat_dev(p1,p2,p3, p1,p2,p3, bn1,bn2,bn3, bn1,bn2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat23, m23_d2,m23_d3,m23_d4,m23_d5,m23_d6, f23);

    } else if (basis_u == 2) {
        // V2 (Hdiv): comp1 N-D-D, comp2 D-N-D, comp3 D-D-N
        fill_mat_vec_dev(p1,pd2,pd3, p1,pd2,pd3, bn1,bd2,bd3, bn1,bd2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat11, m11_d2,m11_d3,m11_d4,m11_d5,m11_d6, f11, vec1, v1_n2,v1_n3, fv[0]);
        fill_mat_vec_dev(pd1,p2,pd3, pd1,p2,pd3, bd1,bn2,bd3, bd1,bn2,bd3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat22, m22_d2,m22_d3,m22_d4,m22_d5,m22_d6, f22, vec2, v2_n2,v2_n3, fv[1]);
        fill_mat_vec_dev(pd1,pd2,p3, pd1,pd2,p3, bd1,bd2,bn3, bd1,bd2,bn3,
            span1,span2,span3, start0,start1,start2, p1,p2,p3,
            mat33, m33_d2,m33_d3,m33_d4,m33_d5,m33_d6, f33, vec3, v3_n2,v3_n3, fv[2]);
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

