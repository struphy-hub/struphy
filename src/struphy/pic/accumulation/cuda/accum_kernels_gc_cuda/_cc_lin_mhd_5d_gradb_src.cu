extern "C" __global__
void cc_lin_mhd_5d_gradB_cuda(
    const double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int mu_idx,
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
    const double* gpb1, const int g1_n2, const int g1_n3,
    const double* gpb2, const int g2_n2, const int g2_n3,
    const double* gpb3, const int g3_n2, const int g3_n3,
    const double* gpq1, const int q1_n2, const int q1_n3,
    const double* gpq2, const int q2_n2, const int q2_n3,
    const double* gpq3, const int q3_n2, const int q3_n3,
    const int basis_u,
    double* vec1, const int v1_n2, const int v1_n3,
    double* vec2, const int v2_n2, const int v2_n3,
    double* vec3, const int v3_n2, const int v3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double weight = row[5];
    const double v = row[3];
    const double mu = row[mu_idx];

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

    double b[3], norm_b1[3], curl_norm_b[3], grad_PB[3], grad_PBeq[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gpb1,g1_n2,g1_n3, gpb2,g2_n2,g2_n3, gpb3,g3_n2,g3_n3, grad_PB);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gpq1,q1_n2,q1_n3, gpq2,q2_n2,q2_n3, gpq3,q3_n2,q3_n3, grad_PBeq);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;
    double tmp[9], tmp_v[3], fv[3];

    if (basis_u == 0) {
        matmat_dev(b_prod, norm_b_prod, tmp);
        matvec_dev(tmp, grad_PB, tmp_v);
        for (int k = 0; k < 3; k++) fv[k] = weight * tmp_v[k] * mu / abs_b_star_para * ep_scale;

        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec1, v1_n2,v1_n3, fv[0]);
        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec2, v2_n2,v2_n3, fv[1]);
        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec3, v3_n2,v3_n3, fv[2]);

    } else if (basis_u == 2) {
        for (int k = 0; k < 3; k++) grad_PB[k] += grad_PBeq[k];
        matmat_dev(b_prod, norm_b_prod, tmp);
        matvec_dev(tmp, grad_PB, tmp_v);
        for (int k = 0; k < 3; k++)
            fv[k] = weight * tmp_v[k] * mu / abs_b_star_para / det_df * ep_scale;

        // Hdiv components: N-D-D, D-N-D, D-D-N
        fill_vec_dev(p1,pd2,pd3, bn1,bd2,bd3, span1,span2,span3, start0,start1,start2,
            vec1, v1_n2,v1_n3, fv[0]);
        fill_vec_dev(pd1,p2,pd3, bd1,bn2,bd3, span1,span2,span3, start0,start1,start2,
            vec2, v2_n2,v2_n3, fv[1]);
        fill_vec_dev(pd1,pd2,p3, bd1,bd2,bn3, span1,span2,span3, start0,start1,start2,
            vec3, v3_n2,v3_n3, fv[2]);
    }
}

