extern "C" __global__
void push_gc_cc_J1_H1vec_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
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
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
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

    double b[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_vectorfield_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u, e);
    const double temp = dot3_dev(e, curl_norm_b);

    row[3] += temp / abs_b_star_para * v * dt;
}

extern "C" __global__
void push_gc_cc_J1_Hcurl_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
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
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
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

    double b[3], u_form[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u_form);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    // g_inv = (DF^T DF)^-1, transforms the 1-form u into H1vec components
    double df_t[9] = {
        dfm[0], dfm[3], dfm[6],
        dfm[1], dfm[4], dfm[7],
        dfm[2], dfm[5], dfm[8],
    };
    double g[9], g_inv[9], u0[3];
    matmat_dev(df_t, dfm, g);
    matrix_inv_dev(g, g_inv);
    matvec_dev(g_inv, u_form, u0);

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = (b[k] + curl_norm_b[k] * v * epsilon) / det_df;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u0, e);
    const double temp = dot3_dev(e, curl_norm_b) / det_df;

    row[3] += temp / abs_b_star_para * v * dt;
}

extern "C" __global__
void push_gc_cc_J1_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const double dt,
    const int kind_map, const double* params,
    const double epsilon,
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
    const double* u_1, const int u1_n2, const int u1_n3,
    const double* u_2, const int u2_n2, const int u2_n3,
    const double* u_3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
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

    double b[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    for (int k = 0; k < 3; k++) u[k] /= det_df;

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = b[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double e[3];
    cross_dev(b, u, e);
    const double temp = dot3_dev(e, curl_norm_b);

    row[3] += temp / abs_b_star_para * v * dt;
}

