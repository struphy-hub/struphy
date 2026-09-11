extern "C" __global__
void push_gc_cc_J2_dg_init_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx,
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
    if (row[first_init_idx] == -1.0) return;

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

    double bb[3], u[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    for (int k = 0; k < 3; k++) e[k] /= (abs_b_star_para * det_df);

    row[0] -= dt * e[0];
    row[1] -= dt * e[1];
    row[2] -= dt * e[2];
}

extern "C" __global__
void push_gc_cc_J2_dg_Hdiv_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx,
    const double dt, const double const_, const double alpha,
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
    const double* u_3, const int u3_n2, const int u3_n3,
    const double* ud_1, const int ud1_n2, const int ud1_n3,
    const double* ud_2, const int ud2_n2, const int ud2_n3,
    const double* ud_3, const int ud3_n2, const int ud3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta_old0 = row[0], eta_old1 = row[1], eta_old2 = row[2];
    double eta_mid[3];
    eta_mid[0] = mod1_dev((row[0] + row[first_init_idx + 0]) / 2.0);
    eta_mid[1] = mod1_dev((row[1] + row[first_init_idx + 1]) / 2.0);
    eta_mid[2] = mod1_dev((row[2] + row[first_init_idx + 2]) / 2.0);
    const double v = row[3];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double bb[3], u[3], ud[3], norm_b1[3], curl_norm_b[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        u_1,u1_n2,u1_n3, u_2,u2_n2,u2_n3, u_3,u3_n2,u3_n3, u);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ud_1,ud1_n2,ud1_n3, ud_2,ud2_n2,ud2_n3, ud_3,ud3_n2,ud3_n3, ud);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);

    double b_prod[9] = {0.0, -bb[2], bb[1], bb[2], 0.0, -bb[0], -bb[1], bb[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    double b_star[3];
    for (int k = 0; k < 3; k++) b_star[k] = bb[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, b_star);

    double tmp[9], e[3], e2[3];
    matmat_dev(norm_b_prod, b_prod, tmp);
    matvec_dev(tmp, u, e);
    matvec_dev(tmp, ud, e2);
    for (int k = 0; k < 3; k++) e[k] = (e[k] + const_ * e2[k]) / (abs_b_star_para * det_df);

    double eta_new[3];
    eta_new[0] = row[first_init_idx + 0] - dt * e[0];
    eta_new[1] = row[first_init_idx + 1] - dt * e[1];
    eta_new[2] = row[first_init_idx + 2] - dt * e[2];

    row[0] = alpha * eta_new[0] + (1.0 - alpha) * eta_old0;
    row[1] = alpha * eta_new[1] + (1.0 - alpha) * eta_old1;
    row[2] = alpha * eta_new[2] + (1.0 - alpha) * eta_old2;
}

