__device__ double dg_mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void cc_lin_mhd_5d_gradB_dg_cuda(
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
    const double* beq_1, const int e1_n2, const int e1_n3,
    const double* beq_2, const int e2_n2, const int e2_n3,
    const double* beq_3, const int e3_n2, const int e3_n3,
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
    const int basis_u, const double konst, const int is_dg,
    double* vec1, const int v1_n2, const int v1_n3,
    double* vec2, const int v2_n2, const int v2_n3,
    double* vec3, const int v3_n2, const int v3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3], eta_diff[3] = {0.0, 0.0, 0.0};
    if (is_dg) {
        for (int k = 0; k < 3; k++) {
            eta[k] = dg_mod1_dev((row[k] + row[first_init_idx + k]) / 2.0);
            eta_diff[k] = row[k] - row[first_init_idx + k];
        }
    } else {
        for (int k = 0; k < 3; k++) eta[k] = row[k];
    }

    const double weight = row[5];
    const double v = row[3];
    const double mu = row[mu_idx];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta[0], eta[1], eta[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    double b[3], beq[3], norm_b1[3], curl_norm_b[3], grad_PB[3], grad_PBeq[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b_1,b1_n2,b1_n3, b_2,b2_n2,b2_n3, b_3,b3_n2,b3_n3, b);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        beq_1,e1_n2,e1_n3, beq_2,e2_n2,e2_n3, beq_3,e3_n2,e3_n3, beq);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        nb1,n1_n2,n1_n3, nb2,n2_n2,n2_n3, nb3,n3_n2,n3_n3, norm_b1);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cnb1,c1_n2,c1_n3, cnb2,c2_n2,c2_n3, cnb3,c3_n2,c3_n3, curl_norm_b);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gpb1,g1_n2,g1_n3, gpb2,g2_n2,g2_n3, gpb3,g3_n2,g3_n3, grad_PB);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gpq1,q1_n2,q1_n3, gpq2,q2_n2,q2_n3, gpq3,q3_n2,q3_n3, grad_PBeq);

    // NOTE: unlike cc_lin_mhd_5d_gradB, B* here includes the equilibrium field.
    double bfull_star[3];
    for (int k = 0; k < 3; k++) bfull_star[k] = b[k] + beq[k] + curl_norm_b[k] * v * epsilon;
    const double abs_b_star_para = dot3_dev(norm_b1, bfull_star);

    double b_prod[9] = {0.0, -b[2], b[1], b[2], 0.0, -b[0], -b[1], b[0], 0.0};
    double beq_prod[9] = {0.0, -beq[2], beq[1], beq[2], 0.0, -beq[0], -beq[1], beq[0], 0.0};
    double norm_b_prod[9] = {
        0.0, -norm_b1[2], norm_b1[1],
        norm_b1[2], 0.0, -norm_b1[0],
        -norm_b1[1], norm_b1[0], 0.0};

    // basis_u == 0 has no 1/det; basis_u == 2 carries one.
    const double inv_det = (basis_u == 2) ? (1.0 / det_df) : 1.0;
    const double w_fac = weight * mu / abs_b_star_para * inv_det * ep_scale;
    const double d_fac = konst / abs_b_star_para * inv_det;

    double tmp[9], tmp_v[3], fv[3] = {0.0, 0.0, 0.0};

    // the two field blocks, Beq first then B, each contributing
    // grad_PBeq, grad_PB and (for `dg`) the eta_diff correction
    for (int blk = 0; blk < 2; blk++) {
        matmat_dev(blk == 0 ? beq_prod : b_prod, norm_b_prod, tmp);

        matvec_dev(tmp, grad_PBeq, tmp_v);
        for (int k = 0; k < 3; k++) fv[k] += tmp_v[k] * w_fac;

        matvec_dev(tmp, grad_PB, tmp_v);
        for (int k = 0; k < 3; k++) fv[k] += tmp_v[k] * w_fac;

        if (is_dg) {
            matvec_dev(tmp, eta_diff, tmp_v);
            for (int k = 0; k < 3; k++) fv[k] += tmp_v[k] * d_fac;
        }
    }

    if (basis_u == 0) {
        // H1vec: N-N-N in all three components
        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec1, v1_n2,v1_n3, fv[0]);
        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec2, v2_n2,v2_n3, fv[1]);
        fill_vec_dev(p1,p2,p3, bn1,bn2,bn3, span1,span2,span3, start0,start1,start2,
            vec3, v3_n2,v3_n3, fv[2]);
    } else if (basis_u == 2) {
        const int pd1 = p1 - 1, pd2 = p2 - 1, pd3 = p3 - 1;
        fill_vec_dev(p1,pd2,pd3, bn1,bd2,bd3, span1,span2,span3, start0,start1,start2,
            vec1, v1_n2,v1_n3, fv[0]);
        fill_vec_dev(pd1,p2,pd3, bd1,bn2,bd3, span1,span2,span3, start0,start1,start2,
            vec2, v2_n2,v2_n3, fv[1]);
        fill_vec_dev(pd1,pd2,p3, bd1,bd2,bn3, span1,span2,span3, start0,start1,start2,
            vec3, v3_n2,v3_n3, fv[2]);
    }
}

