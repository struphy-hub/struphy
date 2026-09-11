extern "C" __global__
void eval_guiding_center_from_6d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b21, const int b1_n2, const int b1_n3,
    const double* b22, const int b2_n2, const int b2_n3,
    const double* b23, const int b3_n2, const int b3_n3,
    const double* absB, const int a_n2, const int a_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double x = row[first_diagnostics_idx];
    const double y = row[first_diagnostics_idx + 1];
    const double z = row[first_diagnostics_idx + 2];
    double v[3] = {row[3], row[4], row[5]};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b2[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        b21, b1_n2, b1_n3, b22, b2_n2, b2_n3, b23, b3_n2, b3_n3, b2);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, a_n2, a_n3);

    // normalized magnetic field, cartesian
    b2[0] /= abs_B; b2[1] /= abs_B; b2[2] /= abs_B;
    double norm_b_cart[3];
    matvec_dev(dfm, b2, norm_b_cart);
    norm_b_cart[0] /= det_df; norm_b_cart[1] /= det_df; norm_b_cart[2] /= det_df;

    const double v_parallel = dot3_dev(norm_b_cart, v);

    double temp[3], v_perp[3];
    cross_dev(v, norm_b_cart, temp);
    cross_dev(norm_b_cart, temp, v_perp);
    const double v_perp_square = v_perp[0]*v_perp[0] + v_perp[1]*v_perp[1] + v_perp[2]*v_perp[2];

    row[first_diagnostics_idx + 6] = v_parallel;
    row[first_diagnostics_idx + 4] = 0.5 * v_perp_square / abs_B;

    double Larmor_r[3];
    cross_dev(norm_b_cart, v_perp, Larmor_r);
    for (int k = 0; k < 3; k++) Larmor_r[k] = Larmor_r[k] / abs_B * epsilon;

    row[first_diagnostics_idx + 0] = x - Larmor_r[0];
    row[first_diagnostics_idx + 1] = y - Larmor_r[1];
    row[first_diagnostics_idx + 2] = z - Larmor_r[2];
}

