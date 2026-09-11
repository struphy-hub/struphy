extern "C" __global__
void push_gc_bxEstar_explicit_multistage_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* unit_b1_1, const int ub1_n2, const int ub1_n3,
    const double* unit_b1_2, const int ub2_n2, const int ub2_n3,
    const double* unit_b1_3, const int ub3_n2, const int ub3_n3,
    const double* grad_b_full_1, const int gb1_n2, const int gb1_n3,
    const double* grad_b_full_2, const int gb2_n2, const int gb2_n3,
    const double* grad_b_full_3, const int gb3_n2, const int gb3_n3,
    const double* B_dot_b_coeffs, const int bdb_n2, const int bdb_n3,
    const double* curl_unit_b_dot_b0, const int cub_n2, const int cub_n3,
    const double* e_field_1, const int e1_n2, const int e1_n3,
    const double* e_field_2, const int e2_n2, const int e2_n3,
    const double* e_field_3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt_a, const double dt_b, const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v = row[3];
    const double mu = row[mu_idx];

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

    double unit_b1[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        unit_b1_1, ub1_n2, ub1_n3, unit_b1_2, ub2_n2, ub2_n3, unit_b1_3, ub3_n2, ub3_n3, unit_b1);

    double e_star[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        grad_b_full_1, gb1_n2, gb1_n3, grad_b_full_2, gb2_n2, gb2_n3, grad_b_full_3, gb3_n2, gb3_n3, e_star);
    e_star[0] *= -epsilon * mu;
    e_star[1] *= -epsilon * mu;
    e_star[2] *= -epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
            start0, start1, start2,
            e_field_1, e1_n2, e1_n3, e_field_2, e2_n2, e2_n3, e_field_3, e3_n2, e3_n3, e_field);
        e_star[0] += e_field[0];
        e_star[1] += e_field[1];
        e_star[2] += e_field[2];
    }

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, B_dot_b_coeffs, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, curl_unit_b_dot_b0, cub_n2, cub_n3);
    b_star_parallel = b_star_parallel * epsilon * v + B_dot_b;
    b_star_parallel *= det_df;

    double Exb[3];
    cross_dev(e_star, unit_b1, Exb);

    double k[3];
    k[0] = Exb[0] / b_star_parallel;
    k[1] = Exb[1] / b_star_parallel;
    k[2] = Exb[2] / b_star_parallel;

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

