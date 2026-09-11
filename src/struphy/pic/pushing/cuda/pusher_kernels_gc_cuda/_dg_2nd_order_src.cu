extern "C" __global__
void push_gc_bxEstar_discrete_gradient_2nd_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* ub1, const int u1_n2, const int u1_n3,
    const double* ub2, const int u2_n2, const int u2_n3,
    const double* ub3, const int u3_n2, const int u3_n3,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_k[k] = row[k] + row[first_shift_idx + k];
        eta_n[k] = row[first_init_idx + k];
        double m = fmod((eta_k[k] + eta_n[k]) / 2.0, 1.0);
        if (m < 0.0) m += 1.0;
        eta_mid[k] = m;
        eta_diff[k] = eta_k[k] - eta_n[k];
    }
    const double v = row[3];
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double H_k = row[first_free_idx + 1];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double unit_b1[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ub1,u1_n2,u1_n3, ub2,u2_n2,u2_n3, ub3,u3_n2,u3_n3, unit_b1);

    double grad_H[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,g1_n2,g1_n3, gb2,g2_n2,g2_n3, gb3,g3_n2,g3_n3, grad_H);
    for (int k = 0; k < 3; k++) grad_H[k] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            ef1,e1_n2,e1_n3, ef2,e2_n2,e2_n3, ef3,e3_n2,e3_n3, e_field);
        for (int k = 0; k < 3; k++) grad_H[k] -= e_field[k];
    }

    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H);
    const double dZ_squared = dot3_dev(eta_diff, eta_diff);

    double grad_I[3];
    if (dZ_squared == 0.0) {
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k];
    } else {
        const double s = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k] + eta_diff[k] * s;
    }

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);
    b_star_parallel = (b_star_parallel * epsilon * v + B_dot_b) * det_df;

    double Exb[3];
    cross_dev(unit_b1, grad_I, Exb);

    double k_vec[3];
    for (int k = 0; k < 3; k++) k_vec[k] = Exb[k] / b_star_parallel;

    row[0] = eta_n[0] + dt * k_vec[0];
    row[1] = eta_n[1] + dt * k_vec[1];
    row[2] = eta_n[2] + dt * k_vec[2];

    const double r0 = row[0] - eta_k[0], r1 = row[1] - eta_k[1], r2 = row[2] - eta_k[2];
    row[residual_idx] = sqrt(r0*r0 + r1*r1 + r2*r2);
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_2nd_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* b2_1, const int b1_n2, const int b1_n3,
    const double* b2_2, const int b2_n2, const int b2_n3,
    const double* b2_3, const int b3_n2, const int b3_n3,
    const double* cb1, const int c1_n2, const int c1_n3,
    const double* cb2, const int c2_n2, const int c2_n3,
    const double* cb3, const int c3_n2, const int c3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_k[k] = row[k] + row[first_shift_idx + k];
        eta_n[k] = row[first_init_idx + k];
        double m = fmod((eta_k[k] + eta_n[k]) / 2.0, 1.0);
        if (m < 0.0) m += 1.0;
        eta_mid[k] = m;
        eta_diff[k] = eta_k[k] - eta_n[k];
    }
    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_mid = (v_k + v_n) / 2.0;
    const double v_diff = v_k - v_n;
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double H_k = row[first_free_idx + 1];

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta_mid[0], eta_mid[1], eta_mid[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double grad_H[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,g1_n2,g1_n3, gb2,g2_n2,g2_n3, gb3,g3_n2,g3_n3, grad_H);
    for (int k = 0; k < 3; k++) grad_H[k] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            ef1,e1_n2,e1_n3, ef2,e2_n2,e2_n3, ef3,e3_n2,e3_n3, e_field);
        for (int k = 0; k < 3; k++) grad_H[k] -= e_field[k];
    }

    const double grad_H_v = epsilon * v_mid;
    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H) + v_diff * grad_H_v;
    const double dZ_squared = dot3_dev(eta_diff, eta_diff) + v_diff * v_diff;

    double grad_I[3];
    double grad_I_v;
    if (dZ_squared == 0.0) {
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k];
        grad_I_v = grad_H_v;
    } else {
        const double s = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int k = 0; k < 3; k++) grad_I[k] = grad_H[k] + eta_diff[k] * s;
        grad_I_v = grad_H_v + v_diff * s;
    }

    double b2[3], b_star[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b2_1,b1_n2,b1_n3, b2_2,b2_n2,b2_n3, b2_3,b3_n2,b3_n3, b2);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cb1,c1_n2,c1_n3, cb2,c2_n2,c2_n3, cb3,c3_n2,c3_n3, b_star);
    for (int k = 0; k < 3; k++) b_star[k] = b_star[k] * epsilon * v_mid + b2[k];

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);
    b_star_parallel = (b_star_parallel * epsilon * v_mid + B_dot_b) * epsilon * det_df;

    double k_vec[3];
    for (int k = 0; k < 3; k++) k_vec[k] = b_star[k] / b_star_parallel * grad_I_v;
    const double k_v = -dot3_dev(b_star, grad_I) / b_star_parallel;

    row[0] = eta_n[0] + dt * k_vec[0];
    row[1] = eta_n[1] + dt * k_vec[1];
    row[2] = eta_n[2] + dt * k_vec[2];
    row[3] = v_n + dt * k_v;

    const double r0 = row[0] - eta_k[0], r1 = row[1] - eta_k[1], r2 = row[2] - eta_k[2];
    const double rv = (row[3] - v_k) / v_k;
    row[residual_idx] = sqrt(r0*r0 + r1*r1 + r2*r2 + rv*rv);
}

