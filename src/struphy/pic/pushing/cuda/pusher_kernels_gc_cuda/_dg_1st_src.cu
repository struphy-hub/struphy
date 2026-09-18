// mod(x, 1.0) matching numpy (result in [0, 1))
__device__ double mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void push_gc_bxEstar_discrete_gradient_1st_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int gb1_n2, const int gb1_n3,
    const double* gb2, const int gb2_n2, const int gb2_n3,
    const double* gb3, const int gb3_n2, const int gb3_n3,
    const double* e1c, const int e1_n2, const int e1_n3,
    const double* e2c, const int e2_n2, const int e2_n3,
    const double* e3c, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int i = 0; i < 3; i++) {
        eta_k[i] = row[i] + row[first_shift_idx + i];
        eta_n[i] = row[first_init_idx + i];
        eta_mid[i] = mod1_dev((eta_k[i] + eta_n[i]) / 2.0);
        eta_diff[i] = eta_k[i] - eta_n[i];
    }

    const double mu = row[mu_idx];
    const double H_n = row[first_free_idx];
    const double b_star_parallel = row[first_free_idx + 1];
    double unit_b1[3] = {
        row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k = row[first_free_idx + 5];

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
        gb1,gb1_n2,gb1_n3, gb2,gb2_n2,gb2_n3, gb3,gb3_n2,gb3_n3, grad_H);
    for (int i = 0; i < 3; i++) grad_H[i] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            e1c,e1_n2,e1_n3, e2c,e2_n2,e2_n3, e3c,e3_n2,e3_n3, e_field);
        for (int i = 0; i < 3; i++) grad_H[i] += -e_field[i];
    }

    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H);
    const double dZ_squared = dot3_dev(eta_diff, eta_diff);

    double grad_I[3];
    if (dZ_squared == 0.0) {
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i];
    } else {
        const double c = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i] + eta_diff[i] * c;
    }

    double Exb[3];
    cross_dev(unit_b1, grad_I, Exb);

    double k[3];
    for (int i = 0; i < 3; i++) k[i] = Exb[i] / b_star_parallel;

    for (int i = 0; i < 3; i++) row[i] = eta_n[i] + dt * k[i];

    row[residual_idx] = sqrt(
        (row[0] - eta_k[0]) * (row[0] - eta_k[0])
      + (row[1] - eta_k[1]) * (row[1] - eta_k[1])
      + (row[2] - eta_k[2]) * (row[2] - eta_k[2]));
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_1st_order_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int gb1_n2, const int gb1_n3,
    const double* gb2, const int gb2_n2, const int gb2_n3,
    const double* gb3, const int gb3_n2, const int gb3_n3,
    const double* e1c, const int e1_n2, const int e1_n3,
    const double* e2c, const int e2_n2, const int e2_n3,
    const double* e3c, const int e3_n2, const int e3_n3,
    const int evaluate_e_field,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_n[3], eta_mid[3], eta_diff[3];
    for (int i = 0; i < 3; i++) {
        eta_k[i] = row[i] + row[first_shift_idx + i];
        eta_n[i] = row[first_init_idx + i];
        eta_mid[i] = mod1_dev((eta_k[i] + eta_n[i]) / 2.0);
        eta_diff[i] = eta_k[i] - eta_n[i];
    }

    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_mid = (v_k + v_n) / 2.0;
    const double v_diff = v_k - v_n;

    const double mu = row[mu_idx];
    const double H_n = row[first_free_idx];
    const double b_star_parallel = epsilon * row[first_free_idx + 1];
    double b_star[3] = {
        row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k = row[first_free_idx + 5];

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
        gb1,gb1_n2,gb1_n3, gb2,gb2_n2,gb2_n3, gb3,gb3_n2,gb3_n3, grad_H);
    for (int i = 0; i < 3; i++) grad_H[i] *= epsilon * mu;

    if (evaluate_e_field) {
        double e_field[3];
        eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
            e1c,e1_n2,e1_n3, e2c,e2_n2,e2_n3, e3c,e3_n2,e3_n3, e_field);
        for (int i = 0; i < 3; i++) grad_H[i] += -e_field[i];
    }

    const double grad_H_v = epsilon * v_mid;
    const double dZ_dot_grad_H = dot3_dev(eta_diff, grad_H) + v_diff * grad_H_v;
    const double dZ_squared = dot3_dev(eta_diff, eta_diff) + v_diff * v_diff;

    double grad_I[3];
    double grad_I_v;
    if (dZ_squared == 0.0) {
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i];
        grad_I_v = grad_H_v;
    } else {
        const double c = (H_k - H_n - dZ_dot_grad_H) / dZ_squared;
        for (int i = 0; i < 3; i++) grad_I[i] = grad_H[i] + eta_diff[i] * c;
        grad_I_v = grad_H_v + v_diff * c;
    }

    double k[3];
    for (int i = 0; i < 3; i++) k[i] = b_star[i] / b_star_parallel * grad_I_v;

    double k_v = dot3_dev(b_star, grad_I);
    k_v /= -b_star_parallel;

    for (int i = 0; i < 3; i++) row[i] = eta_n[i] + dt * k[i];
    row[3] = v_n + dt * k_v;

    row[residual_idx] = sqrt(
        (row[0] - eta_k[0]) * (row[0] - eta_k[0])
      + (row[1] - eta_k[1]) * (row[1] - eta_k[1])
      + (row[2] - eta_k[2]) * (row[2] - eta_k[2])
      + ((row[3] - v_k) / v_k) * ((row[3] - v_k) / v_k));
}

