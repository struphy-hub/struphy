extern "C" __global__
void push_gc_bxEstar_discrete_gradient_1st_order_newton_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const double* phi, const int p_n2, const int p_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        const double eta_k_shifted = row[k] + row[first_shift_idx + k];
        eta_k[k] = row[k];
        eta_diff[k] = eta_k_shifted - row[first_init_idx + k];
    }
    const double v = row[3];
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double b_star_parallel = row[first_free_idx + 1];
    const double unit_b1[3] = {row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k1 = row[first_free_idx + 5];
    const double H_k12 = row[first_free_idx + 6];
    const double grad_H_1 = row[first_free_idx + 7];
    const double grad_H_12[2] = {row[first_free_idx + 8], row[first_free_idx + 9]};

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_k[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_k[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_k[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_k[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_k[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_k[2], span3, bn3, bd3);

    double phi_val = 0.0;
    if (evaluate_e_field) {
        phi_val = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi, p_n2, p_n3);
    }
    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    const double H_k = epsilon * v * v / 2.0 + epsilon * mu * B_dot_b + phi_val;

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

    double grad_I[3];
    grad_I[0] = (eta_diff[0] == 0.0) ? grad_H[0] : (H_k1 - H_n) / eta_diff[0];
    grad_I[1] = (eta_diff[1] == 0.0) ? grad_H[1] : (H_k12 - H_k1) / eta_diff[1];
    grad_I[2] = (eta_diff[2] == 0.0) ? grad_H[2] : (H_k - H_k12) / eta_diff[2];

    double bcross_mat[9] = {
        0.0, -unit_b1[2], unit_b1[1],
        unit_b1[2], 0.0, -unit_b1[0],
        -unit_b1[1], unit_b1[0], 0.0};
    for (int k = 0; k < 9; k++) bcross_mat[k] /= b_star_parallel;

    double func[3];
    matvec_dev(bcross_mat, grad_I, func);
    for (int k = 0; k < 3; k++) func[k] = eta_diff[k] - dt * func[k];

    double Ddg[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (eta_diff[0] != 0.0) Ddg[0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / (eta_diff[0] * eta_diff[0]);
    if (eta_diff[1] != 0.0) {
        Ddg[4] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / (eta_diff[1] * eta_diff[1]);
        Ddg[3] = (grad_H_12[0] - grad_H_1) / eta_diff[1];
    }
    if (eta_diff[2] != 0.0) {
        Ddg[8] = (grad_H[2] * eta_diff[2] - (H_k - H_k12)) / (eta_diff[2] * eta_diff[2]);
        Ddg[6] = (grad_H[0] - grad_H_12[0]) / eta_diff[2];
        Ddg[7] = (grad_H[1] - grad_H_12[1]) / eta_diff[2];
    }

    double Dfunc[9];
    matmat_dev(bcross_mat, Ddg, Dfunc);
    for (int k = 0; k < 9; k++) Dfunc[k] *= -dt;
    Dfunc[0] += 1.0; Dfunc[4] += 1.0; Dfunc[8] += 1.0;

    double Dfunc_inv[9], k_vec[3];
    matrix_inv_dev(Dfunc, Dfunc_inv);
    matvec_dev(Dfunc_inv, func, k_vec);

    row[0] -= k_vec[0];
    row[1] -= k_vec[1];
    row[2] -= k_vec[2];

    row[residual_idx] = sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
}

extern "C" __global__
void push_gc_Bstar_discrete_gradient_1st_order_newton_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int first_shift_idx,
    const int residual_idx, const int first_free_idx, const int mu_idx,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const double* phi, const int p_n2, const int p_n3,
    const int evaluate_e_field, const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0) return;

    double eta_k[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        const double eta_k_shifted = row[k] + row[first_shift_idx + k];
        eta_k[k] = row[k];
        eta_diff[k] = eta_k_shifted - row[first_init_idx + k];
    }
    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v_diff = v_k - v_n;
    const double mu = row[mu_idx];

    const double H_n = row[first_free_idx];
    const double b_star_parallel = epsilon * row[first_free_idx + 1];
    const double b_star[3] = {row[first_free_idx + 2], row[first_free_idx + 3], row[first_free_idx + 4]};
    const double H_k1 = row[first_free_idx + 5];
    const double H_k12 = row[first_free_idx + 6];
    const double grad_H_1 = row[first_free_idx + 7];
    const double grad_H_12[2] = {row[first_free_idx + 8], row[first_free_idx + 9]};

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_k[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_k[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_k[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_k[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_k[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_k[2], span3, bn3, bd3);

    double phi_val = 0.0;
    if (evaluate_e_field) {
        phi_val = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi, p_n2, p_n3);
    }
    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    const double H_k = epsilon * v_k * v_k / 2.0 + epsilon * mu * B_dot_b + phi_val;
    const double H_k123 = epsilon * v_n * v_n / 2.0 + epsilon * mu * B_dot_b + phi_val;

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

    const double grad_H_v = epsilon * v_k;

    double grad_I[3];
    grad_I[0] = (eta_diff[0] == 0.0) ? grad_H[0] : (H_k1 - H_n) / eta_diff[0];
    grad_I[1] = (eta_diff[1] == 0.0) ? grad_H[1] : (H_k12 - H_k1) / eta_diff[1];
    grad_I[2] = (eta_diff[2] == 0.0) ? grad_H[2] : (H_k123 - H_k12) / eta_diff[2];
    const double grad_I_v = (v_diff == 0.0) ? grad_H_v : (H_k - H_k123) / v_diff;

    double J_vec[3];
    for (int k = 0; k < 3; k++) J_vec[k] = b_star[k] / b_star_parallel;

    double func[3];
    for (int k = 0; k < 3; k++) func[k] = eta_diff[k] - dt * (J_vec[k] * grad_I_v);
    double func_v = v_diff + dt * dot3_dev(J_vec, grad_I);

    double Ddg[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (eta_diff[0] != 0.0) Ddg[0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / (eta_diff[0] * eta_diff[0]);
    if (eta_diff[1] != 0.0) {
        Ddg[4] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / (eta_diff[1] * eta_diff[1]);
        Ddg[3] = (grad_H_12[0] - grad_H_1) / eta_diff[1];
    }
    if (eta_diff[2] != 0.0) {
        Ddg[8] = (grad_H[2] * eta_diff[2] - (H_k123 - H_k12)) / (eta_diff[2] * eta_diff[2]);
        Ddg[6] = (grad_H[0] - grad_H_12[0]) / eta_diff[2];
        Ddg[7] = (grad_H[1] - grad_H_12[1]) / eta_diff[2];
    }
    const double Ddg_v = (v_diff == 0.0) ? 0.0 : (grad_H_v * v_diff - (H_k - H_k123)) / (v_diff * v_diff);

    // DF = [[I, B], [C^T, 1]], B = -dt*Ddg_v*J_vec, C = dt*Ddg^T @ J_vec
    double Bv[3], Cv[3];
    for (int k = 0; k < 3; k++) Bv[k] = -dt * Ddg_v * J_vec[k];
    double DdgT[9] = {Ddg[0], Ddg[3], Ddg[6], Ddg[1], Ddg[4], Ddg[7], Ddg[2], Ddg[5], Ddg[8]};
    matvec_dev(DdgT, J_vec, Cv);
    for (int k = 0; k < 3; k++) Cv[k] *= dt;

    const double schur = 1.0 - dot3_dev(Cv, Bv);

    double A_inv[9];
    A_inv[0] = Bv[0]*Cv[0]; A_inv[1] = Bv[0]*Cv[1]; A_inv[2] = Bv[0]*Cv[2];
    A_inv[3] = Bv[1]*Cv[0]; A_inv[4] = Bv[1]*Cv[1]; A_inv[5] = Bv[1]*Cv[2];
    A_inv[6] = Bv[2]*Cv[0]; A_inv[7] = Bv[2]*Cv[1]; A_inv[8] = Bv[2]*Cv[2];
    for (int k = 0; k < 9; k++) A_inv[k] /= schur;
    A_inv[0] += 1.0; A_inv[4] += 1.0; A_inv[8] += 1.0;

    double Binv[3], Cinv[3];
    for (int k = 0; k < 3; k++) { Binv[k] = -Bv[k] / schur; Cinv[k] = -Cv[k] / schur; }

    double k_vec[3];
    matvec_dev(A_inv, func, k_vec);
    for (int k = 0; k < 3; k++) k_vec[k] += Binv[k] * func_v;
    double k_v = dot3_dev(Cinv, func) + func_v / schur;

    row[0] -= k_vec[0];
    row[1] -= k_vec[1];
    row[2] -= k_vec[2];
    row[3] -= k_v;

    row[residual_idx] = sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2] + (k_v/v_k)*(k_v/v_k));
}

