__device__ void weighted_eta_v_dev(
    const double* row, int first_init_idx, int first_shift_idx,
    const double* alpha, double* eta, double* v_out)
{
    for (int k = 0; k < 3; k++) {
        const double eta_k = row[k] + row[first_shift_idx + k];
        const double eta_n = row[first_init_idx + k];
        double e = alpha[k] * eta_k + (1.0 - alpha[k]) * eta_n;
        double r = fmod(e, 1.0);
        if (r < 0.0) r += 1.0;
        eta[k] = r;
    }
    if (v_out) {
        const double v_k = row[3];
        const double v_n = row[first_init_idx + 3];
        *v_out = alpha[3] * v_k + (1.0 - alpha[3]) * v_n;
    }
}

extern "C" __global__
void grad_driftkinetic_hamiltonian_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx, const int mu_idx,
    const double* alpha,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* ef1, const int e1_n2, const int e1_n3,
    const double* ef2, const int e2_n2, const int e2_n3,
    const double* ef3, const int e3_n2, const int e3_n3,
    const int evaluate_e_field)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3];
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, 0);
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

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = grad_H[comps[j]];
}

extern "C" __global__
void bstar_parallel_3form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* bdb, const int bdb_n2, const int bdb_n3,
    const double* cub, const int cub_n2, const int cub_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3], v;
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, &v);

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta[0], eta[1], eta[2], params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bn2[MAXP+1], bn3[MAXP+1];
    double bd1[MAXP], bd2[MAXP], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    const double B_dot_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, bdb, bdb_n2, bdb_n3);
    double b_star_parallel = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, cub, cub_n2, cub_n3);

    b_star_parallel = (b_star_parallel * epsilon * v + B_dot_b) * det_df;

    row[column_nr] = b_star_parallel;
}

extern "C" __global__
void bstar_2form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b1, const int b1_n2, const int b1_n3,
    const double* b2, const int b2_n2, const int b2_n3,
    const double* b3, const int b3_n2, const int b3_n3,
    const double* cb1, const int c1_n2, const int c1_n3,
    const double* cb2, const int c2_n2, const int c2_n3,
    const double* cb3, const int c3_n2, const int c3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3], v;
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, &v);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double bb[3], b_star[3];
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        b1,b1_n2,b1_n3, b2,b2_n2,b2_n3, b3,b3_n2,b3_n3, bb);
    eval_2form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        cb1,c1_n2,c1_n3, cb2,c2_n2,c2_n3, cb3,c3_n2,c3_n3, b_star);

    for (int k = 0; k < 3; k++) b_star[k] = b_star[k] * epsilon * v + bb[k];

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = b_star[comps[j]];
}

extern "C" __global__
void unit_b_1form_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int n_comps, const int* comps,
    const int first_init_idx, const int first_shift_idx,
    const double* alpha,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* ub1, const int u1_n2, const int u1_n3,
    const double* ub2, const int u2_n2, const int u2_n3,
    const double* ub3, const int u3_n2, const int u3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta[3];
    weighted_eta_v_dev(row, first_init_idx, first_shift_idx, alpha, eta, 0);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta[2], span3, bn3, bd3);

    double unit_b1[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        ub1,u1_n2,u1_n3, ub2,u2_n2,u2_n3, ub3,u3_n2,u3_n3, unit_b1);

    for (int j = 0; j < n_comps; j++) row[column_nr + j] = unit_b1[comps[j]];
}

