__device__ double gradb_ediff_mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void eval_gradB_ediff_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int mu_idx, const int idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* pb1, const int p1_n2, const int p1_n3,
    const double* pb2, const int p2_n2, const int p2_n3,
    const double* pb3, const int p3_n2, const int p3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_mid[k] = gradb_ediff_mod1_dev((row[k] + row[first_init_idx + k]) / 2.0);
        eta_diff[k] = row[k] - row[first_init_idx + k];
    }
    const double mu = row[mu_idx];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double gradB[3], grad_PB_b[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,g1_n2,g1_n3, gb2,g2_n2,g2_n3, gb3,g3_n2,g3_n3, gradB);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        pb1,p1_n2,p1_n3, pb2,p2_n2,p2_n3, pb3,p3_n2,p3_n3, grad_PB_b);

    double tmp[3];
    for (int k = 0; k < 3; k++) tmp[k] = gradB[k] + grad_PB_b[k];

    row[idx] = mu * dot3_dev(eta_diff, tmp);
}

