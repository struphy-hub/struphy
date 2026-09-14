#define MAXP 8

__device__ int find_span_dev(const double* t, int p, int len_t, double eta)
{
    int low = p;
    int high = len_t - 1 - p;

    if (eta <= t[low]) return low;
    if (eta >= t[high]) return high - 1;

    int span = (low + high) / 2;
    while (eta < t[span] || eta >= t[span + 1]) {
        if (eta < t[span]) high = span;
        else low = span;
        span = (low + high) / 2;
    }
    return span;
}

__device__ void b_splines_dev(const double* t, int p, double eta, int span, double* bn)
{
    double left[MAXP];
    double right[MAXP];

    for (int i = 0; i <= p; i++) bn[i] = 0.0;
    bn[0] = 1.0;

    for (int j = 0; j < p; j++) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.0;
        for (int r = 0; r <= j; r++) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

__device__ double eval_0form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bn2, const double* bn3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c, int n2x, int n3x)
{
    double out = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out += c[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3]
                     * bn1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    return out;
}

__device__ double mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void driftkinetic_hamiltonian_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr,
    const int first_init_idx, const int first_shift_idx, const int mu_idx,
    const double a0, const double a1, const double a2, const double a3,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* B_dot_b, const int b_n2, const int b_n3,
    const double* phi_c, const int p_n2, const int p_n3,
    const int evaluate_e_field)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double alpha[3] = {a0, a1, a2};
    double eta[3];
    for (int i = 0; i < 3; i++) {
        const double eta_k = row[i] + row[first_shift_idx + i];
        const double eta_n = row[first_init_idx + i];
        eta[i] = mod1_dev(alpha[i] * eta_k + (1.0 - alpha[i]) * eta_n);
    }

    const double v_k = row[3];
    const double v_n = row[first_init_idx + 3];
    const double v = a3 * v_k + (1.0 - a3) * v_n;
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    b_splines_dev(tn1, p1, eta[0], span1, bn1);
    b_splines_dev(tn2, p2, eta[1], span2, bn2);
    b_splines_dev(tn3, p3, eta[2], span3, bn3);

    double phi = 0.0;
    if (evaluate_e_field) {
        phi = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
            start0, start1, start2, phi_c, p_n2, p_n3);
    }

    const double bdb = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, B_dot_b, b_n2, b_n3);

    row[column_nr] = epsilon * v * v / 2.0 + epsilon * mu * bdb + phi;
}

