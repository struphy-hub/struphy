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
                out += c[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3] * bn1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    return out;
}

// markers[ip, first_diagnostics_idx] = mu_p * |B_0(eta_p)|
extern "C" __global__
void eval_magnetic_background_energy_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* abs_B0, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, abs_B0, n2x, n3x);

    row[first_diagnostics_idx] = mu * abs_B;
}

// markers[ip, first_diagnostics_idx] = v_par^2 / 2 + mu_p * |B(eta_p)|
extern "C" __global__
void eval_energy_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_parallel = row[3];
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    row[first_diagnostics_idx] = 0.5 * v_parallel * v_parallel + mu * abs_B;
}

// markers[ip, idx_can_momentum] = shifted canonical toroidal momentum (5D)
extern "C" __global__
void eval_canonical_toroidal_moment_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx, const int idx_can_momentum,
    const double epsilon, const double B0, const double R0,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_para = row[3];
    const double mu = row[mu_idx];
    const double energy = row[first_diagnostics_idx];
    const double psi = row[idx_can_momentum];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    double out = psi - epsilon * B0 * R0 / abs_B * v_para;
    if (energy - mu * B0 > 0.0) {
        // sign(v_para) matches numpy.sign: 0 for exactly 0
        const double sgn = (v_para > 0.0) ? 1.0 : ((v_para < 0.0) ? -1.0 : 0.0);
        out += epsilon * sgn * sqrt(2.0 * (energy - mu * B0)) * R0;
    }
    row[idx_can_momentum] = out;
}

// markers[ip, first_diagnostics_idx + 5] = shifted canonical toroidal momentum (6D)
extern "C" __global__
void eval_canonical_toroidal_moment_6d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const double epsilon, const double B0, const double R0,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double energy = row[first_diagnostics_idx + 3];
    const double mu = row[first_diagnostics_idx + 4];
    const double psi = row[first_diagnostics_idx + 5];
    const double v_para = row[first_diagnostics_idx + 6];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    double out = psi - epsilon * B0 * R0 / abs_B * v_para;
    if (energy - mu * B0 > 0.0) {
        const double sgn = (v_para > 0.0) ? 1.0 : ((v_para < 0.0) ? -1.0 : 0.0);
        out += epsilon * sgn * sqrt(2.0 * (energy - mu * B0)) * R0;
    }
    row[first_diagnostics_idx + 5] = out;
}

// markers[ip, first_diagnostics_idx + 1] = v_perp^2 / (2 |B(eta_p)|)
extern "C" __global__
void eval_magnetic_moment_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_perp = row[4];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    row[first_diagnostics_idx + 1] = 0.5 * v_perp * v_perp / abs_B;
}

// markers[ip, first_diagnostics_idx] = mu_p * (|B_0| + PBb)(eta_p)
// NOTE: the CPU reference also evaluates the Jacobian DF(eta) here, but never
// uses the result, so it is not replicated.
extern "C" __global__
void eval_magnetic_energy_PBb_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* abs_B0, const int a_n2x, const int a_n3x,
    const double* PBb, const int b_n2x, const int b_n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    // eta = mod(markers[0:3], 1.0); fmod can return negative, match numpy mod
    double eta[3];
    for (int k = 0; k < 3; k++) {
        double e = fmod(row[k], 1.0);
        if (e < 0.0) e += 1.0;
        eta[k] = e;
    }

    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    b_splines_dev(tn1, p1, eta[0], span1, bn1);
    b_splines_dev(tn2, p2, eta[1], span2, bn2);
    b_splines_dev(tn3, p3, eta[2], span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, abs_B0, a_n2x, a_n3x);
    const double PB_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, PBb, b_n2x, b_n3x);

    row[first_diagnostics_idx] = mu * (abs_B + PB_b);
}

