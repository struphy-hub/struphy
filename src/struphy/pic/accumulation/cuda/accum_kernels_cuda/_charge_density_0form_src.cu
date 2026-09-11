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

// Only the N-spline values (bn) are needed for an H^1/0-form fill; D-spline
// values are computed alongside (same recursion as
// pusher_kernels_cuda.py's b_d_splines_dev) and simply unused.
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

extern "C" __global__
void charge_density_0form_cuda(
    const double* markers,
    const int n_cols,
    const int n_markers,
    const int weight_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    double* vec, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    const double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double filling = row[weight_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bn1[il1] * filling;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bn2[il2];
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bn3[il3];
                atomicAdd(&vec[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3], b3);
            }
        }
    }
}

