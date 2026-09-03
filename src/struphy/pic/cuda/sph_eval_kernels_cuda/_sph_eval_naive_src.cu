extern "C" __global__
void naive_evaluation_flat_cuda(
    const double* markers,
    const int n_cols,
    const int n_markers,
    const double Np,
    const double* eta1,
    const double* eta2,
    const double* eta3,
    const int n_eval,
    const int* holes,
    const int periodic1,
    const int periodic2,
    const int periodic3,
    const int index,
    const int kernel_type,
    const double h1,
    const double h2,
    const double h3,
    double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_eval) return;

    double e1 = eta1[i], e2 = eta2[i], e3 = eta3[i];

    double acc = 0.0;
    for (int p = 0; p < n_markers; p++) {
        if (!holes[p]) {
            double r1 = distance_dev(e1, markers[(size_t)p * n_cols + 0], (bool)periodic1);
            double r2 = distance_dev(e2, markers[(size_t)p * n_cols + 1], (bool)periodic2);
            double r3 = distance_dev(e3, markers[(size_t)p * n_cols + 2], (bool)periodic3);
            acc += markers[(size_t)p * n_cols + index]
                 * smoothing_kernel_dev(kernel_type, r1, r2, r3, h1, h2, h3);
        }
    }
    out[i] = acc / Np;
}

extern "C" __global__
void naive_evaluation_meshgrid_cuda(
    const double* markers,
    const int n_cols,
    const int n_markers,
    const double Np,
    const double* eta1,
    const double* eta2,
    const double* eta3,
    const int n1_eval,
    const int n2_eval,
    const int n3_eval,
    const int* holes,
    const int periodic1,
    const int periodic2,
    const int periodic3,
    const int index,
    const int kernel_type,
    const double h1,
    const double h2,
    const double h3,
    double* out)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_total = (size_t)n1_eval * n2_eval * n3_eval;
    if (idx >= n_total) return;

    int i = idx / ((size_t)n2_eval * n3_eval);
    int rem = idx % ((size_t)n2_eval * n3_eval);
    int j = rem / n3_eval;
    int k = rem % n3_eval;

    double e1 = eta1[i], e2 = eta2[j], e3 = eta3[k];

    double acc = 0.0;
    for (int p = 0; p < n_markers; p++) {
        if (!holes[p]) {
            double r1 = distance_dev(e1, markers[(size_t)p * n_cols + 0], (bool)periodic1);
            double r2 = distance_dev(e2, markers[(size_t)p * n_cols + 1], (bool)periodic2);
            double r3 = distance_dev(e3, markers[(size_t)p * n_cols + 2], (bool)periodic3);
            acc += markers[(size_t)p * n_cols + index]
                 * smoothing_kernel_dev(kernel_type, r1, r2, r3, h1, h2, h3);
        }
    }
    out[idx] = acc / Np;
}

