#define PI 3.14159265358979323846

__device__ double distance_dev(double x, double y, bool periodic)
{
    double d = x - y;
    if (periodic) {
        while (d > 0.5) d -= 1.0;
        while (d < -0.5) d += 1.0;
    }
    return d;
}

// --- uni-variate kernels (struphy.pic.sph_smoothing_kernels) ---

__device__ double trigonometric_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return 0.785398163397448 / h * cos(x / h * PI / 2.0);
    return 0.0;
}

__device__ double grad_trigonometric_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return -(1.2337005501361697 / (h * h)) * sin(x / h * PI / 2.0);
    return 0.0;
}

__device__ double gaussian_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return 1.0 / (sqrt(PI) * h / 3.0) * exp(-(x * x) / ((h / 3.0) * (h / 3.0)));
    return 0.0;
}

__device__ double grad_gaussian_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return -54.0 * x / (h * h * h * sqrt(PI)) * exp(-(x * x) / ((h / 3.0) * (h / 3.0)));
    return 0.0;
}

__device__ double linear_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return (1.0 - fabs(x / h)) / h;
    return 0.0;
}

__device__ double grad_linear_uni(double x, double h)
{
    if (fabs(x / h) <= 1.0) return (x > 0.0) ? -(1.0 / (h * h)) : (1.0 / (h * h));
    return 0.0;
}

// --- kernel_type dispatch (struphy.pic.sph_smoothing_kernels.smoothing_kernel) ---

__device__ double smoothing_kernel_dev(
    int kernel_type,
    double r1, double r2, double r3,
    double h1, double h2, double h3)
{
    switch (kernel_type) {
        // 1d
        case 100: return trigonometric_uni(r1, h1);
        case 101: return grad_trigonometric_uni(r1, h1);
        case 110: return gaussian_uni(r1, h1);
        case 111: return grad_gaussian_uni(r1, h1);
        case 120: return linear_uni(r1, h1);
        case 121: return grad_linear_uni(r1, h1);

        // 2d (tensor products)
        case 340: return trigonometric_uni(r1, h1) * trigonometric_uni(r2, h2);
        case 341: return grad_trigonometric_uni(r1, h1) * trigonometric_uni(r2, h2);
        case 342: return trigonometric_uni(r1, h1) * grad_trigonometric_uni(r2, h2);
        case 350: return gaussian_uni(r1, h1) * gaussian_uni(r2, h2);
        case 351: return grad_gaussian_uni(r1, h1) * gaussian_uni(r2, h2);
        case 352: return gaussian_uni(r1, h1) * grad_gaussian_uni(r2, h2);
        case 360: return linear_uni(r1, h1) * linear_uni(r2, h2);
        case 361: return grad_linear_uni(r1, h1) * linear_uni(r2, h2);
        case 362: return linear_uni(r1, h1) * grad_linear_uni(r2, h2);

        // 3d (tensor products)
        case 670: return trigonometric_uni(r1, h1) * trigonometric_uni(r2, h2) * trigonometric_uni(r3, h3);
        case 671: return grad_trigonometric_uni(r1, h1) * trigonometric_uni(r2, h2) * trigonometric_uni(r3, h3);
        case 672: return trigonometric_uni(r1, h1) * grad_trigonometric_uni(r2, h2) * trigonometric_uni(r3, h3);
        case 673: return trigonometric_uni(r1, h1) * trigonometric_uni(r2, h2) * grad_trigonometric_uni(r3, h3);
        case 680: return gaussian_uni(r1, h1) * gaussian_uni(r2, h2) * gaussian_uni(r3, h3);
        case 681: return grad_gaussian_uni(r1, h1) * gaussian_uni(r2, h2) * gaussian_uni(r3, h3);
        case 682: return gaussian_uni(r1, h1) * grad_gaussian_uni(r2, h2) * gaussian_uni(r3, h3);
        case 683: return gaussian_uni(r1, h1) * gaussian_uni(r2, h2) * grad_gaussian_uni(r3, h3);
        case 700: return linear_uni(r1, h1) * linear_uni(r2, h2) * linear_uni(r3, h3);
        case 701: return grad_linear_uni(r1, h1) * linear_uni(r2, h2) * linear_uni(r3, h3);
        case 702: return linear_uni(r1, h1) * grad_linear_uni(r2, h2) * linear_uni(r3, h3);
        case 703: return linear_uni(r1, h1) * linear_uni(r2, h2) * grad_linear_uni(r3, h3);

        // 3d, radially symmetric (linear_isotropic_3d and its gradient)
        case 690: {
            double r = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
            double h = h1;
            if (r / h > 1.0) return 0.0;
            return (1.0 - r / h) / (1.0471975512 * h * h * h);
        }
        case 691: {
            double r = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
            double h = h1;
            if (r / h > 1.0) return 0.0;
            if (r == 0.0) return -1.0 / h / (1.0471975512 * h * h * h);
            return -r1 / (r * h) / (1.0471975512 * h * h * h);
        }
        case 692: {
            double r = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
            double h = h1;
            if (r / h > 1.0) return 0.0;
            if (r == 0.0) return -1.0 / h / (1.0471975512 * h * h * h);
            return -r2 / (r * h) / (1.0471975512 * h * h * h);
        }
        case 693: {
            double r = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
            double h = h1;
            if (r / h > 1.0) return 0.0;
            if (r == 0.0) return -1.0 / h / (1.0471975512 * h * h * h);
            return -r3 / (r * h) / (1.0471975512 * h * h * h);
        }
    }
    return 0.0;
}

// --- box lookup (struphy.pic.sorting_kernels.find_box / flatten_index) ---

__device__ int find_box_dev(
    double eta1, double eta2, double eta3,
    int nx, int ny, int nz,
    const double* domain_array)
{
    if (eta1 == domain_array[0]) eta1 += 1e-8;
    if (eta2 == domain_array[3]) eta2 += 1e-8;
    if (eta3 == domain_array[6]) eta3 += 1e-8;
    if (eta1 == domain_array[1]) eta1 -= 1e-8;
    if (eta2 == domain_array[4]) eta2 -= 1e-8;
    if (eta3 == domain_array[7]) eta3 -= 1e-8;

    double x_l = domain_array[0] - (domain_array[1] - domain_array[0]) / nx;
    double x_r = domain_array[1] + (domain_array[1] - domain_array[0]) / nx;
    double y_l = domain_array[3] - (domain_array[4] - domain_array[3]) / ny;
    double y_r = domain_array[4] + (domain_array[4] - domain_array[3]) / ny;
    double z_l = domain_array[6] - (domain_array[7] - domain_array[6]) / nz;
    double z_r = domain_array[7] + (domain_array[7] - domain_array[6]) / nz;

    if (eta1 < x_l || eta1 > x_r || eta2 < y_l || eta2 > y_r || eta3 < z_l || eta3 > z_r)
        return -1;

    int n1 = (int)floor((eta1 - x_l) / (x_r - x_l) * (nx + 2));
    int n2 = (int)floor((eta2 - y_l) / (y_r - y_l) * (ny + 2));
    int n3 = (int)floor((eta3 - z_l) / (z_r - z_l) * (nz + 2));

    // flatten_index, fortran_ordering (the struphy default)
    return n1 + n2 * (nx + 2) + n3 * (nx + 2) * (ny + 2);
}

// --- entry point (struphy.pic.sph_eval_kernels.box_based_evaluation_flat) ---

extern "C" __global__
void box_based_evaluation_flat_cuda(
    const double* markers,
    const int n_cols,
    const double* eta1,
    const double* eta2,
    const double* eta3,
    const int n_eval,
    const int nx,
    const int ny,
    const int nz,
    const double* domain_array,
    const int* boxes,
    const int n_box_cols,
    const int* neighbours,
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

    int loc_box = find_box_dev(e1, e2, e3, nx, ny, nz, domain_array);
    if (loc_box == -1) {
        out[i] = 0.0;
        return;
    }

    double acc = 0.0;
    for (int neigh = 0; neigh < 27; neigh++) {
        int box_to_search = neighbours[loc_box * 27 + neigh];
        int c = 0;
        while (boxes[(size_t)box_to_search * n_box_cols + c] != -1) {
            int p = boxes[(size_t)box_to_search * n_box_cols + c];
            c++;
            if (!holes[p]) {
                double r1 = distance_dev(e1, markers[(size_t)p * n_cols + 0], (bool)periodic1);
                double r2 = distance_dev(e2, markers[(size_t)p * n_cols + 1], (bool)periodic2);
                double r3 = distance_dev(e3, markers[(size_t)p * n_cols + 2], (bool)periodic3);
                acc += markers[(size_t)p * n_cols + index]
                     * smoothing_kernel_dev(kernel_type, r1, r2, r3, h1, h2, h3);
            }
        }
    }
    out[i] = acc;
}

// --- entry point (struphy.pic.sph_eval_kernels.box_based_evaluation_meshgrid) ---
//
// eta1/eta2/eta3 are the 3 distinct 1-D axis vectors of the meshgrid (the
// Pyccel kernel this ports only ever reads eta1[i,0,0]/eta2[0,j,0]/eta3[0,0,k],
// never the broadcast values, so the Python wrapper passes just the axes --
// no reason to transfer the O(n1*n2*n3) redundant meshgrid). One CUDA thread
// per (i, j, k) evaluation point, flattened to match out's C-order layout.

extern "C" __global__
void box_based_evaluation_meshgrid_cuda(
    const double* markers,
    const int n_cols,
    const double* eta1,
    const double* eta2,
    const double* eta3,
    const int n1_eval,
    const int n2_eval,
    const int n3_eval,
    const int nx,
    const int ny,
    const int nz,
    const double* domain_array,
    const int* boxes,
    const int n_box_cols,
    const int* neighbours,
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

    out[idx] = 0.0;

    double e1 = eta1[i];
    if (e1 < domain_array[0] || (e1 >= domain_array[1] && e1 != 1.0)) return;

    double e2 = eta2[j];
    if (e2 < domain_array[3] || (e2 >= domain_array[4] && e2 != 1.0)) return;

    double e3 = eta3[k];
    if (e3 < domain_array[6] || (e3 >= domain_array[7] && e3 != 1.0)) return;

    int loc_box = find_box_dev(e1, e2, e3, nx, ny, nz, domain_array);
    if (loc_box == -1) return;

    double acc = 0.0;
    for (int neigh = 0; neigh < 27; neigh++) {
        int box_to_search = neighbours[loc_box * 27 + neigh];
        int c = 0;
        while (boxes[(size_t)box_to_search * n_box_cols + c] != -1) {
            int p = boxes[(size_t)box_to_search * n_box_cols + c];
            c++;
            if (!holes[p]) {
                double r1 = distance_dev(e1, markers[(size_t)p * n_cols + 0], (bool)periodic1);
                double r2 = distance_dev(e2, markers[(size_t)p * n_cols + 1], (bool)periodic2);
                double r3 = distance_dev(e3, markers[(size_t)p * n_cols + 2], (bool)periodic3);
                acc += markers[(size_t)p * n_cols + index]
                     * smoothing_kernel_dev(kernel_type, r1, r2, r3, h1, h2, h3);
            }
        }
    }
    out[idx] = acc;
}

