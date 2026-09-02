extern "C" __device__ long long flatten_index_dev(
    long long n1, long long n2, long long n3,
    long long nx, long long ny, long long nz)
{
    // fortran_ordering (the struphy default)
    return n1 + n2 * (nx + 2) + n3 * (nx + 2) * (ny + 2);
}

extern "C" __device__ long long find_box_dev(
    double eta1, double eta2, double eta3,
    long long nx, long long ny, long long nz,
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

    long long n1 = (long long)floor((eta1 - x_l) / (x_r - x_l) * (nx + 2));
    long long n2 = (long long)floor((eta2 - y_l) / (y_r - y_l) * (ny + 2));
    long long n3 = (long long)floor((eta3 - z_l) / (z_r - z_l) * (nz + 2));

    return flatten_index_dev(n1, n2, n3, nx, ny, nz);
}

extern "C" __global__
void assign_box_to_each_particle_cuda(
    const double* eta,  // AoS, row p at eta[3*p : 3*p+3]
    const int* holes,
    const long long n_mks,
    const long long nx,
    const long long ny,
    const long long nz,
    const double* domain_array,
    double* box_out)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_mks) return;

    long long n_boxes_total = (nx + 2) * (ny + 2) * (nz + 2);
    long long n_box;

    if (holes[p]) {
        n_box = n_boxes_total;
    } else {
        long long a = find_box_dev(eta[3 * p], eta[3 * p + 1], eta[3 * p + 2], nx, ny, nz, domain_array);
        n_box = (a >= n_boxes_total || a < 0) ? n_boxes_total : a;
    }

    box_out[p] = (double) n_box;
}

extern "C" __global__
void assign_particles_to_boxes_cuda(
    const double* box_id,
    const int* holes,
    const long long n_mks,
    int* boxes,
    int* next_index,
    const long long box_cols)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_mks) return;
    if (holes[p]) return;

    int a = (int) box_id[p];
    int slot = atomicAdd(&next_index[a], 1);
    if (slot < box_cols) {
        boxes[(long long) a * box_cols + slot] = (int) p;
    }
}

