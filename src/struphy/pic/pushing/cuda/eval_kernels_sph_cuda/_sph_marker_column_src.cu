extern "C" __global__
void sph_pressure_coeffs_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);

    const double weight = row[weight_idx];
    const double gamma = 5.0 / 3.0;

    row[column_nr] = n_at_eta;
    row[column_nr + 1] = weight / n_at_eta;
    row[column_nr + 2] = weight * pow(n_at_eta, gamma - 2.0);
}

extern "C" __global__
void sph_mean_velocity_coeffs_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);

    const double weight = row[weight_idx];
    const double scale = weight / n_at_eta;

    row[column_nr + 0] = scale * row[3];
    row[column_nr + 1] = scale * row[4];
    row[column_nr + 2] = scale * row[5];
}

extern "C" __global__
void sph_viscosity_tensor_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int column_nr, const int weight_idx, const int first_free_idx,
    const int* valid_mks,
    const int* boxes, const int n_box_cols,
    const int* neighbours, const int* holes,
    const int periodic1, const int periodic2, const int periodic3,
    const int kernel_type,
    const double h1, const double h2, const double h3,
    const double mu)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;
    if (!valid_mks[ip]) return;

    double* row = markers + (size_t)ip * n_cols;
    const double e1 = row[0], e2 = row[1], e3 = row[2];
    const int loc_box = (int)row[n_cols - 2];

    const double n_at_eta = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
        neighbours, holes, periodic1, periodic2, periodic3, weight_idx, kernel_type, h1, h2, h3);
    const double weight = row[weight_idx];

    double grad_v[3][3];
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            grad_v[j][k] = box_based_kernel_dev(markers, n_cols, e1, e2, e3, loc_box, boxes, n_box_cols,
                neighbours, holes, periodic1, periodic2, periodic3,
                first_free_idx + j, kernel_type + 1 + k, h1, h2, h3);
        }
    }

    double d_dev[3][3];
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++)
            d_dev[j][k] = 0.5 * (grad_v[j][k] + grad_v[k][j]);

    const double mean_trace = (d_dev[0][0] + d_dev[1][1] + d_dev[2][2]) / 3.0;
    d_dev[0][0] -= mean_trace;
    d_dev[1][1] -= mean_trace;
    d_dev[2][2] -= mean_trace;

    const double scale = -2.0 * mu * (weight / n_at_eta);
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            row[column_nr + 3 * j + k] = d_dev[j][k] * scale;
        }
    }
}

