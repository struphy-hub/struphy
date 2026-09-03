extern "C" __global__
void push_eta_stage_cuboid(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const double sx,
    const double sy,
    const double sz,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;

    // skip holes and ghost/boundary particles, matching push_eta_stage
    if (row[first_init_idx] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double kx = sx * row[3];
    const double ky = sy * row[4];
    const double kz = sz * row[5];

    // accumulate for the last stage (must happen before the position update,
    // which reads the just-updated accumulator)
    row[first_free_idx + 0] += dt_b * kx;
    row[first_free_idx + 1] += dt_b * ky;
    row[first_free_idx + 2] += dt_b * kz;

    row[0] = row[first_init_idx + 0] + dt_a * kx + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * ky + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * kz + last * row[first_free_idx + 2];
}

