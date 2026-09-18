extern "C" __global__
void push_eta_rk_periodic(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int first_shift_idx,
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

    if (row[first_init_idx] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double kx = sx * row[3];
    const double ky = sy * row[4];
    const double kz = sz * row[5];

    row[first_free_idx + 0] += dt_b * kx;
    row[first_free_idx + 1] += dt_b * ky;
    row[first_free_idx + 2] += dt_b * kz;

    double e0 = row[first_init_idx + 0] + dt_a * kx + last * row[first_free_idx + 0];
    double e1 = row[first_init_idx + 1] + dt_a * ky + last * row[first_free_idx + 1];
    double e2 = row[first_init_idx + 2] + dt_a * kz + last * row[first_free_idx + 2];

    // periodic wrap + shift bookkeeping, matching the periodic branch of
    // Particles.apply_kinetic_bc (Python's a % 1.0 is always in [0, 1))
    double shift0 = 0.0, shift1 = 0.0, shift2 = 0.0;

    if (e0 > 1.0) { e0 = fmod(e0, 1.0); shift0 = 1.0; }
    else if (e0 < 0.0) { e0 = fmod(e0, 1.0); if (e0 < 0.0) e0 += 1.0; shift0 = -1.0; }

    if (e1 > 1.0) { e1 = fmod(e1, 1.0); shift1 = 1.0; }
    else if (e1 < 0.0) { e1 = fmod(e1, 1.0); if (e1 < 0.0) e1 += 1.0; shift1 = -1.0; }

    if (e2 > 1.0) { e2 = fmod(e2, 1.0); shift2 = 1.0; }
    else if (e2 < 0.0) { e2 = fmod(e2, 1.0); if (e2 < 0.0) e2 += 1.0; shift2 = -1.0; }

    row[0] = e0;
    row[1] = e1;
    row[2] = e2;
    row[first_shift_idx + 0] = shift0;
    row[first_shift_idx + 1] = shift1;
    row[first_shift_idx + 2] = shift2;
}

