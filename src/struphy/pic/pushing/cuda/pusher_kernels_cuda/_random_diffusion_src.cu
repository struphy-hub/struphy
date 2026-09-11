extern "C" __global__
void push_random_diffusion_stage(
    double* markers,
    const int n_cols,
    const int n_markers,
    const double* noise,
    const double scale)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    row[0] += scale * noise[3*ip + 0];
    row[1] += scale * noise[3*ip + 1];
    row[2] += scale * noise[3*ip + 2];
}

