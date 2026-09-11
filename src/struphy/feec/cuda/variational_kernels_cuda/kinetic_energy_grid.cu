extern "C" __global__
void kinetic_energy_grid_cuda(
    const long long* span0, const long long* span1, const long long* span2,
    const double* basis0, const double* basis1, const double* basis2,
    const int n0, const int n1, const int n2,
    const int p0, const int p1, const int p2,
    const int start0, const int start1, const int start2,
    const double* u0, const double* u1, const double* u2,
    const double* v0, const double* v1, const double* v2,
    const int nc1, const int nc2,
    const double* metric, double* out,
    double* ug0, double* ug1, double* ug2,
    double* vg0, double* vg1, double* vg2)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n0 * n1 * n2;
    if (tid >= total) return;

    long long t = tid;
    const int i2 = t % n2; t /= n2;
    const int i1 = t % n1; const int i0 = t / n1;
    double us[3] = {0.0, 0.0, 0.0};
    double vs[3] = {0.0, 0.0, 0.0};

    for (int l0 = 0; l0 <= p0; ++l0) {
        const int c0 = (int)span0[i0] + l0 - start0;
        const double b0 = basis0[(long long)i0 * (p0 + 1) + l0];
        for (int l1 = 0; l1 <= p1; ++l1) {
            const int c1 = (int)span1[i1] + l1 - start1;
            const double b01 = b0 * basis1[(long long)i1 * (p1 + 1) + l1];
            for (int l2 = 0; l2 <= p2; ++l2) {
                const int c2 = (int)span2[i2] + l2 - start2;
                const double weight = b01 * basis2[(long long)i2 * (p2 + 1) + l2];
                const long long ci = ((long long)c0 * nc1 + c1) * nc2 + c2;
                us[0] += u0[ci] * weight; us[1] += u1[ci] * weight; us[2] += u2[ci] * weight;
                vs[0] += v0[ci] * weight; vs[1] += v1[ci] * weight; vs[2] += v2[ci] * weight;
            }
        }
    }
    double value = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            value += us[i] * metric[((long long)i * 3 + j) * total + tid] * vs[j];
    out[tid] = 0.5 * value;
    ug0[tid] = us[0]; ug1[tid] = us[1]; ug2[tid] = us[2];
    vg0[tid] = vs[0]; vg1[tid] = vs[1]; vg2[tid] = vs[2];
}
