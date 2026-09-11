extern "C" __global__
void mass_3d_assemble_cuda(
    const long long* spans1, const long long* spans2, const long long* spans3,
    const int ne1, const int ne2, const int ne3,
    const int pi1, const int pi2, const int pi3, const int pj1, const int pj2, const int pj3,
    const int starts1, const int starts2, const int starts3, const int pads1, const int pads2, const int pads3,
    const double* w1, const double* w2, const double* w3, const int nq1, const int nq2, const int nq3,
    const double* bi1, const double* bi2, const double* bi3, const double* bj1, const double* bj2, const double* bj3,
    const int ni_der1, const int ni_der2, const int ni_der3, const int nj_der1, const int nj_der2, const int nj_der3,
    const double* mat_fun, double* data, const int nd2, const int nd3, const int nd4, const int nd5, const int nd6)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long ni = (long long)(pi1 + 1) * (pi2 + 1) * (pi3 + 1);
    const long long nj = (long long)(pj1 + 1) * (pj2 + 1) * (pj3 + 1);
    const long long total = (long long)ne1 * ne2 * ne3 * ni * nj;
    if (tid >= total) return;
    long long t = tid;
    const long long j = t % nj; t /= nj;
    const long long i = t % ni; t /= ni;
    const int iel3 = t % ne3; t /= ne3;
    const int iel2 = t % ne2; const int iel1 = t / ne2;
    const int il3 = i % (pi3 + 1); const int il2 = (i / (pi3 + 1)) % (pi2 + 1); const int il1 = i / ((pi2 + 1) * (pi3 + 1));
    const int jl3 = j % (pj3 + 1); const int jl2 = (j / (pj3 + 1)) % (pj2 + 1); const int jl1 = j / ((pj2 + 1) * (pj3 + 1));
    const int c1 = pads1 + (int)spans1[iel1] - pi1 + il1 - starts1;
    const int c2 = pads2 + (int)spans2[iel2] - pi2 + il2 - starts2;
    const int c3 = pads3 + (int)spans3[iel3] - pi3 + il3 - starts3;
    const int o1 = pads1 + jl1 - il1, o2 = pads2 + jl2 - il2, o3 = pads3 + jl3 - il3;
    double value = 0.0;
    for (int q1 = 0; q1 < nq1; ++q1) {
        const double wi1 = w1[iel1 * nq1 + q1];
        const double ai1 = bi1[((long long)(iel1 * (pi1 + 1) + il1) * ni_der1) * nq1 + q1];
        const double aj1 = bj1[((long long)(iel1 * (pj1 + 1) + jl1) * nj_der1) * nq1 + q1];
        for (int q2 = 0; q2 < nq2; ++q2) {
            const double wi2 = wi1 * w2[iel2 * nq2 + q2];
            const double ai2 = ai1 * bi2[((long long)(iel2 * (pi2 + 1) + il2) * ni_der2) * nq2 + q2];
            const double aj2 = aj1 * bj2[((long long)(iel2 * (pj2 + 1) + jl2) * nj_der2) * nq2 + q2];
            for (int q3 = 0; q3 < nq3; ++q3) {
                const long long qidx = ((long long)(iel1 * nq1 + q1) * (ne2 * nq2) + iel2 * nq2 + q2) * (ne3 * nq3) + iel3 * nq3 + q3;
                const double ai3 = ai2 * bi3[((long long)(iel3 * (pi3 + 1) + il3) * ni_der3) * nq3 + q3];
                const double aj3 = aj2 * bj3[((long long)(iel3 * (pj3 + 1) + jl3) * nj_der3) * nq3 + q3];
                value += wi2 * w3[iel3 * nq3 + q3] * mat_fun[qidx] * ai3 * aj3;
            }
        }
    }
    const long long didx = (((((long long)c1 * nd2 + c2) * nd3 + c3) * nd4 + o1) * nd5 + o2) * nd6 + o3;
    atomicAdd(&data[didx], value);
}

