extern "C" __global__
void h1vec_divdiv_assemble_cuda(
    const long long* spans1, const long long* spans2, const long long* spans3,
    const int ne1, const int ne2, const int ne3, const int p1, const int p2, const int p3,
    const int starts1, const int starts2, const int starts3, const int pads1, const int pads2, const int pads3,
    const double* b1, const double* b2, const double* b3, const int nder1, const int nder2, const int nder3,
    const int nq1, const int nq2, const int nq3, const double* weighted_rho,
    const int component_test, const int component_trial, double* data,
    const int nd2, const int nd3, const int nd4, const int nd5, const int nd6)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nloc = (long long)(p1 + 1) * (p2 + 1) * (p3 + 1);
    const long long total = (long long)ne1 * ne2 * ne3 * nloc * nloc;
    if (tid >= total) return;
    long long t = tid;
    const long long j = t % nloc; t /= nloc;
    const long long i = t % nloc; t /= nloc;
    const int iel3 = t % ne3; t /= ne3;
    const int iel2 = t % ne2; const int iel1 = t / ne2;
    const int il3 = i % (p3 + 1), il2 = (i / (p3 + 1)) % (p2 + 1), il1 = i / ((p2 + 1) * (p3 + 1));
    const int jl3 = j % (p3 + 1), jl2 = (j / (p3 + 1)) % (p2 + 1), jl1 = j / ((p2 + 1) * (p3 + 1));
    const int c1 = pads1 + (int)spans1[iel1] - p1 + il1 - starts1;
    const int c2 = pads2 + (int)spans2[iel2] - p2 + il2 - starts2;
    const int c3 = pads3 + (int)spans3[iel3] - p3 + il3 - starts3;
    const int o1 = pads1 + jl1 - il1, o2 = pads2 + jl2 - il2, o3 = pads3 + jl3 - il3;
    const int toi1 = component_test == 0, toi2 = component_test == 1, toi3 = component_test == 2;
    const int tro1 = component_trial == 0, tro2 = component_trial == 1, tro3 = component_trial == 2;
    double value = 0.0;
    for (int q1 = 0; q1 < nq1; ++q1) for (int q2 = 0; q2 < nq2; ++q2) for (int q3 = 0; q3 < nq3; ++q3) {
        const long long b1i = ((long long)(iel1 * (p1 + 1) + il1) * nder1) * nq1 + q1;
        const long long b2i = ((long long)(iel2 * (p2 + 1) + il2) * nder2) * nq2 + q2;
        const long long b3i = ((long long)(iel3 * (p3 + 1) + il3) * nder3) * nq3 + q3;
        const long long b1j = ((long long)(iel1 * (p1 + 1) + jl1) * nder1) * nq1 + q1;
        const long long b2j = ((long long)(iel2 * (p2 + 1) + jl2) * nder2) * nq2 + q2;
        const long long b3j = ((long long)(iel3 * (p3 + 1) + jl3) * nder3) * nq3 + q3;
        const double di = b1[b1i + toi1 * nq1] * b2[b2i + toi2 * nq2] * b3[b3i + toi3 * nq3];
        const double dj = b1[b1j + tro1 * nq1] * b2[b2j + tro2 * nq2] * b3[b3j + tro3 * nq3];
        const long long qidx = ((long long)(iel1 * nq1 + q1) * (ne2 * nq2) + iel2 * nq2 + q2) * (ne3 * nq3) + iel3 * nq3 + q3;
        value += weighted_rho[qidx] * di * dj;
    }
    const long long didx = (((((long long)c1 * nd2 + c2) * nd3 + c3) * nd4 + o1) * nd5 + o2) * nd6 + o3;
    atomicAdd(&data[didx], value);
}

