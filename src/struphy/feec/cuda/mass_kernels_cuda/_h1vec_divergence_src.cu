extern "C" __global__
void h1vec_divergence_eval_cuda(
    const long long* spans1, const long long* spans2, const long long* spans3,
    const int ne1, const int ne2, const int ne3,
    const int p1, const int p2, const int p3,
    const int starts1, const int starts2, const int starts3,
    const int pads1, const int pads2, const int pads3,
    const double* b1, const double* b2, const double* b3,
    const int nder1, const int nder2, const int nder3,
    const int nq1, const int nq2, const int nq3,
    const double* dlogj1, const double* dlogj2, const double* dlogj3,
    const int component, const double* coeffs,
    const int nc2, const int nc3, double* values)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long totalq3 = (long long)ne3 * nq3;
    const long long totalq2 = (long long)ne2 * nq2;
    const long long nvalues = (long long)ne1 * nq1 * totalq2 * totalq3;
    if (tid >= nvalues) return;

    const int iq3 = tid % totalq3;
    const long long t12 = tid / totalq3;
    const int iq2 = t12 % totalq2;
    const int iq1 = t12 / totalq2;
    const int iel1 = iq1 / nq1, q1 = iq1 % nq1;
    const int iel2 = iq2 / nq2, q2 = iq2 % nq2;
    const int iel3 = iq3 / nq3, q3 = iq3 % nq3;
    const double dlog = component == 0 ? dlogj1[tid] :
                        (component == 1 ? dlogj2[tid] : dlogj3[tid]);
    double value = 0.0;

    for (int il1 = 0; il1 <= p1; ++il1) {
        const int c1 = pads1 + (int)spans1[iel1] - p1 + il1 - starts1;
        const long long b1base = ((long long)(iel1 * (p1 + 1) + il1) * nder1) * nq1 + q1;
        const double n1 = b1[b1base];
        const double d1 = b1[b1base + nq1];
        for (int il2 = 0; il2 <= p2; ++il2) {
            const int c2 = pads2 + (int)spans2[iel2] - p2 + il2 - starts2;
            const long long b2base = ((long long)(iel2 * (p2 + 1) + il2) * nder2) * nq2 + q2;
            const double n2 = b2[b2base];
            const double d2 = b2[b2base + nq2];
            for (int il3 = 0; il3 <= p3; ++il3) {
                const int c3 = pads3 + (int)spans3[iel3] - p3 + il3 - starts3;
                const long long b3base = ((long long)(iel3 * (p3 + 1) + il3) * nder3) * nq3 + q3;
                const double n3 = b3[b3base];
                const double d3 = b3[b3base + nq3];
                const double basis = n1 * n2 * n3;
                const double derivative = component == 0 ? d1 * n2 * n3 :
                                          (component == 1 ? n1 * d2 * n3 : n1 * n2 * d3);
                value += coeffs[((long long)c1 * nc2 + c2) * nc3 + c3] * (derivative + dlog * basis);
            }
        }
    }
    values[tid] += value;
}

extern "C" __global__
void h1vec_divergence_transpose_cuda(
    const long long* spans1, const long long* spans2, const long long* spans3,
    const int ne1, const int ne2, const int ne3,
    const int p1, const int p2, const int p3,
    const int starts1, const int starts2, const int starts3,
    const int pads1, const int pads2, const int pads3,
    const double* b1, const double* b2, const double* b3,
    const int nder1, const int nder2, const int nder3,
    const int nq1, const int nq2, const int nq3,
    const double* dlogj1, const double* dlogj2, const double* dlogj3,
    const int component, const double* values,
    const int nc2, const int nc3, double* coeffs)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long totalq3 = (long long)ne3 * nq3;
    const long long totalq2 = (long long)ne2 * nq2;
    const long long nvalues = (long long)ne1 * nq1 * totalq2 * totalq3;
    if (tid >= nvalues) return;

    const int iq3 = tid % totalq3;
    const long long t12 = tid / totalq3;
    const int iq2 = t12 % totalq2;
    const int iq1 = t12 / totalq2;
    const int iel1 = iq1 / nq1, q1 = iq1 % nq1;
    const int iel2 = iq2 / nq2, q2 = iq2 % nq2;
    const int iel3 = iq3 / nq3, q3 = iq3 % nq3;
    const double dlog = component == 0 ? dlogj1[tid] :
                        (component == 1 ? dlogj2[tid] : dlogj3[tid]);
    const double qvalue = values[tid];

    for (int il1 = 0; il1 <= p1; ++il1) {
        const int c1 = pads1 + (int)spans1[iel1] - p1 + il1 - starts1;
        const long long b1base = ((long long)(iel1 * (p1 + 1) + il1) * nder1) * nq1 + q1;
        const double n1 = b1[b1base];
        const double d1 = b1[b1base + nq1];
        for (int il2 = 0; il2 <= p2; ++il2) {
            const int c2 = pads2 + (int)spans2[iel2] - p2 + il2 - starts2;
            const long long b2base = ((long long)(iel2 * (p2 + 1) + il2) * nder2) * nq2 + q2;
            const double n2 = b2[b2base];
            const double d2 = b2[b2base + nq2];
            for (int il3 = 0; il3 <= p3; ++il3) {
                const int c3 = pads3 + (int)spans3[iel3] - p3 + il3 - starts3;
                const long long b3base = ((long long)(iel3 * (p3 + 1) + il3) * nder3) * nq3 + q3;
                const double n3 = b3[b3base];
                const double d3 = b3[b3base + nq3];
                const double basis = n1 * n2 * n3;
                const double derivative = component == 0 ? d1 * n2 * n3 :
                                          (component == 1 ? n1 * d2 * n3 : n1 * n2 * d3);
                atomicAdd(&coeffs[((long long)c1 * nc2 + c2) * nc3 + c3], qvalue * (derivative + dlog * basis));
            }
        }
    }
}

