"""CUDA implementations of matrix-free FEEC mass-operator kernels.

These kernels are used only when ``ARRAY_BACKEND=cupy``.  The corresponding
Pyccel kernels accept CuPy arrays, but execute their nested loops on the host;
the routines below keep both the quadrature data and coefficient vectors on
the device.
"""

_H1VEC_DIVERGENCE_SRC = r"""
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
"""

_divergence_eval_kernel = None
_divergence_transpose_kernel = None

_MASS_ASSEMBLY_SRC = r"""
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
"""

_mass_assembly_kernel = None

_WEAK_DIV_ASSEMBLY_SRC = r"""
extern "C" __global__
void weak_div_assemble_cuda(
    const long long* s1, const long long* s2, const long long* s3,
    const int ne1, const int ne2, const int ne3,
    const int pi1, const int pi2, const int pi3,
    const int pj1, const int pj2, const int pj3,
    const int st1, const int st2, const int st3,
    const int pad1, const int pad2, const int pad3,
    const double* w1, const double* w2, const double* w3,
    const int nq1, const int nq2, const int nq3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    const int ndi1, const int ndi2, const int ndi3,
    const int ndj1, const int ndj2, const int ndj3,
    const double* weight, const double* dl1, const double* dl2, const double* dl3,
    const int component, double* data,
    const int dd2, const int dd3, const int dd4, const int dd5, const int dd6)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long ni = (long long)(pi1+1)*(pi2+1)*(pi3+1);
    const long long nj = (long long)(pj1+1)*(pj2+1)*(pj3+1);
    const long long total = (long long)ne1*ne2*ne3*ni*nj;
    if (tid >= total) return;
    long long t=tid;
    const long long j=t%nj; t/=nj;
    const long long i=t%ni; t/=ni;
    const int e3=t%ne3; t/=ne3;
    const int e2=t%ne2; const int e1=t/ne2;
    const int il3=i%(pi3+1), il2=(i/(pi3+1))%(pi2+1), il1=i/((pi2+1)*(pi3+1));
    const int jl3=j%(pj3+1), jl2=(j/(pj3+1))%(pj2+1), jl1=j/((pj2+1)*(pj3+1));
    const int c1=pad1+(int)s1[e1]-pi1+il1-st1;
    const int c2=pad2+(int)s2[e2]-pi2+il2-st2;
    const int c3=pad3+(int)s3[e3]-pi3+il3-st3;
    const int o1=pad1+jl1-il1, o2=pad2+jl2-il2, o3=pad3+jl3-il3;
    double value=0.0;
    for(int q1=0;q1<nq1;++q1) {
      const long long ib1=((long long)(e1*(pi1+1)+il1)*ndi1)*nq1+q1;
      const long long jb1=((long long)(e1*(pj1+1)+jl1)*ndj1)*nq1+q1;
      const double ti1=bi1[ib1], tn1=bj1[jb1], td1=bj1[jb1+nq1];
      for(int q2=0;q2<nq2;++q2) {
        const long long ib2=((long long)(e2*(pi2+1)+il2)*ndi2)*nq2+q2;
        const long long jb2=((long long)(e2*(pj2+1)+jl2)*ndj2)*nq2+q2;
        const double ti12=ti1*bi2[ib2], tn2=bj2[jb2], td2=bj2[jb2+nq2];
        for(int q3=0;q3<nq3;++q3) {
          const long long ib3=((long long)(e3*(pi3+1)+il3)*ndi3)*nq3+q3;
          const long long jb3=((long long)(e3*(pj3+1)+jl3)*ndj3)*nq3+q3;
          const double tn3=bj3[jb3], td3=bj3[jb3+nq3];
          const double basis=tn1*tn2*tn3;
          const double deriv=component==0 ? td1*tn2*tn3 : (component==1 ? tn1*td2*tn3 : tn1*tn2*td3);
          const long long qi=((long long)(e1*nq1+q1)*(ne2*nq2)+e2*nq2+q2)*(ne3*nq3)+e3*nq3+q3;
          const double dl=component==0 ? dl1[qi] : (component==1 ? dl2[qi] : dl3[qi]);
          value += w1[e1*nq1+q1]*w2[e2*nq2+q2]*w3[e3*nq3+q3]
                   * weight[qi] * ti12*bi3[ib3] * (deriv+basis*dl);
        }
      }
    }
    const long long di=(((((long long)c1*dd2+c2)*dd3+c3)*dd4+o1)*dd5+o2)*dd6+o3;
    atomicAdd(&data[di],value);
}
"""

_weak_div_assembly_kernel = None

_H1VEC_DIVDIV_ASSEMBLY_SRC = r"""
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
"""

_h1vec_divdiv_assembly_kernel = None


def _get_h1vec_divdiv_assembly_kernel():
    global _h1vec_divdiv_assembly_kernel
    if _h1vec_divdiv_assembly_kernel is None:
        import cupy as cp

        _h1vec_divdiv_assembly_kernel = cp.RawKernel(_H1VEC_DIVDIV_ASSEMBLY_SRC, "h1vec_divdiv_assemble_cuda")
    return _h1vec_divdiv_assembly_kernel


def _get_mass_assembly_kernel():
    global _mass_assembly_kernel
    if _mass_assembly_kernel is None:
        import cupy as cp

        _mass_assembly_kernel = cp.RawKernel(_MASS_ASSEMBLY_SRC, "mass_3d_assemble_cuda")
    return _mass_assembly_kernel


def _get_weak_div_assembly_kernel():
    global _weak_div_assembly_kernel
    if _weak_div_assembly_kernel is None:
        import cupy as cp

        _weak_div_assembly_kernel = cp.RawKernel(_WEAK_DIV_ASSEMBLY_SRC, "weak_div_assemble_cuda")
    return _weak_div_assembly_kernel


def _get_kernels():
    global _divergence_eval_kernel, _divergence_transpose_kernel
    if _divergence_eval_kernel is None:
        import cupy as cp

        _divergence_eval_kernel = cp.RawKernel(_H1VEC_DIVERGENCE_SRC, "h1vec_divergence_eval_cuda")
        _divergence_transpose_kernel = cp.RawKernel(_H1VEC_DIVERGENCE_SRC, "h1vec_divergence_transpose_cuda")
    return _divergence_eval_kernel, _divergence_transpose_kernel


def _kernel_args(spans, degree, starts, pads, bases, dlogj, component):
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    dlogj = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in dlogj)
    return (
        *spans,
        np.int32(spans[0].size),
        np.int32(spans[1].size),
        np.int32(spans[2].size),
        *(np.int32(x) for x in degree),
        *(np.int32(x) for x in starts),
        *(np.int32(x) for x in pads),
        *bases,
        np.int32(bases[0].shape[2]),
        np.int32(bases[1].shape[2]),
        np.int32(bases[2].shape[2]),
        np.int32(bases[0].shape[3]),
        np.int32(bases[1].shape[3]),
        np.int32(bases[2].shape[3]),
        *dlogj,
        np.int32(component),
    )


def h1vec_divergence_eval_gpu(spans, degree, starts, pads, bases, dlogj, component, coeffs, values):
    """Add one H1-vector component's divergence to device ``values``."""
    import numpy as np

    kernel, _ = _get_kernels()
    args = _kernel_args(spans, degree, starts, pads, bases, dlogj, component)
    nvalues = values.size
    threads = 256
    kernel(
        ((nvalues + threads - 1) // threads,),
        (threads,),
        (*args, coeffs, np.int32(coeffs.shape[1]), np.int32(coeffs.shape[2]), values),
    )


def h1vec_divergence_transpose_gpu(spans, degree, starts, pads, bases, dlogj, component, values, coeffs):
    """Accumulate the transpose of one H1-vector divergence component."""
    import numpy as np

    _, kernel = _get_kernels()
    args = _kernel_args(spans, degree, starts, pads, bases, dlogj, component)
    nvalues = values.size
    threads = 256
    kernel(
        ((nvalues + threads - 1) // threads,),
        (threads,),
        (*args, values, np.int32(coeffs.shape[1]), np.int32(coeffs.shape[2]), coeffs),
    )


def mass_3d_assemble_gpu(spans, degree_i, degree_j, starts, pads, weights, bases_i, bases_j, mat_fun, data):
    """Assemble a 3D weighted mass matrix directly into device stencil data."""
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases_i = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_i)
    bases_j = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_j)
    mat_fun = cp.ascontiguousarray(mat_fun)
    total = int(
        np.prod([x.size for x in spans]) * np.prod([x + 1 for x in degree_i]) * np.prod([x + 1 for x in degree_j])
    )
    threads = 256
    _get_mass_assembly_kernel()(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree_i),
            *(np.int32(x) for x in degree_j),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *weights,
            *(np.int32(x.shape[1]) for x in weights),
            *bases_i,
            *bases_j,
            *(np.int32(x.shape[2]) for x in bases_i),
            *(np.int32(x.shape[2]) for x in bases_j),
            mat_fun,
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )


def weak_divergence_assemble_gpu(
    spans,
    degree_i,
    degree_j,
    starts,
    pads,
    weights,
    bases_i,
    bases_j,
    mat_fun,
    dlogj,
    component,
    data,
):
    """Assemble one L2-by-H1 weak-divergence block on the GPU."""
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases_i = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_i)
    bases_j = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_j)
    dlogj = tuple(cp.ascontiguousarray(x) for x in dlogj)
    mat_fun = cp.ascontiguousarray(mat_fun)
    total = int(
        np.prod([x.size for x in spans]) * np.prod([p + 1 for p in degree_i]) * np.prod([p + 1 for p in degree_j])
    )
    threads = 256
    _get_weak_div_assembly_kernel()(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree_i),
            *(np.int32(x) for x in degree_j),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *weights,
            *(np.int32(x.shape[1]) for x in weights),
            *bases_i,
            *bases_j,
            *(np.int32(x.shape[2]) for x in bases_i),
            *(np.int32(x.shape[2]) for x in bases_j),
            mat_fun,
            *dlogj,
            np.int32(component),
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )


def h1vec_divdiv_assemble_gpu(spans, degree, starts, pads, bases, weighted_rho, component_test, component_trial, data):
    """Assemble one H1-vector div-div block on the GPU.

    This mirrors the existing Pyccel kernel exactly, including its current
    affine-mapping formulation where the log-Jacobian terms vanish.
    """
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    weighted_rho = cp.ascontiguousarray(weighted_rho)
    nloc = int(np.prod([x + 1 for x in degree]))
    total = int(np.prod([x.size for x in spans]) * nloc * nloc)
    threads = 256
    _get_h1vec_divdiv_assembly_kernel()(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *bases,
            *(np.int32(x.shape[2]) for x in bases),
            *(np.int32(x.shape[3]) for x in bases),
            weighted_rho,
            np.int32(component_test),
            np.int32(component_trial),
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )
