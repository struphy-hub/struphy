"""CUDA kernels for fused variational grid evaluations."""

_KINETIC_ENERGY_KERNEL = None

_KINETIC_ENERGY_SOURCE = r"""
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
                us[0] += u0[ci] * weight;
                us[1] += u1[ci] * weight;
                us[2] += u2[ci] * weight;
                vs[0] += v0[ci] * weight;
                vs[1] += v1[ci] * weight;
                vs[2] += v2[ci] * weight;
            }
        }
    }

    double value = 0.0;
    // metric is stored as (3, 3, n0, n1, n2).
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            value += us[i] * metric[((long long)i * 3 + j) * total + tid] * vs[j];
    out[tid] = 0.5 * value;
    ug0[tid] = us[0]; ug1[tid] = us[1]; ug2[tid] = us[2];
    vg0[tid] = vs[0]; vg1[tid] = vs[1]; vg2[tid] = vs[2];
}
"""


def prepare_kinetic_energy_kernel():
    """Compile and cache the fused kinetic-energy CUDA kernel."""
    import cupy as cp

    global _KINETIC_ENERGY_KERNEL
    if _KINETIC_ENERGY_KERNEL is None:
        _KINETIC_ENERGY_KERNEL = cp.RawKernel(
            _KINETIC_ENERGY_SOURCE,
            "kinetic_energy_grid_cuda",
        )
        # Force NVRTC compilation during model setup rather than the first
        # timed propagation step.
        _KINETIC_ENERGY_KERNEL.compile()
    return _KINETIC_ENERGY_KERNEL


def kinetic_energy_grid_gpu(
    spans,
    bases,
    degree,
    starts,
    coefficients,
    coefficients1,
    metric,
    out,
    values,
    values1,
):
    """Evaluate both H1-vector splines and their metric product in one launch."""
    import cupy as cp
    import numpy as np

    kernel = prepare_kinetic_energy_kernel()
    spans = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.int64)) for value in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64)) for value in bases)
    coefficients = tuple(cp.ascontiguousarray(value) for value in coefficients)
    coefficients1 = tuple(cp.ascontiguousarray(value) for value in coefficients1)
    metric = cp.ascontiguousarray(metric)
    total = out.size
    threads = 256
    kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *spans,
            *bases,
            *(np.int32(value.size) for value in spans),
            *(np.int32(value) for value in degree),
            *(np.int32(value) for value in starts),
            *coefficients,
            *coefficients1,
            np.int32(coefficients[0].shape[1]),
            np.int32(coefficients[0].shape[2]),
            metric,
            out,
            *values,
            *values1,
        ),
    )
    return out
