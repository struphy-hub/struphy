"""Hand-written CUDA replacements for the per-marker *diagnostics* kernels in
:mod:`~struphy.pic.utilities_kernels`, used only under ``ARRAY_BACKEND=cupy``.

These run every time step (they back the scalar quantities a model saves,
e.g. ``en_fB`` in :class:`~struphy.models.guiding_center.GuidingCenter`), and
each one writes a diagnostics column of the marker array in place. With
markers now device-resident (see :class:`~struphy.pic.base.Particles`), the
compiled host-only versions were the last thing forcing a host<->device
round trip of the whole marker array in the per-step path -- porting them
removes it.

Both kernels here are plain per-marker 0-form spline evaluations, so they
reuse the ``find_span_dev``/``b_splines_dev``/``eval_0form_dev`` device
functions rather than defining their own.
"""

_UTILITIES_SRC = r"""
#define MAXP 8

__device__ int find_span_dev(const double* t, int p, int len_t, double eta)
{
    int low = p;
    int high = len_t - 1 - p;

    if (eta <= t[low]) return low;
    if (eta >= t[high]) return high - 1;

    int span = (low + high) / 2;
    while (eta < t[span] || eta >= t[span + 1]) {
        if (eta < t[span]) high = span;
        else low = span;
        span = (low + high) / 2;
    }
    return span;
}

__device__ void b_splines_dev(const double* t, int p, double eta, int span, double* bn)
{
    double left[MAXP];
    double right[MAXP];

    for (int i = 0; i <= p; i++) bn[i] = 0.0;
    bn[0] = 1.0;

    for (int j = 0; j < p; j++) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.0;
        for (int r = 0; r <= j; r++) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

__device__ double eval_0form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bn2, const double* bn3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c, int n2x, int n3x)
{
    double out = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out += c[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3] * bn1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    return out;
}

// markers[ip, first_diagnostics_idx] = mu_p * |B_0(eta_p)|
extern "C" __global__
void eval_magnetic_background_energy_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* abs_B0, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, abs_B0, n2x, n3x);

    row[first_diagnostics_idx] = mu * abs_B;
}

// markers[ip, first_diagnostics_idx] = v_par^2 / 2 + mu_p * |B(eta_p)|
extern "C" __global__
void eval_energy_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_parallel = row[3];
    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    row[first_diagnostics_idx] = 0.5 * v_parallel * v_parallel + mu * abs_B;
}

// markers[ip, idx_can_momentum] = shifted canonical toroidal momentum (5D)
extern "C" __global__
void eval_canonical_toroidal_moment_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx, const int idx_can_momentum,
    const double epsilon, const double B0, const double R0,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_para = row[3];
    const double mu = row[mu_idx];
    const double energy = row[first_diagnostics_idx];
    const double psi = row[idx_can_momentum];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    double out = psi - epsilon * B0 * R0 / abs_B * v_para;
    if (energy - mu * B0 > 0.0) {
        // sign(v_para) matches numpy.sign: 0 for exactly 0
        const double sgn = (v_para > 0.0) ? 1.0 : ((v_para < 0.0) ? -1.0 : 0.0);
        out += epsilon * sgn * sqrt(2.0 * (energy - mu * B0)) * R0;
    }
    row[idx_can_momentum] = out;
}

// markers[ip, first_diagnostics_idx + 5] = shifted canonical toroidal momentum (6D)
extern "C" __global__
void eval_canonical_toroidal_moment_6d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const double epsilon, const double B0, const double R0,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double energy = row[first_diagnostics_idx + 3];
    const double mu = row[first_diagnostics_idx + 4];
    const double psi = row[first_diagnostics_idx + 5];
    const double v_para = row[first_diagnostics_idx + 6];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    double out = psi - epsilon * B0 * R0 / abs_B * v_para;
    if (energy - mu * B0 > 0.0) {
        const double sgn = (v_para > 0.0) ? 1.0 : ((v_para < 0.0) ? -1.0 : 0.0);
        out += epsilon * sgn * sqrt(2.0 * (energy - mu * B0)) * R0;
    }
    row[first_diagnostics_idx + 5] = out;
}

// markers[ip, first_diagnostics_idx + 1] = v_perp^2 / (2 |B(eta_p)|)
extern "C" __global__
void eval_magnetic_moment_5d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* absB, const int n2x, const int n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double v_perp = row[4];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_splines_dev(tn1, p1, eta1, span1, bn1);
    b_splines_dev(tn2, p2, eta2, span2, bn2);
    b_splines_dev(tn3, p3, eta3, span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, n2x, n3x);

    row[first_diagnostics_idx + 1] = 0.5 * v_perp * v_perp / abs_B;
}

// markers[ip, first_diagnostics_idx] = mu_p * (|B_0| + PBb)(eta_p)
// NOTE: the CPU reference also evaluates the Jacobian DF(eta) here, but never
// uses the result, so it is not replicated.
extern "C" __global__
void eval_magnetic_energy_PBb_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx, const int mu_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* abs_B0, const int a_n2x, const int a_n3x,
    const double* PBb, const int b_n2x, const int b_n3x)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    // eta = mod(markers[0:3], 1.0); fmod can return negative, match numpy mod
    double eta[3];
    for (int k = 0; k < 3; k++) {
        double e = fmod(row[k], 1.0);
        if (e < 0.0) e += 1.0;
        eta[k] = e;
    }

    const double mu = row[mu_idx];

    double bn1[MAXP + 1], bn2[MAXP + 1], bn3[MAXP + 1];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta[2]);
    b_splines_dev(tn1, p1, eta[0], span1, bn1);
    b_splines_dev(tn2, p2, eta[1], span2, bn2);
    b_splines_dev(tn3, p3, eta[2], span3, bn3);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, abs_B0, a_n2x, a_n3x);
    const double PB_b = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, PBb, b_n2x, b_n3x);

    row[first_diagnostics_idx] = mu * (abs_B + PB_b);
}
"""

_kernels = {}


def _get_kernel(name):
    if name not in _kernels:
        import cupy as cp

        _kernels[name] = cp.RawKernel(_UTILITIES_SRC, name)
    return _kernels[name]


def _launch_0form_diag(kernel_name, markers, args_derham, first_diagnostics_idx, mu_idx, coeffs):
    """Shared launch path for the two 0-form diagnostics kernels above."""
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    coeffs = cp.ascontiguousarray(coeffs)

    _get_kernel(kernel_name)(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            coeffs,
            np.int32(coeffs.shape[1]),
            np.int32(coeffs.shape[2]),
        ),
    )


def eval_magnetic_background_energy_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, abs_B0):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_background_energy`.
    ``markers`` is device-resident and written in place.
    """
    _launch_0form_diag(
        "eval_magnetic_background_energy_cuda",
        markers,
        args_derham,
        first_diagnostics_idx,
        mu_idx,
        abs_B0,
    )


def eval_energy_5d_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, absB):
    """GPU replacement for :func:`~struphy.pic.utilities_kernels.eval_energy_5d`.
    ``markers`` is device-resident and written in place.
    """
    _launch_0form_diag(
        "eval_energy_5d_cuda",
        markers,
        args_derham,
        first_diagnostics_idx,
        mu_idx,
        absB,
    )


def eval_canonical_toroidal_moment_5d_gpu(
    markers, args_derham, first_diagnostics_idx, mu_idx, idx_can_momentum, epsilon, B0, R0, absB
):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_canonical_toroidal_moment_5d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    _get_kernel("eval_canonical_toroidal_moment_5d_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
            np.int32(idx_can_momentum),
            np.float64(epsilon),
            np.float64(B0),
            np.float64(R0),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_canonical_toroidal_moment_6d_gpu(markers, args_derham, first_diagnostics_idx, epsilon, B0, R0, absB):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_canonical_toroidal_moment_6d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    _get_kernel("eval_canonical_toroidal_moment_6d_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.float64(epsilon),
            np.float64(B0),
            np.float64(R0),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_magnetic_moment_5d_gpu(markers, args_derham, first_diagnostics_idx, absB):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_moment_5d`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    absB = cp.ascontiguousarray(absB)
    _get_kernel("eval_magnetic_moment_5d_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


def eval_magnetic_energy_PBb_gpu(markers, args_derham, first_diagnostics_idx, mu_idx, abs_B0, PBb):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_magnetic_energy_PBb`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    abs_B0 = cp.ascontiguousarray(abs_B0)
    PBb = cp.ascontiguousarray(PBb)
    _get_kernel("eval_magnetic_energy_PBb_cuda")(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(mu_idx),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            abs_B0,
            np.int32(abs_B0.shape[1]),
            np.int32(abs_B0.shape[2]),
            PBb,
            np.int32(PBb.shape[1]),
            np.int32(PBb.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# eval_guiding_center_from_6d needs the domain Jacobian and a 2-form (magnetic
# field) evaluation, so unlike the pure 0-form diagnostics above it is built
# on top of pusher_kernels_cuda's shared geometry/spline device functions
# rather than the small self-contained source in this module.
# ---------------------------------------------------------------------------

_GC_FROM_6D_SRC = r"""
extern "C" __global__
void eval_guiding_center_from_6d_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_diagnostics_idx,
    const int kind_map, const double* params,
    const double epsilon,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b21, const int b1_n2, const int b1_n3,
    const double* b22, const int b2_n2, const int b2_n3,
    const double* b23, const int b3_n2, const int b3_n3,
    const double* absB, const int a_n2, const int a_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    const double x = row[first_diagnostics_idx];
    const double y = row[first_diagnostics_idx + 1];
    const double z = row[first_diagnostics_idx + 2];
    double v[3] = {row[3], row[4], row[5]};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;
    const double det_df = det3_dev(dfm);

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b2[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3,
        start0, start1, start2,
        b21, b1_n2, b1_n3, b22, b2_n2, b2_n3, b23, b3_n2, b3_n3, b2);

    const double abs_B = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
        start0, start1, start2, absB, a_n2, a_n3);

    // normalized magnetic field, cartesian
    b2[0] /= abs_B; b2[1] /= abs_B; b2[2] /= abs_B;
    double norm_b_cart[3];
    matvec_dev(dfm, b2, norm_b_cart);
    norm_b_cart[0] /= det_df; norm_b_cart[1] /= det_df; norm_b_cart[2] /= det_df;

    const double v_parallel = dot3_dev(norm_b_cart, v);

    double temp[3], v_perp[3];
    cross_dev(v, norm_b_cart, temp);
    cross_dev(norm_b_cart, temp, v_perp);
    const double v_perp_square = v_perp[0]*v_perp[0] + v_perp[1]*v_perp[1] + v_perp[2]*v_perp[2];

    row[first_diagnostics_idx + 6] = v_parallel;
    row[first_diagnostics_idx + 4] = 0.5 * v_perp_square / abs_B;

    double Larmor_r[3];
    cross_dev(norm_b_cart, v_perp, Larmor_r);
    for (int k = 0; k < 3; k++) Larmor_r[k] = Larmor_r[k] / abs_B * epsilon;

    row[first_diagnostics_idx + 0] = x - Larmor_r[0];
    row[first_diagnostics_idx + 1] = y - Larmor_r[1];
    row[first_diagnostics_idx + 2] = z - Larmor_r[2];
}
"""

_gc6d_kernel = None


def _get_gc_from_6d_kernel():
    global _gc6d_kernel
    if _gc6d_kernel is None:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _gc6d_kernel = cp.RawKernel(
            _GENERAL_GEOMETRY_SRC + _GC_FROM_6D_SRC,
            "eval_guiding_center_from_6d_cuda",
        )
    return _gc6d_kernel


def eval_guiding_center_from_6d_gpu(
    markers, args_derham, kind_map, params_dev, first_diagnostics_idx, epsilon, b21, b22, b23, absB
):
    """GPU replacement for
    :func:`~struphy.pic.utilities_kernels.eval_guiding_center_from_6d`, for any
    domain in :data:`~struphy.pic.pushing.pusher_kernels_cuda.SUPPORTED_GENERAL_KIND_MAPS`.
    ``markers`` is device-resident and written in place.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
    tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
    tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
    b21 = cp.ascontiguousarray(b21)
    b22 = cp.ascontiguousarray(b22)
    b23 = cp.ascontiguousarray(b23)
    absB = cp.ascontiguousarray(absB)
    _get_gc_from_6d_kernel()(
        (blocks,),
        (threads,),
        (
            markers,
            np.int32(markers.shape[1]),
            np.int32(n_markers),
            np.int32(first_diagnostics_idx),
            np.int32(kind_map),
            params_dev,
            np.float64(epsilon),
            np.int32(args_derham.pn[0]),
            np.int32(args_derham.pn[1]),
            np.int32(args_derham.pn[2]),
            tn1,
            np.int32(tn1.shape[0]),
            tn2,
            np.int32(tn2.shape[0]),
            tn3,
            np.int32(tn3.shape[0]),
            np.int32(args_derham.starts[0]),
            np.int32(args_derham.starts[1]),
            np.int32(args_derham.starts[2]),
            b21,
            np.int32(b21.shape[1]),
            np.int32(b21.shape[2]),
            b22,
            np.int32(b22.shape[1]),
            np.int32(b22.shape[2]),
            b23,
            np.int32(b23.shape[1]),
            np.int32(b23.shape[2]),
            absB,
            np.int32(absB.shape[1]),
            np.int32(absB.shape[2]),
        ),
    )


# ---------------------------------------------------------------------------
# eval_gradB_ediff: writes markers[:, idx] = mu * dot(eta_diff, gradB +
# grad_PB_b), evaluated at the midpoint eta_mid = mod((eta+eta_init)/2, 1).
# Called once per fixed-point iteration by CurrentCoupling5DGradB's
# discrete-gradient algorithm. Needs 1-form spline evaluation (unlike the
# 0-form diagnostics above), so this is built from
# pusher_kernels_cuda._GENERAL_GEOMETRY_SRC instead of the private
# find_span_dev/b_splines_dev/eval_0form_dev helpers used by _UTILITIES_SRC.
# ---------------------------------------------------------------------------

_GRADB_EDIFF_SRC = r"""
__device__ double gradb_ediff_mod1_dev(double x)
{
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

extern "C" __global__
void eval_gradB_ediff_cuda(
    double* markers, const int n_cols, const int n_markers,
    const int first_init_idx, const int mu_idx, const int idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* gb1, const int g1_n2, const int g1_n3,
    const double* gb2, const int g2_n2, const int g2_n3,
    const double* gb3, const int g3_n2, const int g3_n3,
    const double* pb1, const int p1_n2, const int p1_n3,
    const double* pb2, const int p2_n2, const int p2_n3,
    const double* pb3, const int p3_n2, const int p3_n3)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    double eta_mid[3], eta_diff[3];
    for (int k = 0; k < 3; k++) {
        eta_mid[k] = gradb_ediff_mod1_dev((row[k] + row[first_init_idx + k]) / 2.0);
        eta_diff[k] = row[k] - row[first_init_idx + k];
    }
    const double mu = row[mu_idx];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta_mid[0]);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta_mid[1]);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta_mid[2]);
    double bn1[MAXP+1], bd1[MAXP];
    double bn2[MAXP+1], bd2[MAXP];
    double bn3[MAXP+1], bd3[MAXP];
    b_d_splines_dev(tn1, p1, eta_mid[0], span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta_mid[1], span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta_mid[2], span3, bn3, bd3);

    double gradB[3], grad_PB_b[3];
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        gb1,g1_n2,g1_n3, gb2,g2_n2,g2_n3, gb3,g3_n2,g3_n3, gradB);
    eval_1form_dev(p1,p2,p3, bn1,bd1,bn2,bd2,bn3,bd3, span1,span2,span3, start0,start1,start2,
        pb1,p1_n2,p1_n3, pb2,p2_n2,p2_n3, pb3,p3_n2,p3_n3, grad_PB_b);

    double tmp[3];
    for (int k = 0; k < 3; k++) tmp[k] = gradB[k] + grad_PB_b[k];

    row[idx] = mu * dot3_dev(eta_diff, tmp);
}
"""

_gradb_ediff_kernel = None


def _get_gradb_ediff_kernel():
    global _gradb_ediff_kernel
    if _gradb_ediff_kernel is None:
        import cupy as cp

        from struphy.pic.pushing.pusher_kernels_cuda import _GENERAL_GEOMETRY_SRC

        _gradb_ediff_kernel = cp.RawKernel(_GENERAL_GEOMETRY_SRC + _GRADB_EDIFF_SRC, "eval_gradB_ediff_cuda")
    return _gradb_ediff_kernel


def eval_gradB_ediff_gpu(
    markers, first_init_idx, mu_idx,
    pn, tn1_dev, tn2_dev, tn3_dev, starts,
    gradB1_dev, grad_PB_b1_dev, idx,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.utilities_kernels.eval_gradB_ediff`.

    ``gradB1_dev``/``grad_PB_b1_dev`` are each a 3-tuple of device arrays
    (the 1-form's 3 components), matching the (unpacked) ``gradB1, gradB2,
    gradB3`` / ``grad_PB_b1, grad_PB_b2, grad_PB_b3`` arguments of the CPU
    kernel.
    """
    import cupy as cp
    import numpy as np

    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    def d(a):
        a = cp.ascontiguousarray(a)
        return (a, np.int32(a.shape[1]), np.int32(a.shape[2]))

    _get_gradb_ediff_kernel()(
        (blocks,),
        (threads,),
        (
            markers, np.int32(markers.shape[1]), np.int32(n_markers),
            np.int32(first_init_idx), np.int32(mu_idx), np.int32(idx),
            np.int32(pn[0]), np.int32(pn[1]), np.int32(pn[2]),
            tn1_dev, np.int32(tn1_dev.shape[0]),
            tn2_dev, np.int32(tn2_dev.shape[0]),
            tn3_dev, np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]), np.int32(starts[1]), np.int32(starts[2]),
            *d(gradB1_dev[0]), *d(gradB1_dev[1]), *d(gradB1_dev[2]),
            *d(grad_PB_b1_dev[0]), *d(grad_PB_b1_dev[1]), *d(grad_PB_b1_dev[2]),
        ),
    )
