"""Hand-written CUDA replacements for select pusher kernels, used only under
``ARRAY_BACKEND=cupy``.

Unlike the generic Pyccel kernels in :mod:`~struphy.pic.pushing.pusher_kernels`
(which operate on plain host NumPy arrays regardless of backend), the kernels
here are real ``cupy.RawKernel`` CUDA source, executed directly on the GPU.
They are deliberately narrow: each one reproduces the exact arithmetic of one
Pyccel kernel, specialized for one :class:`~struphy.geometry.domains.Domain`
whose Jacobian is cheap enough that hand-specializing pays off.

Currently covered: :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`
for the :class:`~struphy.geometry.domains.Cuboid` domain (``kind_map == 10``),
whose Jacobian ``DF = diag(r1 - l1, r2 - l2, r3 - l3)`` is constant, so the
whole stage update collapses to an elementwise scale-and-accumulate per marker
row with no spline evaluation at all.

Two entry points are provided:

* :func:`push_eta_stage_cuboid_gpu` replaces a single ``push_eta_stage`` call.
  It round-trips the full marker array through the device on every call, which
  is fine at the marker counts used in early testing but becomes the dominant
  cost at a few hundred thousand markers and up (H2D/D2H bandwidth, not
  compute, ends up dominating the run).

* :func:`push_eta_rk_periodic_gpu` additionally fuses in the boundary-condition
  bookkeeping that :meth:`~struphy.pic.base.Particles.apply_kinetic_bc` would
  otherwise do on the host between stages (periodic wrap + shift-column
  bookkeeping), restricted to an all-periodic ``bc``. That lets the whole
  multi-stage RK push run with the marker array resident on the device the
  entire time, doing exactly one H2D and one D2H transfer per
  :meth:`~struphy.pic.pushing.pusher.Pusher.__call__`, instead of one round
  trip per stage per kernel.

Also covered: :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`,
again for the Cuboid domain. Unlike ``push_eta_stage``, this one does need a
real (small-degree) tensor-product B-spline evaluation -- the electric field
is a 1-form FEEC spline, not a constant -- so :func:`push_v_with_efield_cuboid_gpu`
ports ``find_span`` and the combined N-/D-spline basis recursion
(:func:`~struphy.bsplines.bsplines_kernels.b_d_splines_slim`) to device code
alongside the local stencil sum
(:func:`~struphy.bsplines.evaluation_kernels_3d.eval_spline_mpi_kernel`).
Basis arrays are sized to a compile-time ``MAXP`` (spline degree 8), which
comfortably covers Struphy's usual degrees. The FE coefficient arrays
(``e1_1``, ``e1_2``, ``e1_3``) are the raw ``._data`` of the field's
:class:`~feectools.linalg.stencil.StencilVector` components; under the CuPy
backend these already live on the device (``StencilVector`` allocates via
``cunumpy``'s array-backend-aware ``xp``) and are never reassigned after
:meth:`~struphy.propagators.push_vin_efield.PushVinEfield.allocate` runs, so
they are passed straight through with no transfer at all -- only the marker
array round-trips through the device, exactly once per call.
"""

_PUSH_ETA_CUBOID_SRC = r"""
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
"""

_push_eta_cuboid_kernel = None


def _get_kernel():
    global _push_eta_cuboid_kernel
    if _push_eta_cuboid_kernel is None:
        import cupy as cp

        _push_eta_cuboid_kernel = cp.RawKernel(_PUSH_ETA_CUBOID_SRC, "push_eta_stage_cuboid")
    return _push_eta_cuboid_kernel


def push_eta_stage_cuboid_gpu(
    markers,
    n_cols: int,
    first_init_idx: int,
    first_free_idx: int,
    scale: tuple[float, float, float],
    dt_a: float,
    dt_b: float,
    last: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`, restricted to
    the :class:`~struphy.geometry.domains.Cuboid` domain.

    ``markers`` is the host (pinned-memory) marker array; it is round-tripped
    through the device in full, matching the pattern used by
    :meth:`~struphy.pic.pushing.pusher.Pusher._reset_marker_buffers_gpu`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_kernel()
    n_markers = markers.shape[0]

    dev = cp.asarray(markers)
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(first_init_idx),
            np.int32(first_free_idx),
            np.float64(scale[0]),
            np.float64(scale[1]),
            np.float64(scale[2]),
            np.float64(dt_a),
            np.float64(dt_b),
            np.float64(last),
        ),
    )
    dev.get(out=markers)


_PUSH_ETA_RK_PERIODIC_SRC = r"""
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
"""

_push_eta_rk_periodic_kernel = None


def _get_periodic_kernel():
    global _push_eta_rk_periodic_kernel
    if _push_eta_rk_periodic_kernel is None:
        import cupy as cp

        _push_eta_rk_periodic_kernel = cp.RawKernel(_PUSH_ETA_RK_PERIODIC_SRC, "push_eta_rk_periodic")
    return _push_eta_rk_periodic_kernel


def push_eta_rk_periodic_gpu(
    markers,
    n_cols: int,
    vdim: int,
    first_init_idx: int,
    first_shift_idx: int,
    first_free_idx: int,
    scale: tuple[float, float, float],
    dt: float,
    a,
    b,
    n_stages: int,
):
    """Run a full multi-stage RK push of :func:`~struphy.pic.pushing.pusher_kernels.push_eta_stage`
    plus periodic boundary handling, entirely on the device.

    Restricted to the :class:`~struphy.geometry.domains.Cuboid` domain and an
    all-``"periodic"`` ``bc``. Equivalent to calling
    :func:`push_eta_stage_cuboid_gpu` once per stage followed by the periodic
    branch of :meth:`~struphy.pic.base.Particles.apply_kinetic_bc`, but with a
    single H2D transfer at the start and a single D2H transfer at the end
    instead of one round trip per stage. Holes and ghost particles are
    invariant under a periodic-only push (positions are wrapped mod 1, never
    set to the -1.0 hole sentinel), so :meth:`~struphy.pic.base.Particles.update_holes`
    does not need to be called.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_periodic_kernel()
    n_markers = markers.shape[0]
    threads = 256
    blocks = (n_markers + threads - 1) // threads

    dev = cp.asarray(markers)

    # reset: save initial phase-space coords, zero shift/free/residual columns
    # (matches Pusher._reset_marker_buffers_gpu, done once instead of per stage)
    dev[:, first_init_idx : first_init_idx + 3 + vdim] = dev[:, : 3 + vdim]
    dev[:, first_shift_idx:-2] = 0.0

    for stage in range(n_stages):
        last = 1.0 if stage == n_stages - 1 else 0.0
        kernel(
            (blocks,),
            (threads,),
            (
                dev,
                np.int32(n_cols),
                np.int32(n_markers),
                np.int32(first_init_idx),
                np.int32(first_free_idx),
                np.int32(first_shift_idx),
                np.float64(scale[0]),
                np.float64(scale[1]),
                np.float64(scale[2]),
                np.float64(dt * float(a[stage])),
                np.float64(dt * float(b[stage])),
                np.float64(last),
            ),
        )

    dev.get(out=markers)


_PUSH_V_EFIELD_CUBOID_SRC = r"""
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

// Combined N-spline (bn, p+1 values) and D-spline (bd, p values) evaluation,
// matching struphy.bsplines.bsplines_kernels.b_d_splines_slim exactly.
__device__ void b_d_splines_dev(const double* t, int p, double eta, int span, double* bn, double* bd)
{
    double left[MAXP];
    double right[MAXP];
    int pd = p - 1;

    for (int i = 0; i <= p; i++) bn[i] = 0.0;
    for (int i = 0; i < p; i++) bd[i] = 0.0;
    bn[0] = 1.0;

    for (int j = 0; j < p; j++) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.0;

        if (j == p - 1) {
            for (int il = 0; il <= pd; il++) {
                bd[pd - il] = (double)p / (t[span - il + p] - t[span - il]) * bn[pd - il];
            }
        }

        for (int r = 0; r <= j; r++) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

extern "C" __global__
void push_v_with_efield_cuboid(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1,
    const int p2,
    const int p3,
    const double* tn1,
    const int len_tn1,
    const double* tn2,
    const int len_tn2,
    const double* tn3,
    const int len_tn3,
    const int start0,
    const int start1,
    const int start2,
    const double* e1_1,
    const int n2x1,
    const int n3x1,
    const double* e1_2,
    const int n2x2,
    const int n3x2,
    const double* e1_3,
    const int n2x3,
    const int n3x3,
    const double sx,
    const double sy,
    const double sz,
    const double dt_const)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;

    // skip holes and ghost/boundary particles, matching Particles.valid_mks
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0];
    const double eta2 = row[1];
    const double eta3 = row[2];

    double bn1[MAXP + 1], bd1[MAXP];
    double bn2[MAXP + 1], bd2[MAXP];
    double bn3[MAXP + 1], bd3[MAXP];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    // e_form[0]: D-spline in direction 1, N-splines in directions 2, 3
    double e_form0 = 0.0;
    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form0 += e1_1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * bd1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }

    // e_form[1]: N-spline in direction 1, D-spline in direction 2, N-spline in direction 3
    double e_form1 = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form1 += e1_2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * bn1[il1] * bd2[il2] * bn3[il3];
            }
        }
    }

    // e_form[2]: N-splines in directions 1, 2, D-spline in direction 3
    double e_form2 = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form2 += e1_3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * bn1[il1] * bn2[il2] * bd3[il3];
            }
        }
    }

    // Cartesian E-field is DF^-T @ e_form; for Cuboid, DF is diag(sx^-1, sy^-1, sz^-1)
    // so DF^-T is diag(sx, sy, sz) -- same convention as push_eta_stage_cuboid's scale.
    row[3] += dt_const * sx * e_form0;
    row[4] += dt_const * sy * e_form1;
    row[5] += dt_const * sz * e_form2;
}
"""

_push_v_efield_cuboid_kernel = None


def _get_v_efield_kernel():
    global _push_v_efield_cuboid_kernel
    if _push_v_efield_cuboid_kernel is None:
        import cupy as cp

        _push_v_efield_cuboid_kernel = cp.RawKernel(_PUSH_V_EFIELD_CUBOID_SRC, "push_v_with_efield_cuboid")
    return _push_v_efield_cuboid_kernel


def push_v_with_efield_cuboid_gpu(
    markers,
    n_cols: int,
    pn: tuple[int, int, int],
    tn1_dev,
    tn2_dev,
    tn3_dev,
    starts: tuple[int, int, int],
    e1_1_dev,
    e1_2_dev,
    e1_3_dev,
    scale: tuple[float, float, float],
    dt_const: float,
):
    """GPU replacement for one call of
    :func:`~struphy.pic.pushing.pusher_kernels.push_v_with_efield`, restricted
    to the :class:`~struphy.geometry.domains.Cuboid` domain.

    ``markers`` is the host marker array and is round-tripped through the
    device once (matching :func:`push_eta_stage_cuboid_gpu`). ``tn1_dev``,
    ``tn2_dev``, ``tn3_dev`` (knot vectors) and ``e1_1_dev``, ``e1_2_dev``,
    ``e1_3_dev`` (FE coefficients of the 1-form E-field) are expected to
    already be CuPy arrays resident on the device -- callers should cache
    them once rather than converting on every call, see
    :class:`~struphy.pic.pushing.pusher.Pusher`.
    """
    import cupy as cp
    import numpy as np

    kernel = _get_v_efield_kernel()
    n_markers = markers.shape[0]

    dev = cp.asarray(markers)
    threads = 256
    blocks = (n_markers + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            dev,
            np.int32(n_cols),
            np.int32(n_markers),
            np.int32(pn[0]),
            np.int32(pn[1]),
            np.int32(pn[2]),
            tn1_dev,
            np.int32(tn1_dev.shape[0]),
            tn2_dev,
            np.int32(tn2_dev.shape[0]),
            tn3_dev,
            np.int32(tn3_dev.shape[0]),
            np.int32(starts[0]),
            np.int32(starts[1]),
            np.int32(starts[2]),
            e1_1_dev,
            np.int32(e1_1_dev.shape[1]),
            np.int32(e1_1_dev.shape[2]),
            e1_2_dev,
            np.int32(e1_2_dev.shape[1]),
            np.int32(e1_2_dev.shape[2]),
            e1_3_dev,
            np.int32(e1_3_dev.shape[1]),
            np.int32(e1_3_dev.shape[2]),
            np.float64(scale[0]),
            np.float64(scale[1]),
            np.float64(scale[2]),
            np.float64(dt_const),
        ),
    )
    dev.get(out=markers)
