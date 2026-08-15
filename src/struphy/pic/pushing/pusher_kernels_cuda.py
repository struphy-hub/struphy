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
