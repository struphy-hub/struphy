"""Hand-written CUDA replacement for the per-particle sorting-box bookkeeping in
:mod:`~struphy.pic.sorting_kernels`, used only under ``ARRAY_BACKEND=cupy``.

:func:`~struphy.pic.sorting_kernels.assign_box_to_each_particle` and
:func:`~struphy.pic.sorting_kernels.assign_particles_to_boxes` are called every
time :meth:`~struphy.pic.base.Particles.put_particles_in_boxes` runs -- which is
every stage of every SPH pusher call (``Pusher._box_comm`` is true for all SPH
particles) and every :meth:`~struphy.pic.base.Particles.eval_density`/
:meth:`~struphy.pic.base.Particles.eval_velocity` call (via ``_eval_sph``) --
so unlike the SPH kernel-density evaluation itself, this is a genuine per-step
hot loop, not just a diagnostics entry point.

Both operations are per-particle and read-mostly:

* :func:`assign_box_to_each_particle_gpu` computes, for every marker, the
  sorting box it currently sits in (:func:`~struphy.pic.sorting_kernels.find_box`,
  identical logic to ``find_box_dev`` in :mod:`~struphy.pic.sph_eval_kernels_cuda`)
  and writes the box id into the marker's box column. Embarrassingly parallel,
  one thread per marker, no cross-thread interaction.
* :func:`assign_particles_to_boxes_gpu` inverts that: for every non-hole
  marker, atomically claims the next free slot in its box's row of the
  ``boxes`` array via ``atomicAdd`` (the parallel equivalent of the CPU
  version's sequential ``next_index[a] += 1`` counter) and writes the marker's
  row index there. The order in which markers land within a box's row is
  therefore not deterministic (unlike the CPU kernel, which fills boxes in
  marker-row order) -- harmless, since ``boxes`` is read as an unordered
  membership list everywhere else (the 27-neighbour SPH sums, ghost-particle
  bookkeeping).

Only ``eta1``/``eta2``/``eta3`` (columns 0:3) and the box column are
transferred for the first kernel, and only the box column for the second --
not the full ``markers`` array, which at ~350 bytes/row would make the
host<->device round trip far more expensive than the kernel itself for these
two lightweight per-particle operations (unlike
:func:`~struphy.pic.sph_eval_kernels_cuda.box_based_evaluation_flat_gpu`,
which genuinely needs every marker column for the density sum).
"""

import numpy as np

_SORT_SRC = r"""
extern "C" __device__ long long flatten_index_dev(
    long long n1, long long n2, long long n3,
    long long nx, long long ny, long long nz)
{
    // fortran_ordering (the struphy default)
    return n1 + n2 * (nx + 2) + n3 * (nx + 2) * (ny + 2);
}

extern "C" __device__ long long find_box_dev(
    double eta1, double eta2, double eta3,
    long long nx, long long ny, long long nz,
    const double* domain_array)
{
    if (eta1 == domain_array[0]) eta1 += 1e-8;
    if (eta2 == domain_array[3]) eta2 += 1e-8;
    if (eta3 == domain_array[6]) eta3 += 1e-8;
    if (eta1 == domain_array[1]) eta1 -= 1e-8;
    if (eta2 == domain_array[4]) eta2 -= 1e-8;
    if (eta3 == domain_array[7]) eta3 -= 1e-8;

    double x_l = domain_array[0] - (domain_array[1] - domain_array[0]) / nx;
    double x_r = domain_array[1] + (domain_array[1] - domain_array[0]) / nx;
    double y_l = domain_array[3] - (domain_array[4] - domain_array[3]) / ny;
    double y_r = domain_array[4] + (domain_array[4] - domain_array[3]) / ny;
    double z_l = domain_array[6] - (domain_array[7] - domain_array[6]) / nz;
    double z_r = domain_array[7] + (domain_array[7] - domain_array[6]) / nz;

    if (eta1 < x_l || eta1 > x_r || eta2 < y_l || eta2 > y_r || eta3 < z_l || eta3 > z_r)
        return -1;

    long long n1 = (long long)floor((eta1 - x_l) / (x_r - x_l) * (nx + 2));
    long long n2 = (long long)floor((eta2 - y_l) / (y_r - y_l) * (ny + 2));
    long long n3 = (long long)floor((eta3 - z_l) / (z_r - z_l) * (nz + 2));

    return flatten_index_dev(n1, n2, n3, nx, ny, nz);
}

extern "C" __global__
void assign_box_to_each_particle_cuda(
    const double* eta,  // AoS, row p at eta[3*p : 3*p+3]
    const int* holes,
    const long long n_mks,
    const long long nx,
    const long long ny,
    const long long nz,
    const double* domain_array,
    double* box_out)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_mks) return;

    long long n_boxes_total = (nx + 2) * (ny + 2) * (nz + 2);
    long long n_box;

    if (holes[p]) {
        n_box = n_boxes_total;
    } else {
        long long a = find_box_dev(eta[3 * p], eta[3 * p + 1], eta[3 * p + 2], nx, ny, nz, domain_array);
        n_box = (a >= n_boxes_total || a < 0) ? n_boxes_total : a;
    }

    box_out[p] = (double) n_box;
}

extern "C" __global__
void assign_particles_to_boxes_cuda(
    const double* box_id,
    const int* holes,
    const long long n_mks,
    int* boxes,
    int* next_index,
    const long long box_cols)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_mks) return;
    if (holes[p]) return;

    int a = (int) box_id[p];
    int slot = atomicAdd(&next_index[a], 1);
    if (slot < box_cols) {
        boxes[(long long) a * box_cols + slot] = (int) p;
    }
}
"""

_assign_box_kernel = None
_assign_particles_kernel = None


def _get_assign_box_kernel():
    global _assign_box_kernel
    if _assign_box_kernel is None:
        import cupy as cp

        _assign_box_kernel = cp.RawKernel(_SORT_SRC, "assign_box_to_each_particle_cuda")
    return _assign_box_kernel


def _get_assign_particles_kernel():
    global _assign_particles_kernel
    if _assign_particles_kernel is None:
        import cupy as cp

        _assign_particles_kernel = cp.RawKernel(_SORT_SRC, "assign_particles_to_boxes_cuda")
    return _assign_particles_kernel


def assign_box_to_each_particle_gpu(
    markers,
    holes,
    nx,
    ny,
    nz,
    domain_array,
    box_index: int = -2,
):
    """GPU port of :func:`~struphy.pic.sorting_kernels.assign_box_to_each_particle`.

    ``markers`` and ``holes`` are host arrays (see module docstring); only the
    logical-position columns and the box column are round-tripped through the
    device, not the full marker rows.
    """
    import cupy as cp

    n_mks, n_cols = markers.shape
    box_col = n_cols + box_index

    # markers[:, :3] is a strided view (stride n_cols) of the marker block;
    # ascontiguousarray on the host packs it into one AoS (n_mks, 3) buffer
    # matching the kernel's eta[3*p:3*p+3] layout, transferred in a single
    # H2D copy instead of three (RawKernel reads the raw device pointer
    # ignoring strides, so a per-axis strided view can't be passed directly --
    # see sph_eval_kernels_cuda.py's meshgrid contiguity fix for the same
    # failure mode).
    dev_eta = cp.asarray(np.ascontiguousarray(markers[:, :3]), dtype=cp.float64)
    dev_holes = cp.asarray(np.ascontiguousarray(holes), dtype=cp.int32)
    dev_domain = cp.asarray(domain_array, dtype=cp.float64)
    dev_box = cp.empty(n_mks, dtype=cp.float64)

    threads = 256
    blocks = (n_mks + threads - 1) // threads
    _get_assign_box_kernel()(
        (blocks,),
        (threads,),
        (
            dev_eta,
            dev_holes,
            n_mks,
            int(nx),
            int(ny),
            int(nz),
            dev_domain,
            dev_box,
        ),
    )

    markers[:, box_col] = cp.asnumpy(dev_box)


def assign_particles_to_boxes_gpu(
    markers,
    holes,
    boxes,
    next_index,
    box_index: int = -2,
):
    """GPU port of :func:`~struphy.pic.sorting_kernels.assign_particles_to_boxes`.

    Fills ``boxes``/``next_index`` via an atomic scatter instead of the CPU
    kernel's sequential counter -- see module docstring for why the resulting
    (unordered) box membership is equivalent.
    """
    import cupy as cp

    n_mks, n_cols = markers.shape
    box_col = n_cols + box_index
    n_box_rows, box_cols = boxes.shape

    dev_box_id = cp.asarray(np.ascontiguousarray(markers[:, box_col]), dtype=cp.float64)
    dev_holes = cp.asarray(np.ascontiguousarray(holes), dtype=cp.int32)
    dev_boxes = cp.full((n_box_rows, box_cols), -1, dtype=cp.int32)
    dev_next_index = cp.zeros(n_box_rows, dtype=cp.int32)

    threads = 256
    blocks = (n_mks + threads - 1) // threads
    _get_assign_particles_kernel()(
        (blocks,),
        (threads,),
        (
            dev_box_id,
            dev_holes,
            n_mks,
            dev_boxes,
            dev_next_index,
            box_cols,
        ),
    )

    boxes[:, :] = cp.asnumpy(dev_boxes)
    next_index[:] = cp.asnumpy(dev_next_index)
