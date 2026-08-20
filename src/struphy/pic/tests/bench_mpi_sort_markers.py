"""Standalone MPI benchmark for :meth:`Particles.mpi_sort_markers`.

The benchmark does not construct or run a Struphy simulation.  It creates a
mesh-less :class:`~struphy.pic.particles.Particles6D` instance, redistributes
uniformly placed markers, and measures the complete marker exchange.  Marker
IDs are checked after every measured call, so this is useful while changing
the implementation of ``mpi_sort_markers``.

Run with MPI (one process per rank/GPU), for example::

    mpiexec -n 4 python src/struphy/pic/tests/bench_mpi_sort_markers.py \
        --sizes 10000,100000,1000000 --repeats 10

For the NumPy baseline, set ``ARRAY_BACKEND`` before Python imports
``cunumpy`` (the benchmark does not change the backend at runtime)::

    ARRAY_BACKEND=numpy python src/struphy/pic/tests/bench_mpi_sort_markers.py \
        --sizes 100000,1000000 --repeats 10

For CuPy, use instead (one rank per GPU; device binding and the MPI opt-in for the
CuPy backend are handled automatically, see below)::

    ARRAY_BACKEND=cupy mpiexec -n 4 python src/struphy/pic/tests/bench_mpi_sort_markers.py

The reported time is the maximum wall time over ranks (the useful MPI step
time).  GPU streams are synchronized around the timed region.
"""

import argparse
import os
import time

# cunumpy does not import feectools.ddm.mpi (verified), so this is safe to import
# first and do CUDA-only, MPI-independent setup (device binding, the MPI opt-in env
# var below) before struphy -- which does transitively import feectools.ddm.mpi as
# part of its own __init__ -- gets imported next.
import cunumpy as xp
import numpy as np

# Under CuPy with more than one MPI rank per node, every rank must bind to its own
# GPU -- cunumpy defaults to device 0, so without this every rank on a node would
# contend for the same GPU instead of getting one each (same pattern as the
# profiling/examples/*/params_*_scaling.py cases). SLURM_LOCALID (the rank's index
# within its node) is set by srun before this process starts, so it works without
# MPI being initialized yet. Falls back to device 0 outside SLURM.
if xp.cupy_backend:
    xp.set_device(int(os.environ.get("SLURM_LOCALID", 0)))

    # feectools.ddm.mpi disables MPI by default on the CuPy backend; this benchmark
    # is specifically meant to exercise the real multi-rank exchange, so opt back in.
    # Must be set before struphy (hence feectools.ddm.mpi) is imported below --
    # feectools.ddm.mpi reads this env var once, at its own import time, so setting
    # it any later would silently have no effect.
    os.environ.setdefault("FEECTOOLS_ENABLE_MPI", "1")

# struphy must be imported before feectools.ddm.mpi is imported anywhere else:
# struphy/__init__.py sets MPI4PY_RC_THREAD_LEVEL=funneled and disables hcoll
# *before* mpi4py's MPI_Init_thread runs, which is required to avoid a
# hcoll/Alltoallv segfault on this cluster (see the comment there). struphy itself
# imports feectools.ddm.mpi as part of this same import, so this line satisfies
# both ordering requirements at once.
from feectools.ddm.mpi import mpi as MPI

from struphy import BoundaryParameters, LoadingParameters, SortingParameters
from struphy.pic.particles import Particles6D


def _sync_device():
    if xp.cupy_backend:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def _host(a):
    """Convert either backend's array to a NumPy array."""
    if xp.cupy_backend:
        import cupy as cp

        return cp.asnumpy(a)
    return np.asarray(a)


def _id_signature(comm, local_ids):
    """Return global count/sum/sum-of-squares using scalar MPI reductions.

    ``feectools`` supplies a singleton ``MockComm`` when the script is run
    without ``mpiexec``; unlike mpi4py, its lowercase ``allgather`` does not
    return a Python list.  Scalar reductions work for both communicators and
    avoid any CuPy/NumPy array dispatch in the validation path.

    The sum-of-squares must be accumulated as Python (arbitrary-precision)
    ints, not float64: at Np in the millions the sum of squared IDs reaches
    ~1e17-1e20, past float64's exact-integer range (2^53 ~= 9e15), so
    summing the same values in a different order -- which is exactly what
    happens here, since mpi_sort_markers regroups which rank holds which
    IDs -- rounds to a different (both inexact) result and produces a
    false-positive "lost or duplicated" mismatch despite the exchange being
    exact. Python ints have no such limit and integer addition is exactly
    associative, so this is order-independent regardless of Np.
    """
    ids = _host(local_ids).astype(np.int64, copy=False)
    local_sumsq = sum(int(v) * int(v) for v in ids.tolist())
    if comm.Get_size() == 1:
        return int(ids.size), int(ids.sum(dtype=np.int64)), local_sumsq
    return (
        int(comm.allreduce(int(ids.size), op=MPI.SUM)),
        int(comm.allreduce(int(ids.sum(dtype=np.int64)), op=MPI.SUM)),
        int(comm.allreduce(local_sumsq, op=MPI.SUM)),
    )


def _global_max(comm, value):
    return value if comm.Get_size() == 1 else comm.allreduce(value, op=MPI.MAX)


def _randomize_positions(particles, seed):
    # Keep marker rows/IDs intact; only move valid markers to create traffic.
    xp.random.seed(seed)
    # ``n_mks_loc`` is a backend scalar under CuPy; shape tuples require a
    # native Python integer.
    n_local = int(particles.n_mks_loc)
    particles.positions = xp.random.random((n_local, 3))


def _check(particles, comm, expected_ids):
    got_ids = _id_signature(comm, particles.markers[particles.valid_mks, -1])
    if got_ids != expected_ids:
        raise AssertionError("mpi_sort_markers lost or duplicated marker IDs")

    # Check ownership on the host, avoiding a device-to-host synchronization
    # inside the timed region.  Every real marker must be strictly inside its
    # rank's three-dimensional subdomain.
    positions = _host(particles.positions)
    bounds = np.asarray(particles.domain_array[particles.mpi_rank]).reshape(3, 3)
    if positions.size:
        inside = np.all((positions > bounds[:, 0]) & (positions < bounds[:, 1]), axis=1)
        if not np.all(inside):
            raise AssertionError("mpi_sort_markers left markers on the wrong rank")


def _one_size(comm, np_global, repeats, seed, check):
    loading = LoadingParameters(
        Np=np_global,
        seed=seed,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        spatial="uniform",
    )
    sorting = SortingParameters(boxes_per_dim=None)
    boundary = BoundaryParameters(bc=["periodic", "periodic", "periodic"])
    particles = Particles6D(
        comm_world=comm,
        loading_params=loading,
        sorting_params=sorting,
        boundary_params=boundary,
    )
    # Loading is intentionally unsorted: this gives every rank a global,
    # uniform sample that must be exchanged by the first sort.
    particles.draw_markers(sort=False)
    comm.Barrier()
    _sync_device()

    expected_ids = _id_signature(comm, particles.markers[particles.valid_mks, -1])

    def prepare(i):
        _randomize_positions(particles, seed + 1009 * (i + 1) + comm.Get_rank())
        _sync_device()
        comm.Barrier()

    # Warm up MPI requests, allocation paths, and CuPy kernels.
    prepare(-1)
    particles.mpi_sort_markers(apply_bc=False, do_test=False)
    _sync_device()
    comm.Barrier()
    if check:
        _check(particles, comm, expected_ids)

    samples = []
    for i in range(repeats):
        prepare(i)
        _sync_device()
        t0 = time.perf_counter()
        particles.mpi_sort_markers(apply_bc=False, do_test=False)
        _sync_device()
        elapsed = time.perf_counter() - t0
        samples.append(_global_max(comm, elapsed))
        if check:
            _check(particles, comm, expected_ids)

    return float(np.median(samples)), float(np.percentile(samples, q=95))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="10000,100000,1000000", help="global marker counts to sweep")
    parser.add_argument("--repeats", type=int, default=10, help="measured calls per size")
    parser.add_argument("--seed", type=int, default=1607)
    parser.add_argument("--no-check", action="store_true", help="skip ID/ownership checks after each call")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    sizes = [int(value) for value in args.sizes.split(",") if value]
    if args.repeats < 1 or any(size < comm.Get_size() for size in sizes):
        raise ValueError("repeats must be positive and every size must be at least the MPI rank count")

    if comm.Get_rank() == 0:
        backend = "cupy" if xp.cupy_backend else "numpy"
        print(f"mpi_sort_markers benchmark: ranks={comm.Get_size()}, backend={backend}")
        print(f"{'Np (global)':>14} {'median [ms]':>14} {'p95 [ms]':>14} {'markers/s':>16}")

    for size in sizes:
        median, p95 = _one_size(comm, size, args.repeats, args.seed, not args.no_check)
        if comm.Get_rank() == 0:
            rate = size / median if median else float("inf")
            print(f"{size:>14d} {median * 1e3:>14.3f} {p95 * 1e3:>14.3f} {rate:>16.3e}")


if __name__ == "__main__":
    main()
