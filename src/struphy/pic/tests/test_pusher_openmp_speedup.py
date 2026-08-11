"""
Demonstrates and checks the OpenMP speedup of the Vlasov pusher kernel
``struphy.pic.pushing.pusher_kernels.push_vxb_analytic``.

The kernel loops over all markers (particles) and is embarrassingly parallel,
so it is a good candidate for the ``#$ omp parallel``/``#$ omp for`` pragmas
in :mod:`struphy.pic.pushing.pusher_kernels`.

This test only produces a meaningful result if Struphy's PIC kernels were
compiled with OpenMP support, e.g.::

    struphy compile --openmp

Since the whole point of the test is to measure wall-clock time under a
varying number of OpenMP threads, the number of threads must be fixed
*before* the pyccel/Fortran extension is loaded (the OpenMP runtime reads
``OMP_NUM_THREADS`` once at start-up). Each measurement is therefore
performed in a fresh subprocess with the desired ``OMP_NUM_THREADS``.
"""

import logging
import os
import statistics
import subprocess
import sys

import pytest

logger = logging.getLogger("struphy")

WORKER = os.path.join(os.path.dirname(__file__), "_bench_push_vxb_analytic.py")

# how many physical particles to push; large enough that the parallel
# region dominates the run time of the worker script
N_ELEMENTS = [24, 24, 24]
DEGREE = [3, 3, 3]
PPC = 50

N_THREADS_SERIAL = 1
N_THREADS_PARALLEL = 4
N_REPEATS = 3


def _median_runtime(n_threads: int) -> float:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(n_threads)
    # thread pinning matters a lot for this memory-bandwidth-bound kernel:
    # unbound threads may migrate/spread across NUMA nodes and make more
    # threads *slower* than fewer.
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    times = []
    for _ in range(N_REPEATS):
        out = subprocess.run(
            [sys.executable, WORKER, str(N_ELEMENTS[0]), str(DEGREE[0]), str(PPC)],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        times.append(float(out.stdout.strip().splitlines()[-1]))

    return statistics.median(times)


@pytest.mark.mpi_skip
@pytest.mark.skipif(
    (os.cpu_count() or 1) < N_THREADS_PARALLEL,
    reason=f"need at least {N_THREADS_PARALLEL} CPUs to demonstrate OpenMP speedup",
)
def test_push_vxb_analytic_openmp_speedup():
    """Compares the wall time of ``push_vxb_analytic`` with 1 vs. several OpenMP threads.

    The pusher kernel is memory-bandwidth bound (particle gather/scatter of
    spline coefficients), so the speedup is far from linear in the number of
    threads; we only assert a modest, robust improvement.
    """

    t_serial = _median_runtime(N_THREADS_SERIAL)
    t_parallel = _median_runtime(N_THREADS_PARALLEL)

    speedup = t_serial / t_parallel

    logger.info(f"push_vxb_analytic: {N_THREADS_SERIAL} thread(s): {t_serial * 1e3:.1f} ms")
    logger.info(f"push_vxb_analytic: {N_THREADS_PARALLEL} thread(s): {t_parallel * 1e3:.1f} ms")
    logger.info(f"push_vxb_analytic: speedup = {speedup:.2f}x")

    assert speedup > 1.05, (
        f"Expected push_vxb_analytic to be faster with {N_THREADS_PARALLEL} OpenMP threads "
        f"than with 1, got speedup={speedup:.2f}x "
        f"({t_serial * 1e3:.1f} ms -> {t_parallel * 1e3:.1f} ms). "
        "Make sure Struphy was compiled with OpenMP support (`struphy compile --openmp`)."
    )


if __name__ == "__main__":
    test_push_vxb_analytic_openmp_speedup()
