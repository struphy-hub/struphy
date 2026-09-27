"""
Standalone benchmark (not a pytest test) that measures the OpenMP speedup of
``struphy.pic.pushing.pusher_kernels.push_vxb_analytic`` as a function of the
number of OpenMP threads.

Reuses the same worker subprocess as :mod:`test_pusher_openmp_speedup`, so the
numbers reported here are directly comparable to what that test checks.

Usage::

    python src/struphy/pic/tests/bench_openmp_speedup.py [--threads 1,2,4,8,16,32] [--repeats 3]

Thread pinning (``OMP_PROC_BIND=close``, ``OMP_PLACES=cores``) matters a lot
for this memory-bandwidth-bound kernel on multi-socket/NUMA machines; without
it more threads can be *slower* than fewer due to remote-memory-access
effects. To keep the benchmark meaningful on such machines, threads are also
confined to a single NUMA node (via ``taskset``) whenever the machine has
more than one and ``taskset`` is available, matching how the OpenMP speedup
test itself is intended to be interpreted.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys

WORKER = os.path.join(os.path.dirname(__file__), "_bench_push_vxb_analytic.py")

N_ELEMENTS = 24
DEGREE = 3
PPC = 50


def _numa_node0_cpu_range() -> str | None:
    path = "/sys/devices/system/node/node0/cpulist"
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read().strip()


def _median_runtime(n_threads: int, repeats: int, cpu_range: str | None) -> float:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    base_cmd = [sys.executable, WORKER, str(N_ELEMENTS), str(DEGREE), str(PPC)]
    if cpu_range is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", cpu_range] + base_cmd
    else:
        cmd = base_cmd

    times = []
    for _ in range(repeats):
        out = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        times.append(float(out.stdout.strip().splitlines()[-1]))

    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8,16,32",
        help="comma-separated list of OMP_NUM_THREADS values to benchmark (default: 1,2,4,8,16,32)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="number of repeats per thread count (default: 3)")
    args = parser.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",")]
    cpu_range = _numa_node0_cpu_range()

    print(f"Benchmarking push_vxb_analytic: num_elements={N_ELEMENTS}, degree={DEGREE}, ppc={PPC}")
    if cpu_range is not None:
        print(f"Pinning to NUMA node 0 (cpus {cpu_range}) via taskset, OMP_PROC_BIND=close, OMP_PLACES=cores")
    else:
        print("No NUMA pinning available, using OMP_PROC_BIND=close, OMP_PLACES=cores")
    print()

    results = {}
    for n_threads in thread_counts:
        t = _median_runtime(n_threads, args.repeats, cpu_range)
        results[n_threads] = t

    t_serial = results[thread_counts[0]]
    print(f"{'threads':>8} {'time (ms)':>12} {'speedup':>10}")
    for n_threads in thread_counts:
        t = results[n_threads]
        speedup = t_serial / t
        print(f"{n_threads:>8} {t * 1e3:>12.2f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
