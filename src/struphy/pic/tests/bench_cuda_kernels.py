"""
Standalone benchmark (not a pytest test) that measures the NumPy-vs-CuPy
speedup of the CUDA ``RawKernel``-ported particle operations:

* ``push_eta``            -- :func:`~struphy.pic.pushing.pusher_kernels_cuda.push_eta_rk_periodic_gpu`
* ``push_v``               -- :func:`~struphy.pic.pushing.pusher_kernels_cuda.push_v_with_efield_cuboid_gpu`
* ``eval_density_flat``     -- :func:`~struphy.pic.sph_eval_kernels_cuda.box_based_evaluation_flat_gpu`
* ``eval_density_mesh``     -- :func:`~struphy.pic.sph_eval_kernels_cuda.box_based_evaluation_meshgrid_gpu`

across a marker-count (``Np``) sweep, one subprocess per (backend, op, Np)
combination (``ARRAY_BACKEND`` is read once at import time by ``cunumpy`` and
can't be changed within a running process -- see the worker script,
``_bench_cuda_kernels_worker.py``).

Usage::

    python src/struphy/pic/tests/bench_cuda_kernels.py [--ops push_eta,push_v,eval_density_flat,eval_density_mesh] \\
        [--sizes 2000,20000,200000] [--repeats 5]

Requires a CUDA-capable GPU and the CuPy backend to be installed; the NumPy
side of each comparison runs regardless.
"""

import argparse
import os
import subprocess
import sys

WORKER = os.path.join(os.path.dirname(__file__), "_bench_cuda_kernels_worker.py")

ALL_OPS = ("push_eta", "push_v", "eval_density_flat", "eval_density_mesh")


def _median_runtime(backend: str, op: str, Np: int, n_reps: int) -> float:
    env = dict(os.environ)
    env["ARRAY_BACKEND"] = backend

    cmd = [sys.executable, WORKER, op, str(Np), str(n_reps)]
    out = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(out.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ops",
        type=str,
        default=",".join(ALL_OPS),
        help=f"comma-separated list of operations to benchmark (default: all of {ALL_OPS})",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="2000,20000,200000",
        help="comma-separated list of Np values to sweep (default: 2000,20000,200000)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="repeats per (op, Np) point, median is reported")
    args = parser.parse_args()

    ops = args.ops.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    for op in ops:
        print(f"\n=== {op} ===")
        print(f"{'Np':>10} {'numpy [ms]':>12} {'cupy [ms]':>12} {'speedup':>10}")
        for Np in sizes:
            t_numpy = _median_runtime("numpy", op, Np, args.repeats)
            t_cupy = _median_runtime("cupy", op, Np, args.repeats)
            speedup = t_numpy / t_cupy
            print(f"{Np:>10} {t_numpy * 1e3:>12.3f} {t_cupy * 1e3:>12.3f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
