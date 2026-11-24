"""Low-mode TAE driven by energetic ions (LinearMHDDriftkineticCC): 16 CPU MPI ranks vs 1 GPU, on the same kind of
Pitagora booster node.

This file defines the profiling case (the `ProfilingCase`) and submits its two runs of the same problem, the profiling
setup of `params_lowmode_tae.py` (`--setup profiling`: 16 x 16 x 4 elements, 20 steps, no accumulation filter, which
runs on the host only) with `--Np` energetic-ion markers:

- CPU: 16 MPI ranks with one thread each, NumPy backend. The grid is decomposed along eta1 and eta2 (4 x 4 ranks,
  4 elements each); cubic splines need at least 3 elements per rank, so e.g. 28 ranks (7 x 4) do not fit;
- GPU: 1 MPI rank with one thread on one H100, CuPy backend.

Both runs use the `pitagora_booster` preset, so they run on identical hardware; the CPU run does not touch the GPUs.
Without a batch system (or with STRUPHY_PROFILING_LOCAL=1) the runs execute directly on this machine, where the
16-rank run is skipped unless there are that many cores. `finalize_run` then packages (and with `--upload` uploads)
each run as soon as its own job finishes. Each generated script runs the simulation by invoking
`params_lowmode_tae.py` directly (its `__main__` block is the worker) and then writes the scope-profiler report and
the cost summary with `pproc_lowmode_tae.py --save-results`.

The time step is a mix of particle work (guiding-centre pushers and current-coupling accumulations, all CUDA kernels
on the CuPy backend) and FEEC solves (the Schur solve of ShearAlfvenCurrentCoupling5D and the MHD propagators), whose
cost does not depend on the number of markers; `--Np` sets the balance between the two.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    parser.add_argument(
        "--language",
        default="fortran",
        choices=["fortran", "c"],
        help="Pyccel language to compile the Struphy kernels with.",
    )
    parser.add_argument(
        "--compiler",
        default="GNU",
        help='Pyccel compiler family ("GNU", "intel", "PGI", "nvidia" or "LLVM").',
    )
    parser.add_argument("--Np", type=int, default=1_000_000, help="Number of energetic-ion markers.")
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    param_flags = ["--setup", "profiling", "--save-results", "--Np", str(args.Np)]

    profiling_case = ProfilingCase(
        # one label per size: runs of different sizes are separate cases in the profiling-data repo
        label=f"linear_mhd_driftkinetic_cc_lowmode_tae_cpu_vs_gpu_Np{args.Np}",
        name="LinearMHDDriftkineticCC low-mode TAE: 16 CPU ranks vs 1 GPU",
        description=f"The LinearMHDDriftkineticCC model on the low-mode TAE (HollowTorus, 16 x 16 x 4 elements, cubic splines, {args.Np / 1e6:.2g}M drift-kinetic energetic-ion markers, delta-f, no filter, 20 steps) on 16 MPI ranks x 1 thread of a booster node (NumPy) vs 1 MPI rank x 1 thread on 1 H100 GPU (CuPy).",
        physics_problem="Toroidal Alfven eigenmode driven unstable by energetic ions (linear MHD with drift-kinetic current coupling).",
        struphy_model_used="LinearMHDDriftkineticCC",
        params_source=script_dir / "params_lowmode_tae.py",
        pproc_source=script_dir / "pproc_lowmode_tae.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # CPU: 16 MPI ranks x 1 thread on a booster node, NumPy backend (4 x 4 decomposition of the 16 x 16 x 4 grid)
    profiling_case.launch(16, num_nodes=1, param_flags=param_flags, backend="numpy", slurm_preset="pitagora_booster")
    # GPU: 1 MPI rank x 1 thread on one H100 (the rank binds to GPU 0 of the node), CuPy backend
    profiling_case.launch(1, num_nodes=1, param_flags=param_flags, backend="cupy", slurm_preset="pitagora_booster")

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
