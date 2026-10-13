"""Vlasov-Maxwell Weibel instability profiling case.

This file defines the Vlasov-Maxwell Weibel instability profiling case (the `ProfilingCase`) and submits
it: for each rank count, `ProfilingCase.launch` builds and submits a SLURM script
(using `clusters.SLURM_PRESETS` by default), or, without a batch system, runs
directly on this machine. `finalize_run` then packages and uploads each run as
soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_vlasov_maxwell.py`
directly (its `__main__` block is the worker).

The run is profiled with scope-profiler. `--backend cupy` runs it on the GPU with CUDA-event
timing per region (default: the backend stored with `struphy backend set`); the job script exports
the choice as CUNUMPY_BACKEND. `--ppc`, `--num-elements` and `--steps` are forwarded to `params_vlasov_maxwell.py`
(defaults: 1000 markers per cell on 32 cells, 20 steps).
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/, two levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    parser.add_argument(
        "--backend",
        default=None,
        choices=["numpy", "cupy"],
        help="cunumpy backend to run on (default: 'struphy backend get'); cupy records CUDA-event GPU times.",
    )
    parser.add_argument("--ppc", type=int, default=1000, help="Markers per cell.")
    parser.add_argument("--num-elements", type=int, default=32, help="Cells in x.")
    parser.add_argument("--steps", type=int, default=20, help="Number of time steps.")
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "weibel_instability"

    profiling_case = ProfilingCase(
        label="vlasov_maxwell_weibel_instability",
        name="Vlasov-Maxwell Weibel instability",
        description="Strong scaling of the VlasovMaxwellOneSpecies model on the Weibel instability (Kraus et al. 2017, Sec. 6.2: anisotropic Maxwellian, seed field B3 = -1e-4 cos(k x)), on the NumPy or CuPy backend.",
        physics_problem="Weibel instability driven by a temperature anisotropy in a collisionless electron plasma.",
        struphy_model_used="VlasovMaxwellOneSpecies",
        params_source=params_dir / "params_vlasov_maxwell.py",
        language=args.language,
        compiler=args.compiler,
        backend=args.backend,
        upload=args.upload,
    )

    param_flags = [
        "--ppc",
        str(args.ppc),
        "--num-elements",
        str(args.num_elements),
        "--steps",
        str(args.steps),
    ]

    # Launch one run per rank count
    for num_tasks in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks, param_flags=param_flags)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
