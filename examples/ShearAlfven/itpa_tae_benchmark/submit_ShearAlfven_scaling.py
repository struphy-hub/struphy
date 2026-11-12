"""ITPA TAE (ShearAlfven): strong scaling on one node, per space of the magnetic field and preconditioner.

This file defines the profiling case (the `ProfilingCase`) of one variant (space of the magnetic field and
preconditioner of the shear Alfven solve) and submits its runs on 4, 8, 16, 32 and 64 MPI ranks, all on a single
node (e.g. Raven, 72 cores). For each rank count, `ProfilingCase.launch` builds and submits a SLURM script (using
`clusters.SLURM_PRESETS` by default), or, without a batch system, runs directly on this machine, where the rank counts
that exceed the local cores are skipped. `finalize_run` then packages (and with `--upload` uploads) each run as soon as
its own job finishes. Each generated script runs the simulation by invoking `params_ShearAlfven.py` directly (its
`__main__` block is the worker) with `--setup profiling` (`--steps` time steps, default 50, solver iterations
logged), and then writes the scope-profiler report and the cost summary with `pproc_ShearAlfven.py --save-results`.

All variants run on the same grid (`--num-elements`, default 24 x 88 x 8, cubic splines), the resolution at which the
H(div) version has the right TAE frequency (see the description of `params_ShearAlfven.py`), with the time step 0.2.
The variant is chosen with `--variant`; run this script once per variant. Each variant is a separate case in the
profiling-data repo.

Cost per step on 4 ranks (24 x 88 x 8, dt = 0.2, local workstation): hcurl_hiptmair_xu 0.76 s (9 PCG iterations),
hdiv_stiffness 1.3 s (15), hcurl_mass 6.1 s (150), hdiv_mass 11.5 s (210), hcurl_projected_mass 12.6 s (110),
hdiv_multigrid 360 s (740), hdiv_toroidal_fourier: 2 iterations, but a setup of about 10 min and dense poloidal
factorizations on every rank (about 5-6 GB per rank, extrapolated from 12 x 44 x 8). WALLTIME holds the default SLURM
time limit per variant (the 4-rank run with 50 steps, with a margin); `--time` overrides it.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (label suffix, name, description of the solver, flags of the params file)
VARIANTS = {
    "hdiv_stiffness": (
        "hdiv_stiffness",
        "B in H(div), Kronecker preconditioner",
        "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by "
        "the Kronecker (fast diagonalization) preconditioner of the Schur operator with the E x B auxiliary space "
        "correction (StiffnessPreconditioner)",
        ["--b-space", "Hdiv", "--precond", "StiffnessPreconditioner"],
    ),
    "hdiv_toroidal_fourier": (
        "hdiv_toroidal_fourier",
        "B in H(div), toroidal Fourier preconditioner",
        "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by "
        "the exact inverse of the Schur operator of the axisymmetric equilibrium (FFT along eta3, one dense poloidal "
        "factorization per toroidal mode, ToroidalFourierPreconditioner)",
        ["--b-space", "Hdiv", "--precond", "ToroidalFourierPreconditioner"],
    ),
    "hdiv_mass": (
        "hdiv_mass",
        "B in H(div), mass matrix preconditioner",
        "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by "
        "the Kronecker approximation of the inverse mass matrix (MassMatrixPreconditioner)",
        ["--b-space", "Hdiv", "--precond", "MassMatrixPreconditioner"],
    ),
    "hdiv_multigrid": (
        "hdiv_multigrid",
        "B in H(div), multigrid preconditioner",
        "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by "
        "a geometric multigrid V-cycle of the Schur operator (MultiGrid, experimental)",
        ["--b-space", "Hdiv", "--precond", "MultiGrid"],
    ),
    "hcurl_hiptmair_xu": (
        "hcurl_hiptmair_xu",
        "B in H(curl), Hiptmair-Xu preconditioner",
        "magnetic field in H(curl), implicit curl-curl solve for the magnetic field (numerically damped), PCG "
        "preconditioned by Hiptmair-Xu (HiptmairXu)",
        ["--b-space", "Hcurl", "--precond", "HiptmairXu"],
    ),
    "hcurl_mass": (
        "hcurl_mass",
        "B in H(curl), mass matrix preconditioner",
        "magnetic field in H(curl), implicit curl-curl solve for the magnetic field (numerically damped), PCG "
        "preconditioned by the Kronecker approximation of the inverse mass matrix (MassMatrixPreconditioner)",
        ["--b-space", "Hcurl", "--precond", "MassMatrixPreconditioner"],
    ),
    "hcurl_projected_mass": (
        "hcurl_projected_mass",
        "B in H(curl), projected, mass matrix preconditioner",
        "magnetic field in H(curl) with the projected Lorentz force (exactly energy conserving), implicit curl-curl "
        "solve with an inner mass matrix solve, PCG preconditioned by the Kronecker approximation of the inverse "
        "mass matrix (MassMatrixPreconditioner)",
        ["--b-space", "Hcurl", "--projected", "--precond", "MassMatrixPreconditioner"],
    ),
}

# default SLURM time limit per variant, for the slowest (4-rank) run of 50 steps
WALLTIME = {
    "hdiv_stiffness": "00:15:00",
    "hdiv_toroidal_fourier": "01:00:00",
    "hdiv_mass": "00:30:00",
    "hdiv_multigrid": "08:00:00",
    "hcurl_hiptmair_xu": "00:15:00",
    "hcurl_mass": "00:20:00",
    "hcurl_projected_mass": "00:30:00",
}

# MPI ranks of the strong scaling, all on one node
RANKS = (4, 8, 16, 32, 64)


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS),
        help="Space of the magnetic field and preconditioner of the runs.",
    )
    parser.add_argument(
        "--num-elements",
        type=int,
        nargs=3,
        default=[24, 88, 8],
        help="Number of elements per direction (default 24 88 8).",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=list(RANKS),
        help=f"MPI rank counts of the scaling (default {' '.join(map(str, RANKS))}).",
    )
    parser.add_argument("--steps", type=int, default=50, help="Time steps per run (default 50).")
    parser.add_argument(
        "--time",
        default=None,
        help="SLURM time limit of each run (default: WALLTIME of the variant, for 50 steps).",
    )
    parser.add_argument(
        "--slurm-preset",
        default=None,
        help="Key of clusters.SLURM_PRESETS to submit with (default: the preset of the detected cluster).",
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
    args = parser.parse_args()
    suffix, name, solver, variant_flags = VARIANTS[args.variant]
    assert set(WALLTIME) == set(VARIANTS)
    grid = " x ".join(map(str, args.num_elements))
    param_flags = [
        "--setup",
        "profiling",
        "--save-results",
        "--profiling-steps",
        str(args.steps),
        "--num-elements",
        *map(str, args.num_elements),
        *variant_flags,
    ]

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=f"shear_alfven_itpa_tae_strong_scaling_{suffix}",
        name=f"ShearAlfven ITPA TAE strong scaling, {name}",
        description=f"Strong scaling on one node of the ShearAlfven model on the ITPA TAE benchmark (HollowTorus with "
        f"tor_period 6, AdhocTorus, {grid} elements, cubic splines, dt = 0.2): {solver}.",
        physics_problem="Toroidal Alfven eigenmode (TAE) in a large-aspect-ratio tokamak (ITPA benchmark).",
        struphy_model_used="ShearAlfven",
        params_source=script_dir / "params_ShearAlfven.py",
        pproc_source=script_dir / "pproc_ShearAlfven.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count, all on a single node
    for num_tasks in args.ranks:
        profiling_case.launch(
            num_tasks,
            num_nodes=1,
            param_flags=param_flags,
            slurm_preset=args.slurm_preset,
            slurm_options={"time": args.time or WALLTIME[args.variant]},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
