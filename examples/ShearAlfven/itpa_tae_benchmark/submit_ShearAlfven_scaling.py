"""ITPA TAE (ShearAlfven): strong scaling on one node, per space of the magnetic field, preconditioner and grid.

This file defines the profiling case (the `ProfilingCase`) of one variant (space of the magnetic field, preconditioner
of the shear Alfven solve and grid) and submits its runs on 4, 8, 16, 32 and 64 MPI ranks, all on a single node (a
Raven CPU node, 72 cores, with the `raven` preset of `clusters.SLURM_PRESETS`, which allocates the whole node also
for the small rank counts). For each rank count, `ProfilingCase.launch` builds and submits a SLURM script, or, without
a batch system, runs directly on this machine, where the rank counts that exceed the local cores are skipped.
`finalize_run` then packages (and with `--upload` uploads) each run as soon as its own job finishes. Each generated
script runs the simulation by invoking `params_ShearAlfven.py` directly (its `__main__` block is the worker) with
`--setup profiling` (`--steps` time steps, default 100, solver iterations logged), and then writes the scope-profiler
report and the cost summary (time per step, PCG iterations per step) with `pproc_ShearAlfven.py --save-results`.

The variant is chosen with `--variant`; run this script once per variant, or all variants at once with
`sweep_ShearAlfven_scaling.py`. Each variant is a separate case in the profiling-data repo. All variants use cubic
splines and the time step 0.2. The H(div) variants run on 24 x 88 x 8 elements, the resolution at which the H(div)
version has the right TAE frequency (see the description of `params_ShearAlfven.py`); the H(curl) variants on the same
grid and on 24 x 44 x 8, the default grid of the example, on which H(curl) already has the right frequency.

Cost per step on 4 ranks (24 x 88 x 8, local workstation): hcurl_hiptmair_xu 0.76 s (9 PCG iterations),
hdiv_stiffness 1.3 s (15), hcurl_mass 6.1 s (150), hdiv_mass 11.5 s (210), hcurl_projected_mass 12.6 s (110).
hdiv_toroidal_fourier needs 2 iterations, but its setup takes about 10 min and holds dense poloidal factorizations on
every rank (about 5-6 GB per rank, extrapolated from 12 x 44 x 8), so it runs on at most 32 ranks. The geometric
multigrid preconditioner (6 min and 740 iterations per step) is left out. The `walltime` of a variant is the SLURM time
limit of its runs (the 4-rank run of 100 steps, with a margin), scaled with `--steps`; `--time` overrides it.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# MPI ranks of the strong scaling, all on one node
RANKS = (4, 8, 16, 32, 64)

FINE, COARSE = (24, 88, 8), (24, 44, 8)

# the solvers: key -> (name, description, flags of the params file)
HDIV = "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by"
HCURL = "magnetic field in H(curl), implicit curl-curl solve for the magnetic field (numerically damped), PCG preconditioned by"
MASS = "the Kronecker approximation of the inverse mass matrix (MassMatrixPreconditioner)"
SOLVERS = {
    "hdiv_stiffness": (
        "B in H(div), Kronecker preconditioner",
        f"{HDIV} the Kronecker (fast diagonalization) preconditioner of the Schur operator with the E x B auxiliary "
        "space correction (StiffnessPreconditioner)",
        ["--b-space", "Hdiv", "--precond", "StiffnessPreconditioner"],
    ),
    "hdiv_toroidal_fourier": (
        "B in H(div), toroidal Fourier preconditioner",
        f"{HDIV} the exact inverse of the Schur operator of the axisymmetric equilibrium (FFT along eta3, one dense "
        "poloidal factorization per toroidal mode on every rank, ToroidalFourierPreconditioner)",
        ["--b-space", "Hdiv", "--precond", "ToroidalFourierPreconditioner"],
    ),
    "hdiv_mass": (
        "B in H(div), mass matrix preconditioner",
        f"{HDIV} {MASS}",
        ["--b-space", "Hdiv", "--precond", "MassMatrixPreconditioner"],
    ),
    "hcurl_hiptmair_xu": (
        "B in H(curl), Hiptmair-Xu preconditioner",
        f"{HCURL} Hiptmair-Xu (HiptmairXu)",
        ["--b-space", "Hcurl", "--precond", "HiptmairXu"],
    ),
    "hcurl_mass": (
        "B in H(curl), mass matrix preconditioner",
        f"{HCURL} {MASS}",
        ["--b-space", "Hcurl", "--precond", "MassMatrixPreconditioner"],
    ),
    "hcurl_projected_mass": (
        "B in H(curl), projected, mass matrix preconditioner",
        "magnetic field in H(curl) with the projected Lorentz force (exactly energy conserving), implicit curl-curl "
        f"solve with an inner mass matrix solve, PCG preconditioned by {MASS}",
        ["--b-space", "Hcurl", "--projected", "--precond", "MassMatrixPreconditioner"],
    ),
}


def _variant(solver, grid, walltime, ranks=RANKS):
    return {"solver": solver, "grid": grid, "walltime": walltime, "ranks": ranks}


# variant (key = label suffix) -> solver, grid, SLURM time limit (4-rank run of 100 steps), rank counts
VARIANTS = {
    "hdiv_stiffness_24x88x8": _variant("hdiv_stiffness", FINE, "00:15:00"),
    "hdiv_toroidal_fourier_24x88x8": _variant("hdiv_toroidal_fourier", FINE, "01:00:00", ranks=(4, 8, 16, 32)),
    "hdiv_mass_24x88x8": _variant("hdiv_mass", FINE, "00:45:00"),
    "hcurl_hiptmair_xu_24x88x8": _variant("hcurl_hiptmair_xu", FINE, "00:15:00"),
    "hcurl_mass_24x88x8": _variant("hcurl_mass", FINE, "00:30:00"),
    "hcurl_projected_mass_24x88x8": _variant("hcurl_projected_mass", FINE, "00:45:00"),
    "hcurl_hiptmair_xu_24x44x8": _variant("hcurl_hiptmair_xu", COARSE, "00:15:00"),
    "hcurl_mass_24x44x8": _variant("hcurl_mass", COARSE, "00:20:00"),
    "hcurl_projected_mass_24x44x8": _variant("hcurl_projected_mass", COARSE, "00:30:00"),
}


def scaled_walltime(walltime: str, factor: float) -> str:
    """The time limit ``walltime`` (HH:MM:SS) times ``factor`` (not below 15 min), as HH:MM:SS."""
    h, m, s = map(int, walltime.split(":"))
    seconds = max(15 * 60, round((3600 * h + 60 * m + s) * factor))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS),
        help="Space of the magnetic field, preconditioner and grid of the runs.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="MPI rank counts of the scaling (default: those of the variant, 4 8 16 32 64; up to 32 for "
        "hdiv_toroidal_fourier).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Time steps per run (default 100).")
    parser.add_argument(
        "--time",
        default=None,
        help="SLURM time limit of each run (default: the walltime of the variant, for 100 steps, scaled with --steps).",
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
    variant = VARIANTS[args.variant]
    name, solver, solver_flags = SOLVERS[variant["solver"]]
    grid = " x ".join(map(str, variant["grid"]))
    param_flags = [
        "--setup",
        "profiling",
        "--save-results",
        "--profiling-steps",
        str(args.steps),
        "--num-elements",
        *map(str, variant["grid"]),
        *solver_flags,
    ]

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=f"shear_alfven_itpa_tae_strong_scaling_{args.variant}",
        name=f"ShearAlfven ITPA TAE strong scaling, {name}, {grid}",
        description=f"Strong scaling on one node of the ShearAlfven model on the ITPA TAE benchmark (HollowTorus with "
        f"tor_period 6, AdhocTorus, {grid} elements, cubic splines, dt = 0.2, {args.steps} steps): {solver}.",
        physics_problem="Toroidal Alfven eigenmode (TAE) in a large-aspect-ratio tokamak (ITPA benchmark).",
        struphy_model_used="ShearAlfven",
        params_source=script_dir / "params_ShearAlfven.py",
        pproc_source=script_dir / "pproc_ShearAlfven.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count, all on a single node
    for num_tasks in args.ranks or variant["ranks"]:
        profiling_case.launch(
            num_tasks,
            num_nodes=1,
            param_flags=param_flags,
            slurm_preset=args.slurm_preset,
            slurm_options={"time": args.time or scaled_walltime(variant["walltime"], args.steps / 100)},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
