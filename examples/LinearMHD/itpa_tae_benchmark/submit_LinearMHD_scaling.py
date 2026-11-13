"""ITPA TAE (LinearMHD): strong scaling on one node, per space of the magnetic field, preconditioner and grid.

This file defines the profiling case (the `ProfilingCase`) of one variant (space of the magnetic field, preconditioner
of the shear Alfven solve and grid) and submits its runs on 4, 8, 16, 32 and 64 MPI ranks, all on a single node (a Raven
CPU node, 72 cores, with the `raven` preset of `clusters.SLURM_PRESETS`); the toroidal Fourier variants on at most 32
ranks (see the memory below). For each rank count, `ProfilingCase.launch` builds and submits a SLURM script, or,
without a batch system, runs directly on this machine, where the rank counts that exceed the local cores are skipped.
`finalize_run` then packages (and with `--upload` uploads) each run as soon as its own job finishes. Each generated
script runs the simulation by invoking `params_TAE_benchmark.py` directly (its `__main__` block is the worker) with
`--setup profiling` (`--steps` time steps, default 50, solver iterations logged), and then writes the scope-profiler
report and the cost summary (time per step, PCG iterations per call of the shear Alfven, sound wave and background
current solves) with `pproc_TAE_benchmark.py --save-results`. The job scripts, SLURM logs and runs go to the profiling
folder of the driver: on Raven `/ptmp/<user>/struphy_profiling` (the scratch file system, `clusters.SCRATCH_DIRS`),
elsewhere `_profiling_jobs/` in the repo; `$STRUPHY_PROFILING_DIR/struphy_profiling` if that variable is set.

The variants are those of the ShearAlfven ITPA case (`examples/ShearAlfven/itpa_tae_benchmark/
submit_ShearAlfven_scaling.py`), so that the two models can be compared: LinearMHD adds the compressional steps
(background current, background pressure gradient and the sound wave, with mass matrix solves preconditioned by the
Kronecker approximation of the inverse mass matrix, polar with `--polar`) to the same shear Alfven step. The variant is
chosen with `--variant`; run this script once per variant, or all variants at once with `sweep_LinearMHD_scaling.py`.
Each variant is a separate case in the profiling-data repo. All variants use cubic splines and the time step 0.2. The
H(div) variants run on 24 x 88 x 8 elements, the resolution at which the H(div) version has the right TAE frequency
(see the description of `params_TAE_benchmark.py`); the H(curl) variants on the same grid and on 24 x 44 x 8, the
default grid of the example. The H(div) variants also run with C1 polar splines on the full torus without hole
(IGAPolarTorus, `--polar`) on 24 x 90 x 8 (polar splines need a multiple of 3 elements along eta2). The
tensor-product variants have a hole of radius 0.1 (the default of this example; ShearAlfven: 0.01), and two of them
(hdiv_stiffness, hcurl_hiptmair_xu on 24 x 88 x 8) are also run with the hole radius 0.01 (the label contains
`a1_0.01`). The geometric multigrid preconditioner is left out (as for ShearAlfven).

Cost per step on 4 ranks (local workstation, hole a1 = 0.1; PCG iterations per step of the shear Alfven and the sound
wave solves): hcurl_hiptmair_xu 0.71 s (11, 4) on 24 x 44 x 8 and 1.2 s (9, 6) on 24 x 88 x 8, hdiv_stiffness 2.0 s
(15, 7), hdiv_polar_stiffness 5.6 s (8, 21), hcurl_mass 9.2 s (146, 9), hdiv_mass 12.5 s (211, 8),
hcurl_projected_mass 14.8 s (106, 10), hdiv_polar_mass 133 s (870, 24). The sound wave step (mass matrix
preconditioner) takes about 20-25 % of a step with tensor-product splines and about half with polar splines; with
polar splines its default is now the 2D x 1D Kronecker preconditioner of its Schur complement (on 24 x 48 x 8: 3
instead of 11 iterations, the step 30 % faster), so the polar variants are faster than these numbers. The
toroidal Fourier variants were not run on their full grids locally (too much memory). The `walltime` of a variant is
the SLURM time limit of its runs (the 4-rank run of 50 steps, with a margin), scaled with `--steps`; `--time`
overrides it.
ToroidalFourierPreconditioner holds dense poloidal factorizations on every rank (about 5-6 GB per rank at
24 x 88 x 8), so it runs on at most 32 ranks.

Memory: on Raven, the runs on at most 36 ranks share their node with other jobs and request 64000 MB by default (the
`raven` preset), the runs on more ranks get the whole node (240000 MB, `clusters.SHARED_NODES`); `--mem` sets the memory
per node of all runs. Peak memory (local workstation, 2 steps, the setup dominates) is about 0.56 GB per rank plus 5.3
GB divided among the ranks (H(div), Kronecker preconditioner; about 8 GB on 4 ranks and 41 GB on 64 ranks). Polar
splines hold the polar operators and the poloidal factors of the preconditioner on every rank: about 2.8 GB per rank
plus 5.9 GB (extrapolated from ShearAlfven, about 1.2 times more on 24 x 48 x 8): about 95 GB on 32 ranks and 185 GB on
64 ranks, so the polar variants request 120000 MB, the most a shared Raven job can request, on up to 32 ranks. The
toroidal Fourier variants (dense poloidal factorizations on every rank, about 5-6 GB per rank, not measured on the full
grid) run on at most 32 ranks and request a whole node (240000 MB) for all runs (shared Raven jobs can request at most
120000 MB, so the driver allocates all 72 cores of the node and runs the ranks of the run in it); on 32 ranks this is
tight (about 200 GB, more with polar splines).
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

from utils import scaled_walltime  # noqa: E402

# MPI ranks of the strong scaling, all on one node
RANKS = (4, 8, 16, 32, 64)
# memory per node (MB) of the runs on shared Raven nodes (at most 36 ranks) of the variants that hold global data on
# every rank: polar splines the most a shared job can request; the toroidal Fourier preconditioner a whole node, on
# at most 32 ranks
MEM_POLAR = "120000"
MEM_TOROIDAL_FOURIER = "240000"
RANKS_TOROIDAL_FOURIER = (4, 8, 16, 32)

FINE, COARSE, POLAR = (24, 88, 8), (24, 44, 8), (24, 90, 8)

# the solvers: key -> (name, description, flags of the params file)
HDIV = "magnetic field in H(div), implicit solve for the velocity (exactly energy conserving), PCG preconditioned by"
HCURL = (
    "magnetic field in H(curl), implicit curl-curl solve for the magnetic field (numerically damped), PCG "
    "preconditioned by"
)
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
    "hdiv_polar_stiffness": (
        "B in H(div), polar splines, Kronecker preconditioner",
        f"{HDIV} the 2D x 1D polar Kronecker preconditioner of the Schur operator (StiffnessPreconditioner with polar "
        "splines: exact in the poloidal plane, diagonal in the eigenbasis along eta3)",
        ["--b-space", "Hdiv", "--polar", "--precond", "StiffnessPreconditioner"],
    ),
    "hdiv_polar_toroidal_fourier": (
        "B in H(div), polar splines, toroidal Fourier preconditioner",
        f"{HDIV} the exact inverse of the Schur operator of the axisymmetric equilibrium (FFT along eta3, one dense "
        "poloidal factorization per toroidal mode on every rank, ToroidalFourierPreconditioner)",
        ["--b-space", "Hdiv", "--polar", "--precond", "ToroidalFourierPreconditioner"],
    ),
    "hdiv_polar_mass": (
        "B in H(div), polar splines, mass matrix preconditioner",
        f"{HDIV} the approximate inverse of the polar mass matrix (PolarMassMatrixPreconditioner)",
        ["--b-space", "Hdiv", "--polar", "--precond", "PolarMassMatrixPreconditioner"],
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


def _variant(solver, grid, walltime, ranks=RANKS, a1=None, mem_shared=None):
    """A variant: the solver (key of SOLVERS), grid, SLURM time limit, rank counts, the hole radius (None for the
    default 0.1 of the example; ignored with polar splines) and the memory per node of the runs on
    shared nodes (`mem_shared` of `ProfilingCase.launch`, Raven only; None for the mem of the SLURM preset)."""
    return {"solver": solver, "grid": grid, "walltime": walltime, "ranks": ranks, "a1": a1, "mem_shared": mem_shared}


# variant (key = label suffix) -> solver, grid, SLURM time limit (4-rank run of 50 steps), rank counts, hole radius a1,
# memory of the runs on shared nodes
VARIANTS = {
    "hdiv_stiffness_24x88x8": _variant("hdiv_stiffness", FINE, "00:20:00"),
    "hdiv_toroidal_fourier_24x88x8": _variant(
        "hdiv_toroidal_fourier", FINE, "01:00:00", ranks=RANKS_TOROIDAL_FOURIER, mem_shared=MEM_TOROIDAL_FOURIER
    ),
    "hdiv_mass_24x88x8": _variant("hdiv_mass", FINE, "01:00:00"),
    "hdiv_polar_stiffness_24x90x8": _variant("hdiv_polar_stiffness", POLAR, "00:30:00", mem_shared=MEM_POLAR),
    "hdiv_polar_toroidal_fourier_24x90x8": _variant(
        "hdiv_polar_toroidal_fourier", POLAR, "01:00:00", ranks=RANKS_TOROIDAL_FOURIER, mem_shared=MEM_TOROIDAL_FOURIER
    ),
    "hdiv_polar_mass_24x90x8": _variant("hdiv_polar_mass", POLAR, "03:00:00", mem_shared=MEM_POLAR),
    "hcurl_hiptmair_xu_24x88x8": _variant("hcurl_hiptmair_xu", FINE, "00:20:00"),
    # the influence of the hole radius on the runtime (not the default of the example)
    "hdiv_stiffness_a1_0.01_24x88x8": _variant("hdiv_stiffness", FINE, "00:20:00", a1=0.01),
    "hcurl_hiptmair_xu_a1_0.01_24x88x8": _variant("hcurl_hiptmair_xu", FINE, "00:20:00", a1=0.01),
    "hcurl_mass_24x88x8": _variant("hcurl_mass", FINE, "00:40:00"),
    "hcurl_projected_mass_24x88x8": _variant("hcurl_projected_mass", FINE, "01:00:00"),
    "hcurl_hiptmair_xu_24x44x8": _variant("hcurl_hiptmair_xu", COARSE, "00:15:00"),
    "hcurl_mass_24x44x8": _variant("hcurl_mass", COARSE, "00:20:00"),
    "hcurl_projected_mass_24x44x8": _variant("hcurl_projected_mass", COARSE, "00:30:00"),
}


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
        help="MPI rank counts of the scaling (default: those of the variant, 4 8 16 32 64; up to 32 for the "
        "toroidal Fourier variants).",
    )
    parser.add_argument("--steps", type=int, default=50, help="Time steps per run (default 50).")
    parser.add_argument(
        "--time",
        default=None,
        help="SLURM time limit of each run (default: the walltime of the variant, for 50 steps, scaled with --steps).",
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="SLURM memory per node of all runs, in MB or with a unit (e.g. 120000 or 100G; default on Raven: on "
        "at most 36 ranks 64000, 120000 for the polar and 240000 for the toroidal Fourier variants; on more ranks "
        "the whole node, 240000).",
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
        *([] if variant["a1"] is None else ["--a1", str(variant["a1"])]),
    ]

    domain = (
        "IGAPolarTorus with tor_period 6, C1 polar splines without hole"
        if "--polar" in solver_flags
        else f"HollowTorus with tor_period 6 and a hole of radius {variant['a1'] or 0.1}"
    )

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=f"linear_mhd_itpa_tae_strong_scaling_{args.variant}",
        name=f"LinearMHD ITPA TAE strong scaling, {name}, {grid}"
        + ("" if variant["a1"] is None else f", hole a1 = {variant['a1']}"),
        description=f"Strong scaling on one node of the LinearMHD model on the ITPA TAE benchmark ({domain}, "
        f"AdhocTorus, {grid} elements, cubic splines, dt = 0.2, {args.steps} steps): shear Alfven step with {solver}; "
        "background current, background pressure gradient and sound wave steps with mass matrix preconditioners.",
        physics_problem="Toroidal Alfven eigenmode (TAE) in a large-aspect-ratio tokamak (ITPA benchmark).",
        struphy_model_used="LinearMHD",
        params_source=script_dir / "params_TAE_benchmark.py",
        pproc_source=script_dir / "pproc_TAE_benchmark.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count, all on a single node
    slurm_options = {"time": args.time or scaled_walltime(variant["walltime"], args.steps / 50)}
    if args.mem:
        slurm_options["mem"] = args.mem
    elif variant["mem_shared"]:
        slurm_options["mem_shared"] = variant["mem_shared"]
    for num_tasks in args.ranks or variant["ranks"]:
        profiling_case.launch(
            num_tasks,
            num_nodes=1,
            param_flags=param_flags,
            slurm_preset=args.slurm_preset,
            slurm_options=slurm_options,
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
