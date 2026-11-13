"""ITPA TAE (ShearAlfven): strong scaling on one node, per space of the magnetic field, preconditioner and grid.

This file defines the profiling case (the `ProfilingCase`) of one variant (space of the magnetic field, preconditioner
of the shear Alfven solve and grid) and submits its runs on 4, 8, 16, 32 and 64 MPI ranks, all on a single node (a
Raven CPU node, 72 cores, with the `raven` preset of `clusters.SLURM_PRESETS`, which allocates the whole node also
for the small rank counts). For each rank count, `ProfilingCase.launch` builds and submits a SLURM script, or, without
a batch system, runs directly on this machine, where the rank counts that exceed the local cores are skipped.
`finalize_run` then packages (and with `--upload` uploads) each run as soon as its own job finishes. Each generated
script runs the simulation by invoking `params_ShearAlfven.py` directly (its `__main__` block is the worker) with
`--setup profiling` (`--steps` time steps, default 50, solver iterations logged), and then writes the scope-profiler
report and the cost summary (time per step, PCG iterations per step) with `pproc_ShearAlfven.py --save-results`.
The job scripts, SLURM logs and runs go to the profiling folder of the driver: on Raven
`/ptmp/<user>/struphy_profiling` (the scratch file system, `clusters.SCRATCH_DIRS`), elsewhere `_profiling_jobs/` in
the repo; `$STRUPHY_PROFILING_DIR/struphy_profiling` if that variable is set.

The variant is chosen with `--variant`; run this script once per variant, or all variants at once with
`sweep_ShearAlfven_scaling.py`. Each variant is a separate case in the profiling-data repo. All variants use cubic
splines and the time step 0.2. The H(div) variants run on 24 x 88 x 8 elements, the resolution at which the H(div)
version has the right TAE frequency (see the description of `params_ShearAlfven.py`); the H(curl) variants on the same
grid and on 24 x 44 x 8, the default grid of the example, on which H(curl) already has the right frequency. The
H(div) variants also run with C1 polar splines on the full torus without hole (IGAPolarTorus, `--polar`) on
24 x 90 x 8 (polar splines need a multiple of 3 elements along eta2), with the 2D x 1D polar Kronecker, the toroidal
Fourier and the polar mass matrix preconditioner; the tensor-product variants have a hole of radius 0.01 (the default
of the example), and two of them (hdiv_stiffness, hcurl_hiptmair_xu on 24 x 88 x 8) are also run with the hole radius
0.1, to see the influence of the hole on the runtime (the label contains `a1_0.1`).

Cost per step on 4 ranks (24 x 88 x 8, local workstation, measured with a hole of radius a1 = 0.1): hcurl_hiptmair_xu
0.76 s (9 PCG iterations), hdiv_stiffness 1.3 s (15), hcurl_mass 6.1 s (150), hdiv_mass 11.5 s (210),
hcurl_projected_mass 12.6 s (110). With the default a1 = 0.01 the mass matrix preconditioners need about 3 times more
iterations (8 x 32 x 4), while Hiptmair-Xu and the Kronecker preconditioner are unchanged; the time limits allow for it.
hdiv_toroidal_fourier needs 2 iterations, but its setup takes about 10 min and holds dense poloidal factorizations on
every rank (about 5-6 GB per rank, extrapolated from 12 x 44 x 8), so it runs on at most 32 ranks. The geometric
multigrid preconditioner (6 min and 740 iterations per step) is left out. With polar splines (serial, 8 x 33 x 4):
polar Kronecker 5-6, toroidal Fourier 2, polar mass matrix about 80 PCG iterations per step; on 24 x 90 x 8 (4 ranks,
LinearMHD, whose shear Alfven step is the same) the polar mass matrix preconditioner needs about 870 iterations and
130 s per step. The `walltime` of a variant is the SLURM time limit of its runs (the 4-rank run of 50 steps, with a
margin), scaled with `--steps`; `--time` overrides it.
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

FINE, COARSE, POLAR = (24, 88, 8), (24, 44, 8), (24, 90, 8)

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


def _variant(solver, grid, walltime, ranks=RANKS, a1=None):
    """A variant: the solver (key of SOLVERS), grid, SLURM time limit, rank counts and the hole radius (None for the
    default 0.01 of the example; ignored with polar splines)."""
    return {"solver": solver, "grid": grid, "walltime": walltime, "ranks": ranks, "a1": a1}


# variant (key = label suffix) -> solver, grid, SLURM time limit (4-rank run of 50 steps), rank counts, hole radius a1
VARIANTS = {
    "hdiv_stiffness_24x88x8": _variant("hdiv_stiffness", FINE, "00:15:00"),
    "hdiv_toroidal_fourier_24x88x8": _variant("hdiv_toroidal_fourier", FINE, "01:00:00", ranks=(4, 8, 16, 32)),
    "hdiv_mass_24x88x8": _variant("hdiv_mass", FINE, "01:00:00"),
    "hdiv_polar_stiffness_24x90x8": _variant("hdiv_polar_stiffness", POLAR, "00:20:00"),
    "hdiv_polar_toroidal_fourier_24x90x8": _variant(
        "hdiv_polar_toroidal_fourier", POLAR, "01:00:00", ranks=(4, 8, 16, 32)
    ),
    "hdiv_polar_mass_24x90x8": _variant("hdiv_polar_mass", POLAR, "03:00:00"),
    "hcurl_hiptmair_xu_24x88x8": _variant("hcurl_hiptmair_xu", FINE, "00:15:00"),
    # the influence of the hole radius on the runtime (not the default of the example)
    "hdiv_stiffness_a1_0.1_24x88x8": _variant("hdiv_stiffness", FINE, "00:15:00", a1=0.1),
    "hcurl_hiptmair_xu_a1_0.1_24x88x8": _variant("hcurl_hiptmair_xu", FINE, "00:15:00", a1=0.1),
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
        help="MPI rank counts of the scaling (default: those of the variant, 4 8 16 32 64; up to 32 for "
        "hdiv_toroidal_fourier).",
    )
    parser.add_argument("--steps", type=int, default=50, help="Time steps per run (default 50).")
    parser.add_argument(
        "--time",
        default=None,
        help="SLURM time limit of each run (default: the walltime of the variant, for 50 steps, scaled with --steps).",
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
        else f"HollowTorus with tor_period 6 and a hole of radius {variant['a1'] or 0.01}"
    )

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=f"shear_alfven_itpa_tae_strong_scaling_{args.variant}",
        name=f"ShearAlfven ITPA TAE strong scaling, {name}, {grid}"
        + ("" if variant["a1"] is None else f", hole a1 = {variant['a1']}"),
        description=f"Strong scaling on one node of the ShearAlfven model on the ITPA TAE benchmark ({domain}, "
        f"AdhocTorus, {grid} elements, cubic splines, dt = 0.2, {args.steps} steps): {solver}.",
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
            slurm_options={"time": args.time or scaled_walltime(variant["walltime"], args.steps / 50)},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
