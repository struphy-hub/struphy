"""ITPA TAE driven by energetic ions (LinearMHDDriftkineticCC): strong scaling from half a node to four nodes, per
space of the magnetic field.

This file defines the profiling case (the `ProfilingCase`) of one variant (space of the magnetic field, with its
default preconditioner and grid) and submits its runs on 36 and 72 MPI ranks on one node, and on 144 ranks (2 nodes,
2 clones) and 288 ranks (4 nodes, 4 clones): with clones (`--num-clones` of the params file), each node holds one clone
with the whole FEEC part (domain decomposition on its 72 ranks) and its share of the markers, so that the FEEC solves
are not spread thinner than over one node while the marker work is divided among all ranks. Raven CPU nodes (72
cores), `raven` preset of `clusters.SLURM_PRESETS`, with the whole memory of a node (120000 MB, not measured on the full
case; `--mem` overrides it). Fewer ranks are not run: the markers make them too
long. For each run, `ProfilingCase.launch` builds and submits a SLURM script, or, without a batch
system, runs directly on this machine, where the rank counts that exceed the local cores are skipped. `finalize_run`
then packages (and with `--upload` uploads) each run as soon as its own job finishes. Each generated script runs the
simulation by invoking `params_itpa_tae_B0.py` directly (its `__main__` block is the worker) with `--setup profiling`
(`--steps` time steps, default 50, solver iterations logged), and then writes the scope-profiler report and the cost
summary (time per step, solver iterations per call of the shear Alfven step, the current couplings and the
compressional steps) with `pproc_itpa_tae_B0.py --save-results`. The job scripts, SLURM logs and runs go to the
profiling folder of the driver: on Raven `/ptmp/<user>/struphy_profiling` (the scratch file system,
`clusters.SCRATCH_DIRS`), elsewhere `_profiling_jobs/` in the repo; `$STRUPHY_PROFILING_DIR/struphy_profiling` if that
variable is set.

The example (`params_itpa_tae_B0.py`) as it is: HollowTorus with tor_period 6 and a hole of radius 0.1, AdhocTorus,
cubic splines, dt = 0.25, 1e6 drift-kinetic energetic deuterons (delta-f, canonical Maxwellian with shift="B0"), the
toroidal Fourier filter keeping n = 6 (no domain decomposition along eta3), second-order discrete-gradient
guiding-center pushers. The variants:
- `hdiv_stiffness_24x96x16`: magnetic field in H(div) (2-form couplings and pushers), the shear Alfven step
  preconditioned by the Kronecker preconditioner of the Schur operator with the E x B correction, on 24 x 96 x 16
  elements (H(div) needs about 8 elements per poloidal wavelength for the right TAE frequency);
- `hcurl_hiptmair_xu_24x44x16`: magnetic field in H(curl) (1-form couplings and pushers), the curl-curl shear Alfven
  step preconditioned by Hiptmair-Xu, on 24 x 44 x 16 elements.
The variant is chosen with `--variant`; run this script once per variant, or both at once with
`sweep_LinearMHDDriftkineticCC_scaling.py`. Each variant is a separate case in the profiling-data repo.

Cost per step on 4 ranks (local workstation, a quarter of the grid and of the markers, since the full case does not
fit into its 15 GB): H(div) on 24 x 48 x 8 with 2.5e5 markers 19 s (24 PCG iterations of the shear Alfven step),
H(curl) on 24 x 22 x 8 with 2.5e5 markers 21 s (69 iterations); most of it is the markers (pushers, accumulations).
The full case should take about 4 times as long, i.e. roughly 10-20 s per step on 36 ranks. The `walltime` of a
variant is the SLURM time limit of its runs (the 36-rank run of 50 steps, with a margin), scaled with `--steps`;
`--time` overrides it.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

from utils import scaled_walltime  # noqa: E402

# the runs of the strong scaling: (MPI ranks, nodes, clones); one clone per node with more than one node
RUNS = ((36, 1, 1), (72, 1, 1), (144, 2, 2), (288, 4, 4))

# number of markers of the energetic ions (the default of the example)
NP = 1_000_000

# SLURM memory per node (MB): the most a Raven job can request (the memory of the full case is not measured)
MEM = "120000"

# variant (key = label suffix) -> name, description, flags of the params file, grid, SLURM time limit (36-rank run of
# 50 steps), runs
VARIANTS = {
    "hdiv_stiffness_24x96x16": {
        "name": "B in H(div), Kronecker preconditioner",
        "solver": "magnetic field in H(div) (2-form current couplings and guiding-center pushers), shear Alfven step "
        "with PCG preconditioned by the Kronecker preconditioner of the Schur operator with the E x B auxiliary space "
        "correction (StiffnessPreconditioner)",
        "flags": ["--b-space", "Hdiv"],
        "grid": (24, 96, 16),
        "walltime": "01:00:00",
        "runs": RUNS,
    },
    "hcurl_hiptmair_xu_24x44x16": {
        "name": "B in H(curl), Hiptmair-Xu preconditioner",
        "solver": "magnetic field in H(curl) (1-form current couplings and guiding-center pushers), curl-curl shear "
        "Alfven step with PCG preconditioned by Hiptmair-Xu",
        "flags": ["--b-space", "Hcurl"],
        "grid": (24, 44, 16),
        "walltime": "01:00:00",
        "runs": RUNS,
    },
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
        help="Space of the magnetic field (with its preconditioner and grid) of the runs.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="MPI rank counts of the runs to submit, among those of RUNS (default: all, 36 72 144 288).",
    )
    parser.add_argument("--steps", type=int, default=50, help="Time steps per run (default 50).")
    parser.add_argument(
        "--time",
        default=None,
        help="SLURM time limit of each run (default: the walltime of the variant, for 50 steps, scaled with --steps).",
    )
    parser.add_argument(
        "--mem",
        default=MEM,
        help=f"SLURM memory per node of each run, in MB or with a unit (e.g. 120000 or 100G; default {MEM}).",
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
    grid = " x ".join(map(str, variant["grid"]))
    param_flags = [
        "--setup",
        "profiling",
        "--save-results",
        "--profiling-steps",
        str(args.steps),
        "--num-elements",
        *map(str, variant["grid"]),
        "--Np",
        str(NP),
        *variant["flags"],
    ]

    runs = variant["runs"]
    if args.ranks is not None:
        unknown = sorted(set(args.ranks) - {ranks for ranks, _, _ in runs})
        if unknown:
            parser.error(f"--ranks {unknown}: not among the runs {[ranks for ranks, _, _ in runs]}.")
        runs = [run for run in runs if run[0] in args.ranks]

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=f"linear_mhd_driftkinetic_cc_itpa_tae_strong_scaling_{args.variant}",
        name=f"LinearMHDDriftkineticCC ITPA TAE strong scaling, {variant['name']}, {grid}, {NP:.0e} markers",
        description="Strong scaling (36 and 72 MPI ranks on one node, 144 and 288 on 2 and 4 nodes with one clone per "
        "node) of the LinearMHDDriftkineticCC model on the ITPA TAE benchmark driven "
        f"by energetic ions (HollowTorus with tor_period 6 and a hole of radius 0.1, AdhocTorus, {grid} elements, "
        f"cubic splines, dt = 0.25, {args.steps} steps, {NP:.0e} drift-kinetic deuterons of 400 keV with delta-f "
        "weights, canonical Maxwellian with shift B0, toroidal Fourier filter n = 6, second-order discrete-gradient "
        f"guiding-center pushers): {variant['solver']}.",
        physics_problem="Toroidal Alfven eigenmode (TAE) driven by energetic ions in a large-aspect-ratio tokamak "
        "(ITPA benchmark).",
        struphy_model_used="LinearMHDDriftkineticCC",
        params_source=script_dir / "params_itpa_tae_B0.py",
        pproc_source=script_dir / "pproc_itpa_tae_B0.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per (ranks, nodes, clones)
    for num_tasks, num_nodes, num_clones in runs:
        profiling_case.launch(
            num_tasks,
            num_nodes=num_nodes,
            param_flags=[*param_flags, "--num-clones", str(num_clones)],
            slurm_preset=args.slurm_preset,
            slurm_options={"time": args.time or scaled_walltime(variant["walltime"], args.steps / 50), "mem": args.mem},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
