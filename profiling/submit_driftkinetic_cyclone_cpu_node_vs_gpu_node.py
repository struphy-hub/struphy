"""DriftKineticElectrostaticAdiabatic (ITG cyclone) full-node CPU-vs-GPU comparison case.

The other two DriftKineticElectrostaticAdiabatic cases each answer a narrower question:
`submit_driftkinetic_cyclone_numpy_vs_cupy.py` compares backends at a fixed single rank
(where CuPy only wins once `ppc` is raised enough to give the CUDA-ported pushers real
work, see that case's params file docstring), and `submit_driftkinetic_cyclone_cupy_scaling.py`
measures CuPy strong-scaling alone. Neither answers the practical question a user
actually has: given one full CPU node and one full GPU node, which one do you point a
real ITG run at? This case runs exactly that comparison -- `ARRAY_BACKEND=numpy` using
every core of one `pitagora_dcgp` node against `ARRAY_BACKEND=cupy` using every GPU of
one Booster node -- at the same grid/marker configuration on both sides, mirroring
`submit_guidingcenter_cpu_node_vs_gpu_node.py`'s pattern for the toy model.

**Solver forced to `pcg`.** `params_cyclone.py` defaults to `solver="direct"`
(`feectools.linalg.solvers.DirectSolver`, see `ISSUE_add_direct_solver_for_constant_operators.md`
and that params file's own docstring for why and by how much it helps), but `DirectSolver`
only supports a single MPI rank -- it asserts on `nprocs > 1`, since the sparse-direct
factorization it wraps (`feectools.linalg.direct_solvers.SparseSolver`) has no
distributed variant. Both sides of this comparison run with many ranks, so both are
pinned to `--solver pcg` here regardless of the file's own default -- this measures the
field solve's *old*, unoptimized behavior at full-node scale, which is also useful data
(see `ISSUE_add_direct_solver_for_constant_operators.md`'s "Known limitations": whether a
distributed direct solve would still win at this scale is an open question this case does
not answer).

Both sides run with more than one rank, so both exercise the domain-decomposed
marker-exchange path (`mpi_sort_markers`/`apply_kinetic_bc`), not just single-rank kernel
throughput -- the grid is domain-decomposed along the two poloidal-plane directions only
(`mpi_dims_mask=(True, True, False)` in `params_cyclone.py`, matching the toroidal
Fourier filter's own `nprocs[2] == 1` requirement), so the rank count on either side is
bounded by that grid's first two `num_elements` (default 16 x 64 = at most 1024 ranks
before `--num-elements` needs to grow too).
"""

import argparse
from pathlib import Path

from clusters import HARDWARE_INFO, SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# `ProfilingCase.launch` picks a preset from the dict it is given by cluster name
# (`detect_machine_name`), so both dicts below are keyed by the *detected* name rather
# than by the preset's own name: on Pitagora detection always returns "pitagora_dcgp"
# for both partitions (it cannot tell the Booster partition apart), and the GPU run must
# still get the Booster preset. Keying on the detected name also keeps this working,
# without a KeyError, on a machine detection does not recognise (name None).
CPU_PRESET = SLURM_PRESETS["pitagora_dcgp"]
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

# One CPU-node's worth of ranks (`HARDWARE_INFO["pitagora_dcgp"]["cpus_per_node"]`), and
# one GPU-node's worth (the Booster preset requests `gres=gpu:4`), one rank per GPU as in
# params_cyclone.py's `SLURM_LOCALID` binding.
CPU_RANKS_PER_NODE = HARDWARE_INFO["pitagora_dcgp"]["cpus_per_node"]
GPU_RANKS_PER_NODE = 4


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
        "--cpu-ranks",
        type=int,
        default=CPU_RANKS_PER_NODE,
        help=f"MPI ranks for the NumPy/CPU-node run (default: {CPU_RANKS_PER_NODE}, one full pitagora_dcgp node).",
    )
    parser.add_argument(
        "--gpu-ranks",
        type=int,
        default=GPU_RANKS_PER_NODE,
        help=f"MPI ranks for the CuPy/GPU-node run, one rank per GPU (default: {GPU_RANKS_PER_NODE}, one full Booster node).",
    )
    parser.add_argument("--ppc", type=int, default=None, help="Markers per cell (overrides params_cyclone.py's default, 200).")
    parser.add_argument("--Tend", type=float, default=None, help="End time (overrides params_cyclone.py's default, 0.01 -> 10 steps).")
    parser.add_argument(
        "--num-elements",
        type=int,
        nargs=3,
        default=None,
        help=(
            "Grid resolution (overrides params_cyclone.py's default, 16 64 4). Must "
            "support at least as many ranks as --cpu-ranks in num_elements[0] * "
            "num_elements[1] (the only two domain-decomposed directions), so raise this "
            "before raising --cpu-ranks much past the default grid's 16*64=1024 cap."
        ),
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "DriftKineticElectrostaticAdiabatic"
    params_source = params_dir / "params_cyclone.py"

    param_flags = ["--solver", "pcg"]
    if args.ppc is not None:
        param_flags += ["--ppc", str(args.ppc)]
    if args.Tend is not None:
        param_flags += ["--Tend", str(args.Tend)]
    if args.num_elements is not None:
        param_flags += ["--num-elements", *[str(n) for n in args.num_elements]]

    profiling_case = ProfilingCase(
        label="driftkinetic_cyclone_cpu_node_vs_gpu_node",
        name="DriftKineticElectrostaticAdiabatic Cyclone, 1 CPU node vs 1 GPU node",
        description=(
            "Cyclone-instability ITG turbulence case for DriftKineticElectrostaticAdiabatic, "
            "run once with the NumPy array backend across every core of one CPU node and "
            "once with the CuPy array backend across every GPU of one GPU node, at the same "
            "grid/marker configuration, to compare realistic full-node throughput rather "
            "than single-rank kernel speed. The field solve is pinned to solver='pcg' on "
            "both sides -- DirectSolver (params_cyclone.py's own default) does not support "
            "more than one MPI rank."
        ),
        physics_problem="Electrostatic drift-kinetic ITG turbulence with adiabatic electrons in toroidal geometry.",
        struphy_model_used="DriftKineticElectrostaticAdiabatic",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    # One full CPU node, NumPy backend.
    profiling_case.launch(
        args.cpu_ranks,
        num_nodes=1,
        param_flags=["--backend", "numpy", *param_flags],
        slurm_presets={cluster_name: CPU_PRESET},
    )

    # One full GPU node, CuPy backend, one rank per GPU.
    profiling_case.launch(
        args.gpu_ranks,
        num_nodes=1,
        param_flags=["--backend", "cupy", *param_flags],
        slurm_presets={cluster_name: GPU_PRESET},
    )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
