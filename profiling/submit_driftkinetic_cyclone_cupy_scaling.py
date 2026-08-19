"""DriftKineticElectrostaticAdiabatic (ITG cyclone) CuPy multi-GPU/multi-rank scaling case.

This is a strong-scaling study, not a backend comparison (see
`submit_driftkinetic_cyclone_numpy_vs_cupy.py` for that, and
`submit_driftkinetic_cyclone_cpu_node_vs_gpu_node.py` for the full-node version): the
same grid/marker configuration is run with `ARRAY_BACKEND=cupy` at increasing MPI rank
counts, one rank per GPU, on the Booster partition -- so it measures whether adding more
rank+GPU pairs actually speeds up this fixed-size ITG run, and whether the CUDA-ported
kernels behave correctly under MPI for this model (domain-decomposed markers, particle
sorting/communication, the toroidal Fourier filter's `nprocs[2] == 1` requirement, etc.),
mirroring `submit_guidingcenter_cupy_scaling.py`'s pattern for the toy model.

Each rank binds to its own GPU via `SLURM_LOCALID` in `params_cyclone.py` (see the
comment there) -- without that, every rank on a node would default to CuPy's device 0
and contend for the same GPU, which would make this scaling study meaningless.
`SLURM_LOCALID` is a rank's index *within its node*, so this binding is correct on
multi-node runs too without any extra handling.

`--ranks 1 2 4 8` (the default) covers both intra-node scaling (1/2/4 ranks, all on a
single Booster node, 4 GPUs/node) and one inter-node step (8 ranks = 2 nodes x 4 GPUs),
so the 4->8 step is the first data point that includes cross-node MPI exchange traffic
instead of only intra-node/NVLink-less PCIe traffic. `launch()` derives
`num_nodes = ceil(num_tasks / GPUS_PER_NODE)` and requires `num_tasks % num_nodes == 0`,
so rank counts must stay multiples of `GPUS_PER_NODE` once they exceed it (8, 12, 16, ...).

The grid is only domain-decomposed along the two poloidal-plane directions
(`mpi_dims_mask=(True, True, False)` in `params_cyclone.py`, matching the toroidal
Fourier filter's own `nprocs[2] == 1` requirement), so the rank count is bounded by
`num_elements[0] * num_elements[1]` (default 16 x 64 = at most 1024) -- pass a larger
`--num-elements` before pushing `--ranks` much higher than the default.

**Solver forced to `pcg`.** `params_cyclone.py` defaults to `solver="direct"`
(`feectools.linalg.solvers.DirectSolver`, see
`ISSUE_add_direct_solver_for_constant_operators.md`), but `DirectSolver` only supports a
single MPI rank -- it has no distributed sparse-direct factorization. Since this case
compares rank counts against each other, using the same solver at every rank count
matters more than using the fastest one available only at rank 1, so `--solver pcg` is
forced across the whole sweep (including at `--ranks 1`) for a consistent, apples-to-apples
comparison. Whether a (currently nonexistent) distributed direct solve would change this
picture is an open question -- see the ISSUE file's "Known limitations".

**`Tend` shortened to 3 steps.** `params_cyclone.py`'s own default (`Tend=0.01`, `dt=0.001`
-> 10 steps) was written for the `direct`-solver comparison, where the field solve is
essentially free after the first call. Here `--solver pcg` is forced instead, and the
case's own profiling notes (`driftkinetic_cyclone_numpy_vs_cupy`'s run metadata) recorded
a single CuPy `pcg` step taking anywhere from ~14.5 s (clean, isolated GPU) up to 174 s
under GPU contention on this shared partition -- 10 such steps plus ~90 s of CUDA
setup/compile do not reliably fit in `pitagora_boost_fua_dbg`'s 15-30 min walltime (its
partition-enforced cap). 3 steps is enough to warm up and get a stable per-step timing for
the scaling comparison this case cares about; pass `--Tend` to override.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# The preset is looked up by cluster name inside `launch`, so the dict is keyed by the
# *detected* name here rather than by the preset's own name: on Pitagora detection
# always returns "pitagora_dcgp" (it cannot tell the Booster partition apart), and this
# case must still get the Booster preset. Keying on the detected name also keeps this
# working, without a KeyError, on a machine detection does not recognise (name None).
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_cyclone.py.
GPUS_PER_NODE = 4


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
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2], #, 4, 8],
        help="MPI rank counts to run with, one GPU per rank (default: 1 2 4 8; 8 spans 2 Booster nodes).",
    )
    parser.add_argument("--ppc", type=int, default=None, help="Markers per cell (overrides params_cyclone.py's default, 200).")
    parser.add_argument(
        "--Tend",
        type=float,
        default=0.003,
        help=(
            "End time (default: 0.003 -> 3 steps, shortened from params_cyclone.py's own "
            "0.01/10 steps so the forced pcg solver reliably fits in the debug partition's "
            "walltime; see the module docstring)."
        ),
    )
    parser.add_argument(
        "--num-elements",
        type=int,
        nargs=3,
        default=None,
        help=(
            "Grid resolution (overrides params_cyclone.py's default, 16 64 4). Must "
            "support at least as many ranks as the largest --ranks value in "
            "num_elements[0] * num_elements[1] (the only two domain-decomposed "
            "directions)."
        ),
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "DriftKineticElectrostaticAdiabatic"
    params_source = params_dir / "params_cyclone.py"

    profiling_case = ProfilingCase(
        label="driftkinetic_cyclone_cupy_scaling",
        name="DriftKineticElectrostaticAdiabatic Cyclone, CuPy multi-GPU scaling",
        description=(
            "Cyclone-instability ITG turbulence case for DriftKineticElectrostaticAdiabatic, "
            "run with the CuPy array backend at increasing MPI rank counts (one GPU per "
            "rank) on a fixed grid/marker configuration, to measure strong-scaling "
            "speedup and verify the CUDA-ported kernels and MPI exchange paths work "
            "correctly for this model under multi-rank/multi-GPU. The field solve is "
            "pinned to solver='pcg' at every rank count -- DirectSolver "
            "(params_cyclone.py's own default) does not support more than one MPI rank."
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

    param_flags = ["--backend", "cupy", "--solver", "pcg", "--Tend", str(args.Tend)]
    if args.ppc is not None:
        param_flags += ["--ppc", str(args.ppc)]
    if args.num_elements is not None:
        param_flags += ["--num-elements", *[str(n) for n in args.num_elements]]

    # Launch one run per rank count. One node per GPUS_PER_NODE ranks -- `launch` would
    # otherwise derive the node count from `cpus_per_node`, which on a GPU partition packs
    # far more ranks per node than there are GPUs.
    for num_tasks in args.ranks:
        num_nodes = -(-num_tasks // GPUS_PER_NODE)
        profiling_case.launch(
            num_tasks,
            num_nodes=num_nodes,
            param_flags=param_flags,
            slurm_presets={cluster_name: GPU_PRESET},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
