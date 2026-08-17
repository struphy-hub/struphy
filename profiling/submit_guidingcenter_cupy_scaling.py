"""Guiding-centre CuPy multi-GPU/multi-rank scaling case.

This is a strong-scaling study, not a backend comparison (see
`submit_guidingcenter_numpy_vs_cupy.py` for that): the same total marker count
(`LoadingParameters.Np` is the *total* across ranks, see
`struphy.particles.parameters.LoadingParameters`) is run with `ARRAY_BACKEND=cupy` at
increasing MPI rank counts, one rank per GPU, on the Booster partition -- so it measures
whether adding more rank+GPU pairs actually speeds up a fixed-size problem, and whether
the CUDA-ported kernels behave correctly under MPI (domain-decomposed markers, particle
sorting/communication across ranks, etc.), not just single-GPU.

Each rank binds to its own GPU via `SLURM_LOCALID` in `params_GuidingCenter_scaling.py`
(see the comment there) -- without that, every rank on a node would default to CuPy's
device 0 and contend for the same GPU, which would make this scaling study meaningless.
`SLURM_LOCALID` is a rank's index *within its node*, so this binding is correct on
multi-node runs too without any extra handling.

`--ranks 1 2 4 8` (the default) covers both intra-node scaling (1/2/4 ranks, all on a
single Booster node, 4 GPUs/node) and one inter-node step (8 ranks = 2 nodes x 4 GPUs),
so the 4->8 step is the first data point that includes cross-node MPI exchange traffic
(mpi_sort_markers) instead of only intra-node/NVLink-less PCIe traffic. `launch()` derives
`num_nodes = ceil(num_tasks / GPUS_PER_NODE)` and requires `num_tasks % num_nodes == 0`,
so rank counts must stay multiples of `GPUS_PER_NODE` once they exceed it (8, 12, 16, ...).

Uses `params_GuidingCenter_scaling.py`, not `params_GuidingCenter.py` (the single-GPU
NumPy-vs-CuPy comparison case): its much larger default Np (50,000,000 vs. 200,000) is
needed for a scaling study specifically because at smaller sizes, per-rank compute between
marker exchanges is too small to outweigh the exchange cost -- adding ranks there measured
*slower*, not faster (see that file's docstring for the numbers), and even at 10,000,000
the 4->8-rank (cross-node) step regressed. Whether 50,000,000 gives enough per-rank compute
to push the crossover point past 8 ranks, and how far, is what this case measures; see
`params_GuidingCenter_scaling.py`'s docstring for the 10,000,000 numbers this raise is
responding to.

`GuidingCenter` is used here rather than `LinearMHDDriftkineticCC` because its runtime is
actually dominated by the particle kernels this comparison is meant to measure. Its whole
propagator stack is CUDA-ported and it has no FEEC field solve, so the backend difference
shows up in the total wall clock. `LinearMHDDriftkineticCC` is dominated by MHD field
propagators and one-off setup instead, which masks any particle-kernel speedup.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# `ProfilingCase.launch` picks a preset from the dict it is given by cluster name
# (`detect_machine_name`), so the dict is keyed by the *detected* name here rather than
# by the preset's own name: on Pitagora detection always returns "pitagora_dcgp" (it
# cannot tell the Booster partition apart), and this case must still get the Booster
# preset. Keying on the detected name also keeps this working, without a KeyError, on a
# machine detection does not recognise (name None).
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

# GPUs per node on the Booster partition (the preset requests `gres=gpu:4`). Runs are
# spread so that no node holds more ranks than it has GPUs, matching the one-GPU-per-rank
# binding in params_GuidingCenter_scaling.py.
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
        default=[2, 4, 8],
        help="MPI rank counts to run with, one GPU per rank (default: 1 2 4 8; 8 spans 2 Booster nodes).",
    )
    parser.add_argument(
        "--Np",
        type=int,
        default=None,
        help="Total marker count, overriding params_GuidingCenter_scaling.py's default (50,000,000).",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "GuidingCenter"
    params_source = params_dir / "params_GuidingCenter_scaling.py"

    profiling_case = ProfilingCase(
        label="guidingcenter_cupy_scaling",
        name="Guiding-centre particles on cube, CuPy multi-GPU scaling",
        description="5D guiding-centre test particles (Np=50,000,000) in a homogeneous slab on a 3D cube, run with the CuPy array backend at increasing MPI rank counts (one GPU per rank) to measure strong-scaling speedup and verify the CUDA-ported kernels work correctly under MPI.",
        physics_problem="Guiding-centre drift-kinetic particle motion; the particle-push hot loop common to all PIC/drift-kinetic models.",
        struphy_model_used="GuidingCenter",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # The preset is looked up by cluster name inside `launch`, so build a one-entry dict
    # under whatever name detection reports for this machine.
    cluster_name = detect_machine_name()

    param_flags = ["--backend", "cupy"]
    if args.Np is not None:
        param_flags += ["--Np", str(args.Np)]

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
