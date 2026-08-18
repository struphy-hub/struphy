"""Guiding-centre full-node CPU-vs-GPU comparison case.

The other two GuidingCenter cases each answer a narrower question:
`submit_guidingcenter_numpy_vs_cupy.py` compares backends at a fixed, small rank count
(default 1), and `submit_guidingcenter_cupy_scaling.py` measures CuPy strong-scaling
alone. Neither answers the practical question a user actually has: given one full CPU
node and one full GPU node, which one do you point a job at? This case runs exactly
that comparison -- `ARRAY_BACKEND=numpy` using every core of one `pitagora_dcgp` node
against `ARRAY_BACKEND=cupy` using every GPU of one Booster node -- at the same total
marker count on both sides.

Both sides run with more than one rank, so both exercise the domain-decomposed
marker-exchange path (`mpi_sort_markers`/`apply_kinetic_bc`) this session's performance
work targeted, not just single-rank kernel throughput. `params_GuidingCenter_scaling.py`
is used (not `params_GuidingCenter.py`) for the same reason `submit_guidingcenter_cupy_scaling.py`
uses it: its default Np is large enough that per-rank compute between exchanges has a
chance of outweighing the exchange cost -- see that file's docstring for the measurements
this default responds to. `--Np` overrides it if a different problem size is of interest.

`GuidingCenter` is used, as in the other two cases, because its whole propagator stack is
CUDA-ported and it has no FEEC field solve, so wall-clock time is dominated by the
particle kernels and their MPI exchange rather than by anything the backend choice
doesn't touch.
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
# `params_GuidingCenter_scaling.py`'s `SLURM_LOCALID` binding.
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
    parser.add_argument(
        "--Np",
        type=int,
        default=None,
        help="Total marker count, overriding params_GuidingCenter_scaling.py's default (10,000,000).",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "GuidingCenter"
    params_source = params_dir / "params_GuidingCenter_scaling.py"

    profiling_case = ProfilingCase(
        label="guidingcenter_cpu_node_vs_gpu_node",
        name="Guiding-centre particles on cube, 1 CPU node vs 1 GPU node",
        description=(
            "5D guiding-centre test particles in a homogeneous slab on a 3D cube, run once "
            "with the NumPy array backend across every core of one CPU node and once with "
            "the CuPy array backend across every GPU of one GPU node, at the same total "
            "marker count, to compare realistic full-node throughput rather than "
            "single-rank kernel speed."
        ),
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

    Np_flags = ["--Np", str(args.Np)] if args.Np is not None else []

    # One full CPU node, NumPy backend.
    profiling_case.launch(
        args.cpu_ranks,
        num_nodes=1,
        param_flags=["--backend", "numpy", *Np_flags],
        slurm_presets={cluster_name: CPU_PRESET},
    )

    # One full GPU node, CuPy backend, one rank per GPU.
    profiling_case.launch(
        args.gpu_ranks,
        num_nodes=1,
        param_flags=["--backend", "cupy", *Np_flags],
        slurm_presets={cluster_name: GPU_PRESET},
    )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
