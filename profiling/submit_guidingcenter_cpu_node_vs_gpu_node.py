"""Guiding-centre full-node CPU-vs-GPU comparison.

Runs `params_GuidingCenter_scaling.py` once across every core of one CPU node
(`ARRAY_BACKEND=numpy`) and once across every GPU of one GPU node (`ARRAY_BACKEND=cupy`),
same total marker count on both sides -- the realistic "which node do I use" comparison,
as opposed to `submit_guidingcenter_numpy_vs_cupy.py` (single-rank backend comparison) or
`submit_guidingcenter_cupy_scaling.py` (CuPy-only rank scaling).
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
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "GuidingCenter"
    params_source = params_dir / "params_GuidingCenter_scaling.py"

    profiling_case = ProfilingCase(
        label="guidingcenter_cpu_node_vs_gpu_node",
        name="GuidingCenter: CPU node vs GPU node",
        description=(
            "GuidingCenter particles on a cube, run across one full CPU node (NumPy) vs "
            "one full GPU node (CuPy) at the same marker count."
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

    # One full CPU node, NumPy backend.
    profiling_case.launch(
        CPU_RANKS_PER_NODE,
        num_nodes=1,
        param_flags=["--backend", "numpy"],
        slurm_presets={cluster_name: CPU_PRESET},
    )

    # One full GPU node, CuPy backend, one rank per GPU.
    profiling_case.launch(
        GPU_RANKS_PER_NODE,
        num_nodes=1,
        param_flags=["--backend", "cupy"],
        slurm_presets={cluster_name: GPU_PRESET},
    )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
