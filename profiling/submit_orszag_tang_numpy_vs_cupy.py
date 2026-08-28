#!/usr/bin/env python3
"""Orszag--Tang NumPy-vs-CuPy profiling case."""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase


# Matched benchmark configuration. Keep these fixed so uploaded runs remain
# directly comparable.
RANKS = 1
NUM_ELEMENTS = 96
NUM_STEPS = 5

# Pitagora resources validated for this case. The 96x96 NumPy setup exceeds
# the default per-core memory allocation, hence the explicit 16 GB request.
CPU_PRESET = {
    **SLURM_PRESETS["pitagora_dcgp"],
    "partition": "dcgp_fua_prod",
    "account": "FUSIO_HLST_7",
    "mem": "16GB",
    "time": "00:05:00",
}
GPU_PRESET = {
    **SLURM_PRESETS["pitagora_boost_fua_prod"],
    "partition": "boost_fua_prod",
    "account": "FUSIO_HLST_6",
    "gres": "gpu:1,tmpfs:10g",
    "mem": "16GB",
    "time": "00:05:00",
}

BACKEND_PRESETS = {
    "numpy": CPU_PRESET,
    "cupy": GPU_PRESET,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the fixed one-rank Orszag--Tang NumPy-vs-CuPy profiling comparison.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repository.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    params_source = (
        script_dir
        / "examples"
        / "ViscoResistiveMHD"
        / "params_orszag_tang_numpy_vs_cupy.py"
    )

    profiling_case = ProfilingCase(
        label="orszag_tang_numpy_vs_cupy",
        name="Orszag-Tang: NumPy vs CuPy",
        description=(
            f"Matched {NUM_ELEMENTS}x{NUM_ELEMENTS}x1 Orszag--Tang run for "
            f"{NUM_STEPS} timesteps using one NumPy rank and one CuPy/GPU rank."
        ),
        physics_problem="Regularized ideal-MHD Orszag--Tang vortex.",
        struphy_model_used="ViscoResistiveMHD",
        params_source=params_source,
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    cluster_name = detect_machine_name()
    for backend, preset in BACKEND_PRESETS.items():
        profiling_case.launch(
            RANKS,
            num_nodes=1,
            param_flags=["--backend", backend],
            slurm_presets={cluster_name: preset},
        )

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
