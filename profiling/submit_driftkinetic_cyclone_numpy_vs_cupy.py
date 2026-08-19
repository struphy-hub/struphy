"""DriftKineticElectrostaticAdiabatic (ITG cyclone) NumPy-vs-CuPy profiling case.

Runs `params_cyclone.py` once with `ARRAY_BACKEND=numpy` and once with
`ARRAY_BACKEND=cupy`. Unlike GuidingCenter, this model has a real per-step FEEC field
solve (PoissonAdiabaticGyrokinetic, default solver='direct'), so it measures whether
the CUDA port helps end to end, not just the particle kernels.
"""

import argparse
from pathlib import Path

from clusters import SLURM_PRESETS, detect_machine_name
from profiling_job import ProfilingCase

# The preset is looked up by cluster name inside `launch`, so both dicts below are keyed
# by the *detected* name rather than by the preset's own name: on Pitagora detection
# always returns "pitagora_dcgp" for both partitions (it cannot tell the Booster
# partition apart), and the GPU run must still get the Booster preset. Keying on the
# detected name also keeps this working, without a KeyError, on a machine detection does
# not recognise (name None).
CPU_PRESET = SLURM_PRESETS["pitagora_dcgp"]
GPU_PRESET = SLURM_PRESETS["pitagora_boost_fua_dbg"]

BACKEND_PRESETS = {
    "numpy": CPU_PRESET,
    "cupy": GPU_PRESET,
}


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
    params_dir = script_dir / "examples" / "DriftKineticElectrostaticAdiabatic"
    params_source = params_dir / "params_cyclone.py"

    profiling_case = ProfilingCase(
        label="driftkinetic_cyclone_numpy_vs_cupy",
        name="ITG cyclone: NumPy vs CuPy",
        description=(
            "Cyclone-instability ITG turbulence (DriftKineticElectrostaticAdiabatic), run "
            "once on NumPy and once on CuPy."
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

    # Launch one run per backend, one rank each -- this case is a backend comparison, not
    # a scaling study (see submit_guidingcenter_cupy_scaling.py for that pattern).
    for backend, preset in BACKEND_PRESETS.items():
        profiling_case.launch(
            1,
            num_nodes=1,
            param_flags=["--backend", backend],
            slurm_presets={cluster_name: preset},
        )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
