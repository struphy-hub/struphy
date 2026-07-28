"""Diocotron profiling case.

This file defines the diocotron profiling case (the `ProfilingCase` and the rank
counts to profile, using the shared `cluster_presets.CLUSTER_PRESETS`) and submits
it: for each rank count in `RANKS`, `ProfilingCase.launch` builds and submits a SLURM
script (or, without a batch system, runs directly on this machine). Once every rank
count has finished running, the comparison plot across rank counts is built, and the
case is packaged/uploaded. Each generated script runs the simulation itself by
invoking `params_diocotron.py` directly (its `__main__` block is the worker).

Use this file as a template for defining other profiling cases.
"""

from pathlib import Path

from cluster_presets import CLUSTER_PRESETS
from profiling_job import ProfilingCase, local_ranks

# Paths relative to this script's location, so it can be run from anywhere.
script_dir = Path(__file__).resolve().parent
params_dir = script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability"

RANKS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)

profiling_case = ProfilingCase(
    label="diocotron_instability",
    name="Diocotron instability",
    description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
    physics_problem="Diocotron instability in a non-neutral plasma.",
    struphy_model_used="ToyDrift",
    params_source=params_dir / "params_diocotron.py",
    cluster_presets=CLUSTER_PRESETS,
)


def main() -> None:
    profiling_case.setup_run()
    ranks = RANKS if profiling_case.use_slurm else local_ranks(RANKS)

    # Launch one run per rank count, without waiting for any of them yet, so they all
    # run concurrently. Pass num_nodes=... to spread ranks across multiple nodes,
    # param_flags=[...] to forward CLI flags to params_diocotron.py, or
    # case_commands=[...] to fully override the script.
    for ntasks in ranks:
        profiling_case.launch(ntasks)

    # Wait for all jobs to finish, and then build the comparison plot and package the case.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
