"""Diocotron profiling case.

This file defines the diocotron profiling case (the `ProfilingCase` and the rank
counts to profile, using the shared `cluster_presets.CLUSTER_PRESETS`) and submits
it: for each rank count in `RANKS`, one SLURM script is built and submitted (or,
without a batch system, run directly on this machine). Once every rank count has
finished running, the comparison plot across rank counts is built, and the case is
packaged/uploaded. Each generated script runs the simulation itself by invoking
`params_diocotron.py` directly (its `__main__` block is the worker).

The per-rank command list and `SlurmScript` are built right here (not hidden behind a
shared "submit" helper), so a case-specific flag (e.g. an arbitrary `ppc`) can be
added to `case_commands` before it's wrapped in a script.

Use this file as a template for defining other profiling cases.
"""

import subprocess
from pathlib import Path

from cluster_presets import CLUSTER_PRESETS
from profiling_job import ProfilingCase, local_ranks
from slurm_script_generator.slurm_script import SlurmScript

# ------------------------------------------------------------------------ #
# Setup paths relative to this script's location, so this script can be run from
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
params_dir = script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability"
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
# Define relevant parameters for this profiling case.
RANKS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
# ------------------------------------------------------------------------ #


# ------------------------------------------------------------------------ #
# Define the profiling case
profiling_case = ProfilingCase(
    label="diocotron_instability",
    name="Diocotron instability",
    description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
    physics_problem="Diocotron instability in a non-neutral plasma.",
    struphy_model_used="ToyDrift",
    params_source=params_dir / "params_diocotron.py",
    cluster_presets=CLUSTER_PRESETS,
)
# ------------------------------------------------------------------------ #


def main() -> None:
    profiling_case.setup_run()
    ranks = RANKS if profiling_case.use_slurm else local_ranks(RANKS)

    job_infos: list[dict] = []
    job_ids: list[int] = []
    local_processes: list[tuple[int, subprocess.Popen, Path]] = []

    # Build (and submit/launch) one script per rank count, without waiting for any of
    # them yet, so they all run concurrently.
    for ntasks in ranks:
        case_commands = profiling_case.build_commands(ntasks)
        # Case-specific tweaks go here
        script_path = repo_root / f"job_profile_{profiling_case.label}_ranks{ntasks}.sh"

        if profiling_case.use_slurm:
            script = SlurmScript(
                job_name=f"profiling_{profiling_case.label}_ranks{ntasks}",
                ntasks_per_node=ntasks,
                custom_commands=case_commands,
                **profiling_case.cluster_preset,
            )
            script_text = str(script)
            job_id = script.submit_job(str(script_path))
            print(f"Submitted '{profiling_case.label}' ({ntasks} MPI ranks) as job {job_id}.")

            job_infos.append(
                {
                    "ranks": ntasks,
                    "job_script_path": str(script_path),
                    "job_script": script_text,
                    "slurm_dict": script.to_dict(),
                },
            )
            job_ids.append(job_id)
        else:
            script_text = "\n".join(["#!/bin/bash", *case_commands, ""])
            script_path.write_text(script_text, encoding="utf-8")
            script_path.chmod(0o755)

            print(
                f"No batch system found; launching '{profiling_case.label}' ({ntasks} MPI ranks) "
                f"locally via {script_path} ...",
            )
            process = subprocess.Popen(["bash", str(script_path)], cwd=repo_root)

            job_infos.append(
                {
                    "ranks": ntasks,
                    "job_script_path": str(script_path),
                    "job_script": script_text,
                    "slurm_dict": None,
                },
            )
            local_processes.append((ntasks, process, script_path))

    profiling_case.finalize_run(job_infos, job_ids, local_processes)


if __name__ == "__main__":
    main()
