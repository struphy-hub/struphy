"""Shared machinery for running profiling jobs.

A concrete profiling job (e.g. ``submit_diocotron_job.py``) only needs to define
its own list of `ProfilingCase` objects and call `run_profiling_job`.

With a batch system the case is submitted to SLURM; without one (a laptop or
workstation) the very same commands run directly on this machine, with `mpirun`
instead of `srun` and without the environment-module lines. Profiling, packaging and
upload are identical either way.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from package_profiling_results import MACHINE_PARAMS_FILE, detect_machine_name, package_testcase
from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue

from struphy import Compiler

from upload import _push_profiling_data

from utils import _git_commit, _git_commit_short, _make_unique_results_root

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
profiling_results_base = repo_root / "results" / "profiling"
latest_results_root_path = profiling_results_base / "latest_run_root.txt"

default_cluster_name = "pitagora"

# Installs `whereami` and `load_modules` into the directory passed to the script.
whereami_install_url = "https://raw.githubusercontent.com/max-models/whereami/main/install.sh"


def install_whereami(venv_path: Path) -> None:
    """Install `whereami` into the venv's bin directory, from the login node.

    Compute nodes have no outbound network, so this must happen before `sbatch`. The
    venv lives on a shared filesystem, so the job picks the executable up from `PATH`
    once it activates the same venv.

    A failed install is not fatal: the job then produces no `machine_params.json` and
    packaging records `machine_params_file: null`.
    """
    install_dir = venv_path / "bin"
    print(f"Installing whereami into {install_dir} ...")
    result = subprocess.run(
        f'curl -fsSL {whereami_install_url} | bash -s -- "{install_dir}"',
        shell=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"WARNING: installing whereami failed (exit code {result.returncode}); "
            "the job will not record machine parameters.",
        )


def detect_cluster_name(cluster_presets: dict[str, dict]) -> str:
    """Pick the cluster preset for the current machine.

    The detected machine name ("Pitagora (DCGP)", "TOK", ...) is matched against the
    preset keys ("pitagora", "tok", ...). Falls back to `default_cluster_name` when the
    machine is unknown or has no preset (e.g. when submitting from a laptop), so
    submission behaves as before.
    """
    machine_name = detect_machine_name()
    if machine_name:
        for preset_name in cluster_presets:
            if preset_name.lower() in machine_name.lower():
                print(f"Detected machine '{machine_name}'; using cluster preset '{preset_name}'.")
                return preset_name

    print(
        f"No cluster preset matches the detected machine ({machine_name!r}); using '{default_cluster_name}'.",
    )
    return default_cluster_name


@dataclass(frozen=True)
class ProfilingCase:
    label: str
    name: str
    description: str
    physics_problem: str
    struphy_model_used: str
    ranks: tuple[int, ...]
    params_source: Path
    run_script: Path
    cluster_presets: dict[str, dict]


def has_module_system() -> bool:
    """True on a machine with environment modules (Lmod / Tcl modules)."""
    return bool(
        os.environ.get("MODULESHOME") or os.environ.get("LMOD_CMD") or shutil.which("modulecmd"),
    )


def detect_launcher() -> str:
    """The MPI launcher to use: `srun` under SLURM, otherwise `mpirun`/`mpiexec`."""
    if shutil.which("srun"):
        return "srun"
    for launcher in ("mpirun", "mpiexec"):
        if shutil.which(launcher):
            return launcher
    raise RuntimeError(
        "No MPI launcher found; install an MPI implementation providing `mpirun` (or `mpiexec`).",
    )


def local_ranks(ranks: tuple[int, ...]) -> tuple[int, ...]:
    """Drop rank counts that exceed the number of local cores.

    Oversubscribing simply makes `mpirun` refuse to start, so a case designed for a
    64-core node still runs its small rank counts on a laptop.
    """
    available = os.cpu_count() or 1
    usable = tuple(ntasks for ntasks in ranks if ntasks <= available)
    skipped = [ntasks for ntasks in ranks if ntasks > available]
    if skipped:
        print(f"Skipping rank counts that exceed the {available} local cores: {skipped}")
    if not usable:
        raise RuntimeError(
            f"None of the requested rank counts {list(ranks)} fit on {available} local cores.",
        )
    return usable


def build_case_commands(
    case: ProfilingCase,
    output_root: Path,
    venv_path: Path,
    *,
    launcher: str = "srun",
    use_modules: bool = True,
    ranks: tuple[int, ...] | None = None,
) -> list[str]:
    activate_path = venv_path / "bin" / "activate"
    case_ranks = case.ranks if ranks is None else ranks
    commands = [
        # Environment modules only exist on the clusters, not on a laptop.
        *(
            [
                "module purge",
                "source ./setup/modules.sh load",
                "module list",
            ]
            if use_modules
            else []
        ),
        f"source {activate_path!s}",
        'echo "----------------------------------------"',
        f'echo "Running profiling case: {case.label}"',
        f'echo "Description: {case.description}"',
        f'echo "Physics problem: {case.physics_problem}"',
        f'echo "Struphy model used: {case.struphy_model_used}"',
        f'echo "Case directory: {output_root}"',
        'echo "----------------------------------------"',
        f'mkdir -p "{output_root}"',
        f'cp "{case.params_source}" "{output_root / "parameters.py"}"',
        # Record the machine parameters of the node this job runs on. `whereami` was
        # installed into the venv by `install_whereami` before submission, since compute
        # nodes have no network access.
        f'whereami --output "{output_root / MACHINE_PARAMS_FILE}"',
        f'ls -l "{output_root}"',
    ]

    commands.append("existing_h5_files=()")

    for ntasks in case_ranks:
        sim_dir = output_root / f"sim_ranks{ntasks}"
        h5_file = sim_dir / "profiling_data.h5"
        mpirun_log = output_root / f"mpirun_ranks{ntasks}.log"
        commands.extend(
            [
                "",
                f'echo "Running {case.label} with {ntasks} MPI ranks"',
                f'{launcher} -n {ntasks} python {case.run_script} --out-root "{output_root}"',
                f'scope-profiler pproc "{h5_file}" -o "{sim_dir}"',
                f'existing_h5_files+=("{h5_file}")',
            ]
            # [
            #     "",
            #     f'echo "Running {case.label} with {ntasks} MPI ranks"',
            #     (
            #         f"if srun -n {ntasks} python {case.run_script} "
            #         f'--out-root "{output_root}" > "{mpirun_log}" 2>&1; then'
            #     ),
            #     f'    echo "srun ({ntasks} ranks) succeeded"',
            #     "else",
            #     f'    echo "srun ({ntasks} ranks) FAILED with exit code $?; log follows:"',
            #     f'    cat "{mpirun_log}"',
            #     "fi",
            #     f'if [ -f "{h5_file}" ]; then',
            #     f'    scope-profiler pproc "{h5_file}" -o "{sim_dir}"',
            #     f'    existing_h5_files+=("{h5_file}")',
            #     "else",
            #     f'    echo "No profiling data found at {h5_file}; skipping scope-profiler pproc for this rank count."',
            #     "fi",
            # ],
        )

    commands.extend(
        [
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {case.label}"',
            'echo "----------------------------------------"',
            "# Postprocessing comparison plots",
            f'scope-profiler pproc "${{existing_h5_files[@]}}" --rank 0 -o "{output_root / "figures"}"',
        ],
    )

    return commands


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload.")
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Root folder for this profiling run. By default a unique "
            "results/profiling/DATETIME-COMMIT folder is created."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="fortran",
        help='Pyccel language to compile the Struphy kernels with: "fortran" (default) or "c".',
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="GNU",
        help='Pyccel compiler family to use: "GNU" (default), "intel", "PGI", "nvidia", or "LLVM".',
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    return parser


def run_profiling_job(case: ProfilingCase) -> None:
    """Compile Struphy, run `case` (via SLURM or locally), and package/push the results."""

    # Parse command-line arguments and validate the virtual environment
    args = build_arg_parser().parse_args()

    # Validate that a virtual environment is active
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError(
            "VIRTUAL_ENV is not set; activate a virtual environment before submitting the job.",
        )
    venv_path = Path(virtual_env)

    # Install whereami here, on the login node: the compute nodes running the job have
    # no outbound network access.
    install_whereami(venv_path)

    # Compile Struphy kernels with the specified language and compiler
    compiler = Compiler(language=args.language, compiler=args.compiler)
    if not compiler.compiled(language=args.language):
        print("Compiling Struphy kernels ...")
        compiler.compile()
    print("Done compiling Struphy kernels.")

    # Create a unique results root for this profiling run
    # and write it to the "latest_run_root.txt" marker file.
    output_root = Path("profiling-results-export").resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profiling_results_base.mkdir(parents=True, exist_ok=True)

    # Determine the current git commit hash for the Struphy repo
    run_commit = _git_commit(repo_root)
    if args.results_root is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_token = f"{timestamp}-{_git_commit_short(repo_root)}"
        run_results_root = _make_unique_results_root(profiling_results_base, run_token)
    else:
        run_results_root = args.results_root.expanduser().resolve()

    run_results_root.mkdir(parents=True, exist_ok=True)
    latest_results_root_path.write_text(str(run_results_root), encoding="utf-8")
    print(f"Profiling run root: {run_results_root}")

    packaged_dirs: list[Path] = []

    # Create a subdirectory for this case's output under the run results root
    case_output_root = run_results_root / case.label
    case_output_root.mkdir(parents=True, exist_ok=True)

    script_path = repo_root / f"job_profile_{case.label}.sh"
    use_slurm = shutil.which("sbatch") is not None

    if use_slurm:
        case_commands = build_case_commands(
            case,
            case_output_root,
            venv_path,
            launcher=detect_launcher(),
            use_modules=has_module_system(),
        )
        # Create a SLURM script for this case using the cluster preset and case-specific commands
        cluster_preset = case.cluster_presets[detect_cluster_name(case.cluster_presets)]
        script = SlurmScript(
            job_name=f"profiling_{case.label}",
            ntasks_per_node=max(case.ranks),
            custom_commands=case_commands,
            **cluster_preset,
        )
        script_text = str(script)
        script_pragmas = script.to_dict()
    else:
        # No batch system: run the very same commands directly on this machine.
        case_commands = build_case_commands(
            case,
            case_output_root,
            venv_path,
            launcher=detect_launcher(),
            use_modules= False, # has_module_system(),
            ranks=local_ranks(case.ranks),
        )
        script_text = "\n".join(["#!/bin/bash", *case_commands, ""])
        script_pragmas = None

    print(
        f"Writing metadata for '{case.label}' to {case_output_root / 'profiling_case_info.json'}",
    )
    (case_output_root / "profiling_case_info.json").write_text(
        json.dumps(
            {
                "test_case_identifier": case.label,
                "test_case_name": case.name,
                "test_case_description": case.description,
                "physics_problem": case.physics_problem,
                "struphy_model_used": case.struphy_model_used,
                "struphy_commit": run_commit,
                "compiler": compiler.to_dict(),
                "scheduler": "slurm" if use_slurm else "local",
                "job_script_path": str(script_path),
                "job_script": script_text,
                "slurm_dict": script_pragmas,
                "parameter_file": str(case.params_source),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if use_slurm:
        # Submit the SLURM job and wait for it to complete
        job_id = script.submit_job(script_path)
        print(
            f"Submitted profiling case '{case.label}' as job {job_id}. Waiting for completion...",
        )
        SQueue().wait_until_done(job_id=job_id, poll_interval=10)
    else:
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        print(f"No batch system found; running '{case.label}' locally via {script_path} ...")
        result = subprocess.run(["bash", str(script_path)], check=False, cwd=repo_root)
        if result.returncode != 0:
            print(f"WARNING: local run of '{case.label}' exited with code {result.returncode}.")

    # Package the results of each profiling case and push to the profiling-data repo
    case_output_root = run_results_root / case.label
    # Package only what this job actually produced, so cases that never ran
    # (or failed before writing output) are not packaged/uploaded.
    packaged_dir = package_testcase(
        testcase_dir=case_output_root,
        results_root=run_results_root,
        language=compiler.language,
        commit=run_commit,
        output_root=output_root,
        verbose=True,
    )
    if packaged_dir is not None:
        print(f"Packaged profiling data for '{case.label}' into {packaged_dir}")
        latest_results_root_path.write_text(str(run_results_root), encoding="utf-8")
        print(f"Updated latest profiling root marker: {latest_results_root_path}")

        print(f"Packaged profiling data for '{case.label}' into {output_root}:")
        print(f" - {packaged_dir}")
        if args.upload:
            print("Uploading packaged profiling data to the profiling-data repo ...")
            _push_profiling_data([packaged_dir], run_commit)
        else:
            print("Upload skipped; use --upload to push the packaged profiling data to the profiling-data repo.")
            print("Plot the results locally by opening the HTML files in the packaged directories, e.g.:")
            print(f"scope-profiler pproc {packaged_dir / '*.h5'} --rank 0")
    else:
        print(f"No profiling output found for '{case.label}'; nothing to package.")
