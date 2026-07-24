"""Shared machinery for submitting profiling jobs to a SLURM cluster.

A concrete profiling job (e.g. ``submit_diocotron_job.py``) only needs to define
its own list of `ProfilingCase` objects and call `run_profiling_job`.
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

from package_profiling_results import package_testcase
from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue

from struphy import Compiler

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
profiling_results_base = repo_root / "results" / "profiling"
latest_results_root_path = profiling_results_base / "latest_run_root.txt"

# Static SLURM settings per cluster. `job_name`, `ntasks_per_node`, and
# `custom_commands` are filled in per profiling case at submission time.
CLUSTER_PRESETS: dict[str, dict] = {
    "pitagora": {
        "nodes": 1,
        "cpus_per_task": 1,
        "mem_per_cpu": "8GB",
        "partition": "dcgp_fua_dbg",
        "account": "FUSIO_HLST_7",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    "tok": {
        "nodes": 1,
        "cpus_per_task": 1,
        "mem_per_cpu": "1GB",
        "partition": "s.tok",
        "qos": "tok.debug",
        "chdir": "./",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
}


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
    """Python script invoked as `srun -n <ntasks> python {run_script} <ntasks> --out-root ...`.

    Must accept a positional `nranks` argument and an `--out-root` option, as
    `profiling/run_diocotron.py` does. Each case's output goes under
    `<run_results_root>/<case.label>`, created by `run_profiling_job`.
    """


from upload import _push_profiling_data

from utils import _git_commit, _git_commit_short, _make_unique_results_root


def build_case_commands(
    case: ProfilingCase, output_root: Path, venv_path: Path
) -> list[str]:
    activate_path = venv_path / "bin" / "activate"
    commands = [
        "module purge",
        "source ./setup/modules.sh load",
        "module list",
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
        f'ls -l "{output_root}"',
    ]

    commands.append("existing_h5_files=()")

    for ntasks in case.ranks:
        sim_dir = output_root / f"sim_ranks{ntasks}"
        h5_file = sim_dir / "profiling_data.h5"
        mpirun_log = output_root / f"mpirun_ranks{ntasks}.log"
        commands.extend(
            [
                "",
                f'echo "Running {case.label} with {ntasks} MPI ranks"',
                (
                    f"if srun -n {ntasks} python {case.run_script} "
                    f'{ntasks} --out-root "{output_root}" > "{mpirun_log}" 2>&1; then'
                ),
                f'    echo "srun ({ntasks} ranks) succeeded"',
                "else",
                f'    echo "srun ({ntasks} ranks) FAILED with exit code $?; log follows:"',
                f'    cat "{mpirun_log}"',
                "fi",
                f'if [ -f "{h5_file}" ]; then',
                f'    scope-profiler pproc "{h5_file}" -o "{sim_dir}"',
                f'    existing_h5_files+=("{h5_file}")',
                "else",
                f'    echo "No profiling data found at {h5_file}; skipping scope-profiler pproc for this rank count."',
                "fi",
            ],
        )

    commands.extend(
        [
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {case.label}"',
            'echo "----------------------------------------"',
            "# Postprocessing comparison plots",
            f'scope-profiler pproc "${{existing_h5_files[@]}}" --rank 0 -o "{output_root / "figures"}"',
        ]
    )

    return commands

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        "--cluster",
        type=str,
        default="pitagora",
        choices=sorted(CLUSTER_PRESETS),
        help='SLURM cluster preset to submit to: "pitagora" (default) or "tok".',
    )
    return parser


def run_profiling_job(cases: list[ProfilingCase], description: str) -> None:
    """Compile Struphy, submit `cases` as SLURM jobs one by one, and package/push the results."""
    args = build_arg_parser(description).parse_args()

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError(
            "VIRTUAL_ENV is not set; activate a virtual environment before submitting the job."
        )
    venv_path = Path(virtual_env)

    compiler = Compiler(language=args.language, compiler=args.compiler)
    compiler.compile()
    print("Done compiling Struphy kernels.")

    output_root = Path("profiling-results-export").resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profiling_results_base.mkdir(parents=True, exist_ok=True)
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

    cluster_preset = CLUSTER_PRESETS[args.cluster]
    packaged_dirs: list[Path] = []

    for case in cases:
        case_output_root = run_results_root / case.label
        case_output_root.mkdir(parents=True, exist_ok=True)
        case_commands = build_case_commands(case, case_output_root, venv_path)

        script = SlurmScript(
            job_name=f"profiling_{case.label}",
            ntasks_per_node=max(case.ranks),
            custom_commands=case_commands,
            **cluster_preset,
        )

        output_path = repo_root / f"job_profile_{case.label}.sh"

        print(
            f"Writing metadata for '{case.label}' to {case_output_root / 'profiling_case_info.json'}"
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
                    "output_path": str(output_path),
                    "slurm_script": str(script),
                    "slurm_dict": script.to_dict(),
                    "parameter_file": str(case.params_source),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        script.save(str(output_path))
        print(
            f"Saved SLURM script for '{case.label}' to {output_path} from {os.getcwd()}"
        )

        print("=== Script contents ===")
        print(Path(output_path).read_text())

        result = subprocess.run(
            ["sbatch", "--parsable", str(output_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        print("stdout:", repr(result.stdout))
        print("stderr:", repr(result.stderr))
        print("returncode:", result.returncode)
        print("cwd:", os.getcwd())

        job_id = result.stdout.strip().split()[-1]
        print(
            f"Submitted profiling case '{case.label}' as job {job_id}. Waiting for completion..."
        )

        SQueue().wait_until_done(job_id=job_id, poll_interval=10)

        print(
            f"Profiling case '{case.label}' completed. Output saved in {case_output_root}"
        )

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
            packaged_dirs.append(packaged_dir)
            print(f"Packaged profiling data for '{case.label}' into {packaged_dir}")
        else:
            print(f"No profiling output found for '{case.label}'; nothing to package.")

    latest_results_root_path.write_text(str(run_results_root), encoding="utf-8")
    print(f"Updated latest profiling root marker: {latest_results_root_path}")

    print(f"Packaged {len(packaged_dirs)} profiling case(s) into {output_root}:")
    for packaged_dir in packaged_dirs:
        print(f" - {packaged_dir}")

    _push_profiling_data(packaged_dirs, run_commit)
