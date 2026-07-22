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


@dataclass(frozen=True)
class ProfilingCase:
    label: str
    name: str
    description: str
    physics_problem: str
    struphy_model_used: str
    ranks: tuple[int, ...]
    output_root: Path
    params_source: Path


script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
profiling_results_base = repo_root / "results" / "profiling"
latest_results_root_path = profiling_results_base / "latest_run_root.txt"


def _make_unique_results_root(base_dir: Path, run_token: str) -> Path:
    candidate = base_dir / run_token
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = base_dir / f"{run_token}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _git_commit_short(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=8", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _push_profiling_data(packaged_dirs: list[Path], run_commit: str) -> None:
    """Push newly packaged profiling data folders to the struphy-hub/profiling-data repo."""
    if not packaged_dirs:
        print("No packaged profiling data to push; skipping profiling-data repo push.")
        return

    repo_url = os.environ.get("PROFILING_DATA_REPO_URL", "git@github.com:struphy-hub/profiling-data.git")

    with tempfile.TemporaryDirectory(prefix="profiling-data-") as clone_dir_str:
        clone_dir = Path(clone_dir_str)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
        )

        for packaged_dir in packaged_dirs:
            destination = clone_dir / packaged_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(packaged_dir, destination)

        subprocess.run(["git", "-C", str(clone_dir), "add", "."], check=True)
        status = subprocess.run(["git", "-C", str(clone_dir), "diff", "--cached", "--quiet"])
        if status.returncode == 0:
            print("No changes to push to profiling-data repo.")
            return

        subprocess.run(
            [
                "git",
                "-C",
                str(clone_dir),
                "commit",
                "-m",
                f"Add profiling data for struphy commit {run_commit[:8]}",
            ],
            check=True,
        )

        for attempt in range(5):
            push_result = subprocess.run(["git", "-C", str(clone_dir), "push"])
            if push_result.returncode == 0:
                print(f"Pushed {len(packaged_dirs)} profiling data folder(s) to {repo_url}.")
                return
            print(f"Push attempt {attempt + 1} failed; fetching and rebasing before retrying.")
            subprocess.run(["git", "-C", str(clone_dir), "fetch", "origin"], check=True)
            subprocess.run(["git", "-C", str(clone_dir), "rebase", "origin/HEAD"], check=True)
            time.sleep(random.randint(1, 5))

        raise RuntimeError("Failed to push profiling data to profiling-data repo after retries.")


def build_case_commands(case: ProfilingCase, venv_path: Path) -> list[str]:
    activate_path = venv_path / "bin" / "activate"
    commands = [
        "module purge",
        "source ./setup/modules.sh load",
        "module list",
        f"source {str(activate_path)}",
        'echo "----------------------------------------"',
        f'echo "Running profiling case: {case.label}"',
        f'echo "Description: {case.description}"',
        f'echo "Physics problem: {case.physics_problem}"',
        f'echo "Struphy model used: {case.struphy_model_used}"',
        f'echo "Case directory: {case.output_root}"',
        "pwd",
        'echo "----------------------------------------"',
        f'mkdir -p "{case.output_root}"',
        f'cp "{case.params_source}" "{case.output_root / "parameters.py"}"',
        f'ls -l "{case.output_root}"',
        'echo "===== Diagnostics ====="',
        "pwd",
        "which python",
        "python --version",
        "which mpirun",
        'echo "VIRTUAL_ENV=$VIRTUAL_ENV"',
        'echo "PATH=$PATH"',
        'echo "======================="',
        "srun -n1 hostname",
        "srun -n1 python -c \"print('hello')\"",
        "srun -n1 mpirun -n1 hostname",
    ]

    commands.append("existing_h5_files=()")

    for ntasks in case.ranks:
        sim_dir = case.output_root / f"sim_ranks{ntasks}"
        h5_file = sim_dir / "profiling_data.h5"
        mpirun_log = case.output_root / f"mpirun_ranks{ntasks}.log"
        commands.extend(
            [
                "",
                f'echo "Running {case.label} with {ntasks} MPI ranks"',
                (
                    f"if srun -n {ntasks} python profiling/run_diocotron.py "
                    f'{ntasks} --out-root "{case.output_root}" > "{mpirun_log}" 2>&1; then'
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
            'if [ "${#existing_h5_files[@]}" -gt 0 ]; then',
            f'    scope-profiler pproc "${{existing_h5_files[@]}}" --rank 0 -o "{case.output_root / "figures"}"',
            "else",
            '    echo "No profiling data was produced for any rank count; skipping comparison plots."',
            "fi",
        ]
    )

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the diocotron profiling job.")
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
    args = parser.parse_args()

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError("VIRTUAL_ENV is not set; activate a virtual environment before submitting the job.")
    venv_path = Path(virtual_env)

    compiler = Compiler(language=args.language, compiler=args.compiler)
    print(f"Using compiler: {compiler.language} ({compiler.compiler})")
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

    cases = [
        ProfilingCase(
            label="diocotron_poisson_scaling",
            name="Diocotron Poisson scaling",
            description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
            physics_problem="Diocotron instability in a non-neutral plasma.",
            struphy_model_used="ToyDrift",
            ranks=(1, 2, 4),  # , 8),
            output_root=run_results_root / "diocotron_poisson_scaling",
            params_source=(repo_root / "examples" / "ToyGyrokinetic" / "diocotron_instability" / "params_diocotron.py"),
        ),
    ]

    packaged_dirs: list[Path] = []

    for case in cases:
        case.output_root.mkdir(parents=True, exist_ok=True)
        case_commands = build_case_commands(case, venv_path)

        # TOK
        # script = SlurmScript(
        #     job_name=f"profiling_{case.label}",
        #     nodes=1,
        #     ntasks_per_node=max(case.ranks),
        #     cpus_per_task=1,
        #     mem_per_cpu="1GB",
        #     partition="s.tok",
        #     qos="tok.debug",
        #     output="./%x.%j.out",
        #     error="./%x.%j.err",
        #     chdir="./",
        #     mail_type="none",
        #     time="00:15:00",
        #     custom_commands=case_commands,
        # )

        # Pitagora
        script = SlurmScript(
            job_name=f"profiling_{case.label}",
            nodes=1,
            ntasks_per_node=max(case.ranks),
            cpus_per_task=1,
            mem_per_cpu="8GB",
            partition="dcgp_fua_dbg",
            account="FUSIO_HLST_7",
            output="./%x.%j.out",
            error="./%x.%j.err",
            mail_type="none",
            time="00:15:00",
            custom_commands=case_commands,
        )

        # print(script)

        output_path = repo_root / f"job_profile_{case.label}.sh"

        print(f"Writing metadata for '{case.label}' to {case.output_root / 'profiling_case_info.json'}")
        (case.output_root / "profiling_case_info.json").write_text(
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
        print(f"Saved SLURM script for '{case.label}' to {output_path} from {os.getcwd()}")
        # result = subprocess.run(["sbatch", str(output_path)], capture_output=True, text=True)

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

        # job_id = script.submit_job(str(output_path), verbose=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted profiling case '{case.label}' as job {job_id}. Waiting for completion...")

        SQueue().wait_until_done(job_id=job_id, poll_interval=10)
        # SQueue().wait_until_done(job_name="profiling_*", poll_interval=10)

        print(f"Profiling case '{case.label}' completed. Output saved in {case.output_root}")

        # Package only what this job actually produced, so cases that never ran
        # (or failed before writing output) are not packaged/uploaded.
        packaged_dir = package_testcase(
            testcase_dir=case.output_root,
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


if __name__ == "__main__":
    main()
