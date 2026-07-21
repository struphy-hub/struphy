import argparse
import json
import os
import shutil
import subprocess
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


def build_case_commands(case: ProfilingCase, venv_path: Path) -> list[str]:
    activate_path = venv_path / "bin" / "activate"
    commands = [
        "module purge",
        "source ./setup/modules.sh load",
        "module list",
        f"source {str(activate_path)}",
        'echo "----------------------------------------"',
        f'echo "Running profiling case: {case.label}"',
        'echo "----------------------------------------"',
        f'mkdir -p "{case.output_root}"',
        f'cp "{case.params_source}" "{case.output_root / "parameters.py"}"',
    ]

    for ntasks in case.ranks:
        sim_dir = case.output_root / f"sim_ranks{ntasks}"
        commands.extend(
            [
                "",
                f'echo "Running {case.label} with {ntasks} MPI ranks"',
                (
                    f'mpirun -n {ntasks} python profiling/run_diocotron.py '
                    f'{ntasks} --out-root "{case.output_root}"'
                ),
                f"scope-profiler pproc {sim_dir / 'profiling_data.h5'} -o {sim_dir}",
            ],
        )

    sim_dirs = [case.output_root / f"sim_ranks{ntasks}" for ntasks in case.ranks]
    commands.extend(
        [
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {case.label}"',
            'echo "----------------------------------------"',
            '# Postprocessing comparison plots',
            (
                f"scope-profiler pproc "
                f"{' '.join(str(sim_dir / 'profiling_data.h5') for sim_dir in sim_dirs)} "
                f"--rank 0 -o {case.output_root / 'figures'}"
            ),
        ]
    )

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the diocotron profiling job."
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
    args = parser.parse_args()

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError(
            "VIRTUAL_ENV is not set; activate a virtual environment before submitting the job."
        )
    venv_path = Path(virtual_env)

    compiler = Compiler(language=args.language, compiler=args.compiler)
    compiler.compile()

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
            ranks=(1, 2, 4, 8),
            output_root=run_results_root / "diocotron_poisson_scaling",
            params_source=(
                repo_root
                / "examples"
                / "ToyGyrokinetic"
                / "diocotron_instability"
                / "params_diocotron.py"
            ),
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
            mem_per_cpu="1GB",
            partition="dcgp_fua_dbg",
            account="FUSIO_HLST_7",
            output="./%x.%j.out",
            error="./%x.%j.err",
            chdir="./",
            mail_type="none",
            time="00:15:00",
            custom_commands=case_commands,
        )

        print(script)

        output_path = repo_root / f"job_profile_{case.label}.sh"

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

        job_id = script.submit_job(str(output_path), verbose=True)

        SQueue().wait_until_done(job_id=job_id, poll_interval=10)

        print(f"Profiling case '{case.label}' completed. Output saved in {case.output_root}")

        # Package only what this job actually produced, so cases that never ran
        # (or failed before writing output) are not packaged/uploaded.
        packaged_dir = package_testcase(
            testcase_dir=case.output_root,
            results_root=run_results_root,
            language=compiler.language,
            commit=run_commit,
            output_root=output_root,
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


if __name__ == "__main__":
    main()
