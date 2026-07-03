from dataclasses import dataclass
from pathlib import Path

from slurm_script_generator.slurm_script import SlurmScript


@dataclass(frozen=True)
class ProfilingCase:
    name: str
    command: str
    ranks: tuple[int, ...]


script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
common_commands_path = script_dir / "common_commands.sh"
output_root = repo_root / "profiling" / "results"


def load_common_commands() -> list[str]:
    return common_commands_path.read_text(encoding="utf-8").splitlines()


def build_case_commands(case: ProfilingCase) -> list[str]:
    commands = [
        'echo "----------------------------------------"',
        f'echo "Running profiling case: {case.name}"',
        'echo "----------------------------------------"',
    ]

    for ntasks in case.ranks:
        case_dir = output_root / case.name / f"n{ntasks}"
        log_file = case_dir / "run.log"
        commands.extend(
            [
                "",
                f'echo "Running {case.name} with {ntasks} MPI ranks"',
                f'mpirun -n {ntasks} {case.command.format(nranks=ntasks)} > "{log_file}" 2>&1',
            ]
        )

    return commands


def main() -> None:
    cases = [
        ProfilingCase(
            name="diocotron_poisson_scaling",
            command="python profiling/run_diocotron.py {nranks}",
            ranks=(1, 2, 4, 8),
        ),
    ]

    common_commands = load_common_commands()

    for case in cases:
        case_commands = (
            [f'cd "{repo_root}"'] + common_commands + build_case_commands(case)
        )
        script = SlurmScript(
            job_name=f"profiling_{case.name}",
            nodes=1,
            ntasks_per_node=max(case.ranks),
            cpus_per_task=1,
            mem_per_cpu="1GB",
            partition="s.tok",
            qos="s.tok.standard",
            stdout="./%x.%j.out",
            stderr="./%x.%j.err",
            chdir="./",
            mail_type="none",
            time="00:15:00",
            custom_commands=case_commands,
        )

        output_path = repo_root / f"job_profile_{case.name}.sh"
        script.save(output_path)
        text = output_path.read_text(encoding="utf-8")
        output_path.write_text(text.replace("#!/bin/bash\n", "#!/bin/bash -l\n", 1), encoding="utf-8")
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
