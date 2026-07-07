import subprocess
from dataclasses import dataclass
from pathlib import Path

from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue


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
        # case_dir = output_root / case.name / f"n{ntasks}"
        # log_file = case_dir / "run.log"
        sim_dir = output_root / case.name / f"sim_ranks{ntasks}"
        commands.extend(
            [
                "",
                f'echo "Running {case.name} with {ntasks} MPI ranks"',
                f'mpirun -n {ntasks} {case.command.format(nranks=ntasks)}', # > "{log_file}" 2>&1',
                f"scope-profiler-pproc {sim_dir / 'profiling_data.h5'} -o {sim_dir}",
            ],
        )
    

    sim_dirs = [output_root / case.name / f"sim_ranks{ntasks}" for ntasks in case.ranks]
    commands.extend(
        [
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {case.name}"',
            'echo "----------------------------------------"',
            '# Postprocessing comparison plots',
            f"scope-profiler-pproc {' '.join(str(sim_dir / 'profiling_data.h5') for sim_dir in sim_dirs)} --rank 0 -o {output_root / case.name / 'figures'}"
        ]
    )

    return commands


def main() -> None:
    cases = [
        ProfilingCase(
            name="diocotron_poisson_scaling",
            command="python /toks/work/maxlin/git_repos/struphy/profiling/run_diocotron.py {nranks}",
            ranks=(1, 2, 4, 8),
        ),
    ]

    common_commands = load_common_commands()

    for case in cases:
        case_commands = (
            [f'cd "{repo_root}"'] + common_commands + build_case_commands(case)
        )

        # TOK
        # script = SlurmScript(
        #     job_name=f"profiling_{case.name}",
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
            job_name=f"profiling_{case.name}",
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
        
        output_path = repo_root / f"job_profile_{case.name}.sh"

        job_id = script.submit_job(str(output_path), verbose=True)
        
        print(f"Submitted profiling case '{case.name}' with job ID {job_id}. Waiting for completion...")

        SQueue().wait_until_done(job_id=job_id, poll_interval=10)

        print(f"Profiling case '{case.name}' completed. Output saved in {output_root / case.name}")

if __name__ == "__main__":
    main()
