"""Shared machinery for running profiling jobs.

A concrete profiling job (e.g. ``run_diocotron.py``) defines its own `ProfilingCase`
and drives the run itself, looping over whichever rank counts it wants to profile:

- `ProfilingCase.setup_run` parses CLI args, validates the venv, compiles Struphy, and
  creates the results directories.
- For each rank count, the caller builds `ProfilingCase.build_commands` and either
  constructs its own `SlurmScript` (under SLURM) or writes/launches a plain bash
  script (otherwise) — see `run_diocotron.py` for the loop.
- `ProfilingCase.finalize_run` waits for every rank count to finish, then builds the
  comparison plot and packages/uploads the results, once per case.

The remaining module-level functions are generic helpers with no case-specific
knowledge (installing `whereami`, detecting the MPI launcher/module system, parsing
CLI args, filtering rank counts to what fits locally).

See `run_diocotron.py` for a template.
"""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from package_profiling_results import MACHINE_PARAMS_FILE, detect_machine_name, package_testcase
from slurm_script_generator.squeue import SQueue
from upload import _push_profiling_data

from struphy import Compiler
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


@dataclass(frozen=True)
class ProfilingRunSetup:
    """Everything the caller's per-rank loop and `ProfilingCase.finalize_run` need.

    Built once per profiling run by `ProfilingCase.setup_run`, then reused for every
    rank count in the caller's loop.
    """

    args: argparse.Namespace
    venv_path: Path
    compiler: Compiler
    run_commit: str
    output_root: Path
    run_results_root: Path
    case_output_root: Path
    use_slurm: bool
    use_modules: bool
    launcher: str
    cluster_preset: dict | None


@dataclass(frozen=True)
class ProfilingCase:
    label: str
    name: str
    description: str
    physics_problem: str
    struphy_model_used: str
    params_source: Path
    cluster_presets: dict[str, dict]

    def detect_cluster_name(self) -> str:
        """Pick the cluster preset for the current machine.

        The detected machine name ("Pitagora (DCGP)", "TOK", ...) is matched against
        the preset keys ("pitagora", "tok", ...). Falls back to `default_cluster_name`
        when the machine is unknown or has no preset (e.g. when submitting from a
        laptop), so submission behaves as before.
        """
        machine_name = detect_machine_name()
        if machine_name:
            for preset_name in self.cluster_presets:
                if preset_name.lower() in machine_name.lower():
                    print(f"Detected machine '{machine_name}'; using cluster preset '{preset_name}'.")
                    return preset_name

        print(
            f"No cluster preset matches the detected machine ({machine_name!r}); using '{default_cluster_name}'.",
        )
        return default_cluster_name

    def build_commands(
        self,
        output_root: Path,
        venv_path: Path,
        ntasks: int,
        *,
        launcher: str = "srun",
        use_modules: bool = True,
    ) -> list[str]:
        """Build the shell commands that run this case with a single MPI rank count.

        One script is built (and submitted) per rank count, by the caller's loop over
        rank counts, so this only ever covers one `ntasks` value. The comparison plot
        across rank counts is built separately, once every rank count has finished
        running.
        """
        activate_path = venv_path / "bin" / "activate"
        sim_dir = output_root / f"sim_ranks{ntasks}"
        h5_file = sim_dir / "profiling_data.h5"

        return [
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
            f'echo "Running profiling case: {self.label} ({ntasks} MPI ranks)"',
            f'echo "Description: {self.description}"',
            f'echo "Physics problem: {self.physics_problem}"',
            f'echo "Struphy model used: {self.struphy_model_used}"',
            f'echo "Case directory: {output_root}"',
            'echo "----------------------------------------"',
            f'mkdir -p "{output_root}"',
            f'cp "{self.params_source}" "{output_root / "parameters.py"}"',
            # Record the machine parameters of the node this job runs on. `whereami`
            # was installed into the venv by `install_whereami` before submission,
            # since compute nodes have no network access.
            f'whereami --output "{output_root / MACHINE_PARAMS_FILE}"',
            f'ls -l "{output_root}"',
            "",
            f'echo "Running {self.label} with {ntasks} MPI ranks"',
            f'cd "{output_root}"',
            f"{launcher} -n {ntasks} python {self.params_source}",
            # f'scope-profiler pproc "{h5_file}" -o "{sim_dir}"',
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {self.label} ({ntasks} MPI ranks)"',
            'echo "----------------------------------------"',
        ]

    def setup_run(self) -> ProfilingRunSetup:
        """Parse CLI args, validate the venv, compile Struphy, and create the results dirs.

        Called once per profiling run, before looping over rank counts.
        """

        # Parse command-line arguments and validate the virtual environment
        args = build_arg_parser().parse_args()

        # Validate that a virtual environment is active
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if not virtual_env:
            raise RuntimeError(
                "VIRTUAL_ENV is not set; activate a virtual environment before submitting the job.",
            )
        venv_path = Path(virtual_env)

        # Install whereami here, on the login node: the compute nodes running the job
        # have no outbound network access.
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

        case_output_root = run_results_root / self.label
        case_output_root.mkdir(parents=True, exist_ok=True)

        use_slurm = shutil.which("sbatch") is not None
        cluster_preset = self.cluster_presets[self.detect_cluster_name()] if use_slurm else None

        return ProfilingRunSetup(
            args=args,
            venv_path=venv_path,
            compiler=compiler,
            run_commit=run_commit,
            output_root=output_root,
            run_results_root=run_results_root,
            case_output_root=case_output_root,
            use_slurm=use_slurm,
            use_modules=use_slurm and has_module_system(),
            launcher=detect_launcher(),
            cluster_preset=cluster_preset,
        )

    def finalize_run(
        self,
        setup: ProfilingRunSetup,
        job_infos: list[dict],
        job_ids: list[int],
        local_processes: list[tuple[int, subprocess.Popen, Path]],
    ) -> None:
        """Wait for every rank count to finish, then build comparison plots and package/push results.

        Called once per case, after every rank count has been submitted/launched.
        """
        print(
            f"Writing metadata for '{self.label}' to {setup.case_output_root / 'profiling_case_info.json'}",
        )
        (setup.case_output_root / "profiling_case_info.json").write_text(
            json.dumps(
                {
                    "test_case_identifier": self.label,
                    "test_case_name": self.name,
                    "test_case_description": self.description,
                    "physics_problem": self.physics_problem,
                    "struphy_model_used": self.struphy_model_used,
                    "struphy_commit": setup.run_commit,
                    "compiler": setup.compiler.to_dict(),
                    "scheduler": "slurm" if setup.use_slurm else "local",
                    "parameter_file": str(self.params_source),
                    "jobs": job_infos,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Wait for every rank count to finish before packaging anything.
        if setup.use_slurm:
            print(
                f"Submitted {len(job_ids)} job(s) for '{self.label}'. Waiting for all of them to complete...",
            )
            SQueue().wait_until_done(job_id=job_ids, poll_interval=10)
        else:
            print(f"Waiting for {len(local_processes)} local run(s) of '{self.label}' to complete...")
            for ntasks, process, script_path in local_processes:
                returncode = process.wait()
                if returncode != 0:
                    print(
                        f"WARNING: local run of '{self.label}' ({ntasks} MPI ranks) via {script_path} "
                        f"exited with code {returncode}.",
                    )

        # Comparison plot across all rank counts, now that every one of them has finished.
        h5_files = sorted(setup.case_output_root.glob("sim_ranks*/profiling_data.h5"))
        if h5_files:
            figures_dir = setup.case_output_root / "figures"
            subprocess.run(
                [
                    "scope-profiler",
                    "pproc",
                    *[str(h5_file) for h5_file in h5_files],
                    "--rank",
                    "0",
                    "-o",
                    str(figures_dir),
                ],
                check=False,
            )
        else:
            print(f"No profiling_data.h5 produced for '{self.label}'; skipping comparison plots.")

        # Package the results of this profiling case and push to the profiling-data repo.
        # Packaging only what this job actually produced means a case that never ran
        # (or failed before writing output) is not packaged/uploaded.
        packaged_dir = package_testcase(
            testcase_dir=setup.case_output_root,
            results_root=setup.run_results_root,
            language=setup.compiler.language,
            commit=setup.run_commit,
            output_root=setup.output_root,
            verbose=True,
        )
        if packaged_dir is not None:
            print(f"Packaged profiling data for '{self.label}' into {packaged_dir}")
            latest_results_root_path.write_text(str(setup.run_results_root), encoding="utf-8")
            print(f"Updated latest profiling root marker: {latest_results_root_path}")

            print(f"Packaged profiling data for '{self.label}' into {setup.output_root}:")
            print(f" - {packaged_dir}")
            if setup.args.upload:
                print("Uploading packaged profiling data to the profiling-data repo ...")
                _push_profiling_data([packaged_dir], setup.run_commit)
            else:
                print("Upload skipped; use --upload to push the packaged profiling data to the profiling-data repo.")
                print("Plot the results locally by opening the HTML files in the packaged directories, e.g.:")
                print(f"scope-profiler pproc {packaged_dir / '*.h5'} --rank 0")
        else:
            print(f"No profiling output found for '{self.label}'; nothing to package.")
