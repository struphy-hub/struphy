"""Shared machinery for running profiling jobs.

A concrete profiling job (e.g. ``profile_diocotron_scaling.py``) defines its own `ProfilingCase`
and drives the run itself, looping over whichever rank counts it wants to profile.

- `ProfilingCase.setup_run` validates the venv, compiles Struphy, and creates the
  results directories. It stores everything it computes as attributes on the case
  itself (`venv_path`, `use_slurm`, `launcher`, `case_output_root`, ...), so there's
  a single object to thread through the rest of the run.
- For each rank count, the caller calls `ProfilingCase.launch(ntasks)`, which builds
  the per-rank shell commands and either submits a `SlurmScript` (under SLURM) or
  writes/launches a plain bash script (otherwise) — the caller doesn't need to know
  which. Pass `case_commands` to override the default commands (e.g. to add a
  case-specific flag such as an arbitrary `ppc`) before they're wrapped in a script.
  Under SLURM, the cluster preset is picked on each call from
  `clusters.SLURM_PRESETS` unless a `slurm_presets` argument overrides it.
- `ProfilingCase.finalize_run` waits for every rank count to finish, then builds the
  comparison plot and packages/uploads the results, once per case.

The remaining module-level functions are generic helpers with no case-specific
knowledge (detecting the MPI launcher/module system).

See `profile_diocotron_scaling.py` for a template.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from clusters import SLURM_PRESETS, HARDWARE_INFO, detect_machine_name
from package_profiling_results import package_testcase
from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue
from upload import _push_profiling_data

from struphy import Compiler
from utils import _git_commit, _git_commit_short, _make_unique_results_root

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
profiling_results_base = repo_root / "results" / "profiling"

default_cluster_name = "pitagora"


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


@dataclass
class ProfilingCase:
    label: str
    name: str
    description: str
    physics_problem: str
    struphy_model_used: str
    params_source: Path
    language: str = "fortran"  # Pyccel language to compile the Struphy kernels with: "fortran" or "c".
    compiler: str = "GNU"  # Pyccel compiler family: "GNU", "intel", "PGI", "nvidia", or "LLVM".

    # Populated by `setup_run`; unset (None) until then. Not constructor arguments.
    venv_path: Path | None = field(init=False, default=None)
    compiler_instance: Compiler | None = field(init=False, default=None)
    run_commit: str | None = field(init=False, default=None)
    output_root: Path | None = field(init=False, default=None)
    run_results_root: Path | None = field(init=False, default=None)
    case_output_root: Path | None = field(init=False, default=None)
    use_slurm: bool | None = field(init=False, default=None)
    use_modules: bool | None = field(init=False, default=None)
    launcher: str | None = field(init=False, default=None)

    # Populated by `launch`, one entry per rank count; read by `finalize_run`.
    job_infos: list[dict] = field(init=False, default_factory=list)
    job_ids: list[int] = field(init=False, default_factory=list)
    local_processes: list[tuple[int, subprocess.Popen, Path]] = field(init=False, default_factory=list)

    # Incremented on every `launch` call, to give each run a unique script filename.
    launch_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.setup_run()

    def build_commands(self, ntasks: int, param_flags: list[str] | None = None) -> list[str]:
        """Build the shell commands that run this case with a single MPI rank count.

        Reads the setup computed by `setup_run` (`case_output_root`, `venv_path`,
        `launcher`, `use_modules`). One script is built (and submitted) per rank
        count, by the caller's loop over rank counts, so this only ever covers one
        `ntasks` value. The comparison plot across rank counts is built separately,
        once every rank count has finished running.

        Args:
            ntasks: Number of MPI ranks to run `params_source` with.
            param_flags: Extra CLI flags appended to the `params_source` invocation,
                e.g. `["--ppc", "10"]`. Omit for none.

        Returns:
            The shell commands to run, in order, as one script.
        """
        output_root = self.case_output_root
        activate_path = self.venv_path / "bin" / "activate"
        sim_dir = output_root / f"sim_ranks{ntasks}"
        h5_file = sim_dir / "profiling_data.h5"
        flags = " ".join(param_flags or [])

        return [
            # Environment modules only exist on the clusters, not on a laptop.
            *(
                [
                    "module purge",
                    "source ./setup/modules.sh load",
                    "module list",
                ]
                if self.use_modules
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
            f'ls -l "{output_root}"',
            "",
            f'echo "Running {self.label} with {ntasks} MPI ranks"',
            f'cd "{output_root}"',
            f"{self.launcher} -n {ntasks} python {self.params_source} {flags}",
            "",
            'echo "----------------------------------------"',
            f'echo "Completed profiling case: {self.label} ({ntasks} MPI ranks)"',
            'echo "----------------------------------------"',
        ]

    def launch(
        self,
        num_tasks: int = 1,
        num_nodes: int | None = None,
        param_flags: list[str] | None = None,
        slurm_presets: dict[str, dict] | None = None,
    ) -> None:
        """Build, submit/launch, and record the run for a single rank count.

        Under SLURM, submits a `SlurmScript`; otherwise writes and runs a plain bash
        script locally. Either way, the resulting job/process is recorded on `self`
        (`job_infos`, plus `job_ids` or `local_processes`) for `finalize_run` to wait
        on and package. Each call gets a unique script filename via `launch_count`.

        Without SLURM, `num_tasks` oversubscribing the local machine simply makes
        `mpirun` refuse to start, so a run that exceeds the number of local cores is
        skipped instead of being launched (a case designed for a 64-core node still
        runs its small rank counts on a laptop).

        Args:
            num_tasks: Number of MPI ranks to run with.
            num_nodes: Number of SLURM nodes to spread `num_tasks` ranks across (the
                cluster preset's own node count is overridden with this). Ignored
                outside SLURM, where every rank runs in a single local process group.
            param_flags: Extra CLI flags forwarded to `build_commands` and appended
                to the `params_source` invocation, e.g. `["--ppc", "10"]`. Ignored if
                `case_commands` is given.
            slurm_presets: Candidate SLURM presets, keyed by cluster name; one is
                picked via `detect_machine_name` on every call. Defaults to
                `clusters.SLURM_PRESETS`. Ignored outside SLURM.

        Raises:
            ValueError: If `num_tasks` is not evenly divisible by `num_nodes`
                (SLURM only).
        """
        if not self.use_slurm:
            available = os.cpu_count() or 1
            if num_tasks > available:
                print(f"Skipping '{self.label}' ({num_tasks} MPI ranks): exceeds the {available} local cores.")
                return

        # Build the commands to run this case with the given rank count, and write/submit
        case_commands = self.build_commands(num_tasks, param_flags)

        # Increment the launch count and build a unique script filename for this run.
        self.launch_count += 1
        script_path = repo_root / f"job_profile_{self.label}_{self.launch_count:02d}.sh"

        # Submit a SLURM script or run a local bash script, depending on whether SLURM is available.
        if self.use_slurm:
            cluster_name = detect_machine_name()
            if num_nodes is None:
                # Compute the number of nodes needed to run `num_tasks` ranks, given the cluster's hardware.
                cpus_per_node = HARDWARE_INFO[cluster_name]["cpus_per_node"]
                num_nodes = (num_tasks + cpus_per_node - 1) // cpus_per_node
            if num_tasks % num_nodes != 0:
                raise ValueError(f"num_tasks ({num_tasks}) is not evenly divisible by num_nodes ({num_nodes}).")

            # Pick the cluster preset for this run, either from the caller's override or the default.
            if slurm_presets is not None:
                cluster_preset = slurm_presets[cluster_name]
            else:
                cluster_preset = SLURM_PRESETS[cluster_name]

            # Build the slurm script
            script = SlurmScript(
                job_name=f"profiling_{self.label}_ranks{num_tasks}",
                ntasks_per_node=num_tasks // num_nodes,
                custom_commands=case_commands,
                **{**cluster_preset, "nodes": num_nodes},
            )
            script_text = str(script)
            job_id = script.submit_job(str(script_path))
            print(f"Submitted '{self.label}' ({num_tasks} MPI ranks) as job {job_id}.")
            script_dict = script.to_dict()
            self.job_ids.append(job_id)
        else:
            script_text = "\n".join(["#!/bin/bash", *case_commands, ""])
            script_path.write_text(script_text, encoding="utf-8")
            script_path.chmod(0o755)

            print(
                f"No batch system found; launching '{self.label}' ({num_tasks} MPI ranks) "
                f"locally via {script_path} ...",
            )
            process = subprocess.Popen(["bash", str(script_path)], cwd=repo_root)
            script_dict = None
            self.local_processes.append((num_tasks, process, script_path))

        # Record the job/process info for `finalize_run` to wait on and package.
        self.job_infos.append(
            {
                "ranks": num_tasks,
                "job_script_path": str(script_path),
                "job_script": script_text,
                "slurm_dict": script_dict,
            },
        )

    def setup_run(self) -> None:
        """Validate the venv, compile Struphy, and create the results dirs.

        Called once per profiling run, before looping over rank counts. Stores
        everything it computes as attributes on `self` (`venv_path`,
        `compiler_instance`, `run_commit`, `output_root`, `run_results_root`,
        `case_output_root`, `use_slurm`, `use_modules`, `launcher`), for the
        caller's loop and `finalize_run` to read. The cluster preset itself is
        resolved by `launch`, since it needs the cluster presets passed there.

        Raises:
            RuntimeError: If no virtual environment is active, or if no MPI
                launcher (`srun`, `mpirun`, `mpiexec`) can be found.
        """

        # Validate that a virtual environment is active
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if not virtual_env:
            raise RuntimeError(
                "VIRTUAL_ENV is not set; activate a virtual environment before submitting the job.",
            )
        self.venv_path = Path(virtual_env)

        # Compile Struphy kernels with the case's language and compiler
        self.compiler_instance = Compiler(language=self.language, compiler=self.compiler)
        if not self.compiler_instance.compiled(language=self.language):
            print("Compiling Struphy kernels ...")
            self.compiler_instance.compile()
        print("Done compiling Struphy kernels.")

        # Create a unique results root for this profiling run. Its name is prefixed with a
        # sortable timestamp, so the latest run is discoverable later as the lexicographically
        # greatest subdirectory of `profiling_results_base` (see `utils.latest_run_root`).
        self.output_root = Path("profiling-results-export").resolve()
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        profiling_results_base.mkdir(parents=True, exist_ok=True)

        # Determine the current git commit hash for the Struphy repo
        self.run_commit = _git_commit(repo_root)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_token = f"{timestamp}-{_git_commit_short(repo_root)}"
        self.run_results_root = _make_unique_results_root(profiling_results_base, run_token)

        self.run_results_root.mkdir(parents=True, exist_ok=True)
        print(f"Profiling run root: {self.run_results_root}")

        self.case_output_root = self.run_results_root / self.label
        self.case_output_root.mkdir(parents=True, exist_ok=True)

        self.use_slurm = shutil.which("sbatch") is not None
        self.use_modules = self.use_slurm and has_module_system()
        self.launcher = detect_launcher()

    def case_info_dict(self) -> dict:
        """This case's metadata plus every submitted/launched job, as written to `profiling_case_info.json`.

        Shared between `finalize_run` (which writes it to disk for later, standalone
        packaging via `package_profiling_results.py`) and the immediate in-process
        packaging call in `finalize_run` itself.
        """
        return {
            "test_case_identifier": self.label,
            "test_case_name": self.name,
            "test_case_description": self.description,
            "physics_problem": self.physics_problem,
            "struphy_model_used": self.struphy_model_used,
            "struphy_commit": self.run_commit,
            "compiler": self.compiler_instance.to_dict(),
            "scheduler": "slurm" if self.use_slurm else "local",
            "parameter_file": str(self.params_source),
            "jobs": self.job_infos,
        }

    def finalize_run(self, upload: bool = False) -> None:
        """Wait for every rank count to finish, then build comparison plots and package/push results.

        Called once per case, after every rank count has been submitted/launched via
        `launch`. Writes `profiling_case_info.json` (case metadata plus `job_infos`)
        into `case_output_root`, blocks until every SLURM job (`job_ids`) or local
        process (`local_processes`) has finished, builds a comparison plot across
        rank counts from the `profiling_data.h5` files produced (if any), and then
        packages the case's results, pushing them to the profiling-data repo if
        `upload` is set. Does nothing beyond writing the metadata file if no output
        was produced.

        Args:
            upload: Whether to push the packaged results to the profiling-data repo.
        """
        print(
            f"Writing metadata for '{self.label}' to {self.case_output_root / 'profiling_case_info.json'}",
        )
        case_info = self.case_info_dict()
        (self.case_output_root / "profiling_case_info.json").write_text(
            json.dumps(case_info, indent=2),
            encoding="utf-8",
        )

        # Wait for every rank count to finish before packaging anything.
        if self.use_slurm:
            print(
                f"Submitted {len(self.job_ids)} job(s) for '{self.label}'. Waiting for all of them to complete...",
            )
            SQueue().wait_until_done(job_id=self.job_ids, poll_interval=10)
        else:
            print(f"Waiting for {len(self.local_processes)} local run(s) of '{self.label}' to complete...")
            for ntasks, process, script_path in self.local_processes:
                returncode = process.wait()
                if returncode != 0:
                    print(
                        f"WARNING: local run of '{self.label}' ({ntasks} MPI ranks) via {script_path} "
                        f"exited with code {returncode}.",
                    )

        # Comparison plot across all rank counts, now that every one of them has finished.
        h5_files = sorted(self.case_output_root.glob("sim_ranks*/profiling_data.h5"))
        if h5_files:
            figures_dir = self.case_output_root / "figures"
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
            testcase_dir=self.case_output_root,
            results_root=self.run_results_root,
            language=self.compiler_instance.language,
            commit=self.run_commit,
            output_root=self.output_root,
            verbose=True,
            case_info=case_info,
        )
        if packaged_dir is not None:
            print(f"Packaged profiling data for '{self.label}' into {packaged_dir}")
            print(f"Packaged profiling data for '{self.label}' into {self.output_root}:")
            print(f" - {packaged_dir}")
            if upload:
                print("Uploading packaged profiling data to the profiling-data repo ...")
                _push_profiling_data([packaged_dir], self.run_commit)
            else:
                print("Upload skipped; use --upload to push the packaged profiling data to the profiling-data repo.")
                print("Plot the results locally by opening the HTML files in the packaged directories, e.g.:")
                print(f"scope-profiler pproc {packaged_dir / '*.h5'} --rank 0")
        else:
            print(f"No profiling output found for '{self.label}'; nothing to package.")
