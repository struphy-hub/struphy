"""Shared machinery for running profiling jobs.

A concrete profiling job (e.g. ``profile_diocotron_scaling.py``) defines its own `ProfilingCase`
and drives the run itself, looping over whichever rank counts it wants to profile.

- `ProfilingCase.setup_run` validates the venv, compiles Struphy, and creates the
  results directories. It stores everything it computes as attributes on the case
  itself (`venv_path`, `use_slurm`, `launcher`, `case_output_root`, ...), so there's
  a single object to thread through the rest of the run.
- For each rank count, the caller calls `ProfilingCase.launch(ntasks, param_flags=None)`, which builds
  the per-rank shell commands and either submits a `SlurmScript` (under SLURM) or
  writes and runs a plain bash script to completion (otherwise) — the caller doesn't
  need to know which. Pass `param_flags` to override the default commands (e.g. to add a
  case-specific flag such as an arbitrary `ppc`) before they're wrapped in a script.
  Under SLURM, the cluster preset is picked on each call from
  `clusters.SLURM_PRESETS` unless a `slurm_presets` argument overrides it.
- `ProfilingCase.finalize_run` packages and pushes the case-level metadata straight
  away, then waits on the runs one by one, packaging and pushing each run's results
  into that same folder as soon as its own job finishes. With `upload=True`, the
  packaged folder lives inside a clone of the profiling-data repo (made by
  `setup_run`), so a push is just a commit there — results are never staged in a
  separate export folder first. Runs are identified by their launch id throughout —
  job scripts, SLURM job/log names, run directories and packaged files all carry it,
  and nothing is named after its rank count.

The remaining module-level functions are generic helpers with no case-specific
knowledge (detecting the MPI launcher/module system).

See `profile_diocotron_scaling.py` for a template.
"""

import getpass

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from clusters import HARDWARE_INFO, SLURM_PRESETS, detect_machine_name
from package_profiling_results import (
    RESULTS_DIR_NAME,
    _build_output_name,
    _collect_hardware_info,
    _collect_software_info,
    _copy_run_results,
    _ensure_testcase_parameters_file,
    _read_sim_metadata_from_parameters,
    _run_folder_name,
    _write_run_metadata,
)
from slurm_script_generator.slurm_script import SlurmScript
from slurm_script_generator.squeue import SQueue, job_states
from upload import _clone_profiling_data, _push_profiling_data

from struphy import Compiler
from utils import _git_commit, _git_commit_short, _make_unique_results_root, _slug

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
profiling_results_base = repo_root / "_profiling_jobs"

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
    # Whether the packaged results are pushed to the profiling-data repo. Set here rather
    # than at `finalize_run` time because `setup_run` clones that repo up front, so a
    # missing/unreachable repo fails before any job is submitted rather than after.
    upload: bool = False

    # Populated by `setup_run`; unset (None) until then. Not constructor arguments.
    venv_path: Path | None = field(init=False, default=None)
    compiler_instance: Compiler | None = field(init=False, default=None)
    run_commit: str | None = field(init=False, default=None)
    run_timestamp: datetime | None = field(init=False, default=None)
    output_root: Path | None = field(init=False, default=None)
    run_results_root: Path | None = field(init=False, default=None)
    case_output_root: Path | None = field(init=False, default=None)
    use_slurm: bool | None = field(init=False, default=None)
    use_modules: bool | None = field(init=False, default=None)
    launcher: str | None = field(init=False, default=None)

    # Packaging state, grown incrementally by `package_run` as each job finishes and
    # read by `package_case_metadata` when it (re)writes `case_metadata.json`.
    destination_dir: Path | None = field(init=False, default=None)
    packaged_runs: list[dict] = field(init=False, default_factory=list)

    # Populated by `launch`, one entry per launch; read by `finalize_run`.
    job_infos: list[dict] = field(init=False, default_factory=list)

    # Incremented on every `launch` call, to give each run a unique script filename.
    launch_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.setup_run()

    def build_commands(self, ntasks: int, param_flags: list[str] | None = None) -> list[str]:
        """Build the shell commands that run this case with a single MPI rank count.

        Reads the setup computed by `setup_run` (`case_output_root`, `venv_path`,
        `launcher`, `use_modules`). One script is built (and submitted) per launch, by
        the caller's loop, so this only ever covers one `ntasks` value.

        Args:
            ntasks: Number of MPI ranks to run `params_source` with.
            param_flags: Extra CLI flags appended to the `params_source` invocation,
                e.g. `["--ppc", "10"]`. Omit for none. `--id` is always passed on top
                of these.

        Returns:
            The shell commands to run, in order, as one script.
        """
        output_root = self.case_output_root
        activate_path = self.venv_path / "bin" / "activate"
        # Every run is identified by its launch id alone — nothing in the naming carries
        # the rank count, so two launches that share one stay separate. The rank count is
        # recorded in the run's `run_metadata.json`. `params_source` is passed the same
        # counter as `--id` and names its output folder from it.
        sim_dir = output_root / f"sim_{self.launch_count:02d}"
        python = self.venv_path / "bin" / "python"
        flags = " ".join(["--id", str(self.launch_count), *(param_flags or [])])

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
            "set -e",
            f"source {activate_path!s}",
            'echo "----------------------------------------"',
            f'echo "Running profiling case: {self.label} ({ntasks} MPI ranks)"',
            f'echo "Description: {self.description}"',
            f'echo "Physics problem: {self.physics_problem}"',
            f'echo "Struphy model used: {self.struphy_model_used}"',
            f'echo "Case directory: {output_root}"',
            'echo "----------------------------------------"',
            f'mkdir -p "{sim_dir}"',
            f'cp "{self.params_source}" "{output_root / "parameters.py"}"',
            "",
            f'echo "Running {self.label} with {ntasks} MPI ranks"',
            f'cd "{output_root}"',
            # The run's log lives next to its output, so each run keeps its own record
            # instead of sharing the driver's terminal or the SLURM log. STRUPHY_LOG_FILE
            # is needed on top of the shared cwd above: several rank-count runs of the same
            # case share `output_root` as their cwd (often concurrently, as separate SLURM
            # jobs), and struphy's default relative "struphy.log" would otherwise resolve to
            # the same file for all of them, racing on log rotation across processes.
            f'STRUPHY_LOG_FILE="{sim_dir / "struphy.log"}" '
            f'{self.launcher} -n {ntasks} {python} {self.params_source} {flags} > "{sim_dir / "struphy.out"}" 2>&1',
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
        """Build, submit/run, and record the run for a single rank count.

        Under SLURM, submits a `SlurmScript`; otherwise writes and runs a plain bash
        script locally, blocking until it finishes. Either way, the resulting job is
        recorded on `self` (`job_infos`) for `finalize_run` to wait on and package.
        Each call gets a unique script filename via `launch_count`.

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
                to the `params_source` invocation, e.g. `["--ppc", "10"]`.
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

        # Increment the launch count first: it identifies both the script filename and
        # the run directory that `build_commands` writes into.
        self.launch_count += 1
        script_path = profiling_results_base / f"job_profile_{self.label}_{self.launch_count:02d}.sh"

        # Build the commands to run this case with the given rank count, and write/submit
        case_commands = self.build_commands(num_tasks, param_flags)

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
                # The job name is what `%x` expands to in the preset's output/error
                # paths, so this also names the SLURM log files.
                job_name=f"profiling_{self.label}_{self.launch_count:02d}",
                ntasks_per_node=num_tasks // num_nodes,
                custom_commands=case_commands,
                **{**cluster_preset, "nodes": num_nodes},
            )
            script_text = str(script)
            job_id = script.submit_job(str(script_path))
            print(f"Submitted '{self.label}' ({num_tasks} MPI ranks) as job {job_id}.")
            script_dict = script.to_dict()
        else:
            job_id = None
            script_text = "\n".join(["#!/bin/bash", *case_commands, ""])
            script_path.write_text(script_text, encoding="utf-8")
            script_path.chmod(0o755)

            print(
                f"No batch system found; running '{self.label}' ({num_tasks} MPI ranks) locally via {script_path} ...",
            )

            result = subprocess.run(["bash", script_path], cwd=repo_root, check=False)

            if result.returncode != 0:
                print(
                    f"WARNING: local run of '{self.label}' ({num_tasks} MPI ranks) via {script_path} "
                    f"exited with code {result.returncode}.",
                )
            script_dict = None

        # Record the job/process info for `finalize_run` to wait on and package.
        # `launch_id` identifies this run everywhere: its output directory (`sim_<id>`),
        # its job script, its SLURM job name and its packaged files. `ranks` is metadata.
        job_info = {
            "launch_id": self.launch_count,
            "ranks": num_tasks,
            "job_id": job_id,
            "job_script_path": str(script_path),
            "job_script": script_text,
            "slurm_dict": script_dict,
        }
        self.job_infos.append(job_info)

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

        profiling_results_base.mkdir(parents=True, exist_ok=True)

        # Results are packaged straight into a clone of the profiling-data repo, so
        # pushing a run is just a commit — there is no intermediate export folder to
        # copy out of. Without `--upload` the clone is pointless, so the packaged
        # folders are simply written into the same, plain directory.
        self.output_root = profiling_results_base / "profiling-data"
        if self.upload:
            _clone_profiling_data(self.output_root)
        else:
            if self.output_root.exists():
                shutil.rmtree(self.output_root)
            self.output_root.mkdir(parents=True, exist_ok=True)

        # Determine the current git commit hash for the Struphy repo
        self.run_commit = _git_commit(repo_root)
        # Fixed once per run, so the packaged folder keeps the same name across the
        # incremental pushes `finalize_run` makes as each job finishes.
        self.run_timestamp = datetime.now(UTC)
        timestamp = self.run_timestamp.strftime("%Y%m%dT%H%M%SZ")
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

    def _packaged_language_and_commit(self, case_info: dict) -> tuple[str, str]:
        """The Pyccel language and Struphy commit the packaged folder is named after."""
        testcase = self.case_output_root.name
        case_language = case_info.get("pyccel_language") or self.compiler_instance.language
        case_commit = case_info.get("struphy_commit") or self.run_commit
        if case_language is None:
            raise RuntimeError(
                f"Missing pyccel language for testcase '{testcase}'. "
                "Provide it in profiling_case_info.json or via --language.",
            )
        if case_commit is None:
            raise RuntimeError(
                f"Missing commit hash for testcase '{testcase}'. "
                "Provide it in profiling_case_info.json or via --commit.",
            )
        return case_language, case_commit

    def package_case_metadata(self, case_info: dict) -> Path:
        """Create the packaged folder and (re)write `parameters.py` and `case_metadata.json`.

        `case_metadata.json` describes the case: what was run, on which machine, with
        which software, plus a `runs` list referencing the metadata file of each run
        packaged so far. Everything specific to a single run lives in that file instead.

        Called once before any job has finished — so the case's metadata can be pushed
        while the runs are still queued — and again after every run `package_run` adds,
        to refresh the `runs` list. The folder name is derived from `run_timestamp`,
        fixed in `setup_run`, so every call targets the same folder and each push
        updates it in place.

        `case_info` is `self.case_info_dict()` for the case being packaged.
        """
        testcase = self.case_output_root.name
        case_language, case_commit = self._packaged_language_and_commit(case_info)
        datetime_token = self.run_timestamp.strftime("%Y%m%dT%H%M%SZ")

        if self.destination_dir is None:
            # The Pyccel language is recorded in `case_metadata.json`, not in the name.
            folder_name = f"{datetime_token}-{case_commit[:8]}-{_slug(testcase)}"
            self.destination_dir = self.output_root / folder_name
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        sim_name = testcase
        sim_description = ""
        parameters_path = _ensure_testcase_parameters_file(self.case_output_root)
        if parameters_path is not None:
            sim_name, sim_description = _read_sim_metadata_from_parameters(
                parameters_path=parameters_path,
                fallback_name=testcase,
            )
            shutil.copy2(parameters_path, self.destination_dir / "parameters.py")

        metadata = {
            "general_information": {
                "time_date_utc": self.run_timestamp.isoformat(),
                "datetime_token": datetime_token,
                "test_case_identifier": case_info.get("test_case_identifier", testcase),
                "test_case_name": case_info.get("test_case_name", sim_name),
                "test_case_description": case_info.get(
                    "test_case_description",
                    sim_description,
                ),
                "physics_problem": case_info.get("physics_problem", sim_name),
                "struphy_model_used": case_info.get("struphy_model_used"),
                "simulation_name": sim_name,
                "simulation_description": sim_description,
                "results_root": str(self.run_results_root),
            },
            "hardware_information": _collect_hardware_info(),
            "software_information": _collect_software_info(
                language=case_language,
                commit=case_commit,
                parameters_path=parameters_path,
                case_info=case_info,
            ),
            "scheduler": case_info.get("scheduler", "slurm"),
            # One entry per packaged run, each pointing at that run's own metadata file:
            # the job it ran as, the rank count, and everything packaged with it live
            # there, not here.
            "runs": self.packaged_runs,
        }
        (self.destination_dir / "case_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return self.destination_dir

    def package_run(self, job_info: dict, verbose: bool = False) -> bool:
        """Copy one finished run's output into the packaged folder.

        Everything one run produced goes into a folder of its own,
        `results-run<launch_id>`: its `.h5` files, anything the parameter file
        post-processed into `sim_<launch_id>/results` (`.png` figures, `.npy` arrays),
        and a `run<launch_id>.json` describing all of it. Only packages what that run
        actually produced, so a run whose job never started (or failed before writing
        output) is skipped instead of being uploaded. Appends to `packaged_runs`, which
        the next `package_case_metadata` call writes into `case_metadata.json` as the
        `runs` list — paths everywhere are relative to the packaged case folder.

        The run is located and named by its `launch_id`: `build_commands` names each run
        directory `sim_<launch_id>`, and the packaged folder and `.h5` files follow the
        same id, so two launches sharing a rank count stay separate.

        Args:
            job_info: The `job_infos` entry of the run that just finished.
            verbose: Print what is being packaged.

        Returns:
            Whether any output (`.h5` files or `results` artifacts) was found and packaged.

        Raises:
            RuntimeError: If called before `package_case_metadata` has created the
                packaged folder this copies into.
        """
        if self.destination_dir is None:
            raise RuntimeError(
                "package_run was called before package_case_metadata; the packaged folder does not exist yet.",
            )

        sim_dir = self.case_output_root / f"sim_{job_info['launch_id']:02d}"
        run_dir = self.destination_dir / _run_folder_name(job_info["launch_id"])

        h5_files = sorted(sim_dir.rglob("*.h5")) if sim_dir.is_dir() else []
        if verbose:
            print(f"Found {len(h5_files)} .h5 file(s) in {sim_dir}")

        # The launch id is unique per run, so the files of this run cannot collide with
        # those of any other; `index` only separates several `.h5` files of this one run.
        profiling_data = []
        for index, source_h5 in enumerate(h5_files):
            output_name = _build_output_name(
                launch_id=job_info["launch_id"],
                index=index,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_h5, run_dir / output_name)
            profiling_data.append(f"{run_dir.name}/{output_name}")

        # Whatever the parameter file post-processed into `sim_dir/results` (figures,
        # small arrays) joins the run's `.h5` files in the same folder.
        results_files = _copy_run_results(sim_dir, run_dir)
        if verbose and results_files:
            print(
                f"Packaged {len(results_files)} results file(s) from {sim_dir / RESULTS_DIR_NAME} into {run_dir.name}/",
            )

        if not profiling_data and not results_files:
            return False

        # Everything about this run goes into its own metadata file; the case metadata
        # only points at it.
        self.packaged_runs.append(
            {
                "launch_id": job_info["launch_id"],
                "folder": run_dir.name,
                "run_metadata": _write_run_metadata(
                    sim_dir=sim_dir,
                    run_dir=run_dir,
                    job_info=job_info,
                    profiling_data=profiling_data,
                    results=results_files,
                ),
            },
        )
        return True

    def _iter_finished_runs(self, poll_interval: float = 10.0):
        """Yield each run's `job_infos` entry as soon as that run finishes, in completion order.

        Under SLURM, polls the queue instead of blocking on all jobs at once, so
        `finalize_run` can package and push each run while the others are still
        running. Locally the runs are already done, since `launch` runs them
        sequentially, and they are yielded in launch order.

        Args:
            poll_interval: Seconds between polls of the SLURM queue. Unused for
                local runs.

        Yields:
            The `job_infos` entry of each run that has just finished.
        """
        if self.use_slurm:
            pending = {info["job_id"]: info for info in self.job_infos if info.get("job_id") is not None}
            # Filter by user so each poll reads back only our own jobs rather than the
            # whole cluster queue.
            queue = SQueue(user=getpass.getuser())
            while pending:
                queue.refresh()
                active = {job.job_id for job in queue.jobs() if job.is_active}
                # A job that has left the queue is finished, including one that was
                # already gone by the first poll.
                finished = [job_id for job_id in pending if job_id not in active]
                # Leaving the queue only means the job is over, not that it succeeded.
                # Without the accounting state, a crashed job is indistinguishable from
                # one that produced no output, and the scaling point is lost silently.
                # `job_states` is one `sacct` call for however many finished this poll,
                # and gives None when the state cannot be determined — which must not
                # be reported as a failure.
                states = job_states(finished)
                for job_id in finished:
                    job_info = pending.pop(job_id)
                    state = states.get(job_id)
                    if state is not None and state != "COMPLETED":
                        print(
                            f"WARNING: job {job_id} for '{self.label}' ({job_info['ranks']} MPI ranks) "
                            f"ended in state {state}.",
                        )
                    else:
                        print(f"Job {job_id} for '{self.label}' ({job_info['ranks']} MPI ranks) finished.")
                    yield job_info
                if pending:
                    time.sleep(poll_interval)
        else:
            # Local runs are executed synchronously by `launch`, so by the time we get
            # here every one of them has already finished, in launch order.
            yield from self.job_infos

    def finalize_run(self, poll_interval: float = 10.0) -> None:
        """Package and push the case metadata up front, then each run as its job finishes.

        Called once per case, after every rank count has been submitted/run via
        `launch`. Writes `profiling_case_info.json` (case metadata plus `job_infos`)
        into `case_output_root`, then packages and pushes the case-level metadata
        (`parameters.py`, `case_metadata.json`) straight away, so the packaged folder
        exists in the profiling-data repo before any run has finished. It then waits on
        the runs one by one, packaging each run's `.h5` files and run metadata into that
        same folder and pushing it again as soon as that run's job finishes, rather than
        waiting for every run to complete first.

        Whether the pushes happen at all is the case's `upload` flag, set at construction.

        Args:
            poll_interval: Seconds between polls for finished runs.
        """
        print(
            f"Writing metadata for '{self.label}' to {self.case_output_root / 'profiling_case_info.json'}",
        )
        case_info = self.case_info_dict()
        (self.case_output_root / "profiling_case_info.json").write_text(
            json.dumps(case_info, indent=2),
            encoding="utf-8",
        )

        # Package and push the case-level metadata before waiting on any run, so the
        # packaged folder is already in place when the per-run pushes start updating it.
        self.package_case_metadata(case_info)
        print(f"Packaged case metadata for '{self.label}' into {self.destination_dir}")
        if self.upload:
            print("Uploading case metadata to the profiling-data repo ...")
            _push_profiling_data(self.output_root, self.run_commit)

        # Then package and push each run's results as soon as its own job finishes,
        # rather than waiting for every run to complete first.
        num_runs = len(self.job_infos)
        print(f"Waiting for {num_runs} run(s) of '{self.label}' to complete...")
        packaged_launch_ids = []
        for job_info in self._iter_finished_runs(poll_interval=poll_interval):
            # The launch id identifies the run; the rank count is just context.
            run = f"run {job_info['launch_id']:02d} ({job_info['ranks']} MPI ranks)"
            if not self.package_run(job_info, verbose=True):
                print(f"No profiling output for '{self.label}' {run}; nothing to package.")
                continue
            packaged_launch_ids.append(job_info["launch_id"])
            # Refresh `case_metadata.json` so it lists the runs packaged so far.
            self.package_case_metadata(case_info)
            print(f"Packaged '{self.label}' {run} into {self.destination_dir}")
            if self.upload:
                print(f"Uploading '{self.label}' {run} to the profiling-data repo ...")
                _push_profiling_data(self.output_root, self.run_commit)

        if packaged_launch_ids:
            print(f"Packaged {len(packaged_launch_ids)} run(s) of '{self.label}' (ids {sorted(packaged_launch_ids)}):")
            print(f" - {self.destination_dir}")
            if not self.upload:
                print("Upload skipped; nothing was pushed to the profiling-data repo.")
                print("Push this case later, exactly as packaged, without re-running it:")
                print(f"    python {script_dir / 'upload.py'} {self.destination_dir}")
                first_run_glob = self.destination_dir / _run_folder_name(sorted(packaged_launch_ids)[0]) / "*.h5"
                plot_dir = self.destination_dir / "figures"
                print("Or plot the results locally from the packaged profiling_data.h5 files, e.g.:")
                print(
                    "    scope-profiler plot all "
                    f"{shlex.quote(str(first_run_glob))} --ranks 0 -o {shlex.quote(str(plot_dir))}",
                )
        else:
            print(f"No profiling output found for '{self.label}'; nothing to package.")
