import ast
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from clusters import detect_machine_name

from utils import _run_command, _slug

# Written by `Simulation.run()`, one per `sim_<id>` run directory.
RUN_METADATA_FILE = "run_metadata.json"

# Written by the parameter file itself (see the Poisson example), one per `sim_<id>` run
# directory: post-processing figures and small arrays, next to the profiling output.
RESULTS_DIR_NAME = "results"
RESULTS_FILE_SUFFIXES = (".png", ".npy")


def _read_mpi_ranks(source_h5: Path) -> int | None:
    """Rank count of the run that produced `source_h5`, or None if it cannot be read.

    Read from the `run_metadata.json` Struphy writes next to it. Nothing in the run's
    naming carries the rank count — runs are identified by their launch id — so this
    is the only place it comes from. Recorded as metadata only; it never names a file.
    Returned as an `int` so it matches the `ranks` recorded per job in
    `_collect_job_info`; the caller falls back to the requested rank count on None.
    """
    metadata_path = source_h5.parent / RUN_METADATA_FILE
    if metadata_path.exists():
        mpi_ranks = json.loads(metadata_path.read_text(encoding="utf-8")).get("mpi_ranks")
        if isinstance(mpi_ranks, int):
            return mpi_ranks
    return None


def _build_output_name(launch_id: int, index: int) -> str:
    """Name of a packaged `.h5`, identifying its run by launch id.

    Named after the run alone (`run02.h5`), matching the `results-run02` folder of the
    same run: the test case is already the packaged folder's name, so repeating it in
    every file inside only makes the names longer.

    `index` disambiguates a run that produced more than one `.h5` file; the first keeps
    the plain name.
    """
    base = f"run{launch_id:02d}"
    if index > 0:
        base = f"{base}-{index}"
    return f"{base}.h5"


def _copy_run_metadata(source_h5: Path, destination_h5: Path) -> tuple[str | None, str | None]:
    """Copy the `run_metadata.json` that Struphy wrote next to `source_h5`.

    Each `sim_<id>` run directory holds its own `run_metadata.json`, so it is packaged
    per run, named after the corresponding `.h5` file (`run02.h5` -> `run02.json`).
    Returns ``(packaged file name, source path)``, both None if the run produced no
    metadata.
    """
    source = source_h5.parent / RUN_METADATA_FILE
    if not source.exists():
        print(f"No {RUN_METADATA_FILE} next to {source_h5}; skipping.")
        return None, None

    output_name = f"{destination_h5.stem}.json"
    shutil.copy2(source, destination_h5.parent / output_name)
    return output_name, str(source)


def _run_token(sim_dir_name: str) -> str:
    """`sim_03` -> `run03`; any other run directory name is slugged as it is."""
    match = re.fullmatch(r"sim_?(\d+)", sim_dir_name)
    if match:
        return f"run{match.group(1)}"
    return _slug(sim_dir_name)


def _copy_run_results(sim_dir: Path, destination_dir: Path) -> dict[str, Any] | None:
    """Package the figures and arrays a run wrote into `<sim_dir>/results`.

    The parameter file of a case may post-process its own output (see the Poisson
    example), writing `.png`/`.npy` files into a `results` folder inside its run
    directory. Every run of a case writes its own, so they are packaged into a per-run
    subfolder (`results-run03`) instead of being merged into the flat case folder.

    Any subfolder structure of `results` is kept, and `files` lists every packaged file
    by its path relative to the packaged folder, so a consumer can open them without
    reconstructing any names.

    Returns the packaged entry for `case_metadata.json`, or None if the run wrote no
    such files.
    """
    source_dir = sim_dir / RESULTS_DIR_NAME
    if not source_dir.is_dir():
        return None

    sources = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in RESULTS_FILE_SUFFIXES
    )
    if not sources:
        return None

    output_name = f"{RESULTS_DIR_NAME}-{_run_token(sim_dir.name)}"
    destination = destination_dir / output_name
    relative_paths = []
    for source in sources:
        relative_path = source.relative_to(source_dir)
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        relative_paths.append(str(target.relative_to(destination_dir)))

    return {
        "source": str(source_dir),
        "destination": output_name,
        "files": relative_paths,
    }


def _extract_string_node(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _is_simulation_constructor(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Name):
        return call.func.id == "Simulation"
    if isinstance(call.func, ast.Attribute):
        return call.func.attr == "Simulation"
    return False


def _read_sim_metadata_from_parameters(
    parameters_path: Path,
    fallback_name: str,
) -> tuple[str, str]:
    tree = ast.parse(parameters_path.read_text(encoding="utf-8"))
    string_constants: dict[str, str] = {}
    sim_name: str | None = None
    sim_description: str | None = None

    for node in tree.body:
        assign_target_name: str | None = None
        assign_value: ast.AST | None = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assign_target_name = node.targets[0].id
            assign_value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assign_target_name = node.target.id
            assign_value = node.value

        if assign_target_name is None or assign_value is None:
            continue

        value = _extract_string_node(assign_value, string_constants)
        if value is not None:
            string_constants[assign_target_name] = value

        if (
            assign_target_name == "sim"
            and isinstance(assign_value, ast.Call)
            and _is_simulation_constructor(assign_value)
        ):
            for keyword in assign_value.keywords:
                if keyword.arg == "name":
                    sim_name = _extract_string_node(keyword.value, string_constants)
                elif keyword.arg == "description":
                    sim_description = _extract_string_node(
                        keyword.value,
                        string_constants,
                    )

    if sim_name is None:
        sim_name = string_constants.get("name", fallback_name)
    if sim_description is None:
        sim_description = string_constants.get("description", "")

    return sim_name, sim_description


def _ensure_testcase_parameters_file(testcase_dir: Path) -> Path | None:
    testcase_parameters = testcase_dir / "parameters.py"
    if testcase_parameters.exists():
        return testcase_parameters

    candidate_parameters = sorted(testcase_dir.rglob("parameters.py"))
    if not candidate_parameters:
        return None

    chosen_parameters = candidate_parameters[0]
    chosen_content = chosen_parameters.read_text(encoding="utf-8")
    for candidate in candidate_parameters[1:]:
        if candidate.read_text(encoding="utf-8") != chosen_content:
            raise RuntimeError(
                f"Found multiple different parameters.py files under testcase directory: {testcase_dir}",
            )

    shutil.copy2(chosen_parameters, testcase_parameters)
    return testcase_parameters


def _collect_hardware_info() -> dict[str, Any]:
    """Description of the machine the profiling job ran on."""
    cluster_name = (
        detect_machine_name()
        or os.environ.get("SLURM_CLUSTER_NAME")
        or _run_command(["hostname", "-f"])["stdout"]
        or None
    )

    slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST")
    resolved_nodes: list[str] = []
    if slurm_nodelist:
        hostnames_cmd = _run_command(["scontrol", "show", "hostnames", slurm_nodelist])
        if hostnames_cmd["returncode"] == 0 and hostnames_cmd["stdout"]:
            resolved_nodes = hostnames_cmd["stdout"].splitlines()
    if not resolved_nodes:
        resolved_nodes = [_run_command(["hostname"])["stdout"]]

    return {
        "cluster_name": cluster_name,
        "node_hostnames": resolved_nodes,
    }


def _collect_software_info(
    *,
    language: str,
    commit: str,
    parameters_path: Path | None,
    case_info: dict[str, Any],
) -> dict[str, Any]:
    """Struphy-specific software description.

    No environment variables are collected here. The interpreter, the module stack, the
    toolchain and the batch job are described by scope-profiler in every
    `profiling_data.h5`; anything important that it misses belongs there, not in a
    second, struphy-only copy.
    """
    compiler_info = case_info.get("compiler") or {}
    compiler_options = {key: value for key, value in compiler_info.items() if key not in ("language", "compiler")}

    return {
        "struphy_commit": commit,
        "pyccel_language": case_info.get("pyccel_language") or compiler_info.get("language") or language,
        "pyccel_compiler_family": case_info.get("pyccel_compiler_family") or compiler_info.get("compiler"),
        "compiler_options": compiler_options,
        "parameter_file": (str(parameters_path) if parameters_path is not None else case_info.get("parameter_file")),
        "parameter_file_source": case_info.get("parameter_file"),
        "python_environment_pip_freeze": _run_command(
            ["python", "-m", "pip", "freeze"],
        )["stdout"],
    }


def _collect_job_info(case_info: dict[str, Any]) -> dict[str, Any]:
    """Job description: one entry per launch, each with its own script.

    Covers both schedulers: a SLURM batch script with `pragmas`, or the plain bash
    script of a local run (`scheduler: "local"`, no pragmas). Each launch is submitted
    (or run locally) as its own job/script, identified by its launch id, instead of
    looping over rank counts inside a single script.
    ``slurm_dict["custom_commands"]`` is dropped because those commands are already
    part of ``script``, and the `SLURM_*` variables because scope-profiler stores them
    in every `profiling_data.h5`.
    """
    return {
        "scheduler": case_info.get("scheduler", "slurm"),
        "jobs": [
            {
                "launch_id": job.get("launch_id"),
                "ranks": job.get("ranks"),
                "script_path": job.get("job_script_path"),
                "script": job.get("job_script"),
                "pragmas": (job.get("slurm_dict") or {}).get("pragmas"),
            }
            for job in case_info.get("jobs", [])
        ],
    }
