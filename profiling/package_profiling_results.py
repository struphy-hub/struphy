import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any

from clusters import detect_machine_name

from utils import _run_command

# Written by `Simulation.run()`, one per `sim_<id>` run directory.
RUN_METADATA_FILE = "run_metadata.json"

# Written by the parameter file itself (see the Poisson example), one per `sim_<id>` run
# directory: post-processing figures and small arrays, next to the profiling output.
RESULTS_DIR_NAME = "results"
RESULTS_FILE_SUFFIXES = (".png", ".npy")


def _run_name(launch_id: int) -> str:
    """How a run is named everywhere it is packaged: `run03`.

    Runs are identified by their launch id alone. The rank count names nothing — it is
    recorded in the run's metadata file — so two launches sharing a rank count stay
    apart.
    """
    return f"run{launch_id:02d}"


def _run_folder_name(launch_id: int) -> str:
    """Name of a run's own folder inside the packaged case folder: `results-run03`.

    Everything a run produced lives in there together: its `.h5` files, its metadata
    file, and whatever the parameter file post-processed into `results`.
    """
    return f"{RESULTS_DIR_NAME}-{_run_name(launch_id)}"


def _build_output_name(launch_id: int, index: int) -> str:
    """Name of a packaged `.h5`: the run's name (`run03.h5`).

    The test case is already the packaged folder's name, so repeating it in every file
    inside only makes the names longer. `index` disambiguates a run that produced more
    than one `.h5` file; the first keeps the plain name.
    """
    base = _run_name(launch_id)
    if index > 0:
        base = f"{base}-{index}"
    return f"{base}.h5"


def _write_run_metadata(
    sim_dir: Path,
    run_dir: Path,
    job_info: dict[str, Any],
    profiling_data: list[str],
    results: list[str],
) -> str:
    """Write the run's metadata file, the one place run-specific data lives.

    Starts from the `run_metadata.json` Struphy wrote in `sim_dir` (empty if the run
    never got that far), and adds what only packaging knows: the script the run was
    submitted as, and where its files ended up. The case metadata just references this
    file, so nothing about a single run is spelled out twice.

    `slurm_script` is `SlurmScript.to_dict()` as it stands — `pragmas`, `modules` and
    `custom_commands`, everything the submitted script was built from — and nothing
    else. It is None for a run that was not submitted to SLURM but run locally.
    `slurm_script_str` is the script as it was written to disk and run, `str(script)`
    under SLURM and the plain bash script of a local run.

    Packaged paths are relative to the case folder (`run_dir.parent`), the same base the
    case metadata uses.

    Returns the path of the written file, relative to the case folder.
    """
    source = sim_dir / RUN_METADATA_FILE
    metadata: dict[str, Any] = {}
    if source.exists():
        metadata = json.loads(source.read_text(encoding="utf-8"))
    else:
        print(f"No {RUN_METADATA_FILE} in {sim_dir}; recording only what packaging knows about the run.")

    # Struphy records the rank count it actually ran with; fall back to the requested one
    # for a run that wrote no metadata of its own.
    metadata.setdefault("mpi_ranks", job_info.get("ranks"))
    metadata["slurm_script"] = job_info.get("slurm_dict")
    metadata["slurm_script_str"] = job_info.get("job_script")
    metadata["packaged_files"] = {
        "profiling_data": profiling_data[0] if profiling_data else None,
        "additional_profiling_data": profiling_data[1:],
        "results": results,
        "run_directory": str(sim_dir),
    }

    output_name = f"{_run_name(job_info['launch_id'])}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / output_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return f"{run_dir.name}/{output_name}"


def _copy_run_results(sim_dir: Path, destination_dir: Path) -> list[str]:
    """Copy the figures and arrays a run wrote into `<sim_dir>/results`.

    The parameter file of a case may post-process its own output (see the Poisson
    example), writing `.png`/`.npy` files into a `results` folder inside its run
    directory. They are copied straight into `destination_dir`, the run's own packaged
    folder, next to its `.h5` and run metadata; any subfolder structure of `results` is
    kept.

    Returns the packaged files as paths relative to the case folder
    (`destination_dir.parent`), so a consumer can open them without reconstructing any
    names. Empty if the run wrote no such files.
    """
    source_dir = sim_dir / RESULTS_DIR_NAME
    if not source_dir.is_dir():
        return []

    sources = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in RESULTS_FILE_SUFFIXES
    )

    relative_paths = []
    for source in sources:
        target = destination_dir / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        relative_paths.append(str(target.relative_to(destination_dir.parent)))

    return relative_paths


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
