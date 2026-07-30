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


def _extract_ranks(source_h5: Path) -> str:
    """Rank count of the run that produced `source_h5`.

    Read from the `run_metadata.json` Struphy writes next to it, since the run
    directory is named after the run id alone and no longer carries the rank count.
    Falls back to a rank count spelled out in the path, for older result trees.
    """
    metadata_path = source_h5.parent / RUN_METADATA_FILE
    if metadata_path.exists():
        mpi_ranks = json.loads(metadata_path.read_text(encoding="utf-8")).get("mpi_ranks")
        if isinstance(mpi_ranks, int):
            return str(mpi_ranks)

    for part in source_h5.parts:
        part_match = re.search(r"ranks(\d+)", part)
        if part_match:
            return part_match.group(1)
    return "unknown"


def _build_output_name(testcase: str, language: str, ranks: str, index: int) -> str:
    ranks_token = f"{int(ranks):04d}" if ranks.isdigit() else _slug(ranks)
    base = f"{_slug(testcase)}-ranks{ranks_token}-{_slug(language)}"
    if index > 0:
        base = f"{base}-{index}"
    return f"{base}.h5"


def _copy_run_metadata(source_h5: Path, destination_h5: Path) -> tuple[str | None, str | None]:
    """Copy the `run_metadata.json` that Struphy wrote next to `source_h5`.

    Each `sim_<id>` run directory holds its own `run_metadata.json`, so it is
    packaged per run, named after the corresponding `.h5` file. Returns
    ``(packaged file name, source path)``, both None if the run produced no metadata.
    """
    source = source_h5.parent / RUN_METADATA_FILE
    if not source.exists():
        print(f"No {RUN_METADATA_FILE} next to {source_h5}; skipping.")
        return None, None

    output_name = f"{destination_h5.stem}-{RUN_METADATA_FILE}"
    shutil.copy2(source, destination_h5.parent / output_name)
    return output_name, str(source)


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
    """Job description: one entry per rank count, each with its own script.

    Covers both schedulers: a SLURM batch script with `pragmas`, or the plain bash
    script of a local run (`scheduler: "local"`, no pragmas). Each rank count is
    submitted (or run locally) as its own job/script, since the caller's loop over
    rank counts builds and submits one script per rank count instead of looping over
    rank counts inside a single script.
    ``slurm_dict["custom_commands"]`` is dropped because those commands are already
    part of ``script``, and the `SLURM_*` variables because scope-profiler stores them
    in every `profiling_data.h5`.
    """
    return {
        "scheduler": case_info.get("scheduler", "slurm"),
        "jobs": [
            {
                "ranks": job.get("ranks"),
                "script_path": job.get("job_script_path"),
                "script": job.get("job_script"),
                "pragmas": (job.get("slurm_dict") or {}).get("pragmas"),
            }
            for job in case_info.get("jobs", [])
        ],
    }
