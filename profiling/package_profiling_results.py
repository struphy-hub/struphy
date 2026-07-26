import argparse
import ast
import getpass
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Written by `whereami` during the job, one per profiling case.
MACHINE_PARAMS_FILE = "machine_params.json"
# Written by `Simulation.run()`, one per `sim_ranks<N>` run directory.
RUN_METADATA_FILE = "run_metadata.json"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def _extract_ranks(path: Path) -> str:
    rank_match = re.search(r"(?:^|[-_])ranks?(\d+)(?:$|[-_])", str(path))
    if rank_match:
        return rank_match.group(1)

    for part in path.parts:
        part_match = re.search(r"sim_ranks(\d+)", part)
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

    Each `sim_ranks<N>` run directory holds its own `run_metadata.json`, so it is
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


def _discover_results_root(search_root: Path) -> Path:
    marker_path = search_root / "results" / "profiling" / "latest_run_root.txt"
    if marker_path.exists():
        marker_root = Path(marker_path.read_text(encoding="utf-8").strip())
        if not marker_root.is_absolute():
            marker_root = (marker_path.parent / marker_root).resolve()
        if marker_root.exists():
            print(f"Discovered results root from marker: {marker_root}")
            return marker_root

    candidates: set[Path] = set()

    for h5_path in search_root.rglob("profiling_data.h5"):
        parts = h5_path.parts
        for idx in range(len(parts) - 1):
            if parts[idx] == "profiling" and parts[idx + 1] == "results":
                candidates.add(Path(*parts[: idx + 2]))
                break
            if idx + 2 < len(parts) and parts[idx] == "results" and parts[idx + 1] == "profiling":
                candidates.add(Path(*parts[: idx + 3]))
                break

    if not candidates:
        raise FileNotFoundError(
            f"Results folder does not exist and no profiling_data.h5 files were found under: {search_root}",
        )

    if len(candidates) > 1:
        discovered = "\n".join(f" - {path}" for path in sorted(candidates))
        raise RuntimeError(
            f"Found multiple possible profiling results roots; pass --results-root explicitly:\n{discovered}",
        )

    discovered_root = next(iter(candidates))
    print(f"Discovered results root: {discovered_root}")
    return discovered_root


def _resolve_results_root_arg(results_root: Path) -> Path:
    marker_path = results_root / "latest_run_root.txt"
    if marker_path.exists():
        marker_root = Path(marker_path.read_text(encoding="utf-8").strip())
        if not marker_root.is_absolute():
            marker_root = (results_root / marker_root).resolve()
        if marker_root.exists():
            print(f"Resolved run results root from marker: {marker_root}")
            return marker_root
    return results_root


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        # e.g. `lscpu` or `scontrol` are not available outside a Linux/SLURM machine.
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _run_shell_command(command: str) -> dict[str, Any]:
    result = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _read_case_info(testcase_dir: Path) -> dict[str, Any]:
    case_info_path = testcase_dir / "profiling_case_info.json"
    if not case_info_path.exists():
        return {}
    return json.loads(case_info_path.read_text(encoding="utf-8"))


def _collect_environment_variables() -> dict[str, str]:
    """Environment variables of interest, excluding those stored elsewhere in the metadata.

    ``SLURM_*`` variables live in ``job_information.variables`` and ``LOADEDMODULES``
    is expanded into ``software_information.modules``, so both are skipped here.
    """
    allowed_prefixes = (
        "OMP_",
        "PYTHON",
        "VIRTUAL_ENV",
        "CONDA",
        "MODULE",
        "GITHUB_",
    )
    allowed_names = {"PATH", "LD_LIBRARY_PATH"}
    filtered = {
        key: value for key, value in os.environ.items() if key.startswith(allowed_prefixes) or key in allowed_names
    }
    return dict(sorted(filtered.items()))


def _collect_slurm_environment_variables() -> dict[str, str]:
    slurm_variables = {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}
    return dict(sorted(slurm_variables.items()))


def detect_machine_name() -> str | None:
    """Name of the current HPC machine, following the detection order of `whereami`.

    This is a Python port of the `MACHINE_NAME` branch of
    https://github.com/max-models/whereami; the other parameters (`CPU_VENDOR`,
    `CHIP`, ...) are not duplicated here, they come from the `machine_params.json`
    that `whereami` itself writes during the job.

    Returns None on an unrecognised machine (e.g. a laptop).
    """
    host = os.environ.get("HOST", "")
    hostname = os.environ.get("HOSTNAME") or socket.gethostname()
    lmod_admin_file = os.environ.get("LMOD_ADMIN_FILE", "")
    hpc_system = os.environ.get("HPC_SYSTEM", "")
    nersc_host = os.environ.get("NERSC_HOST", "")
    runner_tags = os.environ.get("CI_RUNNER_TAGS", "")
    partition = os.environ.get("PARTITION", "")

    if "raven" in host:
        return "Raven"
    if "viper12" in hostname:
        return "Viper-GPU"
    if "viper" in hostname:
        return "Viper-CPU"
    if "cobra" in host:
        return "Cobra"
    if "lumi" in lmod_admin_file:
        lumi_partition = partition or "LUMI-G"
        if lumi_partition in ("LUMI-G", "LUMI-C", "LUMI-D"):
            return lumi_partition
        print(f"Unsupported LUMI partition: {lumi_partition}")
        return None
    if "leonardo" in hpc_system:
        return "Leonardo (Booster)" if (partition or "Booster") == "Booster" else "Leonardo (DCGP)"
    if "marconi" in hpc_system:
        return "Marconi"
    if "pitagora" in hpc_system:
        return "Pitagora (DCGP)"
    if "toki" in host:
        return "TOK"
    if "vega" in hostname:
        return "Vega (GPU)" if (partition or "GPU") == "GPU" else "Vega (CPU)"
    if "perlmutter" in nersc_host:
        return "Perlmutter"
    if "runner" in hostname:
        if "nvidia-cc80" in runner_tags:
            return "Shared GPU Runner (NVIDIA)"
        if "amd-mi200" in runner_tags:
            return "Shared GPU Runner (AMD)"
        return "Shared Runner"
    return None


def _copy_machine_params(testcase_dir: Path, destination_dir: Path) -> str | None:
    """Copy the `whereami` JSON export produced by the job next to the packaged data.

    The file is stored verbatim, not parsed. If the job did not produce one (older
    run, or `whereami` install failed), it is regenerated here as a best effort.
    Returns the packaged file name, or None if no parameters could be obtained.
    """
    source = testcase_dir / MACHINE_PARAMS_FILE
    destination = destination_dir / MACHINE_PARAMS_FILE

    if not source.exists():
        executable = shutil.which("whereami")
        if executable is None:
            print(f"No {MACHINE_PARAMS_FILE} in {testcase_dir} and `whereami` is not on PATH; skipping.")
            return None
        result = _run_command([executable, "--output", str(destination)])
        if result["returncode"] != 0 or not destination.exists():
            print(f"`whereami --output` failed: {result['stderr'] or result['stdout']}")
            return None
        return MACHINE_PARAMS_FILE

    shutil.copy2(source, destination)
    return MACHINE_PARAMS_FILE


def _collect_hardware_info() -> dict[str, Any]:
    """Description of the machine the profiling job ran on.

    CPU/GPU details are not repeated here: they live in the `whereami` export
    (`machine_params.json`) that is packaged alongside this metadata.
    """
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
        "platform": platform.platform(),
        "hostname": _run_command(["hostname"])["stdout"],
        "uname": _run_command(["uname", "-a"])["stdout"],
        "chip_information": _run_command(["lscpu"])["stdout"],
        "node_hostnames": resolved_nodes,
    }


def _collect_software_info(
    *,
    language: str,
    commit: str,
    parameters_path: Path | None,
    case_info: dict[str, Any],
) -> dict[str, Any]:
    module_list_cmd = _run_shell_command("module list -t 2>&1")
    loaded_modules = [
        line
        for line in (module_list_cmd["stdout"].splitlines() if module_list_cmd["returncode"] == 0 else [])
        if line
        and not line.startswith("Currently Loaded Modulefiles:")
        and not line.startswith("No Modulefiles Currently Loaded.")
    ]
    if not loaded_modules and os.environ.get("LOADEDMODULES"):
        loaded_modules = [entry for entry in os.environ["LOADEDMODULES"].split(":") if entry]

    compiler_info = case_info.get("compiler") or {}
    compiler_options = {key: value for key, value in compiler_info.items() if key not in ("language", "compiler")}

    return {
        "struphy_commit": commit,
        "pyccel_language": case_info.get("pyccel_language") or compiler_info.get("language") or language,
        "pyccel_compiler_family": case_info.get("pyccel_compiler_family") or compiler_info.get("compiler"),
        "compiler_options": compiler_options,
        "parameter_file": (str(parameters_path) if parameters_path is not None else case_info.get("parameter_file")),
        "parameter_file_source": case_info.get("parameter_file"),
        "modules": loaded_modules,
        "environment_variables": _collect_environment_variables(),
        "python_environment_pip_freeze": _run_command(
            ["python", "-m", "pip", "freeze"],
        )["stdout"],
    }


def _collect_job_info(case_info: dict[str, Any]) -> dict[str, Any]:
    """Job description; the script is stored once, as a single string.

    Covers both schedulers: a SLURM batch script with `pragmas`, or the plain bash
    script of a local run (`scheduler: "local"`, no pragmas and no SLURM variables).
    ``slurm_dict["custom_commands"]`` is dropped because those commands are already
    part of ``script``.
    """
    slurm_dict = case_info.get("slurm_dict") or {}
    return {
        "scheduler": case_info.get("scheduler", "slurm"),
        "script_path": case_info.get("job_script_path"),
        "script": case_info.get("job_script"),
        "pragmas": slurm_dict.get("pragmas"),
        "variables": case_info.get("slurm_variables", _collect_slurm_environment_variables()),
    }


def package_testcase(
    testcase_dir: Path,
    results_root: Path,
    language: str | None,
    commit: str | None,
    output_root: Path,
    timestamp: datetime | None = None,
    verbose: bool = False,
) -> Path | None:
    """Package a single testcase directory (e.g. one `ProfilingCase.output_root`) into `output_root`.

    Only packages the testcase if it actually produced `.h5` output, so a case whose
    SLURM job never ran (or failed before writing output) is silently skipped instead
    of being uploaded. Returns the created destination folder, or None if skipped.
    """
    if verbose:
        print(f"Packaging testcase directory: {testcase_dir}")
    h5_files = sorted(testcase_dir.rglob("*.h5"))
    if verbose:
        print(f"Found {len(h5_files)} .h5 file(s) in {testcase_dir}")
    if not h5_files:
        return None

    if timestamp is None:
        timestamp = datetime.now(UTC)
    datetime_token = timestamp.strftime("%Y%m%dT%H%M%SZ")

    testcase = testcase_dir.name
    parameters_path = _ensure_testcase_parameters_file(testcase_dir)
    case_info = _read_case_info(testcase_dir)
    case_language = case_info.get("pyccel_language") or language
    case_commit = case_info.get("struphy_commit") or commit
    if case_language is None:
        raise RuntimeError(
            f"Missing pyccel language for testcase '{testcase}'. "
            "Provide it in profiling_case_info.json or via --language.",
        )
    if case_commit is None:
        raise RuntimeError(
            f"Missing commit hash for testcase '{testcase}'. Provide it in profiling_case_info.json or via --commit.",
        )

    commit_short = case_commit[:8]
    folder_name = f"{datetime_token}-{commit_short}-{_slug(testcase)}-{_slug(case_language)}"
    destination_dir = output_root / folder_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    sim_name = testcase
    sim_description = ""
    if parameters_path is not None:
        sim_name, sim_description = _read_sim_metadata_from_parameters(
            parameters_path=parameters_path,
            fallback_name=testcase,
        )
        shutil.copy2(parameters_path, destination_dir / "parameters.py")

    files_metadata = []
    name_counts: dict[str, int] = {}
    for source_h5 in h5_files:
        relative_source = source_h5.relative_to(testcase_dir)
        ranks = _extract_ranks(relative_source)
        base_key = f"{_slug(testcase)}-ranks{ranks}-{_slug(case_language)}"
        output_name = _build_output_name(
            testcase=testcase,
            language=case_language,
            ranks=ranks,
            index=name_counts.get(base_key, 0),
        )
        name_counts[base_key] = name_counts.get(base_key, 0) + 1
        destination_h5 = destination_dir / output_name
        shutil.copy2(source_h5, destination_h5)
        run_metadata_name, run_metadata_source = _copy_run_metadata(
            source_h5=source_h5,
            destination_h5=destination_h5,
        )
        files_metadata.append(
            {
                "source": str(source_h5),
                "relative_source": str(relative_source),
                "ranks": ranks,
                "destination": output_name,
                "run_metadata_source": run_metadata_source,
                "run_metadata_destination": run_metadata_name,
            },
        )

    hardware_information = _collect_hardware_info()
    hardware_information["machine_params_file"] = _copy_machine_params(
        testcase_dir=testcase_dir,
        destination_dir=destination_dir,
    )

    metadata = {
        "general_information": {
            "time_date_utc": timestamp.isoformat(),
            "datetime_token": datetime_token,
            "user": getpass.getuser(),
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
            "results_root": str(results_root),
        },
        "hardware_information": hardware_information,
        "software_information": _collect_software_info(
            language=case_language,
            commit=case_commit,
            parameters_path=parameters_path,
            case_info=case_info,
        ),
        "job_information": _collect_job_info(case_info),
        "files": files_metadata,
    }
    (destination_dir / "case_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return destination_dir


def package_results(
    results_root: Path,
    language: str | None,
    commit: str | None,
    output_root: Path,
) -> list[Path]:
    results_root = _resolve_results_root_arg(results_root)
    if not results_root.exists():
        results_root = _discover_results_root(search_root=Path.cwd().resolve())

    timestamp = datetime.now(UTC)
    created_dirs: list[Path] = []

    testcase_dirs = [path for path in sorted(results_root.iterdir()) if path.is_dir()]
    for testcase_dir in testcase_dirs:
        destination_dir = package_testcase(
            testcase_dir=testcase_dir,
            results_root=results_root,
            language=language,
            commit=commit,
            output_root=output_root,
            timestamp=timestamp,
        )
        if destination_dir is not None:
            created_dirs.append(destination_dir)

    if not created_dirs:
        raise RuntimeError(f"No .h5 profiling files found under: {results_root}")

    return created_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package profiling .h5 outputs into DATETIME-COMMIT-TESTCASE-LANGUAGE folders.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/profiling"),
        help=(
            "Folder containing testcase result directories for one profiling run "
            "(default: results/profiling; marker/discovery may resolve to latest run)."
        ),
    )
    parser.add_argument(
        "--language",
        required=False,
        help="Optional compile language fallback if not present in profiling_case_info.json.",
    )
    parser.add_argument(
        "--commit",
        required=False,
        help="Optional commit SHA fallback if not present in profiling_case_info.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("profiling-results-export"),
        help="Folder where packaged result folders are created.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    created_dirs = package_results(
        results_root=args.results_root.resolve(),
        language=args.language,
        commit=args.commit,
        output_root=output_root,
    )

    print("Packaged result folders:")
    for path in created_dirs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
