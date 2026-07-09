import argparse
import ast
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path


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
    base = f"{_slug(testcase)}-ranks{ranks}-{_slug(language)}"
    if index > 0:
        base = f"{base}-{index}"
    return f"{base}.h5"


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
    parameters_path: Path, fallback_name: str
) -> tuple[str, str]:
    tree = ast.parse(parameters_path.read_text(encoding="utf-8"))
    string_constants: dict[str, str] = {}
    sim_name: str | None = None
    sim_description: str | None = None

    for node in tree.body:
        assign_target_name: str | None = None
        assign_value: ast.AST | None = None

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
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
                        keyword.value, string_constants
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
                "Found multiple different parameters.py files under testcase "
                f"directory: {testcase_dir}"
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
            if (
                idx + 2 < len(parts)
                and parts[idx] == "results"
                and parts[idx + 1] == "profiling"
            ):
                candidates.add(Path(*parts[: idx + 3]))
                break

    if not candidates:
        raise FileNotFoundError(
            f"Results folder does not exist and no profiling_data.h5 files were found under: {search_root}"
        )

    if len(candidates) > 1:
        discovered = "\n".join(f" - {path}" for path in sorted(candidates))
        raise RuntimeError(
            "Found multiple possible profiling results roots; pass --results-root explicitly:\n"
            f"{discovered}"
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


def package_results(
    results_root: Path,
    language: str,
    commit: str,
    output_root: Path,
) -> list[Path]:
    results_root = _resolve_results_root_arg(results_root)
    if not results_root.exists():
        results_root = _discover_results_root(search_root=Path.cwd().resolve())

    timestamp = datetime.now(UTC)
    datetime_token = timestamp.strftime("%Y%m%dT%H%M%SZ")
    commit_short = commit[:8]
    created_dirs: list[Path] = []

    testcase_dirs = [path for path in sorted(results_root.iterdir()) if path.is_dir()]
    for testcase_dir in testcase_dirs:
        h5_files = sorted(testcase_dir.rglob("*.h5"))
        if not h5_files:
            continue

        testcase = testcase_dir.name
        folder_name = f"{datetime_token}-{commit_short}-{_slug(testcase)}-{_slug(language)}"
        destination_dir = output_root / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        parameters_path = _ensure_testcase_parameters_file(testcase_dir)
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
            base_key = f"{_slug(testcase)}-ranks{ranks}-{_slug(language)}"
            output_name = _build_output_name(
                testcase=testcase,
                language=language,
                ranks=ranks,
                index=name_counts.get(base_key, 0),
            )
            name_counts[base_key] = name_counts.get(base_key, 0) + 1
            destination_h5 = destination_dir / output_name
            shutil.copy2(source_h5, destination_h5)
            files_metadata.append(
                {
                    "source": str(source_h5),
                    "relative_source": str(relative_source),
                    "ranks": ranks,
                    "destination": output_name,
                }
            )

        metadata = {
            "name": sim_name,
            "description": sim_description,
            "datetime_utc": timestamp.isoformat(),
            "datetime_token": datetime_token,
            "commit": commit,
            "commit_short": commit_short,
            "testcase": testcase,
            "language": language,
            "source_results_root": str(results_root),
            "source_parameters_file": (
                str(parameters_path) if parameters_path is not None else None
            ),
            "files": files_metadata,
            "github": {
                "repository": os.environ.get("GITHUB_REPOSITORY"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "workflow": os.environ.get("GITHUB_WORKFLOW"),
                "job": os.environ.get("GITHUB_JOB"),
            },
        }
        (destination_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        created_dirs.append(destination_dir)

    if not created_dirs:
        raise RuntimeError(f"No .h5 profiling files found under: {results_root}")

    return created_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package profiling .h5 outputs into DATETIME-COMMIT-TESTCASE-LANGUAGE folders."
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
        required=True,
        help="Compile language label to include in output naming.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Commit SHA used in destination folder naming.",
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
