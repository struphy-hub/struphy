import argparse
import importlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path


EXCLUDED_DEPENDENCIES = {"psydac"}
UPPER_BOUND_OPERATORS = {"<", "<="}
SPECIFIER_PATTERN = re.compile(r"(<=|>=|==|!=|~=|<|>)\s*([^,;]+)")


def normalize_name(package_name):
    return re.sub(r"[-_.]+", "-", package_name).lower()


def split_requirement(entry):
    marker = None
    requirement_part = entry.strip()
    if ";" in requirement_part:
        requirement_part, marker = requirement_part.split(";", 1)
        marker = marker.strip()

    requirement_part = requirement_part.strip()
    match = re.search(r"(<=|>=|==|!=|~=|<|>)", requirement_part)
    if match:
        name_with_extras = requirement_part[: match.start()].strip()
        specifier_part = requirement_part[match.start() :].strip()
    else:
        name_with_extras = requirement_part
        specifier_part = ""

    if "[" in name_with_extras and name_with_extras.endswith("]"):
        name, extras_part = name_with_extras[:-1].split("[", 1)
        extras = [extra.strip() for extra in extras_part.split(",") if extra.strip()]
    else:
        name = name_with_extras
        extras = []

    specifiers = []
    for operator, version in SPECIFIER_PATTERN.findall(specifier_part):
        specifiers.append(f"{operator}{version.strip()}")

    return {
        "name": name.strip(),
        "extras": sorted(extras),
        "marker": marker,
        "specifiers": specifiers,
    }


def format_requirement_name(requirement):
    extras = ""
    if requirement["extras"]:
        extras = "[" + ",".join(requirement["extras"]) + "]"
    return f"{requirement['name']}{extras}"


def get_preserved_specifiers(requirement):
    preserved = []
    for specifier in requirement["specifiers"]:
        for operator in UPPER_BOUND_OPERATORS:
            if specifier.startswith(operator):
                break
        else:
            preserved.append(specifier)
    return sorted(preserved)


def build_dependency_entry(entry, resolved_versions, project_name):
    requirement = split_requirement(entry)
    normalized_name = normalize_name(requirement["name"])

    if normalized_name in EXCLUDED_DEPENDENCIES:
        return None

    if normalized_name == normalize_name(project_name):
        return entry

    resolved_version = resolved_versions.get(normalized_name)
    if resolved_version is None:
        print(f"Warning: {requirement['name']} is not available in the tested dependency snapshot, skipping...", file=sys.stderr)
        return entry

    specifiers = get_preserved_specifiers(requirement)
    specifiers.append(f"<={resolved_version}")
    updated_entry = format_requirement_name(requirement)
    if specifiers:
        updated_entry += specifiers[0]
        if len(specifiers) > 1:
            updated_entry += ", " + ", ".join(specifiers[1:])
    if requirement["marker"]:
        updated_entry += f"; {requirement['marker']}"
    return updated_entry


def update_dependency_group(dependencies, resolved_versions, project_name):
    updated_dependencies = []
    for entry in dependencies:
        updated_entry = build_dependency_entry(entry, resolved_versions, project_name)
        if updated_entry is not None:
            updated_dependencies.append(updated_entry)
    return updated_dependencies


def get_selected_optional_groups(pyproject_data, optional_groups):
    declared_optional_groups = pyproject_data["project"].get("optional-dependencies", {})
    if optional_groups is None:
        return None

    missing_groups = [group_name for group_name in optional_groups if group_name not in declared_optional_groups]
    if missing_groups:
        missing_display = ", ".join(sorted(missing_groups))
        raise ValueError(f"Unknown optional dependency group(s): {missing_display}")

    return set(optional_groups)


def iter_dependency_entries(pyproject_data, optional_groups=None):
    for entry in pyproject_data["project"]["dependencies"]:
        yield entry

    selected_optional_groups = get_selected_optional_groups(pyproject_data, optional_groups)
    optional_dependencies = pyproject_data["project"].get("optional-dependencies", {})
    for group_name, group_dependencies in optional_dependencies.items():
        if selected_optional_groups is not None and group_name not in selected_optional_groups:
            continue
        for entry in group_dependencies:
            yield entry


def collect_installed_versions(pyproject_data, optional_groups=None):
    project_name = pyproject_data["project"]["name"]
    versions = {}
    for entry in iter_dependency_entries(pyproject_data, optional_groups=optional_groups):
        requirement = split_requirement(entry)
        normalized_name = normalize_name(requirement["name"])

        if normalized_name in EXCLUDED_DEPENDENCIES or normalized_name == normalize_name(project_name):
            continue

        if normalized_name in versions:
            continue

        try:
            versions[normalized_name] = importlib.metadata.version(requirement["name"])
        except importlib.metadata.PackageNotFoundError:
            print(f"Warning: {requirement['name']} not installed, skipping...", file=sys.stderr)
    return versions


def load_pyproject(pyproject_path):
    try:
        toml_reader = importlib.import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        toml_reader = importlib.import_module("tomli")

    with pyproject_path.open("rb") as handle:
        return toml_reader.load(handle)


def load_versions(versions_path):
    with versions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "versions" in payload:
        payload = payload["versions"]

    return {normalize_name(name): version for name, version in payload.items()}


def write_versions_snapshot(pyproject_data, output_path, optional_groups=None):
    payload = {
        "project": pyproject_data["project"]["name"],
        "versions": collect_installed_versions(pyproject_data, optional_groups=optional_groups),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_pyproject(pyproject_data, resolved_versions, optional_groups=None):
    project_name = pyproject_data["project"]["name"]
    pyproject_data["project"]["dependencies"] = update_dependency_group(
        pyproject_data["project"]["dependencies"],
        resolved_versions,
        project_name,
    )

    selected_optional_groups = get_selected_optional_groups(pyproject_data, optional_groups)
    for group_name, group_dependencies in pyproject_data["project"].get("optional-dependencies", {}).items():
        if selected_optional_groups is not None and group_name not in selected_optional_groups:
            continue
        pyproject_data["project"]["optional-dependencies"][group_name] = update_dependency_group(
            group_dependencies,
            resolved_versions,
            project_name,
        )


def dump_pyproject(pyproject_data, pyproject_path):
    toml_writer = importlib.import_module("tomli_w")

    with pyproject_path.open("wb") as handle:
        toml_writer.dump(pyproject_data, handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Update Struphy release dependency bounds.")
    parser.add_argument("--pyproject-file", default="pyproject.toml", help="Path to pyproject.toml.")
    parser.add_argument(
        "--optional-group",
        action="append",
        dest="optional_groups",
        help="Optional dependency group to include. If omitted, all optional groups are processed.",
    )
    parser.add_argument(
        "--versions-file",
        help="Path to a JSON file containing the tested dependency versions.",
    )
    parser.add_argument(
        "--write-versions-file",
        help="Write the currently installed dependency versions for declared dependencies to this JSON file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if pyproject.toml would change.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pyproject_path = Path(args.pyproject_file)
    pyproject_data = load_pyproject(pyproject_path)

    try:
        if args.write_versions_file:
            write_versions_snapshot(
                pyproject_data,
                Path(args.write_versions_file),
                optional_groups=args.optional_groups,
            )
            return 0

        original_payload = json.dumps(pyproject_data, sort_keys=True)
        resolved_versions = (
            load_versions(Path(args.versions_file))
            if args.versions_file
            else collect_installed_versions(pyproject_data, optional_groups=args.optional_groups)
        )
        update_pyproject(pyproject_data, resolved_versions, optional_groups=args.optional_groups)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    updated_payload = json.dumps(pyproject_data, sort_keys=True)

    if args.check:
        if updated_payload != original_payload:
            print("pyproject.toml is not in sync with the tested dependency bounds.", file=sys.stderr)
            return 1
        return 0

    dump_pyproject(pyproject_data, pyproject_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
