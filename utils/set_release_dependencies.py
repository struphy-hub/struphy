import argparse
import importlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

EXCLUDED_DEPENDENCIES = {"psydac"}
DEFAULT_OPTIONAL_GROUPS = ["phys", "mpi"]
UPPER_BOUND_OPERATORS = {"<", "<="}
SPECIFIER_PATTERN = re.compile(r"(<=|>=|==|!=|~=|<|>)\s*([^,;]+)")


def normalize_name(package_name):
    """Return a normalized package name for case-insensitive matching.

    Example: ``JAX_Finufft`` -> ``jax-finufft``.
    """
    return re.sub(r"[-_.]+", "-", package_name).lower()


def split_requirement(entry):
    """Parse a requirement string into structured parts.

    - extras: optional dependency extras from square brackets (for example, ``pkg[a,b]``).
    - marker: an environment marker after ``;`` (for example, ``python_version < '3.12'``).
    - specifiers: version constraints such as ``>=1.0``, ``<2.0``, ``!=1.5``.
    """
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
    """Build the requirement name portion including optional extras.

    Example: ``{'name': 'foo', 'extras': ['bar']}`` -> ``foo[bar]``.
    """
    extras = ""
    if requirement["extras"]:
        extras = "[" + ",".join(requirement["extras"]) + "]"
    return f"{requirement['name']}{extras}"


def get_preserved_specifiers(requirement):
    """Return non-upper-bound specifiers that should be kept unchanged.

    Example: ``['>=1.0', '<=2.0', '!=1.5']`` -> ``['!=1.5', '>=1.0']``.
    """
    preserved = []
    for specifier in requirement["specifiers"]:
        for operator in UPPER_BOUND_OPERATORS:
            if specifier.startswith(operator):
                break
        else:
            preserved.append(specifier)
    return sorted(preserved)


def build_dependency_entry(entry, resolved_versions, project_name):
    """Return an updated dependency entry with an upper bound from resolved versions.

    Example: ``numpy>=1.25`` + resolved ``1.26.4`` -> ``numpy<=1.26.4, >=1.25``.
    """
    requirement = split_requirement(entry)
    normalized_name = normalize_name(requirement["name"])

    if normalized_name in EXCLUDED_DEPENDENCIES:
        return None

    if normalized_name == normalize_name(project_name):
        return entry

    if any(specifier.startswith("==") for specifier in requirement["specifiers"]):
        # exact pins are deliberate (e.g. ruff) and not managed
        return entry

    resolved_version = resolved_versions.get(normalized_name)
    if resolved_version is None:
        print(
            f"Warning: {requirement['name']} is not available in the tested dependency snapshot, skipping...",
            file=sys.stderr,
        )
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
    """Update all dependencies in a single group and drop excluded entries."""
    updated_dependencies = []
    for entry in dependencies:
        updated_entry = build_dependency_entry(entry, resolved_versions, project_name)
        if updated_entry is not None:
            updated_dependencies.append(updated_entry)
    return updated_dependencies


def get_selected_optional_groups(pyproject_data, optional_groups):
    """Validate and return the selected optional dependency groups.

    Example: ``['phys', 'mpi']`` -> ``{'phys', 'mpi'}``.
    """
    declared_optional_groups = pyproject_data["project"].get("optional-dependencies", {})
    if optional_groups is None:
        return None

    missing_groups = [group_name for group_name in optional_groups if group_name not in declared_optional_groups]
    if missing_groups:
        missing_display = ", ".join(sorted(missing_groups))
        raise ValueError(f"Unknown optional dependency group(s): {missing_display}")

    return set(optional_groups)


def iter_dependency_entries(pyproject_data, optional_groups=None):
    """Yield dependency entries from core dependencies and selected optional groups.

    Example: with ``optional_groups=['phys']``, yields core deps plus ``optional:phys`` entries.
    """
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
    """Collect installed versions for the managed dependency scope.

    Example: returns ``{'numpy': '1.26.4', 'scipy': '1.13.1'}``.
    """
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


def get_toml_reader():
    """Return the TOML reader module (``tomllib``, or ``tomli`` on Python 3.10)."""
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        return importlib.import_module("tomli")


def load_pyproject(pyproject_path):
    """Load and return parsed pyproject.toml content."""
    with pyproject_path.open("rb") as handle:
        return get_toml_reader().load(handle)


def load_versions(versions_path):
    """Load dependency versions from a JSON versions snapshot.

    Example: ``{'versions': {'numpy': '1.26.4'}}`` -> ``{'numpy': '1.26.4'}``.
    """
    with versions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "versions" in payload:
        payload = payload["versions"]

    return {normalize_name(name): version for name, version in payload.items()}


def write_versions_snapshot(pyproject_data, output_path, optional_groups=None):
    """Write installed versions for managed dependencies to a JSON file."""
    payload = {
        "project": pyproject_data["project"]["name"],
        "versions": collect_installed_versions(pyproject_data, optional_groups=optional_groups),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_pyproject(pyproject_data, resolved_versions, optional_groups=None):
    """Update pyproject dependency bounds in-place for the managed scope."""
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


def find_array_span(text, header, key):
    """Return ``(start, end)`` of the TOML array ``key = [...]`` in table ``[header]``.

    ``start`` is the index after the opening ``[``, ``end`` the index of the closing ``]``.
    Quoted strings (which may contain brackets, e.g. ``pkg[extra]``) and comments are skipped.
    """
    table_match = re.search(rf"^\[{re.escape(header)}\][ \t]*$", text, flags=re.MULTILINE)
    if table_match is None:
        raise ValueError(f"Table [{header}] not found in pyproject.toml")
    next_table = re.search(r"^\[", text[table_match.end() :], flags=re.MULTILINE)
    table_end = table_match.end() + next_table.start() if next_table else len(text)

    key_match = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*=[ \t]*\[", flags=re.MULTILINE).search(
        text, table_match.end(), table_end
    )
    if key_match is None:
        raise ValueError(f"Array {key} not found in table [{header}] of pyproject.toml")

    depth = 1
    pos = key_match.end()
    while pos < len(text):
        char = text[pos]
        if char in "\"'":
            pos = text.index(char, pos + 1)
        elif char == "#":
            pos = text.index("\n", pos)
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return key_match.end(), pos
        pos += 1
    raise ValueError(f"Unterminated array {key} in table [{header}] of pyproject.toml")


def rewrite_array(array_text, old_entries, new_entries):
    """Rewrite the changed entries of one dependency array, keeping comments and layout.

    Old and new entries are matched in order by package name. Old entries without a match were dropped;
    they must stand on a line of their own, which is removed.
    """
    replacements = []
    dropped = []
    new_index = 0
    for old_entry in old_entries:
        old_name = normalize_name(split_requirement(old_entry)["name"])
        if (
            new_index < len(new_entries)
            and normalize_name(split_requirement(new_entries[new_index])["name"]) == old_name
        ):
            replacements.append((old_entry, new_entries[new_index]))
            new_index += 1
        else:
            dropped.append(old_entry)
    if new_index != len(new_entries):
        raise ValueError("Dependency arrays can only be changed entry by entry or by dropping entries")

    for entry in dropped:
        for quote in "\"'":
            line_pattern = rf"^[ \t]*{quote}{re.escape(entry)}{quote}[ \t]*,?[ \t]*(#[^\n]*)?\n"
            array_text = re.sub(line_pattern, "", array_text, flags=re.MULTILINE)

    for old_entry, new_entry in replacements:
        if old_entry == new_entry:
            continue
        for quote in "\"'":
            array_text = array_text.replace(f"{quote}{old_entry}{quote}", f'"{new_entry}"')
    return array_text


def dump_pyproject(pyproject_data, pyproject_path):
    """Write the dependency lists of ``pyproject_data`` back to pyproject.toml.

    Only the changed requirement strings in ``[project].dependencies`` and ``[project.optional-dependencies]``
    are rewritten in the text, so comments, ordering and formatting of the file are kept. Raises ``ValueError``
    (and leaves the file unchanged) if the rewritten file does not parse to the expected dependency lists.
    """
    original_text = pyproject_path.read_text(encoding="utf-8")
    original_data = load_pyproject(pyproject_path)

    arrays = [
        ("project", "dependencies", original_data["project"]["dependencies"], pyproject_data["project"]["dependencies"])
    ]
    old_optional = original_data["project"].get("optional-dependencies", {})
    new_optional = pyproject_data["project"].get("optional-dependencies", {})
    for group_name, new_entries in new_optional.items():
        arrays.append(("project.optional-dependencies", group_name, old_optional.get(group_name), new_entries))

    text = original_text
    for header, key, old_entries, new_entries in arrays:
        if old_entries == new_entries:
            continue
        if old_entries is None:
            raise ValueError(f"Cannot add the new dependency group {key} to pyproject.toml")
        start, end = find_array_span(text, header, key)
        text = text[:start] + rewrite_array(text[start:end], old_entries, new_entries) + text[end:]

    written = get_toml_reader().loads(text)["project"]
    if (
        written["dependencies"] != pyproject_data["project"]["dependencies"]
        or written.get("optional-dependencies", {}) != new_optional
    ):
        raise ValueError("Rewriting the dependency bounds in pyproject.toml failed; the file was left unchanged")

    pyproject_path.write_text(text, encoding="utf-8")


def parse_args():
    """Parse command-line arguments for dependency bound updates."""
    parser = argparse.ArgumentParser(description="Update Struphy release dependency bounds.")
    parser.add_argument("--pyproject-file", default="pyproject.toml", help="Path to pyproject.toml.")
    parser.add_argument(
        "--optional-group",
        action="append",
        dest="optional_groups",
        help="Optional dependency group to include. If omitted, only `phys` and `mpi` are processed.",
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
    """Run the dependency-bound update workflow and return an exit code.

    Example: returns ``1`` in ``--check`` mode when ``pyproject.toml`` would change.
    """
    args = parse_args()
    pyproject_path = Path(args.pyproject_file)
    pyproject_data = load_pyproject(pyproject_path)
    optional_groups = args.optional_groups if args.optional_groups is not None else DEFAULT_OPTIONAL_GROUPS

    try:
        if args.write_versions_file:
            write_versions_snapshot(
                pyproject_data,
                Path(args.write_versions_file),
                optional_groups=optional_groups,
            )
            return 0

        original_payload = json.dumps(pyproject_data, sort_keys=True)
        resolved_versions = (
            load_versions(Path(args.versions_file))
            if args.versions_file
            else collect_installed_versions(pyproject_data, optional_groups=optional_groups)
        )
        update_pyproject(pyproject_data, resolved_versions, optional_groups=optional_groups)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    updated_payload = json.dumps(pyproject_data, sort_keys=True)

    if args.check:
        if updated_payload != original_payload:
            print("pyproject.toml is not in sync with the tested dependency bounds.", file=sys.stderr)
            return 1
        return 0

    try:
        dump_pyproject(pyproject_data, pyproject_path)
    except ValueError as exc:
        print(f"Could not rewrite {pyproject_path}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
