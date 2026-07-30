"""Apply dependency upper bound updates from a check_dependency_bounds.py report.

Reads the JSON report and rewrites only the affected upper bounds in
pyproject.toml — no package installation required.

Example: ``python utils/update_dependency_bounds.py``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.set_release_dependencies import (
    dump_pyproject,
    format_requirement_name,
    get_preserved_specifiers,
    load_pyproject,
    normalize_name,
    split_requirement,
)


def parse_args():
    """Parse command-line arguments for report-driven bound updates."""
    parser = argparse.ArgumentParser(
        description="Apply dependency upper bound updates from a check_dependency_bounds.py report.",
    )
    parser.add_argument("--pyproject-file", default="pyproject.toml", help="Path to pyproject.toml.")
    parser.add_argument(
        "--report-file",
        default="dependency-bounds-report.json",
        help="Path to the JSON report produced by check_dependency_bounds.py.",
    )
    return parser.parse_args()


def apply_update(entry, latest_version):
    """Return the requirement string with the upper bound replaced by <=latest_version.

    Example: ``numpy>=1.25`` + ``1.26.4`` -> ``numpy<=1.26.4, >=1.25``.
    """
    requirement = split_requirement(entry)
    specifiers = get_preserved_specifiers(requirement)
    specifiers.append(f"<={latest_version}")
    updated = format_requirement_name(requirement)
    if specifiers:
        updated += specifiers[0]
        if len(specifiers) > 1:
            updated += ", " + ", ".join(specifiers[1:])
    if requirement["marker"]:
        updated += f"; {requirement['marker']}"
    return updated


def build_update_map(report):
    """Return ({(group_key, normalized_name): latest_stable}, [skipped_items]).

    Example: maps ``('optional:phys', 'plotly')`` -> ``'6.7.0'``.
    """
    updates = {}
    skipped = []
    for item in report.get("outdated", []):
        if item["kind"] == "newer-release-available":
            updates[(item["group"], normalize_name(item["dependency"]))] = item["latest_stable"]
        elif item["kind"] == "missing-upper-bound":
            skipped.append(item)
    return updates, skipped


def apply_updates_to_pyproject(pyproject_data, updates):
    """Apply report-derived updates to matching dependency entries in pyproject data.

    Example: updates only entries whose ``(group, normalized_name)`` key exists in ``updates``.
    """
    changed = False

    new_deps = []
    for entry in pyproject_data["project"]["dependencies"]:
        req = split_requirement(entry)
        key = ("dependencies", normalize_name(req["name"]))
        if key in updates:
            entry = apply_update(entry, updates[key])
            changed = True
        new_deps.append(entry)
    pyproject_data["project"]["dependencies"] = new_deps

    for group_name, group_deps in pyproject_data["project"].get("optional-dependencies", {}).items():
        new_group = []
        for entry in group_deps:
            req = split_requirement(entry)
            key = (f"optional:{group_name}", normalize_name(req["name"]))
            if key in updates:
                entry = apply_update(entry, updates[key])
                changed = True
            new_group.append(entry)
        pyproject_data["project"]["optional-dependencies"][group_name] = new_group

    return changed


def main():
    """Run the update workflow from report loading through pyproject rewrite.

    Example: exits with ``0`` when no updates are needed.
    """
    args = parse_args()
    report = json.loads(Path(args.report_file).read_text(encoding="utf-8"))

    if report.get("status") not in ("outdated",):
        print(f"Report status is '{report.get('status')}', nothing to update.")
        return 0

    updates, skipped = build_update_map(report)

    if skipped:
        print(
            f"Warning: {len(skipped)} item(s) with missing upper bounds cannot be updated automatically "
            "(latest version was not fetched by check_dependency_bounds.py for those):",
            file=sys.stderr,
        )
        for item in skipped:
            print(f"  - {item['dependency']} ({item['group']}): {item['requirement']}", file=sys.stderr)

    if not updates:
        print("No newer-release-available items to update.")
        return 0

    pyproject_path = Path(args.pyproject_file)
    pyproject_data = load_pyproject(pyproject_path)
    changed = apply_updates_to_pyproject(pyproject_data, updates)

    if not changed:
        print("No entries matched the report updates — pyproject.toml unchanged.")
        return 0

    dump_pyproject(pyproject_data, pyproject_path)

    print(f"Updated {len(updates)} upper bound(s) in {pyproject_path}:")
    for item in report["outdated"]:
        if item["kind"] == "newer-release-available":
            print(f"  - {item['dependency']} ({item['group']}): {item['upper_bound']} -> {item['latest_stable']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
