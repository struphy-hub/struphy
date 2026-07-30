import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.set_release_dependencies import load_pyproject, normalize_name, split_requirement

EXIT_IN_SYNC = 0
EXIT_OUTDATED = 1
EXIT_ERROR = 2
DEFAULT_OPTIONAL_GROUPS = ["phys", "mpi"]
NUMERIC_VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


def parse_args():
    """Parse command-line options for dependency bound freshness checks."""
    parser = argparse.ArgumentParser(
        description="Check whether dependency upper bounds lag behind the latest stable releases.",
    )
    parser.add_argument("--pyproject-file", default="pyproject.toml", help="Path to pyproject.toml.")
    parser.add_argument(
        "--optional-group",
        action="append",
        dest="optional_groups",
        help="Optional dependency group to include. If omitted, only `phys` and `mpi` are checked.",
    )
    parser.add_argument(
        "--version-scope",
        choices=("major-minor", "any"),
        default="major-minor",
        help="Version difference that should trigger a failure.",
    )
    parser.add_argument(
        "--report-file",
        default="dependency-bounds-report.json",
        help="Write a machine-readable JSON report to this path.",
    )
    return parser.parse_args()


def parse_numeric_version(version_text):
    """Parse a dotted numeric version string into a tuple of integers.

    Example: ``'1.2.3'`` -> ``(1, 2, 3)``.
    """
    if not NUMERIC_VERSION_PATTERN.match(version_text):
        return None
    return tuple(int(part) for part in version_text.split("."))


def get_managed_dependency_entries(pyproject_data, optional_groups):
    """Return managed dependency entries and unknown optional-group names.

    Example: with ``optional_groups=['phys']``, include core deps plus ``optional:phys``.
    """
    managed_entries = [("dependencies", entry) for entry in pyproject_data["project"]["dependencies"]]

    declared_optional_groups = pyproject_data["project"].get("optional-dependencies", {})
    if not optional_groups:
        return managed_entries, []

    missing_groups = [group_name for group_name in optional_groups if group_name not in declared_optional_groups]
    for group_name in optional_groups:
        for entry in declared_optional_groups.get(group_name, []):
            managed_entries.append((f"optional:{group_name}", entry))
    return managed_entries, missing_groups


def get_upper_bound(requirement):
    """Extract the tightest numeric upper bound from parsed requirement specifiers.

    Example: ``['>=1.0', '<2.0', '<=1.9']`` -> upper bound ``1.9``.
    """
    upper_bound = None
    upper_bound_text = None
    for specifier in requirement["specifiers"]:
        if specifier.startswith("<=") or specifier.startswith("<"):
            version_text = specifier[2:] if specifier.startswith("<=") else specifier[1:]
            candidate = parse_numeric_version(version_text)
            if candidate is None:
                return None, None, f"Invalid upper bound version: {version_text}"

            if upper_bound is None or candidate < upper_bound:
                upper_bound = candidate
                upper_bound_text = version_text

    return upper_bound, upper_bound_text, None


def release_is_usable(files):
    """Return True when a release has at least one non-yanked file.

    Example: ``[{'yanked': True}, {'yanked': False}]`` -> ``True``.
    """
    if not files:
        return True
    return any(not file_info.get("yanked", False) for file_info in files)


def fetch_pypi_payload(package_name, normalized_name, attempts=3, timeout=15):
    """Fetch package metadata from PyPI, retrying and trying normalized names.

    Example: try ``JAX_Finufft`` first, then ``jax-finufft``.
    """
    candidate_names = [package_name]
    if normalized_name not in candidate_names:
        candidate_names.append(normalized_name)

    errors = []
    for candidate_name in candidate_names:
        encoded_name = urllib.parse.quote(candidate_name)
        url = f"https://pypi.org/pypi/{encoded_name}/json"
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    return json.load(response), None
            except urllib.error.HTTPError as exc:
                errors.append(f"{candidate_name}: HTTP {exc.code}")
                if exc.code == 404:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                errors.append(f"{candidate_name}: {exc}")
                if attempt < attempts:
                    time.sleep(attempt)

    return None, "; ".join(errors) if errors else "Unknown PyPI lookup error"


def latest_stable_version(package_name, normalized_name):
    """Return the latest stable numeric release on PyPI for a package.

    Example: returns ``((1, 13, 1), '1.13.1')`` for a latest stable release.
    """
    payload, error_message = fetch_pypi_payload(package_name, normalized_name)
    if payload is None:
        return None, error_message

    stable_versions = []
    for version_text, files in payload.get("releases", {}).items():
        version = parse_numeric_version(version_text)
        if version is None:
            continue
        if not release_is_usable(files):
            continue
        stable_versions.append((version, version_text))

    if not stable_versions:
        return None, "No stable numeric releases found on PyPI"

    return max(stable_versions), None


def version_scope_tuple(version_tuple):
    """Normalize a version tuple to major/minor components for comparison.

    Example: ``(2,)`` -> ``(2, 0)``.
    """
    if len(version_tuple) >= 2:
        return version_tuple[0], version_tuple[1]
    if len(version_tuple) == 1:
        return version_tuple[0], 0
    return 0, 0


def is_outdated(upper_bound, latest_version, version_scope):
    """Decide whether the latest version exceeds the declared upper bound.

    Example: major-minor mode treats ``1.2.9`` vs ``1.2.3`` as in-scope equal.
    """
    latest_version_tuple = latest_version[0]
    if version_scope == "any":
        return latest_version_tuple > upper_bound
    return version_scope_tuple(latest_version_tuple) > version_scope_tuple(upper_bound)


def build_report(pyproject_data, optional_groups, version_scope):
    """Build a structured report describing in-sync, outdated, and error entries.

    Example: report status is ``'outdated'`` when at least one managed bound lags.
    """
    managed_entries, missing_groups = get_managed_dependency_entries(pyproject_data, optional_groups)
    report = {
        "status": "in_sync",
        "version_scope": version_scope,
        "scope": {
            "project_dependencies": True,
            "optional_groups": optional_groups or [],
        },
        "checked": [],
        "outdated": [],
        "errors": [],
        "missing_optional_groups": missing_groups,
    }

    if missing_groups:
        report["status"] = "error"
        report["errors"].append(
            {
                "kind": "missing-optional-group",
                "message": f"Unknown optional dependency group(s): {', '.join(sorted(missing_groups))}",
            }
        )
        return report

    for source_group, entry in managed_entries:
        requirement = split_requirement(entry)
        package_name = requirement["name"]
        normalized_name = normalize_name(package_name)

        upper_bound, upper_bound_text, upper_bound_error = get_upper_bound(requirement)
        entry_report = {
            "group": source_group,
            "name": package_name,
            "requirement": entry,
            "upper_bound": upper_bound_text,
            "latest_stable": None,
            "status": "in_sync",
        }

        if upper_bound_error is not None:
            entry_report["status"] = "error"
            entry_report["message"] = upper_bound_error
            report["errors"].append(
                {
                    "kind": "invalid-upper-bound",
                    "dependency": package_name,
                    "message": upper_bound_error,
                }
            )
            report["checked"].append(entry_report)
            report["status"] = "error"
            continue

        if upper_bound is None:
            entry_report["status"] = "outdated"
            entry_report["message"] = "No upper bound declared"
            report["checked"].append(entry_report)
            report["outdated"].append(
                {
                    "kind": "missing-upper-bound",
                    "dependency": package_name,
                    "group": source_group,
                    "requirement": entry,
                }
            )
            report["status"] = "outdated"
            continue

        latest_version, lookup_error = latest_stable_version(package_name, normalized_name)
        if lookup_error is not None:
            entry_report["status"] = "error"
            entry_report["message"] = lookup_error
            report["errors"].append(
                {
                    "kind": "lookup-error",
                    "dependency": package_name,
                    "message": lookup_error,
                }
            )
            report["checked"].append(entry_report)
            report["status"] = "error"
            continue

        latest_version_tuple, latest_version_text = latest_version
        entry_report["latest_stable"] = latest_version_text
        if is_outdated(upper_bound, latest_version, version_scope):
            entry_report["status"] = "outdated"
            report["outdated"].append(
                {
                    "kind": "newer-release-available",
                    "dependency": package_name,
                    "group": source_group,
                    "requirement": entry,
                    "upper_bound": upper_bound_text,
                    "latest_stable": latest_version_text,
                }
            )
            if report["status"] != "error":
                report["status"] = "outdated"

        report["checked"].append(entry_report)

    return report


def write_report(report, report_file):
    """Write the machine-readable report JSON to disk."""
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report):
    """Print a human-readable summary and return the appropriate exit code.

    Example: returns ``1`` when report status is ``'outdated'``.
    """
    if report["status"] == "error":
        print("Dependency freshness check failed with lookup/configuration errors.", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error['kind']}: {error['message']}", file=sys.stderr)
        return EXIT_ERROR

    if report["status"] == "outdated":
        print("Dependency bounds are outdated for the managed dependency scope.", file=sys.stderr)
        for item in report["outdated"]:
            if item["kind"] == "missing-upper-bound":
                print(
                    f"- {item['dependency']} ({item['group']}): no upper bound declared in {item['requirement']}",
                    file=sys.stderr,
                )
                continue
            print(
                f"- {item['dependency']} ({item['group']}): upper bound {item['upper_bound']} < latest stable {item['latest_stable']}",
                file=sys.stderr,
            )
        return EXIT_OUTDATED

    print("Managed dependency bounds are in sync with the configured version policy.")
    return EXIT_IN_SYNC


def main():
    """Run the freshness check workflow and return a process exit code.

    Example: ``python utils/check_dependency_bounds.py``.
    """
    args = parse_args()
    pyproject_data = load_pyproject(Path(args.pyproject_file))
    optional_groups = args.optional_groups if args.optional_groups is not None else DEFAULT_OPTIONAL_GROUPS
    report = build_report(
        pyproject_data,
        optional_groups=optional_groups,
        version_scope=args.version_scope,
    )

    if args.report_file:
        write_report(report, args.report_file)

    return print_summary(report)


if __name__ == "__main__":
    raise SystemExit(main())