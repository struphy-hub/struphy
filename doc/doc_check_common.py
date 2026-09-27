#!/usr/bin/env python3
"""Shared utilities for documentation coverage checks."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckConfig:
    """Configuration for a documentation coverage check."""

    label_singular: str
    label_plural: str
    expected_items_source: str
    documentation_source: str
    directive_label: str
    directive_pattern: str
    expected_resolver: callable


def _read_text(path: Path) -> str:
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    return path.read_text()


def get_documented_items(rst_path: Path, directive_pattern: str) -> set[str]:
    """Extract documented items from an RST file using a regex pattern."""

    return set(re.findall(directive_pattern, _read_text(rst_path)))


def validate_python_syntax(path: Path) -> tuple[bool, str | None]:
    """Validate that a Python file parses successfully."""

    if not path.exists():
        return False, f"File not found: {path}"

    try:
        compile(path.read_text(), str(path), "exec")
        return True, None
    except SyntaxError as error:
        return False, f"Syntax error: {error}"
    except Exception as error:  # pragma: no cover - defensive reporting
        return False, f"Error: {error}"


def get_exported_names(init_file: Path) -> set[str]:
    """Read __all__ from a package __init__.py file."""

    module = ast.parse(_read_text(init_file), filename=str(init_file))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    raise ValueError(f"Unsupported __all__ definition in {init_file}")
                names: set[str] = set()
                for element in node.value.elts:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        raise ValueError(f"Unsupported __all__ entry in {init_file}")
                    names.add(element.value)
                return names
    raise ValueError(f"Could not find __all__ in {init_file}")


def get_python_modules(directory: Path, excluded_files: set[str] | None = None) -> set[str]:
    """Return Python module names from a directory, excluding selected files."""

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    excluded = excluded_files or set()
    modules = {
        path.stem
        for path in directory.glob("*.py")
        if path.name not in excluded
    }
    return modules


def run_check(
    config: CheckConfig,
    expected_items: set[str],
    documented_items: set[str],
    syntax_targets: list[tuple[str, Path]],
) -> int:
    """Compare expected and documented items and print a CI-friendly report."""

    print(f"{config.label_singular} Documentation Coverage Check")
    print("=" * 50)
    print(f"Found {len(expected_items)} {config.label_plural} in {config.expected_items_source}")
    print(
        f"Found {len(documented_items)} documented {config.label_plural} "
        f"via {config.directive_label} in {config.documentation_source}"
    )
    print()

    validation_errors: list[tuple[str, str]] = []
    for name, path in sorted(syntax_targets):
        is_valid, error_message = validate_python_syntax(path)
        if not is_valid and error_message is not None:
            validation_errors.append((name, error_message))

    missing_items = expected_items - documented_items
    orphaned_items = documented_items - expected_items

    has_issues = bool(validation_errors or missing_items or orphaned_items)

    if validation_errors:
        print("Validation Errors:")
        for name, error_message in validation_errors:
            print(f"  - {name}: {error_message}")
        print()

    if missing_items:
        print(f"Missing Documentation ({len(missing_items)}):")
        for name in sorted(missing_items):
            print(f"  - {name}")
        print()

    if orphaned_items:
        print(f"Orphaned Documentation ({len(orphaned_items)}):")
        for name in sorted(orphaned_items):
            print(f"  - {name}")
        print()

    if not has_issues:
        print(f"All {config.label_plural} are properly documented.")
        return 0

    print("Documentation check failed.")
    return 1