#!/usr/bin/env python3
"""Check that all propagator modules have documentation in propagators-avail.rst."""

from __future__ import annotations

import sys
from pathlib import Path

from doc_check_common import CheckConfig, get_documented_items, get_python_modules, run_check


DOC_FILE = Path("doc/sections/subsections/propagators-avail.rst")
PROPAGATORS_DIR = Path("src/struphy/propagators")


def main() -> int:
    config = CheckConfig(
        label_singular="Propagator",
        label_plural="propagators",
        expected_items_source=str(PROPAGATORS_DIR),
        documentation_source=str(DOC_FILE),
        directive_label="automodule",
        directive_pattern=r"\.\. automodule:: struphy\.propagators\.([a-z_0-9]+)",
        expected_resolver=get_python_modules,
    )

    expected_items = get_python_modules(PROPAGATORS_DIR, {"base.py", "__init__.py"})
    documented_items = get_documented_items(DOC_FILE, config.directive_pattern)
    syntax_targets = [
        (path.stem, path)
        for path in sorted(PROPAGATORS_DIR.glob("*.py"))
        if path.name not in {"base.py", "__init__.py"}
    ]
    return run_check(config, expected_items, documented_items, syntax_targets)


if __name__ == "__main__":
    sys.exit(main())
