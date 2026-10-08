#!/usr/bin/env python3
"""Check that all public models have documentation in models-all.rst."""

from __future__ import annotations

import sys
from pathlib import Path

from doc_check_common import CheckConfig, get_documented_items, get_exported_names, run_check

DOC_FILE = Path("doc/sections/subsections/models-all.rst")
MODELS_INIT = Path("src/struphy/models/__init__.py")
MODELS_DIR = Path("src/struphy/models")


def main() -> int:
    config = CheckConfig(
        label_singular="Model",
        label_plural="models",
        expected_items_source=str(MODELS_INIT),
        documentation_source=str(DOC_FILE),
        directive_label="autoclass",
        directive_pattern=r"\.\. autoclass:: struphy\.models\.([A-Za-z_][A-Za-z0-9_]*)",
        expected_resolver=get_exported_names,
    )

    expected_items = get_exported_names(MODELS_INIT)
    documented_items = get_documented_items(DOC_FILE, config.directive_pattern)
    syntax_targets = [
        ("models.__init__", MODELS_INIT),
        *[(path.stem, path) for path in sorted(MODELS_DIR.glob("*.py")) if path.name != "__init__.py"],
    ]
    return run_check(config, expected_items, documented_items, syntax_targets)


if __name__ == "__main__":
    sys.exit(main())
