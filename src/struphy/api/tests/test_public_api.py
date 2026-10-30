"""Regression coverage for the public lazy imports."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mpi_skip


def test_post_processing_exports():
    # A fresh interpreter also verifies that package import stays lazy.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import struphy

assert 'struphy.api.post_processing' not in sys.modules
assert 'struphy.post_processing.output' not in sys.modules
assert {'Output', 'open_output'} <= set(dir(struphy))

from struphy import Output, open_output
from struphy.post_processing import output

assert Output is output.Output
assert open_output is output.open_output
assert struphy.Output is Output
assert struphy.open_output is open_output
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_api_models_match_models():
    # Parse instead of import: importing struphy.api.models loads every model.
    import struphy.models

    api_init = Path(struphy.models.__file__).parents[1] / "api" / "models" / "__init__.py"
    tree = ast.parse(api_init.read_text())
    (api_all,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets)
    ]
    api_imports = {
        alias.name: node.module for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert sorted(api_all) == sorted(struphy.models.__all__)
    assert sorted(struphy.models.__all__) == sorted(struphy.models._LAZY_IMPORTS)
    # every exported name is imported from the module that defines it
    assert api_imports == struphy.models._LAZY_IMPORTS
