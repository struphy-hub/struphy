"""Regression coverage for the public lazy imports."""

import subprocess
import sys


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
