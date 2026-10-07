"""The numba kernel generated from ``bstar_parallel_3form`` agrees with pyccel (issue #690 prototype).

Skipped unless numba (with numba-cuda for the simulator) is installed; numba is not a struphy dependency.
Runs in a subprocess because ``NUMBA_ENABLE_CUDASIM`` must be set before numba is imported.
"""

import importlib.util
import subprocess
import sys

import pytest

from struphy.geometry.tests.test_domain import serial_child_env

requires_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba is not installed")


@requires_numba
@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_generated_kernel_agrees_with_pyccel(target):
    """``cpu``: the translation compiled with numba.njit; ``cuda``: the numba.cuda kernel in the CUDA simulator."""
    extra = {"NUMBA_ENABLE_CUDASIM": "1"} if target == "cuda" else {}
    result = subprocess.run(
        [sys.executable, "-m", "struphy.pic.tests.codegen_spike.compare", target],
        env=serial_child_env(**extra),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
