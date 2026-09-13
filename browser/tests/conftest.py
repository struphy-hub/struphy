"""Test the built wheel, not the full Struphy installation."""

from pathlib import Path
import sys

if sys.platform != "emscripten":
    wheels = sorted((Path(__file__).resolve().parents[1] / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Run python browser/build.py and keep exactly one wheel in browser/dist")
    sys.path.insert(0, str(wheels[0]))
