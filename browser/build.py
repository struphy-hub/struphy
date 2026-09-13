"""Build a pure-Python geometry wheel from the shared Struphy sources.

Run with Python >= 3.11: python browser/build.py
No compiler, installed Struphy, or third-party build dependency is needed.
"""

import argparse
import ast
import base64
import csv
import hashlib
import io
from pathlib import Path
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "geometry/base.py",
    "geometry/domains.py",
    "geometry/utilities.py",
    "geometry/mappings_kernels.py",
    "geometry/evaluation_kernels.py",
    "geometry/transform_kernels.py",
    "geometry/utilities_kernels.py",
    "geometry/flux.py",
    "geometry/portable.py",
    "bsplines/bsplines.py",
    "bsplines/bsplines_kernels.py",
    "bsplines/evaluation_kernels_2d.py",
    "bsplines/evaluation_kernels_3d.py",
    "kernel_arguments/pusher_args_kernels.py",
    "linear_algebra/linalg_kernels.py",
    "linear_algebra/linalg_kron.py",
    "utils/class_helpers.py",
    "utils/docstring_converter.py",
)


class BrowserSource(ast.NodeTransformer):
    """Remove explicitly unsupported adapters and compiler-only annotations.

    Numerical function bodies are shared verbatim in meaning with Struphy.
    cuNumPy, including its callable wrapper, remains a runtime dependency.
    """

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return None
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name):
            if node.test.left.id == "__name__":
                return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name in ("GVECunit", "DESCunit"):
            return None
        if node.name == "Domain":
            omitted = {"create_geometry_mesh", "show_3d", "export_geometry"}
            node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name not in omitted]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.decorator_list = [
            d
            for d in node.decorator_list
            if not (isinstance(d, ast.Name) and d.id == "pure")
            and not (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "stack_array")
        ]
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "pyccel.decorators":
            if any(n.name not in ("pure", "stack_array") for n in node.names):
                raise ValueError("Review new Pyccel decorators before adding them to the browser build")
            return None
        if node.module and node.module.startswith("struphy.fields_background"):
            names = {n.name for n in node.names}
            if "EQDSKequilibrium" in names:
                message = (
                    "Tokamak requires equilibrium=CircularFlux() or EQDSKFlux.from_text(text) in the browser package"
                )
            elif "GVECequilibrium" in names:
                message = (
                    "Spline requires explicit cx, cy, cz control points; use load_geometry for exported equilibria"
                )
            else:
                message = (
                    "Equilibrium reconstruction is unavailable; use save_geometry/load_geometry for spline domains"
                )
            return ast.Raise(
                exc=ast.Call(func=ast.Name(id="ValueError", ctx=ast.Load()), args=[ast.Constant(message)], keywords=[])
            )
        return node


def browser_source(source):
    tree = BrowserSource().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    # Preserve leading copyright comments as well as the module docstring.
    header = []
    for line in source.splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            break
    text = "\n".join(header) + "\n" + ast.unparse(tree) + "\n"
    return text.replace("struphy.", "struphy_geometry.")


def build(output):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    package = "struphy_geometry"
    files = {}
    for path in SOURCES:
        files[f"{package}/{path}"] = browser_source((ROOT / "src/struphy" / path).read_text()).encode()
        files[f"{package}/{Path(path).parent}/__init__.py"] = b""
    files[f"{package}/__init__.py"] = f'''"""Struphy geometry on NumPy, including Pyodide. No compilation required."""
from .geometry import domains
from .geometry.base import Domain, Spline, PoloidalSplineStraight, PoloidalSplineTorus, interp_mapping, spline_interpolation_nd
from .geometry.flux import CircularFlux, EQDSKFlux
from .geometry.portable import sample_surface, save_geometry, load_geometry
__version__ = {version!r}
__all__ = ["domains", "Domain", "Spline", "PoloidalSplineStraight", "PoloidalSplineTorus", "interp_mapping", "spline_interpolation_nd", "CircularFlux", "EQDSKFlux", "sample_surface", "save_geometry", "load_geometry"]
'''.encode()
    info = f"struphy_geometry-{version}.dist-info"
    files[f"{info}/METADATA"] = f"""Metadata-Version: 2.1
Name: struphy-geometry
Version: {version}
Summary: Portable Struphy domains, mappings, and flux geometry for Python and Pyodide
Requires-Python: >=3.10
Requires-Dist: numpy
Requires-Dist: scipy
Requires-Dist: cunumpy>=0.1.4,<=0.1.5
Provides-Extra: plot
Requires-Dist: matplotlib; extra == "plot"
License-File: LICENSE

Built from the shared Struphy geometry sources. See browser/README.md in the Struphy repository.
""".encode()
    files[f"{info}/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    files[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: struphy-geometry-build\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for path, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([path, "sha256=" + digest, len(data)])
    writer.writerow([f"{info}/RECORD", "", ""])
    files[f"{info}/RECORD"] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"struphy_geometry-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path, data in sorted(files.items()):
            entry = zipfile.ZipInfo(path, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            wheel.writestr(entry, data)
    print(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dist")
    build(parser.parse_args().output)
