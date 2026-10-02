"""Tests for parallel kernel compilation (``struphy compile -j N``).

The first group of tests checks the static dependency detection in :mod:`struphy.dependencies`,
which make relies on to compile kernels in the correct order. The second group checks that the
number of jobs is passed from the command line and the :class:`struphy.Compiler` class to make.
None of the tests actually compile kernels.
"""

import importlib.util
import os
import sys
import sysconfig
import types
from unittest import mock

import pytest

import struphy as struphy_lib
from struphy.console.main import struphy
from struphy.dependencies import get_dependencies

LIBPATH = struphy_lib.__path__[0]
SO_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")


def find_kernels():
    """All kernel sources, detected with the same rule as in struphy.console.compile."""
    kernels = []
    for subdir, _, files in os.walk(LIBPATH):
        for file in files:
            if (
                "kernels" in file
                and ".py" in file
                and "_tmp.py" not in file
                and "test" not in file
                and "__pycache__" not in subdir
                and "__pyccel__" not in subdir
            ):
                kernels += [os.path.join(subdir, file)]
    return sorted(kernels)


def deps_of(kernel_py):
    """Dependencies of a kernel as a set of .py paths."""
    # get_dependencies expects and returns paths of compiled targets (.so)
    deps = get_dependencies(kernel_py.replace(".py", SO_SUFFIX))
    return {d.replace(SO_SUFFIX, ".py") for d in deps.split()}


def deps_by_import(kernel_py):
    """Reference implementation: execute the source and collect all kernel modules bound at module level,
    as well as the kernel modules defining functions and classes bound at module level.

    The former is what get_dependencies did before it was made static.
    """
    stem = os.path.dirname(LIBPATH) + "/"
    # load under a unique name so that the already imported struphy module is not replaced
    spec = importlib.util.spec_from_file_location("_reference_" + os.path.basename(kernel_py)[:-3], kernel_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = set()
    for v in vars(mod).values():
        if isinstance(v, types.ModuleType):
            names.add(v.__name__)
        elif callable(v) and isinstance(getattr(v, "__module__", None), str):
            # functions and classes, also those of compiled kernels
            names.add(v.__module__)
    return {stem + name.replace(".", "/") + ".py" for name in names if "kernels" in name and name.startswith("struphy")}


@pytest.mark.mpi_skip
def test_dependencies_static_rules(tmp_path):
    """Only modules imported at module level, with "kernels" in their name, are dependencies."""
    # fake package tmp_path/struphy/pkg, so that tmp_path acts as the stem of the struphy package
    pkg = tmp_path / "struphy" / "pkg"
    pkg.mkdir(parents=True)
    for name in (
        "a_kernels",
        "b_kernels",
        "d_kernels",
        "e_kernels",
        "f_kernels",
        "g_kernels",
        "h_kernels",
        "plain",
        "__init__",
    ):
        (pkg / f"{name}.py").write_text("")

    (pkg / "c_kernels.py").write_text(
        "\n".join(
            [
                "import struphy.pkg.a_kernels as a_kernels",  # dependency
                "from struphy.pkg import b_kernels",  # dependency (imported name is a module)
                "from . import f_kernels",  # dependency (relative import)
                "if True:",
                "    import struphy.pkg.d_kernels as d_kernels",  # dependency (nested in if)
                "from struphy.pkg.h_kernels import some_function",  # dependency (object imported from a kernel)
                "from struphy.pkg.plain import helper",  # no "kernels" in name
                "import struphy.pkg.plain as plain",  # no "kernels" in name
                "import struphy.pkg.g_kernels",  # binds "struphy", not the kernel
                "import numpy as np",
                "def func():",
                "    import struphy.pkg.e_kernels as e_kernels",  # not module level
                "",
            ]
        )
    )

    kernel = str(pkg / "c_kernels.py")
    expected = {str(pkg / f"{name}.py") for name in ("a_kernels", "b_kernels", "d_kernels", "f_kernels", "h_kernels")}
    assert deps_of(kernel) == expected

    # a kernel without dependencies
    assert get_dependencies(str(pkg / "a_kernels") + SO_SUFFIX) == ""


@pytest.mark.mpi_skip
def test_dependencies_static_does_not_import(tmp_path):
    """Determining dependencies must not execute the kernel source."""
    pkg = tmp_path / "struphy" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "a_kernels.py").write_text("")
    (pkg / "b_kernels.py").write_text(
        "raise RuntimeError('must not be executed')\nimport struphy.pkg.a_kernels as a_kernels\n",
    )

    assert deps_of(str(pkg / "b_kernels.py")) == {str(pkg / "a_kernels.py")}


@pytest.mark.mpi_skip
def test_dependencies_match_import_based_result():
    """For every Struphy kernel, the static dependencies equal those found by executing the source."""
    kernels = find_kernels()
    assert len(kernels) > 0

    for kernel in kernels:
        assert deps_of(kernel) == deps_by_import(kernel), kernel


@pytest.mark.mpi_skip
def test_dependency_graph_is_consistent():
    """Dependencies exist, are themselves kernels (so make can build them) and contain no cycles."""
    kernels = find_kernels()
    graph = {k: deps_of(k) for k in kernels}

    for kernel, deps in graph.items():
        assert kernel not in deps, f"{kernel} depends on itself"
        for dep in deps:
            assert os.path.isfile(dep), f"{kernel} depends on non-existing {dep}"
            assert dep in graph, f"{kernel} depends on {dep}, which is not compiled by struphy compile"

    # topological sort (Kahn's algorithm) has to consume all kernels if there is no cycle
    remaining = {k: set(d) for k, d in graph.items()}
    while remaining:
        ready = [k for k, d in remaining.items() if not d]
        assert ready, f"cyclic dependencies among {sorted(remaining)}"
        for k in ready:
            del remaining[k]
        for d in remaining.values():
            d.difference_update(ready)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("jobs", [None, 1, 4])
def test_compile_passes_jobs_to_make(jobs):
    """struphy_compile calls make with -j<jobs> (default: -j1) for the Struphy kernels."""
    from struphy.console.compile import struphy_compile

    # minimal state with the same language/compiler as requested, so that no prompt is triggered

    state = {
        "kernels": ["some_kernels.py"],
        "last_used_language": "fortran",
        "last_used_compiler": "GNU",
        "last_used_omp": "",
    }
    kwargs = {} if jobs is None else {"jobs": jobs}

    # mock all subprocess calls (psydac-accelerate, make, ...) and the state file
    with (
        mock.patch("struphy.console.compile.subp_run") as subp_run,
        mock.patch("struphy.utils.utils.read_state", return_value=state),
        mock.patch("struphy.utils.utils.save_state"),
    ):
        struphy_compile("fortran", "GNU", None, False, False, False, False, False, False, True, **kwargs)

    make_cmds = [c.args[0] for c in subp_run.call_args_list if c.args[0][0] == "make"]
    assert len(make_cmds) == 1
    assert f"-j{1 if jobs is None else jobs}" in make_cmds[0]
    assert "compile_struphy.mk" in make_cmds[0]


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "args, jobs",
    [
        (["compile"], 1),
        (["compile", "-j", "4"], 4),
        (["compile", "--jobs", "8"], 8),
    ],
)
def test_cli_jobs_option(args, jobs):
    """The command line option -j/--jobs reaches struphy_compile."""
    with (
        mock.patch("struphy.console.compile.struphy_compile") as compile_mock,
        mock.patch.object(sys, "argv", ["struphy"] + args),
    ):
        struphy()

    compile_mock.assert_called_once()
    assert compile_mock.call_args.kwargs["jobs"] == jobs


@pytest.mark.mpi_skip
def test_compiler_class_jobs():
    """Compiler(jobs=...) and Compiler.compile(jobs=...) reach struphy_compile."""
    from struphy import Compiler

    with mock.patch("struphy.utils.compiler.struphy_compile") as compile_mock:
        Compiler(jobs=3).compile()
        assert compile_mock.call_args.kwargs["jobs"] == 3

        Compiler(jobs=3).compile(jobs=8)
        assert compile_mock.call_args.kwargs["jobs"] == 8

        Compiler().compile()
        assert compile_mock.call_args.kwargs["jobs"] == 1
