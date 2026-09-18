import logging
import os

import pytest

from struphy import set_logging_level

# Tests marked "needs_host_kernels" call a Pyccel kernel that is not yet
# ported for the CuPy backend (e.g. FEEC mass-matrix/basis-projection
# assembly, particle-to-grid accumulation). Under ARRAY_BACKEND=cupy they are
# skipped rather than run to failure, so a GPU CI run reports the state of
# the actually-ported code paths instead of drowning in known gaps.
_NEEDS_HOST_KERNELS_SKIP_REASON = (
    "needs_host_kernels: not yet ported to the CuPy backend (ARRAY_BACKEND=cupy)"
)


def set_logging_level_pytest(config):
    level_name = str(config.getoption("--logging-level")).upper()
    level = getattr(logging, level_name, None)
    if level is None or not isinstance(level, int):
        raise pytest.UsageError(
            f"Invalid --logging-level '{level_name}'. Use one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    set_logging_level(level)


def pytest_unconfigure(config):
    if hasattr(config, "testmon_data"):
        config.testmon_data.db.con.close()


def pytest_configure(config):
    set_logging_level_pytest(config)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ARRAY_BACKEND") != "cupy":
        return
    skip_host_only = pytest.mark.skip(reason=_NEEDS_HOST_KERNELS_SKIP_REASON)
    for item in items:
        if "needs_host_kernels" in item.keywords:
            item.add_marker(skip_host_only)


def pytest_addoption(parser):
    parser.addoption("--with-desc", action="store_true")
    parser.addoption("--vrbose", action="store_true")
    parser.addoption("--show-plots", action="store_true")
    parser.addoption("--nclones", type=int, default=1)
    parser.addoption("--model-name", type=str, default="Maxwell")
    parser.addoption("--logging-level", type=str, default="WARNING")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".])

    option_value = metafunc.config.option.with_desc
    if "with_desc" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("with_desc", [option_value])

    option_value = metafunc.config.option.vrbose
    if "vrbose" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("vrbose", [option_value])

    option_value = metafunc.config.option.nclones
    if "nclones" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("nclones", [option_value])

    option_value = metafunc.config.option.show_plots
    if "show_plots" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("show_plots", [option_value])

    option_value = metafunc.config.option.model_name
    if "model_name" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("model_name", [option_value])
