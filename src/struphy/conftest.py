import logging
import re
import shutil

import numpy as np
import pytest
from maybempi import MPI

from struphy import set_logging_level


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


def pytest_addoption(parser):
    parser.addoption("--with-desc", action="store_true")
    parser.addoption("--vrbose", action="store_true")
    parser.addoption("--show-plots", action="store_true")
    parser.addoption("--nclones", type=int, default=1)
    parser.addoption("--model-name", type=str, default="Maxwell")
    parser.addoption("--logging-level", type=str, default="WARNING")
    parser.addoption(
        "--update-baselines",
        action="store_true",
        help="Overwrite the baselines of the model regression tests (models/tests/regression/) instead of comparing.",
    )


@pytest.fixture(autouse=True)
def _seed_numpy_random():
    """Reproducible random numbers in every test: seeds NumPy's global generator (``np.random.rand`` etc.).
    The seed differs between MPI ranks, so that the ranks draw different numbers, as without a seed."""
    np.random.seed(1234 + MPI.COMM_WORLD.Get_rank())


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Store the report of each test phase on the item (``item.rep_call``), used by :func:`mpi_tmp_path`."""
    rep = yield
    setattr(item, f"rep_{rep.when}", rep)
    return rep


@pytest.fixture
def mpi_tmp_path(request, tmp_path_factory) -> str:
    """Output folder of one test, the same on all MPI ranks (use it as ``EnvironmentOptions(out_folders=...)``).

    Rank 0 creates the folder in pytest's temporary directory (``--basetemp``; on clusters with node-local /tmp,
    set it to a shared file system) and broadcasts its path. After a passed test it is removed (after a barrier);
    after a failed test it is kept for inspection (pytest keeps the temporary directories of the last 3 sessions).
    Unlike the current working directory, the folder is unique per test, so tests cannot overwrite each other's
    output.
    """
    comm = MPI.COMM_WORLD
    parallel = comm.Get_size() > 1
    path = None
    if comm.Get_rank() == 0:
        name = re.sub(r"[^\w.-]", "_", request.node.name)[:60]
        path = str(tmp_path_factory.mktemp(name))
    if parallel:
        path = comm.bcast(path, root=0)

    yield path

    if parallel:
        comm.Barrier()
    rep = getattr(request.node, "rep_call", None)
    if comm.Get_rank() == 0 and rep is not None and rep.passed:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def update_baselines(request) -> bool:
    """Value of the ``--update-baselines`` option, see models/tests/regression/."""
    return request.config.getoption("--update-baselines")


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
