import logging

import pytest

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
