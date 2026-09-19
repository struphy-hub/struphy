# Logging parameters and filters
import atexit
import logging
import logging.config
import os
from typing import TYPE_CHECKING

from struphy.utils.mpi_launch import launched_under_mpi


class RankZeroFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


class BelowWarningFilter(logging.Filter):
    """Let only DEBUG and INFO records pass (WARNING and above go to stderr)."""

    def filter(self, record):
        return record.levelno < logging.WARNING


# logger configuration
config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(message)s"},
        "detailed": {
            "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "filters": {
        "below_warning": {"()": BelowWarningFilter},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filters": ["below_warning"],
            "stream": "ext://sys.stdout",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            # Overridable via STRUPHY_LOG_FILE so processes sharing a cwd (e.g. several
            # profiling jobs launched from the same case directory) don't rotate the
            # same file concurrently -- RotatingFileHandler's rollover isn't safe across
            # separate OS processes and races with FileNotFoundError when they collide.
            "filename": os.environ.get("STRUPHY_LOG_FILE", "struphy.log"),
            "maxBytes": 10000,
            "backupCount": 3,
        },
    },
    "loggers": {"struphy": {"level": "WARNING", "handlers": ["stdout", "stderr", "file"]}},
}


def set_logging_level(level: int = logging.WARNING):
    """Set logging level for struphy logger and its handlers.

    Useful levels are:
    * logging.DEBUG: for detailed debugging information.
    * logging.INFO: for general informational messages about the simulation setup and progress, plus key events.
    * logging.WARNING: for warnings about potential issues that do not stop the simulation.
    * logging.ERROR: for errors that occur during the simulation, which may affect results but do not necessarily stop the simulation.
    * logging.CRITICAL: for critical errors that likely cause the simulation to stop or produce invalid results.

    Which handler a record ends up in is fixed by the configuration and not changed here:
    DEBUG/INFO go to stdout, WARNING and above to stderr; records that pass the logger level are also written to the log file.
    """
    logger = logging.getLogger("struphy")
    logger.setLevel(level)

    logger.debug(
        f"\nNew logger level: {logger.level}, effective: {logger.getEffectiveLevel()}, propagate: {logger.propagate}"
    )
    for h in logger.handlers:
        logger.debug(f"{type(h).__name__}: handler level: {h.level}")


def setup_logging(logging_level: int = logging.WARNING):
    """Setup logging configuration for struphy."""
    logger = logging.getLogger("struphy")

    log_path = config["handlers"]["file"]["filename"]
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    logging.config.dictConfig(config)

    set_logging_level(logging_level)

    # Add RankZeroFilter to all handlers
    # This helper function figures out whether
    # the current process is launched with mpirun
    # or not without importing mpi4py, which would initialize MPI
    # and cause issues if imported prematurely.
    # Instead, it checks for the presence of certain environment
    # variables that are typically set by MPI launchers (like mpirun or mpiexec).
    if not launched_under_mpi():
        rank = 0
    else:
        # deferred: importing feectools (and thereby mpi4py) is expensive
        from feectools.ddm.mpi import mpi as MPI

        rank = MPI.COMM_WORLD.Get_rank()
    rank_filter = RankZeroFilter(rank)

    # Apply filter to struphy logger handlers
    for handler in logger.handlers:
        handler.addFilter(rank_filter)

    # Apply filter to root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(rank_filter)

    # Start queue handler listener if present
    queue_handler = None
    for handler in logger.handlers:
        if hasattr(handler, "listener"):
            queue_handler = handler
            break
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


# Default logging setup
logger = logging.getLogger("struphy")
setup_logging(logging_level=logging.WARNING)
logger.info(f"Logging setup complete, log-file at {config['handlers']['file']['filename']}")

# Public API components.
#
# They are resolved lazily (PEP 562): ``import struphy`` stays cheap, and
# ``from struphy import X`` only imports the module that actually provides ``X``.
# Importing everything eagerly pulled in the whole package (geometry, models,
# post-processing, feectools, ...), costing several seconds.
_LAZY_API = {
    "Compiler": "struphy.api.compiler",
    "domains": "struphy.api.domains",
    "equils": "struphy.api.equils",
    "grids": "struphy.api.grids",
    "maxwellians": "struphy.api.maxwellians",
    "ButcherTableau": "struphy.api.ode",
    "BaseUnits": "struphy.api.options",
    "DerhamOptions": "struphy.api.options",
    "EnvironmentOptions": "struphy.api.options",
    "FieldsBackground": "struphy.api.options",
    "ProfilingOptions": "struphy.api.options",
    "Time": "struphy.api.options",
    "BinningPlot": "struphy.api.particles",
    "BoundaryParameters": "struphy.api.particles",
    "KernelDensityPlot": "struphy.api.particles",
    "LoadingParameters": "struphy.api.particles",
    "SavingParameters": "struphy.api.particles",
    "SortingParameters": "struphy.api.particles",
    "WeightsParameters": "struphy.api.particles",
    "perturbations": "struphy.api.perturbations",
    "PlottingData": "struphy.api.post_processing",
    "PostProcessor": "struphy.api.post_processing",
    "Simulation": "struphy.api.simulation",
}

if TYPE_CHECKING:  # static analysis and IDEs see the eager imports
    from struphy.api.compiler import Compiler
    from struphy.api.domains import domains
    from struphy.api.equils import equils
    from struphy.api.grids import grids
    from struphy.api.maxwellians import maxwellians
    from struphy.api.ode import ButcherTableau
    from struphy.api.options import (
        BaseUnits,
        DerhamOptions,
        EnvironmentOptions,
        FieldsBackground,
        ProfilingOptions,
        Time,
    )
    from struphy.api.particles import (
        BinningPlot,
        BoundaryParameters,
        KernelDensityPlot,
        LoadingParameters,
        SavingParameters,
        SortingParameters,
        WeightsParameters,
    )
    from struphy.api.perturbations import perturbations
    from struphy.api.post_processing import PlottingData, PostProcessor
    from struphy.api.simulation import Simulation

__all__ = [
    "Compiler",
    "domains",
    "equils",
    "grids",
    "maxwellians",
    "EnvironmentOptions",
    "BaseUnits",
    "Time",
    "ProfilingOptions",
    "perturbations",
    "LoadingParameters",
    "WeightsParameters",
    "BoundaryParameters",
    "SortingParameters",
    "SavingParameters",
    "BinningPlot",
    "KernelDensityPlot",
    "DerhamOptions",
    "FieldsBackground",
    "ButcherTableau",
    "PostProcessor",
    "PlottingData",
    "Simulation",
]


def __getattr__(name: str):
    module_name = _LAZY_API.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_API))
