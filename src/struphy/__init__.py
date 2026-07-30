# Logging parameters and filters
import atexit
import logging
import logging.config
import os

from feectools.ddm.mpi import mpi as MPI


class RankZeroFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


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
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "WARNING",
            "formatter": "detailed",
            "filename": "struphy.log",
            "maxBytes": 10000,
            "backupCount": 3,
        },
    },
    "loggers": {"struphy": {"level": "WARNING", "handlers": ["stderr", "file"]}},
}


def set_logging_level(level: int = logging.WARNING):
    """Set logging level for struphy logger and its handlers.

    Useful levels are:
    * logging.DEBUG: for detailed debugging information.
    * logging.INFO: for general informational messages about the simulation setup and progress, plus key events.
    * logging.WARNING: for warnings about potential issues that do not stop the simulation.
    * logging.ERROR: for errors that occur during the simulation, which may affect results but do not necessarily stop the simulation.
    * logging.CRITICAL: for critical errors that likely cause the simulation to stop or produce invalid results.
    """
    logger = logging.getLogger("struphy")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

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

# Import API components
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
    "BaseUnits",
    "BinningPlot",
    "BoundaryParameters",
    "ButcherTableau",
    "Compiler",
    "DerhamOptions",
    "EnvironmentOptions",
    "FieldsBackground",
    "KernelDensityPlot",
    "LoadingParameters",
    "PlottingData",
    "PostProcessor",
    "SavingParameters",
    "Simulation",
    "SortingParameters",
    "Time",
    "WeightsParameters",
    "domains",
    "equils",
    "grids",
    "maxwellians",
    "perturbations",
]
