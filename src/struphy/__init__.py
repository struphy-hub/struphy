# Logging parameters and filters
import logging
import logging.config
import atexit
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
    "simple": {
      "format": "%(message)s"
    },
    "detailed": {
      "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
      "datefmt": "%Y-%m-%dT%H:%M:%S%z"
    }
  },
  "handlers": {
    "stderr": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "simple",
      "stream": "ext://sys.stderr"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "/tmp/struphy.log",
      "maxBytes": 10000,
      "backupCount": 3
    }
  },
  "loggers": {
    "root": {
      "level": "DEBUG",
      "handlers": [
        "stderr",
        "file"
      ]
    }
  }
}

def setup_logging(logging_level: int = logging.INFO):
    logger = logging.getLogger("struphy")

    # config["handlers"]["file"]["level"] = logging_level
    config["handlers"]["stderr"]["level"] = logging_level

    logging.config.dictConfig(config)

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
logger.info("Logging setup complete.")

# Import API components
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
    WeightsParameters,
)
from struphy.api.perturbations import perturbations
from struphy.api.post_processing import PlottingData, PostProcessor
from struphy.api.simulation import Simulation

__all__ = [
    "domains",
    "equils",
    "grids",
    "maxwellians",
    "EnvironmentOptions",
    "BaseUnits",
    "Time",
    "perturbations",
    "LoadingParameters",
    "WeightsParameters",
    "BoundaryParameters",
    "BinningPlot",
    "KernelDensityPlot",
    "DerhamOptions",
    "FieldsBackground",
    "ButcherTableau",
    "PostProcessor",
    "PlottingData",
    "Simulation",
]