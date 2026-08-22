# Logging parameters and filters
import atexit
import logging
import logging.config
import os

# HDF5's file locking relies on flock(), which is unreliable/unsupported on
# parallel filesystems such as Lustre or GPFS (common on HPC clusters) and
# causes spurious `BlockingIOError: Unable to synchronously open file` errors.
# Disable it unless the user has explicitly configured it.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# mpi4py defaults to requesting MPI_THREAD_MULTIPLE (thread level 3) from
# MPI_Init_thread. On at least one cluster this repo runs on (Pitagora's Booster
# partition, OpenMPI 4.1.6 + UCX 1.20), the UCX worker does not support that level,
# which OpenMPI reports at every multi-rank run ("UCP worker does not support
# MPI_THREAD_MULTIPLE" / "failed to init ucx" / hcoll init failure) and works
# around by making hcoll (its GPU-aware collective component) fail to initialize,
# silently falling back to a different, working collective implementation. Struphy
# only ever calls MPI from the main Python thread (CuPy's internal CUDA driver
# threads don't touch MPI), so requesting the weaker MPI_THREAD_FUNNELED guarantee
# instead is sufficient and avoids the warnings -- but it ALSO lets hcoll
# successfully initialize where it previously failed to, and hcoll's own
# Alltoallv implementation on this cluster then segfaults
# (hmca_bcol_ucx_p2p_alltoallv_pairwise_chunk_progress) the first time it's
# actually used, something the failed init was silently protecting us from.
# hcoll must therefore stay disabled explicitly alongside the thread-level
# change, not just left to fail its own init. Both must be set before mpi4py.MPI
# is imported anywhere (thread level can't change after MPI_Init), and only if
# the user hasn't already configured them themselves.
os.environ.setdefault("MPI4PY_RC_THREAD_LEVEL", "funneled")
os.environ.setdefault("OMPI_MCA_coll_hcoll_enable", "0")

from feectools.ddm.mpi import mpi as MPI

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
            "filename": "struphy.log",
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
