"""Detection of whether the process was launched by an MPI launcher.

Importing ``mpi4py.MPI`` calls ``MPI_Init``, and any collective (``bcast``,
``Barrier``, ...) issued afterwards costs something even on a single process.
A plain ``python script.py`` run should therefore never touch MPI at all, even
when mpi4py happens to be installed. This module answers the only question
that decides it: was this process started by ``mpirun``/``mpiexec``/``srun``
(or an equivalent launcher)?

The answer is read from the environment the launcher sets up, so it is
available before mpi4py is imported.
"""

import os
import sys

# Per-process variables exported by the process managers behind the common
# launchers. Each is set only for processes started *by* the launcher, so the
# presence of any one of them means "this rank belongs to an MPI job".
# SLURM_PROCID is deliberately absent: it is also set for the script of a
# plain `sbatch` job, which is not an MPI launch. `srun` is covered by the
# PMI/PMIX variables its MPI plugin exports.
_LAUNCHER_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI (and derivatives: Spectrum, ...)
    "PMI_RANK",  # MPICH, Intel MPI, MS-MPI, Cray, srun (pmi2)
    "PMIX_RANK",  # PMIx, used by srun --mpi=pmix and Open MPI 5
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Hydra (mpiexec.hydra)
    "ALPS_APP_PE",  # Cray ALPS aprun
    "PALS_RANKID",  # Cray PALS palsrun
)

# Escape hatch: force the decision either way without touching code, e.g. for
# a launcher whose variables are not listed above.
_OVERRIDE_ENV_VAR = "SCOPE_PROFILER_MPI"

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _override() -> bool | None:
    """Value of ``SCOPE_PROFILER_MPI``, or None if unset/unrecognized."""
    value = os.environ.get(_OVERRIDE_ENV_VAR)
    if value is None:
        return None
    value = value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def launched_under_mpi() -> bool:
    """Whether this process was started by an MPI launcher.

    Returns
    -------
    bool
        True if a launcher's per-rank environment variable is present, or if
        the application itself already initialized MPI (in which case using
        the communicator is free). ``SCOPE_PROFILER_MPI=0``/``1`` overrides
        the detection.
    """
    override = _override()
    if override is not None:
        return override

    if any(var in os.environ for var in _LAUNCHER_ENV_VARS):
        return True

    # The application may have initialized MPI itself (embedded interpreter,
    # or an explicit `from mpi4py import MPI`). Only inspect mpi4py if it is
    # already imported: importing it here is exactly what must be avoided.
    mpi_module = sys.modules.get("mpi4py.MPI")
    if mpi_module is not None:
        try:
            return bool(mpi_module.Is_initialized())
        except AttributeError:
            return False

    return False


def get_comm(use_mpi: bool | None = None):
    """Return ``MPI.COMM_WORLD``, or None when MPI must not be used.

    Parameters
    ----------
    use_mpi : bool or None, optional
        None (default) decides via :func:`launched_under_mpi`. True forces the
        communicator (and fails loudly if mpi4py is missing); False disables
        MPI unconditionally.

    Returns
    -------
    mpi4py.MPI.Intracomm or None
        The world communicator, or None if this is not an MPI run or mpi4py
        is unavailable.
    """
    if use_mpi is False:
        return None

    if use_mpi is None and not launched_under_mpi():
        return None

    try:
        from mpi4py import MPI
    except ImportError:
        if use_mpi:
            raise ImportError(
                "MPI profiling was requested (use_mpi=True) but mpi4py is not "
                "installed."
            )
        # Launched by mpirun without mpi4py available: fall back to treating
        # this rank as a standalone process rather than failing the run.
        return None

    return MPI.COMM_WORLD
