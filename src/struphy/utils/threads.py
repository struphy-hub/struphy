"""Thread limits for threaded BLAS libraries in MPI runs.

numpy and scipy each ship a threaded BLAS (usually OpenBLAS) that starts one thread per core
it can see. Under MPI with several ranks per node the BLAS threads of all ranks oversubscribe
the cores, which makes small BLAS calls, e.g. the SuperLU solves of the commuting projectors,
up to a few hundred times slower (struphy issue #743).

The OpenMP kernels of Struphy (pushers, ...) run in a separate thread pool (the OpenMP runtime,
controlled by ``OMP_NUM_THREADS``) and are not limited here.
"""

import logging
import os

from threadpoolctl import threadpool_info, threadpool_limits

logger = logging.getLogger("struphy")

BLAS_THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
"""Environment variables by which the user sets the number of BLAS threads explicitly."""


def _num_threads(user_api: str) -> list[int]:
    """Number of threads of each loaded library of the given ``user_api`` ("blas" or "openmp")."""
    return [lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == user_api]


def limit_blas_threads(comm, num_threads: int = 1) -> bool:
    """Limit the threads of all loaded BLAS libraries to ``num_threads`` in MPI runs.

    Nothing is done if ``comm`` has only one rank (or is None), or if the user has set one of
    :data:`BLAS_THREAD_ENV_VARS`. The OpenMP thread pool of the Struphy kernels is left unchanged
    (it is restored if a BLAS library built with OpenMP changes it).

    :meth:`Simulation.run <struphy.simulation.sim.Simulation.run>` calls this function at
    start-up. Scripts that use the FEEC/PIC data structures directly (without ``Simulation``)
    should call it once after creating the communicator.

    Parameters
    ----------
    comm : MPI.Intracomm | None
        MPI communicator of the run, None in serial runs.

    num_threads : int
        Maximum number of threads per BLAS library and rank.

    Returns
    -------
    bool
        True if the BLAS threads were limited.
    """
    comm_size = 1 if comm is None else comm.Get_size()
    if comm_size == 1:
        return False

    user_vars = [var for var in BLAS_THREAD_ENV_VARS if var in os.environ]
    if user_vars:
        logger.info(f"BLAS threads set by the user ({', '.join(user_vars)}), not limited by Struphy.")
        return False

    # threadpoolctl only acts on libraries that are already loaded;
    # scipy brings its own BLAS copy (used by SuperLU and LAPACK), separate from numpy's.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import scipy.sparse.linalg  # noqa: F401

    omp_threads = _num_threads("openmp")
    threadpool_limits(limits=num_threads, user_api="blas")

    # a BLAS built with OpenMP may have changed the OpenMP pool, too
    if omp_threads and _num_threads("openmp") != omp_threads:
        threadpool_limits(limits=max(omp_threads), user_api="openmp")

    logger.info(
        f"Limited BLAS to {num_threads} thread(s) per MPI rank ({comm_size} ranks); "
        f"OpenMP threads per rank: {max(omp_threads) if omp_threads else 'no OpenMP runtime loaded'}. "
        f"Set {BLAS_THREAD_ENV_VARS[0]} to override.",
    )
    return True
