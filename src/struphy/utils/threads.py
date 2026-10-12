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

from threadpoolctl import ThreadpoolController

logger = logging.getLogger("struphy")

BLAS_THREAD_ENV_VARS = {
    "openblas": ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"),
    "mkl": ("MKL_NUM_THREADS",),
    "blis": ("BLIS_NUM_THREADS",),
}
"""Environment variables by which the user sets the number of threads of each BLAS library
(keys: threadpoolctl's ``internal_api``) explicitly."""


def limit_blas_threads(comm, num_threads: int = 1) -> bool:
    """Limit the threads of all loaded BLAS libraries to ``num_threads`` in MPI runs.

    Nothing is done if ``comm`` has only one rank (or is None). A BLAS library is left
    unchanged if the user has set its environment variable, see :data:`BLAS_THREAD_ENV_VARS`.
    The OpenMP thread pools of the Struphy kernels are left unchanged (they are restored if a
    BLAS library built with OpenMP changes them).

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
        True if at least one BLAS library was limited and all limited libraries kept the limit.
    """
    comm_size = 1 if comm is None else comm.Get_size()
    if comm_size == 1:
        return False

    # threadpoolctl only sees libraries that are already loaded;
    # scipy brings its own BLAS copy (used by SuperLU and LAPACK), separate from numpy's.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import scipy.sparse.linalg  # noqa: F401

    libs = ThreadpoolController().lib_controllers
    blas_libs = [lib for lib in libs if lib.user_api == "blas"]
    omp_threads = [(lib, lib.num_threads) for lib in libs if lib.user_api == "openmp"]

    if not blas_libs:
        logger.info("No BLAS library with a controllable thread pool found, BLAS threads not limited.")
        return False

    limited = []
    user_vars = []
    for lib in blas_libs:
        set_vars = [var for var in BLAS_THREAD_ENV_VARS.get(lib.internal_api, ()) if var in os.environ]
        if set_vars:
            user_vars += set_vars
        else:
            lib.set_num_threads(num_threads)
            limited.append(lib)

    # a BLAS built with OpenMP may have changed the OpenMP pools, too
    for lib, n in omp_threads:
        if lib.num_threads != n:
            lib.set_num_threads(n)

    if user_vars:
        logger.info(f"BLAS threads set by {', '.join(sorted(set(user_vars)))}, not limited by Struphy.")
    if not limited:
        return False

    # such a BLAS follows the OpenMP pool again after the restore
    if any(lib.num_threads > num_threads for lib in limited):
        logger.warning(
            "Could not limit the BLAS threads without changing the OpenMP threads (BLAS built with OpenMP). "
            "Set OMP_NUM_THREADS (and OPENBLAS_NUM_THREADS) per rank to avoid oversubscribing the cores.",
        )
        return False

    omp_info = max(n for _, n in omp_threads) if omp_threads else "no OpenMP runtime loaded"
    logger.info(
        f"Limited BLAS to {num_threads} thread(s) per MPI rank ({comm_size} ranks); "
        f"OpenMP threads per rank: {omp_info}. Set OPENBLAS_NUM_THREADS to override.",
    )
    return True
