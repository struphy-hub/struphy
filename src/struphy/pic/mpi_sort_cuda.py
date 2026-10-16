"""The destinations of :meth:`~struphy.pic.base.Particles.mpi_sort_markers` on the GPU (``mpi_sort_cuda.cu``).

:meth:`~struphy.pic.base.Particles._mpi_sort_device` finds the rank of every leaving marker with one kernel
(:func:`sort_destinations`) instead of a pass over the buffer per rank and per direction.
"""

from functools import cache
from pathlib import Path

import numpy as np


@cache
def _kernel():
    from cunumpy.kernels import CudaKernel

    return CudaKernel.from_file(Path(__file__).with_name("mpi_sort_cuda.cu"), "sort_destinations")


def domain_bounds(domain_array) -> np.ndarray:
    """The ends of each rank's domain per direction, as :meth:`~struphy.pic.base.Particles._is_on_domain_of_rank` tests
    them: an array (n_ranks, 3, 2) of [left, right), with the right end 1 shifted to the next float (eta = 1 belongs to
    the last rank).

    Parameters
    ----------
    domain_array : array
        (n_ranks, 9) host array of :attr:`~struphy.pic.base.Particles.domain_array` (left, right, cells per direction).
    """
    domain_array = np.asarray(domain_array, dtype=float)
    left = domain_array[:, 0::3]
    right = domain_array[:, 1::3]
    right = np.where(right == 1.0, np.nextafter(right, 2.0), right)
    return np.ascontiguousarray(np.stack([left, right], axis=-1))


def sort_destinations(
    markers,
    holes,
    ghosts,
    shift_row: int,
    init_row: int,
    alpha,
    use_alpha: bool,
    periodic,
    bounds,
    ranks,
    rank: int,
    n_used: int,
    dest,
    counts,
):
    """Launch the kernel ``sort_destinations`` (one thread per marker, see ``mpi_sort_cuda.cu``).

    Parameters
    ----------
    markers : cupy.ndarray
        Marker buffer (n_cols, n_markers_max).
    holes, ghosts : cupy.ndarray
        Hole and ghost masks (bool, n_markers_max).
    shift_row, init_row : int
        First rows of the boundary shifts and of the initial positions.
    alpha : cupy.ndarray
        The three sorting weights (float64, 3).
    use_alpha : bool
        Whether to use them; else the sorting position is position + shift (alpha = 1, computed exactly).
    periodic : cupy.ndarray
        Per direction, 1 if periodic (int32, 3).
    bounds : cupy.ndarray
        :func:`domain_bounds` on the device (float64).
    ranks : cupy.ndarray
        The other ranks, neighbours first (int32).
    rank, n_used : int
        This rank and the number of markers (columns) that can hold markers.
    dest : cupy.ndarray
        Output, int32 (n_markers_max): -1 stay, -2 lost, else the destination rank.
    counts : cupy.ndarray
        Output, uint64 (n_ranks + 2), added to: leaving markers per destination + 2.
    """
    if n_used == 0:
        return
    _kernel()(
        markers,
        holes,
        ghosts,
        int(shift_row),
        int(init_row),
        alpha,
        int(bool(use_alpha)),
        periodic,
        bounds,
        ranks,
        int(ranks.size),
        int(rank),
        int(n_used),
        dest,
        counts,
        n_threads=int(n_used),
    )
