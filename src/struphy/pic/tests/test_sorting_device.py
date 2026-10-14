"""Marker exchange between ranks on the CuPy backend (:meth:`Particles.mpi_sort_markers`).

The marker rows are exchanged as device buffers with CUDA-aware MPI, and staged through
host memory otherwise. Without a GPU, the tests run on cunumpy's fake CuPy, which must be
installed before Python imports cunumpy, so they are skipped unless the fake is active::

    mpirun -n 2 env CUNUMPY_FAKE_CUPY=1 pytest --with-mpi src/struphy/pic/tests/test_sorting_device.py

Fake CuPy arrays live in host memory, so a non-CUDA-aware MPI library takes them, and the
CUDA-aware path (the probe of :func:`cunumpy.mpi.mpi_is_cuda_aware` succeeds) is what runs by
default. The staged path is forced with :func:`cunumpy.mpi.set_mpi_cuda_aware`.
"""

import sys
from contextlib import contextmanager

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy
from maybempi import MPI

from struphy import BoundaryParameters, LoadingParameters
from struphy.pic.particles import Particles6D


def _fake_cupy_active() -> bool:
    return bool(getattr(sys.modules.get("cupy"), "__cunumpy_fake__", False))


requires_fake_cupy = pytest.mark.skipif(
    not _fake_cupy_active(),
    reason="needs the fake CuPy: run under `mpirun -n N env CUNUMPY_FAKE_CUPY=1 pytest ...`",
)


@contextmanager
def _count_host_reads():
    """Count reads of fake CuPy arrays by the host (``int()``, ``bool()``, iteration, ``.get()``, ...).

    :func:`cunumpy.profiling.count_transfers` sees the copies made through cunumpy; this also sees
    the implicit ones (a device scalar in an ``if``, ``list(device_array)``), which synchronize
    and copy on a GPU.
    """
    import cupy

    reads = []
    names = ("__int__", "__float__", "__bool__", "__index__", "__iter__", "get")
    originals = {name: getattr(cupy.ndarray, name) for name in names}
    original_getattr = cupy.ndarray.__getattr__

    def counting(name, method):
        def wrapper(self, *args, **kwargs):
            reads.append(name)
            return method(self, *args, **kwargs)

        return wrapper

    def counting_getattr(self, name):
        if name in ("item", "tolist"):
            reads.append(name)
        return original_getattr(self, name)

    for name, method in originals.items():
        setattr(cupy.ndarray, name, counting(name, method))
    cupy.ndarray.__getattr__ = counting_getattr
    try:
        yield reads
    finally:
        for name, method in originals.items():
            setattr(cupy.ndarray, name, method)
        cupy.ndarray.__getattr__ = original_getattr


def _make_particles(comm, bc):
    return Particles6D(
        comm_world=comm,
        loading_params=LoadingParameters(Np=2000, seed=1234),
        boundary_params=BoundaryParameters(bc=(bc, bc, bc)),
    )


def _shift_positions(markers: np.ndarray, valid: np.ndarray, step: int) -> np.ndarray:
    """New positions in [0, 1) for the valid rows, far enough away that most markers change rank."""
    new = markers.copy()
    shift = np.array([0.37, 0.21, 0.53]) * (step + 1)
    new[valid, :3] = (markers[valid, :3] + shift) % 1.0
    return new


def check_mpi_sort_markers_device(cuda_aware: bool | None, bc: str = "periodic", count_reads: bool = True):
    """Sort the same markers on NumPy and CuPy across all ranks and compare the marker arrays.

    ``cuda_aware=None`` uses the probe (:func:`cunumpy.mpi.mpi_is_cuda_aware`); ``True``/``False``
    force the device-buffer or the host-staged exchange.
    """
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    ref = _make_particles(comm, bc)
    ref.draw_markers(sort=False)
    ref.mpi_sort_markers()

    # the reference is sorted on NumPy, the device markers inside `use_backend("cupy")` blocks
    on_device = lambda: xp.use_backend("cupy")  # noqa: E731

    previous = xp.mpi.get_mpi_cuda_aware()
    try:
        with on_device():
            if cuda_aware is not None:
                xp.mpi.set_mpi_cuda_aware(cuda_aware)
            else:
                xp.mpi.set_mpi_cuda_aware(None)
                cuda_aware = xp.mpi.mpi_is_cuda_aware(comm)

            dev = _make_particles(comm, bc)
            assert xp.get_array_backend(dev.markers) == "cupy"
            assert dev.markers.shape == ref.markers.shape

        for step in range(2):
            new = _shift_positions(ref.markers, ref.valid_mks, step)

            # number of rows this rank has to send, from the host reference
            etas = new[ref.valid_mks, :3]
            left, right = ref.domain_array[rank, 0::3], ref.domain_array[rank, 1::3]
            right = np.where(right == 1.0, np.nextafter(right, 2.0), right)
            n_send = int(np.count_nonzero(~np.all((etas >= left) & (etas < right), axis=1)))
            n_send_total = comm.allreduce(n_send)
            assert size == 1 or n_send_total > 0

            ref.markers[:] = new
            ref.update_holes()
            n_before = ref.n_mks_loc
            ref.mpi_sort_markers(do_test=True)
            n_recv = ref.n_mks_loc - (n_before - n_send)

            with on_device():
                dev.markers[:] = xp.to_cupy(new)
                dev.update_holes()
                with xp.profiling.count_transfers() as transfers, _count_host_reads() as reads:
                    dev.mpi_sort_markers()

            if cuda_aware:
                assert transfers.total == 0, transfers.report()
            else:
                # exchange stages each non-empty send and receive buffer separately
                n_dest = sum(1 for i in range(size) if i != rank and dev._send_list[i].shape[0] > 0)
                n_sources = sum(1 for i in range(size) if i != rank and ref._recvbufs[i].shape[0] > 0)
                assert transfers.to_host == n_dest, transfers.report()
                assert transfers.to_device == n_sources, transfers.report()
                sent_bytes = sum(e.nbytes for e in transfers.events if e.kind == "to_host")
                assert sent_bytes == n_send * ref.markers.shape[1] * 8
                received_bytes = sum(e.nbytes for e in transfers.events if e.kind == "to_device")
                assert received_bytes == n_recv * ref.markers.shape[1] * 8

            if count_reads and cuda_aware:
                # (the staged exchange reads the send buffers with .get(), counted above)
                assert reads == [], f"host reads of device arrays during the exchange: {reads}"

            with on_device():
                assert np.array_equal(xp.to_numpy(dev.markers), ref.markers)
                assert np.array_equal(xp.to_numpy(dev.valid_mks), ref.valid_mks)
                n_dev = dev.n_mks_loc
            assert n_dev == ref.n_mks_loc

            with on_device():
                # all markers are on the right rank now: nothing moves
                dev.mpi_sort_markers(do_test=True)
                assert np.array_equal(xp.to_numpy(dev.markers), ref.markers)
    finally:
        xp.mpi.set_mpi_cuda_aware(previous)


@requires_fake_cupy
@pytest.mark.mpi
@pytest.mark.parametrize("cuda_aware", [None, True, False])
@pytest.mark.parametrize("bc", ["periodic", "remove"])
def test_mpi_sort_markers_fake_cupy(cuda_aware, bc):
    """On the fake CuPy: the exchange gives the NumPy result, without host/device copies when MPI is CUDA-aware."""
    check_mpi_sort_markers_device(cuda_aware, bc=bc)


@requires_cupy
@pytest.mark.mpi
@pytest.mark.parametrize("bc", ["periodic", "remove"])
def test_mpi_sort_markers_gpu(bc):
    """On a GPU (``mpirun -n 2``, one GPU per rank): the exchange gives the NumPy result; with
    CUDA-aware MPI without host/device copies."""
    check_mpi_sort_markers_device(None, bc=bc, count_reads=False)


if __name__ == "__main__":
    check_mpi_sort_markers_device(None)
