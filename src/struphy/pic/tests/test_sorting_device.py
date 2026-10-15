"""Marker exchange between ranks (:meth:`Particles.mpi_sort_markers`) and sorting boxes on the CuPy backend.

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

from struphy import BoundaryParameters, LoadingParameters, SortingParameters
from struphy.pic.particles import Particles6D, ParticlesSPH
from struphy.pic.tests.test_sorting import assert_boxes_match_markers


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
    """New positions in [0, 1) for the valid markers, far enough away that most markers change rank."""
    new = markers.copy()
    shift = np.array([0.37, 0.21, 0.53])[:, None] * (step + 1)
    new[:3, valid] = (markers[:3, valid] + shift) % 1.0
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

            # number of markers this rank has to send, from the host reference
            etas = new[:3, ref.valid_mks]
            left, right = ref.domain_array[rank, 0::3, None], ref.domain_array[rank, 1::3, None]
            right = np.where(right == 1.0, np.nextafter(right, 2.0), right)
            n_send = int(np.count_nonzero(~np.all((etas >= left) & (etas < right), axis=0)))
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
                n_dest = sum(1 for i in range(size) if i != rank and dev._send_list[i].shape[1] > 0)
                n_sources = sum(1 for i in range(size) if i != rank and ref._recvbufs[i].shape[1] > 0)
                assert transfers.to_host == n_dest, transfers.report()
                assert transfers.to_device == n_sources, transfers.report()
                sent_bytes = sum(e.nbytes for e in transfers.events if e.kind == "to_host")
                assert sent_bytes == n_send * ref.n_cols * 8
                received_bytes = sum(e.nbytes for e in transfers.events if e.kind == "to_device")
                assert received_bytes == n_recv * ref.n_cols * 8

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


# divisible by the number of processes in each direction for 1 to 4 ranks
BOXES_PER_DIM = (12, 12, 12)


def check_sorting_boxes_device():
    """Sorting boxes on NumPy and CuPy across all ranks: the same setup, and one :meth:`do_sort` of the same markers.

    On a rank > 0 the box kernels and the setup read the row of that rank from the host copy of ``domain_array``, and
    SPH particles find their neighbouring processes with a real domain decomposition (for more than one rank).
    """
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    def make_particles():
        return Particles6D(
            comm_world=comm,
            loading_params=LoadingParameters(Np=4000, seed=1234),
            sorting_params=SortingParameters(boxes_per_dim=BOXES_PER_DIM),
        )

    ref = make_particles()
    ref.draw_markers(sort=False)
    ref.mpi_sort_markers()
    markers_in = ref.markers.copy()
    ref.do_sort()
    assert_boxes_match_markers(ref)
    assert ref.n_mks_loc > 0
    # all markers of this rank lie in its domain, so none is in the out-of-domain box (last row of the box array):
    # the boxes were found with the domain_array row of this rank
    out_box = ref.sorting_boxes.boxes.shape[0] - 1
    assert np.all(ref.markers[ref.sorting_boxes.box_index, ref.valid_mks] < out_box)

    with xp.use_backend("cupy"):
        dev = make_particles()
        assert xp.get_array_backend(dev.domain_array) == "cupy"
        assert isinstance(dev._domain_array_host, np.ndarray)
        assert np.array_equal(dev._domain_array_host, ref.domain_array)

        dev.markers[:] = xp.to_cupy(markers_in)
        dev.update_holes()
        with xp.profiling.count_transfers() as transfers:
            dev.do_sort()
        assert np.array_equal(xp.to_numpy(dev.markers), ref.markers)
        for name in ("_boxes", "_next_index", "_cumul_next_index"):
            assert np.array_equal(xp.to_numpy(getattr(dev.sorting_boxes, name)), getattr(ref.sorting_boxes, name))
        assert_boxes_match_markers(dev)

    # the markers cross between device and host once in each direction (#694: no CUDA box kernels yet)
    markers_bytes = ref.markers.nbytes
    assert sum(e.kind == "to_host" and e.nbytes == markers_bytes for e in transfers.events) == 1, transfers.report()
    assert sum(e.kind == "to_device" and e.nbytes == markers_bytes for e in transfers.events) == 1, transfers.report()

    def make_sph():
        return ParticlesSPH(
            comm_world=comm,
            loading_params=LoadingParameters(Np=4000, seed=1234),
            sorting_params=SortingParameters(boxes_per_dim=BOXES_PER_DIM),
        )

    ref_sph = make_sph()
    with xp.use_backend("cupy"):
        dev_sph = make_sph()

    assert dev_sph.sorting_boxes.communicate
    is_domain_boundary = dev_sph.sorting_boxes.is_domain_boundary
    assert is_domain_boundary == ref_sph.sorting_boxes.is_domain_boundary
    assert all(type(v) is bool for v in is_domain_boundary.values())
    assert np.array_equal(xp.to_numpy(dev_sph.sorting_boxes.neighbours), ref_sph.sorting_boxes.neighbours)
    for side in ("x_m", "x_p", "y_m", "y_p", "z_m", "z_p"):
        name = f"_bnd_boxes_{side}"
        assert getattr(dev_sph.sorting_boxes, name) == getattr(ref_sph.sorting_boxes, name), name

    # the neighbouring process in each of the 26 directions
    procs = sorted(name for name in vars(ref_sph) if name.endswith("_proc"))
    assert len(procs) == 26, procs
    dev_procs = {name: getattr(dev_sph, name) for name in procs}
    ref_procs = {name: getattr(ref_sph, name) for name in procs}
    assert all(p is None or isinstance(p, (int, np.integer)) for p in dev_procs.values()), dev_procs
    assert dev_procs == ref_procs
    if size > 1:
        # a real decomposition: some neighbours are other ranks
        assert any(p is not None and p != rank for p in ref_procs.values()), ref_procs


@requires_fake_cupy
@pytest.mark.mpi
def test_sorting_boxes_fake_cupy():
    """On the fake CuPy, on every rank: sorting boxes and :meth:`do_sort` give the NumPy result."""
    check_sorting_boxes_device()


@requires_cupy
@pytest.mark.mpi
def test_sorting_boxes_gpu():
    """On a GPU (``mpirun -n 2``, one GPU per rank): sorting boxes and :meth:`do_sort` give the NumPy result."""
    check_sorting_boxes_device()


if __name__ == "__main__":
    check_mpi_sort_markers_device(None)
