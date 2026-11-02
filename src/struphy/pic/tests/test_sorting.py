import logging
from time import time

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import BoundaryParameters, LoadingParameters, SortingParameters, WeightsParameters, domains
from struphy.feec.psydac_derham import Derham
from struphy.geometry.tests.test_domain import _cupy_installed, run_fake_cupy_child
from struphy.io.options import DerhamOptions
from struphy.pic.particles import Particles6D, ParticlesSPH
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


def assert_boxes_match_markers(particles):
    """The box array lists, for every box, exactly the markers (columns, not holes) whose box row is that box,
    in index order and padded with -1, and ``next_index`` holds their number."""
    sorting_boxes = particles.sorting_boxes
    boxes = xp.to_numpy(sorting_boxes.boxes)
    next_index = xp.to_numpy(sorting_boxes._next_index)
    markers = xp.to_numpy(particles.markers)
    holes = xp.to_numpy(particles.holes)
    box_of_row = markers[sorting_boxes.box_index].astype(int)
    for box in range(boxes.shape[0]):
        rows = np.flatnonzero(~holes & (box_of_row == box))
        assert next_index[box] == rows.size, box
        assert np.array_equal(boxes[box, : rows.size], rows), box
        assert np.all(boxes[box, rows.size :] == -1), box


@pytest.mark.parametrize("nx", [8, 70])
@pytest.mark.parametrize("ny", [16, 80])
@pytest.mark.parametrize("nz", [32, 90])
@pytest.mark.parametrize("algo", ["fortran_ordering", "c_ordering"])
def test_flattening_fortran(nx, ny, nz, algo):
    from struphy.pic.sorting_kernels import flatten_index, unflatten_index

    n1s = xp.array(xp.random.rand(10) * (nx + 1), dtype=int)
    n2s = xp.array(xp.random.rand(10) * (ny + 1), dtype=int)
    n3s = xp.array(xp.random.rand(10) * (nz + 1), dtype=int)
    for n1 in n1s:
        for n2 in n2s:
            for n3 in n3s:
                n_glob = flatten_index(int(n1), int(n2), int(n3), nx, ny, nz, algo)
                n1n, n2n, n3n = unflatten_index(n_glob, nx, ny, nz, algo)
                assert n1n == n1
                assert n2n == n2
                assert n3n == n3


@pytest.mark.parametrize("nx", [8, 70])
@pytest.mark.parametrize("ny", [16, 80])
@pytest.mark.parametrize("nz", [32, 90])
@pytest.mark.parametrize("algo", ["fortran_ordering", "c_ordering"])
def test_flattening_c(nx, ny, nz, algo):
    from struphy.pic.sorting_kernels import flatten_index, unflatten_index

    n1s = xp.array(xp.random.rand(10) * (nx + 1), dtype=int)
    n2s = xp.array(xp.random.rand(10) * (ny + 1), dtype=int)
    n3s = xp.array(xp.random.rand(10) * (nz + 1), dtype=int)
    for n1 in n1s:
        for n2 in n2s:
            for n3 in n3s:
                n_glob = flatten_index(int(n1), int(n2), int(n3), nx, ny, nz, algo)
                n1n, n2n, n3n = unflatten_index(n_glob, nx, ny, nz, algo)
                assert n1n == n1
                assert n2n == n2
                assert n3n == n3


@pytest.mark.parametrize("nx", [8, 70])
@pytest.mark.parametrize("ny", [16, 80])
@pytest.mark.parametrize("nz", [32, 90])
@pytest.mark.parametrize("algo", ["fortran_ordering", "c_ordering"])
def test_flattening_roundtrip(nx, ny, nz, algo):
    from struphy.pic.sorting_kernels import flatten_index, unflatten_index

    n1s = xp.array(xp.random.rand(10) * (nx + 1), dtype=int)
    n2s = xp.array(xp.random.rand(10) * (ny + 1), dtype=int)
    n3s = xp.array(xp.random.rand(10) * (nz + 1), dtype=int)
    for n1 in n1s:
        for n2 in n2s:
            for n3 in n3s:
                n_glob = flatten_index(int(n1), int(n2), int(n3), nx, ny, nz, algo)
                n1n, n2n, n3n = unflatten_index(n_glob, nx, ny, nz, algo)
                assert n1n == n1
                assert n2n == n2
                assert n3n == n3


@pytest.mark.parametrize("num_elements", [[18, 19, 20]])
@pytest.mark.parametrize("degree", [[2, 3, 4]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), ("free", "free"), None),
        (("free", "free"), None, ("free", "free")),
        (None, ("free", "free"), None),
        (None, None, ("free", "free")),
    ],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Cuboid",
            {
                "l1": 1.0,
                "r1": 2.0,
                "l2": 10.0,
                "r2": 20.0,
                "l3": 100.0,
                "r3": 200.0,
            },
        ],
    ],
)
@pytest.mark.parametrize("Np", [10000])
@pytest.mark.mpi_pic
def test_sorting(num_elements, degree, bcs, mapping, Np):
    mpi_comm = MPI.COMM_WORLD
    # assert mpi_comm.size >= 2
    rank = mpi_comm.Get_rank()

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=mpi_comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    loading_params = LoadingParameters(Np=Np, seed=1607, moments=(0.0, 0.0, 0.0, 1.0, 2.0, 3.0), spatial="uniform")
    # The marked MPI test runs with 1-4 ranks.
    # Use box counts divisible by the process-grid dimensions selected
    # for both 3 and 4 ranks.
    boxes_per_dim = (6, 6, 6)

    sorting_params = SortingParameters(boxes_per_dim=boxes_per_dim)

    particles = Particles6D(
        comm_world=mpi_comm,
        loading_params=loading_params,
        domain_decomp=domain_decomp,
        sorting_params=sorting_params,
    )

    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()

    time_start = time()
    particles.do_sort()
    time_end = time()
    time_sorting = time_end - time_start

    logger.info("Rank : {0} | Sorting time : {1:8.6f}".format(rank, time_sorting))

    box_markers = particles.markers[-2]
    assert all(box_markers[i] <= box_markers[i + 1] for i in range(len(box_markers) - 1))
    assert_boxes_match_markers(particles)


@pytest.mark.parametrize("use_numpy_argsort", [False, True])
def test_boxes_valid_after_do_sort(use_numpy_argsort):
    """After :meth:`do_sort` the box array points to the sorted rows (it is rebuilt after the rows are reordered)."""
    particles = Particles6D(
        comm_world=MPI.COMM_WORLD,
        loading_params=LoadingParameters(Np=3000, seed=1234),
        sorting_params=SortingParameters(boxes_per_dim=(12, 12, 12)),
    )
    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()
    rows_before = np.array(particles.markers, copy=True)

    particles.do_sort(use_numpy_argsort=use_numpy_argsort)

    # the rows were really reordered, so a box array from before the sort would be wrong
    assert not np.array_equal(particles.markers, rows_before)
    assert_boxes_match_markers(particles)


@pytest.mark.parametrize("bc", ["periodic", "remove"])
@pytest.mark.mpi
@pytest.mark.mpi_pic
def test_mpi_sort_markers_on_rank_boundary(bc):
    """Markers exactly on a process boundary (or on eta = 0, 1) must be kept and sent to exactly one process."""
    mpi_comm = MPI.COMM_WORLD

    particles = Particles6D(
        comm_world=mpi_comm,
        loading_params=LoadingParameters(Np=1000, seed=1234),
        boundary_params=BoundaryParameters(bc=(bc, bc, bc)),
    )
    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()

    # one tagged marker (tag in v1) for each process boundary in eta1, eta2 and eta3
    dom = particles.domain_array
    boundaries = [sorted(set(dom[:, 3 * n].tolist()) | set(dom[:, 3 * n + 1].tolist())) for n in range(3)]
    special = [(e, 0.3, 0.7) for e in boundaries[0]]
    special += [(0.3, e, 0.7) for e in boundaries[1]]
    special += [(0.3, 0.7, e) for e in boundaries[2]]
    tags = 1000.0 + xp.arange(len(special))

    if mpi_comm.Get_rank() == 0:
        holes = xp.nonzero(particles.holes)[0][: len(special)]
        particles.markers[:, holes] = 0.0
        particles.markers[:3, holes] = xp.array(special).T
        particles.markers[3, holes] = tags
        # markers written into holes by hand: count the columns that can hold markers again
        particles.count_markers_used()
        particles.update_holes()

    n_before = mpi_comm.allreduce(particles.n_mks_loc)
    particles.mpi_sort_markers(do_test=True)
    n_after = mpi_comm.allreduce(particles.n_mks_loc)
    assert n_after == n_before

    v1 = particles.markers[3, particles.valid_mks]
    for tag, eta in zip(tags, special):
        n_found = mpi_comm.allreduce(int(xp.count_nonzero(v1 == tag)))
        assert n_found == 1, f"marker at {eta} found on {n_found} processes"


def check_sorting_boxes_on_cupy():
    """Create ``Particles6D`` with sorting boxes on the CuPy backend, sort the same markers as on NumPy and compare.

    Checks the setup (``SortingBoxes.neighbours``, also for SPH particles) and one :meth:`do_sort`: the sorted markers,
    the box arrays (which must point to the sorted rows) and that the markers cross between device and host once in
    each direction. Box sorting has no CUDA port yet (#694); on CuPy the box kernels run on the host. Serial (no MPI
    communicator); the multi-rank version is ``check_sorting_boxes_device`` in ``test_sorting_device.py``. Runs with
    the real CuPy on a GPU and with cunumpy's fake CuPy (``CUNUMPY_FAKE_CUPY=1``).
    """

    def make_particles():
        return Particles6D(
            loading_params=LoadingParameters(Np=3000, seed=1234),
            sorting_params=SortingParameters(boxes_per_dim=(4, 3, 2)),
        )

    with xp.use_backend("numpy"):
        ref = make_particles()
        ref.draw_markers(sort=False)
        markers_in = np.array(ref.markers, copy=True)
        ref.do_sort()
        assert_boxes_match_markers(ref)

    with xp.use_backend("cupy"):
        particles = make_particles()
        boxes = particles.sorting_boxes
        assert xp.get_array_backend(boxes.neighbours) == "cupy"
        assert xp.get_array_backend(boxes.boxes) == "cupy"
        # the only box array with content before the first sort
        assert np.array_equal(xp.to_numpy(boxes.neighbours), ref.sorting_boxes.neighbours)

        # the same markers as on NumPy, then sort
        particles.markers[:] = xp.to_cunumpy(markers_in)
        particles.update_holes()
        with xp.profiling.count_transfers() as transfers:
            particles.do_sort()
        assert np.array_equal(xp.to_numpy(particles.markers), ref.markers)
        for name in ("_boxes", "_next_index", "_cumul_next_index"):
            assert np.array_equal(xp.to_numpy(getattr(boxes, name)), getattr(ref.sorting_boxes, name)), name
        assert_boxes_match_markers(particles)

    # one host round trip of the markers: assign the boxes, sort and refill the box array in one kernel call
    markers_bytes = ref.markers.nbytes
    assert sum(e.kind == "to_host" and e.nbytes == markers_bytes for e in transfers.events) == 1, transfers.report()
    assert sum(e.kind == "to_device" and e.nbytes == markers_bytes for e in transfers.events) == 1, transfers.report()

    # SPH particles also set up the boundary boxes and the neighbouring processes (on the host copy of domain_array)
    def make_sph():
        return ParticlesSPH(
            loading_params=LoadingParameters(Np=3000, seed=1234),
            sorting_params=SortingParameters(boxes_per_dim=(4, 3, 2)),
        )

    with xp.use_backend("numpy"):
        ref_sph = make_sph()
    with xp.use_backend("cupy"):
        sph = make_sph()
    assert sph.sorting_boxes.communicate
    assert sph.sorting_boxes.is_domain_boundary == ref_sph.sorting_boxes.is_domain_boundary
    assert all(type(v) is bool for v in sph.sorting_boxes.is_domain_boundary.values())
    assert sph.sorting_boxes._bnd_boxes_x_m == ref_sph.sorting_boxes._bnd_boxes_x_m
    assert np.array_equal(xp.to_numpy(sph.sorting_boxes.neighbours), ref_sph.sorting_boxes.neighbours)


@pytest.mark.mpi_skip
@pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")
def test_sorting_boxes_on_cupy():
    """Sorting boxes on the CuPy backend match the NumPy ones (see :func:`check_sorting_boxes_on_cupy`).

    Serial; skipped under ``mpirun``, where every rank would run the same check on the same GPU.
    """
    check_sorting_boxes_on_cupy()


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
def test_sorting_boxes_on_cupy_fake_cupy():
    """Without a GPU: :func:`check_sorting_boxes_on_cupy` with cunumpy's fake CuPy, which rejects host/device mixing.

    Runs in a serial subprocess because the fake CuPy must be installed before cunumpy is imported; under ``mpirun``
    only rank 0 starts it.
    """
    run_fake_cupy_child(
        "from struphy.pic.tests.test_sorting import check_sorting_boxes_on_cupy; check_sorting_boxes_on_cupy()"
    )


if __name__ == "__main__":
    test_flattening_roundtrip(8, 8, 8, "c_ordering")
    # test_sorting(
    #     [8, 9, 10],
    #     [2, 3, 4],
    #     [False, True, False],
    #     [
    #         "Cuboid",
    #         {
    #             "l1": 1.0,
    #             "r1": 2.0,
    #             "l2": 10.0,
    #             "r2": 20.0,
    #             "l3": 100.0,
    #             "r3": 200.0,
    #         },
    #     ],
    #     1000000,
    # )
