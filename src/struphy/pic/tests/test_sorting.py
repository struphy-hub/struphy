from time import time

import cunumpy as xp
import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import Particles6D
from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters


@pytest.mark.parametrize("nx", [8, 70])
@pytest.mark.parametrize("ny", [16, 80])
@pytest.mark.parametrize("nz", [32, 90])
@pytest.mark.parametrize("algo", ["fortran_ordering", "c_ordering"])
def test_flattening(nx, ny, nz, algo):
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
def test_flattening(nx, ny, nz, algo):
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
def test_flattening(nx, ny, nz, algo):
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


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[2, 3, 4]])
@pytest.mark.parametrize(
    "spl_kind", [[False, False, True], [False, True, False], [True, False, True], [True, True, False]]
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
def test_sorting(Nel, p, spl_kind, mapping, Np, verbose=False):
    mpi_comm = MPI.COMM_WORLD
    # assert mpi_comm.size >= 2
    rank = mpi_comm.Get_rank()

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    loading_params = LoadingParameters(Np=Np, seed=1607, moments=(0.0, 0.0, 0.0, 1.0, 2.0, 3.0), spatial="uniform")
    boxes_per_dim = (3, 3, 6)

    particles = Particles6D(
        comm_world=mpi_comm,
        loading_params=loading_params,
        domain_decomp=domain_decomp,
        boxes_per_dim=boxes_per_dim,
    )

    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()

    time_start = time()
    particles.do_sort()
    time_end = time()
    time_sorting = time_end - time_start

    print("Rank : {0} | Sorting time : {1:8.6f}".format(rank, time_sorting))

    box_markers = particles.markers[:, -2]
    assert all(box_markers[i] <= box_markers[i + 1] for i in range(len(box_markers) - 1))


if __name__ == "__main__":
    test_flattening(8, 8, 8, "c_orderwding")
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
