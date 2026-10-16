import logging

import pytest

pytestmark = pytest.mark.mpi_pic

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), ("free", "free"), None),
        (("free", "free"), None, ("free", "free")),
        (None, ("free", "free"), ("free", "free")),
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
        [
            "ShafranovDshapedCylinder",
            {
                "R0": 4.0,
                "Lz": 5.0,
                "delta_x": 0.06,
                "delta_y": 0.07,
                "delta_gs": 0.08,
                "epsilon_gs": 9.0,
                "kappa_gs": 10.0,
            },
        ],
    ],
)
def test_draw(num_elements, degree, bcs, mapping, ppc=10):
    """Asserts whether all particles are on the correct process after `particles.mpi_sort_markers()`."""

    import cunumpy as xp
    from maybempi import MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = 1234

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info("")
        logger.info("Domain decomposition according to : ")
        logger.info(derham.domain_array)

    # create particles
    loading_params = LoadingParameters(
        ppc=ppc,
        seed=seed,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        spatial="uniform",
    )

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
        domain=domain,
    )

    particles.draw_markers()

    # test weights
    particles.initialize_weights()
    _w0 = particles.weights
    logger.info("Test weights:")
    logger.info(f"rank {rank}: {_w0.shape} {xp.min(_w0)} {xp.max(_w0)}")

    comm.Barrier()
    logger.info("Number of particles w/wo holes on each process before sorting : ")
    logger.info(f"Rank {rank} : {particles.n_mks_loc} {particles.n_markers_max}")

    # sort particles according to domain decomposition
    comm.Barrier()
    particles.mpi_sort_markers(do_test=True)

    comm.Barrier()
    logger.info("Number of particles w/wo holes on each process after sorting : ")
    logger.info(f"Rank {rank} : {particles.n_mks_loc} {particles.n_markers_max}")

    # are all markers in the correct domain?
    conds = xp.logical_and(
        particles.markers[:3] > derham.domain_array[rank, 0::3, None],
        particles.markers[:3] < derham.domain_array[rank, 1::3, None],
    )
    holes = particles.markers[0] == -1.0
    stay = xp.all(conds, axis=0)

    error_mks = particles.markers[:, xp.logical_and(~stay, ~holes)]

    assert error_mks.size == 0, (
        f"rank {rank} | markers not on correct process: {xp.nonzero(xp.logical_and(~stay, ~holes))} \n corresponding positions:\n {error_mks[:3]}"
    )


@pytest.mark.parametrize("loading", ["pseudo_random", "sobol_standard", "sobol_antithetic"])
def test_marker_ids(loading, Np=1000):
    """Asserts that marker IDs are unique and contiguous across processes after drawing,
    and that no marker is left at the origin (``sobol_antithetic`` with Np not divisible by 64)."""

    import cunumpy as xp
    from maybempi import MPI

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=[8, 5, 3])
    derham = Derham(grid, DerhamOptions(degree=[1, 1, 1]), comm=comm)
    domain_decomp = (derham.domain_array, derham.domain_decomposition.nprocs)

    loading_params = LoadingParameters(
        Np=Np,
        seed=1234,
        loading=loading,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
        domain=domains.Cuboid(),
    )

    particles.draw_markers(sort=False)

    valid = particles.markers[:, ~particles.holes]
    ids = valid[-1]
    n_at_origin = int(xp.sum(xp.all(valid[:3] == 0.0, axis=0)))
    if particles.mpi_size > 1:
        ids = xp.concatenate(particles.mpi_comm.allgather(ids))
        n_at_origin = particles.mpi_comm.allreduce(n_at_origin)

    assert xp.array_equal(xp.sort(ids), xp.arange(Np, dtype=float))
    assert n_at_origin == 0


@pytest.mark.parametrize("loading", ["pseudo_random", "sobol_standard", "sobol_antithetic"])
@pytest.mark.parametrize("sort", [True, False])
def test_draw_without_boxes_on_rank_domain(loading, sort, Np=1000):
    """Without sorting boxes, every marker is on the domain of its process right after drawing (#764),
    for ``sort=True`` and ``sort=False`` (which only concerns the boxes), and no marker is lost."""

    import cunumpy as xp
    from maybempi import MPI

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=[8, 5, 3])
    derham = Derham(grid, DerhamOptions(degree=[1, 1, 1]), comm=comm)
    domain_decomp = (derham.domain_array, derham.domain_decomposition.nprocs)

    loading_params = LoadingParameters(
        Np=Np,
        seed=1234,
        loading=loading,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )

    # no sorting_params: no sorting boxes
    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
        domain=domains.Cuboid(),
    )
    assert particles.sorting_boxes is None

    particles.draw_markers(sort=sort)

    positions = particles.markers_wo_holes[:3]
    on_rank = xp.all(particles._is_on_domain_of_rank(positions, particles.mpi_rank), axis=0)
    n_off_rank = int(xp.sum(~on_rank))
    n_markers = positions.shape[1]
    if particles.mpi_size > 1:
        n_markers = particles.mpi_comm.allreduce(n_markers)

    assert n_off_rank == 0, f"rank {particles.mpi_rank}: {n_off_rank} markers off the process domain after drawing"
    assert n_markers == Np


@pytest.mark.parametrize("loading", ["pseudo_random", "sobol_standard"])
def test_draw_disc_on_own_process(loading, ppc=50):
    """Markers drawn uniformly on the disc are on their own process after `draw_markers` (without sorting boxes)."""

    import numpy as np
    from maybempi import MPI

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=4.0)
    derham = Derham(TensorProductGrid(num_elements=(8, 9, 4)), DerhamOptions(degree=(1, 1, 1)), comm=comm)
    domain_decomp = (derham.domain_array, derham.domain_decomposition.nprocs)

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=LoadingParameters(ppc=ppc, loading=loading, seed=1234, spatial="disc"),
        domain=domain,
    )
    particles.draw_markers()

    # all markers are on their own process and none is lost
    # (reduce before asserting, such that a failure on one process does not deadlock the others)
    positions = particles.positions
    on_own_process = comm.allreduce(bool(np.all(particles._is_on_domain_of_rank(positions, rank))), op=MPI.LAND)
    assert on_own_process
    assert comm.allreduce(positions.shape[1]) == particles.Np

    # uniform on the disc: eta1 has the density 2 eta1 with mean 2/3 and variance 1/18
    mean = comm.allreduce(np.sum(positions[0])) / particles.Np
    assert abs(mean - 2.0 / 3.0) < 5.0 * np.sqrt(1.0 / 18.0 / particles.Np)


if __name__ == "__main__":
    # test_draw([8, 9, 10], [2, 3, 4], [False, False, True], ['Cuboid', {
    #     'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
    test_draw(
        [8, 9, 10],
        [2, 3, 4],
        [False, False, True],
        [
            "Cuboid",
            {
                "l1": 0.0,
                "r1": 1.0,
                "l2": 0.0,
                "r2": 1.0,
                "l3": 0.0,
                "r3": 1.0,
            },
        ],
    )
