import logging

import pytest

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

    import numpy as np
    from cunumpy import to_numpy
    from feectools.ddm.mpi import mpi as MPI

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
    logger.info(f"rank {rank}: {_w0.shape} {np.min(_w0)} {np.max(_w0)}")

    comm.Barrier()
    logger.info("Number of particles w/wo holes on each process before sorting : ")
    logger.info(f"Rank {rank} : {particles.n_mks_loc} {particles.markers.shape[0]}")

    # sort particles according to domain decomposition
    comm.Barrier()
    particles.mpi_sort_markers(do_test=True)

    comm.Barrier()
    logger.info("Number of particles w/wo holes on each process after sorting : ")
    logger.info(f"Rank {rank} : {particles.n_mks_loc} {particles.markers.shape[0]}")

    # are all markers in the correct domain?
    # particles.markers is always host (NumPy); derham.domain_array may be a
    # device array under CuPy, so bring it to the host for this comparison.
    domain_array_host = to_numpy(derham.domain_array)
    conds = np.logical_and(
        particles.markers[:, :3] > domain_array_host[rank, 0::3],
        particles.markers[:, :3] < domain_array_host[rank, 1::3],
    )
    holes = particles.markers[:, 0] == -1.0
    stay = np.all(conds, axis=1)

    error_mks = particles.markers[np.logical_and(~stay, ~holes)]

    assert error_mks.size == 0, (
        f"rank {rank} | markers not on correct process: {np.nonzero(np.logical_and(~stay, ~holes))} \n corresponding positions:\n {error_mks[:, :3]}"
    )


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
