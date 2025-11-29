import pytest

from struphy.utils.pyccel import Pyccelkernel


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[2, 3, 4]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, False, True], [False, True, True], [True, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
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
        [
            "Cuboid",
            {
                "l1": 0.0,
                "r1": 2.0,
                "l2": 0.0,
                "r2": 3.0,
                "l3": 0.0,
                "r3": 4.0,
            },
        ],
    ],
)
@pytest.mark.parametrize("num_clones", [1, 2])
def test_accum_poisson(Nel, p, spl_kind, mapping, num_clones, Np=1000):
    r"""DRAFT: test the accumulation of the rhs (H1-space) in Poisson's equation .

    Tests:

        * Whether all weights are initialized as \sqrt(g) = const. (Cuboid mappings).
        * Whether the sum oaver all MC integrals is 1.
    """

    import copy

    import cunumpy as xp
    from psydac.ddm.mpi import MockComm
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.pic.accumulation import accum_kernels
    from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
    from struphy.pic.particles import Particles6D
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters
    from struphy.utils.clone_config import CloneConfig

    if isinstance(MPI.COMM_WORLD, MockComm):
        mpi_comm = None
        mpi_rank = 0
    else:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()

    # domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    params = {
        "grid": {"Nel": Nel},
        "kinetic": {"test_particles": {"markers": {"Np": Np, "ppc": Np / xp.prod(Nel)}}},
    }
    if mpi_comm is None:
        clone_config = None

        derham = Derham(
            Nel,
            p,
            spl_kind,
            comm=None,
        )
    else:
        if mpi_comm.Get_size() % num_clones == 0:
            clone_config = CloneConfig(comm=mpi_comm, params=params, num_clones=num_clones)
        else:
            return

        derham = Derham(
            Nel,
            p,
            spl_kind,
            comm=clone_config.sub_comm,
        )

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if mpi_rank == 0:
        print("Domain decomposition according to", derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    loading_params = LoadingParameters(
        Np=Np,
        seed=1607,
        moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        spatial="uniform",
    )

    particles = Particles6D(
        comm_world=mpi_comm,
        clone_config=clone_config,
        loading_params=loading_params,
        domain=domain,
        domain_decomp=domain_decomp,
    )

    particles.draw_markers()
    if mpi_comm is not None:
        particles.mpi_sort_markers()
    particles.initialize_weights()

    _vdim = particles.vdim
    _w0 = particles.weights

    print("Test weights:")
    print(f"rank {mpi_rank}:", _w0.shape, xp.min(_w0), xp.max(_w0))

    _sqrtg = domain.jacobian_det(0.5, 0.5, 0.5)

    assert xp.isclose(xp.min(_w0), _sqrtg)
    assert xp.isclose(xp.max(_w0), _sqrtg)

    # mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    # instance of the accumulator
    acc = AccumulatorVector(
        particles,
        "H1",
        Pyccelkernel(accum_kernels.charge_density_0form),
        mass_ops,
        domain.args_domain,
    )

    acc()

    # sum all MC integrals
    _sum_within_clone = xp.empty(1, dtype=float)
    _sum_within_clone[0] = xp.sum(acc.vectors[0].toarray())
    if clone_config is not None:
        clone_config.sub_comm.Allreduce(MPI.IN_PLACE, _sum_within_clone, op=MPI.SUM)

    print(f"rank {mpi_rank}: {_sum_within_clone =}, {_sqrtg =}")

    # Check within clone
    assert xp.isclose(_sum_within_clone, _sqrtg)

    # Check for all clones
    _sum_between_clones = xp.empty(1, dtype=float)
    _sum_between_clones[0] = xp.sum(acc.vectors[0].toarray())

    if mpi_comm is not None:
        mpi_comm.Allreduce(MPI.IN_PLACE, _sum_between_clones, op=MPI.SUM)
        clone_config.inter_comm.Allreduce(MPI.IN_PLACE, _sqrtg, op=MPI.SUM)

    print(f"rank {mpi_rank}: {_sum_between_clones =}, {_sqrtg =}")

    # Check within clone
    assert xp.isclose(_sum_between_clones, _sqrtg)


if __name__ == "__main__":
    for num_clones in [1, 2]:
        test_accum_poisson(
            [8, 9, 10],
            [2, 3, 4],
            [False, False, True],
            [
                "Cuboid",
                {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0},
            ],
            num_clones=num_clones,
            Np=1000,
        )
