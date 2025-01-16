import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[2, 3, 4]])
@pytest.mark.parametrize(
    "spl_kind", [[False, False, True], [False, True, True], [True, False, True], [True, True, True]]
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
@pytest.mark.parametrize("Nclones", [1, 2])
def test_accum_poisson(Nel, p, spl_kind, mapping, Nclones, Np=1000):
    r"""DRAFT: test the accumulation of the rhs (H1-space) in Poisson's equation .

    Tests:

        * Whether all weights are initialized as \sqrt(g) = const. (Cuboid mappings).
        * Whether the sum oaver all MC integrals is 1.
    """

    import copy

    import numpy as np
    from mpi4py import MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.setup import setup_domain_cloning
    from struphy.pic.accumulation import accum_kernels
    from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
    from struphy.pic.particles import Particles6D

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    # domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    params = {
        "grid": {"Nclones": Nclones, "Nel": Nel},
        "kinetic": {"test_particles": {"markers": {"Np": Np, "ppc": Np / np.prod(Nel)}}},
    }
    params, inter_comm, sub_comm = setup_domain_cloning(mpi_comm, copy.deepcopy(params), Nclones)

    # DeRham object
    derham = Derham(
        Nel,
        p,
        spl_kind,
        comm=sub_comm,
        inter_comm=inter_comm,
    )

    if mpi_rank == 0:
        print("Domain decomposition according to", derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    loading_params = {
        "seed": 1607,
        "moments": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "spatial": "uniform",
    }

    particles = Particles6D(
        "test_particles",
        Np=Np,
        bc=["periodic"] * 3,
        loading="pseudo_random",
        loading_params=loading_params,
        comm=sub_comm,
        inter_comm=inter_comm,
        domain=domain,
        domain_array=derham.domain_array,
    )

    particles.draw_markers()
    particles.mpi_sort_markers()
    particles.initialize_weights()

    _vdim = particles.vdim
    _w0 = particles.weights

    print("Test weights:")
    print(f"rank {mpi_rank}:", _w0.shape, np.min(_w0), np.max(_w0))

    _sqrtg = domain.jacobian_det(0.5, 0.5, 0.5)

    assert np.isclose(np.min(_w0), _sqrtg)
    assert np.isclose(np.max(_w0), _sqrtg)

    # mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    # instance of the accumulator
    acc = AccumulatorVector(
        particles,
        "H1",
        accum_kernels.charge_density_0form,
        mass_ops,
        domain.args_domain,
    )

    acc(particles.vdim)

    # sum all MC integrals
    _sum_within_clone = np.empty(1, dtype=float)
    _sum_within_clone[0] = np.sum(acc.vectors[0].toarray())
    sub_comm.Allreduce(MPI.IN_PLACE, _sum_within_clone, op=MPI.SUM)

    print(f"rank {mpi_rank}: {_sum_within_clone = }, {_sqrtg = }")

    # Check within clone
    assert np.isclose(_sum_within_clone, _sqrtg)

    # Check for all clones
    _sum_between_clones = np.empty(1, dtype=float)
    _sum_between_clones[0] = np.sum(acc.vectors[0].toarray())
    mpi_comm.Allreduce(MPI.IN_PLACE, _sum_between_clones, op=MPI.SUM)
    inter_comm.Allreduce(MPI.IN_PLACE, _sqrtg, op=MPI.SUM)

    print(f"rank {mpi_rank}: {_sum_between_clones = }, {_sqrtg = }")

    # Check within clone
    assert np.isclose(_sum_between_clones, _sqrtg)


if __name__ == "__main__":
    for Nclones in [1, 2]:
        test_accum_poisson(
            [8, 9, 10],
            [2, 3, 4],
            [False, False, True],
            [
                "Cuboid",
                {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0},
            ],
            Nclones=Nclones,
            Np=1000,
        )
