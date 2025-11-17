import pytest

from struphy.utils.pyccel import Pyccelkernel

def pc_lin_mhd_6d_step_ph_full(Nel, p, spl_kind, mapping, Np, verbose=False):
    from time import time

    import cunumpy as xp
    from psydac.ddm.mpi import MockComm
    from psydac.ddm.mpi import mpi as MPI

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays
    from struphy.geometry import domains
    from struphy.pic.accumulation import accum_kernels
    from struphy.pic.accumulation.particles_to_grid import Accumulator
    from struphy.pic.particles import Particles6D
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    if isinstance(MPI.COMM_WORLD, MockComm):
        mpi_comm = None
        rank = 0
        mpi_size = 1
    else:
        mpi_comm = MPI.COMM_WORLD
        # assert mpi_comm.size >= 2
        rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()

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

    mass_ops = WeightedMassOperators(derham, domain)

    if rank == 0:
        print(derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    loading_params = LoadingParameters(Np=Np, seed=1607, moments=(0.0, 0.0, 0.0, 1.0, 2.0, 3.0), spatial="uniform")

    particles = Particles6D(
        comm_world=mpi_comm,
        loading_params=loading_params,
        domain=domain,
        domain_decomp=domain_decomp,
    )

    particles.draw_markers()

    # set random weights on each process
    particles.markers[
        ~particles.holes,
        6,
    ] = xp.random.rand(particles.n_mks_loc)

    # gather all particles for legacy kernel
    if mpi_comm is None:
        marker_shapes = xp.array([particles.markers.shape[0]])
    else:
        marker_shapes = xp.zeros(mpi_size, dtype=int)
        mpi_comm.Allgather(xp.array([particles.markers.shape[0]]), marker_shapes)
    print(rank, marker_shapes)

    particles_leg = xp.zeros(
        (sum(marker_shapes), particles.markers.shape[1]),
        dtype=float,
    )

    if rank == 0:
        particles_leg[: marker_shapes[0], :] = particles.markers

        cumulative_lengths = marker_shapes[0]

        for i in range(1, mpi_size):
            arr_recv = xp.zeros(
                (marker_shapes[i], particles.markers.shape[1]),
                dtype=float,
            )
            mpi_comm.Recv(arr_recv, source=i)
            particles_leg[cumulative_lengths : cumulative_lengths + marker_shapes[i]] = arr_recv

            cumulative_lengths += marker_shapes[i]
    else:
        mpi_comm.Send(particles.markers, dest=0)

    if mpi_comm is not None:
        mpi_comm.Bcast(particles_leg, root=0)

    # sort new particles
    if particles.mpi_comm:
        particles.mpi_sort_markers()

    # =========================
    # ====== Legacy Part ======
    # =========================

    spaces_FEM_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vec = [0, 0, 0]

    for a in range(3):
        Ni = SPACES.Nbase_1form[a]
        vec[a] = xp.zeros((Ni[0], Ni[1], Ni[2], 3), dtype=float)

        for b in range(3):
            mat[a][b] = xp.zeros(
                (
                    Ni[0],
                    Ni[1],
                    Ni[2],
                    2 * SPACES.p[0] + 1,
                    2 * SPACES.p[1] + 1,
                    2 * SPACES.p[2] + 1,
                    3,
                    3,
                ),
                dtype=float,
            )

    basis_u = 1

    # =========================
    # ======== New Part =======
    # =========================
    ACC = Accumulator(
        particles,
        "Hcurl",
        Pyccelkernel(accum_kernels.pc_lin_mhd_6d_full),
        mass_ops,
        domain.args_domain,
        add_vector=True,
        symmetry="pressure",
    )

    start_time = time()
    ACC(
        1.0,
    )

    end_time = time()
    tot_time = xp.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f"Step ph New took {tot_time} seconds.")


if __name__ == "__main__":
    # test_accumulation(
    #     [8, 9, 10],
    #     [2, 3, 4],
    #     [False, False, True],
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
    # )
    pass