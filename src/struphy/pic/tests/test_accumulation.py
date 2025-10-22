import pytest

from struphy.utils.pyccel import Pyccelkernel


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
def test_accumulation(Nel, p, spl_kind, mapping, Np=40, verbose=False):
    """
    A test to compare the old accumulation routine of step1 and step3 of cc_lin_mhd_6d with the old way (files stored in
    ../test_pic_legacy_files) and the new way using the Accumulator object (ghost_region_sender, particle_to_mat_kernels).

    The two accumulation matrices are computed with the same random magnetic field produced by
    feec.utilities.create_equal_random_arrays and compared against each other at the bottom using
    feec.utilities.compare_arrays().

    The times for both legacy and the new way are printed if verbose == True. This comparison only makes sense if the
    ..test_pic_legacy_files/ are also all compiled.
    """
    from psydac.ddm.mpi import mpi as MPI

    rank = MPI.COMM_WORLD.Get_rank()

    pc_lin_mhd_6d_step_ph_full(Nel, p, spl_kind, mapping, Np, verbose)
    if verbose and rank == 0:
        print("\nTest for Step ph passed\n")


def pc_lin_mhd_6d_step_ph_full(Nel, p, spl_kind, mapping, Np, verbose=False):
    from time import time

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
    from struphy.pic.tests.test_pic_legacy_files.accumulation_kernels_3d import kernel_step_ph_full
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters
    from struphy.utils.arrays import xp

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

    start_time = time()
    kernel_step_ph_full(
        particles_leg,
        SPACES.T[0],
        SPACES.T[1],
        SPACES.T[2],
        xp.array(SPACES.p),
        xp.array(Nel),
        xp.array(SPACES.NbaseN),
        xp.array(SPACES.NbaseD),
        particles_leg.shape[0],
        domain.kind_map,
        domain.params_numpy,
        domain.T[0],
        domain.T[1],
        domain.T[2],
        xp.array(domain.p),
        xp.array(
            domain.Nel,
        ),
        xp.array(domain.NbaseN),
        domain.cx,
        domain.cy,
        domain.cz,
        mat[0][0],
        mat[0][1],
        mat[0][2],
        mat[1][1],
        mat[1][2],
        mat[2][2],
        vec[0],
        vec[1],
        vec[2],
        basis_u,
    )

    end_time = time()
    tot_time = xp.round(end_time - start_time, 3)

    mat[0][0] /= Np
    mat[0][1] /= Np
    mat[0][2] /= Np
    mat[1][1] /= Np
    mat[1][2] /= Np
    mat[2][2] /= Np

    vec[0] /= Np
    vec[1] /= Np
    vec[2] /= Np

    if rank == 0 and verbose:
        print(f"Step ph Legacy took {tot_time} seconds.")

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
    ACC(1.0, 1.0, 0.0)

    end_time = time()
    tot_time = xp.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f"Step ph New took {tot_time} seconds.")

    # =========================
    # ======== Compare ========
    # =========================

    atol = 1e-10

    # mat_temp11 = [[mat[0][0][:,:,:,:,:,:,0,0], mat[0][1][:,:,:,:,:,:,0,0], mat[0][2][:,:,:,:,:,:,0,0]],
    #              [ mat[0][1][:,:,:,:,:,:,0,0].transpose(), mat[1][1][:,:,:,:,:,:,0,0], mat[1][2][:,:,:,:,:,:,0,0]],
    #              [ mat[0][2][:,:,:,:,:,:,0,0].transpose(), mat[1][2][:,:,:,:,:,:,0,0].transpose(), mat[2][2][:,:,:,:,:,:,0,0]]]
    # mat_temp12 = [[mat[0][0][:,:,:,:,:,:,0,1], mat[0][1][:,:,:,:,:,:,0,1], mat[0][2][:,:,:,:,:,:,0,1]],
    #              [ mat[0][1][:,:,:,:,:,:,0,1].transpose(), mat[1][1][:,:,:,:,:,:,0,1], mat[1][2][:,:,:,:,:,:,0,1]],
    #              [ mat[0][2][:,:,:,:,:,:,0,1].transpose(), mat[1][2][:,:,:,:,:,:,0,1].transpose(), mat[2][2][:,:,:,:,:,:,0,1]]]
    # mat_temp13 = [[mat[0][0][:,:,:,:,:,:,0,2], mat[0][1][:,:,:,:,:,:,0,2], mat[0][2][:,:,:,:,:,:,0,2]],
    #              [ mat[0][1][:,:,:,:,:,:,0,2].transpose(), mat[1][1][:,:,:,:,:,:,0,2], mat[1][2][:,:,:,:,:,:,0,2]],
    #              [ mat[0][2][:,:,:,:,:,:,0,2].transpose(), mat[1][2][:,:,:,:,:,:,0,2].transpose(), mat[2][2][:,:,:,:,:,:,0,2]]]
    # mat_temp22 = [[mat[0][0][:,:,:,:,:,:,1,1], mat[0][1][:,:,:,:,:,:,1,1], mat[0][2][:,:,:,:,:,:,1,1]],
    #              [ mat[0][1][:,:,:,:,:,:,1,1].transpose(), mat[1][1][:,:,:,:,:,:,1,1], mat[1][2][:,:,:,:,:,:,1,1]],
    #              [ mat[0][2][:,:,:,:,:,:,1,1].transpose(), mat[1][2][:,:,:,:,:,:,1,1].transpose(), mat[2][2][:,:,:,:,:,:,1,1]]]
    # mat_temp23 = [[mat[0][0][:,:,:,:,:,:,1,2], mat[0][1][:,:,:,:,:,:,1,2], mat[0][2][:,:,:,:,:,:,1,2]],
    #              [ mat[0][1][:,:,:,:,:,:,1,2].transpose(), mat[1][1][:,:,:,:,:,:,1,2], mat[1][2][:,:,:,:,:,:,1,2]],
    #              [ mat[0][2][:,:,:,:,:,:,1,2].transpose(), mat[1][2][:,:,:,:,:,:,1,2].transpose(), mat[2][2][:,:,:,:,:,:,1,2]]]
    # mat_temp33 = [[mat[0][0][:,:,:,:,:,:,2,2], mat[0][1][:,:,:,:,:,:,2,2], mat[0][2][:,:,:,:,:,:,2,2]],
    #              [ mat[0][1][:,:,:,:,:,:,2,2].transpose(), mat[1][1][:,:,:,:,:,:,2,2], mat[1][2][:,:,:,:,:,:,2,2]],
    #              [ mat[0][2][:,:,:,:,:,:,2,2].transpose(), mat[1][2][:,:,:,:,:,:,2,2].transpose(), mat[2][2][:,:,:,:,:,:,2,2]]]
    vec_temp1 = [vec[0][:, :, :, 0], vec[1][:, :, :, 0], vec[2][:, :, :, 0]]
    vec_temp2 = [vec[0][:, :, :, 1], vec[1][:, :, :, 1], vec[2][:, :, :, 1]]
    vec_temp3 = [vec[0][:, :, :, 2], vec[1][:, :, :, 2], vec[2][:, :, :, 2]]

    compare_arrays(
        ACC.operators[0].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_11 passed test")
    compare_arrays(
        ACC.operators[0].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_11 passed test")
    compare_arrays(
        ACC.operators[0].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_11 passed test")
    compare_arrays(
        ACC.operators[0].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_11 passed test")
    compare_arrays(
        ACC.operators[0].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_11 passed test")
    compare_arrays(
        ACC.operators[0].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 0, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_11 passed test")

    compare_arrays(
        ACC.operators[1].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_12 passed test")
    compare_arrays(
        ACC.operators[1].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_12 passed test")
    compare_arrays(
        ACC.operators[1].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_12 passed test")
    compare_arrays(
        ACC.operators[1].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_12 passed test")
    compare_arrays(
        ACC.operators[1].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_12 passed test")
    compare_arrays(
        ACC.operators[1].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 0, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_12 passed test")

    compare_arrays(
        ACC.operators[2].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_13 passed test")
    compare_arrays(
        ACC.operators[2].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_13 passed test")
    compare_arrays(
        ACC.operators[2].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_13 passed test")
    compare_arrays(
        ACC.operators[2].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_13 passed test")
    compare_arrays(
        ACC.operators[2].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_13 passed test")
    compare_arrays(
        ACC.operators[2].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 0, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_13 passed test")

    compare_arrays(
        ACC.operators[3].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_22 passed test")
    compare_arrays(
        ACC.operators[3].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_22 passed test")
    compare_arrays(
        ACC.operators[3].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_22 passed test")
    compare_arrays(
        ACC.operators[3].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_22 passed test")
    compare_arrays(
        ACC.operators[3].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_22 passed test")
    compare_arrays(
        ACC.operators[3].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 1, 1],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_22 passed test")

    compare_arrays(
        ACC.operators[4].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_23 passed test")
    compare_arrays(
        ACC.operators[4].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_23 passed test")
    compare_arrays(
        ACC.operators[4].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_23 passed test")
    compare_arrays(
        ACC.operators[4].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_23 passed test")
    compare_arrays(
        ACC.operators[4].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_23 passed test")
    compare_arrays(
        ACC.operators[4].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 1, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_23 passed test")

    compare_arrays(
        ACC.operators[5].matrix.blocks[0][0],
        mat[0][0][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat11_33 passed test")
    compare_arrays(
        ACC.operators[5].matrix.blocks[0][1],
        mat[0][1][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat12_33 passed test")
    compare_arrays(
        ACC.operators[5].matrix.blocks[0][2],
        mat[0][2][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat13_33 passed test")
    compare_arrays(
        ACC.operators[5].matrix.blocks[1][1],
        mat[1][1][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat22_33 passed test")
    compare_arrays(
        ACC.operators[5].matrix.blocks[1][2],
        mat[1][2][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat23_33 passed test")
    compare_arrays(
        ACC.operators[5].matrix.blocks[2][2],
        mat[2][2][:, :, :, :, :, :, 2, 2],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("mat33_33 passed test")

    compare_arrays(
        ACC.vectors[0].blocks[0],
        vec[0][:, :, :, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("vec1_1 passed test")
    compare_arrays(
        ACC.vectors[0].blocks[1],
        vec[1][:, :, :, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("vec2_1 passed test")
    compare_arrays(
        ACC.vectors[0].blocks[2],
        vec[2][:, :, :, 0],
        rank,
        atol=atol,
        verbose=verbose,
    )
    if verbose:
        print("vec3_1 passed test")
    # compare_arrays(ACC.operators[0].matrix, mat_temp11, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_11 passed test')
    # compare_arrays(ACC.operators[1].matrix, mat_temp12, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_12 passed test')
    # compare_arrays(ACC.operators[2].matrix, mat_temp13, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_13 passed test')
    # compare_arrays(ACC.operators[3].matrix, mat_temp22, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_22 passed test')
    # compare_arrays(ACC.operators[4].matrix, mat_temp23, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_23 passed test')
    # compare_arrays(ACC.operators[5].matrix, mat_temp33, rank, atol=atol, verbose=verbose)
    # if verbose:
    #     print('full block matrix_33 passed test')
    compare_arrays(ACC.vectors[0], vec_temp1, rank, atol=atol, verbose=verbose)
    if verbose:
        print("full block vector_1 passed test")
    compare_arrays(ACC.vectors[1], vec_temp2, rank, atol=atol, verbose=verbose)
    if verbose:
        print("full block vector_2 passed test")
    compare_arrays(ACC.vectors[2], vec_temp3, rank, atol=atol, verbose=verbose)
    if verbose:
        print("full block vector_3 passed test")


if __name__ == "__main__":
    test_accumulation(
        [8, 9, 10],
        [2, 3, 4],
        [False, False, True],
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
    )
