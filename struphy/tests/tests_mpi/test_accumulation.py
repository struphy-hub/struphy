import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, True], [True, True, False]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_accumulation(Nel, p, spl_kind, mapping, Np=40, verbose=False):
    """
    A test to compare the old accumulation routine of step1 and step3 of cc_lin_mhd_6d with the old way (files stored in
    ../test_pic_legacy_files) and the new way using the Accumulator object (ghost_region_sender, mat_vec_filler).

    The two accumulation matrices are computed with the same random magnetic field produced by
    psydac_api.utilities.create_equal_random_arrays and compared against each other at the bottom using
    psydac_api.utilities.compare_arrays().

    The times for both legacy and the new way are printed if verbose == True. This comparison only makes sense if the
    ..test_pic_legacy_files/ are also all compiled.
    """
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    cc_lin_mhd_6d_step_1(Nel, p, spl_kind, mapping, Np, verbose)
    if verbose and rank == 0:
        print('\nTest for Step 1 passed\n')

    cc_lin_mhd_6d_step_3(Nel, p, spl_kind, mapping, Np, verbose)
    if verbose and rank == 0:
        print('\nTest for Step 3 passed\n')

    pc_lin_mhd_6d_step_ph_full(Nel, p, spl_kind, mapping, Np, verbose)
    if verbose and rank == 0:
        print('\nTest for Step ph passed\n')


def cc_lin_mhd_6d_step_1(Nel, p, spl_kind, mapping, Np, verbose=False):
    import numpy as np
    from mpi4py import MPI
    from time import time

    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    from struphy.pic.particles import Particles6D
    from struphy.pic.particles_to_grid import Accumulator
    from struphy.tests.tests_mpi.test_pic_legacy_files.accumulation import Accumulator as Accumulator_leg

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # space of test
    space = 'Hdiv'

    # domain object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # space key: one of '1', '2' and 'v'
    space_key = derham.spaces_dict[space]

    if mpi_rank == 0:
        print('Domain decomposition according to', derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    params_markers = {'Np': Np, 'eps': .25,
                      'loading': {'type': 'pseudo_random', 'seed': 1607, 'moments': [0., 0., 0., 1., 2., 3.]}
                      }

    particles = Particles6D(
        'test_particles', **params_markers, domain_array=derham.domain_array, comm=mpi_comm)

    # set random weights on each process
    particles.markers[~particles.holes,
                      6] = np.random.rand(particles.n_mks_loc)

    # copy of markers for legacy kernel
    particles_leg = particles.markers.copy()

    # sort new particles
    particles.mpi_sort_markers()

    # create a random array for the magnetic field
    b2, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=4657, flattened=True)

    # =========================
    # ====== Legacy Part ======
    # =========================

    spaces_FEM_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2])

    spaces = Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    if space == 'H1vec':
        acc_leg = Accumulator_leg(
            spaces, domain, 0, mpi_comm, use_control=False)
    else:
        acc_leg = Accumulator_leg(spaces, domain, int(
            space_key), mpi_comm, use_control=False)

    start_time = time()
    acc_leg.accumulate_step1(particles_leg, Np, 0., b2, mpi_comm)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if mpi_rank == 0 and verbose:
        print(f'Step 1 Legacy took {tot_time} seconds.')

    # =========================
    # ======== New Part =======
    # =========================
    acc = Accumulator(derham, domain, space, 'cc_lin_mhd_6d_1',
                      add_vector=False, symmetry='asym')

    start_time = time()

    if space == 'H1vec':
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       0, 1.)
    else:
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       int(space_key), 1.)

    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if mpi_rank == 0 and verbose:
        print(f'Step 1 New took {tot_time} seconds.')

    # accumulate a 2nd time to check whether calling accumulate successively gives the same result
    if space == 'H1vec':
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       0, 1.)
    else:
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       int(space_key), 1.)

    # =========================
    # ======== Compare ========
    # =========================

    # compare blocks
    atol = 1e-10
    
    compare_arrays(acc.operators[0].matrix[0, 1], acc_leg.blocks_glo[0][1], mpi_rank, atol=atol, verbose=verbose)
    if verbose and mpi_rank == 0:
        print('mat12 passed test')
    compare_arrays(acc.operators[0].matrix[0, 2], acc_leg.blocks_glo[0][2], mpi_rank, atol=atol, verbose=verbose)
    if verbose and mpi_rank == 0:
        print('mat13 passed test')
    compare_arrays(acc.operators[0].matrix[1, 2], acc_leg.blocks_glo[1][2], mpi_rank, atol=atol, verbose=verbose)
    if verbose and mpi_rank == 0:
        print('mat23 passed test')

    # compare matrix-vector product
    x, x_psy = create_equal_random_arrays(derham.Vh_fem[space_key], seed=5624, flattened=True)
    
    r_psy = acc.operators[0].dot(x_psy)
    
    r = acc_leg.to_sparse_step1().dot(x)

    compare_arrays(r_psy, r, mpi_rank, atol=atol, verbose=verbose)
    if verbose and mpi_rank == 0:
        print('matrix-vector product passed test')


def cc_lin_mhd_6d_step_3(Nel, p, spl_kind, mapping, Np, verbose=False):
    import numpy as np
    from mpi4py import MPI
    from time import time

    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    from struphy.pic.particles import Particles6D
    from struphy.pic.particles_to_grid import Accumulator
    from struphy.tests.tests_mpi.test_pic_legacy_files.accumulation import Accumulator as Accumulator_leg

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # space of test
    space = 'Hdiv'

    # domain object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # space key: one of '1', '2' and 'v'
    space_key = derham.spaces_dict[space]

    if mpi_rank == 0:
        print(derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    params_markers = {'Np': Np, 'eps': .25,
                      'loading': {'type': 'pseudo_random', 'seed': 1607, 'moments': [0., 0., 0., 1., 2., 3.]}
                      }

    particles = Particles6D(
        'test_particles', **params_markers, domain_array=derham.domain_array, comm=mpi_comm)

    # set random weights on each process
    particles.markers[~particles.holes,
                      6] = np.random.rand(particles.n_mks_loc)

    # copy of markers for legacy kernel
    particles_leg = particles.markers.copy()

    # sort new particles
    particles.mpi_sort_markers()

    # create a random array for the magnetic field
    b2, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=1597, flattened=True)

    # =========================
    # ====== Legacy Part ======
    # =========================

    spaces_FEM_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2])

    spaces = Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    if space == 'H1vec':
        acc_leg = Accumulator_leg(
            spaces, domain, 0, mpi_comm, use_control=False)
    else:
        acc_leg = Accumulator_leg(spaces, domain, int(
            space_key), mpi_comm, use_control=False)

    start_time = time()
    acc_leg.accumulate_step3(particles_leg, Np, 0., b2, mpi_comm)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if mpi_rank == 0 and verbose:
        print(f'Step 3 Legacy took {tot_time} seconds.')

    # =========================
    # ======== New Part =======
    # =========================
    acc = Accumulator(derham, domain, space, 'cc_lin_mhd_6d_2',
                      add_vector=True, symmetry='symm')

    start_time = time()

    if space == 'H1vec':
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       0, 1., 1.)
    else:
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       int(space_key), 1., 1.)

    # accumulate a 2nd time to check whether calling accumulate successively gives the same result
    if space == 'H1vec':
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       0, 1., 1.)
    else:
        acc.accumulate(particles,
                       b2_psy[0]._data,
                       b2_psy[1]._data,
                       b2_psy[2]._data,
                       int(space_key), 1., 1.)

    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if mpi_rank == 0 and verbose:
        print(f'Step 3 New took {tot_time} seconds.')

    # =========================
    # ======== Compare ========
    # =========================

    # compare blocks
    atol = 1e-10

    compare_arrays(acc.operators[0].matrix[0, 0], acc_leg.blocks_glo[0][0], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat11 passed test')
    compare_arrays(acc.operators[0].matrix[0, 1], acc_leg.blocks_glo[0][1], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat12 passed test')
    compare_arrays(acc.operators[0].matrix[0, 2], acc_leg.blocks_glo[0][2], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat13 passed test')
    compare_arrays(acc.operators[0].matrix[1, 1], acc_leg.blocks_glo[1][1], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat22 passed test')
    compare_arrays(acc.operators[0].matrix[1, 2], acc_leg.blocks_glo[1][2], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat23 passed test')
    compare_arrays(acc.operators[0].matrix[2, 2], acc_leg.blocks_glo[2][2], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat33 passed test')
    
    compare_arrays(acc.vectors[0][0], acc_leg.vecs_glo[0], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec1 passed test')
    compare_arrays(acc.vectors[0][1], acc_leg.vecs_glo[1], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec2 passed test')
    compare_arrays(acc.vectors[0][2], acc_leg.vecs_glo[2], mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec3 passed test')

    compare_arrays(acc.vectors[0], acc_leg.vecs_glo, mpi_rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block vector passed test')

    # compare matrix-vector product
    x, x_psy = create_equal_random_arrays(derham.Vh_fem[space_key], seed=5624, flattened=True)
    
    r_psy = acc.operators[0].dot(x_psy)
    
    r = acc_leg.to_sparse_step3().dot(x)

    compare_arrays(r_psy, r, mpi_rank, atol=atol, verbose=verbose)
    if verbose and mpi_rank == 0:
        print('matrix-vector product passed test')


def pc_lin_mhd_6d_step_ph_full(Nel, p, spl_kind, mapping, Np, verbose=False):
    import numpy as np
    from mpi4py import MPI
    from time import time

    from struphy.psydac_api.utilities import compare_arrays

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    from struphy.tests.tests_mpi.test_pic_legacy_files.accumulation_kernels_3d import kernel_step_ph_full
    from struphy.pic.particles import Particles6D
    from struphy.pic.particles_to_grid import Accumulator

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

    if rank == 0:
        print(derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    params_markers = {'Np': Np, 'eps': .25,
                      'loading': {'type': 'pseudo_random', 'seed': 1607, 'moments': [0., 0., 0., 1., 2., 3.]}
                      }

    particles = Particles6D(
        'test_particles', **params_markers, domain_array=derham.domain_array, comm=mpi_comm)

    # set random weights on each process
    particles.markers[~particles.holes,
                      6] = np.random.rand(particles.n_mks_loc)

    # gather all particles for legacy kernel
    marker_shapes = np.zeros(mpi_size, dtype=int)

    mpi_comm.Allgather(np.array([particles.markers.shape[0]]), marker_shapes)
    print(rank, marker_shapes)

    particles_leg = np.zeros(
        (sum(marker_shapes), particles.markers.shape[1]), dtype=float)

    if rank == 0:

        particles_leg[:marker_shapes[0], :] = particles.markers

        cumulative_lengths = marker_shapes[0]

        for i in range(1, mpi_size):
            arr_recv = np.zeros(
                (marker_shapes[i], particles.markers.shape[1]), dtype=float)
            mpi_comm.Recv(arr_recv, source=i)
            particles_leg[cumulative_lengths:cumulative_lengths +
                          marker_shapes[i]] = arr_recv

            cumulative_lengths += marker_shapes[i]
    else:
        mpi_comm.Send(particles.markers, dest=0)

    mpi_comm.Bcast(particles_leg, root=0)

    # sort new particles
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
        vec[a] = np.zeros((Ni[0], Ni[1], Ni[2], 3), dtype=float)

        for b in range(3):
            mat[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*SPACES.p[0] + 1,
                                 2*SPACES.p[1] + 1, 2*SPACES.p[2] + 1, 3, 3), dtype=float)

    basis_u = 1

    start_time = time()
    kernel_step_ph_full(particles_leg,
                        SPACES.T[0], SPACES.T[1], SPACES.T[2],
                        np.array(SPACES.p), np.array(Nel),
                        np.array(SPACES.NbaseN), np.array(SPACES.NbaseD),
                        particles_leg.shape[0],
                        domain.kind_map, domain.params_numpy,
                        domain.T[0], domain.T[1], domain.T[2],
                        np.array(domain.p), np.array(
                            domain.Nel), np.array(domain.NbaseN),
                        domain.cx, domain.cy, domain.cz,
                        mat[0][0], mat[0][1], mat[0][2],
                        mat[1][1], mat[1][2], mat[2][2],
                        vec[0], vec[1], vec[2],
                        basis_u)

    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

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
        print(f'Step 1 Legacy took {tot_time} seconds.')

    # =========================
    # ======== New Part =======
    # =========================
    ACC = Accumulator(derham, domain, 'Hcurl', 'pc_lin_mhd_6d_full',
                      add_vector=True, symmetry='pressure')

    start_time = time()
    ACC.accumulate(particles)

    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f'Step ph New took {tot_time} seconds.')

    # =========================
    # ======== Compare ========
    # =========================

    atol = 1e-10

    mat_temp11 = [[mat[0][0][:,:,:,:,:,:,0,0], mat[0][1][:,:,:,:,:,:,0,0], mat[0][2][:,:,:,:,:,:,0,0]],
                 [0                          , mat[1][1][:,:,:,:,:,:,0,0], mat[1][2][:,:,:,:,:,:,0,0]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,0,0]]]
    mat_temp12 = [[mat[0][0][:,:,:,:,:,:,0,1], mat[0][1][:,:,:,:,:,:,0,1], mat[0][2][:,:,:,:,:,:,0,1]],
                 [0                          , mat[1][1][:,:,:,:,:,:,0,1], mat[1][2][:,:,:,:,:,:,0,1]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,0,1]]]
    mat_temp13 = [[mat[0][0][:,:,:,:,:,:,0,2], mat[0][1][:,:,:,:,:,:,0,2], mat[0][2][:,:,:,:,:,:,0,2]],
                 [0                          , mat[1][1][:,:,:,:,:,:,0,2], mat[1][2][:,:,:,:,:,:,0,2]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,0,2]]]
    mat_temp22 = [[mat[0][0][:,:,:,:,:,:,1,1], mat[0][1][:,:,:,:,:,:,1,1], mat[0][2][:,:,:,:,:,:,1,1]],
                 [0                          , mat[1][1][:,:,:,:,:,:,1,1], mat[1][2][:,:,:,:,:,:,1,1]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,1,1]]]
    mat_temp23 = [[mat[0][0][:,:,:,:,:,:,1,2], mat[0][1][:,:,:,:,:,:,1,2], mat[0][2][:,:,:,:,:,:,1,2]],
                 [0                          , mat[1][1][:,:,:,:,:,:,1,2], mat[1][2][:,:,:,:,:,:,1,2]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,1,2]]]
    mat_temp33 = [[mat[0][0][:,:,:,:,:,:,2,2], mat[0][1][:,:,:,:,:,:,2,2], mat[0][2][:,:,:,:,:,:,2,2]],
                 [0                          , mat[1][1][:,:,:,:,:,:,2,2], mat[1][2][:,:,:,:,:,:,2,2]],
                 [0                          , 0                         , mat[2][2][:,:,:,:,:,:,2,2]]]
    vec_temp1 = [vec[0][:,:,:,0], vec[1][:,:,:,0], vec[2][:,:,:,0]]
    vec_temp2 = [vec[0][:,:,:,1], vec[1][:,:,:,1], vec[2][:,:,:,1]]
    vec_temp3 = [vec[0][:,:,:,2], vec[1][:,:,:,2], vec[2][:,:,:,2]]

    compare_arrays(ACC.operators[0].matrix.blocks[0][0], mat[0][0][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat11_11 passed test')
    compare_arrays(ACC.operators[0].matrix.blocks[0][1], mat[0][1][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat12_11 passed test')
    compare_arrays(ACC.operators[0].matrix.blocks[0][2], mat[0][2][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat13_11 passed test')
    compare_arrays(ACC.operators[0].matrix.blocks[1][1], mat[1][1][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat22_11 passed test')
    compare_arrays(ACC.operators[0].matrix.blocks[1][2], mat[1][2][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat23_11 passed test')
    compare_arrays(ACC.operators[0].matrix.blocks[2][2], mat[2][2][:,:,:,:,:,:,0,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat33_11 passed test')
    compare_arrays(ACC.vectors[0].blocks[0], vec[0][:,:,:,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec1_1 passed test')
    compare_arrays(ACC.vectors[0].blocks[1], vec[1][:,:,:,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec2_1 passed test')
    compare_arrays(ACC.vectors[0].blocks[2], vec[2][:,:,:,0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec3_1 passed test')
    compare_arrays(ACC.operators[0].matrix, mat_temp11, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_11 passed test')
    compare_arrays(ACC.operators[1].matrix, mat_temp12, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_12 passed test')
    compare_arrays(ACC.operators[2].matrix, mat_temp13, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_13 passed test')
    compare_arrays(ACC.operators[3].matrix, mat_temp22, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_22 passed test')
    compare_arrays(ACC.operators[4].matrix, mat_temp23, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_23 passed test')
    compare_arrays(ACC.operators[5].matrix, mat_temp33, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix_33 passed test')
    compare_arrays(ACC.vectors[0], vec_temp1, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block vector_1 passed test')
    compare_arrays(ACC.vectors[1], vec_temp2, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block vector_2 passed test')
    compare_arrays(ACC.vectors[2], vec_temp3, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block vector_3 passed test')


if __name__ == '__main__':
    import itertools

    for kind in itertools.product([True, False], repeat=2):
        print(kind)
        spl_kind = list(kind) + [False]
        test_accumulation([18, 10, 10], [2, 3, 4], spl_kind, ['Cuboid', {
                          'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 4.}], Np=40, verbose=True)
