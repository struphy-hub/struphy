import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, True], [True, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_accumulation(Nel, p, spl_kind, mapping, n_markers=10, verbose=False):
    """
    A test to compare the old accumulation routine of step1 and step3 of cc_lin_mhd_6d with the old way (files stored in
    ../test_accum_legacy_files) and the new way using the Accumulator object (ghost_region_sender, mat_vec_filler).

    The two accumulation matrices are computed with the same random magnetic field produced by
    psydac_api.utilities.create_equal_random_arrays and compared against each other at the bottom using
    psydac_api.utilities.compare_arrays().

    The times for both legacy and the new way are printed if verbose == True. This comparison only makes sense if the
    ..test_accum_legacy_files/ are also all compiled.
    """
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    cc_lin_mhd_6d_step_1(Nel, p, spl_kind, mapping, n_markers, verbose)
    if verbose and rank == 0:
        print('\nTest for Step 1 passed\n')
    cc_lin_mhd_6d_step_3(Nel, p, spl_kind, mapping, n_markers, verbose)
    if verbose and rank == 0:
        print('\nTest for Step 3 passed\n')


def cc_lin_mhd_6d_step_1(Nel, p, spl_kind, mapping, n_markers=10, verbose=False):
    import numpy as np
    from mpi4py import MPI
    from time import time

    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space

    from struphy.tests_mpi.test_accum_legacy_files.accumulation_kernels_3d import kernel_step1
    from struphy.pic.particles_to_grid import Accumulator

    mpi_comm = MPI.COMM_WORLD
    assert mpi_comm.size >= 2
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    DOMAIN = Domain(dom_type, dom_params)

    # DeRham object
    DR = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # draw particles randomly on each rank (for distribution) and use mpi.Allgather
    # to combine them into particles_leg for legacy routine
    Np = n_markers
    particles = np.zeros((Np, 7), dtype=float)

    dom = DR.domain_array[rank]
    particles[:, 0] = np.random.rand(Np)*(dom[1] - dom[0]) + dom[0]
    particles[:, 1] = np.random.rand(Np)*(dom[4] - dom[3]) + dom[3]
    particles[:, 2] = np.random.rand(Np)*(dom[7] - dom[6]) + dom[6]

    particles[:, 3:7] = np.random.rand(Np, 4)

    Np_leg = mpi_size*Np
    particles_leg = np.zeros((Np_leg, 7), dtype=float)

    mpi_comm.Allgather(particles, particles_leg)
    assert np.sum(np.abs(particles_leg[rank*Np:(rank+1)*Np, :]) -
                  np.abs(particles)) == 0.

    # create a random array for the magnetic field
    seed = 1404
    B2, B2_psy = create_equal_random_arrays(DR.V2, seed)

    # =========================
    # ====== Legacy Part ======
    # =========================

    spaces_FEM_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for a in range(3):
        Ni = SPACES.Nbase_1form[a]

        for b in range(a, 3):
            mat[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*SPACES.p[0] + 1,
                                 2*SPACES.p[1] + 1, 2*SPACES.p[2] + 1), dtype=float)

    basis_u = 1

    start_time = time()
    kernel_step1(particles_leg,
                 SPACES.T[0], SPACES.T[1], SPACES.T[2],
                 np.array(SPACES.p), np.array(Nel),
                 np.array(SPACES.NbaseN), np.array(SPACES.NbaseD),
                 Np_leg,
                 B2[0], B2[1], B2[2],
                 DOMAIN.kind_map, DOMAIN.params_map,
                 DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2],
                 np.array(DOMAIN.p), np.array(
                     DOMAIN.Nel), np.array(DOMAIN.NbaseN),
                 DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                 mat[0][1], mat[0][2], mat[1][2],
                 basis_u)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f'Step 1 Legacy took {tot_time} seconds.')

    # =========================
    # ======== New Part =======
    # =========================

    args = []

    for k in range(3):
        args += [B2_psy[k]._data[:, :, :]]

    for k in range(3):
        args += [np.array(B2_psy[k].starts)]

    ACC = Accumulator(DOMAIN, DR, 'Hcurl', 'cc_lin_mhd_6d_1',
                      *args, do_vector=False, symmetry='asym')

    start_time = time()
    ACC.accumulate(particles)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f'Step 1 New took {tot_time} seconds.')

    atol = 1e-10

    compare_arrays(ACC.matrix.blocks[0][1], mat[0][1], rank, atol=atol, verbose=verbose)
    if verbose and rank == 0:
        print('mat12 passed test')
    compare_arrays(ACC.matrix.blocks[0][2], mat[0][2], rank, atol=atol, verbose=verbose)
    if verbose and rank == 0:
        print('mat13 passed test')
    compare_arrays(ACC.matrix.blocks[1][2], mat[1][2], rank, atol=atol, verbose=verbose)
    if verbose and rank == 0:
        print('mat23 passed test')
    compare_arrays(ACC.matrix, mat, rank, atol=atol, verbose=verbose)
    if verbose and rank == 0:
        print('full block matrix passed test')

def cc_lin_mhd_6d_step_3(Nel, p, spl_kind, mapping, n_markers=10, verbose=False):
    import numpy as np
    from mpi4py import MPI
    from time import time

    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space

    from struphy.tests_mpi.test_accum_legacy_files.accumulation_kernels_3d import kernel_step3
    from struphy.pic.particles_to_grid import Accumulator

    mpi_comm = MPI.COMM_WORLD
    assert mpi_comm.size >= 2
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    DOMAIN = Domain(dom_type, dom_params)

    # DeRham object
    DR = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # draw particles randomly on each rank (for distribution) and use mpi.Allgather
    # to combine them into particles_leg for legacy routine
    Np = n_markers
    particles = np.zeros((Np, 7), dtype=float)

    dom = DR.domain_array[rank]
    particles[:, 0] = np.random.rand(Np)*(dom[1] - dom[0]) + dom[0]
    particles[:, 1] = np.random.rand(Np)*(dom[4] - dom[3]) + dom[3]
    particles[:, 2] = np.random.rand(Np)*(dom[7] - dom[6]) + dom[6]

    particles[:, 3:7] = np.random.rand(Np, 4)

    Np_leg = mpi_size*Np
    particles_leg = np.zeros((Np_leg, 7), dtype=float)

    mpi_comm.Allgather(particles, particles_leg)
    assert np.sum(np.abs(particles_leg[rank*Np:(rank+1)*Np, :]) -
                  np.abs(particles)) == 0.

    # create a random array for the magnetic field
    seed = 1404
    B2, B2_psy = create_equal_random_arrays(DR.V2, seed)

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
        vec[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)

        for b in range(a, 3):
            mat[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*SPACES.p[0] + 1,
                                 2*SPACES.p[1] + 1, 2*SPACES.p[2] + 1), dtype=float)

    basis_u = 1

    start_time = time()
    kernel_step3(particles_leg,
                 SPACES.T[0], SPACES.T[1], SPACES.T[2],
                 np.array(SPACES.p), np.array(Nel),
                 np.array(SPACES.NbaseN), np.array(SPACES.NbaseD),
                 Np_leg,
                 B2[0], B2[1], B2[2],
                 DOMAIN.kind_map, DOMAIN.params_map,
                 DOMAIN.T[0], DOMAIN.T[1], DOMAIN.T[2],
                 np.array(DOMAIN.p), np.array(
                     DOMAIN.Nel), np.array(DOMAIN.NbaseN),
                 DOMAIN.cx, DOMAIN.cy, DOMAIN.cz,
                 mat[0][0], mat[0][1], mat[0][2],
                 mat[1][1], mat[1][2], mat[2][2],
                 vec[0], vec[1], vec[2],
                 basis_u)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f'Step 3 Legacy took {tot_time} seconds.')

    # =========================
    # ======== New Part =======
    # =========================

    args = []

    for k in range(3):
        args += [B2_psy[k]._data[:, :, :]]

    for k in range(3):
        args += [np.array(B2_psy[k].starts)]

    ACC = Accumulator(DOMAIN, DR, 'Hcurl', 'cc_lin_mhd_6d_2',
                      *args, do_vector=True, symmetry='symm')

    start_time = time()
    ACC.accumulate(particles)
    end_time = time()
    tot_time = np.round(end_time - start_time, 3)

    if rank == 0 and verbose:
        print(f'Step 3 New took {tot_time} seconds.')

    atol = 1e-10

    compare_arrays(ACC.matrix.blocks[0][0], mat[0][0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat11 passed test')
    compare_arrays(ACC.matrix.blocks[0][1], mat[0][1], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat12 passed test')
    compare_arrays(ACC.matrix.blocks[0][2], mat[0][2], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat13 passed test')
    compare_arrays(ACC.matrix.blocks[1][1], mat[1][1], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat22 passed test')
    compare_arrays(ACC.matrix.blocks[1][2], mat[1][2], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat23 passed test')
    compare_arrays(ACC.matrix.blocks[2][2], mat[2][2], rank, atol=atol, verbose=verbose)
    if verbose:
        print('mat33 passed test')
    compare_arrays(ACC.vector.blocks[0], vec[0], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec1 passed test')
    compare_arrays(ACC.vector.blocks[1], vec[1], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec2 passed test')
    compare_arrays(ACC.vector.blocks[2], vec[2], rank, atol=atol, verbose=verbose)
    if verbose:
        print('vec3 passed test')
    compare_arrays(ACC.matrix, mat, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block matrix passed test')
    compare_arrays(ACC.vector, vec, rank, atol=atol, verbose=verbose)
    if verbose:
        print('full block vector passed test')


if __name__ == '__main__':
    import itertools

    for kind in itertools.product([True, False], repeat=2):
        spl_kind = list(kind) + [True]
        test_accumulation([18, 20, 20], [2, 4, 4], spl_kind, ['cuboid', {
                          'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}], n_markers=10, verbose=True)
