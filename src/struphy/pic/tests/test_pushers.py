import pytest

# ==================================================================================


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_vxb_analytic(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    starts0 = np.array(derham.Vh['0'].starts)
    starts1 = np.array(derham.Vh['1'].starts)
    starts2 = np.array(derham.Vh['2'].starts)
    starts3 = np.array(derham.Vh['3'].starts)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    b2_str, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=3456, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=2, bc_pos=0)

    pusher_psy = Pusher_psy(derham, domain, 'push_vxb_analytic')

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step5(markers_str, dt, b2_str)

    pusher_psy(particles, dt,
               b2_eq_psy[0]._data + b2_psy[0]._data,
               b2_eq_psy[1]._data + b2_psy[1]._data,
               b2_eq_psy[2]._data + b2_psy[2]._data)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_bxu_Hdiv(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field and velocity field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    b2_str, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=3456, flattened=True)
    u2_str, u2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=4567, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=2, bc_pos=0)
    mu0_str = np.zeros(markers_str.shape[1], dtype=float)
    pow_str = np.zeros(markers_str.shape[1], dtype=float)

    pusher_psy = Pusher_psy(derham, domain, 'push_bxu_Hdiv')

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step3(markers_str, dt, b2_str, u2_str, mu0_str, pow_str)

    pusher_psy(particles, dt,
               b2_eq_psy[0]._data + b2_psy[0]._data,
               b2_eq_psy[1]._data + b2_psy[1]._data,
               b2_eq_psy[2]._data + b2_psy[2]._data,
               u2_psy[0]._data,
               u2_psy[1]._data,
               u2_psy[2]._data)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_bxu_Hcurl(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    b2_str, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=3456, flattened=True)
    u1_str, u1_psy = create_equal_random_arrays(
        derham.Vh_fem['1'], seed=4567, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=1, bc_pos=0)
    mu0_str = np.zeros(markers_str.shape[1], dtype=float)
    pow_str = np.zeros(markers_str.shape[1], dtype=float)

    pusher_psy = Pusher_psy(derham, domain, 'push_bxu_Hcurl')

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step3(markers_str, dt, b2_str, u1_str, mu0_str, pow_str)

    pusher_psy(particles, dt,
               b2_eq_psy[0]._data + b2_psy[0]._data,
               b2_eq_psy[1]._data + b2_psy[1]._data,
               b2_eq_psy[2]._data + b2_psy[2]._data,
               u1_psy[0]._data,
               u1_psy[1]._data,
               u1_psy[2]._data)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_bxu_H1vec(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    b2_str, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=3456, flattened=True)
    uv_str, uv_psy = create_equal_random_arrays(
        derham.Vh_fem['v'], seed=4567, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=0, bc_pos=0)
    mu0_str = np.zeros(markers_str.shape[1], dtype=float)
    pow_str = np.zeros(markers_str.shape[1], dtype=float)

    pusher_psy = Pusher_psy(derham, domain, 'push_bxu_H1vec')

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step3(markers_str, dt, b2_str, uv_str, mu0_str, pow_str)

    pusher_psy(particles, dt,
               b2_eq_psy[0]._data + b2_psy[0]._data,
               b2_eq_psy[1]._data + b2_psy[1]._data,
               b2_eq_psy[2]._data + b2_psy[2]._data,
               uv_psy[0]._data,
               uv_psy[1]._data,
               uv_psy[2]._data)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_bxu_Hdiv_pauli(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    b2_str, b2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=3456, flattened=True)
    u2_str, u2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=4567, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=2, bc_pos=0)
    mu0_str = np.random.rand(markers_str.shape[1])
    pow_str = np.zeros(markers_str.shape[1], dtype=float)

    pusher_psy = Pusher_psy(derham, domain, 'push_bxu_Hdiv_pauli')

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step3(markers_str, dt, b2_str, u2_str, mu0_str, pow_str)

    pusher_psy(particles, dt,
               b2_eq_psy[0]._data + b2_psy[0]._data,
               b2_eq_psy[1]._data + b2_psy[1]._data,
               b2_eq_psy[2]._data + b2_psy[2]._data,
               u2_psy[0]._data,
               u2_psy[1]._data,
               u2_psy[2]._data,
               b0_eq_psy._data,
               mu0_str)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}]])
def test_push_eta_rk4(Nel, p, spl_kind, mapping, show_plots=False):

    from mpi4py import MPI

    import numpy as np

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.pic.tests.test_pic_legacy_files.pusher import Pusher as Pusher_str
    from struphy.pic.pushing.pusher import ButcherTableau

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('')

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac and legacy struphy)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print('Domain decomposition : \n', derham.domain_array)

    spaces = [Spline_space_1d(Nel, p, spl_kind)
              for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    space = Tensor_spline_space(spaces)

    # particle loading and sorting
    seed = int(np.random.rand()*1000)
    loader_params = {'type': 'pseudo_random',
                     'seed': seed, 'moments': [0., 0., 0., 1., 1., 1.], 'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': 2, 'eps': .25, 'loading': loader_params,
                     'bc': bc_params}

    particles = Particles6D(
        'energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # make copy of markers (legacy struphy uses transposed markers!)
    markers_str = particles.markers.copy().T

    # create random FEM coefficients for magnetic field
    b0_eq_str, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    b2_eq_str, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=2345, flattened=True)

    # create legacy struphy pusher and psydac based pusher
    pusher_str = Pusher_str(domain, space, space.extract_0(
        b0_eq_str), space.extract_2(b2_eq_str), basis_u=0, bc_pos=0)

    a = [1/2, 1/2, 1.]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0., 1/2, 1/2, 1.]
    butcher = ButcherTableau(a, b, c)

    pusher_psy = Pusher_psy(derham, domain, 'push_eta_stage', n_stages=butcher.n_stages)

    # compare if markers are the same BEFORE push
    assert np.allclose(particles.markers, markers_str.T)

    # push markers
    dt = 0.1

    pusher_str.push_step4(markers_str, dt)
    pusher_psy(particles, dt,
               butcher.a, butcher.b, butcher.c)

    # compare if markers are the same AFTER push
    assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
# @pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
# @pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
# @pytest.mark.parametrize('mapping', [
#     ['Colella', {
#         'Lx' : 2., 'Ly' : 3., 'alpha' : .1, 'Lz' : 4.}]])
# def test_push_pc_GXu(Nel, p, spl_kind, mapping, show_plots=False):

#     from mpi4py import MPI

#     import numpy as np

#     from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
#     from struphy.geometry import domains
#     from struphy.feec.psydac_derham import Derham
#     from struphy.pic.particles import Particles6D
#     from struphy.pic.pushing.pusher import Pusher as Pusher_psy
#     from struphy.feec.utilities import create_equal_random_arrays
#     from struphy.tests_mpi.test_pic_legacy_files.pusher import Pusher as Pusher_str

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     print('')

#     # domain object
#     domain_class = getattr(domains, mapping[0])
#     domain = domain_class(mapping[1])

#     # discrete Derham sequence (psydac and legacy struphy)
#     derham = Derham(Nel, p, spl_kind, comm=comm)

#     starts0 = np.array(derham.Vh['0'].starts)
#     starts1 = np.array(derham.Vh['1'].starts)
#     starts2 = np.array(derham.Vh['2'].starts)
#     starts3 = np.array(derham.Vh['3'].starts)

#     if rank == 0: print('Domain decomposition : \n', derham.domain_array)

#     spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]
#     space = Tensor_spline_space(spaces)

#     # particle loading and sorting
#     loader_params = {'type': 'pseudo_random', 'seed': 1234, 'moments': [0., 0., 0., 1., 1., 1.]}
#     marker_params = {'ppc': 2, 'loading': loader_params, 'bc_type' : ['periodic', 'periodic', 'periodic']}

#     particles = Particles6D('energetic_ions', marker_params, derham.domain_array, comm)

#     if show_plots: particles.show_physical()
#     comm.Barrier()
#     particles.mpi_sort_markers()
#     comm.Barrier()
#     if show_plots: particles.show_physical()

#     # make copy of markers (legacy struphy uses transposed markers!)
#     markers_str = particles.markers.copy().T

#     # test_push_pc_GXu_Full ----------------------------------------------------------------
#     # create random FEM coefficients for magnetic field
#     b0_eq_str, b0_eq_psy = create_equal_random_arrays(derham.Vh_fem['0'], seed=1234, flattened=True)
#     b2_eq_str, b2_eq_psy = create_equal_random_arrays(derham.Vh_fem['2'], seed=2345, flattened=True)

#     # create random FEM coefficients corresponding to three of GXu
#     V_1_str, V_1_psy = create_equal_random_arrays(derham.Vh_fem['1'], seed=1234, flattened=True)
#     V_2_str, V_2_psy = create_equal_random_arrays(derham.Vh_fem['1'], seed=2345, flattened=True)
#     V_3_str, V_3_psy = create_equal_random_arrays(derham.Vh_fem['1'], seed=3456, flattened=True)

#     # create legacy struphy pusher and psydac based pusher
#     pusher_str = Pusher_str(domain, space, space.extract_0(b0_eq_str), space.extract_2(b2_eq_str), basis_u=1, bc_pos=0)
#     pusher_psy = Pusher_psy(derham, domain, 'push_pc_GXu_full')

#     # compare if markers are the same BEFORE push
#     assert np.allclose(particles.markers, markers_str.T)

#     # push markers
#     dt = 0.1

#     pusher_str.push_vel_pc_full(markers_str, dt, V_1_str, V_2_str, V_3_str)
#     pusher_psy(particles, dt,
#                V_1_psy[0]._data, V_1_psy[1]._data, V_1_psy[2]._data,
#                V_2_psy[0]._data, V_2_psy[1]._data, V_2_psy[2]._data,
#                V_3_psy[0]._data, V_3_psy[1]._data, V_3_psy[2]._data)

#     # compare if markers are the same AFTER push
#     assert np.allclose(particles.markers, markers_str.T)


# ==================================================================================
# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize('Nel', [[8, 9, 5], [7, 8, 9]])
# @pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
# @pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True], [False, False, True], [True, True, True]])
# @pytest.mark.parametrize('mapping', [
#     ['Colella', {
#         'Lx' : 2., 'Ly' : 3., 'alpha' : .1, 'Lz' : 4.}]])
# def test_push_eta_pc(Nel, p, spl_kind, mapping, show_plots=False):

#     from mpi4py import MPI

#     import numpy as np

#     from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
#     from struphy.geometry import domains
#     from struphy.feec.psydac_derham import Derham
#     from struphy.pic.particles import Particles6D
#     from struphy.pic.pushing.pusher import Pusher as Pusher_psy
#     from struphy.feec.utilities import create_equal_random_arrays
#     from struphy.tests_mpi.test_pic_legacy_files.pusher import Pusher as Pusher_str

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     print('')

#     # domain object
#     domain_class = getattr(domains, mapping[0])
#     domain = domain_class(mapping[1])

#     # discrete Derham sequence (psydac and legacy struphy)
#     derham = Derham(Nel, p, spl_kind, comm=comm)

#     if rank == 0: print('Domain decomposition : \n', derham.domain_array)

#     spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

#     space = Tensor_spline_space(spaces)

#     # particle loading and sorting
#     loader_params = {'type': 'pseudo_random', 'seed': 1234, 'moments': [0., 0., 0., 1., 1., 1.]}
#     marker_params = {'ppc': 1, 'loading': loader_params, 'bc_type' : ['periodic', 'periodic', 'periodic']}

#     particles = Particles6D('energetic_ions', marker_params, derham.domain_array, comm)

#     comm.Barrier()
#     particles.mpi_sort_markers()
#     comm.Barrier()

#     # make copy of markers (legacy struphy uses transposed markers!)
#     markers_str = particles.markers.copy().T

#     # test_push_pc_GXu_Full ----------------------------------------------------------------
#     # create random FEM coefficients for magnetic field
#     b0_eq_str, b0_eq_psy = create_equal_random_arrays(derham.Vh_fem['0'], seed=1234, flattened=True)
#     b2_eq_str, b2_eq_psy = create_equal_random_arrays(derham.Vh_fem['2'], seed=2345, flattened=True)

#     # create random FEM coefficients corresponding to three of GXu
#     up_str, up_psy = create_equal_random_arrays(derham.Vh_fem['1'], seed=1234, flattened=True)

#     # create legacy struphy pusher and psydac based pusher
#     pusher_str = Pusher_str(domain, space, space.extract_0(b0_eq_str), space.extract_2(b2_eq_str), basis_u=1, bc_pos=0)
#     pusher_psy = Pusher_psy(derham, domain, 'push_pc_eta_rk4_Hcurl_full', stage_num = 4)

#     # compare if markers are the same BEFORE push
#     assert np.allclose(particles.markers, markers_str.T)

#     # push markers
#     dt = 0.1

#     pusher_str.push_eta_pc_full(markers_str, dt, up_str)
#     pusher_psy(particles, dt, up_psy[0]._data, up_psy[1]._data, up_psy[2]._data, do_mpi_sort=True)

#     n_mks_load = np.zeros(size, dtype=int)

#     comm.Allgather(np.array(np.shape(particles.markers)[0]), n_mks_load)

#     sendcounts = np.zeros(size, dtype=int)
#     displacements = np.zeros(size, dtype=int)
#     accum_sendcounts = 0.

#     for i in range(size) :
#         sendcounts[i] = n_mks_load[i]*3
#         displacements[i] = accum_sendcounts
#         accum_sendcounts += sendcounts[i]

#     all_particles_psy = np.zeros((int(accum_sendcounts)*3,), dtype = float)
#     all_particles_str = np.zeros((int(accum_sendcounts)*3,), dtype = float)

#     comm.Barrier()
#     comm.Allgatherv(np.array(particles.markers[:,0:3]), [all_particles_psy, sendcounts, displacements, MPI.DOUBLE])
#     comm.Barrier()
#     comm.Allgatherv(np.array(markers_str.T[:,0:3]), [all_particles_str, sendcounts, displacements, MPI.DOUBLE])
#     comm.Barrier()

#     unique_psy = np.unique(all_particles_psy)
#     unique_str = np.unique(all_particles_str)

#     assert np.allclose(unique_psy, unique_str)


if __name__ == '__main__':
    test_push_vxb_analytic([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    test_push_bxu_Hdiv([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    test_push_bxu_Hcurl([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    test_push_bxu_H1vec([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    test_push_bxu_Hdiv_pauli([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    test_push_eta_rk4([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
        'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    # test_push_pc_GXu([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
    #     'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    # test_push_eta_pc([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
    #      'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
