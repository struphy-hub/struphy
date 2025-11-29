import pytest

from struphy.utils.pyccel import Pyccelkernel


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_vxb_analytic(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    _, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=3456,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_vxb_analytic),
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_bxu_Hdiv(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # create random FEM coefficients for magnetic field and velocity field
    _, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=3456,
        flattened=True,
    )
    _, u2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_bxu_Hdiv),
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
            u2_psy[0]._data,
            u2_psy[1]._data,
            u2_psy[2]._data,
            0.0,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_bxu_Hcurl(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # create random FEM coefficients for magnetic field
    _, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=3456,
        flattened=True,
    )
    _, u1_psy = create_equal_random_arrays(
        derham.Vh_fem["1"],
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_bxu_Hcurl),
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
            u1_psy[0]._data,
            u1_psy[1]._data,
            u1_psy[2]._data,
            0.0,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_bxu_H1vec(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # create random FEM coefficients for magnetic field
    _, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=3456,
        flattened=True,
    )
    _, uv_psy = create_equal_random_arrays(
        derham.Vh_fem["v"],
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_bxu_H1vec),
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
            uv_psy[0]._data,
            uv_psy[1]._data,
            uv_psy[2]._data,
            0.0,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_bxu_Hdiv_pauli(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # create random FEM coefficients for magnetic field
    _, b0_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["0"],
        seed=1234,
        flattened=True,
    )
    _, b2_eq_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=3456,
        flattened=True,
    )
    _, u2_psy = create_equal_random_arrays(
        derham.Vh_fem["2"],
        seed=4567,
        flattened=True,
    )

    mu0 = xp.zeros(particles.markers.copy().T.shape[1], dtype=float)

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_bxu_Hdiv_pauli),
        (
            derham.args_derham,
            *derham.p,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
            u2_psy[0]._data,
            u2_psy[1]._data,
            u2_psy[2]._data,
            b0_eq_psy._data,
            mu0,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("Nel", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("p", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "spl_kind",
    [[False, True, True], [True, False, True], [False, False, True], [True, True, True]],
)
@pytest.mark.parametrize(
    "mapping",
    [
        [
            "Colella",
            {
                "Lx": 2.0,
                "Ly": 3.0,
                "alpha": 0.1,
                "Lz": 4.0,
            },
        ],
    ],
)
def test_push_eta_rk4(Nel, p, spl_kind, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry import domains
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.pic.utilities import BoundaryParameters, LoadingParameters, WeightsParameters

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    derham = Derham(Nel, p, spl_kind, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        print("Domain decomposition : \n", derham.domain_array)

    # particle loading and sorting
    seed = 1234
    loading_params = LoadingParameters(ppc=2, seed=seed, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=domain_decomp,
        loading_params=loading_params,
    )

    particles.draw_markers()

    if show_plots:
        particles.show_physical()
    comm.Barrier()
    particles.mpi_sort_markers()
    comm.Barrier()
    if show_plots:
        particles.show_physical()

    # create legacy struphy pusher and psydac based pusher

    butcher = ButcherTableau("rk4")
    # temp fix due to refactoring of ButcherTableau:
    butcher._a = xp.diag(butcher.a, k=-1)
    butcher._a = xp.array(list(butcher._a) + [0.0])

    pusher_psy = Pusher_psy(
        particles,
        Pyccelkernel(pusher_kernels.push_eta_stage),
        (butcher.a, butcher.b, butcher.c),
        domain.args_domain,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)

    n_mks_load = xp.zeros(size, dtype=int)

    comm.Allgather(xp.array(xp.shape(particles.markers)[0]), n_mks_load)

    sendcounts = xp.zeros(size, dtype=int)
    displacements = xp.zeros(size, dtype=int)
    accum_sendcounts = 0.0

    for i in range(size):
        sendcounts[i] = n_mks_load[i] * 3
        displacements[i] = accum_sendcounts
        accum_sendcounts += sendcounts[i]

    all_particles_psy = xp.zeros((int(accum_sendcounts) * 3,), dtype=float)

    comm.Barrier()
    comm.Allgatherv(xp.array(particles.markers[:, :3]), [all_particles_psy, sendcounts, displacements, MPI.DOUBLE])
    comm.Barrier()


if __name__ == "__main__":
    test_push_vxb_analytic(
        [8, 9, 5],
        [4, 2, 3],
        [False, True, True],
        ["Colella", {"Lx": 2.0, "Ly": 2.0, "alpha": 0.1, "Lz": 4.0}],
        False,
    )
    # test_push_bxu_Hdiv([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
    #     'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    # test_push_bxu_Hcurl([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
    #     'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    # test_push_bxu_H1vec([8, 9, 5], [4, 2, 3], [False, True, True], ['Colella', {
    #     'Lx': 2., 'Ly': 2., 'alpha': 0.1, 'Lz': 4.}], False)
    # test_push_bxu_Hdiv_pauli([8, 9, 5], [2, 3, 1], [False, True, True], ['Colella', {
    #     'Lx': 2., 'Ly': 3., 'alpha': .1, 'Lz': 4.}], False)
    # test_push_eta_rk4(
    #     [8, 9, 5],
    #     [4, 2, 3],
    #     [False, True, True],
    #     [
    #         "Colella",
    #         {
    #             "Lx": 2.0,
    #             "Ly": 2.0,
    #             "alpha": 0.1,
    #             "Lz": 4.0,
    #         },
    #     ],
    #     False,
    # )
