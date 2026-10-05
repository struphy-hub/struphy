import logging

import pytest

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_vxb_analytic(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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
        derham.V2fem,
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=3456,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_vxb_analytic"],
        (
            derham.args_derham,
            b2_eq_psy[0]._data + b2_psy[0]._data,
            b2_eq_psy[1]._data + b2_psy[1]._data,
            b2_eq_psy[2]._data + b2_psy[2]._data,
        ),
        domain.args_domain,
        alpha_in_kernel=1.0,
        pushes_eta=False,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_bxu_Hdiv(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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
        derham.V2fem,
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=3456,
        flattened=True,
    )
    _, u2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_bxu_Hdiv"],
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
        pushes_eta=False,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_bxu_Hcurl(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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
        derham.V2fem,
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=3456,
        flattened=True,
    )
    _, u1_psy = create_equal_random_arrays(
        derham.V1fem,
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_bxu_Hcurl"],
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
        pushes_eta=False,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_bxu_H1vec(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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
        derham.V2fem,
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=3456,
        flattened=True,
    )
    _, uv_psy = create_equal_random_arrays(
        derham.Vvfem,
        seed=4567,
        flattened=True,
    )

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_bxu_H1vec"],
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
        pushes_eta=False,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_bxu_Hdiv_pauli(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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
        derham.V0fem,
        seed=1234,
        flattened=True,
    )
    _, b2_eq_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=2345,
        flattened=True,
    )

    _, b2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=3456,
        flattened=True,
    )
    _, u2_psy = create_equal_random_arrays(
        derham.V2fem,
        seed=4567,
        flattened=True,
    )

    mu0 = xp.zeros(particles.markers.copy().T.shape[1], dtype=float)

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_bxu_Hdiv_pauli"],
        (
            derham.args_derham,
            *derham.degree,
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
        pushes_eta=False,
    )

    # push markers
    dt = 0.1

    pusher_psy(dt)


@pytest.mark.parametrize("num_elements", [[8, 9, 5], [7, 8, 9]])
@pytest.mark.parametrize("degree", [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (("free", "free"), ("free", "free"), None),
        (None, None, None),
    ],
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
def test_push_eta_rk4(num_elements, degree, bcs, mapping, show_plots=False):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.pushing.pusher import Pusher as Pusher_psy
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger.info("")

    # domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # discrete Derham sequence (psydac)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain_array = derham.domain_array
    nprocs = derham.domain_decomposition.nprocs
    domain_decomp = (domain_array, nprocs)

    if rank == 0:
        logger.info(f"Domain decomposition : \n{derham.domain_array}")

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

    pusher_psy = Pusher_psy(
        particles,
        catalog["push_eta_stage"],
        (butcher.a_stage, butcher.b, butcher.c, butcher.n_stages),
        domain.args_domain,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        pushes_eta=True,
        local_eval_only=True,
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


@pytest.mark.parametrize("bc", ["periodic", "reflect", "remove"])
@pytest.mark.parametrize("mapping", [["Cuboid", {}], ["Colella", {"Lx": 2.0, "Ly": 3.0, "alpha": 0.1, "Lz": 4.0}]])
def test_kinetic_bc_in_kernel(bc, mapping):
    """The per-marker boundary conditions applied inside push_eta_stage
    (apply_kinetic_bc_marker) must give the same result as Particles.apply_kinetic_bc."""
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels import catalog

    domain = getattr(domains, mapping[0])(**mapping[1])

    loading_params = LoadingParameters(Np=10000, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")
    particles = Particles6D(
        comm_world=MPI.COMM_WORLD,
        loading_params=loading_params,
        boundary_params=BoundaryParameters(bc=(bc, bc, bc)),
        domain=domain,
    )
    particles.draw_markers()
    particles.mpi_sort_markers()

    butcher = ButcherTableau("forward_euler")
    markers = particles.markers
    first_pusher_idx = particles.first_pusher_idx
    first_shift_idx = particles.first_shift_idx
    bc_type = particles.args_markers.bc_type
    bc_type_kernel = bc_type.copy()

    # large time step such that many markers leave the unit cube
    markers[:, first_pusher_idx:first_shift_idx] = markers[:, :6]
    markers[:, first_shift_idx:-2] = 0.0
    markers_init = markers.copy()
    n_holes_init = xp.count_nonzero(particles.holes)

    # reference: kernel without boundary conditions, then apply_kinetic_bc in Python
    bc_type[:] = 3
    catalog["push_eta_stage"](
        0.2, 0, particles.args_markers, domain.args_domain, butcher.a_stage, butcher.b, butcher.c, butcher.n_stages
    )
    n_outside = xp.count_nonzero(
        xp.logical_or(markers[~particles.holes, :3] > 1.0, markers[~particles.holes, :3] < 0.0)
    )
    assert n_outside > 0
    particles.apply_kinetic_bc()
    markers_ref = markers.copy()

    # boundary conditions inside the kernel
    markers[:] = markers_init
    particles.update_holes()
    n_lost_before = particles.n_lost_markers
    bc_type[:] = bc_type_kernel
    catalog["push_eta_stage"](
        0.2, 0, particles.args_markers, domain.args_domain, butcher.a_stage, butcher.b, butcher.c, butcher.n_stages
    )
    particles.finish_kernel_bc()

    assert xp.array_equal(markers[:, :first_shift_idx], markers_ref[:, :first_shift_idx])
    assert xp.all(markers[~particles.holes, :3] >= 0.0)
    assert xp.all(markers[~particles.holes, :3] <= 1.0)
    if bc == "periodic":
        shift_slice = slice(first_shift_idx, first_shift_idx + 3)
        assert xp.array_equal(markers[:, shift_slice], markers_ref[:, shift_slice])
    if bc == "remove":
        n_new_holes = xp.count_nonzero(particles.holes) - n_holes_init
        assert n_new_holes > 0
        assert particles.n_lost_markers - n_lost_before == n_new_holes


def test_kinetic_bc_remove_counts_each_marker_once():
    """A marker outside the unit cube on several "remove" axes is lost (and counted) only once."""
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.pic.particles import Particles6D

    particles = Particles6D(
        comm_world=MPI.COMM_WORLD,
        loading_params=LoadingParameters(Np=100, seed=1234, spatial="uniform"),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "remove")),
        domain=domains.Cuboid(),
    )
    particles.draw_markers()
    particles.update_holes()

    valid = xp.nonzero(particles.valid_mks)[0]
    n_valid = valid.size
    particles.markers[valid[0], :3] = [1.5, -0.5, 0.5]  # outside on two axes
    particles.markers[valid[1], :3] = [1.5, 1.5, 1.5]  # outside on all three axes
    particles.markers[valid[2], :3] = [0.5, 0.5, -0.1]  # outside on one axis

    n_lost_before = particles.n_lost_markers
    particles.apply_kinetic_bc()
    particles.update_holes()

    assert particles.n_lost_markers - n_lost_before == 3
    assert xp.count_nonzero(particles.valid_mks) == n_valid - 3


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
