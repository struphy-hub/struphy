import pytest

# ===========================================
# ========== single-threaded tests ==========
# ===========================================


@pytest.mark.mpi_skip
@pytest.mark.parametrize("comp", ((0,), (1, 2), (0, 1, 2)))
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
                "l3": 3.0,
                "r3": 4.0,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08>
    ],
)
def test_set_zero_velocity(mapping, comp: int, show_plot=False):
    """Test set_zero_velocity argument in LoadingParameters.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    comp : tuple[int]
        components to set velocity to zero
    """
    import cunumpy as xp
    from matplotlib import pyplot as plt

    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.pic.particles import Particles6D

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    zero_velocity_comp = [False, False, False]
    for c in comp:
        zero_velocity_comp[c] = True
    loading_params = LoadingParameters(Np=1e6, set_zero_velocity=zero_velocity_comp)

    boundary_params = BoundaryParameters()

    # create particles
    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
    )

    particles.draw_markers()
    particles.set_velocities_comp(velocity=0.0, comp=comp)
    particles.initialize_weights()

    # binning
    v_bins = xp.linspace(-5, 5, 500, endpoint=True)
    binned_result = []
    for i in range(3):
        components = [False] * 6
        components[3 + i] = True
        binned_result.append(particles.binning(components, [v_bins])[0])

    # tests
    if show_plot:
        dv = v_bins[1] - v_bins[0]
        v_plot = v_bins[:-1] + dv / 2
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)

        for i in range(3):
            axs[i].plot(v_plot, binned_result[i])
            axs[i].set_xlabel(rf"$v_{i + 1}$")

        fig.suptitle("Velocity distribution along 3 dimensions")
        axs[0].set_ylabel(r"f(v_i)")
        axs[0].set_ylim(0, 4)
        plt.show()

    for i in range(3):
        unique_v = xp.unique(binned_result[i]).tolist()
        if i in comp:
            # delta function has 2 unique values in distribution
            assert len(unique_v) == 2, f"velocity along component {i} not set to 0.0"
        else:
            assert len(unique_v) > 2, f"velocity along component {i} is wrongly set to 0.0"


# ==========================================
# ========== multi-threaded tests ==========
# ==========================================
@pytest.mark.parametrize("comp", ((0,), (1, 2), (0, 1, 2)))
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
                "l3": 3.0,
                "r3": 4.0,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08>
    ],
)
def test_set_zero_velocity_mpi(mapping, comp: int, show_plot=False):
    """Test set_zero_velocity argument in LoadingParameters with mpi.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    comp : tuple[int]
        components to set velocity to zero
    """

    import cunumpy as xp
    from matplotlib import pyplot as plt

    from feectools.ddm.mpi import MockComm
    from feectools.ddm.mpi import mpi as MPI
    from struphy import BoundaryParameters, LoadingParameters, domains
    from struphy.pic.particles import Particles6D

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        size = 1
        rank = 0
    else:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

    zero_velocity_comp = [False, False, False]
    for c in comp:
        zero_velocity_comp[c] = True
    loading_params = LoadingParameters(Np=1e6, set_zero_velocity=zero_velocity_comp)

    boundary_params = BoundaryParameters()

    # create particles
    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
    )

    particles.draw_markers()
    particles.set_velocities_comp(velocity=0.0, comp=comp)
    particles.initialize_weights()

    # binning
    v_bins = xp.linspace(-5, 5, 500, endpoint=True)
    binned_result = []
    for i in range(3):
        components = [False] * 6
        components[3 + i] = True
        binned_result.append(particles.binning(components, [v_bins])[0])

    # Reduce all threads to get complete result
    if comm is None:
        mpi_result = binned_result
    else:
        mpi_result = xp.zeros_like(binned_result)
        for i in range(3):
            comm.Allreduce(binned_result[i], mpi_result[i], op=MPI.SUM)
        comm.Barrier()

    # tests
    if show_plot and rank == 0:
        dv = v_bins[1] - v_bins[0]
        v_plot = v_bins[:-1] + dv / 2
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)

        for i in range(3):
            axs[i].plot(v_plot, mpi_result[i])
            axs[i].set_xlabel(rf"$v_{i + 1}$")

        fig.suptitle("Velocity distribution along 3 dimensions")
        axs[0].set_ylabel(r"f(v_i)")
        axs[0].set_ylim(0, 4)
        plt.show()

    for i in range(3):
        unique_v = xp.unique(mpi_result[i]).tolist()
        if i in comp:
            # delta function has 2 unique values in distribution
            assert len(unique_v) == 2, f"velocity along component {i} not set to 0.0"
        else:
            assert len(unique_v) > 2, f"velocity along component {i} is wrongly set to 0.0"


if __name__ == "__main__":
    from feectools.ddm.mpi import MockComm
    from feectools.ddm.mpi import mpi as MPI

    if isinstance(MPI.COMM_WORLD, MockComm):
        comm = None
        size = 1
        rank = 0

    else:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

    if comm is None or size == 1:
        test_set_zero_velocity(
            mapping=[
                # "Cuboid",
                # {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0}
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
            comp=(1, 2),
            show_plot=True,
        )

    else:
        test_set_zero_velocity_mpi(
            mapping=[
                # "Cuboid",
                # {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0}
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
            comp=(1, 2),
            show_plot=True,
        )
