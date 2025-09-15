import pytest

# TODO: add tests for Particles5D

# ===========================================
# ========== single-threaded tests ==========
# ===========================================


@pytest.mark.mpi_skip
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
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_full_f(mapping, show_plot=False):
    """Test Maxwellian in v1-direction and cosine perturbation for full-f Particles6D.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D
    from struphy.pic.utilities import (
        BoundaryParameters,
        LoadingParameters,
        WeightsParameters,
    )

    # Set seed
    seed = 1234

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # create particles
    bc_params = ("periodic", "periodic", "periodic")

    # ===========================================
    # ===== Test Maxwellian in v1 direction =====
    # ===========================================
    loading_params = LoadingParameters(Np=Np, seed=seed, spatial="uniform")
    boundary_params = BoundaryParameters(bc=bc_params)

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
    )

    particles.draw_markers()

    # test weights
    particles.initialize_weights()

    v1_bins = np.linspace(-5.0, 5.0, 200, endpoint=True)
    dv = v1_bins[1] - v1_bins[0]

    binned_res, r2 = particles.binning(
        [False, False, False, True, False, False],
        [v1_bins],
    )

    v1_plot = v1_bins[:-1] + dv / 2

    ana_res = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(v1_plot**2) / 2.0)

    if show_plot:
        plt.plot(v1_plot, ana_res, label="Analytical result")
        plt.plot(v1_plot, binned_res, "r*", label="From binning")
        plt.title(r"Full-$f$: Maxwellian in $v_1$-direction")
        plt.xlabel(r"$v_1$")
        plt.ylabel(r"$f(v_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.02, f"Error between binned data and analytical result was {l2_error}"

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    # test weights
    amp_n = 0.1
    l_n = 2
    pert = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    maxwellian = Maxwellian3D(n=(1.0, pert))

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
        background=maxwellian,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = 1.0 + amp_n * np.cos(2 * np.pi * l_n * e1_plot)

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, "r*", label="From binning")
        plt.title(r"Full-$f$: Cosine in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.02, f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    n1 = 0.8
    n2 = 0.2

    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4

    pert_1 = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    pert_2 = perturbations.ModesCos(ls=(l_n2,), amps=(amp_n2,))
    maxw_1 = Maxwellian3D(n=(n1, pert_1))
    maxw_2 = Maxwellian3D(n=(n2, pert_2), u1=(4.5, None), vth1=(0.5, None))
    background = maxw_1 + maxw_2

    # adapt s0 for importance sampling
    loading_params = LoadingParameters(
        Np=Np,
        seed=seed,
        spatial="uniform",
        moments=(2.5, 0, 0, 3, 1, 1),
    )

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = n1 + amp_n1 * np.cos(2 * np.pi * l_n1 * e1_plot) + n2 + amp_n2 * np.cos(2 * np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot:
        s0 = Maxwellian3D(
            n=(1.0, None),
            u1=(particles.loading_params.moments[0], None),
            u2=(particles.loading_params.moments[1], None),
            u3=(particles.loading_params.moments[2], None),
            vth1=(particles.loading_params.moments[3], None),
            vth2=(particles.loading_params.moments[4], None),
            vth3=(particles.loading_params.moments[5], None),
        )

        v1 = np.linspace(-10.0, 10.0, 400)
        phase_space = np.meshgrid(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            v1,
            np.array([0.0]),
            np.array([0.0]),
        )

        s0_vals = s0(*phase_space).squeeze()
        f0_vals = particles._f_init(*phase_space).squeeze()

        plt.plot(v1, s0_vals, label=r"$s_0$")
        plt.plot(v1, f0_vals, label=r"$f_0$")
        plt.legend()
        plt.xlabel(r"$v_1$")
        plt.title(r"Drawing from $s_0$ and initializing from $f_0$")
        plt.show()

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, "r*", label="From binning")
        plt.title(r"Full-$f$: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.04, f"Error between binned data and analytical result was {l2_error}"


@pytest.mark.mpi_skip
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
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_delta_f(mapping, show_plot=False):
    """Test Maxwellian in v1-direction and cosine perturbation for delta-f Particles6D.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import DeltaFParticles6D
    from struphy.pic.utilities import (
        BoundaryParameters,
        LoadingParameters,
        WeightsParameters,
    )

    # Set seed
    seed = 1234

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # create particles
    bc_params = ("periodic", "periodic", "periodic")

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    loading_params = LoadingParameters(Np=Np, seed=seed, spatial="uniform")
    boundary_params = BoundaryParameters(bc=bc_params)

    # test weights
    amp_n = 0.1
    l_n = 2
    pert = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    background = Maxwellian3D(n=(1.0, pert))

    particles = DeltaFParticles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = amp_n * np.cos(2 * np.pi * l_n * e1_plot)

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, "r*", label="From binning")
        plt.title(r"$\delta f$: Cosine in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.02, f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    n1 = 0.8
    n2 = 0.2

    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4

    pert_1 = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    pert_2 = perturbations.ModesCos(ls=(l_n2,), amps=(amp_n2,))
    maxw_1 = Maxwellian3D(n=(n1, pert_1))
    maxw_2 = Maxwellian3D(n=(n2, pert_2), u1=(4.5, None), vth1=(0.5, None))
    background = maxw_1 + maxw_2

    # adapt s0 for importance sampling
    loading_params = LoadingParameters(
        Np=Np,
        seed=seed,
        spatial="uniform",
        moments=(2.5, 0, 0, 2, 1, 1),
    )

    particles = DeltaFParticles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = amp_n1 * np.cos(2 * np.pi * l_n1 * e1_plot) + amp_n2 * np.cos(2 * np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot:
        s0 = Maxwellian3D(
            n=(1.0, None),
            u1=(particles.loading_params.moments[0], None),
            u2=(particles.loading_params.moments[1], None),
            u3=(particles.loading_params.moments[2], None),
            vth1=(particles.loading_params.moments[3], None),
            vth2=(particles.loading_params.moments[4], None),
            vth3=(particles.loading_params.moments[5], None),
        )

        v1 = np.linspace(-10.0, 10.0, 400)
        phase_space = np.meshgrid(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            v1,
            np.array([0.0]),
            np.array([0.0]),
        )

        s0_vals = s0(*phase_space).squeeze()
        f0_vals = particles._f_init(*phase_space).squeeze()

        plt.plot(v1, s0_vals, label=r"$s_0$")
        plt.plot(v1, f0_vals, label=r"$f_0$")
        plt.legend()
        plt.xlabel(r"$v_1$")
        plt.title(r"Drawing from $s_0$ and initializing from $f_0$")
        plt.show()

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, "r*", label="From binning")
        plt.title(r"$\delta f$: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.04, f"Error between binned data and analytical result was {l2_error}"


# ==========================================
# ========== multi-threaded tests ==========
# ==========================================
@pytest.mark.mpi(min_size=2)
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
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_full_f_mpi(mapping, show_plot=False):
    """Test Maxwellian in v1-direction and cosine perturbation for full-f Particles6D with mpi.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D
    from struphy.pic.utilities import (
        BoundaryParameters,
        LoadingParameters,
        WeightsParameters,
    )

    # Set seed
    seed = 1234

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    assert size > 1

    # create particles
    bc_params = ("periodic", "periodic", "periodic")

    # ===========================================
    # ===== Test Maxwellian in v1 direction =====
    # ===========================================
    loading_params = LoadingParameters(Np=Np, seed=seed, spatial="uniform")
    boundary_params = BoundaryParameters(bc=bc_params)

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        comm_world=comm,
        domain=domain,
    )
    particles.draw_markers()

    # test weights
    particles.initialize_weights()

    v1_bins = np.linspace(-5.0, 5.0, 200, endpoint=True)
    dv = v1_bins[1] - v1_bins[0]

    binned_res, r2 = particles.binning(
        [False, False, False, True, False, False],
        [v1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    v1_plot = v1_bins[:-1] + dv / 2

    ana_res = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(v1_plot**2) / 2.0)

    if show_plot and rank == 0:
        plt.plot(v1_plot, ana_res, label="Analytical result")
        plt.plot(v1_plot, mpi_res, "r*", label="From binning")
        plt.title(r"Full-$f$ with MPI: Maxwellian in $v_1$-direction")
        plt.xlabel(r"$v_1$")
        plt.ylabel(r"$f(v_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.03, f"Error between binned data and analytical result was {l2_error}"

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    # test weights
    amp_n = 0.1
    l_n = 2
    pert = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    maxwellian = Maxwellian3D(n=(1.0, pert))

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        comm_world=comm,
        domain=domain,
        background=maxwellian,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = 1.0 + amp_n * np.cos(2 * np.pi * l_n * e1_plot)

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, "r*", label="From binning")
        plt.title(r"Full-$f$ with MPI: Cosine in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.03, f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        "Maxwellian3D_1": {
            "n": n1,
        },
        "Maxwellian3D_2": {
            "n": n2,
            "vth1": 0.5,
            "u1": 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        "Maxwellian3D_1": {
            "n": {
                "ModesCos": {
                    "given_in_basis": "0",
                    "ls": [l_n1],
                    "amps": [amp_n1],
                }
            }
        },
        "Maxwellian3D_2": {
            "n": {
                "ModesCos": {
                    "given_in_basis": "0",
                    "ls": [l_n2],
                    "amps": [amp_n2],
                }
            }
        },
    }
    pert_1 = perturbations.ModesCos(ls=(l_n1,), amps=(amp_n1,))
    pert_2 = perturbations.ModesCos(ls=(l_n2,), amps=(amp_n2,))
    maxw_1 = Maxwellian3D(n=(n1, pert_1))
    maxw_2 = Maxwellian3D(n=(n2, pert_2), u1=(4.5, None), vth1=(0.5, None))
    background = maxw_1 + maxw_2

    # adapt s0 for importance sampling
    loading_params = LoadingParameters(
        Np=Np,
        seed=seed,
        spatial="uniform",
        moments=(2.5, 0, 0, 2, 1, 1),
    )

    particles = Particles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        comm_world=comm,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = n1 + amp_n1 * np.cos(2 * np.pi * l_n1 * e1_plot) + n2 + amp_n2 * np.cos(2 * np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot and rank == 0:
        s0 = Maxwellian3D(
            n=(1.0, None),
            u1=(particles.loading_params.moments[0], None),
            u2=(particles.loading_params.moments[1], None),
            u3=(particles.loading_params.moments[2], None),
            vth1=(particles.loading_params.moments[3], None),
            vth2=(particles.loading_params.moments[4], None),
            vth3=(particles.loading_params.moments[5], None),
        )

        v1 = np.linspace(-10.0, 10.0, 400)
        phase_space = np.meshgrid(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            v1,
            np.array([0.0]),
            np.array([0.0]),
        )

        s0_vals = s0(*phase_space).squeeze()
        f0_vals = particles._f_init(*phase_space).squeeze()

        plt.plot(v1, s0_vals, label=r"$s_0$")
        plt.plot(v1, f0_vals, label=r"$f_0$")
        plt.legend()
        plt.xlabel(r"$v_1$")
        plt.title(r"Drawing from $s_0$ and initializing from $f_0$")
        plt.show()

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, "r*", label="From binning")
        plt.title(r"Full-$f$ with MPI: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.04, f"Error between binned data and analytical result was {l2_error}"


@pytest.mark.mpi(min_size=2)
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
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_delta_f_mpi(mapping, show_plot=False):
    """Test Maxwellian in v1-direction and cosine perturbation for delta-f Particles6D with mpi.

    Parameters
    ----------
    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import DeltaFParticles6D
    from struphy.pic.utilities import (
        BoundaryParameters,
        LoadingParameters,
        WeightsParameters,
    )

    # Set seed
    seed = 1234

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    assert size > 1

    # create particles
    bc_params = ("periodic", "periodic", "periodic")

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    loading_params = LoadingParameters(Np=Np, seed=seed, spatial="uniform")
    boundary_params = BoundaryParameters(bc=bc_params)

    # test weights
    amp_n = 0.1
    l_n = 2
    pert_params = {
        "n": {
            "ModesCos": {
                "given_in_basis": "0",
                "ls": [l_n],
                "amps": [amp_n],
            }
        }
    }
    pert = perturbations.ModesCos(ls=(l_n,), amps=(amp_n,))
    background = Maxwellian3D(n=(1.0, pert))

    particles = DeltaFParticles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        comm_world=comm,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = amp_n * np.cos(2 * np.pi * l_n * e1_plot)

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, "r*", label="From binning")
        plt.title(r"$\delta f$ with MPI: Cosine in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.02, f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        "Maxwellian3D_1": {
            "n": n1,
        },
        "Maxwellian3D_2": {
            "n": n2,
            "vth1": 0.5,
            "u1": 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        "Maxwellian3D_1": {
            "use_background_n": False,
            "n": {
                "ModesCos": {
                    "given_in_basis": "0",
                    "ls": [l_n1],
                    "amps": [amp_n1],
                }
            },
        },
        "Maxwellian3D_2": {
            "use_background_n": True,
            "n": {
                "ModesCos": {
                    "given_in_basis": "0",
                    "ls": [l_n2],
                    "amps": [amp_n2],
                }
            },
        },
    }
    pert_1 = perturbations.ModesCos(ls=(l_n1,), amps=(amp_n1,))
    pert_2 = perturbations.ModesCos(ls=(l_n2,), amps=(amp_n2,))
    maxw_1 = Maxwellian3D(n=(n1, pert_1))
    maxw_2 = Maxwellian3D(n=(n2, pert_2), u1=(4.5, None), vth1=(0.5, None))
    background = maxw_1 + maxw_2

    # adapt s0 for importance sampling
    loading_params = LoadingParameters(
        Np=Np,
        seed=seed,
        spatial="uniform",
        moments=(2.5, 0, 0, 2, 1, 1),
    )

    particles = DeltaFParticles6D(
        loading_params=loading_params,
        boundary_params=boundary_params,
        comm_world=comm,
        domain=domain,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0.0, 1.0, 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de / 2

    ana_res = amp_n1 * np.cos(2 * np.pi * l_n1 * e1_plot) + amp_n2 * np.cos(2 * np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot and rank == 0:
        s0 = Maxwellian3D(
            n=(1.0, None),
            u1=(particles.loading_params.moments[0], None),
            u2=(particles.loading_params.moments[1], None),
            u3=(particles.loading_params.moments[2], None),
            vth1=(particles.loading_params.moments[3], None),
            vth2=(particles.loading_params.moments[4], None),
            vth3=(particles.loading_params.moments[5], None),
        )

        v1 = np.linspace(-10.0, 10.0, 400)
        phase_space = np.meshgrid(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            v1,
            np.array([0.0]),
            np.array([0.0]),
        )

        s0_vals = s0(*phase_space).squeeze()
        f0_vals = particles._f_init(*phase_space).squeeze()

        plt.plot(v1, s0_vals, label=r"$s_0$")
        plt.plot(v1, f0_vals, label=r"$f_0$")
        plt.legend()
        plt.xlabel(r"$v_1$")
        plt.title(r"Drawing from $s_0$ and initializing from $f_0$")
        plt.show()

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, "r*", label="From binning")
        plt.title(r"$\delta f$ with MPI: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r"$\eta_1$")
        plt.ylabel(r"$f(\eta_1)$")
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res) ** 2)) / np.sqrt(np.sum((ana_res) ** 2))

    assert l2_error <= 0.04, f"Error between binned data and analytical result was {l2_error}"


if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if comm is None or size == 1:
        test_binning_6D_full_f(
            mapping=[
                "Cuboid",
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0},
                # 'ShafranovDshapedCylinder',
                # {'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07,
                #     'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}
            ],
            show_plot=True,
        )
        test_binning_6D_delta_f(
            mapping=[
                "Cuboid",
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0},
            ],
            show_plot=True,
        )
    else:
        test_binning_6D_full_f_mpi(
            mapping=[
                "Cuboid",
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0},
                # 'ShafranovDshapedCylinder',
                # {'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07,
                #     'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}
            ],
            show_plot=True,
        )
        test_binning_6D_delta_f_mpi(
            mapping=[
                "Cuboid",
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 10.0, "r3": 20.0},
            ],
            show_plot=True,
        )
