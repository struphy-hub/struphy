import pytest

# TODO: add tests for Particles5D


# ===========================================
# ========== single-threaded tests ==========
# ===========================================
@pytest.mark.mpi_skip
@pytest.mark.parametrize('Nel', [[12, 9, 2]])
@pytest.mark.parametrize('p', [[3, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize(
    'mapping', [
        [
            'Cuboid', {
                'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 3., 'r3': 4.,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_full_f(Nel, p, spl_kind, mapping, show_plot=False):
    """ Test Maxwellian in v1-direction and cosine perturbation for full-f Particles6D.

    Parameters
    ----------
    Nel : list[int]
        number of elements in each space-direction

    p : list[int]
        number of spline degrees in each space-direction

    spl_kind : list[int]
        periodicity of splines in each space-direction

    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D

    # Set seed
    seed = int(np.random.rand()*1000)

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size == 1
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create particles
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    bc_params = {'type': ['periodic', 'periodic', 'periodic']}

    # ===========================================
    # ===== Test Maxwellian in v1 direction =====
    # ===========================================
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params, derham=derham,
    )
    particles.draw_markers()

    # test weights
    particles.initialize_weights()

    v1_bins = np.linspace(-5., 5., 200, endpoint=True)
    dv = v1_bins[1] - v1_bins[0]

    binned_res, r2 = particles.binning(
        [False, False, False, True, False, False],
        [v1_bins],
    )

    v1_plot = v1_bins[:-1] + dv/2

    ana_res = 1. / np.sqrt(2.*np.pi) * np.exp(- v1_plot**2 / 2.)

    if show_plot:
        plt.plot(v1_plot, ana_res, label="Analytical result")
        plt.plot(v1_plot, binned_res, 'r*', label="From binning")
        plt.title(r"Full-$f$: Maxwellian in $v_1$-direction")
        plt.xlabel(r'$v_1$')
        plt.ylabel(r'$f(v_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.02, \
        f"Error between binned data and analytical result was {l2_error}"

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None
    # test weights
    amp_n = 0.1
    l_n = 2
    pert_params = {
        'type': 'ModesCos',
        'ModesCos': {
            'comps': {'n': '0'},
            'ls': {'n': [l_n]},
            'amps': {'n': [amp_n]},
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de/2

    ana_res = (1. + amp_n * np.cos(2*np.pi * l_n * e1_plot))

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, 'r*', label="From binning")
        plt.title(r"Full-$f$: Cosine in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.02, \
        f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        'type': ['Maxwellian3D_1', 'Maxwellian3D_2'],
        'Maxwellian3D_1': {
            'n': n1,
        },
        'Maxwellian3D_2': {
            'n': n2,
            'vth1': 0.5,
            'u1': 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        'Maxwellian3D_1': {
            'type': "ModesCos",
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n1]},
                'amps': {'n': [amp_n1]},
            },
        },
        'Maxwellian3D_2': {
            'type': "ModesCos",
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n2]},
                'amps': {'n': [amp_n2]},
            },
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de/2

    ana_res = n1 + amp_n1 * np.cos(2*np.pi * l_n1 * e1_plot) + \
        n2 + amp_n2 * np.cos(2*np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot:
        s0_dict = {
            'n': 1.,
            'u1': particles.marker_params['loading']['moments'][0],
            'u2': particles.marker_params['loading']['moments'][1],
            'u3': particles.marker_params['loading']['moments'][2],
            'vth1': particles.marker_params['loading']['moments'][3],
            'vth2': particles.marker_params['loading']['moments'][4],
            'vth3': particles.marker_params['loading']['moments'][5],
        }
        s0 = Maxwellian3D(maxw_params=s0_dict)

        v1 = np.linspace(-10., 10., 400)
        phase_space = np.meshgrid(
            np.array([0.]),
            np.array([0.]),
            np.array([0.]),
            v1,
            np.array([0.]),
            np.array([0.]),
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
        plt.plot(e1_plot, binned_res, 'r*', label="From binning")
        plt.title(r"Full-$f$: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.04, \
        f"Error between binned data and analytical result was {l2_error}"


@pytest.mark.mpi_skip
@pytest.mark.parametrize('Nel', [[12, 9, 2]])
@pytest.mark.parametrize('p', [[3, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize(
    'mapping', [
        [
            'Cuboid', {
                'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 3., 'r3': 4.,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_delta_f(Nel, p, spl_kind, mapping, show_plot=False):
    """ Test Maxwellian in v1-direction and cosine perturbation for delta-f Particles6D.

    Parameters
    ----------
    Nel : list[int]
        number of elements in each space-direction

    p : list[int]
        number of spline degrees in each space-direction

    spl_kind : list[int]
        periodicity of splines in each space-direction

    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D

    # Set seed
    seed = int(np.random.rand()*1000)

    # Set number of particles for which error is known <= 0.1
    Np = int(1e6)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    assert size == 1
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create particles
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    bc_params = {'type': ['periodic', 'periodic', 'periodic']}

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    marker_params = {
        'Np': Np,
        'type': 'delta_f',
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None
    # test weights
    amp_n = 0.1
    l_n = 2
    pert_params = {
        'type': 'ModesCos',
        'ModesCos': {
            'comps': {'n': '0'},
            'ls': {'n': [l_n]},
            'amps': {'n': [amp_n]},
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de/2

    ana_res = (amp_n * np.cos(2*np.pi * l_n * e1_plot))

    if show_plot:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, binned_res, 'r*', label="From binning")
        plt.title(r"$\delta f$: Cosine in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.02, \
        f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    marker_params = {
        'Np': Np,
        'type': 'delta_f',
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        'type': ['Maxwellian3D_1', 'Maxwellian3D_2'],
        'Maxwellian3D_1': {
            'n': n1,
        },
        'Maxwellian3D_2': {
            'n': n2,
            'vth1': 0.5,
            'u1': 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        'Maxwellian3D_1': {
            'type': "ModesCos",
            'use_background_n': False,
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n1]},
                'amps': {'n': [amp_n1]},
            },
        },
        'Maxwellian3D_2': {
            'type': "ModesCos",
            'use_background_n': True,
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n2]},
                'amps': {'n': [amp_n2]},
            },
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    e1_plot = e1_bins[:-1] + de/2

    ana_res = amp_n1 * np.cos(2*np.pi * l_n1 * e1_plot) + \
        n2 + amp_n2 * np.cos(2*np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot:
        s0_dict = {
            'n': 1.,
            'u1': particles.marker_params['loading']['moments'][0],
            'u2': particles.marker_params['loading']['moments'][1],
            'u3': particles.marker_params['loading']['moments'][2],
            'vth1': particles.marker_params['loading']['moments'][3],
            'vth2': particles.marker_params['loading']['moments'][4],
            'vth3': particles.marker_params['loading']['moments'][5],
        }
        s0 = Maxwellian3D(maxw_params=s0_dict)

        v1 = np.linspace(-10., 10., 400)
        phase_space = np.meshgrid(
            np.array([0.]),
            np.array([0.]),
            np.array([0.]),
            v1,
            np.array([0.]),
            np.array([0.]),
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
        plt.plot(e1_plot, binned_res, 'r*', label="From binning")
        plt.title(r"$\delta f$: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - binned_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.04, \
        f"Error between binned data and analytical result was {l2_error}"


# ==========================================
# ========== multi-threaded tests ==========
# ==========================================


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[12, 9, 2]])
@pytest.mark.parametrize('p', [[3, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize(
    'mapping', [
        [
            'Cuboid', {
                'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 3., 'r3': 4.,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_full_f_mpi(Nel, p, spl_kind, mapping, show_plot=False):
    """ Test Maxwellian in v1-direction and cosine perturbation for full-f Particles6D with mpi.

    Parameters
    ----------
    Nel : list[int]
        number of elements in each space-direction

    p : list[int]
        number of spline degrees in each space-direction

    spl_kind : list[int]
        periodicity of splines in each space-direction

    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D

    # Set seed
    seed = int(np.random.rand()*1000)

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
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create particles
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    bc_params = {'type': ['periodic', 'periodic', 'periodic']}

    # ===========================================
    # ===== Test Maxwellian in v1 direction =====
    # ===========================================
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params, derham=derham,
    )
    particles.draw_markers()

    # test weights
    particles.initialize_weights()

    v1_bins = np.linspace(-5., 5., 200, endpoint=True)
    dv = v1_bins[1] - v1_bins[0]

    binned_res, r2 = particles.binning(
        [False, False, False, True, False, False],
        [v1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    v1_plot = v1_bins[:-1] + dv/2

    ana_res = (1. / np.sqrt(2.*np.pi) * np.exp(- v1_plot**2 / 2.))

    if show_plot and rank == 0:
        plt.plot(v1_plot, ana_res, label="Analytical result")
        plt.plot(v1_plot, mpi_res, 'r*', label="From binning")
        plt.title(r"Full-$f$ with MPI: Maxwellian in $v_1$-direction")
        plt.xlabel(r'$v_1$')
        plt.ylabel(r'$f(v_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.03, \
        f"Error between binned data and analytical result was {l2_error}"

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None
    # test weights
    amp_n = 0.1
    l_n = 2
    pert_params = {
        'type': 'ModesCos',
        'ModesCos': {
            'comps': {'n': '0'},
            'ls': {'n': [l_n]},
            'amps': {'n': [amp_n]},
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de/2

    ana_res = 1. + amp_n * np.cos(2*np.pi * l_n * e1_plot)

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, 'r*', label="From binning")
        plt.title(r"Full-$f$ with MPI: Cosine in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.03, \
        f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    marker_params = {
        'Np': Np,
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        'type': ['Maxwellian3D_1', 'Maxwellian3D_2'],
        'Maxwellian3D_1': {
            'n': n1,
        },
        'Maxwellian3D_2': {
            'n': n2,
            'vth1': 0.5,
            'u1': 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        'Maxwellian3D_1': {
            'type': "ModesCos",
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n1]},
                'amps': {'n': [amp_n1]},
            },
        },
        'Maxwellian3D_2': {
            'type': "ModesCos",
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n2]},
                'amps': {'n': [amp_n2]},
            },
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de/2

    ana_res = n1 + amp_n1 * np.cos(2*np.pi * l_n1 * e1_plot) + \
        n2 + amp_n2 * np.cos(2*np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot and rank == 0:
        s0_dict = {
            'n': 1.,
            'u1': particles.marker_params['loading']['moments'][0],
            'u2': particles.marker_params['loading']['moments'][1],
            'u3': particles.marker_params['loading']['moments'][2],
            'vth1': particles.marker_params['loading']['moments'][3],
            'vth2': particles.marker_params['loading']['moments'][4],
            'vth3': particles.marker_params['loading']['moments'][5],
        }
        s0 = Maxwellian3D(maxw_params=s0_dict)

        v1 = np.linspace(-10., 10., 400)
        phase_space = np.meshgrid(
            np.array([0.]),
            np.array([0.]),
            np.array([0.]),
            v1,
            np.array([0.]),
            np.array([0.]),
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
        plt.plot(e1_plot, mpi_res, 'r*', label="From binning")
        plt.title(r"Full-$f$ with MPI: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.04, \
        f"Error between binned data and analytical result was {l2_error}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[12, 9, 2]])
@pytest.mark.parametrize('p', [[3, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize(
    'mapping', [
        [
            'Cuboid', {
                'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 3., 'r3': 4.,
            },
        ],
        # ['ShafranovDshapedCylinder', {
        #     'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
    ],
)
def test_binning_6D_delta_f_mpi(Nel, p, spl_kind, mapping, show_plot=False):
    """ Test Maxwellian in v1-direction and cosine perturbation for delta-f Particles6D with mpi.

    Parameters
    ----------
    Nel : list[int]
        number of elements in each space-direction

    p : list[int]
        number of spline degrees in each space-direction

    spl_kind : list[int]
        periodicity of splines in each space-direction

    mapping : tuple[String, dict] (or list with 2 entries)
        name and specification of the mapping
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.pic.particles import Particles6D

    # Set seed
    seed = int(np.random.rand()*1000)

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
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create particles
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    bc_params = {'type': ['periodic', 'periodic', 'periodic']}

    # =========================================
    # ===== Test cosine in eta1 direction =====
    # =========================================
    marker_params = {
        'Np': Np,
        'type': 'delta_f',
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    bckgr_params = None
    # test weights
    amp_n = 0.1
    l_n = 2
    pert_params = {
        'type': 'ModesCos',
        'ModesCos': {
            'comps': {'n': '0'},
            'ls': {'n': [l_n]},
            'amps': {'n': [amp_n]},
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de/2

    ana_res = (amp_n * np.cos(2*np.pi * l_n * e1_plot))

    if show_plot and rank == 0:
        plt.plot(e1_plot, ana_res, label="Analytical result")
        plt.plot(e1_plot, mpi_res, 'r*', label="From binning")
        plt.title(r"$\delta f$ with MPI: Cosine in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.02, \
        f"Error between binned data and analytical result was {l2_error}"

    # ==============================================================
    # ===== Test cosines for two backgrounds in eta1 direction =====
    # ==============================================================
    loading_params = {
        'type': 'pseudo_random',
        'seed': seed,
        'spatial': 'uniform',
    }
    marker_params = {
        'Np': Np,
        'type': 'delta_f',
        'eps': .25,
        'loading': loading_params,
        'bc': bc_params,
        'domain': domain,
    }
    n1 = 0.8
    n2 = 0.2
    bckgr_params = {
        'type': ['Maxwellian3D_1', 'Maxwellian3D_2'],
        'Maxwellian3D_1': {
            'n': n1,
        },
        'Maxwellian3D_2': {
            'n': n2,
            'vth1': 0.5,
            'u1': 4.5,
        },
    }
    # test weights
    amp_n1 = 0.1
    amp_n2 = 0.1
    l_n1 = 2
    l_n2 = 4
    pert_params = {
        'Maxwellian3D_1': {
            'type': "ModesCos",
            'use_background_n': False,
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n1]},
                'amps': {'n': [amp_n1]},
            },
        },
        'Maxwellian3D_2': {
            'type': "ModesCos",
            'use_background_n': True,
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [l_n2]},
                'amps': {'n': [amp_n2]},
            },
        },
    }

    particles = Particles6D(
        'energetic_ions', **marker_params,
        bckgr_params=bckgr_params,
        pert_params=pert_params,
        derham=derham,
    )
    particles.draw_markers()
    particles.initialize_weights()

    e1_bins = np.linspace(0., 1., 200, endpoint=True)
    de = e1_bins[1] - e1_bins[0]

    binned_res, r2 = particles.binning(
        [True, False, False, False, False, False],
        [e1_bins],
    )

    # Reduce all threads to get complete result
    mpi_res = np.zeros_like(binned_res)
    comm.Allreduce(binned_res, mpi_res, op=MPI.SUM)
    comm.Barrier()

    e1_plot = e1_bins[:-1] + de/2

    ana_res = amp_n1 * np.cos(2*np.pi * l_n1 * e1_plot) + \
        n2 + amp_n2 * np.cos(2*np.pi * l_n2 * e1_plot)

    # Compare s0 and the sum of two Maxwellians
    if show_plot and rank == 0:
        s0_dict = {
            'n': 1.,
            'u1': particles.marker_params['loading']['moments'][0],
            'u2': particles.marker_params['loading']['moments'][1],
            'u3': particles.marker_params['loading']['moments'][2],
            'vth1': particles.marker_params['loading']['moments'][3],
            'vth2': particles.marker_params['loading']['moments'][4],
            'vth3': particles.marker_params['loading']['moments'][5],
        }
        s0 = Maxwellian3D(maxw_params=s0_dict)

        v1 = np.linspace(-10., 10., 400)
        phase_space = np.meshgrid(
            np.array([0.]),
            np.array([0.]),
            np.array([0.]),
            v1,
            np.array([0.]),
            np.array([0.]),
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
        plt.plot(e1_plot, mpi_res, 'r*', label="From binning")
        plt.title(r"$\delta f$ with MPI: Two backgrounds with cosines in $\eta_1$-direction")
        plt.xlabel(r'$\eta_1$')
        plt.ylabel(r'$f(\eta_1)$')
        plt.legend()
        plt.show()

    l2_error = np.sqrt(np.sum((ana_res - mpi_res)**2)) / \
        np.sqrt(np.sum((ana_res)**2))

    assert l2_error <= 0.04, \
        f"Error between binned data and analytical result was {l2_error}"


if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size == 1:
        test_binning_6D_full_f(
            Nel=[24, 1, 1],
            p=[3, 1, 1],
            spl_kind=[True, False, False],
            mapping=[
                'Cuboid',
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 10., 'r3': 20.},
                # 'ShafranovDshapedCylinder',
                # {'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07,
                #     'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}
            ],
            show_plot=True,
        )
        test_binning_6D_delta_f(
            Nel=[24, 1, 1],
            p=[3, 1, 1],
            spl_kind=[True, False, False],
            mapping=[
                'Cuboid',
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 10., 'r3': 20.},
            ],
            show_plot=True,
        )
    else:
        test_binning_6D_full_f_mpi(
            Nel=[24, 1, 1],
            p=[3, 1, 1],
            spl_kind=[True, False, False],
            mapping=[
                'Cuboid',
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 10., 'r3': 20.},
                # 'ShafranovDshapedCylinder',
                # {'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07,
                #     'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}
            ],
            show_plot=True,
        )
        test_binning_6D_delta_f_mpi(
            Nel=[24, 1, 1],
            p=[3, 1, 1],
            spl_kind=[True, False, False],
            mapping=[
                'Cuboid',
                # {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
                {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 10., 'r3': 20.},
            ],
            show_plot=True,
        )
