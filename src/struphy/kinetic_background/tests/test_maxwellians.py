import pytest


@pytest.mark.parametrize('Nel', [[64, 1, 1]])
def test_maxwellian_6d_uniform(Nel, show_plot=False):
    """ Tests the Maxwellian6D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian6D

    e1 = np.linspace(0., 1., Nel[0])
    e2 = np.linspace(0., 1., Nel[1])
    e3 = np.linspace(0., 1., Nel[2])

    # ==========================================================
    # ==== Test uniform non-shifted, isothermal Maxwellian =====
    # ==========================================================
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': 2.
            }
        },
        'perturbation': {
            'type': None
        }
    }

    maxwellian = Maxwellian6D(**params)

    meshgrids = np.meshgrid(
        e1, e2, e3,
        [0.], [0.], [0.]
    )

    # Test constant value at v=0
    res = maxwellian(*meshgrids).squeeze()
    assert np.allclose(
        res,
        2. / (2 * np.pi)**(3/2) + 0*e1,
        atol=10e-10
    ), f"{res=},\n {2. / (2 * np.pi)**(3/2)}"

    # test Maxwellian profile in v
    v1 = np.linspace(-5, 5, 128)
    meshgrids = np.meshgrid(
        [0.], [0.], [0.],
        v1, [0.], [0.],
    )
    res = maxwellian(*meshgrids).squeeze()
    res_ana = 2. * np.exp(- v1**2 / 2.) / (2 * np.pi)**(3/2)
    assert np.allclose(
        res,
        res_ana,
        atol=10e-10
    ), f"{res=},\n {res_ana}"

    # =======================================================
    # ===== Test non-zero shifts and thermal velocities =====
    # =======================================================
    n = 2.
    u1 = 1.
    u2 = -0.2
    u3 = 0.1
    vth1 = 1.2
    vth2 = 0.5
    vth3 = 0.3
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': n,
                    'u1': u1,
                    'u2': u2,
                    'u3': u3,
                    'vth1': vth1,
                    'vth2': vth2,
                    'vth3': vth3,
            }
        }
    }

    maxwellian = Maxwellian6D(**params)

    # test Maxwellian profile in v
    for i in range(3):
        vs = [0, 0, 0]
        vs[i] = np.linspace(-5, 5, 128)
        meshgrids = np.meshgrid(
            [0.], [0.], [0.],
            *vs
        )
        res = maxwellian(*meshgrids).squeeze()

        res_ana = np.exp(- (vs[0] - u1)**2 / (2*vth1**2))
        res_ana *= np.exp(- (vs[1] - u2)**2 / (2*vth2**2))
        res_ana *= np.exp(- (vs[2] - u3)**2 / (2*vth3**2))
        res_ana *= n / ((2 * np.pi)**(3/2) * vth1 * vth2 * vth3)

        if show_plot:
            plt.plot(vs[i], res_ana, label='analytical')
            plt.plot(vs[i], res, 'r*', label='Maxwellian class')
            plt.legend()
            plt.title("Test non-zero shifts and thermal velocities")
            plt.ylabel('f(v_' + str(i+1) + ')')
            plt.xlabel('v_' + str(i+1))
            plt.show()

        assert np.allclose(
            res,
            res_ana,
            atol=10e-10
        ), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize('Nel', [[64, 1, 1]])
def test_maxwellian_6d_perturbed(Nel, show_plot=False):
    '''Tests the Maxwellian6D class for perturbations.'''

    import numpy as np
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian6D

    e1 = np.linspace(0., 1., Nel[0])
    v1 = np.linspace(-5., 5., 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': 2.
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [mode]},
                'amps': {'n': [amp]}
            }
        }
    }

    maxwellian = Maxwellian6D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        [0.], [0.], [0.]
    )

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2. + amp * np.cos(2 * np.pi * mode * e1)) / (2 * np.pi)**(3/2)

    if show_plot:
        plt.plot(e1, ana_res, label='analytical')
        plt.plot(e1, res, 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')
        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test cosine perturbation in shift =====
    # =============================================
    amp = 0.1
    mode = 1
    n = 2.
    u1 = 1.2
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': n,
                    'u1': u1,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'u1': '0'},
                'ls': {'u1': [mode]},
                'amps': {'u1': [amp]}
            }
        }
    }

    maxwellian = Maxwellian6D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        v1, [0.], [0.],
    )

    res = maxwellian(*meshgrids).squeeze()
    shift = u1 + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- (v1 - shift[:, None])**2 / 2)
    ana_res *= n / (2 * np.pi)**(3/2)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label='analytical')
        plt.plot(e1, res[:, 0], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift")
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ===========================================
    # ===== Test cosine perturbation in vth =====
    # ===========================================
    amp = 0.1
    mode = 1
    n = 2.
    vth1 = 1.2
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': n,
                    'vth1': vth1,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'vth1': '0'},
                'ls': {'vth1': [mode]},
                'amps': {'vth1': [amp]}
            }
        }
    }

    maxwellian = Maxwellian6D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        v1, [0.], [0.],
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth1 + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- v1**2 / (2. * thermal[:, None]**2))
    ana_res *= n / ((2 * np.pi)**(3/2) * thermal[:, None])

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label='analytical')
        plt.plot(e1, res[:, 0], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth")
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.491230, 0.298228, 0.198739, 0.521298]
    params = {
        'background': {
            'type': 'Maxwellian6D',
            'Maxwellian6D': {
                    'n': 0.
            }
        },
        'perturbation': {
            'type': 'ITPA_density',
            'ITPA_density': {
                'comps': {'n': '0'},
                'n0': {'n': n0},
                'c': {'n': c}
            }
        }
    }

    maxwellian = Maxwellian6D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        [0.], [0.], [0.]
    )

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0 * c[3] * np.exp(
        -c[2]/c[1] * np.tanh((e1 - c[0])/c[2])
    ) / (2 * np.pi)**(3/2)

    if show_plot:
        plt.plot(e1, ana_res, label='analytical')
        plt.plot(e1, res, 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')
        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"


@pytest.mark.parametrize('Nel', [[64, 1, 1]])
def test_maxwellian_5d_uniform(Nel, show_plot=False):
    """ Tests the Maxwellian5D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian5D

    e1 = np.linspace(0., 1., Nel[0])
    e2 = np.linspace(0., 1., Nel[1])
    e3 = np.linspace(0., 1., Nel[2])

    # ===========================================================
    # ===== Test uniform non-shifted, isothermal Maxwellian =====
    # ===========================================================
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': 2.
            }
        },
        'perturbation': {
            'type': None
        }
    }

    maxwellian = Maxwellian5D(**params)

    meshgrids = np.meshgrid(
        e1, e2, e3,
        [0.01], [0.01]
    )

    # Test constant value at v_para = v_perp = 0.01
    res = maxwellian(*meshgrids).squeeze()
    assert np.allclose(
        res,
        2. / (2 * np.pi)**(1/2) * 0.01 * np.exp(- 0.01**2) + 0*e1,
        atol=10e-10
    ), f"{res=},\n {2. / (2 * np.pi)**(3/2)}"

    # test Maxwellian profile in v
    v_para = np.linspace(-5, 5, 64)
    v_perp = np.linspace(0, 2.5, 64)
    vpara, vperp = np.meshgrid(v_para, v_perp)

    meshgrids = np.meshgrid(
        [0.], [0.], [0.],
        v_para, v_perp,
    )
    res = maxwellian(*meshgrids).squeeze()

    res_ana = 2. / (2 * np.pi)**(1/2) * v_perp * \
        np.exp(- vpara.T**2 / 2. - vperp.T**2 / 2.)
    assert np.allclose(
        res,
        res_ana,
        atol=10e-10
    ), f"{res=},\n {res_ana}"

    # =======================================================
    # ===== Test non-zero shifts and thermal velocities =====
    # =======================================================
    n = 2.
    u_para = 0.1
    u_perp = 0.2
    vth_para = 1.2
    vth_perp = 0.5
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': n,
                    'u_para': u_para,
                    'u_perp': u_perp,
                    'vth_para': vth_para,
                    'vth_perp': vth_perp,
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    # test Maxwellian profile in v
    v_para = np.linspace(-5, 5, 64)
    v_perp = np.linspace(0, 2.5, 64)
    vpara, vperp = np.meshgrid(v_para, v_perp)

    meshgrids = np.meshgrid(
        [0.], [0.], [0.],
        v_para, v_perp
    )
    res = maxwellian(*meshgrids).squeeze()

    res_ana = np.exp(- (vpara.T - u_para)**2 / (2*vth_para**2))
    res_ana *= np.exp(- (vperp.T - u_perp)**2 / (2*vth_perp**2))
    res_ana *= n / ((2 * np.pi)**(1/2) * vth_para * vth_perp**2) * vperp.T

    if show_plot:
        plt.plot(v_para, res_ana[:, 32], label='analytical')
        plt.plot(v_para, res[:, 32], 'r*', label='Maxwellian class')
        plt.legend()
        plt.title("Test non-zero shifts and thermal velocities")
        plt.ylabel('f(v_' + 'para' + ')')
        plt.xlabel('v_' + 'para')
        plt.show()

        plt.plot(v_perp, res_ana[32, :], label='analytical')
        plt.plot(v_perp, res[32, :], 'r*', label='Maxwellian class')
        plt.legend()
        plt.title("Test non-zero shifts and thermal velocities")
        plt.ylabel('f(v_' + 'perp' + ')')
        plt.xlabel('v_' + 'perp')
        plt.show()

    assert np.allclose(
        res,
        res_ana,
        atol=10e-10
    ), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize('Nel', [[6, 1, 1]])
def test_maxwellian_5d_perturbed(Nel, show_plot=False):
    '''Tests the Maxwellian5D class for perturbations.'''

    import numpy as np
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian5D

    e1 = np.linspace(0., 1., Nel[0])
    v1 = np.linspace(-5., 5., 128)
    v2 = np.linspace(0, 2.5, 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': 2.
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'n': '0'},
                'ls': {'n': [mode]},
                'amps': {'n': [amp]}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    v_perp = 0.1
    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        [0.], v_perp
    )

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2. + amp * np.cos(2 * np.pi * mode * e1)) / (2 * np.pi)**(1/2)
    ana_res *= v_perp * np.exp(- v_perp**2/2)

    if show_plot:
        plt.plot(e1, ana_res, label='analytical')
        plt.plot(e1, res, 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')
        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ====================================================
    # ===== Test cosine perturbation in shift (para) =====
    # ====================================================
    amp = 0.1
    mode = 1
    n = 2.
    u_para = 1.2
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': n,
                    'u_para': u_para,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'u_para': '0'},
                'ls': {'u_para': [mode]},
                'amps': {'u_para': [amp]}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    v_perp = 0.1
    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        v1, v_perp
    )

    res = maxwellian(*meshgrids).squeeze()
    shift = u_para + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- (v1 - shift[:, None])**2 / 2.)
    ana_res *= n / (2 * np.pi)**(1/2) * v_perp * np.exp(- v_perp**2 / 2.)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 20], label='analytical')
        plt.plot(e1, res[:, 20], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift (para)")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift (para)")
        plt.xlabel('v_para')
        plt.ylabel('f(v_para)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ====================================================
    # ===== Test cosine perturbation in shift (perp) =====
    # ====================================================
    amp = 0.1
    mode = 1
    n = 2.
    u_perp = 1.2
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': n,
                    'u_perp': u_perp,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'u_perp': '0'},
                'ls': {'u_perp': [mode]},
                'amps': {'u_perp': [amp]}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        0., v2
    )

    res = maxwellian(*meshgrids).squeeze()
    shift = u_perp + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- (v2 - shift[:, None])**2 / 2.)
    ana_res *= n / (2 * np.pi)**(1/2) * v2

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 20], label='analytical')
        plt.plot(e1, res[:, 20], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift (perp)")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in shift (perp)")
        plt.xlabel('v_perp')
        plt.ylabel('f(v_perp)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (para) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.
    vth_para = 1.2
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': n,
                    'vth_para': vth_para,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'vth_para': '0'},
                'ls': {'vth_para': [mode]},
                'amps': {'vth_para': [amp]}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    v_perp = 0.1
    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        v1, v_perp,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_para + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- v1**2 / (2. * thermal[:, None]**2))
    ana_res *= n / ((2 * np.pi)**(1/2) * thermal[:, None])
    ana_res *= v_perp * np.exp(- v_perp**2 / 2.)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label='analytical')
        plt.plot(e1, res[:, 0], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth (para)")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth (para)")
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (perp) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.
    vth_perp = 1.2
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': n,
                    'vth_perp': vth_perp,
            }
        },
        'perturbation': {
            'type': 'ModesCos',
            'ModesCos': {
                'comps': {'vth_perp': '0'},
                'ls': {'vth_perp': [mode]},
                'amps': {'vth_perp': [amp]}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        0., v2,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_perp + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(- v2**2 / (2. * thermal[:, None]**2))
    ana_res *= n / ((2 * np.pi)**(1/2) * thermal[:, None]**2) * v2

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label='analytical')
        plt.plot(e1, res[:, 0], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth (perp)")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test cosine perturbation in vth (perp)")
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.491230, 0.298228, 0.198739, 0.521298]
    params = {
        'background': {
            'type': 'Maxwellian5D',
            'Maxwellian5D': {
                    'n': 0.
            }
        },
        'perturbation': {
            'type': 'ITPA_density',
            'ITPA_density': {
                'comps': {'n': '0'},
                'n0': {'n': n0},
                'c': {'n': c}
            }
        }
    }

    maxwellian = Maxwellian5D(**params)

    v_perp = 0.1
    meshgrids = np.meshgrid(
        e1, [0.], [0.],
        [0.], v_perp
    )

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0*c[3]*np.exp(-c[2]/c[1] *
                             np.tanh((e1 - c[0])/c[2])) / (2 * np.pi)**(1/2)
    ana_res *= v_perp * np.exp(- v_perp**2 / 2.)

    if show_plot:
        plt.plot(e1, ana_res, label='analytical')
        plt.plot(e1, res, 'r*', label='Maxwellian Class')
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')
        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"


if __name__ == '__main__':
    test_maxwellian_6d_uniform(Nel=[64, 1, 1], show_plot=True)
    test_maxwellian_6d_perturbed(Nel=[64, 1, 1], show_plot=True)
    test_maxwellian_5d_uniform(Nel=[64, 1, 1], show_plot=True)
    test_maxwellian_5d_perturbed(Nel=[64, 1, 1], show_plot=True)
