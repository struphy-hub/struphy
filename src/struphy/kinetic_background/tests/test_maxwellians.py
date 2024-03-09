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

    # ======================================================
    # ==== Test non-zero shifts and thermal velocities =====
    # ======================================================
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

    # ==============================================
    # ==== Test cosine perturbation in density =====
    # ==============================================
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
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')
        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"

    # ============================================
    # ==== Test cosine perturbation in shift =====
    # ============================================
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
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    # ==========================================
    # ==== Test cosine perturbation in vth =====
    # ==========================================
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
        plt.xlabel('eta_1')
        plt.ylabel('f(eta_1)')

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label='analytical')
        plt.plot(v1, res[0, :], 'r*', label='Maxwellian Class')
        plt.legend()
        plt.xlabel('v_1')
        plt.ylabel('f(v_1)')

        plt.show()

    assert np.allclose(
        res,
        ana_res,
        atol=10e-10
    ), f"{res=},\n {ana_res}"


if __name__ == '__main__':
    test_maxwellian_6d_uniform(Nel=[64, 1, 1], show_plot=True)
    test_maxwellian_6d_perturbed(Nel=[64, 1, 1], show_plot=True)
