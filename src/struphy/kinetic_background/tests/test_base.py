import pytest


def test_kinetic_background_magics(show_plot=False):
    """Test the magic commands __sum__, __mul__ and __sub__
    of the Maxwellian base class."""
    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian3D

    num_elements = [32, 1, 1]
    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])
    v1 = xp.linspace(-7.0, 7.0, 128)

    m1_params = {"n": 0.5, "u1": 3.0}
    m2_params = {"n": 0.5, "u1": -3.0}

    m1 = Maxwellian3D(n=(0.5, None), u1=(3.0, None))
    m2 = Maxwellian3D(n=(0.5, None), u1=(-3.0, None))

    m_add = m1 + m2
    m_rmul_int = 2 * m1
    m_mul_int = m1 * 2
    m_mul_float = 2.0 * m1
    m_mul_npint = xp.ones(1, dtype=int)[0] * m1
    m_sub = m1 - m2

    # compare distribution function
    meshgrids = xp.meshgrid(e1, e2, e3, v1, [0.0], [0.0])

    m1_vals = m1(*meshgrids)
    m2_vals = m2(*meshgrids)

    m_add_vals = m_add(*meshgrids)
    m_rmul_int_vals = m_rmul_int(*meshgrids)
    m_mul_int_vals = m_mul_int(*meshgrids)
    m_mul_float_vals = m_mul_float(*meshgrids)
    m_mul_npint_vals = m_mul_npint(*meshgrids)
    m_sub_vals = m_sub(*meshgrids)

    assert xp.allclose(m1_vals + m2_vals, m_add_vals)
    assert xp.allclose(2 * m1_vals, m_rmul_int_vals)
    assert xp.allclose(2 * m1_vals, m_mul_int_vals)
    assert xp.allclose(2.0 * m1_vals, m_mul_float_vals)
    assert xp.allclose(xp.ones(1, dtype=int)[0] * m1_vals, m_mul_npint_vals)
    assert xp.allclose(m1_vals - m2_vals, m_sub_vals)

    # compare first two moments
    meshgrids = xp.meshgrid(e1, e2, e3)

    n1_vals = m1.n(*meshgrids)
    n2_vals = m2.n(*meshgrids)
    u11, u12, u13 = m1.u(*meshgrids)
    u21, u22, u23 = m2.u(*meshgrids)

    n_add_vals = m_add.n(*meshgrids)
    u_add1, u_add2, u_add3 = m_add.u(*meshgrids)
    n_sub_vals = m_sub.n(*meshgrids)

    assert xp.allclose(n1_vals + n2_vals, n_add_vals)
    assert xp.allclose(u11 + u21, u_add1)
    assert xp.allclose(u12 + u22, u_add2)
    assert xp.allclose(u13 + u23, u_add3)
    assert xp.allclose(n1_vals - n2_vals, n_sub_vals)

    if show_plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 2, 1)
        plt.plot(v1, m1_vals[0, 0, 0, :, 0, 0])
        plt.title("M1")
        plt.subplot(3, 2, 3)
        plt.plot(v1, m2_vals[0, 0, 0, :, 0, 0])
        plt.title("M2")
        plt.subplot(3, 2, 5)
        plt.plot(v1, m_add_vals[0, 0, 0, :, 0, 0])
        plt.title("M1 + M2")
        plt.subplot(3, 2, 2)
        plt.plot(v1, m_mul_int_vals[0, 0, 0, :, 0, 0])
        plt.title("2 * M1")
        plt.subplot(3, 2, 6)
        plt.plot(v1, m_sub_vals[0, 0, 0, :, 0, 0])
        plt.title("M1 - M2")

        plt.show()


@pytest.mark.mpi_skip
def test_plotting_function():

    import cunumpy as xp

    from struphy import domains, equils, maxwellians

    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)
    equil.domain = domains.Cuboid()

    # definition of test functions
    l, m, n = 3, 4, 5

    def n_init(*etas):
        if len(etas) == 1:
            e1, e2, e3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 1]
        else:
            assert len(etas) == 3
            e1, e2, e3 = etas[0], etas[1], etas[2]
        return 1 + 0.5 * xp.cos(2 * xp.pi * e1 * l) * xp.cos(2 * xp.pi * e2 * m) * xp.cos(2 * xp.pi * e3 * n)

    def vth(*etas):
        if len(etas) == 1:
            e1, e2, e3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 1]
        else:
            assert len(etas) == 3
            e1, e2, e3 = etas[0], etas[1], etas[2]
        return 1 + 0.2 * xp.cos(2 * xp.pi * e1 * l) * xp.cos(2 * xp.pi * e2 * m) * xp.cos(2 * xp.pi * e3 * n)

    # Testing with GyroMaxwellian2Dvperp:
    background = maxwellians.GyroMaxwellian2Dvperp(
        n=(n_init, None), vth_para=(vth, None), vth_perp=(vth, None), equil=equil
    )
    background.plot_density_profile("e1")
    background.plot_density_profile("e2")
    background.plot_density_profile("e3")
    background.plot_density_profile("v1")
    background.plot_density_profile("v2")
    background.plot_density_profile("e1", "e2")
    background.plot_density_profile("e1", "e2", domain=domains.HollowCylinder(), proj_axis=(0, 1), in_physical=True)
    background.plot_density_profile("e1", "e2", domain=domains.HollowTorus(), proj_axis=(1, 2), in_physical=True)
    background.plot_density_profile(
        "e1", "e2", domain=domains.HollowTorus(), proj_axis=(0, 2), in_physical=True, plot_3D=True
    )
    background.plot_density_profile(
        "e2", "e3", domain=domains.HollowTorus(), proj_axis=(1, 2), in_physical=True, plot_3D=True
    )
    background.plot_density_profile("v1", "v2")
    background.plot_density_profile("v1", "v2", use_mu=True)

    # Testing with Maxwellian3D:
    background = maxwellians.Maxwellian3D(n=(n_init, None), vth1=(vth, None), vth2=(vth, None), vth3=(vth, None))
    background.plot_density_profile("v1", "v2")
    background.plot_density_profile("v1", "v3")
    background.plot_density_profile("e1", "v3")


if __name__ == "__main__":
    # test_kinetic_background_magics(show_plot=True)
    test_plotting_function()
