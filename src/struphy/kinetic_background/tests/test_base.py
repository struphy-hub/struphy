def test_kinetic_background_magics(show_plot=False):
    """Test the magic commands __sum__, __mul__ and __sub__
    of the Maxwellian base class."""
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian3D
    from struphy.utils.arrays import xp

    Nel = [32, 1, 1]
    e1 = xp.linspace(0.0, 1.0, Nel[0])
    e2 = xp.linspace(0.0, 1.0, Nel[1])
    e3 = xp.linspace(0.0, 1.0, Nel[2])
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


if __name__ == "__main__":
    test_kinetic_background_magics(show_plot=True)
