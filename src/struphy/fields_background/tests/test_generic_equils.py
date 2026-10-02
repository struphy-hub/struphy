import cunumpy as xp
from matplotlib import pyplot as plt

from struphy import domains, equils
from struphy.fields_background.base import FluidEquilibriumWithB


def test_generic_equils(show=False):
    fun_vec = lambda x, y, z: (xp.cos(2 * xp.pi * x), xp.cos(2 * xp.pi * y), z)
    fun_n = lambda x, y, z: xp.exp(-((x - 1) ** 2) - (y) ** 2)
    fun_p = lambda x, y, z: x**2
    gen_eq = equils.GenericCartesianFluidEquilibrium(
        u_xyz=fun_vec,
        p_xyz=fun_p,
        n_xyz=fun_n,
    )
    gen_eq_B = equils.GenericCartesianFluidEquilibriumWithB(
        u_xyz=fun_vec,
        p_xyz=fun_p,
        n_xyz=fun_n,
        b_xyz=fun_vec,
        gradB_xyz=fun_vec,
    )

    x = xp.linspace(-3, 3, 32)
    y = xp.linspace(-4, 4, 32)
    z = 1.0
    xx, yy, zz = xp.meshgrid(x, y, z)

    # gen_eq
    assert all([xp.all(tmp == fun_i) for tmp, fun_i in zip(gen_eq.u_xyz(xx, yy, zz), fun_vec(xx, yy, zz))])
    assert xp.all(gen_eq.p_xyz(xx, yy, zz) == fun_p(xx, yy, zz))
    assert xp.all(gen_eq.n_xyz(xx, yy, zz) == fun_n(xx, yy, zz))

    # gen_eq_B
    assert all([xp.all(tmp == fun_i) for tmp, fun_i in zip(gen_eq_B.u_xyz(xx, yy, zz), fun_vec(xx, yy, zz))])
    assert xp.all(gen_eq_B.p_xyz(xx, yy, zz) == fun_p(xx, yy, zz))
    assert xp.all(gen_eq_B.n_xyz(xx, yy, zz) == fun_n(xx, yy, zz))
    assert all([xp.all(tmp == fun_i) for tmp, fun_i in zip(gen_eq_B.b_xyz(xx, yy, zz), fun_vec(xx, yy, zz))])
    assert all([xp.all(tmp == fun_i) for tmp, fun_i in zip(gen_eq_B.gradB_xyz(xx, yy, zz), fun_vec(xx, yy, zz))])

    # derived B-field methods of FluidEquilibriumWithB
    assert isinstance(gen_eq_B, FluidEquilibriumWithB)
    assert "b_xyz" in gen_eq_B.params and "gradB_xyz" in gen_eq_B.params
    gen_eq_B_const = equils.GenericCartesianFluidEquilibriumWithB(
        b_xyz=lambda x, y, z: (0.0 * x, 0.0 * x, 2.0 + 0.0 * x),
    )
    gen_eq_B_const.domain = domains.Cuboid(r1=2.0, r2=3.0, r3=4.0)
    e = xp.linspace(0.1, 0.9, 3)
    assert xp.allclose(gen_eq_B_const.absB0(e, e, e), 2.0)
    assert xp.allclose(gen_eq_B_const.b2_3(e, e, e), 12.0)
    assert xp.allclose(gen_eq_B_const.unit_b1_3(e, e, e), 4.0)
    assert xp.allclose(gen_eq_B_const.gradB1_1(e, e, e), 0.0)

    if show:
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 2, 1)
        plt.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            gen_eq.u_xyz(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0])[0],
        )
        plt.colorbar()
        plt.title("u_1")
        plt.subplot(3, 2, 3)
        plt.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            gen_eq.u_xyz(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0])[1],
        )
        plt.colorbar()
        plt.title("u_2")
        plt.subplot(3, 2, 5)
        plt.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            gen_eq.u_xyz(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0])[2],
        )
        plt.colorbar()
        plt.title("u_3")
        plt.subplot(3, 2, 2)
        plt.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            gen_eq.p_xyz(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0]),
        )
        plt.colorbar()
        plt.title("p")
        plt.subplot(3, 2, 4)
        plt.contourf(
            xx[:, :, 0],
            yy[:, :, 0],
            gen_eq.n_xyz(xx[:, :, 0], yy[:, :, 0], zz[:, :, 0]),
        )
        plt.colorbar()
        plt.title("n")

        plt.show()


if __name__ == "__main__":
    test_generic_equils(show=True)
