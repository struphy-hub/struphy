import cunumpy as xp
import pytest
from matplotlib import pyplot as plt

from struphy.fields_background.generic import (
    GenericCartesianFluidEquilibrium,
    GenericCartesianFluidEquilibriumWithB,
)


def test_generic_equils(show=False):
    fun_vec = lambda x, y, z: (xp.cos(2 * xp.pi * x), xp.cos(2 * xp.pi * y), z)
    fun_n = lambda x, y, z: xp.exp(-((x - 1) ** 2) - (y) ** 2)
    fun_p = lambda x, y, z: x**2
    gen_eq = GenericCartesianFluidEquilibrium(
        u_xyz=fun_vec,
        p_xyz=fun_p,
        n_xyz=fun_n,
    )
    gen_eq_B = GenericCartesianFluidEquilibriumWithB(
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
