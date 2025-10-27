import importlib.util

import cunumpy as xp
import pytest
from matplotlib import pyplot as plt

desc_spec = importlib.util.find_spec("desc")


@pytest.mark.mpi_skip
@pytest.mark.skipif(desc_spec is None, reason="desc-opt not installed.")
def test_desc_equil(do_plot=False):
    """Test the workflow of creating a DESC mhd equilibirum and compares
    push forwards to native DESC results."""

    import desc
    from desc.grid import Grid

    from struphy.fields_background import base, equils

    # default case, with and without use of toroidal field periods
    desc_eq = desc.examples.get("W7-X")
    nfps = [1, desc_eq.NFP]
    rmin = 0.01

    struphy_eqs = {}
    for nfp in nfps:
        struphy_eqs[nfp] = equils.DESCequilibrium(use_nfp=nfp != 1)

    # grid
    n1 = 8
    n2 = 9
    n3 = 11

    e1 = xp.linspace(0.0001, 1, n1)
    e2 = xp.linspace(0, 1, n2)
    e3 = xp.linspace(0, 1 - 1e-6, n3)

    # desc grid and evaluation
    vars = [
        "X",
        "Y",
        "Z",
        "R",
        "phi",
        "sqrt(g)",
        "p",
        "B",
        "J",
        "B_R",
        "B_phi",
        "B_Z",
        "J_R",
        "J_phi",
        "J_Z",
        "B^rho",
        "B^theta",
        "B^zeta",
        "J^rho",
        "J^theta",
        "J^zeta",
        "|B|_r",
        "|B|_t",
        "|B|_z",
    ]

    outs = {}
    for nfp in nfps:
        outs[nfp] = {}

        rho = rmin + e1 * (1.0 - rmin)
        theta = 2 * xp.pi * e2
        zeta = 2 * xp.pi * e3 / nfp

        r, t, ze = xp.meshgrid(rho, theta, zeta, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        ze = ze.flatten()

        nodes = xp.stack((r, t, ze)).T
        grid_3d = Grid(nodes, spacing=xp.ones_like(nodes), jitable=False)

        for var in vars:
            node_values = desc_eq.compute(var, grid=grid_3d, override_grid=False)

            if node_values[var].ndim == 1:
                out = node_values[var].reshape((rho.size, theta.size, zeta.size), order="C")
                outs[nfp][var] = xp.ascontiguousarray(out)
            else:
                B = []
                for i in range(3):
                    Bcomp = node_values[var][:, i].reshape((rho.size, theta.size, zeta.size), order="C")
                    Bcomp = xp.ascontiguousarray(Bcomp)
                    B += [Bcomp]
                    outs[nfp][var + str(i + 1)] = Bcomp
                outs[nfp][var] = xp.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

        assert xp.allclose(outs[nfp]["B1"], outs[nfp]["B_R"])
        assert xp.allclose(outs[nfp]["B2"], outs[nfp]["B_phi"])
        assert xp.allclose(outs[nfp]["B3"], outs[nfp]["B_Z"])

        assert xp.allclose(outs[nfp]["J1"], outs[nfp]["J_R"])
        assert xp.allclose(outs[nfp]["J2"], outs[nfp]["J_phi"])
        assert xp.allclose(outs[nfp]["J3"], outs[nfp]["J_Z"])

        outs[nfp]["Bx"] = xp.cos(outs[nfp]["phi"]) * outs[nfp]["B_R"] - xp.sin(outs[nfp]["phi"]) * outs[nfp]["B_phi"]

        outs[nfp]["By"] = xp.sin(outs[nfp]["phi"]) * outs[nfp]["B_R"] + xp.cos(outs[nfp]["phi"]) * outs[nfp]["B_phi"]

        outs[nfp]["Bz"] = outs[nfp]["B_Z"]

    # struphy evaluation
    outs_struphy = {}
    for nfp in nfps:
        outs_struphy[nfp] = {}
        s_eq = struphy_eqs[nfp]

        assert isinstance(s_eq, base.MHDequilibrium)

        x, y, z = s_eq.domain(e1, e2, e3)
        outs_struphy[nfp]["X"] = x
        outs_struphy[nfp]["Y"] = y
        outs_struphy[nfp]["Z"] = z

        outs_struphy[nfp]["R"] = xp.sqrt(x**2 + y**2)
        tmp = xp.arctan2(y, x)
        tmp[tmp < -1e-6] += 2 * xp.pi
        outs_struphy[nfp]["phi"] = tmp

        outs_struphy[nfp]["sqrt(g)"] = s_eq.domain.jacobian_det(e1, e2, e3) / (4 * xp.pi**2 / nfp)

        outs_struphy[nfp]["p"] = s_eq.p0(e1, e2, e3)

        # include push forward to DESC logical coordinates
        bv = s_eq.bv(e1, e2, e3)
        outs_struphy[nfp]["B^rho"] = bv[0] * (1 - rmin)
        outs_struphy[nfp]["B^theta"] = bv[1] * 2 * xp.pi
        outs_struphy[nfp]["B^zeta"] = bv[2] * 2 * xp.pi / nfp

        outs_struphy[nfp]["B"] = s_eq.absB0(e1, e2, e3)

        # include push forward to DESC logical coordinates
        jv = s_eq.jv(e1, e2, e3)
        outs_struphy[nfp]["J^rho"] = jv[0] * (1 - rmin)
        outs_struphy[nfp]["J^theta"] = jv[1] * 2 * xp.pi
        outs_struphy[nfp]["J^zeta"] = jv[2] * 2 * xp.pi / nfp

        j1 = s_eq.j1(e1, e2, e3)

        outs_struphy[nfp]["J"] = xp.sqrt(jv[0] * j1[0] + jv[1] * j1[1] + jv[2] * j1[2])

        b_cart, xyz = s_eq.b_cart(e1, e2, e3)
        outs_struphy[nfp]["Bx"] = b_cart[0]
        outs_struphy[nfp]["By"] = b_cart[1]
        outs_struphy[nfp]["Bz"] = b_cart[2]

        # include push forward to DESC logical coordinates
        gradB1 = s_eq.gradB1(e1, e2, e3)
        outs_struphy[nfp]["|B|_r"] = gradB1[0] / (1 - rmin)
        outs_struphy[nfp]["|B|_t"] = gradB1[1] / (2 * xp.pi)
        outs_struphy[nfp]["|B|_z"] = gradB1[2] / (2 * xp.pi / nfp)

    # comparisons
    vars += ["Bx", "By", "Bz"]
    print(vars)

    err_lim = 0.09

    for nfp in nfps:
        print(f"\n{nfp =}")
        for var in vars:
            if var in ("B_R", "B_phi", "B_Z", "J_R", "J_phi", "J_Z"):
                continue
            else:
                max_norm = xp.max(xp.abs(outs[nfp][var]))
                if max_norm < 1e-16:
                    max_norm = 1.0
                err = xp.max(xp.abs(outs[nfp][var] - outs_struphy[nfp][var])) / max_norm

                assert err < err_lim
                print(
                    f"compare {var}: {err =}",
                )

                if do_plot:
                    fig = plt.figure(figsize=(12, 13))

                    levels = xp.linspace(xp.min(outs[nfp][var]) - 1e-10, xp.max(outs[nfp][var]), 20)

                    # poloidal plot
                    R = outs[nfp]["R"][:, :, 0].squeeze()
                    Z = outs[nfp]["Z"][:, :, 0].squeeze()

                    plt.subplot(2, 2, 1)
                    map1 = plt.contourf(R, Z, outs[nfp][var][:, :, 0], levels=levels)
                    plt.title(f"DESC, {var =}, {nfp =}")
                    plt.xlabel("$R$")
                    plt.ylabel("$Z$")
                    plt.axis("equal")
                    plt.colorbar(map1, location="right")

                    plt.subplot(2, 2, 2)
                    map2 = plt.contourf(R, Z, outs_struphy[nfp][var][:, :, 0], levels=levels)
                    plt.title(f"Struphy, {err =}")
                    plt.xlabel("$R$")
                    plt.ylabel("$Z$")
                    plt.axis("equal")
                    plt.colorbar(map2, location="right")

                    # top view plot
                    x1 = outs[nfp]["X"][:, 0, :].squeeze()
                    y1 = outs[nfp]["Y"][:, 0, :].squeeze()

                    x2 = outs[nfp]["X"][:, n2 // 2, :].squeeze()
                    y2 = outs[nfp]["Y"][:, n2 // 2, :].squeeze()

                    plt.subplot(2, 2, 3)
                    map3 = plt.contourf(x1, y1, outs[nfp][var][:, 0, :], levels=levels)
                    map3b = plt.contourf(x2, y2, outs[nfp][var][:, n2 // 2, :], levels=levels)
                    plt.title(f"DESC, {var =}, {nfp =}")
                    plt.xlabel("$x$")
                    plt.ylabel("$y$")
                    plt.axis("equal")
                    plt.colorbar(map3, location="right")

                    plt.subplot(2, 2, 4)
                    map4 = plt.contourf(x1, y1, outs_struphy[nfp][var][:, 0, :], levels=levels)
                    map4b = plt.contourf(x2, y2, outs_struphy[nfp][var][:, n2 // 2, :], levels=levels)
                    plt.title(f"Struphy, {err =}")
                    plt.xlabel("$x$")
                    plt.ylabel("$y$")
                    plt.axis("equal")
                    plt.colorbar(map4, location="right")

    if do_plot:
        plt.show()


if __name__ == "__main__":
    test_desc_equil(do_plot=True)
