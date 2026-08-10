import logging

import pytest

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("num_elements", [[64, 1, 1]])
def test_maxwellian_3d_uniform(num_elements, show_plot=False):
    """Tests the Maxwellian3D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])

    # ==========================================================
    # ==== Test uniform non-shifted, isothermal Maxwellian =====
    # ==========================================================
    maxwellian = Maxwellian3D(n=(2.0, None))

    meshgrids = xp.meshgrid(e1, e2, e3, [0.0], [0.0], [0.0])

    # Test constant value at v=0
    res = maxwellian(*meshgrids).squeeze()
    assert xp.allclose(res, 2.0 / (2 * xp.pi) ** (3 / 2) + 0 * e1, atol=10e-10), (
        f"{res=},\n {2.0 / (2 * xp.pi) ** (3 / 2)}"
    )

    # test Maxwellian profile in v
    v1 = xp.linspace(-5, 5, 128)
    meshgrids = xp.meshgrid(
        [0.0],
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )
    res = maxwellian(*meshgrids).squeeze()
    res_ana = 2.0 * xp.exp(-(v1**2) / 2.0) / (2 * xp.pi) ** (3 / 2)
    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # =======================================================
    # ===== Test non-zero shifts and thermal velocities =====
    # =======================================================
    n = 2.0
    u1 = 1.0
    u2 = -0.2
    u3 = 0.1
    vth1 = 1.2
    vth2 = 0.5
    vth3 = 0.3

    maxwellian = Maxwellian3D(
        n=(2.0, None),
        u1=(1.0, None),
        u2=(-0.2, None),
        u3=(0.1, None),
        vth1=(1.2, None),
        vth2=(0.5, None),
        vth3=(0.3, None),
    )

    # test Maxwellian profile in v
    for i in range(3):
        vs = [0, 0, 0]
        vs[i] = xp.linspace(-5, 5, 128)
        meshgrids = xp.meshgrid([0.0], [0.0], [0.0], *vs)
        res = maxwellian(*meshgrids).squeeze()

        res_ana = xp.exp(-((vs[0] - u1) ** 2) / (2 * vth1**2))
        res_ana *= xp.exp(-((vs[1] - u2) ** 2) / (2 * vth2**2))
        res_ana *= xp.exp(-((vs[2] - u3) ** 2) / (2 * vth3**2))
        res_ana *= n / ((2 * xp.pi) ** (3 / 2) * vth1 * vth2 * vth3)

        if show_plot:
            plt.plot(vs[i], res_ana, label="analytical")
            plt.plot(vs[i], res, "r*", label="Maxwellian class")
            plt.legend()
            plt.title("Test non-zero shifts and thermal velocities")
            plt.ylabel("f(v_" + str(i + 1) + ")")
            plt.xlabel("v_" + str(i + 1))
            plt.show()

        assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize("num_elements", [[64, 1, 1]])
def test_maxwellian_3d_perturbed(num_elements, show_plot=False):
    """Tests the Maxwellian3D class for perturbations."""

    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    v1 = xp.linspace(-5.0, 5.0, 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(2.0, pert))

    meshgrids = xp.meshgrid(e1, [0.0], [0.0], [0.0], [0.0], [0.0])

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2.0 + amp * xp.cos(2 * xp.pi * mode * e1)) / (2 * xp.pi) ** (3 / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test cosine perturbation in shift =====
    # =============================================
    amp = 0.1
    mode = 1
    n = 2.0
    u1 = 1.2

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(n, None), u1=(u1, pert))

    meshgrids = xp.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )

    res = maxwellian(*meshgrids).squeeze()
    shift = u1 + amp * xp.cos(2 * xp.pi * mode * e1)
    ana_res = xp.exp(-((v1 - shift[:, None]) ** 2) / 2)
    ana_res *= n / (2 * xp.pi) ** (3 / 2)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label="analytical")
        plt.plot(e1, res[:, 0], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift")
        plt.xlabel("v_1")
        plt.ylabel("f(v_1)")

        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ===========================================
    # ===== Test cosine perturbation in vth =====
    # ===========================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth1 = 1.2

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(n, None), vth1=(vth1, pert))

    meshgrids = xp.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth1 + amp * xp.cos(2 * xp.pi * mode * e1)
    ana_res = xp.exp(-(v1**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * xp.pi) ** (3 / 2) * thermal[:, None])

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label="analytical")
        plt.plot(e1, res[:, 0], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth")
        plt.xlabel("v_1")
        plt.ylabel("f(v_1)")

        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = (0.491230, 0.298228, 0.198739, 0.521298)

    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian = Maxwellian3D(n=(0.0, pert))

    meshgrids = xp.meshgrid(e1, [0.0], [0.0], [0.0], [0.0], [0.0])

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0 * c[3] * xp.exp(-c[2] / c[1] * xp.tanh((e1 - c[0]) / c[2])) / (2 * xp.pi) ** (3 / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"


@pytest.mark.parametrize("num_elements", [[8, 11, 12]])
def test_maxwellian_3d_mhd(num_elements, with_desc, show_plot=False):
    """Tests the Maxwellian3D class for mhd equilibrium moments."""

    import inspect

    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy import domains, equils, perturbations
    from struphy.fields_background.base import FluidEquilibrium
    from struphy.initial.base import Perturbation
    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])
    v1 = [0.0]
    v2 = [0.0, -1.0]
    v3 = [0.0, -1.0, -1.3]

    meshgrids = xp.meshgrid(e1, e2, e3, v1, v2, v3, indexing="ij")
    e_meshgrids = xp.meshgrid(e1, e2, e3, indexing="ij")

    n_mks = 17
    e1_fl = xp.random.rand(n_mks)
    e2_fl = xp.random.rand(n_mks)
    e3_fl = xp.random.rand(n_mks)
    v1_fl = xp.random.randn(n_mks)
    v2_fl = xp.random.randn(n_mks)
    v3_fl = xp.random.randn(n_mks)
    args_fl = [e1_fl, e2_fl, e3_fl, v1_fl, v2_fl, v3_fl]
    e_args_fl = xp.concatenate((e1_fl[:, None], e2_fl[:, None], e3_fl[:, None]), axis=1)

    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and val.__module__ == equils.__name__:
            logger.info(f"{key =}")

            if "DESCequilibrium" in key and not with_desc:
                logger.info(f"Attention: {with_desc =}, DESC not tested here !!")
                continue

            if "GVECequilibrium" in key:
                logger.info("Attention: GVEC not tested here !!")
                # logger.info("Attention: flat (marker) evaluation not tested for GVEC at the moment.")
                continue

            mhd_equil = val()
            assert isinstance(mhd_equil, FluidEquilibrium)
            logger.info(f"{mhd_equil.params =}")
            if "AdhocTorus" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    R0=mhd_equil.params["R0"],
                    tor_period=1,
                )
            elif "EQDSKequilibrium" in key:
                mhd_equil.domain = domains.Tokamak(equilibrium=mhd_equil)
            elif "CircularTokamak" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    R0=mhd_equil.params["R0"],
                    tor_period=1,
                )
            elif "HomogenSlab" in key:
                mhd_equil.domain = domains.Cuboid()
            elif "ShearedSlab" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["a"] * 2 * xp.pi,
                    r3=mhd_equil.params["R0"] * 2 * xp.pi,
                )
            elif "ShearFluid" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["b"],
                    r3=mhd_equil.params["c"],
                )
            elif "ScrewPinch" in key:
                mhd_equil.domain = domains.HollowCylinder(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    Lz=mhd_equil.params["R0"] * 2 * xp.pi,
                )
            else:
                try:
                    mhd_equil.domain = domains.Cuboid()
                except:
                    logger.info(f"Not setting domain for {key}.")

            maxwellian = Maxwellian3D(
                n=(mhd_equil.n0, None),
                u1=(mhd_equil.u_cart_1, None),
                u2=(mhd_equil.u_cart_2, None),
                u3=(mhd_equil.u_cart_3, None),
                vth1=(mhd_equil.vth0, None),
                vth2=(mhd_equil.vth0, None),
                vth3=(mhd_equil.vth0, None),
            )

            maxwellian_1 = Maxwellian3D(
                n=(1.0, None),
                u1=(mhd_equil.u_cart_1, None),
                u2=(mhd_equil.u_cart_2, None),
                u3=(mhd_equil.u_cart_3, None),
                vth1=(mhd_equil.vth0, None),
                vth2=(mhd_equil.vth0, None),
                vth3=(mhd_equil.vth0, None),
            )

            # test meshgrid evaluation
            n0 = mhd_equil.n0(*e_meshgrids)
            assert xp.allclose(
                maxwellian(*meshgrids)[:, :, :, 0, 0, 0],
                n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 0, 0],
            )

            assert xp.allclose(
                maxwellian(*meshgrids)[:, :, :, 0, 1, 2],
                n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 1, 2],
            )

            # test flat evaluation
            if "GVECequilibrium" in key:
                logger.info("Attention: GVEC not tested here !!")
                # logger.info("Attention: flat (marker) evaluation not tested for GVEC at the moment.")
                continue
                pass
            else:
                assert xp.allclose(maxwellian(*args_fl), mhd_equil.n0(e_args_fl) * maxwellian_1(*args_fl))
                assert xp.allclose(maxwellian.n(e1_fl, e2_fl, e3_fl), mhd_equil.n0(e_args_fl))

                u_maxw = maxwellian.u(e1_fl, e2_fl, e3_fl)
                u_eq = mhd_equil.u_cart(e_args_fl)[0]
                assert all([xp.allclose(m, e) for m, e in zip(u_maxw, u_eq)])

                vth_maxw = maxwellian.vth(e1_fl, e2_fl, e3_fl)
                vth_eq = xp.sqrt(mhd_equil.p0(e_args_fl) / mhd_equil.n0(e_args_fl))
                assert all([xp.allclose(v, vth_eq) for v in vth_maxw])

            # plotting moments
            if show_plot:
                plt.figure(f"{mhd_equil =}", figsize=(24, 16))
                x, y, z = mhd_equil.domain(*e_meshgrids)

                # density plots
                n_cart = mhd_equil.domain.push(maxwellian.n, *e_meshgrids)

                levels = xp.linspace(xp.min(n_cart) - 1e-10, xp.max(n_cart), 20)

                plt.subplot(2, 5, 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        z[:, num_elements[1] // 2 - 1, :],
                        n_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        y[:, num_elements[1] // 2 - 1, :],
                        n_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian density $n$, top view (e1-e3)")
                plt.subplot(2, 5, 5 + 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0], n_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0], n_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian density $n$, poloidal view (e1-e2)")

                # velocity plots
                us = maxwellian.u(*e_meshgrids)
                for i, u in enumerate(us):
                    levels = xp.linspace(xp.min(u) - 1e-10, xp.max(u), 20)

                    plt.subplot(2, 5, 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(
                            x[:, num_elements[1] // 2, :],
                            z[:, num_elements[1] // 2, :],
                            u[:, num_elements[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(
                            x[:, num_elements[1] // 2, :],
                            y[:, num_elements[1] // 2, :],
                            u[:, num_elements[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Maxwellian velocity $u_{i + 1}$, top view (e1-e3)")
                    plt.subplot(2, 5, 5 + 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0], u[:, :, 0], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0], u[:, :, 0], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Maxwellian velocity $u_{i + 1}$, poloidal view (e1-e2)")

                # thermal velocity plots
                vth = maxwellian.vth(*e_meshgrids)[0]
                vth_cart = mhd_equil.domain.push(vth, *e_meshgrids)

                levels = xp.linspace(xp.min(vth_cart) - 1e-10, xp.max(vth_cart), 20)

                plt.subplot(2, 5, 5)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        z[:, num_elements[1] // 2 - 1, :],
                        vth_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        y[:, num_elements[1] // 2 - 1, :],
                        vth_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian thermal velocity $v_t$, top view (e1-e3)")
                plt.subplot(2, 5, 10)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0], vth_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0], vth_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian thermal velocity $v_t$, poloidal view (e1-e2)")

                plt.show()

            # test perturbations
            if "EQDSKequilibrium" in key:
                for key_2, val_2 in inspect.getmembers(perturbations):
                    if inspect.isclass(val_2) and val_2.__module__ == perturbations.__name__:
                        pert = val_2()
                        assert isinstance(pert, Perturbation)
                        logger.info(f"{pert =}")
                        if isinstance(pert, perturbations.Noise):
                            continue

                        # background + perturbation
                        maxwellian_perturbed = Maxwellian3D(
                            n=(mhd_equil.n0, pert),
                            u1=(mhd_equil.u_cart_1, pert),
                            u2=(mhd_equil.u_cart_2, pert),
                            u3=(mhd_equil.u_cart_3, pert),
                            vth1=(mhd_equil.vth0, pert),
                            vth2=(mhd_equil.vth0, pert),
                            vth3=(mhd_equil.vth0, pert),
                        )

                        # test meshgrid evaluation
                        assert maxwellian_perturbed(*meshgrids).shape == meshgrids[0].shape

                        # test flat evaluation
                        assert maxwellian_perturbed(*args_fl).shape == args_fl[0].shape

                        # pure perturbation
                        maxwellian_zero_bckgr = Maxwellian3D(
                            n=(0.0, pert),
                            u1=(0.0, pert),
                            u2=(0.0, pert),
                            u3=(0.0, pert),
                            vth1=(0.0, pert),
                            vth2=(0.0, pert),
                            vth3=(0.0, pert),
                        )

                        assert xp.allclose(maxwellian_zero_bckgr.n(*e_meshgrids), pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[2], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[2], pert(*e_meshgrids))

                        # plotting perturbations
                        if show_plot:  # and 'Torus' in key_2:
                            plt.figure(f"perturbation = {key_2}", figsize=(24, 16))
                            x, y, z = mhd_equil.domain(*e_meshgrids)

                            # density plots
                            n_cart = mhd_equil.domain.push(maxwellian_zero_bckgr.n, *e_meshgrids)

                            levels = xp.linspace(xp.min(n_cart) - 1e-10, xp.max(n_cart), 20)

                            plt.subplot(2, 5, 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    z[:, num_elements[1] // 2, :],
                                    n_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    y[:, num_elements[1] // 2, :],
                                    n_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed density $n$, top view (e1-e3)")
                            plt.subplot(2, 5, 5 + 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, :, 0], y[:, :, 0], n_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, :, 0], z[:, :, 0], n_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed density $n$, poloidal view (e1-e2)")

                            # velocity plots
                            us = maxwellian_zero_bckgr.u(*e_meshgrids)
                            for i, u in enumerate(us):
                                levels = xp.linspace(xp.min(u) - 1e-10, xp.max(u), 20)

                                plt.subplot(2, 5, 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, num_elements[1] // 2, :],
                                        z[:, num_elements[1] // 2, :],
                                        u[:, num_elements[1] // 2, :],
                                        levels=levels,
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                else:
                                    plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, num_elements[1] // 2, :],
                                        y[:, num_elements[1] // 2, :],
                                        u[:, num_elements[1] // 2, :],
                                        levels=levels,
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                plt.axis("equal")
                                plt.colorbar()
                                plt.title(f"Maxwellian perturbed velocity $u_{i + 1}$, top view (e1-e3)")
                                plt.subplot(2, 5, 5 + 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, :, 0], y[:, :, 0], u[:, :, 0], levels=levels)
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                else:
                                    plt.contourf(x[:, :, 0], z[:, :, 0], u[:, :, 0], levels=levels)
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                plt.axis("equal")
                                plt.colorbar()
                                plt.title(f"Maxwellian perturbed velocity $u_{i + 1}$, poloidal view (e1-e2)")

                            # thermal velocity plots
                            vth = maxwellian_zero_bckgr.vth(*e_meshgrids)[0]
                            vth_cart = mhd_equil.domain.push(vth, *e_meshgrids)

                            levels = xp.linspace(xp.min(vth_cart) - 1e-10, xp.max(vth_cart), 20)

                            plt.subplot(2, 5, 5)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    z[:, num_elements[1] // 2, :],
                                    vth_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    y[:, num_elements[1] // 2, :],
                                    vth_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed thermal velocity $v_t$, top view (e1-e3)")
                            plt.subplot(2, 5, 10)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, :, 0], y[:, :, 0], vth_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, :, 0], z[:, :, 0], vth_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed thermal velocity $v_t$, poloidal view (e1-e2)")

                            plt.show()


@pytest.mark.parametrize("num_elements", [[64, 1, 1]])
def test_maxwellian_2d_uniform(num_elements, show_plot=False):
    """Tests the GyroMaxwellian2D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy.kinetic_background.maxwellians import GyroMaxwellian2Dvperp

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])

    # ===========================================================
    # ===== Test uniform non-shifted, isothermal Maxwellian =====
    # ===========================================================
    maxwellian = GyroMaxwellian2Dvperp(n=(2.0, None), volume_form=False)

    meshgrids = xp.meshgrid(e1, e2, e3, [0.01], [0.01])

    # Test constant value at v_para = v_perp = 0.01
    res = maxwellian(*meshgrids).squeeze()
    assert xp.allclose(res, 2.0 / (2 * xp.pi) ** (1 / 2) * xp.exp(-(0.01**2)) + 0 * e1, atol=10e-10), (
        f"{res=},\n {2.0 / (2 * xp.pi) ** (3 / 2)}"
    )

    # test Maxwellian profile in v
    v_para = xp.linspace(-5, 5, 64)
    v_perp = xp.linspace(0, 2.5, 64)
    vpara, vperp = xp.meshgrid(v_para, v_perp)

    meshgrids = xp.meshgrid(
        [0.0],
        [0.0],
        [0.0],
        v_para,
        v_perp,
    )
    res = maxwellian(*meshgrids).squeeze()

    res_ana = 2.0 / (2 * xp.pi) ** (1 / 2) * xp.exp(-(vpara.T**2) / 2.0 - vperp.T**2 / 2.0)
    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # =======================================================
    # ===== Test non-zero shifts and thermal velocities =====
    # =======================================================
    n = 2.0
    u_para = 0.1
    u_perp = 0.0
    vth_para = 1.2
    vth_perp = 0.5

    maxwellian = GyroMaxwellian2Dvperp(
        n=(n, None),
        u_para=(u_para, None),
        u_perp=(u_perp, None),
        vth_para=(vth_para, None),
        vth_perp=(vth_perp, None),
        volume_form=False,
    )

    # test Maxwellian profile in v
    v_para = xp.linspace(-5, 5, 64)
    v_perp = xp.linspace(0, 2.5, 64)
    vpara, vperp = xp.meshgrid(v_para, v_perp)

    meshgrids = xp.meshgrid([0.0], [0.0], [0.0], v_para, v_perp)
    res = maxwellian(*meshgrids).squeeze()

    res_ana = xp.exp(-((vpara.T - u_para) ** 2) / (2 * vth_para**2))
    res_ana *= xp.exp(-((vperp.T - u_perp) ** 2) / (2 * vth_perp**2))
    res_ana *= n / ((2 * xp.pi) ** (1 / 2) * vth_para * vth_perp**2)

    if show_plot:
        plt.plot(v_para, res_ana[:, 32], label="analytical")
        plt.plot(v_para, res[:, 32], "r*", label="Maxwellian class")
        plt.legend()
        plt.title("Test non-zero shifts and thermal velocities")
        plt.ylabel("f(v_" + "para" + ")")
        plt.xlabel("v_" + "para")
        plt.show()

        plt.plot(v_perp, res_ana[32, :], label="analytical")
        plt.plot(v_perp, res[32, :], "r*", label="Maxwellian class")
        plt.legend()
        plt.title("Test non-zero shifts and thermal velocities")
        plt.ylabel("f(v_" + "perp" + ")")
        plt.xlabel("v_" + "perp")
        plt.show()

    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize("num_elements", [[6, 1, 1]])
def test_maxwellian_2d_perturbed(num_elements, show_plot=False):
    """Tests the GyroMaxwellian2D class for perturbations."""

    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy import perturbations
    from struphy.kinetic_background.maxwellians import GyroMaxwellian2Dvperp

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    v1 = xp.linspace(-5.0, 5.0, 128)
    v2 = xp.linspace(0, 2.5, 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2Dvperp(n=(2.0, pert), volume_form=False)

    v_perp = 0.1
    meshgrids = xp.meshgrid(e1, [0.0], [0.0], [0.0], v_perp)

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2.0 + amp * xp.cos(2 * xp.pi * mode * e1)) / (2 * xp.pi) ** (1 / 2)
    ana_res *= xp.exp(-(v_perp**2) / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ====================================================
    # ===== Test cosine perturbation in shift (para) =====
    # ====================================================
    amp = 0.1
    mode = 1
    n = 2.0
    u_para = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2Dvperp(
        n=(2.0, None),
        u_para=(u_para, pert),
        volume_form=False,
    )

    v_perp = 0.1
    meshgrids = xp.meshgrid(e1, [0.0], [0.0], v1, v_perp)

    res = maxwellian(*meshgrids).squeeze()
    shift = u_para + amp * xp.cos(2 * xp.pi * mode * e1)
    ana_res = xp.exp(-((v1 - shift[:, None]) ** 2) / 2.0)
    ana_res *= n / (2 * xp.pi) ** (1 / 2) * xp.exp(-(v_perp**2) / 2.0)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 20], label="analytical")
        plt.plot(e1, res[:, 20], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift (para)")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift (para)")
        plt.xlabel("v_para")
        plt.ylabel("f(v_para)")

        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (para) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth_para = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2Dvperp(
        n=(2.0, None),
        vth_para=(vth_para, pert),
        volume_form=False,
    )

    v_perp = 0.1
    meshgrids = xp.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        v_perp,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_para + amp * xp.cos(2 * xp.pi * mode * e1)
    ana_res = xp.exp(-(v1**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * xp.pi) ** (1 / 2) * thermal[:, None])
    ana_res *= xp.exp(-(v_perp**2) / 2.0)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label="analytical")
        plt.plot(e1, res[:, 0], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth (para)")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth (para)")
        plt.xlabel("v_1")
        plt.ylabel("f(v_1)")

        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (perp) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth_perp = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2Dvperp(
        n=(2.0, None),
        vth_perp=(vth_perp, pert),
        volume_form=False,
    )

    meshgrids = xp.meshgrid(
        e1,
        [0.0],
        [0.0],
        0.0,
        v2,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_perp + amp * xp.cos(2 * xp.pi * mode * e1)
    ana_res = xp.exp(-(v2**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * xp.pi) ** (1 / 2) * thermal[:, None] ** 2)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 0], label="analytical")
        plt.plot(e1, res[:, 0], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth (perp)")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in vth (perp)")
        plt.xlabel("v_1")
        plt.ylabel("f(v_1)")

        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.491230, 0.298228, 0.198739, 0.521298]
    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian = GyroMaxwellian2Dvperp(n=(0.0, pert), volume_form=False)

    v_perp = 0.1
    meshgrids = xp.meshgrid(e1, [0.0], [0.0], [0.0], v_perp)

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0 * c[3] * xp.exp(-c[2] / c[1] * xp.tanh((e1 - c[0]) / c[2])) / (2 * xp.pi) ** (1 / 2)
    ana_res *= xp.exp(-(v_perp**2) / 2.0)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert xp.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"


@pytest.mark.parametrize("num_elements", [[8, 12, 12]])
def test_maxwellian_2d_mhd(num_elements, with_desc, show_plot=False):
    """Tests the GyroMaxwellian2D class for mhd equilibrium moments."""

    import inspect

    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy import domains, equils, perturbations
    from struphy.fields_background.base import MHDequilibrium
    from struphy.initial.base import Perturbation
    from struphy.kinetic_background.maxwellians import GyroMaxwellian2Dvperp

    e1 = xp.linspace(0.0, 1.0, num_elements[0])
    e2 = xp.linspace(0.0, 1.0, num_elements[1])
    e3 = xp.linspace(0.0, 1.0, num_elements[2])
    v1 = [0.0]
    v2 = [0.0, 2.0]

    meshgrids = xp.meshgrid(e1, e2, e3, v1, v2, indexing="ij")
    e_meshgrids = xp.meshgrid(e1, e2, e3, indexing="ij")

    n_mks = 17
    e1_fl = xp.random.rand(n_mks)
    e2_fl = xp.random.rand(n_mks)
    e3_fl = xp.random.rand(n_mks)
    v1_fl = xp.random.randn(n_mks)
    v2_fl = xp.random.rand(n_mks)
    args_fl = [e1_fl, e2_fl, e3_fl, v1_fl, v2_fl]
    e_args_fl = xp.concatenate((e1_fl[:, None], e2_fl[:, None], e3_fl[:, None]), axis=1)

    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and val.__module__ == equils.__name__:
            logger.info(f"{key =}")

            if "DESCequilibrium" in key and not with_desc:
                logger.info(f"Attention: {with_desc =}, DESC not tested here !!")
                continue

            if "GVECequilibrium" in key:
                logger.info("Attention: GVEC not tested here !!")
                # logger.info("Attention: flat (marker) evaluation not tested for GVEC at the moment.")
                continue

            mhd_equil = val()
            if not isinstance(mhd_equil, MHDequilibrium):
                continue

            logger.info(f"{mhd_equil.params =}")
            if "AdhocTorus" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    R0=mhd_equil.params["R0"],
                    tor_period=1,
                )
            elif "EQDSKequilibrium" in key:
                mhd_equil.domain = domains.Tokamak(equilibrium=mhd_equil)
            elif "CircularTokamak" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    R0=mhd_equil.params["R0"],
                    tor_period=1,
                )
            elif "HomogenSlab" in key:
                mhd_equil.domain = domains.Cuboid()
            elif "ShearedSlab" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["a"] * 2 * xp.pi,
                    r3=mhd_equil.params["R0"] * 2 * xp.pi,
                )
            elif "ShearFluid" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["b"],
                    r3=mhd_equil.params["c"],
                )
            elif "ScrewPinch" in key:
                mhd_equil.domain = domains.HollowCylinder(
                    a1=1e-3,
                    a2=mhd_equil.params["a"],
                    Lz=mhd_equil.params["R0"] * 2 * xp.pi,
                )
            else:
                try:
                    mhd_equil.domain = domains.Cuboid()
                except:
                    logger.info(f"Not setting domain for {key}.")

            maxwellian = GyroMaxwellian2Dvperp(
                n=(mhd_equil.n0, None),
                u_para=(mhd_equil.u_para0, None),
                vth_para=(mhd_equil.vth0, None),
                vth_perp=(mhd_equil.vth0, None),
                volume_form=False,
            )

            maxwellian_1 = GyroMaxwellian2Dvperp(
                n=(1.0, None),
                u_para=(mhd_equil.u_para0, None),
                vth_para=(mhd_equil.vth0, None),
                vth_perp=(mhd_equil.vth0, None),
                volume_form=False,
            )

            # test meshgrid evaluation
            n0 = mhd_equil.n0(*e_meshgrids)
            assert xp.allclose(maxwellian(*meshgrids)[:, :, :, 0, 0], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 0])

            assert xp.allclose(maxwellian(*meshgrids)[:, :, :, 0, 1], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 1])

            # test flat evaluation
            if "GVECequilibrium" in key:
                logger.info("Attention: GVEC not tested here !!")
                # logger.info("Attention: flat (marker) evaluation not tested for GVEC at the moment.")
                continue
                pass
            else:
                assert xp.allclose(maxwellian(*args_fl), mhd_equil.n0(e_args_fl) * maxwellian_1(*args_fl))
                assert xp.allclose(maxwellian.n(e1_fl, e2_fl, e3_fl), mhd_equil.n0(e_args_fl))

                u_maxw = maxwellian.u(e1_fl, e2_fl, e3_fl)
                tmp_jv = mhd_equil.jv(e_args_fl) / mhd_equil.n0(e_args_fl)
                tmp_unit_b1 = mhd_equil.unit_b1(e_args_fl)
                # j_parallel = jv.b1
                j_para = sum([ji * bi for ji, bi in zip(tmp_jv, tmp_unit_b1)])
                assert xp.allclose(u_maxw[0], j_para)

                vth_maxw = maxwellian.vth(e1_fl, e2_fl, e3_fl)
                vth_eq = xp.sqrt(mhd_equil.p0(e_args_fl) / mhd_equil.n0(e_args_fl))
                assert all([xp.allclose(v, vth_eq) for v in vth_maxw])

            # plotting moments
            if show_plot:
                plt.figure(f"{mhd_equil =}", figsize=(24, 16))
                x, y, z = mhd_equil.domain(*e_meshgrids)

                # density plots
                n_cart = mhd_equil.domain.push(maxwellian.n, *e_meshgrids)

                levels = xp.linspace(xp.min(n_cart) - 1e-10, xp.max(n_cart), 20)

                plt.subplot(2, 4, 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        z[:, num_elements[1] // 2 - 1, :],
                        n_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        y[:, num_elements[1] // 2 - 1, :],
                        n_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian density $n$, top view (e1-e3)")
                plt.subplot(2, 4, 4 + 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0], n_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0], n_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian density $n$, poloidal view (e1-e2)")

                # velocity plots
                us = maxwellian.u(*e_meshgrids)
                for i, u in enumerate(us[:1]):
                    levels = xp.linspace(xp.min(u) - 1e-10, xp.max(u), 20)

                    plt.subplot(2, 4, 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(
                            x[:, num_elements[1] // 2, :],
                            z[:, num_elements[1] // 2, :],
                            u[:, num_elements[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(
                            x[:, num_elements[1] // 2, :],
                            y[:, num_elements[1] // 2, :],
                            u[:, num_elements[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Maxwellian velocity $u_{i + 1}$, top view (e1-e3)")
                    plt.subplot(2, 4, 4 + 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0], u[:, :, 0], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0], u[:, :, 0], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Maxwellian velocity $u_{i + 1}$, poloidal view (e1-e2)")

                # thermal velocity plots
                vth = maxwellian.vth(*e_meshgrids)[0]
                vth_cart = mhd_equil.domain.push(vth, *e_meshgrids)

                levels = xp.linspace(xp.min(vth_cart) - 1e-10, xp.max(vth_cart), 20)

                plt.subplot(2, 4, 4)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        z[:, num_elements[1] // 2 - 1, :],
                        vth_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, num_elements[1] // 2, :],
                        y[:, num_elements[1] // 2 - 1, :],
                        vth_cart[:, num_elements[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian thermal velocity $v_t$, top view (e1-e3)")
                plt.subplot(2, 4, 8)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0], vth_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0], vth_cart[:, :, 0], levels=levels)
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Maxwellian density $v_t$, poloidal view (e1-e2)")

                plt.show()

            # test perturbations
            if "EQDSKequilibrium" in key:
                for key_2, val_2 in inspect.getmembers(perturbations):
                    if inspect.isclass(val_2) and val_2.__module__ == perturbations.__name__:
                        pert = val_2()
                        logger.info(f"{pert =}")
                        assert isinstance(pert, Perturbation)

                        if isinstance(pert, perturbations.Noise):
                            continue

                        # background + perturbation
                        maxwellian_perturbed = GyroMaxwellian2Dvperp(
                            n=(mhd_equil.n0, pert),
                            u_para=(mhd_equil.u_para0, pert),
                            vth_para=(mhd_equil.vth0, pert),
                            vth_perp=(mhd_equil.vth0, pert),
                            volume_form=False,
                        )

                        # test meshgrid evaluation
                        assert maxwellian_perturbed(*meshgrids).shape == meshgrids[0].shape

                        # test flat evaluation
                        assert maxwellian_perturbed(*args_fl).shape == args_fl[0].shape

                        # pure perturbation
                        maxwellian_zero_bckgr = GyroMaxwellian2Dvperp(
                            n=(0.0, pert),
                            u_para=(0.0, pert),
                            u_perp=(0.0, pert),
                            vth_para=(0.0, pert),
                            vth_perp=(0.0, pert),
                            volume_form=False,
                        )

                        assert xp.allclose(maxwellian_zero_bckgr.n(*e_meshgrids), pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert xp.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[1], pert(*e_meshgrids))

                        # plotting perturbations
                        if show_plot and "EQDSKequilibrium" in key:  # and 'Torus' in key_2:
                            plt.figure(f"perturbation = {key_2}", figsize=(24, 16))
                            x, y, z = mhd_equil.domain(*e_meshgrids)

                            # density plots
                            n_cart = mhd_equil.domain.push(maxwellian_zero_bckgr.n, *e_meshgrids)

                            levels = xp.linspace(xp.min(n_cart) - 1e-10, xp.max(n_cart), 20)

                            plt.subplot(2, 4, 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    z[:, num_elements[1] // 2, :],
                                    n_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    y[:, num_elements[1] // 2, :],
                                    n_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed density $n$, top view (e1-e3)")
                            plt.subplot(2, 4, 4 + 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, :, 0], y[:, :, 0], n_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, :, 0], z[:, :, 0], n_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed density $n$, poloidal view (e1-e2)")

                            # velocity plots
                            us = maxwellian_zero_bckgr.u(*e_meshgrids)
                            for i, u in enumerate(us):
                                levels = xp.linspace(xp.min(u) - 1e-10, xp.max(u), 20)

                                plt.subplot(2, 4, 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, num_elements[1] // 2, :],
                                        z[:, num_elements[1] // 2, :],
                                        u[:, num_elements[1] // 2, :],
                                        levels=levels,
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                else:
                                    plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, num_elements[1] // 2, :],
                                        y[:, num_elements[1] // 2, :],
                                        u[:, num_elements[1] // 2, :],
                                        levels=levels,
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                plt.axis("equal")
                                plt.colorbar()
                                plt.title(f"Maxwellian perturbed velocity $u_{i + 1}$, top view (e1-e3)")
                                plt.subplot(2, 4, 4 + 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, :, 0], y[:, :, 0], u[:, :, 0], levels=levels)
                                    plt.xlabel("x")
                                    plt.ylabel("y")
                                else:
                                    plt.contourf(x[:, :, 0], z[:, :, 0], u[:, :, 0], levels=levels)
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                plt.axis("equal")
                                plt.colorbar()
                                plt.title(f"Maxwellian perturbed velocity $u_{i + 1}$, poloidal view (e1-e2)")

                            # thermal velocity plots
                            vth = maxwellian_zero_bckgr.vth(*e_meshgrids)[0]
                            vth_cart = mhd_equil.domain.push(vth, *e_meshgrids)

                            levels = xp.linspace(xp.min(vth_cart) - 1e-10, xp.max(vth_cart), 20)

                            plt.subplot(2, 4, 4)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    z[:, num_elements[1] // 2, :],
                                    vth_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, num_elements[1] // 2, :],
                                    y[:, num_elements[1] // 2, :],
                                    vth_cart[:, num_elements[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed thermal velocity $v_t$, top view (e1-e3)")
                            plt.subplot(2, 4, 8)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, :, 0], y[:, :, 0], vth_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("y")
                            else:
                                plt.contourf(x[:, :, 0], z[:, :, 0], vth_cart[:, :, 0], levels=levels)
                                plt.xlabel("x")
                                plt.ylabel("z")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title("Maxwellian perturbed density $v_t$, poloidal view (e1-e2)")

                            plt.show()


@pytest.mark.parametrize("num_markers", [200])
def test_canonical_maxwellian_uniform(num_markers, show_plot=False):
    """Tests the CanonicalMaxwellian2D evaluation scheme in
    :math:`(\\eta_1, \\eta_2, \\eta_3, v_\\parallel, \\mu)` coordinates
    (flat/marker evaluation), including caching of the canonical toroidal
    momentum :math:`\\psi_c`.

    Asserts that the results match an independently computed reference:

    .. math::
        f(\\eta, v_\\parallel, \\mu) = \\frac{n(\\psi_c)}{\\sqrt{2\\pi}\\,v_\\text{th}(\\psi_c)}
        \\exp\\left[-\\frac{v_\\parallel^2}{2 v_\\text{th}(\\psi_c)^2}\\right]
        \\frac{|B_0(\\eta)|}{v_\\text{th}(\\psi_c)^2}\\exp\\left[-\\frac{\\mu |B_0(\\eta)|}{v_\\text{th}(\\psi_c)^2}\\right],

    where the second factor drops the :math:`|B_0|` prefactor for ``volume_form=False``.
    """
    import cunumpy as xp
    import matplotlib.pyplot as plt

    from struphy import domains, equils, perturbations
    from struphy.kinetic_background.maxwellians import CanonicalMaxwellian2D

    epsilon = 1.0

    AdhocTorus_params = {
        "a": 1.0,
        "R0": 10.0,
        "B0": 3.0,
        "q_kind": 0.0,
        "q0": 1.71,
        "q1": 1.87,
        "n1": 0.0,
        "n2": 0.0,
        "na": 1.0,
        "p_kind": 1.0,
        "p1": 0.95,
        "p2": 0.05,
        "beta": 0.0018,
    }

    HollowTorus_params = {"a1": 0.1, "a2": 1.0, "R0": 10.0, "sfl": False, "tor_period": 6}

    mhd_equil = equils.AdhocTorus(**AdhocTorus_params)
    mhd_equil.domain = domains.HollowTorus(**HollowTorus_params)

    R0 = mhd_equil.params["R0"]
    B0_const = mhd_equil.params["B0"]

    def ref_psic_and_absB(eta1, eta2, eta3, v_para, mu):
        """Independent reference computation of psi_c and |B0|, mirroring
        CanonicalMaxwellian2D.eval_psic but without using the class under test."""
        etas = xp.concatenate((eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)
        absB = mhd_equil.absB0(etas)
        x, y, z = mhd_equil.domain(etas)
        R, P, Z = mhd_equil.inverse_map(x, y, z)
        psi = mhd_equil.psi(R, Z)

        energy = 0.5 * v_para**2 + mu * absB
        psic = psi - epsilon * B0_const * R0 / absB * v_para

        pos_mask = (energy - mu * B0_const) > 0
        correction = xp.zeros_like(psic)
        correction[pos_mask] = (
            epsilon * xp.sign(v_para[pos_mask]) * xp.sqrt(2 * (energy[pos_mask] - mu[pos_mask] * B0_const)) * R0
        )
        return psic + correction, absB

    def ref_eval(n_of_psic, vth_val, eta1, eta2, eta3, v_para, mu, volume_form=True):
        """Independent analytical reference for the canonical Maxwellian evaluated
        at phase space coordinates (eta1, eta2, eta3, v_para, mu)."""
        psic, absB = ref_psic_and_absB(eta1, eta2, eta3, v_para, mu)
        n_val = n_of_psic(psic) if callable(n_of_psic) else n_of_psic
        g_para = 1.0 / (vth_val * xp.sqrt(2 * xp.pi)) * xp.exp(-(v_para**2) / (2 * vth_val**2))
        g_mu = 1.0 / vth_val**2 * xp.exp(-mu * absB / vth_val**2)
        if volume_form:
            g_mu = g_mu * absB
        return n_val * g_para * g_mu

    xp.random.seed(1234)
    eta1 = xp.random.rand(num_markers)
    eta2 = xp.random.rand(num_markers)
    eta3 = xp.random.rand(num_markers)
    v_para = (xp.random.rand(num_markers) - 0.5) * 4.0
    mu = xp.random.rand(num_markers) * 0.5

    # ===========================================================
    # ===== Test uniform, isothermal canonical Maxwellian =====
    # ===========================================================
    n_val, vth_val = 2.0, 1.3

    maxwellian = CanonicalMaxwellian2D(n=(n_val, None), vth=(vth_val, None), equil=mhd_equil)

    res = maxwellian(eta1, eta2, eta3, v_para, mu)
    res_ana = ref_eval(n_val, vth_val, eta1, eta2, eta3, v_para, mu)
    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana=}"

    # test canonical Maxwellian profile in v_para, at fixed eta and mu
    v_para_p = xp.linspace(-5, 5, 64)
    eta1_p = 0.5 + 0.0 * v_para_p
    eta2_p = 0.5 + 0.0 * v_para_p
    eta3_p = 0.5 + 0.0 * v_para_p
    mu_p = 0.1 + 0.0 * v_para_p

    res = maxwellian(eta1_p, eta2_p, eta3_p, v_para_p, mu_p)
    res_ana = ref_eval(n_val, vth_val, eta1_p, eta2_p, eta3_p, v_para_p, mu_p)

    if show_plot:
        plt.plot(v_para_p, res_ana, label="analytical")
        plt.plot(v_para_p, res, "r*", label="CanonicalMaxwellian2D class")
        plt.legend()
        plt.title("Profile in v_para (eta=0.5, mu=0.1)")
        plt.ylabel("f(v_para)")
        plt.xlabel("v_para")
        plt.show()

    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana=}"

    # test canonical Maxwellian profile in mu, at fixed eta and v_para
    mu_p2 = xp.linspace(0, 2.0, 64)
    eta1_p2 = 0.5 + 0.0 * mu_p2
    eta2_p2 = 0.5 + 0.0 * mu_p2
    eta3_p2 = 0.5 + 0.0 * mu_p2
    v_para_p2 = 0.1 + 0.0 * mu_p2

    res = maxwellian(eta1_p2, eta2_p2, eta3_p2, v_para_p2, mu_p2)
    res_ana = ref_eval(n_val, vth_val, eta1_p2, eta2_p2, eta3_p2, v_para_p2, mu_p2)

    if show_plot:
        plt.plot(mu_p2, res_ana, label="analytical")
        plt.plot(mu_p2, res, "r*", label="CanonicalMaxwellian2D class")
        plt.legend()
        plt.title("Profile in mu (eta=0.5, v_para=0.1)")
        plt.ylabel("f(mu)")
        plt.xlabel("mu")
        plt.show()

    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana=}"

    # =====================================================================
    # ===== Test non-uniform n(psi_c): psi_c evaluation and caching =====
    # =====================================================================
    def n_of_psic(psic):
        return 1.5 + 0.1 * psic

    maxwellian_nc = CanonicalMaxwellian2D(n=(n_of_psic, None), vth=(vth_val, None), equil=mhd_equil)

    res = maxwellian_nc(eta1, eta2, eta3, v_para, mu)
    res_ana = ref_eval(n_of_psic, vth_val, eta1, eta2, eta3, v_para, mu)
    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana=}"

    # calling again with the same markers must hit the internal psi_c cache
    # and still return the (correct) result
    res_cached = maxwellian_nc(eta1, eta2, eta3, v_para, mu)
    assert xp.allclose(res_cached, res_ana, atol=10e-10), f"{res_cached=},\n {res_ana=}"

    # calling with different markers must invalidate the cache instead of
    # silently reusing the stale psi_c from the previous call
    eta1_b = xp.random.rand(num_markers)
    eta2_b = xp.random.rand(num_markers)
    eta3_b = xp.random.rand(num_markers)
    v_para_b = (xp.random.rand(num_markers) - 0.5) * 4.0
    mu_b = xp.random.rand(num_markers) * 0.5

    res_b = maxwellian_nc(eta1_b, eta2_b, eta3_b, v_para_b, mu_b)
    res_b_ana = ref_eval(n_of_psic, vth_val, eta1_b, eta2_b, eta3_b, v_para_b, mu_b)
    assert xp.allclose(res_b, res_b_ana, atol=10e-10), f"{res_b=},\n {res_b_ana=}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.46623, 0.17042, 0.11357, 0.521298]
    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian_pert = CanonicalMaxwellian2D(n=(0.0, pert), vth=(vth_val, None), equil=mhd_equil, volume_form=False)

    res = maxwellian_pert(eta1, eta2, eta3, v_para, mu)

    # the perturbation is added at the raw (eta1, eta2, eta3) position (not via psi_c/rc)
    n_pert = n0 * c[3] * xp.exp(-c[2] / c[1] * xp.tanh((eta1 - c[0]) / c[2]))
    res_ana = ref_eval(n_pert, vth_val, eta1, eta2, eta3, v_para, mu, volume_form=False)

    if show_plot:
        order = xp.argsort(eta1)
        plt.plot(eta1[order], res_ana[order], label="analytical")
        plt.plot(eta1[order], res[order], "r*", label="CanonicalMaxwellian2D class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f")
        plt.show()

    assert xp.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana=}"


if __name__ == "__main__":
    # test_maxwellian_3d_uniform(num_elements=[64, 1, 1], show_plot=True)
    # test_maxwellian_3d_perturbed(num_elements=[64, 1, 1], show_plot=True)
    # test_maxwellian_3d_mhd(num_elements=[8, 11, 12], with_desc=None, show_plot=False)
    # test_maxwellian_2d_uniform(num_elements=[64, 1, 1], show_plot=True)
    # test_maxwellian_2d_perturbed(num_elements=[64, 1, 1], show_plot=True)
    # test_maxwellian_2d_mhd(num_elements=[8, 12, 12], with_desc=None, show_plot=False)
    test_canonical_maxwellian_uniform(num_markers=200, show_plot=True)
