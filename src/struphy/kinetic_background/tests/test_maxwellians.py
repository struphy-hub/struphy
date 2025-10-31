import pytest


@pytest.mark.parametrize("Nel", [[64, 1, 1]])
def test_maxwellian_3d_uniform(Nel, show_plot=False):
    """Tests the Maxwellian3D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])

    # ==========================================================
    # ==== Test uniform non-shifted, isothermal Maxwellian =====
    # ==========================================================
    maxwellian = Maxwellian3D(n=(2.0, None))

    meshgrids = np.meshgrid(e1, e2, e3, [0.0], [0.0], [0.0])

    # Test constant value at v=0
    res = maxwellian(*meshgrids).squeeze()
    assert np.allclose(res, 2.0 / (2 * np.pi) ** (3 / 2) + 0 * e1, atol=10e-10), (
        f"{res=},\n {2.0 / (2 * np.pi) ** (3 / 2)}"
    )

    # test Maxwellian profile in v
    v1 = np.linspace(-5, 5, 128)
    meshgrids = np.meshgrid(
        [0.0],
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )
    res = maxwellian(*meshgrids).squeeze()
    res_ana = 2.0 * np.exp(-(v1**2) / 2.0) / (2 * np.pi) ** (3 / 2)
    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

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
        vs[i] = np.linspace(-5, 5, 128)
        meshgrids = np.meshgrid([0.0], [0.0], [0.0], *vs)
        res = maxwellian(*meshgrids).squeeze()

        res_ana = np.exp(-((vs[0] - u1) ** 2) / (2 * vth1**2))
        res_ana *= np.exp(-((vs[1] - u2) ** 2) / (2 * vth2**2))
        res_ana *= np.exp(-((vs[2] - u3) ** 2) / (2 * vth3**2))
        res_ana *= n / ((2 * np.pi) ** (3 / 2) * vth1 * vth2 * vth3)

        if show_plot:
            plt.plot(vs[i], res_ana, label="analytical")
            plt.plot(vs[i], res, "r*", label="Maxwellian class")
            plt.legend()
            plt.title("Test non-zero shifts and thermal velocities")
            plt.ylabel("f(v_" + str(i + 1) + ")")
            plt.xlabel("v_" + str(i + 1))
            plt.show()

        assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize("Nel", [[64, 1, 1]])
def test_maxwellian_3d_perturbed(Nel, show_plot=False):
    """Tests the Maxwellian3D class for perturbations."""

    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    v1 = np.linspace(-5.0, 5.0, 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(2.0, pert))

    meshgrids = np.meshgrid(e1, [0.0], [0.0], [0.0], [0.0], [0.0])

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2.0 + amp * np.cos(2 * np.pi * mode * e1)) / (2 * np.pi) ** (3 / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test cosine perturbation in shift =====
    # =============================================
    amp = 0.1
    mode = 1
    n = 2.0
    u1 = 1.2

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(n, None), u1=(u1, pert))

    meshgrids = np.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )

    res = maxwellian(*meshgrids).squeeze()
    shift = u1 + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-((v1 - shift[:, None]) ** 2) / 2)
    ana_res *= n / (2 * np.pi) ** (3 / 2)

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

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ===========================================
    # ===== Test cosine perturbation in vth =====
    # ===========================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth1 = 1.2

    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = Maxwellian3D(n=(n, None), vth1=(vth1, pert))

    meshgrids = np.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        [0.0],
        [0.0],
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth1 + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-(v1**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * np.pi) ** (3 / 2) * thermal[:, None])

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

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = (0.491230, 0.298228, 0.198739, 0.521298)

    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian = Maxwellian3D(n=(0.0, pert))

    meshgrids = np.meshgrid(e1, [0.0], [0.0], [0.0], [0.0], [0.0])

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0 * c[3] * np.exp(-c[2] / c[1] * np.tanh((e1 - c[0]) / c[2])) / (2 * np.pi) ** (3 / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"


@pytest.mark.parametrize("Nel", [[8, 11, 12]])
def test_maxwellian_3d_mhd(Nel, with_desc, show_plot=False):
    """Tests the Maxwellian3D class for mhd equilibrium moments."""

    import inspect

    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.fields_background import equils
    from struphy.fields_background.base import FluidEquilibrium
    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.initial.base import Perturbation
    from struphy.kinetic_background.maxwellians import Maxwellian3D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])
    v1 = [0.0]
    v2 = [0.0, -1.0]
    v3 = [0.0, -1.0, -1.3]

    meshgrids = np.meshgrid(e1, e2, e3, v1, v2, v3, indexing="ij")
    e_meshgrids = np.meshgrid(e1, e2, e3, indexing="ij")

    n_mks = 17
    e1_fl = np.random.rand(n_mks)
    e2_fl = np.random.rand(n_mks)
    e3_fl = np.random.rand(n_mks)
    v1_fl = np.random.randn(n_mks)
    v2_fl = np.random.randn(n_mks)
    v3_fl = np.random.randn(n_mks)
    args_fl = [e1_fl, e2_fl, e3_fl, v1_fl, v2_fl, v3_fl]
    e_args_fl = np.concatenate((e1_fl[:, None], e2_fl[:, None], e3_fl[:, None]), axis=1)

    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and val.__module__ == equils.__name__:
            print(f"{key = }")

            if "DESCequilibrium" in key and not with_desc:
                print(f"Attention: {with_desc = }, DESC not tested here !!")
                continue

            if "GVECequilibrium" in key:
                print(f"Attention: flat (marker) evaluation not tested for GVEC at the moment.")

            mhd_equil = val()
            assert isinstance(mhd_equil, FluidEquilibrium)
            print(f"{mhd_equil.params = }")
            if "AdhocTorus" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3, a2=mhd_equil.params["a"], R0=mhd_equil.params["R0"], tor_period=1
                )
            elif "EQDSKequilibrium" in key:
                mhd_equil.domain = domains.Tokamak(equilibrium=mhd_equil)
            elif "CircularTokamak" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3, a2=mhd_equil.params["a"], R0=mhd_equil.params["R0"], tor_period=1
                )
            elif "HomogenSlab" in key:
                mhd_equil.domain = domains.Cuboid()
            elif "ShearedSlab" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["a"] * 2 * np.pi,
                    r3=mhd_equil.params["R0"] * 2 * np.pi,
                )
            elif "ShearFluid" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"], r2=mhd_equil.params["b"], r3=mhd_equil.params["c"]
                )
            elif "ScrewPinch" in key:
                mhd_equil.domain = domains.HollowCylinder(
                    a1=1e-3, a2=mhd_equil.params["a"], Lz=mhd_equil.params["R0"] * 2 * np.pi
                )
            else:
                try:
                    mhd_equil.domain = domains.Cuboid()
                except:
                    print(f"Not setting domain for {key}.")

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
            assert np.allclose(
                maxwellian(*meshgrids)[:, :, :, 0, 0, 0], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 0, 0]
            )

            assert np.allclose(
                maxwellian(*meshgrids)[:, :, :, 0, 1, 2], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 1, 2]
            )

            # test flat evaluation
            if "GVECequilibrium" in key:
                pass
            else:
                assert np.allclose(maxwellian(*args_fl), mhd_equil.n0(e_args_fl) * maxwellian_1(*args_fl))
                assert np.allclose(maxwellian.n(e1_fl, e2_fl, e3_fl), mhd_equil.n0(e_args_fl))

                u_maxw = maxwellian.u(e1_fl, e2_fl, e3_fl)
                u_eq = mhd_equil.u_cart(e_args_fl)[0]
                assert all([np.allclose(m, e) for m, e in zip(u_maxw, u_eq)])

                vth_maxw = maxwellian.vth(e1_fl, e2_fl, e3_fl)
                vth_eq = np.sqrt(mhd_equil.p0(e_args_fl) / mhd_equil.n0(e_args_fl))
                assert all([np.allclose(v, vth_eq) for v in vth_maxw])

            # plotting moments
            if show_plot:
                plt.figure(f"{mhd_equil = }", figsize=(24, 16))
                x, y, z = mhd_equil.domain(*e_meshgrids)

                # density plots
                n_cart = mhd_equil.domain.push(maxwellian.n, *e_meshgrids)

                levels = np.linspace(np.min(n_cart) - 1e-10, np.max(n_cart), 20)

                plt.subplot(2, 5, 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2 - 1, :], n_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2 - 1, :], n_cart[:, Nel[1] // 2, :], levels=levels
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
                    levels = np.linspace(np.min(u) - 1e-10, np.max(u), 20)

                    plt.subplot(2, 5, 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels)
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

                levels = np.linspace(np.min(vth_cart) - 1e-10, np.max(vth_cart), 20)

                plt.subplot(2, 5, 5)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2 - 1, :], vth_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2 - 1, :], vth_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title(f"Maxwellian thermal velocity $v_t$, top view (e1-e3)")
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
                plt.title(f"Maxwellian thermal velocity $v_t$, poloidal view (e1-e2)")

                plt.show()

            # test perturbations
            if "EQDSKequilibrium" in key:
                maxw_params_zero = {"n": 0.0, "vth1": 0.0, "vth2": 0.0, "vth3": 0.0}

                for key_2, val_2 in inspect.getmembers(perturbations):
                    if inspect.isclass(val_2) and val_2.__module__ == perturbations.__name__:
                        pert = val_2()
                        assert isinstance(pert, Perturbation)
                        print(f"{pert = }")
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

                        assert np.allclose(maxwellian_zero_bckgr.n(*e_meshgrids), pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[2], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[2], pert(*e_meshgrids))

                        # plotting perturbations
                        if show_plot:  # and 'Torus' in key_2:
                            plt.figure(f"perturbation = {key_2}", figsize=(24, 16))
                            x, y, z = mhd_equil.domain(*e_meshgrids)

                            # density plots
                            n_cart = mhd_equil.domain.push(maxwellian_zero_bckgr.n, *e_meshgrids)

                            levels = np.linspace(np.min(n_cart) - 1e-10, np.max(n_cart), 20)

                            plt.subplot(2, 5, 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], n_cart[:, Nel[1] // 2, :], levels=levels
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], n_cart[:, Nel[1] // 2, :], levels=levels
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
                                levels = np.linspace(np.min(u) - 1e-10, np.max(u), 20)

                                plt.subplot(2, 5, 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                else:
                                    plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels
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

                            levels = np.linspace(np.min(vth_cart) - 1e-10, np.max(vth_cart), 20)

                            plt.subplot(2, 5, 5)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :],
                                    z[:, Nel[1] // 2, :],
                                    vth_cart[:, Nel[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :],
                                    y[:, Nel[1] // 2, :],
                                    vth_cart[:, Nel[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title(f"Maxwellian perturbed thermal velocity $v_t$, top view (e1-e3)")
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
                            plt.title(f"Maxwellian perturbed thermal velocity $v_t$, poloidal view (e1-e2)")

                            plt.show()


@pytest.mark.parametrize("Nel", [[64, 1, 1]])
def test_maxwellian_2d_uniform(Nel, show_plot=False):
    """Tests the GyroMaxwellian2D class as a uniform Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.kinetic_background.maxwellians import GyroMaxwellian2D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])

    # ===========================================================
    # ===== Test uniform non-shifted, isothermal Maxwellian =====
    # ===========================================================
    maxwellian = GyroMaxwellian2D(n=(2.0, None), volume_form=False)

    meshgrids = np.meshgrid(e1, e2, e3, [0.01], [0.01])

    # Test constant value at v_para = v_perp = 0.01
    res = maxwellian(*meshgrids).squeeze()
    assert np.allclose(res, 2.0 / (2 * np.pi) ** (1 / 2) * np.exp(-(0.01**2)) + 0 * e1, atol=10e-10), (
        f"{res=},\n {2.0 / (2 * np.pi) ** (3 / 2)}"
    )

    # test Maxwellian profile in v
    v_para = np.linspace(-5, 5, 64)
    v_perp = np.linspace(0, 2.5, 64)
    vpara, vperp = np.meshgrid(v_para, v_perp)

    meshgrids = np.meshgrid(
        [0.0],
        [0.0],
        [0.0],
        v_para,
        v_perp,
    )
    res = maxwellian(*meshgrids).squeeze()

    res_ana = 2.0 / (2 * np.pi) ** (1 / 2) * np.exp(-(vpara.T**2) / 2.0 - vperp.T**2 / 2.0)
    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # =======================================================
    # ===== Test non-zero shifts and thermal velocities =====
    # =======================================================
    n = 2.0
    u_para = 0.1
    u_perp = 0.2
    vth_para = 1.2
    vth_perp = 0.5

    maxwellian = GyroMaxwellian2D(
        n=(n, None),
        u_para=(u_para, None),
        u_perp=(u_perp, None),
        vth_para=(vth_para, None),
        vth_perp=(vth_perp, None),
        volume_form=False,
    )

    # test Maxwellian profile in v
    v_para = np.linspace(-5, 5, 64)
    v_perp = np.linspace(0, 2.5, 64)
    vpara, vperp = np.meshgrid(v_para, v_perp)

    meshgrids = np.meshgrid([0.0], [0.0], [0.0], v_para, v_perp)
    res = maxwellian(*meshgrids).squeeze()

    res_ana = np.exp(-((vpara.T - u_para) ** 2) / (2 * vth_para**2))
    res_ana *= np.exp(-((vperp.T - u_perp) ** 2) / (2 * vth_perp**2))
    res_ana *= n / ((2 * np.pi) ** (1 / 2) * vth_para * vth_perp**2)

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

    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana =}"


@pytest.mark.parametrize("Nel", [[6, 1, 1]])
def test_maxwellian_2d_perturbed(Nel, show_plot=False):
    """Tests the GyroMaxwellian2D class for perturbations."""

    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import GyroMaxwellian2D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    v1 = np.linspace(-5.0, 5.0, 128)
    v2 = np.linspace(0, 2.5, 128)

    # ===============================================
    # ===== Test cosine perturbation in density =====
    # ===============================================
    amp = 0.1
    mode = 1
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2D(n=(2.0, pert), volume_form=False)

    v_perp = 0.1
    meshgrids = np.meshgrid(e1, [0.0], [0.0], [0.0], v_perp)

    res = maxwellian(*meshgrids).squeeze()
    ana_res = (2.0 + amp * np.cos(2 * np.pi * mode * e1)) / (2 * np.pi) ** (1 / 2)
    ana_res *= np.exp(-(v_perp**2) / 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ====================================================
    # ===== Test cosine perturbation in shift (para) =====
    # ====================================================
    amp = 0.1
    mode = 1
    n = 2.0
    u_para = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2D(
        n=(2.0, None),
        u_para=(u_para, pert),
        volume_form=False,
    )

    v_perp = 0.1
    meshgrids = np.meshgrid(e1, [0.0], [0.0], v1, v_perp)

    res = maxwellian(*meshgrids).squeeze()
    shift = u_para + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-((v1 - shift[:, None]) ** 2) / 2.0)
    ana_res *= n / (2 * np.pi) ** (1 / 2) * np.exp(-(v_perp**2) / 2.0)

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

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ====================================================
    # ===== Test cosine perturbation in shift (perp) =====
    # ====================================================
    amp = 0.1
    mode = 1
    n = 2.0
    u_perp = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2D(
        n=(2.0, None),
        u_perp=(u_perp, pert),
        volume_form=False,
    )

    meshgrids = np.meshgrid(e1, [0.0], [0.0], 0.0, v2)

    res = maxwellian(*meshgrids).squeeze()
    shift = u_perp + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-((v2 - shift[:, None]) ** 2) / 2.0)
    ana_res *= n / (2 * np.pi) ** (1 / 2)

    if show_plot:
        plt.figure(1)
        plt.plot(e1, ana_res[:, 20], label="analytical")
        plt.plot(e1, res[:, 20], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift (perp)")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")

        plt.figure(2)
        plt.plot(v1, ana_res[0, :], label="analytical")
        plt.plot(v1, res[0, :], "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test cosine perturbation in shift (perp)")
        plt.xlabel("v_perp")
        plt.ylabel("f(v_perp)")

        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (para) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth_para = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2D(
        n=(2.0, None),
        vth_para=(vth_para, pert),
        volume_form=False,
    )

    v_perp = 0.1
    meshgrids = np.meshgrid(
        e1,
        [0.0],
        [0.0],
        v1,
        v_perp,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_para + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-(v1**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * np.pi) ** (1 / 2) * thermal[:, None])
    ana_res *= np.exp(-(v_perp**2) / 2.0)

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

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # ==================================================
    # ===== Test cosine perturbation in vth (perp) =====
    # ==================================================
    amp = 0.1
    mode = 1
    n = 2.0
    vth_perp = 1.2
    pert = perturbations.ModesCos(ls=(mode,), amps=(amp,))

    maxwellian = GyroMaxwellian2D(
        n=(2.0, None),
        vth_perp=(vth_perp, pert),
        volume_form=False,
    )

    meshgrids = np.meshgrid(
        e1,
        [0.0],
        [0.0],
        0.0,
        v2,
    )

    res = maxwellian(*meshgrids).squeeze()
    thermal = vth_perp + amp * np.cos(2 * np.pi * mode * e1)
    ana_res = np.exp(-(v2**2) / (2.0 * thermal[:, None] ** 2))
    ana_res *= n / ((2 * np.pi) ** (1 / 2) * thermal[:, None] ** 2)

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

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.491230, 0.298228, 0.198739, 0.521298]
    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian = GyroMaxwellian2D(n=(0.0, pert), volume_form=False)

    v_perp = 0.1
    meshgrids = np.meshgrid(e1, [0.0], [0.0], [0.0], v_perp)

    res = maxwellian(*meshgrids).squeeze()
    ana_res = n0 * c[3] * np.exp(-c[2] / c[1] * np.tanh((e1 - c[0]) / c[2])) / (2 * np.pi) ** (1 / 2)
    ana_res *= np.exp(-(v_perp**2) / 2.0)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="Maxwellian Class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"


@pytest.mark.parametrize("Nel", [[8, 12, 12]])
def test_maxwellian_2d_mhd(Nel, with_desc, show_plot=False):
    """Tests the GyroMaxwellian2D class for mhd equilibrium moments."""

    import inspect

    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.fields_background import equils
    from struphy.fields_background.base import FluidEquilibriumWithB
    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.initial.base import Perturbation
    from struphy.kinetic_background.maxwellians import GyroMaxwellian2D

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])
    v1 = [0.0]
    v2 = [0.0, 2.0]

    meshgrids = np.meshgrid(e1, e2, e3, v1, v2, indexing="ij")
    e_meshgrids = np.meshgrid(e1, e2, e3, indexing="ij")

    n_mks = 17
    e1_fl = np.random.rand(n_mks)
    e2_fl = np.random.rand(n_mks)
    e3_fl = np.random.rand(n_mks)
    v1_fl = np.random.randn(n_mks)
    v2_fl = np.random.rand(n_mks)
    args_fl = [e1_fl, e2_fl, e3_fl, v1_fl, v2_fl]
    e_args_fl = np.concatenate((e1_fl[:, None], e2_fl[:, None], e3_fl[:, None]), axis=1)

    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and val.__module__ == equils.__name__:
            print(f"{key = }")

            if "DESCequilibrium" in key and not with_desc:
                print(f"Attention: {with_desc = }, DESC not tested here !!")
                continue

            if "GVECequilibrium" in key:
                print(f"Attention: flat (marker) evaluation not tested for GVEC at the moment.")

            mhd_equil = val()
            if not isinstance(mhd_equil, FluidEquilibriumWithB):
                continue

            print(f"{mhd_equil.params = }")
            if "AdhocTorus" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3, a2=mhd_equil.params["a"], R0=mhd_equil.params["R0"], tor_period=1
                )
            elif "EQDSKequilibrium" in key:
                mhd_equil.domain = domains.Tokamak(equilibrium=mhd_equil)
            elif "CircularTokamak" in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3, a2=mhd_equil.params["a"], R0=mhd_equil.params["R0"], tor_period=1
                )
            elif "HomogenSlab" in key:
                mhd_equil.domain = domains.Cuboid()
            elif "ShearedSlab" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"],
                    r2=mhd_equil.params["a"] * 2 * np.pi,
                    r3=mhd_equil.params["R0"] * 2 * np.pi,
                )
            elif "ShearFluid" in key:
                mhd_equil.domain = domains.Cuboid(
                    r1=mhd_equil.params["a"], r2=mhd_equil.params["b"], r3=mhd_equil.params["c"]
                )
            elif "ScrewPinch" in key:
                mhd_equil.domain = domains.HollowCylinder(
                    a1=1e-3, a2=mhd_equil.params["a"], Lz=mhd_equil.params["R0"] * 2 * np.pi
                )
            else:
                try:
                    mhd_equil.domain = domains.Cuboid()
                except:
                    print(f"Not setting domain for {key}.")

            maxwellian = GyroMaxwellian2D(
                n=(mhd_equil.n0, None),
                u_para=(mhd_equil.u_para0, None),
                vth_para=(mhd_equil.vth0, None),
                vth_perp=(mhd_equil.vth0, None),
                volume_form=False,
            )

            maxwellian_1 = GyroMaxwellian2D(
                n=(1.0, None),
                u_para=(mhd_equil.u_para0, None),
                vth_para=(mhd_equil.vth0, None),
                vth_perp=(mhd_equil.vth0, None),
                volume_form=False,
            )

            # test meshgrid evaluation
            n0 = mhd_equil.n0(*e_meshgrids)
            assert np.allclose(maxwellian(*meshgrids)[:, :, :, 0, 0], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 0])

            assert np.allclose(maxwellian(*meshgrids)[:, :, :, 0, 1], n0 * maxwellian_1(*meshgrids)[:, :, :, 0, 1])

            # test flat evaluation
            if "GVECequilibrium" in key:
                pass
            else:
                assert np.allclose(maxwellian(*args_fl), mhd_equil.n0(e_args_fl) * maxwellian_1(*args_fl))
                assert np.allclose(maxwellian.n(e1_fl, e2_fl, e3_fl), mhd_equil.n0(e_args_fl))

                u_maxw = maxwellian.u(e1_fl, e2_fl, e3_fl)
                tmp_jv = mhd_equil.jv(e_args_fl) / mhd_equil.n0(e_args_fl)
                tmp_unit_b1 = mhd_equil.unit_b1(e_args_fl)
                # j_parallel = jv.b1
                j_para = sum([ji * bi for ji, bi in zip(tmp_jv, tmp_unit_b1)])
                assert np.allclose(u_maxw[0], j_para)

                vth_maxw = maxwellian.vth(e1_fl, e2_fl, e3_fl)
                vth_eq = np.sqrt(mhd_equil.p0(e_args_fl) / mhd_equil.n0(e_args_fl))
                assert all([np.allclose(v, vth_eq) for v in vth_maxw])

            # plotting moments
            if show_plot:
                plt.figure(f"{mhd_equil = }", figsize=(24, 16))
                x, y, z = mhd_equil.domain(*e_meshgrids)

                # density plots
                n_cart = mhd_equil.domain.push(maxwellian.n, *e_meshgrids)

                levels = np.linspace(np.min(n_cart) - 1e-10, np.max(n_cart), 20)

                plt.subplot(2, 4, 1)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2 - 1, :], n_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2 - 1, :], n_cart[:, Nel[1] // 2, :], levels=levels
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
                    levels = np.linspace(np.min(u) - 1e-10, np.max(u), 20)

                    plt.subplot(2, 4, 2 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels)
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels)
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

                levels = np.linspace(np.min(vth_cart) - 1e-10, np.max(vth_cart), 20)

                plt.subplot(2, 4, 4)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2 - 1, :], vth_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                    plt.contourf(
                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2 - 1, :], vth_cart[:, Nel[1] // 2, :], levels=levels
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title(f"Maxwellian thermal velocity $v_t$, top view (e1-e3)")
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
                plt.title(f"Maxwellian density $v_t$, poloidal view (e1-e2)")

                plt.show()

            # test perturbations
            if "EQDSKequilibrium" in key:
                for key_2, val_2 in inspect.getmembers(perturbations):
                    if inspect.isclass(val_2) and val_2.__module__ == perturbations.__name__:
                        pert = val_2()
                        print(f"{pert = }")
                        assert isinstance(pert, Perturbation)

                        if isinstance(pert, perturbations.Noise):
                            continue

                        # background + perturbation
                        maxwellian_perturbed = GyroMaxwellian2D(
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
                        maxwellian_zero_bckgr = GyroMaxwellian2D(
                            n=(0.0, pert),
                            u_para=(0.0, pert),
                            u_perp=(0.0, pert),
                            vth_para=(0.0, pert),
                            vth_perp=(0.0, pert),
                            volume_form=False,
                        )

                        assert np.allclose(maxwellian_zero_bckgr.n(*e_meshgrids), pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.u(*e_meshgrids)[1], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[0], pert(*e_meshgrids))
                        assert np.allclose(maxwellian_zero_bckgr.vth(*e_meshgrids)[1], pert(*e_meshgrids))

                        # plotting perturbations
                        if show_plot and "EQDSKequilibrium" in key:  # and 'Torus' in key_2:
                            plt.figure(f"perturbation = {key_2}", figsize=(24, 16))
                            x, y, z = mhd_equil.domain(*e_meshgrids)

                            # density plots
                            n_cart = mhd_equil.domain.push(maxwellian_zero_bckgr.n, *e_meshgrids)

                            levels = np.linspace(np.min(n_cart) - 1e-10, np.max(n_cart), 20)

                            plt.subplot(2, 4, 1)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], n_cart[:, Nel[1] // 2, :], levels=levels
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], n_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], n_cart[:, Nel[1] // 2, :], levels=levels
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
                                levels = np.linspace(np.min(u) - 1e-10, np.max(u), 20)

                                plt.subplot(2, 4, 2 + i)
                                if "Slab" in key or "Pinch" in key:
                                    plt.contourf(x[:, 0, :], z[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, Nel[1] // 2, :], z[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels
                                    )
                                    plt.xlabel("x")
                                    plt.ylabel("z")
                                else:
                                    plt.contourf(x[:, 0, :], y[:, 0, :], u[:, 0, :], levels=levels)
                                    plt.contourf(
                                        x[:, Nel[1] // 2, :], y[:, Nel[1] // 2, :], u[:, Nel[1] // 2, :], levels=levels
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

                            levels = np.linspace(np.min(vth_cart) - 1e-10, np.max(vth_cart), 20)

                            plt.subplot(2, 4, 4)
                            if "Slab" in key or "Pinch" in key:
                                plt.contourf(x[:, 0, :], z[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :],
                                    z[:, Nel[1] // 2, :],
                                    vth_cart[:, Nel[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("z")
                            else:
                                plt.contourf(x[:, 0, :], y[:, 0, :], vth_cart[:, 0, :], levels=levels)
                                plt.contourf(
                                    x[:, Nel[1] // 2, :],
                                    y[:, Nel[1] // 2, :],
                                    vth_cart[:, Nel[1] // 2, :],
                                    levels=levels,
                                )
                                plt.xlabel("x")
                                plt.ylabel("y")
                            plt.axis("equal")
                            plt.colorbar()
                            plt.title(f"Maxwellian perturbed thermal velocity $v_t$, top view (e1-e3)")
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
                            plt.title(f"Maxwellian perturbed density $v_t$, poloidal view (e1-e2)")

                            plt.show()


@pytest.mark.parametrize("Nel", [[64, 1, 1]])
def test_canonical_maxwellian_uniform(Nel, show_plot=False):
    """Tests the CanonicalMaxwellian class as a uniform canonical Maxwellian.

    Asserts that the results over the domain and velocity space correspond to the
    analytical computation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from struphy.fields_background import equils
    from struphy.geometry import domains
    from struphy.initial import perturbations
    from struphy.kinetic_background.maxwellians import CanonicalMaxwellian

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])

    eta_meshgrid = np.meshgrid(e1, e2, e3)

    v_para = 0.01
    v_perp = 0.01

    epsilon = 1.0

    # evaluate three constants of motions at AdhocTorus equilibrium
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

    absB = mhd_equil.absB0(*eta_meshgrid)

    # magnetic moment
    mu = v_perp**2 / 2.0 / absB

    # total energy
    energy = 1 / 2 * v_para**2 + mu * absB

    # shifted canonical toroidal momentum
    a1 = mhd_equil.domain.params["a1"]
    R0 = mhd_equil.params["R0"]
    B0 = mhd_equil.params["B0"]

    r = eta_meshgrid[0] * (1 - a1) + a1

    psi = mhd_equil.psi_r(r)

    psic = psi - epsilon * B0 * R0 / absB * v_para
    psic += epsilon * np.sign(v_para) * np.sqrt(2 * (energy - mu * B0)) * R0 * np.heaviside(energy - mu * B0, 0)

    # ===========================================================
    # ===== Test uniform, isothermal canonical Maxwellian =====
    # ===========================================================
    maxw_params = {"n": 2.0, "vth": 1.0}

    maxwellian = CanonicalMaxwellian(n=(2.0, None), vth=(1.0, None))

    # Test constant value at v_para = v_perp = 0.01
    res = maxwellian(energy, mu, psic).squeeze()
    res_ana = (
        maxw_params["n"]
        * 2
        * np.sqrt(energy / np.pi)
        / maxw_params["vth"] ** 3
        * np.exp(-energy / maxw_params["vth"] ** 2)
    )
    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # test canonical Maxwellian profile in v_para
    v_para = np.linspace(-5, 5, 64)
    v_perp = 0.1

    absB = mhd_equil.absB0(0.0, 0.0, 0.0)[0, 0, 0]

    # magnetic moment
    mu = v_perp**2 / 2.0 / absB

    # total energy
    energy = 1 / 2 * v_para**2 + mu * absB

    # shifted canonical toroidal momentum
    r = a1

    psi = mhd_equil.psi_r(r)

    psic = psi - epsilon * B0 * R0 / absB * v_para
    psic += epsilon * np.sign(v_para) * np.sqrt(2 * (energy - mu * B0)) * R0 * np.heaviside(energy - mu * B0, 0)

    com_meshgrids = np.meshgrid(energy, mu, psic)

    res = maxwellian(*com_meshgrids).squeeze()

    res_ana = (
        maxw_params["n"]
        * 2
        * np.sqrt(com_meshgrids[0] / np.pi)
        / maxw_params["vth"] ** 3
        * np.exp(-com_meshgrids[0] / maxw_params["vth"] ** 2)
    )

    if show_plot:
        plt.plot(v_para, res_ana[0, :, 0], label="analytical")
        plt.plot(v_para, res[:, 0], "r*", label="CanonicalMaxwellian class")
        plt.legend()
        plt.title("Profile in v_para (v_perp = 0.1)")
        plt.ylabel("f(v_para)")
        plt.xlabel("v_para")
        plt.show()

    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # test canonical Maxwellian profile in v_perp
    v_para = 0.1
    v_perp = np.linspace(0, 2.5, 64)

    absB = mhd_equil.absB0(0.5, 0.5, 0.5)[0, 0, 0]

    # magnetic moment
    mu = v_perp**2 / 2.0 / absB

    # total energy
    energy = 1 / 2 * v_para**2 + mu * absB

    # shifted canonical toroidal momentum
    r = a1

    psi = mhd_equil.psi_r(r)

    psic = psi - epsilon * B0 * R0 / absB * v_para
    psic += epsilon * np.sign(v_para) * np.sqrt(2 * (energy - mu * B0)) * R0 * np.heaviside(energy - mu * B0, 0)

    com_meshgrids = np.meshgrid(energy, mu, psic)

    res = maxwellian(*com_meshgrids).squeeze()

    res_ana = (
        maxw_params["n"]
        * 2
        * np.sqrt(com_meshgrids[0] / np.pi)
        / maxw_params["vth"] ** 3
        * np.exp(-com_meshgrids[0] / maxw_params["vth"] ** 2)
    )

    if show_plot:
        plt.plot(v_perp, res_ana[0, :, 0], label="analytical")
        plt.plot(v_perp, res[0, :, 0], "r*", label="CanonicalMaxwellian class")
        plt.legend()
        plt.title("Profile in v_perp (v_para = 0.1)")
        plt.ylabel("f(v_perp)")
        plt.xlabel("v_perp")
        plt.show()

    assert np.allclose(res, res_ana, atol=10e-10), f"{res=},\n {res_ana}"

    # =============================================
    # ===== Test ITPA perturbation in density =====
    # =============================================
    n0 = 0.00720655
    c = [0.46623, 0.17042, 0.11357, 0.521298]
    maxw_params = {
        "n": {"ITPA_density": {"n0": n0, "c": c}},
        "vth": 1.0,
    }
    pert = perturbations.ITPA_density(n0=n0, c=c)

    maxwellian = CanonicalMaxwellian(n=(0.0, pert), equil=mhd_equil, volume_form=False)

    e1 = np.linspace(0.0, 1.0, Nel[0])
    e2 = np.linspace(0.0, 1.0, Nel[1])
    e3 = np.linspace(0.0, 1.0, Nel[2])

    eta_meshgrid = np.meshgrid(e1, e2, e3)

    v_para = 0.01
    v_perp = 0.01

    absB = mhd_equil.absB0(*eta_meshgrid)[0, :, 0]

    # magnetic moment
    mu = v_perp**2 / 2.0 / absB

    # total energy
    energy = 1 / 2 * v_para**2 + mu * absB

    # shifted canonical toroidal momentum
    r = eta_meshgrid[0] * (1 - a1) + a1

    psi = mhd_equil.psi_r(r[0, :, 0])

    psic = psi - epsilon * B0 * R0 / absB * v_para
    psic += epsilon * np.sign(v_para) * np.sqrt(2 * (energy - mu * B0)) * R0 * np.heaviside(energy - mu * B0, 0)

    com_meshgrids = np.meshgrid(energy, mu, psic)
    res = maxwellian(energy, mu, psic).squeeze()

    # calculate rc
    rc = maxwellian.rc(psic)

    ana_res = n0 * c[3] * np.exp(-c[2] / c[1] * np.tanh((rc - c[0]) / c[2]))
    ana_res *= 2 * np.sqrt(energy / np.pi) / maxw_params["vth"] ** 3 * np.exp(-energy / maxw_params["vth"] ** 2)

    if show_plot:
        plt.plot(e1, ana_res, label="analytical")
        plt.plot(e1, res, "r*", label="CanonicalMaxwellian Class")
        plt.legend()
        plt.title("Test ITPA perturbation in density")
        plt.xlabel("eta_1")
        plt.ylabel("f(eta_1)")
        plt.show()

    assert np.allclose(res, ana_res, atol=10e-10), f"{res=},\n {ana_res}"


if __name__ == "__main__":
    # test_maxwellian_3d_uniform(Nel=[64, 1, 1], show_plot=True)
    # test_maxwellian_3d_perturbed(Nel=[64, 1, 1], show_plot=True)
    # test_maxwellian_3d_mhd(Nel=[8, 11, 12], with_desc=None, show_plot=False)
    # test_maxwellian_2d_uniform(Nel=[64, 1, 1], show_plot=True)
    # test_maxwellian_2d_perturbed(Nel=[64, 1, 1], show_plot=True)
    # test_maxwellian_2d_mhd(Nel=[8, 12, 12], with_desc=None, show_plot=False)
    test_canonical_maxwellian_uniform(Nel=[64, 1, 1], show_plot=True)
