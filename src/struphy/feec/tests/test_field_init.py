import pytest


@pytest.mark.parametrize("Nel", [[8, 10, 12]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [True, True, False]])
@pytest.mark.parametrize("spaces", [["H1", "Hcurl", "Hdiv"], ["Hdiv", "L2"], ["H1vec"]])
@pytest.mark.parametrize("vec_comps", [[True, True, False], [False, True, True]])
def test_bckgr_init_const(Nel, p, spl_kind, spaces, vec_comps):
    """Test field background initialization of "LogicalConst" with multiple fields in params."""

    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import FieldsBackground

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # evaluation grids for comparisons
    e1 = xp.linspace(0.0, 1.0, Nel[0])
    e2 = xp.linspace(0.0, 1.0, Nel[1])
    e3 = xp.linspace(0.0, 1.0, Nel[2])
    meshgrids = xp.meshgrid(e1, e2, e3, indexing="ij")

    # test values
    xp.random.seed(1234)
    val = xp.random.rand()
    if val > 0.5:
        val = int(val * 10)

    # test
    for i, space in enumerate(spaces):
        field = derham.create_spline_function("name_" + str(i), space)
        if space in ("H1", "L2"):
            background = FieldsBackground(type="LogicalConst", values=(val,))
            field.initialize_coeffs(backgrounds=background)
            print(
                f"\n{rank =}, {space =}, after init:\n {xp.max(xp.abs(field(*meshgrids) - val)) =}",
            )
            # print(f'{field(*meshgrids) = }')
            assert xp.allclose(field(*meshgrids), val)
        else:
            background = FieldsBackground(type="LogicalConst", values=(val, None, val))
            field.initialize_coeffs(backgrounds=background)
            for j, val in enumerate(background.values):
                if val is not None:
                    print(
                        f"\n{rank =}, {space =}, after init:\n {j =}, {xp.max(xp.abs(field(*meshgrids)[j] - val)) =}",
                    )
                    # print(f'{field(*meshgrids)[i] = }')
                    assert xp.allclose(field(*meshgrids)[j], val)


@pytest.mark.parametrize("Nel", [[18, 24, 12]])
@pytest.mark.parametrize("p", [[1, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
def test_bckgr_init_mhd(Nel, p, spl_kind, with_desc=False, with_gvec=False, show_plot=False):
    """Test field background initialization of "MHD" with multiple fields in params."""

    import inspect

    import cunumpy as xp
    from matplotlib import pyplot as plt
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background import equils
    from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB
    from struphy.geometry import domains
    from struphy.io.options import FieldsBackground

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # background parameters
    bckgr_0 = FieldsBackground(type="FluidEquilibrium", variable="absB0")
    bckgr_1 = FieldsBackground(type="FluidEquilibrium", variable="u1")
    bckgr_2 = FieldsBackground(type="FluidEquilibrium", variable="u2")
    bckgr_3 = FieldsBackground(type="FluidEquilibrium", variable="p3")
    bckgr_4 = FieldsBackground(type="FluidEquilibrium", variable="uv")

    # evaluation grids for comparisons
    e1 = xp.linspace(0.0, 1.0, Nel[0])
    e2 = xp.linspace(0.0, 1.0, Nel[1])
    e3 = xp.linspace(0.0, 1.0, Nel[2])
    meshgrids = xp.meshgrid(e1, e2, e3, indexing="ij")

    # test
    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and val.__module__ == equils.__name__:
            print(f"{key =}")
            if "DESC" in key and not with_desc:
                print(f"Attention: {with_desc =}, DESC not tested here !!")
                continue

            if "GVEC" in key and not with_gvec:
                print(f"Attention: {with_gvec =}, GVEC not tested here !!")
                continue

            mhd_equil = val()
            if not isinstance(mhd_equil, FluidEquilibriumWithB):
                continue

            print(f"{mhd_equil.params =}")

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
                    print(f"Not setting domain for {key}.")

            field_0 = derham.create_spline_function(
                "name_0",
                "H1",
                backgrounds=bckgr_0,
                equil=mhd_equil,
            )
            field_1 = derham.create_spline_function(
                "name_1",
                "Hcurl",
                backgrounds=bckgr_1,
                equil=mhd_equil,
            )
            field_2 = derham.create_spline_function(
                "name_2",
                "Hdiv",
                backgrounds=bckgr_2,
                equil=mhd_equil,
            )
            field_3 = derham.create_spline_function(
                "name_3",
                "L2",
                backgrounds=bckgr_3,
                equil=mhd_equil,
            )
            field_4 = derham.create_spline_function(
                "name_4",
                "H1vec",
                backgrounds=bckgr_4,
                equil=mhd_equil,
            )

            # scalar spaces
            print(
                f"{xp.max(xp.abs(field_3(*meshgrids) - mhd_equil.p3(*meshgrids))) / xp.max(xp.abs(mhd_equil.p3(*meshgrids)))}",
            )
            assert (
                xp.max(
                    xp.abs(field_3(*meshgrids) - mhd_equil.p3(*meshgrids)),
                )
                / xp.max(xp.abs(mhd_equil.p3(*meshgrids)))
                < 0.54
            )

            if isinstance(mhd_equil, FluidEquilibriumWithB):
                print(
                    f"{xp.max(xp.abs(field_0(*meshgrids) - mhd_equil.absB0(*meshgrids))) / xp.max(xp.abs(mhd_equil.absB0(*meshgrids)))}",
                )
                assert (
                    xp.max(
                        xp.abs(field_0(*meshgrids) - mhd_equil.absB0(*meshgrids)),
                    )
                    / xp.max(xp.abs(mhd_equil.absB0(*meshgrids)))
                    < 0.057
                )
            print("Scalar asserts passed.")

            # vector-valued spaces
            ref = mhd_equil.u1(*meshgrids)
            if xp.max(xp.abs(ref[0])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[0]))
            print(
                f"{xp.max(xp.abs(field_1(*meshgrids)[0] - ref[0])) / denom =}",
            )
            assert xp.max(xp.abs(field_1(*meshgrids)[0] - ref[0])) / denom < 0.28
            if xp.max(xp.abs(ref[1])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[1]))
            print(
                f"{xp.max(xp.abs(field_1(*meshgrids)[1] - ref[1])) / denom =}",
            )
            assert xp.max(xp.abs(field_1(*meshgrids)[1] - ref[1])) / denom < 0.33
            if xp.max(xp.abs(ref[2])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[2]))
            print(
                f"{xp.max(xp.abs(field_1(*meshgrids)[2] - ref[2])) / denom =}",
            )
            assert (
                xp.max(
                    xp.abs(
                        field_1(*meshgrids)[2] - ref[2],
                    ),
                )
                / denom
                < 0.1
            )
            print("u1 asserts passed.")

            ref = mhd_equil.u2(*meshgrids)
            if xp.max(xp.abs(ref[0])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[0]))
            print(
                f"{xp.max(xp.abs(field_2(*meshgrids)[0] - ref[0])) / denom =}",
            )
            assert xp.max(xp.abs(field_2(*meshgrids)[0] - ref[0])) / denom < 0.86
            if xp.max(xp.abs(ref[1])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[1]))
            print(
                f"{xp.max(xp.abs(field_2(*meshgrids)[1] - ref[1])) / denom =}",
            )
            assert (
                xp.max(
                    xp.abs(
                        field_2(*meshgrids)[1] - ref[1],
                    ),
                )
                / denom
                < 0.4
            )
            if xp.max(xp.abs(ref[2])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[2]))
            print(
                f"{xp.max(xp.abs(field_2(*meshgrids)[2] - ref[2])) / denom =}",
            )
            assert xp.max(xp.abs(field_2(*meshgrids)[2] - ref[2])) / denom < 0.21
            print("u2 asserts passed.")

            ref = mhd_equil.uv(*meshgrids)
            if xp.max(xp.abs(ref[0])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[0]))
            print(
                f"{xp.max(xp.abs(field_4(*meshgrids)[0] - ref[0])) / denom =}",
            )
            assert xp.max(xp.abs(field_4(*meshgrids)[0] - ref[0])) / denom < 0.6
            if xp.max(xp.abs(ref[1])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[1]))
            print(
                f"{xp.max(xp.abs(field_4(*meshgrids)[1] - ref[1])) / denom =}",
            )
            assert (
                xp.max(
                    xp.abs(
                        field_4(*meshgrids)[1] - ref[1],
                    ),
                )
                / denom
                < 0.2
            )
            if xp.max(xp.abs(ref[2])) < 1e-11:
                denom = 1.0
            else:
                denom = xp.max(xp.abs(ref[2]))
            print(
                f"{xp.max(xp.abs(field_4(*meshgrids)[2] - ref[2])) / denom =}",
            )
            assert (
                xp.max(
                    xp.abs(
                        field_4(*meshgrids)[2] - ref[2],
                    ),
                )
                / denom
                < 0.04
            )
            print("uv asserts passed.")

            # plotting fields with equilibrium
            if show_plot and rank == 0:
                plt.figure(f"0/3-forms top, {mhd_equil =}", figsize=(24, 16))
                plt.figure(
                    f"0/3-forms poloidal, {mhd_equil =}",
                    figsize=(24, 16),
                )
                plt.figure(f"1-forms top, {mhd_equil =}", figsize=(24, 16))
                plt.figure(
                    f"1-forms poloidal, {mhd_equil =}",
                    figsize=(24, 16),
                )
                plt.figure(f"2-forms top, {mhd_equil =}", figsize=(24, 16))
                plt.figure(
                    f"2-forms poloidal, {mhd_equil =}",
                    figsize=(24, 16),
                )
                plt.figure(
                    f"vector-fields top, {mhd_equil =}",
                    figsize=(24, 16),
                )
                plt.figure(
                    f"vector-fields poloidal, {mhd_equil =}",
                    figsize=(24, 16),
                )
                x, y, z = mhd_equil.domain(*meshgrids)

                # 0-form
                if isinstance(mhd_equil, FluidEquilibriumWithB):
                    absB0_h = mhd_equil.domain.push(field_0, *meshgrids)
                    absB0 = mhd_equil.domain.push(mhd_equil.absB0, *meshgrids)

                    levels = xp.linspace(xp.min(absB0) - 1e-10, xp.max(absB0), 20)

                    plt.figure(f"0/3-forms top, {mhd_equil =}")
                    plt.subplot(2, 3, 1)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            absB0_h[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            absB0_h[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            absB0_h[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            absB0_h[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("Equilibrium $|B_0|$, top view (e1-e3)")
                    plt.subplot(2, 3, 3 + 1)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            absB0[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            absB0[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            absB0[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            absB0[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, top view (e1-e3)")

                    plt.figure(f"0/3-forms poloidal, {mhd_equil =}")
                    plt.subplot(2, 3, 1)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            absB0_h[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            absB0_h[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("Equilibrium $|B_0|$, poloidal view (e1-e2)")
                    plt.subplot(2, 3, 3 + 1)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            absB0[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            absB0[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, poloidal view (e1-e2)")

                # 3-form
                p3_h = mhd_equil.domain.push(field_3, *meshgrids)
                p3 = mhd_equil.domain.push(mhd_equil.p3, *meshgrids)

                levels = xp.linspace(xp.min(p3) - 1e-10, xp.max(p3), 20)

                plt.figure(f"0/3-forms top, {mhd_equil =}")
                plt.subplot(2, 3, 2)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(
                        x[:, 0, :],
                        z[:, 0, :],
                        p3_h[:, 0, :],
                        levels=levels,
                    )
                    plt.contourf(
                        x[:, Nel[1] // 2, :],
                        z[
                            :,
                            Nel[1] // 2 - 1,
                            :,
                        ],
                        p3_h[:, Nel[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(
                        x[:, 0, :],
                        y[:, 0, :],
                        p3_h[:, 0, :],
                        levels=levels,
                    )
                    plt.contourf(
                        x[:, Nel[1] // 2, :],
                        y[
                            :,
                            Nel[1] // 2 - 1,
                            :,
                        ],
                        p3_h[:, Nel[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Equilibrium $p_0$, top view (e1-e3)")
                plt.subplot(2, 3, 3 + 2)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(
                        x[:, 0, :],
                        z[:, 0, :],
                        p3[:, 0, :],
                        levels=levels,
                    )
                    plt.contourf(
                        x[:, Nel[1] // 2, :],
                        z[
                            :,
                            Nel[1] // 2 - 1,
                            :,
                        ],
                        p3[:, Nel[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                else:
                    plt.contourf(
                        x[:, 0, :],
                        y[:, 0, :],
                        p3[:, 0, :],
                        levels=levels,
                    )
                    plt.contourf(
                        x[:, Nel[1] // 2, :],
                        y[
                            :,
                            Nel[1] // 2 - 1,
                            :,
                        ],
                        p3[:, Nel[1] // 2, :],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                plt.axis("equal")
                plt.colorbar()
                plt.title("reference, top view (e1-e3)")

                plt.figure(f"0/3-forms poloidal, {mhd_equil =}")
                plt.subplot(2, 3, 2)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(
                        x[:, :, 0],
                        y[:, :, 0],
                        p3_h[:, :, 0],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(
                        x[:, :, 0],
                        z[:, :, 0],
                        p3_h[:, :, 0],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("Equilibrium $p_0$, poloidal view (e1-e2)")
                plt.subplot(2, 3, 3 + 2)
                if "Slab" in key or "Pinch" in key:
                    plt.contourf(
                        x[:, :, 0],
                        y[:, :, 0],
                        p3[:, :, 0],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("y")
                else:
                    plt.contourf(
                        x[:, :, 0],
                        z[:, :, 0],
                        p3[:, :, 0],
                        levels=levels,
                    )
                    plt.xlabel("x")
                    plt.ylabel("z")
                plt.axis("equal")
                plt.colorbar()
                plt.title("reference, poloidal view (e1-e2)")

                # 1-form magnetic field plots
                b1h = mhd_equil.domain.push(
                    field_1(*meshgrids),
                    *meshgrids,
                    kind="1",
                )
                b1 = mhd_equil.domain.push(
                    [*mhd_equil.u1(*meshgrids)],
                    *meshgrids,
                    kind="1",
                )

                for i, (bh, b) in enumerate(zip(b1h, b1)):
                    levels = xp.linspace(xp.min(b) - 1e-10, xp.max(b), 20)

                    plt.figure(f"1-forms top, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Equilibrium $B_{i + 1}$, top view (e1-e3)")
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, top view (e1-e3)")

                    plt.figure(f"1-forms poloidal, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(
                        f"Equilibrium $B_{i + 1}$, poloidal view (e1-e2)",
                    )
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, poloidal view (e1-e2)")

                # 2-form magnetic field plots
                b2h = mhd_equil.domain.push(
                    field_2(*meshgrids),
                    *meshgrids,
                    kind="2",
                )
                b2 = mhd_equil.domain.push(
                    [*mhd_equil.u2(*meshgrids)],
                    *meshgrids,
                    kind="2",
                )

                for i, (bh, b) in enumerate(zip(b2h, b2)):
                    levels = xp.linspace(xp.min(b) - 1e-10, xp.max(b), 20)

                    plt.figure(f"2-forms top, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Equilibrium $B_{i + 1}$, top view (e1-e3)")
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, top view (e1-e3)")

                    plt.figure(f"2-forms poloidal, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(
                        f"Equilibrium $B_{i + 1}$, poloidal view (e1-e2)",
                    )
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, poloidal view (e1-e2)")

                # vector-field magnetic field plots
                bvh = mhd_equil.domain.push(
                    field_4(*meshgrids),
                    *meshgrids,
                    kind="v",
                )
                bv = mhd_equil.domain.push(
                    [*mhd_equil.uv(*meshgrids)],
                    *meshgrids,
                    kind="v",
                )

                for i, (bh, b) in enumerate(zip(bvh, bv)):
                    levels = xp.linspace(xp.min(b) - 1e-10, xp.max(b), 20)

                    plt.figure(f"vector-fields top, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            bh[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            bh[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(f"Equilibrium $B_{i + 1}$, top view (e1-e3)")
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, 0, :],
                            z[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            z[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    else:
                        plt.contourf(
                            x[:, 0, :],
                            y[:, 0, :],
                            b[:, 0, :],
                            levels=levels,
                        )
                        plt.contourf(
                            x[:, Nel[1] // 2, :],
                            y[
                                :,
                                Nel[1] // 2 - 1,
                                :,
                            ],
                            b[:, Nel[1] // 2, :],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, top view (e1-e3)")

                    plt.figure(f"vector-fields poloidal, {mhd_equil =}")
                    plt.subplot(2, 3, 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            bh[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title(
                        f"Equilibrium $B_{i + 1}$, poloidal view (e1-e2)",
                    )
                    plt.subplot(2, 3, 3 + 1 + i)
                    if "Slab" in key or "Pinch" in key:
                        plt.contourf(
                            x[:, :, 0],
                            y[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        plt.contourf(
                            x[:, :, 0],
                            z[:, :, 0],
                            b[:, :, 0],
                            levels=levels,
                        )
                        plt.xlabel("x")
                        plt.ylabel("z")
                    plt.axis("equal")
                    plt.colorbar()
                    plt.title("reference, poloidal view (e1-e2)")

                plt.show()


@pytest.mark.parametrize("Nel", [[1, 32, 32]])
@pytest.mark.parametrize("p", [[1, 3, 3]])
@pytest.mark.parametrize("spl_kind", [[True, True, True]])
def test_sincos_init_const(Nel, p, spl_kind, show_plot=False):
    """Test field perturbation with ModesSin + ModesCos on top of of "LogicalConst" with multiple fields in params."""

    import cunumpy as xp
    from matplotlib import pyplot as plt
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.initial.perturbations import ModesCos, ModesSin
    from struphy.io.options import FieldsBackground

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # background parameters
    avg_0 = (1.2,)
    avg_1 = (0.0, 2.6, 3.7)
    avg_2 = (2, 3, 4.2)

    bckgr_0 = FieldsBackground(type="LogicalConst", values=avg_0)
    bckgr_1 = FieldsBackground(type="LogicalConst", values=avg_1)
    bckgr_2 = FieldsBackground(type="LogicalConst", values=avg_2)

    # perturbations
    ms_s = [0, 2]
    ns_s = [1, 1]
    amps = [0.2]
    f_sin_0 = ModesSin(ms=ms_s, ns=ns_s, amps=amps)
    f_sin_11 = ModesSin(ms=ms_s, ns=ns_s, amps=amps, given_in_basis="1", comp=0)
    f_sin_13 = ModesSin(ms=ms_s, ns=ns_s, amps=amps, given_in_basis="1", comp=2)

    ms_c = [1]
    ns_c = [0]
    f_cos_0 = ModesCos(ms=ms_c, ns=ns_c, amps=amps)
    f_cos_11 = ModesCos(ms=ms_c, ns=ns_c, amps=amps, given_in_basis="1", comp=0)
    f_cos_12 = ModesCos(ms=ms_c, ns=ns_c, amps=amps, given_in_basis="1", comp=1)
    f_cos_22 = ModesCos(ms=ms_c, ns=ns_c, amps=amps, given_in_basis="2", comp=1)

    pert_params_0 = {
        "ModesSin": {
            "given_in_basis": "0",
            "ms": ms_s,
            "ns": ns_s,
            "amps": amps,
        },
        "ModesCos": {
            "given_in_basis": "0",
            "ms": ms_c,
            "ns": ns_c,
            "amps": amps,
        },
    }

    pert_params_1 = {
        "ModesSin": {
            "given_in_basis": ["1", None, "1"],
            "ms": [ms_s, None, ms_s],
            "ns": [ns_s, None, ns_s],
            "amps": [amps, None, amps],
        },
        "ModesCos": {
            "given_in_basis": ["1", "1", None],
            "ms": [ms_c, ms_c, None],
            "ns": [ns_c, ns_c, None],
            "amps": [amps, amps, None],
        },
    }

    pert_params_2 = {
        "ModesCos": {
            "given_in_basis": [None, "2", None],
            "ms": [None, ms_c, None],
            "ns": [None, ns_c, None],
            "amps": [None, amps, None],
        },
    }

    # Psydac discrete Derham sequence and fields
    derham = Derham(Nel, p, spl_kind, comm=comm)

    field_0 = derham.create_spline_function("name_0", "H1", backgrounds=bckgr_0, perturbations=[f_sin_0, f_cos_0])
    field_1 = derham.create_spline_function(
        "name_1",
        "Hcurl",
        backgrounds=bckgr_1,
        perturbations=[f_sin_11, f_sin_13, f_cos_11, f_cos_12],
    )
    field_2 = derham.create_spline_function("name_2", "Hdiv", backgrounds=bckgr_2, perturbations=[f_cos_22])

    # evaluation grids for comparisons
    e1 = xp.linspace(0.0, 1.0, Nel[0])
    e2 = xp.linspace(0.0, 1.0, Nel[1])
    e3 = xp.linspace(0.0, 1.0, Nel[2])
    meshgrids = xp.meshgrid(e1, e2, e3, indexing="ij")

    fun_0 = avg_0 + f_sin_0(*meshgrids) + f_cos_0(*meshgrids)

    fun_1 = [
        avg_1[0] + f_sin_11(*meshgrids) + f_cos_11(*meshgrids),
        avg_1[1] + f_cos_12(*meshgrids),
        avg_1[2] + f_sin_13(*meshgrids),
    ]
    fun_2 = [
        avg_2[0] + 0.0 * meshgrids[0],
        avg_2[1] + f_cos_22(*meshgrids),
        avg_2[2] + 0.0 * meshgrids[0],
    ]

    f0_h = field_0(*meshgrids)
    f1_h = field_1(*meshgrids)
    f2_h = field_2(*meshgrids)

    print(f"{xp.max(xp.abs(fun_0 - f0_h)) =}")
    print(f"{xp.max(xp.abs(fun_1[0] - f1_h[0])) =}")
    print(f"{xp.max(xp.abs(fun_1[1] - f1_h[1])) =}")
    print(f"{xp.max(xp.abs(fun_1[2] - f1_h[2])) =}")
    print(f"{xp.max(xp.abs(fun_2[0] - f2_h[0])) =}")
    print(f"{xp.max(xp.abs(fun_2[1] - f2_h[1])) =}")
    print(f"{xp.max(xp.abs(fun_2[2] - f2_h[2])) =}")

    assert xp.max(xp.abs(fun_0 - f0_h)) < 3e-5
    assert xp.max(xp.abs(fun_1[0] - f1_h[0])) < 3e-5
    assert xp.max(xp.abs(fun_1[1] - f1_h[1])) < 3e-5
    assert xp.max(xp.abs(fun_1[2] - f1_h[2])) < 3e-5
    assert xp.max(xp.abs(fun_2[0] - f2_h[0])) < 3e-5
    assert xp.max(xp.abs(fun_2[1] - f2_h[1])) < 3e-5
    assert xp.max(xp.abs(fun_2[2] - f2_h[2])) < 3e-5

    if show_plot and rank == 0:
        levels = xp.linspace(xp.min(fun_0) - 1e-10, xp.max(fun_0), 40)

        plt.figure("0-form", figsize=(10, 16))
        plt.subplot(2, 1, 1)
        plt.contourf(
            meshgrids[1][0, :, :],
            meshgrids[2][0, :, :],
            f0_h[0, :, :],
            levels=levels,
        )
        plt.xlabel("$\\eta_2$")
        plt.ylabel("$\\eta_3$")
        plt.xlim([0, 1.0])
        plt.title("field_0")
        plt.axis("equal")
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.contourf(
            meshgrids[1][0, :, :],
            meshgrids[2][0, :, :],
            fun_0[0, :, :],
            levels=levels,
        )
        plt.xlabel("$\\eta_2$")
        plt.ylabel("$\\eta_3$")
        plt.title("reference")
        # plt.figure('1-form', figsize=(24, 16))
        # plt.figure('2-form', figsize=(24, 16))
        plt.axis("equal")
        plt.colorbar()

        plt.figure("1-form", figsize=(30, 16))
        for i, (f_h, fun) in enumerate(zip(f1_h, fun_1)):
            levels = xp.linspace(xp.min(fun) - 1e-10, xp.max(fun), 40)

            plt.subplot(2, 3, 1 + i)
            plt.contourf(
                meshgrids[1][0, :, :],
                meshgrids[2][0, :, :],
                f_h[0, :, :],
                levels=levels,
            )
            plt.xlabel("$\\eta_2$")
            plt.ylabel("$\\eta_3$")
            plt.xlim([0, 1.0])
            plt.title(f"field_1, component {i + 1}")
            plt.axis("equal")
            plt.colorbar()
            plt.subplot(2, 3, 4 + i)
            plt.contourf(
                meshgrids[1][0, :, :],
                meshgrids[2][0, :, :],
                fun[0, :, :],
                levels=levels,
            )
            plt.xlabel("$\\eta_2$")
            plt.ylabel("$\\eta_3$")
            plt.title("reference")
            # plt.figure('1-form', figsize=(24, 16))
            # plt.figure('2-form', figsize=(24, 16))
            plt.axis("equal")
            plt.colorbar()

        plt.figure("2-form", figsize=(30, 16))
        for i, (f_h, fun) in enumerate(zip(f2_h, fun_2)):
            levels = xp.linspace(xp.min(fun) - 1e-10, xp.max(fun), 40)

            plt.subplot(2, 3, 1 + i)
            plt.contourf(
                meshgrids[1][0, :, :],
                meshgrids[2][0, :, :],
                f_h[0, :, :],
                levels=levels,
            )
            plt.xlabel("$\\eta_2$")
            plt.ylabel("$\\eta_3$")
            plt.xlim([0, 1.0])
            plt.title(f"field_2, component {i + 1}")
            plt.axis("equal")
            plt.colorbar()
            plt.subplot(2, 3, 4 + i)
            plt.contourf(
                meshgrids[1][0, :, :],
                meshgrids[2][0, :, :],
                fun[0, :, :],
                levels=levels,
            )
            plt.xlabel("$\\eta_2$")
            plt.ylabel("$\\eta_3$")
            plt.title("reference")
            # plt.figure('1-form', figsize=(24, 16))
            # plt.figure('2-form', figsize=(24, 16))
            plt.axis("equal")
            plt.colorbar()

        plt.show()


@pytest.mark.parametrize("Nel", [[8, 10, 12]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [True, False, True]])
@pytest.mark.parametrize("space", ["Hcurl", "Hdiv", "H1vec"])
@pytest.mark.parametrize("direction", ["e1", "e2", "e3"])
def test_noise_init(Nel, p, spl_kind, space, direction):
    """Only tests 1d noise ('e1', 'e2', 'e3') !!"""

    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays
    from struphy.initial.perturbations import Noise

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)
    field = derham.create_spline_function("field", space)

    derham_np = Derham(Nel, p, spl_kind, comm=None)
    field_np = derham_np.create_spline_function("field", space)

    # initial conditions
    pert = Noise(direction=direction, amp=0.0001, seed=1234, comp=0)

    field.initialize_coeffs(perturbations=pert)
    field_np.initialize_coeffs(perturbations=pert)

    # print('#'*80)
    # print(f'npts={field.vector[0].space.npts}, npts_np={field_np.vector[0].space.npts}')
    # print(f'rank={rank}: nprocs={derham.domain_array[rank]}')
    # print(f'rank={rank}, field={field.vector[0].toarray_local().shape}, field_np={field_np.vector[0].toarray_local().shape}')
    # print(f'rank={rank}: \ncomp{0}={field.vector[0].toarray_local()}, \ncomp{0}_np={field_np.vector[0].toarray_local()}')

    compare_arrays(
        field.vector,
        [field_np.vector[n].toarray_local() for n in range(3)],
        rank,
    )


if __name__ == "__main__":
    # test_bckgr_init_const([8, 10, 12], [1, 2, 3], [False, False, True], [
    #     'H1', 'Hcurl', 'Hdiv'], [True, True, False])
    # test_bckgr_init_mhd(
    #     [18, 24, 12],
    #     [1, 2, 1],
    #     [
    #         False,
    #         True,
    #         True,
    #     ],
    #     show_plot=False,
    # )
    test_sincos_init_const([1, 32, 32], [1, 3, 3], [True] * 3, show_plot=True)
    test_noise_init([4, 8, 6], [1, 1, 1], [True, True, True], "Hcurl", "e1")
