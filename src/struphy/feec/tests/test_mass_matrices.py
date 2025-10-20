import pytest


@pytest.mark.parametrize("Nel", [[5, 6, 7]])
@pytest.mark.parametrize("p", [[2, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [True, False, True]])
@pytest.mark.parametrize(
    "dirichlet_bc",
    [None, [[False, True], [True, False], [False, False]], [[True, False], [False, True], [False, False]]],
)
@pytest.mark.parametrize("mapping", [["Colella", {"Lx": 1.0, "Ly": 6.0, "alpha": 0.1, "Lz": 10.0}]])
def test_mass(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Compare Struphy mass matrices to Struphy-legacy mass matrices."""

    from psydac.ddm.mpi import mpi as MPI

    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.mass import WeightedMassOperators, WeightedMassOperatorsOldForTesting
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import RotationMatrix, compare_arrays, create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch, ShearedSlab
    from struphy.geometry import domains
    from struphy.utils.arrays import xp as np

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f"Rank {mpi_rank} | Start test_mass with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show()

    # load MHD equilibrium
    if mapping[0] == "Cuboid":
        eq_mhd = ShearedSlab(
            **{
                "a": (mapping[1]["r1"] - mapping[1]["l1"]),
                "R0": (mapping[1]["r3"] - mapping[1]["l3"]) / (2 * np.pi),
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

    elif mapping[0] == "Colella":
        eq_mhd = ShearedSlab(
            **{
                "a": mapping[1]["Lx"],
                "R0": mapping[1]["Lz"] / (2 * np.pi),
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == "HollowCylinder":
        eq_mhd = ScrewPinch(
            **{
                "a": mapping[1]["a2"],
                "R0": 3.0,
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

        if show_plots:
            eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space
    if dirichlet_bc is not None:
        for i, knd in enumerate(spl_kind):
            if knd:
                dirichlet_bc[i] = [False, False]
    else:
        dirichlet_bc = [[False, False]] * 3

    print(f"{dirichlet_bc = }")

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc)

    print(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    fem_spaces = [derham.Vh_fem["0"], derham.Vh_fem["1"], derham.Vh_fem["2"], derham.Vh_fem["3"], derham.Vh_fem["v"]]

    # mass matrices object
    mass_matsold = WeightedMassOperatorsOldForTesting(derham, domain, eq_mhd=eq_mhd)
    mass_matsold_free = WeightedMassOperatorsOldForTesting(derham, domain, eq_mhd=eq_mhd, matrix_free=True)
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    mass_mats_free = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd, matrix_free=True)

    # test calling the diagonal method
    aaa = mass_mats.M0.matrix.diagonal()
    bbb = mass_mats.M1.matrix.diagonal()
    print(f"{aaa = }, {bbb[0, 0] = }, {bbb[0, 1] = }")

    # compare to old STRUPHY
    bc_old = [[None, None], [None, None], [None, None]]
    for i in range(3):
        for j in range(2):
            if dirichlet_bc[i][j]:
                bc_old[i][j] = "d"
            else:
                bc_old[i][j] = "f"

    spaces = [
        Spline_space_1d(Nel[0], p[0], spl_kind[0], p[0] + 1, bc_old[0]),
        Spline_space_1d(Nel[1], p[1], spl_kind[1], p[1] + 1, bc_old[1]),
        Spline_space_1d(Nel[2], p[2], spl_kind[2], p[2] + 1, bc_old[2]),
    ]

    spaces[0].set_projectors()
    spaces[1].set_projectors()
    spaces[2].set_projectors()

    space = Tensor_spline_space(spaces)
    space.set_projectors("general")

    space.assemble_Mk(domain, "V0")
    space.assemble_Mk(domain, "V1")
    space.assemble_Mk(domain, "V2")
    space.assemble_Mk(domain, "V3")
    space.assemble_Mk(domain, "Vv")

    mhd_ops_str = MHDOperators(space, eq_mhd, 2)

    mhd_ops_str.assemble_Mn()
    mhd_ops_str.assemble_MJ()

    mhd_ops_str.set_operators()

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(fem_spaces[0], seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(fem_spaces[2], seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(fem_spaces[3], seed=8196, flattened=True)
    xv_str, xv_psy = create_equal_random_arrays(fem_spaces[4], seed=2038, flattened=True)

    x0_str0 = space.B0.dot(x0_str)
    x1_str0 = space.B1.dot(x1_str)
    x2_str0 = space.B2.dot(x2_str)
    x3_str0 = space.B3.dot(x3_str)
    xv_str0 = space.Bv.dot(xv_str)

    # Test toarray and tosparse
    all_false = all(not bc for bl in dirichlet_bc for bc in bl)
    if all_false:
        r2str_toarray = mass_mats.M2.toarray.dot(x2_str)
        r2psy_compare = mass_mats.M2.dot(x2_psy)
        r2str_tosparse = mass_mats.M2.tosparse.dot(x2_str)
        compare_arrays(r2psy_compare, r2str_toarray, mpi_rank, atol=1e-14)
        compare_arrays(r2psy_compare, r2str_tosparse, mpi_rank, atol=1e-14)

    # perfrom matrix-vector products (with boundary conditions)
    r0_str = space.B0.T.dot(space.M0_0(x0_str0))
    r1_str = space.B1.T.dot(space.M1_0(x1_str0))
    r2_str = space.B2.T.dot(space.M2_0(x2_str0))
    r3_str = space.B3.T.dot(space.M3_0(x3_str0))
    rv_str = space.Bv.T.dot(space.Mv_0(xv_str0))

    rn_str = space.B2.T.dot(mhd_ops_str.Mn(x2_str0))
    rJ_str = space.B2.T.dot(mhd_ops_str.MJ(x2_str0))

    r0_psy = mass_mats.M0.dot(x0_psy, apply_bc=True)
    r1_psy = mass_mats.M1.dot(x1_psy, apply_bc=True)
    r2_psy = mass_mats.M2.dot(x2_psy, apply_bc=True)
    r3_psy = mass_mats.M3.dot(x3_psy, apply_bc=True)
    rv_psy = mass_mats.Mv.dot(xv_psy, apply_bc=True)

    rn_psy = mass_mats.M2n.dot(x2_psy, apply_bc=True)
    rJ_psy = mass_mats.M2J.dot(x2_psy, apply_bc=True)

    r1J_psy = mass_mats.M1J.dot(x2_psy, apply_bc=True)
    r1Jold_psy = mass_matsold.M1J.dot(x2_psy, apply_bc=True)

    # How to test space x1_psy? M1J is space HdivHcurl

    rM1Bninv_psy = mass_mats.M1Bninv.dot(x1_psy, apply_bc=True)
    rM1Bninvold_psy = mass_matsold.M1Bninv.dot(x1_psy, apply_bc=True)
    rM0ad_psy = mass_mats.M0ad.dot(x0_psy, apply_bc=True)
    rM0adold_psy = mass_matsold.M0ad.dot(x0_psy, apply_bc=True)
    rM1ninv_psy = mass_mats.M1ninv.dot(x1_psy, apply_bc=True)
    rM1ninvold_psy = mass_matsold.M1ninv.dot(x1_psy, apply_bc=True)
    rM1gyro_psy = mass_mats.M1gyro.dot(x1_psy, apply_bc=True)
    rM1gyroold_psy = mass_matsold.M1gyro.dot(x1_psy, apply_bc=True)
    rM1perp_psy = mass_mats.M1perp.dot(x1_psy, apply_bc=True)
    rM1perpold_psy = mass_matsold.M1perp.dot(x1_psy, apply_bc=True)

    # Change order of input in callable
    rM1ninvswitch_psy = mass_mats.create_weighted_mass(
        "Hcurl", "Hcurl", weights=["sqrt_g", "1/eq_n0", "Ginv"], name="M1ninv", assemble=True
    ).dot(x1_psy, apply_bc=True)

    rot_B = RotationMatrix(
        mass_mats.weights[mass_mats.selected_weight].b2_1,
        mass_mats.weights[mass_mats.selected_weight].b2_2,
        mass_mats.weights[mass_mats.selected_weight].b2_3,
    )
    rM1Bninvswitch_psy = mass_mats.create_weighted_mass(
        "Hcurl", "Hcurl", weights=["1/eq_n0", "sqrt_g", "Ginv", rot_B, "Ginv"], name="M1Bninv", assemble=True
    ).dot(x1_psy, apply_bc=True)

    # Test matrix free operators
    r0_fre = mass_mats_free.M0.dot(x0_psy, apply_bc=True)
    r1_fre = mass_mats_free.M1.dot(x1_psy, apply_bc=True)
    r2_fre = mass_mats_free.M2.dot(x2_psy, apply_bc=True)
    r3_fre = mass_mats_free.M3.dot(x3_psy, apply_bc=True)
    rv_fre = mass_mats_free.Mv.dot(xv_psy, apply_bc=True)

    rn_fre = mass_mats_free.M2n.dot(x2_psy, apply_bc=True)
    rJ_fre = mass_mats_free.M2J.dot(x2_psy, apply_bc=True)

    rM1Bninv_fre = mass_mats_free.M1Bninv.dot(x1_psy, apply_bc=True)
    rM1Bninvold_fre = mass_matsold_free.M1Bninv.dot(x1_psy, apply_bc=True)
    rM0ad_fre = mass_mats_free.M0ad.dot(x0_psy, apply_bc=True)
    rM0adold_fre = mass_matsold_free.M0ad.dot(x0_psy, apply_bc=True)
    rM1ninv_fre = mass_mats_free.M1ninv.dot(x1_psy, apply_bc=True)
    rM1ninvold_fre = mass_matsold_free.M1ninv.dot(x1_psy, apply_bc=True)
    rM1gyro_fre = mass_mats_free.M1gyro.dot(x1_psy, apply_bc=True)
    rM1gyroold_fre = mass_matsold_free.M1gyro.dot(x1_psy, apply_bc=True)
    rM1perp_fre = mass_mats_free.M1perp.dot(x1_psy, apply_bc=True)
    rM1perpold_fre = mass_matsold_free.M1perp.dot(x1_psy, apply_bc=True)

    # Change order of input in callable
    rM1ninvswitch_fre = mass_mats_free.create_weighted_mass(
        "Hcurl", "Hcurl", weights=["sqrt_g", "1/eq_n0", "Ginv"], name="M1ninvswitch", assemble=True
    ).dot(x1_psy, apply_bc=True)
    rot_B = RotationMatrix(
        mass_mats_free.weights[mass_mats_free.selected_weight].b2_1,
        mass_mats_free.weights[mass_mats_free.selected_weight].b2_2,
        mass_mats_free.weights[mass_mats_free.selected_weight].b2_3,
    )

    rM1Bninvswitch_fre = mass_mats_free.create_weighted_mass(
        "Hcurl", "Hcurl", weights=["1/eq_n0", "sqrt_g", "Ginv", rot_B, "Ginv"], name="M1Bninvswitch", assemble=True
    ).dot(x1_psy, apply_bc=True)

    # compare output arrays
    compare_arrays(r0_psy, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_psy, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_psy, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_psy, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_psy, rv_str, mpi_rank, atol=1e-14)

    compare_arrays(rn_psy, rn_str, mpi_rank, atol=1e-14)
    compare_arrays(rJ_psy, rJ_str, mpi_rank, atol=1e-14)

    compare_arrays(r1J_psy, r1Jold_psy.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(r0_fre, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_fre, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_fre, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_fre, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_fre, rv_str, mpi_rank, atol=1e-14)

    compare_arrays(rn_fre, rn_str, mpi_rank, atol=1e-14)
    compare_arrays(rJ_fre, rJ_str, mpi_rank, atol=1e-14)

    compare_arrays(rM1Bninv_psy, rM1Bninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1Bninv_fre, rM1Bninvold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM1ninv_psy, rM1ninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1ninv_fre, rM1ninvold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM1ninvswitch_psy, rM1ninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1ninvswitch_fre, rM1ninvold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM1Bninvswitch_psy, rM1Bninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1Bninvswitch_fre, rM1Bninvold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM0ad_psy, rM0adold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM0ad_fre, rM0adold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM1gyro_psy, rM1gyroold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1gyro_fre, rM1gyroold_fre.toarray(), mpi_rank, atol=1e-14)

    compare_arrays(rM1perp_psy, rM1perpold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1perp_fre, rM1perpold_fre.toarray(), mpi_rank, atol=1e-14)

    # perfrom matrix-vector products (without boundary conditions)
    r0_str = space.M0(x0_str)
    r1_str = space.M1(x1_str)
    r2_str = space.M2(x2_str)
    r3_str = space.M3(x3_str)
    rv_str = space.Mv(xv_str)

    r0_psy = mass_mats.M0.dot(x0_psy, apply_bc=False)
    r1_psy = mass_mats.M1.dot(x1_psy, apply_bc=False)
    r2_psy = mass_mats.M2.dot(x2_psy, apply_bc=False)
    r3_psy = mass_mats.M3.dot(x3_psy, apply_bc=False)
    rv_psy = mass_mats.Mv.dot(xv_psy, apply_bc=False)

    rM1Bninv_psy = mass_mats.M1Bninv.dot(x1_psy, apply_bc=False)
    rM1Bninvold_psy = mass_matsold.M1Bninv.dot(x1_psy, apply_bc=False)
    rM0ad_psy = mass_mats.M0ad.dot(x0_psy, apply_bc=False)
    rM0adold_psy = mass_matsold.M0ad.dot(x0_psy, apply_bc=False)
    rM1ninv_psy = mass_mats.M1ninv.dot(x1_psy, apply_bc=False)
    rM1ninvold_psy = mass_matsold.M1ninv.dot(x1_psy, apply_bc=False)

    r0_fre = mass_mats_free.M0.dot(x0_psy, apply_bc=False)
    r1_fre = mass_mats_free.M1.dot(x1_psy, apply_bc=False)
    r2_fre = mass_mats_free.M2.dot(x2_psy, apply_bc=False)
    r3_fre = mass_mats_free.M3.dot(x3_psy, apply_bc=False)
    rv_fre = mass_mats_free.Mv.dot(xv_psy, apply_bc=False)

    rM1Bninv_fre = mass_mats_free.M1Bninv.dot(x1_psy, apply_bc=False)
    rM1Bninvold_fre = mass_matsold_free.M1Bninv.dot(x1_psy, apply_bc=False)
    rM0ad_fre = mass_mats_free.M0ad.dot(x0_psy, apply_bc=False)
    rM0adold_fre = mass_matsold_free.M0ad.dot(x0_psy, apply_bc=False)
    rM1ninv_fre = mass_mats_free.M1ninv.dot(x1_psy, apply_bc=False)
    rM1ninvold_fre = mass_matsold_free.M1ninv.dot(x1_psy, apply_bc=False)

    # compare output arrays
    compare_arrays(r0_psy, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_psy, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_psy, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_psy, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_psy, rv_str, mpi_rank, atol=1e-14)

    compare_arrays(r0_fre, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_fre, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_fre, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_fre, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_fre, rv_str, mpi_rank, atol=1e-14)

    compare_arrays(rM1Bninv_psy, rM1Bninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1Bninv_fre, rM1Bninvold_fre.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM0ad_psy, rM0adold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM0ad_fre, rM0adold_fre.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1ninv_psy, rM1ninvold_psy.toarray(), mpi_rank, atol=1e-14)
    compare_arrays(rM1ninv_fre, rM1ninvold_fre.toarray(), mpi_rank, atol=1e-14)

    print(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("Nel", [[8, 12, 6]])
@pytest.mark.parametrize("p", [[2, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
@pytest.mark.parametrize(
    "dirichlet_bc",
    [None, [[False, True], [False, False], [False, True]], [[False, False], [False, False], [True, False]]],
)
@pytest.mark.parametrize("mapping", [["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}]])
def test_mass_polar(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Compare Struphy polar mass matrices to Struphy-legacy polar mass matrices."""

    from psydac.ddm.mpi import mpi as MPI

    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch
    from struphy.geometry import domains
    from struphy.polar.basic import PolarVector
    from struphy.utils.arrays import xp as np

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f"Rank {mpi_rank} | Start test_mass_polar with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**{"Nel": Nel[:2], "p": p[:2], "a": mapping[1]["a"], "Lz": mapping[1]["Lz"]})

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show(grid_info=Nel)

    # load MHD equilibrium
    eq_mhd = ScrewPinch(
        **{
            "a": mapping[1]["a"],
            "R0": mapping[1]["Lz"],
            "B0": 1.0,
            "q0": 1.05,
            "q1": 1.8,
            "n1": 3.0,
            "n2": 4.0,
            "na": 0.0,
            "beta": 0.1,
        }
    )

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space
    if dirichlet_bc is not None:
        for i, knd in enumerate(spl_kind):
            if knd:
                dirichlet_bc[i] = [False, False]
    else:
        dirichlet_bc = [[False, False]] * 3

    # derham object
    derham = Derham(
        Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc, with_projectors=False, polar_ck=1, domain=domain
    )

    print(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    bc_old = [[None, None], [None, None], [None, None]]
    for i in range(3):
        for j in range(2):
            if dirichlet_bc[i][j]:
                bc_old[i][j] = "d"
            else:
                bc_old[i][j] = "f"

    spaces = [
        Spline_space_1d(Nel[0], p[0], spl_kind[0], p[0] + 1, bc_old[0]),
        Spline_space_1d(Nel[1], p[1], spl_kind[1], p[1] + 1, bc_old[1]),
        Spline_space_1d(Nel[2], p[2], spl_kind[2], p[2] + 1, bc_old[2]),
    ]

    spaces[0].set_projectors()
    spaces[1].set_projectors()
    spaces[2].set_projectors()

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors("general")

    space.assemble_Mk(domain, "V0")
    space.assemble_Mk(domain, "V1")
    space.assemble_Mk(domain, "V2")
    space.assemble_Mk(domain, "V3")

    mhd_ops_str = MHDOperators(space, eq_mhd, 2)

    mhd_ops_str.assemble_Mn()
    mhd_ops_str.assemble_MJ()

    mhd_ops_str.set_operators()

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(derham.Vh_fem["0"], seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(derham.Vh_fem["1"], seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(derham.Vh_fem["2"], seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(derham.Vh_fem["3"], seed=8196, flattened=True)

    # set polar vectors
    x0_pol_psy = PolarVector(derham.Vh_pol["0"])
    x1_pol_psy = PolarVector(derham.Vh_pol["1"])
    x2_pol_psy = PolarVector(derham.Vh_pol["2"])
    x3_pol_psy = PolarVector(derham.Vh_pol["3"])

    x0_pol_psy.tp = x0_psy
    x1_pol_psy.tp = x1_psy
    x2_pol_psy.tp = x2_psy
    x3_pol_psy.tp = x3_psy

    np.random.seed(1607)
    x0_pol_psy.pol = [np.random.rand(x0_pol_psy.pol[0].shape[0], x0_pol_psy.pol[0].shape[1])]
    x1_pol_psy.pol = [np.random.rand(x1_pol_psy.pol[n].shape[0], x1_pol_psy.pol[n].shape[1]) for n in range(3)]
    x2_pol_psy.pol = [np.random.rand(x2_pol_psy.pol[n].shape[0], x2_pol_psy.pol[n].shape[1]) for n in range(3)]
    x3_pol_psy.pol = [np.random.rand(x3_pol_psy.pol[0].shape[0], x3_pol_psy.pol[0].shape[1])]

    # apply boundary conditions to old STRUPHY
    x0_pol_str = x0_pol_psy.toarray(True)
    x1_pol_str = x1_pol_psy.toarray(True)
    x2_pol_str = x2_pol_psy.toarray(True)
    x3_pol_str = x3_pol_psy.toarray(True)

    x0_pol_str0 = space.B0.dot(x0_pol_str)
    x1_pol_str0 = space.B1.dot(x1_pol_str)
    x2_pol_str0 = space.B2.dot(x2_pol_str)
    x3_pol_str0 = space.B3.dot(x3_pol_str)

    # perfrom matrix-vector products (with boundary conditions)
    r0_pol_str = space.B0.T.dot(space.M0_0(x0_pol_str0))
    r1_pol_str = space.B1.T.dot(space.M1_0(x1_pol_str0))
    r2_pol_str = space.B2.T.dot(space.M2_0(x2_pol_str0))
    r3_pol_str = space.B3.T.dot(space.M3_0(x3_pol_str0))

    rn_pol_str = space.B2.T.dot(mhd_ops_str.Mn(x2_pol_str0))
    rJ_pol_str = space.B2.T.dot(mhd_ops_str.MJ(x2_pol_str0))

    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=True)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=True)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=True)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=True)

    rn_pol_psy = mass_mats.M2n.dot(x2_pol_psy, apply_bc=True)
    rJ_pol_psy = mass_mats.M2J.dot(x2_pol_psy, apply_bc=True)

    assert np.allclose(r0_pol_str, r0_pol_psy.toarray(True))
    assert np.allclose(r1_pol_str, r1_pol_psy.toarray(True))
    assert np.allclose(r2_pol_str, r2_pol_psy.toarray(True))
    assert np.allclose(r3_pol_str, r3_pol_psy.toarray(True))
    assert np.allclose(rn_pol_str, rn_pol_psy.toarray(True))
    assert np.allclose(rJ_pol_str, rJ_pol_psy.toarray(True))

    # perfrom matrix-vector products (without boundary conditions)
    r0_pol_str = space.M0(x0_pol_str)
    r1_pol_str = space.M1(x1_pol_str)
    r2_pol_str = space.M2(x2_pol_str)
    r3_pol_str = space.M3(x3_pol_str)

    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=False)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=False)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=False)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=False)

    assert np.allclose(r0_pol_str, r0_pol_psy.toarray(True))
    assert np.allclose(r1_pol_str, r1_pol_psy.toarray(True))
    assert np.allclose(r2_pol_str, r2_pol_psy.toarray(True))
    assert np.allclose(r3_pol_str, r3_pol_psy.toarray(True))
    assert np.allclose(rn_pol_str, rn_pol_psy.toarray(True))
    assert np.allclose(rJ_pol_str, rJ_pol_psy.toarray(True))

    print(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("Nel", [[8, 12, 6]])
@pytest.mark.parametrize("p", [[2, 3, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
@pytest.mark.parametrize(
    "dirichlet_bc",
    [None, [[False, True], [False, False], [False, True]], [[False, False], [False, False], [True, False]]],
)
@pytest.mark.parametrize("mapping", [["HollowCylinder", {"a1": 0.1, "a2": 1.0, "Lz": 18.84955592153876}]])
def test_mass_preconditioner(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Compare mass matrix-vector products with Kronecker products of preconditioner,
    check PC * M = Id and test PCs in solve."""

    import time

    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.solvers import inverse

    from struphy.feec.mass import WeightedMassOperators, WeightedMassOperatorsOldForTesting
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch, ShearedSlab
    from struphy.geometry import domains
    from struphy.utils.arrays import xp as np

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f"Rank {mpi_rank} | Start test_mass_preconditioner with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show()

    # load MHD equilibrium
    if mapping[0] == "Cuboid":
        eq_mhd = ShearedSlab(
            **{
                "a": (mapping[1]["r1"] - mapping[1]["l1"]),
                "R0": (mapping[1]["r3"] - mapping[1]["l3"]) / (2 * np.pi),
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

    elif mapping[0] == "Colella":
        eq_mhd = ShearedSlab(
            **{
                "a": mapping[1]["Lx"],
                "R0": mapping[1]["Lz"] / (2 * np.pi),
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == "HollowCylinder":
        eq_mhd = ScrewPinch(
            **{
                "a": mapping[1]["a2"],
                "R0": 3.0,
                "B0": 1.0,
                "q0": 1.05,
                "q1": 1.8,
                "n1": 3.0,
                "n2": 4.0,
                "na": 0.0,
                "beta": 0.1,
            }
        )

        if show_plots:
            eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space
    if dirichlet_bc is not None:
        for i, knd in enumerate(spl_kind):
            if knd:
                dirichlet_bc[i] = [False, False]
    else:
        dirichlet_bc = [[False, False]] * 3

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc)

    fem_spaces = [derham.Vh_fem["0"], derham.Vh_fem["1"], derham.Vh_fem["2"], derham.Vh_fem["3"], derham.Vh_fem["v"]]

    print(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    mass_matsold = WeightedMassOperatorsOldForTesting(derham, domain, eq_mhd=eq_mhd)

    # assemble preconditioners
    if mpi_rank == 0:
        print("Start assembling preconditioners")

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)
    Mvpre = MassMatrixPreconditioner(mass_mats.Mv)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)
    Mvnpre = MassMatrixPreconditioner(mass_mats.Mvn)

    M1Bninvpre = MassMatrixPreconditioner(mass_mats.M1Bninv)
    M1Bninvoldpre = MassMatrixPreconditioner(mass_matsold.M1Bninv)

    if mpi_rank == 0:
        print("Done")

    # create random input arrays
    x0 = create_equal_random_arrays(fem_spaces[0], seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(fem_spaces[1], seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(fem_spaces[2], seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(fem_spaces[3], seed=8196, flattened=True)[1]
    xv = create_equal_random_arrays(fem_spaces[4], seed=2038, flattened=True)[1]

    # compare mass matrix-vector products with Kronecker products of preconditioner
    do_this_test = False

    if (mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder") and do_this_test:
        if mpi_rank == 0:
            print("Start matrix-vector products in stencil format for mapping Cuboid/HollowCylinder")

        r0 = mass_mats.M0.dot(x0)
        r1 = mass_mats.M1.dot(x1)
        r2 = mass_mats.M2.dot(x2)
        r3 = mass_mats.M3.dot(x3)
        rv = mass_mats.Mv.dot(xv)

        r1n = mass_mats.M1n.dot(x1)
        r2n = mass_mats.M2n.dot(x2)
        rvn = mass_mats.Mvn.dot(xv)

        r1Bninv = mass_mats.M1Bninv.dot(x1)
        r1Bninvold = mass_matsold.M1Bninv.dot(x1)

        if mpi_rank == 0:
            print("Done")

        if mpi_rank == 0:
            print("Start matrix-vector products in KroneckerStencil format for mapping Cuboid/HollowCylinder")

        r0_pre = M0pre.matrix.dot(x0)
        r1_pre = M1pre.matrix.dot(x1)
        r2_pre = M2pre.matrix.dot(x2)
        r3_pre = M3pre.matrix.dot(x3)
        rv_pre = Mvpre.matrix.dot(xv)

        r1n_pre = M1npre.matrix.dot(x1)
        r2n_pre = M2npre.matrix.dot(x2)
        rvn_pre = Mvnpre.matrix.dot(xv)

        r1Bninv_pre = M1Bninvpre.matrix.dot(x1)
        r1Bninvold_pre = M1Bninvoldpre.matrix.dot(x1)

        if mpi_rank == 0:
            print("Done")

        # compare output arrays
        assert np.allclose(r0.toarray(), r0_pre.toarray())
        assert np.allclose(r1.toarray(), r1_pre.toarray())
        assert np.allclose(r2.toarray(), r2_pre.toarray())
        assert np.allclose(r3.toarray(), r3_pre.toarray())
        assert np.allclose(rv.toarray(), rv_pre.toarray())

        assert np.allclose(r1n.toarray(), r1n_pre.toarray())
        assert np.allclose(r2n.toarray(), r2n_pre.toarray())
        assert np.allclose(rvn.toarray(), rvn_pre.toarray())

        assert np.allclose(r1Bninv.toarray(), r1Bninv_pre.toarray())
        assert np.allclose(r1Bninv.toarray(), r1Bninvold_pre.toarray())
        assert np.allclose(r1Bninvold.toarray(), r1Bninv_pre.toarray())

    # test if preconditioner satisfies PC * M = Identity
    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert np.allclose(mass_mats.M0.dot(M0pre.solve(x0)).toarray(), derham.boundary_ops["0"].dot(x0).toarray())
        assert np.allclose(mass_mats.M1.dot(M1pre.solve(x1)).toarray(), derham.boundary_ops["1"].dot(x1).toarray())
        assert np.allclose(mass_mats.M2.dot(M2pre.solve(x2)).toarray(), derham.boundary_ops["2"].dot(x2).toarray())
        assert np.allclose(mass_mats.M3.dot(M3pre.solve(x3)).toarray(), derham.boundary_ops["3"].dot(x3).toarray())
        assert np.allclose(mass_mats.Mv.dot(Mvpre.solve(xv)).toarray(), derham.boundary_ops["v"].dot(xv).toarray())

    # test preconditioner in iterative solver
    M0inv = inverse(mass_mats.M0, "pcg", pc=M0pre, tol=1e-8, maxiter=1000)
    M1inv = inverse(mass_mats.M1, "pcg", pc=M1pre, tol=1e-8, maxiter=1000)
    M2inv = inverse(mass_mats.M2, "pcg", pc=M2pre, tol=1e-8, maxiter=1000)
    M3inv = inverse(mass_mats.M3, "pcg", pc=M3pre, tol=1e-8, maxiter=1000)
    Mvinv = inverse(mass_mats.Mv, "pcg", pc=Mvpre, tol=1e-8, maxiter=1000)

    M1ninv = inverse(mass_mats.M1n, "pcg", pc=M1npre, tol=1e-8, maxiter=1000)
    M2ninv = inverse(mass_mats.M2n, "pcg", pc=M2npre, tol=1e-8, maxiter=1000)
    Mvninv = inverse(mass_mats.Mvn, "pcg", pc=Mvnpre, tol=1e-8, maxiter=1000)

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M0 with preconditioner")
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0))
    else:
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M0inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M1 with preconditioner")
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1))
    else:
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M1inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M2 with preconditioner")
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2))
    else:
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M2inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M3 with preconditioner")
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3))
    else:
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M3inv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert Mv with preconditioner")
        rv = Mvinv.dot(derham.boundary_ops["v"].dot(xv))
    else:
        rv = Mvinv.dot(derham.boundary_ops["v"].dot(xv))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert Mvinv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Apply M1n with preconditioner")
        r1n = M1ninv.dot(derham.boundary_ops["1"].dot(x1))
    else:
        r1n = M1ninv.dot(derham.boundary_ops["1"].dot(x1))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M1ninv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Apply M2n with preconditioner")
        r2n = M2ninv.dot(derham.boundary_ops["2"].dot(x2))
    else:
        r2n = M2ninv.dot(derham.boundary_ops["2"].dot(x2))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert M2ninv._info["niter"] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Apply Mvn with preconditioner")
        rvn = Mvninv.dot(derham.boundary_ops["v"].dot(xv))
    else:
        rvn = Mvninv.dot(derham.boundary_ops["v"].dot(xv))

    if mapping[0] == "Cuboid" or mapping[0] == "HollowCylinder":
        assert Mvninv._info["niter"] == 2

    time.sleep(2)
    print(f"Rank {mpi_rank} | All tests passed!")


@pytest.mark.parametrize("Nel", [[8, 9, 6]])
@pytest.mark.parametrize("p", [[2, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
@pytest.mark.parametrize(
    "dirichlet_bc",
    [None, [[False, True], [False, False], [False, True]], [[False, False], [False, False], [True, False]]],
)
@pytest.mark.parametrize("mapping", [["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}]])
def test_mass_preconditioner_polar(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    """Compare polar mass matrix-vector products with Kronecker products of preconditioner,
    check PC * M = Id and test PCs in solve."""

    import time

    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.solvers import inverse

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch
    from struphy.geometry import domains
    from struphy.polar.basic import PolarVector
    from struphy.utils.arrays import xp as np

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f"Rank {mpi_rank} | Start test_mass_preconditioner_polar with " + str(mpi_size) + " MPI processes!")

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**{"Nel": Nel[:2], "p": p[:2], "a": mapping[1]["a"], "Lz": mapping[1]["Lz"]})

    if show_plots:
        import matplotlib.pyplot as plt

        domain.show()

    # load MHD equilibrium
    eq_mhd = ScrewPinch(
        **{
            "a": mapping[1]["a"],
            "R0": mapping[1]["Lz"],
            "B0": 1.0,
            "q0": 1.05,
            "q1": 1.8,
            "n1": 3.0,
            "n2": 4.0,
            "na": 0.0,
            "beta": 0.1,
        }
    )

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space
    if dirichlet_bc is not None:
        for i, knd in enumerate(spl_kind):
            if knd:
                dirichlet_bc[i] = [False, False]
    else:
        dirichlet_bc = [[False, False]] * 3

    # derham object
    derham = Derham(
        Nel, p, spl_kind, comm=mpi_comm, dirichlet_bc=dirichlet_bc, with_projectors=False, polar_ck=1, domain=domain
    )

    print(f"Rank {mpi_rank} | Local domain : " + str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # preconditioners
    if mpi_rank == 0:
        print("Start assembling preconditioners")

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)

    if mpi_rank == 0:
        print("Done")

    # create random input arrays
    x0 = create_equal_random_arrays(derham.Vh_fem["0"], seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(derham.Vh_fem["1"], seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(derham.Vh_fem["2"], seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(derham.Vh_fem["3"], seed=8196, flattened=True)[1]

    # set polar vectors
    x0_pol = PolarVector(derham.Vh_pol["0"])
    x1_pol = PolarVector(derham.Vh_pol["1"])
    x2_pol = PolarVector(derham.Vh_pol["2"])
    x3_pol = PolarVector(derham.Vh_pol["3"])

    x0_pol.tp = x0
    x1_pol.tp = x1
    x2_pol.tp = x2
    x3_pol.tp = x3

    np.random.seed(1607)
    x0_pol.pol = [np.random.rand(x0_pol.pol[0].shape[0], x0_pol.pol[0].shape[1])]
    x1_pol.pol = [np.random.rand(x1_pol.pol[n].shape[0], x1_pol.pol[n].shape[1]) for n in range(3)]
    x2_pol.pol = [np.random.rand(x2_pol.pol[n].shape[0], x2_pol.pol[n].shape[1]) for n in range(3)]
    x3_pol.pol = [np.random.rand(x3_pol.pol[0].shape[0], x3_pol.pol[0].shape[1])]

    # test preconditioner in iterative solver and compare to case without preconditioner
    M0inv = inverse(mass_mats.M0, "pcg", pc=M0pre, tol=1e-8, maxiter=500)
    M1inv = inverse(mass_mats.M1, "pcg", pc=M1pre, tol=1e-8, maxiter=500)
    M2inv = inverse(mass_mats.M2, "pcg", pc=M2pre, tol=1e-8, maxiter=500)
    M3inv = inverse(mass_mats.M3, "pcg", pc=M3pre, tol=1e-8, maxiter=500)

    M1ninv = inverse(mass_mats.M1n, "pcg", pc=M1npre, tol=1e-8, maxiter=500)
    M2ninv = inverse(mass_mats.M2n, "pcg", pc=M2npre, tol=1e-8, maxiter=500)

    M0inv_nopc = inverse(mass_mats.M0, "pcg", pc=None, tol=1e-8, maxiter=500)
    M1inv_nopc = inverse(mass_mats.M1, "pcg", pc=None, tol=1e-8, maxiter=500)
    M2inv_nopc = inverse(mass_mats.M2, "pcg", pc=None, tol=1e-8, maxiter=500)
    M3inv_nopc = inverse(mass_mats.M3, "pcg", pc=None, tol=1e-8, maxiter=500)

    M1ninv_nopc = inverse(mass_mats.M1n, "pcg", pc=None, tol=1e-8, maxiter=500)
    M2ninv_nopc = inverse(mass_mats.M2n, "pcg", pc=None, tol=1e-8, maxiter=500)

    # =============== M0 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M0 with preconditioner")
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0_pol))
        print("Number of iterations : ", M0inv._info["niter"])
    else:
        r0 = M0inv.dot(derham.boundary_ops["0"].dot(x0_pol))

    assert M0inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M0 without preconditioner")
        r0 = M0inv_nopc.dot(derham.boundary_ops["0"].dot(x0_pol))
        print("Number of iterations : ", M0inv_nopc._info["niter"])
    else:
        r0 = M0inv_nopc.dot(derham.boundary_ops["0"].dot(x0_pol))

    assert M0inv._info["niter"] < M0inv_nopc._info["niter"]
    # =======================================================

    # =============== M1 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M1 with preconditioner")
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1_pol))
        print("Number of iterations : ", M1inv._info["niter"])
    else:
        r1 = M1inv.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M1 without preconditioner")
        r1 = M1inv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))
        print("Number of iterations : ", M1inv_nopc._info["niter"])
    else:
        r1 = M1inv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1inv._info["niter"] < M1inv_nopc._info["niter"]
    # =======================================================

    # =============== M2 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M2 with preconditioner")
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2_pol))
        print("Number of iterations : ", M2inv._info["niter"])
    else:
        r2 = M2inv.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M2 without preconditioner")
        r2 = M2inv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))
        print("Number of iterations : ", M2inv_nopc._info["niter"])
    else:
        r2 = M2inv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2inv._info["niter"] < M2inv_nopc._info["niter"]
    # =======================================================

    # =============== M3 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M3 with preconditioner")
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3_pol))
        print("Number of iterations : ", M3inv._info["niter"])
    else:
        r3 = M3inv.dot(derham.boundary_ops["3"].dot(x3_pol))

    assert M3inv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M3 without preconditioner")
        r3 = M3inv_nopc.dot(derham.boundary_ops["3"].dot(x3_pol))
        print("Number of iterations : ", M3inv_nopc._info["niter"])
    else:
        r3 = M3inv_nopc.dot(derham.boundary_ops["3"].dot(x3_pol))

    assert M3inv._info["niter"] < M3inv_nopc._info["niter"]
    # =======================================================

    # =============== M1n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M1n with preconditioner")
        r1 = M1ninv.dot(derham.boundary_ops["1"].dot(x1_pol))
        print("Number of iterations : ", M1ninv._info["niter"])
    else:
        r1 = M1ninv.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1ninv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M1n without preconditioner")
        r1 = M1ninv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))
        print("Number of iterations : ", M1ninv_nopc._info["niter"])
    else:
        r1 = M1ninv_nopc.dot(derham.boundary_ops["1"].dot(x1_pol))

    assert M1ninv._info["niter"] < M1ninv_nopc._info["niter"]
    # =======================================================

    # =============== M2n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M2n with preconditioner")
        r2 = M2ninv.dot(derham.boundary_ops["2"].dot(x2_pol))
        print("Number of iterations : ", M2ninv._info["niter"])
    else:
        r2 = M2ninv.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2ninv._info["success"]

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print("Invert M2n without preconditioner")
        r2 = M2ninv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))
        print("Number of iterations : ", M2ninv_nopc._info["niter"])
    else:
        r2 = M2ninv_nopc.dot(derham.boundary_ops["2"].dot(x2_pol))

    assert M2ninv._info["niter"] < M2ninv_nopc._info["niter"]
    # =======================================================

    time.sleep(2)
    print(f"Rank {mpi_rank} | All tests passed!")


if __name__ == "__main__":
    test_mass(
        [5, 6, 7],
        [2, 2, 3],
        [True, False, True],
        [[False, True], [True, False], [False, False]],
        ["Colella", {"Lx": 1.0, "Ly": 6.0, "alpha": 0.1, "Lz": 10.0}],
        False,
    )
    test_mass(
        [5, 6, 7],
        [2, 2, 3],
        [True, False, True],
        [[False, False], [False, False], [False, False]],
        ["Colella", {"Lx": 1.0, "Ly": 6.0, "alpha": 0.1, "Lz": 10.0}],
        False,
    )
    # # test_mass([8, 6, 4], [2, 3, 2], [False, True, False], [['d', 'd'], [None, None], [None, 'd']], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False)
    # test_mass([8, 6, 4], [2, 2, 2], [False, True, True], [['d', 'd'], [None, None], [None, None]], ['HollowCylinder', {'a1': .1, 'a2': 1., 'Lz': 10.}], False)

    # test_mass_polar([8, 12, 6], [4, 3, 2], [False, True, False], [[False,  True], [False, False], [False, True]], ['IGAPolarCylinder', {'a': 1., 'Lz': 3.}], False)

    # test_mass_preconditioner([8, 6, 4], [2, 2, 2], [False, False, False], [[True, True], [False, False], [False, False]], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False)
    # test_mass_preconditioner([8, 6, 4], [2, 2, 2], [False, False, False], [['d', 'd'], [None, None], [None, None]], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .05, 'Lz' : 10.}], False)
    # test_mass_preconditioner([6, 9, 4], [4, 3, 2], [False, True, False], [[None, 'd'], [None, None], ['d', None]], ['HollowCylinder', {'a1' : .1, 'a2' : 1., 'Lz' : 18.84955592153876}], False)

    # test_mass_preconditioner_polar([8, 12, 6], [4, 3, 2], [False, True, False], [[False, True], [False, False], [True, False]], ['IGAPolarCylinder', {'a': 1., 'Lz': 3.}], False)
