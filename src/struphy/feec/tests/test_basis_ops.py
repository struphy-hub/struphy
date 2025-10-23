import pytest


@pytest.mark.parametrize("Nel", [[8, 12, 4]])
@pytest.mark.parametrize("p", [[2, 3, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [True, False, True]])
@pytest.mark.parametrize("mapping", [["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}]])
def test_some_basis_ops(Nel, p, spl_kind, mapping):
    """Tests the MHD specific projection operators PI_ijk(fun*Lambda_mno).

    Here, PI_ijk is the commuting projector of the output space (codomain),
    Lambda_mno are the basis functions of the input space (domain),
    and fun is an arbitrary (matrix-valued) function.
    """
    from time import time

    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.block import BlockVector
    from psydac.linalg.stencil import StencilVector

    from struphy.eigenvalue_solvers.legacy.mhd_operators_MF import projectors_dot_x
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background.equils import HomogenSlab
    from struphy.geometry import domains

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # de Rham object
    n_quad_el = [5, 5, 5]
    n_quad_pr = [4, 4, 4]

    DERHAM_PSY = Derham(Nel, p, spl_kind, nq_pr=n_quad_pr, nquads=n_quad_el, comm=MPI_COMM)

    # grid parameters
    if mpi_rank == 0:
        print(f"Rank {mpi_rank} | Nel: {Nel}")
        print(f"Rank {mpi_rank} | p: {p}")
        print(f"Rank {mpi_rank} | spl_kind: {spl_kind}")
        print(f"Rank {mpi_rank} | ")

    # Mhd equilibirum (slab)
    mhd_equil_params = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "beta": 2.0, "n0": 1.0}

    EQ_MHD = HomogenSlab(**mhd_equil_params)
    EQ_MHD.domain = domain

    # Psydac spline spaces
    V0 = DERHAM_PSY.Vh_fem["0"]
    V1 = DERHAM_PSY.Vh_fem["1"]
    V2 = DERHAM_PSY.Vh_fem["2"]
    V3 = DERHAM_PSY.Vh_fem["3"]
    V0vec = DERHAM_PSY.Vh_fem["v"]

    if mpi_rank == 0:
        print(f"Rank {mpi_rank} | type(V0) {type(V0)}")
        print(f"Rank {mpi_rank} | type(V1) {type(V1)}")
        print(f"Rank {mpi_rank} | type(V2) {type(V2)}")
        print(f"Rank {mpi_rank} | type(V3) {type(V3)}")
        print(f"Rank {mpi_rank} | type(V0vec) {type(V0vec)}")
        print(f"Rank {mpi_rank} | ")

    # Psydac projectors
    P0 = DERHAM_PSY.P["0"]
    P1 = DERHAM_PSY.P["1"]
    P2 = DERHAM_PSY.P["2"]
    P3 = DERHAM_PSY.P["3"]
    P0vec = DERHAM_PSY.P["v"]
    if mpi_rank == 0:
        print(f"Rank {mpi_rank} | type(P0) {type(P0)}")
        print(f"Rank {mpi_rank} | type(P1) {type(P1)}")
        print(f"Rank {mpi_rank} | type(P2) {type(P2)}")
        print(f"Rank {mpi_rank} | type(P3) {type(P3)}")
        print(f"Rank {mpi_rank} | type(P0vec) {type(P0vec)}")
        print(f"Rank {mpi_rank} | ")

    # Struphy spline spaces
    space_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], n_quad_el[0] + 1)
    space_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], n_quad_el[1] + 1)
    space_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2], n_quad_el[2] + 1)

    space_1.set_projectors(n_quad_pr[0])
    space_2.set_projectors(n_quad_pr[1])
    space_3.set_projectors(n_quad_pr[2])

    # print('\nSTRUPHY point sets:')
    # print('\nDirection 1:')
    # print(f'x_int: {space_1.projectors.x_int}')
    # print(f'x_hisG: {space_1.projectors.x_hisG}')
    # print(f'x_his: {space_1.projectors.x_his}')
    # print('\nDirection 2:')
    # print(f'x_int: {space_2.projectors.x_int}')
    # print(f'x_hisG: {space_2.projectors.x_hisG}')
    # print(f'x_his: {space_2.projectors.x_his}')
    # print('\nDirection 3:')
    # print(f'x_int: {space_3.projectors.x_int}')
    # print(f'x_hisG: {space_3.projectors.x_hisG}')
    # print(f'x_his: {space_3.projectors.x_his}')

    SPACES = Tensor_spline_space([space_1, space_2, space_3])
    SPACES.set_projectors("tensor")

    # Psydac MHD operators
    OPS_PSY = BasisProjectionOperators(DERHAM_PSY, domain, eq_mhd=EQ_MHD)

    # Struphy matrix-free MHD operators
    print(f"Rank {mpi_rank} | Init STRUPHY `projectors_dot_x`...")
    elapsed = time()
    OPS_STR = projectors_dot_x(SPACES, EQ_MHD)
    print(f"Rank {mpi_rank} | Init `projectors_dot_x` done ({time() - elapsed:.4f}s).")

    # Test vectors
    x0 = xp.reshape(xp.arange(V0.nbasis), [space.nbasis for space in V0.spaces])

    x1 = [xp.reshape(xp.arange(comp.nbasis), [space.nbasis for space in comp.spaces]) for comp in V1.spaces]

    x2 = [xp.reshape(xp.arange(comp.nbasis), [space.nbasis for space in comp.spaces]) for comp in V2.spaces]

    x3 = xp.reshape(xp.arange(V3.nbasis), [space.nbasis for space in V3.spaces])

    x0_st = StencilVector(V0.coeff_space)
    x1_st = BlockVector(V1.coeff_space, [StencilVector(comp) for comp in V1.coeff_space])
    x2_st = BlockVector(V2.coeff_space, [StencilVector(comp) for comp in V2.coeff_space])
    x3_st = StencilVector(V3.coeff_space)

    # for testing X1T:
    x0vec_st = BlockVector(V0vec.coeff_space, [StencilVector(comp) for comp in V0vec.coeff_space])

    MPI_COMM.Barrier()

    print(f"rank: {mpi_rank} | x3_starts[0]: {x3_st.starts[0]}, x3_ends[0]: {x3_st.ends[0]}")
    MPI_COMM.Barrier()
    print(f"rank: {mpi_rank} | x3_starts[1]: {x3_st.starts[1]}, x3_ends[1]: {x3_st.ends[1]}")
    MPI_COMM.Barrier()
    print(f"rank: {mpi_rank} | x3_starts[2]: {x3_st.starts[2]}, x3_ends[2]: {x3_st.ends[2]}")
    MPI_COMM.Barrier()

    # Use .copy() in case input will be overwritten (is not the case I guess)
    x0_st[
        x0_st.starts[0] : x0_st.ends[0] + 1,
        x0_st.starts[1] : x0_st.ends[1] + 1,
        x0_st.starts[2] : x0_st.ends[2] + 1,
    ] = x0[
        x0_st.starts[0] : x0_st.ends[0] + 1,
        x0_st.starts[1] : x0_st.ends[1] + 1,
        x0_st.starts[2] : x0_st.ends[2] + 1,
    ].copy()

    for n in range(3):
        x1_st[n][
            x1_st[n].starts[0] : x1_st[n].ends[0] + 1,
            x1_st[n].starts[1] : x1_st[n].ends[1] + 1,
            x1_st[n].starts[2] : x1_st[n].ends[2] + 1,
        ] = x1[n][
            x1_st[n].starts[0] : x1_st[n].ends[0] + 1,
            x1_st[n].starts[1] : x1_st[n].ends[1] + 1,
            x1_st[n].starts[2] : x1_st[n].ends[2] + 1,
        ].copy()

    for n in range(3):
        x2_st[n][
            x2_st[n].starts[0] : x2_st[n].ends[0] + 1,
            x2_st[n].starts[1] : x2_st[n].ends[1] + 1,
            x2_st[n].starts[2] : x2_st[n].ends[2] + 1,
        ] = x2[n][
            x2_st[n].starts[0] : x2_st[n].ends[0] + 1,
            x2_st[n].starts[1] : x2_st[n].ends[1] + 1,
            x2_st[n].starts[2] : x2_st[n].ends[2] + 1,
        ].copy()

    x3_st[
        x3_st.starts[0] : x3_st.ends[0] + 1,
        x3_st.starts[1] : x3_st.ends[1] + 1,
        x3_st.starts[2] : x3_st.ends[2] + 1,
    ] = x3[
        x3_st.starts[0] : x3_st.ends[0] + 1,
        x3_st.starts[1] : x3_st.ends[1] + 1,
        x3_st.starts[2] : x3_st.ends[2] + 1,
    ].copy()

    for n in range(3):
        x0vec_st[n][
            x0vec_st[n].starts[0] : x0vec_st[n].ends[0] + 1,
            x0vec_st[n].starts[1] : x0vec_st[n].ends[1] + 1,
            x0vec_st[n].starts[2] : x0vec_st[n].ends[2] + 1,
        ] = x0[
            x0vec_st[n].starts[0] : x0vec_st[n].ends[0] + 1,
            x0vec_st[n].starts[1] : x0vec_st[n].ends[1] + 1,
            x0vec_st[n].starts[2] : x0vec_st[n].ends[2] + 1,
        ].copy()

    MPI_COMM.Barrier()

    x0_st.update_ghost_regions()
    x1_st.update_ghost_regions()
    x2_st.update_ghost_regions()
    x3_st.update_ghost_regions()

    MPI_COMM.Barrier()

    # Compare to Struphy matrix-free operators
    # See struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x for the definition of these operators

    # operator K3 (V3 --> V3)
    if mpi_rank == 0:
        print("\nK3 (V3 --> V3, Identity operator in this case):")

    res_PSY = OPS_PSY.K3.dot(x3_st)
    res_STR = OPS_STR.K1_dot(x3.flatten())
    res_STR = SPACES.extract_3(res_STR)

    print(f"Rank {mpi_rank} | Asserting MHD operator K3.")
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f"Rank {mpi_rank} | Assertion passed.")

    K3T = OPS_PSY.K3.transpose()
    res_PSY = K3T.dot(x3_st)
    res_STR = OPS_STR.transpose_K1_dot(x3.flatten())
    res_STR = SPACES.extract_3(res_STR)

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator K3T.")
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    # operator K0 (V0 --> V0)
    if mpi_rank == 0:
        print("\nK0 (V0 --> V0, Identity operator in this case):")

    res_PSY = OPS_PSY.K0.dot(x0_st)
    res_STR = OPS_STR.K10_dot(x0.flatten())
    res_STR = SPACES.extract_0(res_STR)

    print(f"Rank {mpi_rank} | Asserting MHD operator K0.")
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f"Rank {mpi_rank} | Assertion passed.")

    K10T = OPS_PSY.K0.transpose()
    res_PSY = K10T.dot(x0_st)
    res_STR = OPS_STR.transpose_K10_dot(x0.flatten())
    res_STR = SPACES.extract_0(res_STR)

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator K10T.")
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    # operator Q1 (V1 --> V2)
    if mpi_rank == 0:
        print("\nQ1 (V1 --> V2):")

    res_PSY = OPS_PSY.Q1.dot(x1_st)
    res_STR = OPS_STR.Q1_dot(xp.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q1, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q1, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q1, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    Q1T = OPS_PSY.Q1.transpose()
    res_PSY = Q1T.dot(x2_st)
    res_STR = OPS_STR.transpose_Q1_dot(xp.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    # operator W1 (V1 --> V1)
    if mpi_rank == 0:
        print("\nW1 (V1 --> V1, Identity operator in this case):")

    res_PSY = OPS_PSY.W1.dot(x1_st)
    res_STR = OPS_STR.W1_dot(xp.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator W1, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator W1, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator W1, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    W1T = OPS_PSY.W1.transpose()
    res_PSY = W1T.dot(x1_st)
    res_STR = OPS_STR.transpose_W1_dot(xp.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    # operator Q2 (V2 --> V2)
    if mpi_rank == 0:
        print("\nQ2 (V2 --> V2, Identity operator in this case):")

    res_PSY = OPS_PSY.Q2.dot(x2_st)
    res_STR = OPS_STR.Q2_dot(xp.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q2, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q2, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator Q2, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    Q2T = OPS_PSY.Q2.transpose()
    res_PSY = Q2T.dot(x2_st)
    res_STR = OPS_STR.transpose_Q2_dot(xp.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    # operator X1 (V1 --> V0 x V0 x V0)
    if mpi_rank == 0:
        print("\nX1 (V1 --> V0 x V0 x V0):")

    res_PSY = OPS_PSY.X1.dot(x1_st)
    res_STR = OPS_STR.X1_dot(xp.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0 = SPACES.extract_0(res_STR[0])
    res_STR_1 = SPACES.extract_0(res_STR[1])
    res_STR_2 = SPACES.extract_0(res_STR[2])

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator X1, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator X1, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting MHD operator X1, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")

    X1T = OPS_PSY.X1.transpose()
    res_PSY = X1T.dot(x0vec_st)
    res_STR = OPS_STR.transpose_X1_dot([x0.flatten(), x0.flatten(), x0.flatten()])
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, first component.")
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, second component.")
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f"Rank {mpi_rank} | Assertion passed.")

    MPI_COMM.Barrier()

    print(f"Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, third component.")
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f"Rank {mpi_rank} | Assertion passed.")


@pytest.mark.parametrize("Nel", [[6, 9, 7]])
@pytest.mark.parametrize("p", [[2, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
@pytest.mark.parametrize(
    "dirichlet_bc",
    [None, [(False, True), (False, False), (False, True)], [(False, False), (False, False), (True, False)]],
)
@pytest.mark.parametrize("mapping", [["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}]])
def test_basis_ops_polar(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):
    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI

    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.fields_background.equils import ScrewPinch
    from struphy.geometry import domains
    from struphy.polar.basic import PolarVector

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    print("number of processes : ", mpi_size)

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
            "R0": 3.0,
            "B0": 1.0,
            "q0": 1.05,
            "q1": 1.80,
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
                dirichlet_bc[i] = (False, False)
    else:
        dirichlet_bc = [(False, False)] * 3

    dirichlet_bc = tuple(dirichlet_bc)

    # derham object
    nq_el = [p[0] + 1, p[1] + 1, p[2] + 1]
    nq_pr = p.copy()

    derham = Derham(
        Nel,
        p,
        spl_kind,
        nquads=p,
        nq_pr=nq_pr,
        comm=mpi_comm,
        dirichlet_bc=dirichlet_bc,
        with_projectors=True,
        polar_ck=1,
        domain=domain,
    )

    if mpi_rank == 0:
        print()
        print(derham.domain_array)

    mhd_ops_psy = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    spaces = [
        Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], dirichlet_bc[0]),
        Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], dirichlet_bc[1]),
        Spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2], dirichlet_bc[2]),
    ]

    spaces[0].set_projectors(nq_pr[0])
    spaces[1].set_projectors(nq_pr[1])
    spaces[2].set_projectors(nq_pr[2])

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors("general")

    mhd_ops_str = MHDOperators(space, eq_mhd, basis_u=2)

    mhd_ops_str.assemble_dofs("MF")
    mhd_ops_str.assemble_dofs("PF")
    mhd_ops_str.assemble_dofs("EF")
    mhd_ops_str.assemble_dofs("PR")

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

    xp.random.seed(1607)
    x0_pol_psy.pol = [xp.random.rand(x0_pol_psy.pol[0].shape[0], x0_pol_psy.pol[0].shape[1])]
    x1_pol_psy.pol = [xp.random.rand(x1_pol_psy.pol[n].shape[0], x1_pol_psy.pol[n].shape[1]) for n in range(3)]
    x2_pol_psy.pol = [xp.random.rand(x2_pol_psy.pol[n].shape[0], x2_pol_psy.pol[n].shape[1]) for n in range(3)]
    x3_pol_psy.pol = [xp.random.rand(x3_pol_psy.pol[0].shape[0], x3_pol_psy.pol[0].shape[1])]

    # apply boundary conditions to legacy vectors for right shape
    x0_pol_str = space.B0.dot(x0_pol_psy.toarray(True))
    x1_pol_str = space.B1.dot(x1_pol_psy.toarray(True))
    x2_pol_str = space.B2.dot(x2_pol_psy.toarray(True))
    x3_pol_str = space.B3.dot(x3_pol_psy.toarray(True))

    # ================================================================================
    #                              MHD velocity is a 2-form
    # ================================================================================

    # ===== operator K3 (V3 --> V3) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print("\nOperator K (V3 --> V3):")

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.K3.dot(x3_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.K3.dot(x3_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PR(x3_pol_str)

    print(f"Rank {mpi_rank} | Asserting MHD operator K3.")
    xp.allclose(space.B3.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.K3.transpose().dot(x3_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.K3.transpose().dot(x3_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PR.T(x3_pol_str)

    print(f"Rank {mpi_rank} | Asserting transpose MHD operator K3.T.")
    xp.allclose(space.B3.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    # ===== operator Q2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print("\nOperator Q2 (V2 --> V2):")

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.Q2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.Q2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.MF(x2_pol_str)

    print(f"Rank {mpi_rank} | Asserting MHD operator Q2.")
    xp.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.Q2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.Q2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.MF.T(x2_pol_str)

    print(f"Rank {mpi_rank} | Asserting transposed MHD operator Q2.T.")
    xp.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    # ===== operator T2 (V2 --> V1) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print("\nOperator T2 (V2 --> V1):")

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.T2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.T2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.EF(x2_pol_str)

    print(f"Rank {mpi_rank} | Asserting MHD operator T2.")
    xp.allclose(space.B1.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.T2.transpose().dot(x1_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.T2.transpose().dot(x1_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.EF.T(x1_pol_str)

    print(f"Rank {mpi_rank} | Asserting transposed MHD operator T2.T.")
    xp.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    # ===== operator S2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print("\nOperator S2 (V2 --> V2):")

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.S2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.S2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PF(x2_pol_str)

    print(f"Rank {mpi_rank} | Asserting MHD operator S2.")
    xp.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.S2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.S2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PF.T(x2_pol_str)

    print(f"Rank {mpi_rank} | Asserting transposed MHD operator S2.T.")
    xp.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f"Rank {mpi_rank} | Assertion passed.")


def assert_ops(mpi_rank, res_PSY, res_STR, verbose=False, MPI_COMM=None):
    """
    TODO
    """

    import cunumpy as xp

    if verbose:
        if MPI_COMM is not None:
            MPI_COMM.Barrier()

        # print(f'Rank {mpi_rank} | ')
        # print(f'Rank {mpi_rank} | res_PSY.shape   : {res_PSY.shape}')
        # print(f'Rank {mpi_rank} | res_PSY[:].shape: {res_PSY[:].shape}')
        # print(f'Rank {mpi_rank} | res_STR.shape   : {res_STR.shape}')

        # print(f'Rank {mpi_rank} | res_PSY starts & ends:')
        # print([
        #     res_PSY.starts[0], res_PSY.ends[0] + 1,
        #     res_PSY.starts[1], res_PSY.ends[1] + 1,
        #     res_PSY.starts[2], res_PSY.ends[2] + 1,
        # ])

        # print(f'Rank {mpi_rank} | res_PSY starts & ends:')
        # print([
        #     res_PSY.starts[0], res_PSY.ends[0] + 1,
        #     res_PSY.starts[1], res_PSY.ends[1] + 1,
        #     res_PSY.starts[2], res_PSY.ends[2] + 1,
        # ])

        # if MPI_COMM is not None: MPI_COMM.Barrier()

        # print(f'Rank {mpi_rank} | res_PSY (local slice at starts[0]):')
        # print(res_PSY[
        #     res_PSY.starts[0],
        #     res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #     res_PSY.starts[2] : res_PSY.ends[2] + 1,
        # ])

        # print(f'Rank {mpi_rank} | res_STR (local slice at starts[0]):')
        # print(res_STR[
        #     res_PSY.starts[0],
        #     res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #     res_PSY.starts[2] : res_PSY.ends[2] + 1,
        # ])
        # print(f'Rank {mpi_rank} | ')

        # for n in range(res_PSY.ends[0] + 1):

        #     print(f'Rank {mpi_rank} | dof_PSY (local slice at starts[0] + {n}):')
        #     print(dof_PSY[
        #         res_PSY.starts[0] + n,
        #         res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #         res_PSY.starts[2] : res_PSY.ends[2] + 1,
        #     ])

        #     print(f'Rank {mpi_rank} | dof_STR (local slice at starts[0] + {n}):')
        #     print(dof_STR[
        #         res_PSY.starts[0] + n,
        #         res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #         res_PSY.starts[2] : res_PSY.ends[2] + 1,
        #     ])
        #     print(f'Rank {mpi_rank} | ')

        # if MPI_COMM is not None: MPI_COMM.Barrier()

        print(
            f"Rank {mpi_rank} | Maximum absolute diference (result):\n",
            xp.max(
                xp.abs(
                    res_PSY[
                        res_PSY.starts[0] : res_PSY.ends[0] + 1,
                        res_PSY.starts[1] : res_PSY.ends[1] + 1,
                        res_PSY.starts[2] : res_PSY.ends[2] + 1,
                    ]
                    - res_STR[
                        res_PSY.starts[0] : res_PSY.ends[0] + 1,
                        res_PSY.starts[1] : res_PSY.ends[1] + 1,
                        res_PSY.starts[2] : res_PSY.ends[2] + 1,
                    ]
                )
            ),
        )

    if MPI_COMM is not None:
        MPI_COMM.Barrier()

    # Compare results. (Works only for Nel=[N, N, N] so far! TODO: Find this bug!)
    assert xp.allclose(
        res_PSY[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ],
        res_STR[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ],
    )

    if MPI_COMM is not None:
        MPI_COMM.Barrier()


if __name__ == "__main__":
    # test_some_basis_ops(
    #     Nel=[8, 8, 8],
    #     p=[2, 2, 2],
    #     spl_kind=[False, True, True],
    #     mapping=["Cuboid", {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}],
    # )
    test_basis_ops_polar(
        [6, 9, 7], [2, 2, 3], [False, True, True], None, ["IGAPolarCylinder", {"a": 1.0, "Lz": 3.0}], False
    )
