import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 6]])
@pytest.mark.parametrize("p", [[3, 2, 4]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_spaces(Nel, p, spl_kind):
    from struphy.feec.psydac_derham import Derham
    from struphy.polar.basic import PolarDerhamSpace, PolarVector

    derham = Derham(Nel, p, spl_kind)

    print("polar V0:")
    V = PolarDerhamSpace(derham, "H1")
    print("dimensions (parent, polar):", derham.Vh_fem["0"].nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.tp[:] = 1.0
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2 * a).toarray())
    print((a * 2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print()

    print("polar V1:")
    V = PolarDerhamSpace(derham, "Hcurl")
    print("dimensions (parent, polar):", derham.Vh_fem["1"].nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2 * a).toarray())
    print((a * 2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print()

    print("polar V2:")
    V = PolarDerhamSpace(derham, "Hdiv")
    print("dimensions (parent, polar):", derham.Vh_fem["2"], V.dimension)
    print(V.dtype)
    print(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2 * a).toarray())
    print((a * 2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print()

    print("polar V3:")
    V = PolarDerhamSpace(derham, "L2")
    print("dimensions (parent, polar):", derham.Vh_fem["3"], V.dimension)
    print(V.dtype)
    print(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.tp[:] = 1.0
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2 * a).toarray())
    print((a * 2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print()

    print("polar V0vec:")
    V = PolarDerhamSpace(derham, "H1vec")
    print("dimensions (parent, polar):", derham.Vh_fem["v"].nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2 * a).toarray())
    print((a * 2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print()


@pytest.mark.parametrize("Nel", [[6, 9, 6]])
@pytest.mark.parametrize("p", [[3, 2, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_extraction_ops_and_derivatives(Nel, p, spl_kind):
    import numpy as np
    from mpi4py import MPI

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.polar.basic import PolarDerhamSpace, PolarVector
    from struphy.polar.extraction_operators import PolarExtractionBlocksC1
    from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create control points
    params_map = {"Nel": Nel[:2], "p": p[:2], "Lz": 3.0, "a": 1.0}
    domain = IGAPolarCylinder(**params_map)

    # create de Rham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm, polar_ck=1, domain=domain, with_projectors=False)

    # create legacy FEM spaces
    spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    for space_i in spaces:
        space_i.set_projectors()

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors("general")

    if rank == 0:
        print()
        print("Domain decomposition : \n", derham.domain_array)
        print()

    comm.Barrier()

    # create polar FEM spaces
    f0_pol = PolarVector(derham.Vh_pol["0"])
    e1_pol = PolarVector(derham.Vh_pol["1"])
    b2_pol = PolarVector(derham.Vh_pol["2"])
    p3_pol = PolarVector(derham.Vh_pol["3"])

    # create pure tensor-product and polar vectors (legacy and distributed)
    f0_tp_leg, f0_tp = create_equal_random_arrays(derham.Vh_fem["0"], flattened=True)
    e1_tp_leg, e1_tp = create_equal_random_arrays(derham.Vh_fem["1"], flattened=True)
    b2_tp_leg, b2_tp = create_equal_random_arrays(derham.Vh_fem["2"], flattened=True)
    p3_tp_leg, p3_tp = create_equal_random_arrays(derham.Vh_fem["3"], flattened=True)

    f0_pol.tp = f0_tp
    e1_pol.tp = e1_tp
    b2_pol.tp = b2_tp
    p3_pol.tp = p3_tp

    np.random.seed(1607)
    f0_pol.pol = [np.random.rand(f0_pol.pol[0].shape[0], f0_pol.pol[0].shape[1])]
    e1_pol.pol = [np.random.rand(e1_pol.pol[n].shape[0], e1_pol.pol[n].shape[1]) for n in range(3)]
    b2_pol.pol = [np.random.rand(b2_pol.pol[n].shape[0], b2_pol.pol[n].shape[1]) for n in range(3)]
    p3_pol.pol = [np.random.rand(p3_pol.pol[0].shape[0], p3_pol.pol[0].shape[1])]

    f0_pol_leg = f0_pol.toarray(True)
    e1_pol_leg = e1_pol.toarray(True)
    b2_pol_leg = b2_pol.toarray(True)
    p3_pol_leg = p3_pol.toarray(True)

    # ==================== test basis extraction operators ===================
    if rank == 0:
        print("----------- Test basis extraction operators ---------")

    # test basis extraction operator
    r0_pol = derham.extraction_ops["0"].dot(f0_tp)
    r1_pol = derham.extraction_ops["1"].dot(e1_tp)
    r2_pol = derham.extraction_ops["2"].dot(b2_tp)
    r3_pol = derham.extraction_ops["3"].dot(p3_tp)

    assert np.allclose(r0_pol.toarray(True), space.E0.dot(f0_tp_leg))
    assert np.allclose(r1_pol.toarray(True), space.E1.dot(e1_tp_leg))
    assert np.allclose(r2_pol.toarray(True), space.E2.dot(b2_tp_leg))
    assert np.allclose(r3_pol.toarray(True), space.E3.dot(p3_tp_leg))

    # test transposed extraction operators
    E0T = derham.extraction_ops["0"].transpose()
    E1T = derham.extraction_ops["1"].transpose()
    E2T = derham.extraction_ops["2"].transpose()
    E3T = derham.extraction_ops["3"].transpose()

    r0 = E0T.dot(f0_pol)
    r1 = E1T.dot(e1_pol)
    r2 = E2T.dot(b2_pol)
    r3 = E3T.dot(p3_pol)

    compare_arrays(r0, space.E0.T.dot(f0_pol_leg), rank)
    compare_arrays(r1, space.E1.T.dot(e1_pol_leg), rank)
    compare_arrays(r2, space.E2.T.dot(b2_pol_leg), rank)
    compare_arrays(r3, space.E3.T.dot(p3_pol_leg), rank)

    if rank == 0:
        print("------------- Test passed ---------------------------")
        print()

    # ==================== test discrete derivatives ======================
    if rank == 0:
        print("----------- Test discrete derivatives ---------")

    # test discrete derivatives
    r1_pol = derham.grad.dot(f0_pol)
    r2_pol = derham.curl.dot(e1_pol)
    r3_pol = derham.div.dot(b2_pol)

    assert np.allclose(r1_pol.toarray(True), space.G.dot(f0_pol_leg))
    assert np.allclose(r2_pol.toarray(True), space.C.dot(e1_pol_leg))
    assert np.allclose(r3_pol.toarray(True), space.D.dot(b2_pol_leg))

    # test transposed derivatives
    GT = derham.grad.transpose()
    CT = derham.curl.transpose()
    DT = derham.div.transpose()

    r0_pol = GT.dot(e1_pol)
    r1_pol = CT.dot(b2_pol)
    r2_pol = DT.dot(p3_pol)

    assert np.allclose(r0_pol.toarray(True), space.G.T.dot(e1_pol_leg))
    assert np.allclose(r1_pol.toarray(True), space.C.T.dot(b2_pol_leg))
    assert np.allclose(r2_pol.toarray(True), space.D.T.dot(p3_pol_leg))

    if rank == 0:
        print("------------- Test passed ---------------------------")


@pytest.mark.parametrize("Nel", [[6, 12, 7]])
@pytest.mark.parametrize("p", [[4, 3, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_projectors(Nel, p, spl_kind):
    import numpy as np
    from mpi4py import MPI

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import IGAPolarCylinder

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create control points
    params_map = {"Nel": Nel[:2], "p": p[:2], "Lz": 3.0, "a": 1.0}
    domain = IGAPolarCylinder(**params_map)

    # create polar de Rham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm, nq_pr=[6, 6, 6], polar_ck=1, domain=domain)

    # create legacy FEM spaces
    spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    for space_i in spaces:
        space_i.set_projectors(nq=6)

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors("general")

    if rank == 0:
        print()
        print("Domain decomposition : \n", derham.domain_array)
        print()

    comm.Barrier()

    # function to project on physical domain
    def fun_scalar(x, y, z):
        return np.sin(2 * np.pi * (x)) * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * z)

    fun_vector = [fun_scalar, fun_scalar, fun_scalar]

    # pull-back to logical domain
    def fun0(e1, e2, e3):
        return domain.pull(fun_scalar, e1, e2, e3, kind="0")

    fun1 = [
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="1")[0],
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="1")[1],
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="1")[2],
    ]

    fun2 = [
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="2")[0],
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="2")[1],
        lambda e1, e2, e3: domain.pull(fun_vector, e1, e2, e3, kind="2")[2],
    ]

    def fun3(e1, e2, e3):
        return domain.pull(fun_scalar, e1, e2, e3, kind="3")

    # ============ project on V0 =========================
    if rank == 0:
        r0_pol = derham.P["0"](fun0)
    else:
        r0_pol = derham.P["0"](fun0)

    r0_pol_leg = space.projectors.pi_0(fun0)

    assert np.allclose(r0_pol.toarray(True), r0_pol_leg)

    if rank == 0:
        print("Test passed for PI_0 polar projector")
        print()

    comm.Barrier()

    # ============ project on V1 =========================
    if rank == 0:
        r1_pol = derham.P["1"](fun1)
    else:
        r1_pol = derham.P["1"](fun1)

    r1_pol_leg = space.projectors.pi_1(fun1, with_subs=False)

    assert np.allclose(r1_pol.toarray(True), r1_pol_leg)

    if rank == 0:
        print("Test passed for PI_1 polar projector")
        print()

    comm.Barrier()

    # ============ project on V2 =========================
    if rank == 0:
        r2_pol = derham.P["2"](fun2)
    else:
        r2_pol = derham.P["2"](fun2)

    r2_pol_leg = space.projectors.pi_2(fun2, with_subs=False)

    assert np.allclose(r2_pol.toarray(True), r2_pol_leg)

    if rank == 0:
        print("Test passed for PI_2 polar projector")
        print()

    comm.Barrier()

    # ============ project on V3 =========================
    if rank == 0:
        r3_pol = derham.P["3"](fun3)
    else:
        r3_pol = derham.P["3"](fun3)

    r3_pol_leg = space.projectors.pi_3(fun3, with_subs=False)

    assert np.allclose(r3_pol.toarray(True), r3_pol_leg)

    if rank == 0:
        print("Test passed for PI_3 polar projector")
        print()


if __name__ == "__main__":
    # test_spaces([6, 9, 4], [2, 2, 2], [False, True, False])
    # test_extraction_ops_and_derivatives([8, 12, 6], [2, 2, 3], [False, True, False])
    test_projectors([8, 15, 6], [2, 2, 3], [False, True, True])
