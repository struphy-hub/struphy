import pytest


@pytest.mark.parametrize("Nel", [[8, 9, 6]])
@pytest.mark.parametrize("p", [[3, 2, 4]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_spaces(Nel, p, spl_kind):
    from struphy.feec.psydac_derham import Derham
    from struphy.polar.basic import PolarDerhamSpace, PolarVector

    derham = Derham(Nel, p, spl_kind)

    logger.info("polar V0:")
    V = PolarDerhamSpace(derham, "H1")
    logger.info("dimensions (parent, polar):", derham.Vh_fem["0"].nbasis, V.dimension)
    logger.info(V.dtype)
    logger.info(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.tp[:] = 1.0
    logger.info(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    logger.info(a.toarray())
    logger.info(a.dot(b))
    logger.info((-a).toarray())
    logger.info((2 * a).toarray())
    logger.info((a * 2).toarray())
    logger.info((a + b).toarray())
    logger.info((a - b).toarray())
    a *= 2
    logger.info(a.toarray())
    a += b
    logger.info(a.toarray())
    a -= b
    logger.info(a.toarray())
    logger.info(a.toarray_tp())

    logger.info()

    logger.info("polar V1:")
    V = PolarDerhamSpace(derham, "Hcurl")
    logger.info("dimensions (parent, polar):", derham.Vh_fem["1"].nbasis, V.dimension)
    logger.info(V.dtype)
    logger.info(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    logger.info(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    logger.info(a.toarray())
    logger.info(a.dot(b))
    logger.info((-a).toarray())
    logger.info((2 * a).toarray())
    logger.info((a * 2).toarray())
    logger.info((a + b).toarray())
    logger.info((a - b).toarray())
    a *= 2
    logger.info(a.toarray())
    a += b
    logger.info(a.toarray())
    a -= b
    logger.info(a.toarray())
    logger.info(a.toarray_tp())

    logger.info()

    logger.info("polar V2:")
    V = PolarDerhamSpace(derham, "Hdiv")
    logger.info("dimensions (parent, polar):", derham.Vh_fem["2"], V.dimension)
    logger.info(V.dtype)
    logger.info(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    logger.info(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    logger.info(a.toarray())
    logger.info(a.dot(b))
    logger.info((-a).toarray())
    logger.info((2 * a).toarray())
    logger.info((a * 2).toarray())
    logger.info((a + b).toarray())
    logger.info((a - b).toarray())
    a *= 2
    logger.info(a.toarray())
    a += b
    logger.info(a.toarray())
    a -= b
    logger.info(a.toarray())
    logger.info(a.toarray_tp())

    logger.info()

    logger.info("polar V3:")
    V = PolarDerhamSpace(derham, "L2")
    logger.info("dimensions (parent, polar):", derham.Vh_fem["3"], V.dimension)
    logger.info(V.dtype)
    logger.info(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.tp[:] = 1.0
    logger.info(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    logger.info(a.toarray())
    logger.info(a.dot(b))
    logger.info((-a).toarray())
    logger.info((2 * a).toarray())
    logger.info((a * 2).toarray())
    logger.info((a + b).toarray())
    logger.info((a - b).toarray())
    a *= 2
    logger.info(a.toarray())
    a += b
    logger.info(a.toarray())
    a -= b
    logger.info(a.toarray())
    logger.info(a.toarray_tp())

    logger.info()

    logger.info("polar V0vec:")
    V = PolarDerhamSpace(derham, "H1vec")
    logger.info("dimensions (parent, polar):", derham.Vh_fem["v"].nbasis, V.dimension)
    logger.info(V.dtype)
    logger.info(V.zeros(), "\n")
    a = PolarVector(V)
    a.pol[0][:] = 1.0
    a.pol[1][:] = 2.0
    a.pol[2][:] = 3.0
    a.tp[0][:] = 1.0
    a.tp[1][:] = 2.0
    a.tp[2][:] = 3.0
    logger.info(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    logger.info(a.toarray())
    logger.info(a.dot(b))
    logger.info((-a).toarray())
    logger.info((2 * a).toarray())
    logger.info((a * 2).toarray())
    logger.info((a + b).toarray())
    logger.info((a - b).toarray())
    a *= 2
    logger.info(a.toarray())
    a += b
    logger.info(a.toarray())
    a -= b
    logger.info(a.toarray())
    logger.info(a.toarray_tp())

    logger.info()


@pytest.mark.parametrize("Nel", [[6, 9, 6]])
@pytest.mark.parametrize("p", [[3, 2, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_extraction_ops_and_derivatives(Nel, p, spl_kind):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

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

    if rank == 0:
        logger.info()
        logger.info("Domain decomposition : \n", derham.domain_array)
        logger.info()

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

    xp.random.seed(1607)
    f0_pol.pol = [xp.random.rand(f0_pol.pol[0].shape[0], f0_pol.pol[0].shape[1])]
    e1_pol.pol = [xp.random.rand(e1_pol.pol[n].shape[0], e1_pol.pol[n].shape[1]) for n in range(3)]
    b2_pol.pol = [xp.random.rand(b2_pol.pol[n].shape[0], b2_pol.pol[n].shape[1]) for n in range(3)]
    p3_pol.pol = [xp.random.rand(p3_pol.pol[0].shape[0], p3_pol.pol[0].shape[1])]

    f0_pol_leg = f0_pol.toarray(True)
    e1_pol_leg = e1_pol.toarray(True)
    b2_pol_leg = b2_pol.toarray(True)
    p3_pol_leg = p3_pol.toarray(True)

    # ==================== test basis extraction operators ===================
    if rank == 0:
        logger.info("----------- Test basis extraction operators ---------")

    # test basis extraction operator
    r0_pol = derham.extraction_ops["0"].dot(f0_tp)
    r1_pol = derham.extraction_ops["1"].dot(e1_tp)
    r2_pol = derham.extraction_ops["2"].dot(b2_tp)
    r3_pol = derham.extraction_ops["3"].dot(p3_tp)

    # test transposed extraction operators
    E0T = derham.extraction_ops["0"].transpose()
    E1T = derham.extraction_ops["1"].transpose()
    E2T = derham.extraction_ops["2"].transpose()
    E3T = derham.extraction_ops["3"].transpose()

    r0 = E0T.dot(f0_pol)
    r1 = E1T.dot(e1_pol)
    r2 = E2T.dot(b2_pol)
    r3 = E3T.dot(p3_pol)

    if rank == 0:
        logger.info("------------- Test passed ---------------------------")
        logger.info()

    # ==================== test discrete derivatives ======================
    if rank == 0:
        logger.info("----------- Test discrete derivatives ---------")

    # test discrete derivatives
    r1_pol = derham.grad.dot(f0_pol)
    r2_pol = derham.curl.dot(e1_pol)
    r3_pol = derham.div.dot(b2_pol)

    # test transposed derivatives
    GT = derham.grad.transpose()
    CT = derham.curl.transpose()
    DT = derham.div.transpose()

    r0_pol = GT.dot(e1_pol)
    r1_pol = CT.dot(b2_pol)
    r2_pol = DT.dot(p3_pol)

    if rank == 0:
        logger.info("------------- Test passed ---------------------------")


@pytest.mark.parametrize("Nel", [[6, 12, 7]])
@pytest.mark.parametrize("p", [[4, 3, 2]])
@pytest.mark.parametrize("spl_kind", [[False, True, True], [False, True, False]])
def test_projectors(Nel, p, spl_kind):
    import cunumpy as xp
    from feectools.ddm.mpi import mpi as MPI

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

    if rank == 0:
        logger.info()
        logger.info("Domain decomposition : \n", derham.domain_array)
        logger.info()

    comm.Barrier()

    # function to project on physical domain
    def fun_scalar(x, y, z):
        return xp.sin(2 * xp.pi * (x)) * xp.cos(2 * xp.pi * y) * xp.sin(2 * xp.pi * z)

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

    if rank == 0:
        logger.info("Test passed for PI_0 polar projector")
        logger.info()

    comm.Barrier()

    # ============ project on V1 =========================
    if rank == 0:
        r1_pol = derham.P["1"](fun1)
    else:
        r1_pol = derham.P["1"](fun1)

    if rank == 0:
        logger.info("Test passed for PI_1 polar projector")
        logger.info()

    comm.Barrier()

    # ============ project on V2 =========================
    if rank == 0:
        r2_pol = derham.P["2"](fun2)
    else:
        r2_pol = derham.P["2"](fun2)

    if rank == 0:
        logger.info("Test passed for PI_2 polar projector")
        logger.info()

    comm.Barrier()

    # ============ project on V3 =========================
    if rank == 0:
        r3_pol = derham.P["3"](fun3)
    else:
        r3_pol = derham.P["3"](fun3)

    if rank == 0:
        logger.info("Test passed for PI_3 polar projector")
        logger.info()


if __name__ == "__main__":
    # test_spaces([6, 9, 4], [2, 2, 2], [False, True, False])
    # test_extraction_ops_and_derivatives([8, 12, 6], [2, 2, 3], [False, True, False])
    test_projectors([8, 15, 6], [2, 2, 3], [False, True, True])
