import logging

import pytest

logger = logging.getLogger("struphy")


def _pole_safe(num_elements):
    """At least five elements in eta1 per process: the process at the pole holds the polar rings of regularity 1
    (three rings of 0-forms) and two tensor-product rings, also when eta1 is decomposed."""
    from maybempi import MPI

    return [max(num_elements[0], 5 * MPI.COMM_WORLD.Get_size()), *num_elements[1:]]


@pytest.mark.parametrize("num_elements", [[8, 9, 6]])
@pytest.mark.parametrize("degree", [[3, 2, 4]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "free"), None, ("free", "free")),
    ],
)
def test_spaces(num_elements, degree, bcs):
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import HollowCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarDerhamSpace, PolarVector
    from struphy.topology.grids import TensorProductGrid

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(grid, derham_opts, domain=HollowCylinder(a1=0.0, a2=1.0, Lz=3.0))

    logger.info("polar V0:")
    V = PolarDerhamSpace(derham, "H1")
    logger.info(f"dimensions (parent, polar): {derham.V0fem.nbasis} {V.dimension}")
    logger.info(V.dtype)
    logger.info(f"{V.zeros()}\n")
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

    logger.info("")

    logger.info("polar V1:")
    V = PolarDerhamSpace(derham, "Hcurl")
    logger.info(f"dimensions (parent, polar): {derham.V1fem.nbasis} {V.dimension}")
    logger.info(V.dtype)
    logger.info(f"{V.zeros()}\n")
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

    logger.info("")

    logger.info("polar V2:")
    V = PolarDerhamSpace(derham, "Hdiv")
    logger.info(f"dimensions (parent, polar): {derham.V2fem} {V.dimension}")
    logger.info(V.dtype)
    logger.info(f"{V.zeros()}\n")
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

    logger.info("")

    logger.info("polar V3:")
    V = PolarDerhamSpace(derham, "L2")
    logger.info(f"dimensions (parent, polar): {derham.V3fem} {V.dimension}")
    logger.info(V.dtype)
    logger.info(f"{V.zeros()}\n")
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

    logger.info("")

    logger.info("polar V0vec:")
    V = PolarDerhamSpace(derham, "H1vec")
    logger.info(f"dimensions (parent, polar): {derham.Vvfem.nbasis} {V.dimension}")
    logger.info(V.dtype)
    logger.info(f"{V.zeros()}\n")
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

    logger.info("")


@pytest.mark.parametrize("num_elements", [[6, 9, 6]])
@pytest.mark.parametrize("degree", [[3, 2, 2]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "free"), None, ("free", "free")),
    ],
)
def test_extraction_ops_and_derivatives(num_elements, degree, bcs):
    import cunumpy as xp
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarDerhamSpace, PolarVector
    from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_elements = _pole_safe(num_elements)

    # create control points
    params_map = {"num_elements": num_elements[:2], "degree": degree[:2], "Lz": 3.0, "a": 1.0}
    domain = IGAPolarCylinder(**params_map)

    # create de Rham sequence
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    # create legacy FEM spaces

    if rank == 0:
        logger.info("")
        logger.info(f"Domain decomposition : \n{derham.domain_array}")
        logger.info("")

    comm.Barrier()

    # create polar FEM spaces
    f0_pol = PolarVector(derham.V0pol)
    e1_pol = PolarVector(derham.V1pol)
    b2_pol = PolarVector(derham.V2pol)
    p3_pol = PolarVector(derham.V3pol)

    # create pure tensor-product and polar vectors (legacy and distributed)
    f0_tp_leg, f0_tp = create_equal_random_arrays(derham.V0fem, flattened=True)
    e1_tp_leg, e1_tp = create_equal_random_arrays(derham.V1fem, flattened=True)
    b2_tp_leg, b2_tp = create_equal_random_arrays(derham.V2fem, flattened=True)
    p3_tp_leg, p3_tp = create_equal_random_arrays(derham.V3fem, flattened=True)

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
        logger.info("")

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

    # polar derivatives are consistent with the tensor-product ones: E_{k+1}^T D_pol = D_tp E_k^T
    E = derham.extraction_ops
    for D, k, x in (
        (derham.grad_bcfree, "0", f0_pol),
        (derham.curl_bcfree, "1", e1_pol),
        (derham.div_bcfree, "2", b2_pol),
    ):
        k1 = str(int(k) + 1)
        lhs = E[k1].T.dot(D.dot(x))
        rhs = D.tp_operator.dot(E[k].T.dot(x))
        d = lhs - rhs
        assert xp.sqrt(d.inner(d)) < 1e-12 * xp.sqrt(rhs.inner(rhs)), k

    # exactness of the polar sequence
    r = derham.curl_bcfree.dot(derham.grad_bcfree.dot(f0_pol))
    assert xp.sqrt(r.inner(r)) < 1e-12
    r = derham.div_bcfree.dot(derham.curl_bcfree.dot(e1_pol))
    assert xp.sqrt(r.inner(r)) < 1e-12

    if rank == 0:
        logger.info("------------- Test passed ---------------------------")


@pytest.mark.parametrize("num_elements", [[8, 3, 2], [8, 6, 2]])
def test_polar_adjoints_small_nel2(num_elements):
    """<A x, y> = <x, A^T y> for polar extraction operators and derivatives, including the smallest number of
    angular elements Nel2 = 3 of regularity 1."""
    import cunumpy as xp
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.topology.grids import TensorProductGrid

    degree = [2, 2, 1]
    num_elements = _pole_safe(num_elements)
    domain = IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], Lz=1.0, a=1.0)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, comm=MPI.COMM_WORLD, domain=domain)

    fem_spaces = {"0": derham.V0fem, "1": derham.V1fem, "2": derham.V2fem, "3": derham.V3fem, "v": derham.Vvfem}

    xp.random.seed(1607)

    def rand_pol(space, seed):
        v = PolarVector(space)
        v.tp = create_equal_random_arrays(fem_spaces[derham.space_to_form[space.space_id]], seed=seed)[1]
        v.pol = [xp.random.rand(*pol.shape) for pol in v.pol]
        return v

    for form, V_fem in fem_spaces.items():
        E = derham.extraction_ops[form]
        x = create_equal_random_arrays(V_fem, seed=123)[1]
        y = rand_pol(E.codomain, 456)
        lhs = E.dot(x).dot(y)
        rhs = x.inner(E.transpose().dot(y))
        assert xp.isclose(lhs, rhs, rtol=1e-12), (form, lhs, rhs)

    for op in (derham.grad, derham.curl, derham.div):
        x = rand_pol(op.domain, 123)
        y = rand_pol(op.codomain, 456)
        lhs = op.dot(x).dot(y)
        rhs = x.dot(op.transpose().dot(y))
        assert xp.isclose(lhs, rhs, rtol=1e-12), (lhs, rhs)


def test_polar_ops_eta1_decomposition():
    """With an MPI decomposition in eta_1 (processes away from the pole): the ghost-region flags of the
    tp parts written by polar operators agree on all processes (they must be set collectively),
    and PolarLinearOperator.dot gives the same result for out=v (in-place)."""
    import cunumpy as xp
    from maybempi import MPI
    from scipy.sparse import identity

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.polar.linear_operators import PolarLinearOperator
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    degree = [2, 2, 1]
    num_elements = _pole_safe([16, 6, 4])
    domain = IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], Lz=1.0, a=1.0)
    grid = TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, False, False))
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    def tp_blocks(vec):
        return [vec] if not hasattr(vec, "blocks") else list(vec.blocks)

    def assert_sync_flags_agree(vec, label):
        flags = [b.ghost_regions_in_sync for b in tp_blocks(vec)]
        if comm.Get_size() > 1:
            all_flags = comm.allgather(flags)
            assert all(f == flags for f in all_flags), (label, all_flags)

    E = derham.extraction_ops
    for k, space in (("0", derham.V0pol), ("1", derham.V1pol), ("2", derham.V2pol)):
        x_pol = _rand_polar_vector(derham, space, seed=123)
        for b in tp_blocks(x_pol.tp):
            b.update_ghost_regions()

        # transposed extraction (polar -> tp): writes the inner rings at the pole only
        x_tp = E[k].T.dot(x_pol)
        assert_sync_flags_agree(x_tp, f"E{k}.T")

        # dofs extraction (tp -> polar): writes the "first tp ring" at the pole only (out starts in sync)
        y_pol = PolarVector(space)
        y_pol.tp.update_ghost_regions()
        derham.dofs_extraction_ops[k].dot(x_tp, out=y_pol)
        assert_sync_flags_agree(y_pol.tp, f"P{k}")

    # square PolarLinearOperator on V0pol with random polar blocks
    V = derham.V0pol
    G = derham.grad_bcfree
    rng = xp.random.default_rng(42)
    op = PolarLinearOperator(V, V, tp_operator=G.tp_operator.T @ G.tp_operator)
    op.blocks_pol_to_ten = [[rng.random(shape) for shape in row] for row in op.blocks_pol_to_ten_shapes]
    op.blocks_pol_to_pol = [[rng.random(shape) for shape in row] for row in op.blocks_pol_to_pol_shapes]
    op.blocks_e3 = [[identity(shape[0]) for shape in row] for row in op.blocks_e3_shapes]

    for A in (op, op.transpose()):
        x = _rand_polar_vector(derham, V, seed=5)
        ref = A.dot(x)
        assert_sync_flags_agree(ref.tp, f"{A.transposed = }")
        A.dot(x, out=x)
        assert _rel_err(x, ref) < 1e-14, A.transposed


def test_polar_vector_ghost_sync():
    """Writing the tp part of a PolarVector (tp setter, set_tp_coeffs_to_zero) marks its ghost regions
    as out of sync on all processes, also if they were in sync before. Otherwise a following stencil dot
    skips the halo exchange and reads stale ghost values (issue #748)."""
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    degree = [2, 2, 1]
    num_elements = _pole_safe([16, 6, 4])
    domain = IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], Lz=1.0, a=1.0)
    grid = TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, False, False))
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    def tp_blocks(vec):
        return [vec] if not hasattr(vec, "blocks") else list(vec.blocks)

    def synced_rand_polar_vector(space, seed):
        v = _rand_polar_vector(derham, space, seed=seed)
        v.tp.update_ghost_regions()
        assert v.ghost_regions_in_sync
        return v

    def assert_out_of_sync(vec, label):
        flags = [vec.ghost_regions_in_sync] + [b.ghost_regions_in_sync for b in tp_blocks(vec)]
        if comm.Get_size() > 1:
            all_flags = comm.allgather(flags)
            assert all(f == flags for f in all_flags), (label, all_flags)
        assert not any(flags), (label, flags)

    for k, space, op in (
        ("0", derham.V0pol, derham.grad_bcfree),
        ("1", derham.V1pol, derham.curl_bcfree),
        ("2", derham.V2pol, derham.div_bcfree),
    ):
        fem_space = derham.fem_spaces[derham.space_to_form[space.space_id]]

        # set_tp_coeffs_to_zero on a synced vector
        y = synced_rand_polar_vector(space, seed=1)
        y.set_tp_coeffs_to_zero()
        assert_out_of_sync(y.tp, f"set_tp_coeffs_to_zero V{k}")

        # tp setter: synced target, source with stale (zero) ghost regions
        y = synced_rand_polar_vector(space, seed=1)
        x_tp = create_equal_random_arrays(fem_space, seed=2)[1]
        y.tp = x_tp
        assert_out_of_sync(y.tp, f"tp setter V{k}")

        # stencil dot must exchange the ghost regions; reference starts from a fresh (unsynced) vector
        ref = PolarVector(space)
        ref.tp = x_tp
        res = op.tp_operator.dot(y.tp)
        exp = op.tp_operator.dot(ref.tp)
        diff = res - exp
        assert diff.inner(diff) <= 1e-28 * exp.inner(exp), k


def _rand_polar_vector(derham, space, seed):
    """Random PolarVector (equal on all processes for the same seed)."""
    import cunumpy as xp

    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.polar.basic import PolarVector

    fem_space = derham.fem_spaces[derham.space_to_form[space.space_id]]
    v = PolarVector(space)
    v.tp = create_equal_random_arrays(fem_space, seed=seed)[1]
    rng = xp.random.default_rng(seed)
    v.pol = [rng.random(pol.shape) for pol in v.pol]
    return v


def _rel_err(a, b):
    import cunumpy as xp

    d = a - b
    return xp.sqrt(d.inner(d)) / xp.sqrt(b.inner(b))


@pytest.mark.parametrize("num_elements", [[8, 12, 4]])
@pytest.mark.parametrize("degree", [[3, 3, 2], [2, 3, 1]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "dirichlet"), None, None),
        (("free", "dirichlet"), None, ("dirichlet", "dirichlet")),
    ],
)
def test_projectors(num_elements, degree, bcs):
    """Polar commuting projectors: the polar inter-/histopolation matrix P * I * E^T is the identity, projection of
    splines is the identity (with and without boundary conditions), grad P0 = P1 grad, and exactness of the polar
    sequence."""
    import cunumpy as xp
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    num_elements = _pole_safe(num_elements)
    domain = IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], Lz=3.0, a=1.0)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, nquads_proj=[6, 6, 6], polar_splines=True)
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    # projection of a spline returns the spline, with and without boundary conditions
    for form, P in derham.projectors_global.items():
        if form not in ("0", "1", "2", "3", "v"):
            continue
        x = _rand_polar_vector(derham, P.I.domain, seed=1234)
        y = _rand_polar_vector(derham, P.I.domain, seed=5678)

        # polar inter-/histopolation matrix P * I * E^T (the identity) and its transpose
        Pd, Eb, B = P.dofs_extraction_op, P.base_extraction_op, P.boundary_op
        assert _rel_err(P.I.dot(x), Pd.dot(P.projector_tensor.imat_kronecker.dot(Eb.T.dot(x)))) < 1e-12, form
        assert xp.isclose(P.IT.dot(x).dot(y), x.dot(P.I.dot(y)), rtol=1e-12), form
        assert xp.isclose(P.I0T.dot(x).dot(y), x.dot(P.I0.dot(y)), rtol=1e-12), form
        assert _rel_err(P.I0.dot(x), B.dot(P.I.dot(B.dot(x)))) < 1e-12, form

        assert _rel_err(P.solve(P.I.dot(x), apply_bc=False), x) < 1e-10, form

        x0 = P.boundary_op.dot(x)
        assert _rel_err(P.solve(P.I0.dot(x0), apply_bc=True), x0) < 1e-10, form

        # transposed solve
        assert _rel_err(P.solve(P.IT.dot(x), transposed=True, apply_bc=False), x) < 1e-10, form

    # commutation grad P0 f = P1 grad f (up to the quadrature error in the histopolation)
    tp = 2 * xp.pi

    def f(e1, e2, e3):
        xh, yh = e1 * xp.cos(tp * e2), e1 * xp.sin(tp * e2)
        return xp.sin(xh + 0.3) * xp.cos(yh) * xp.cos(tp * e3)

    def df(e1, e2, e3):
        xh, yh = e1 * xp.cos(tp * e2), e1 * xp.sin(tp * e2)
        fx = xp.cos(xh + 0.3) * xp.cos(yh) * xp.cos(tp * e3)
        fy = -xp.sin(xh + 0.3) * xp.sin(yh) * xp.cos(tp * e3)
        return (
            fx * xp.cos(tp * e2) + fy * xp.sin(tp * e2),
            -fx * e1 * tp * xp.sin(tp * e2) + fy * e1 * tp * xp.cos(tp * e2),
            -xp.sin(xh + 0.3) * xp.cos(yh) * tp * xp.sin(tp * e3),
        )

    dfs = [lambda e1, e2, e3, i=i: df(e1, e2, e3)[i] for i in range(3)]

    c0 = derham.P0(f, apply_bc=False)
    c1 = derham.P1(dfs, apply_bc=False)
    err = _rel_err(derham.grad_bcfree.dot(c0), c1)
    logger.info(f"{bcs = }, {degree = }: |grad P0 f - P1 grad f| / |P1 grad f| = {err:.2e}")
    assert err < 1e-5

    # exactness of the polar sequence
    assert xp.sqrt((r := derham.curl_bcfree.dot(derham.grad_bcfree.dot(c0))).inner(r)) < 1e-12
    b2 = derham.curl_bcfree.dot(c1)
    assert xp.sqrt((r := derham.div_bcfree.dot(b2)).inner(r)) < 1e-12


def test_poloidal_mass_matrix():
    """2d mass matrices of the poloidal plane (unit weight) are Kronecker products of the 1d mass matrices."""
    import scipy.sparse as spa

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import _mass_matrix_1d, _ones_1d
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.kron_2dx1d import poloidal_mass_matrix
    from struphy.topology.grids import TensorProductGrid

    domain = IGAPolarCylinder(num_elements=(8, 9), degree=(3, 2))
    grid = TensorProductGrid(num_elements=[8, 9, 4])
    derham_opts = DerhamOptions(degree=[3, 2, 2], bcs=(("free", "free"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, domain=domain)
    M0 = WeightedMassOperators(derham, domain).M0

    for bases in ("BB", "BM", "MB", "MM"):
        M = poloidal_mass_matrix(derham, tuple(bases), tuple(bases), None)
        A1, A2 = (_mass_matrix_1d(M0, bases[d], d, _ones_1d)[0].tosparse() for d in range(2))
        assert abs(M - spa.kron(A1, A2)).max() < 1e-12, bases


@pytest.mark.parametrize("mapping", ["IGAPolarCylinder", "IGAPolarTorus"])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "free"), None, None),
        (("free", "dirichlet"), None, ("dirichlet", "dirichlet")),
    ],
)
@pytest.mark.parametrize("poloidal_solver", ["schur", "lu", "ilu"])
def test_polar_mass_preconditioner(mapping, bcs, poloidal_solver):
    """The 2D x 1D Kronecker preconditioner is exact for axisymmetric mass matrices (cylinder, torus)
    with exact poloidal solvers, and a good preconditioner with incomplete LU."""
    import cunumpy as xp
    from feectools.linalg.solvers import inverse
    from maybempi import MPI

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import PolarMassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    num_elements, degree = _pole_safe([8, 12, 4]), [3, 3, 2]
    domain = getattr(domains, mapping)(num_elements=num_elements[:2], degree=degree[:2])
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(grid, derham_opts, comm=MPI.COMM_WORLD, domain=domain)
    mass = WeightedMassOperators(derham, domain)

    for name in ("M0", "M1", "M2", "M3", "Mv"):
        M = getattr(mass, name)
        pc = PolarMassMatrixPreconditioner(M, poloidal_solver=poloidal_solver)
        assert pc.exact == (poloidal_solver != "ilu")

        x = M._V_boundary_op.dot(_rand_polar_vector(derham, M.domain, seed=4321))
        b = M.dot(x)

        # exact preconditioner: pc * M = Id (on the space with boundary conditions)
        if pc.exact:
            assert _rel_err(pc.dot(b), x) < 1e-10, name

        solver = inverse(M, "pcg", pc=pc, tol=1e-10, maxiter=100)
        y = solver.dot(b)
        info = solver.get_info()
        logger.info(f"{mapping}, {bcs = }, {poloidal_solver = }, {name}: {info['niter']} iterations")
        assert info["success"], name
        assert info["niter"] <= (2 if pc.exact else 10), (name, info["niter"])
        assert _rel_err(y, x) < 1e-8, name


@pytest.mark.parametrize("mapping", ["IGAPolarCylinder", "IGAPolarTorus"])
@pytest.mark.parametrize(
    "bcs",
    [
        (("free", "dirichlet"), None, None),
        (("free", "dirichlet"), None, ("dirichlet", "dirichlet")),
    ],
)
@pytest.mark.parametrize("poloidal_solver", ["schur", "lu", "ilu"])
def test_polar_stiffness_preconditioner(mapping, bcs, poloidal_solver):
    """The polar stiffness preconditioner (dispatched from StiffnessPreconditioner) is exact for the
    Laplacian on axisymmetric domains with exact poloidal solvers, and a good preconditioner for curl-curl."""
    from feectools.linalg.basic import IdentityOperator
    from feectools.linalg.solvers import inverse
    from maybempi import MPI

    from struphy.feec.mass import WeightedAverageProjection, WeightedMassOperators
    from struphy.feec.preconditioner import PolarStiffnessPreconditioner, StiffnessPreconditioner, _parse_directions
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    num_elements, degree = _pole_safe([8, 12, 6]), [3, 3, 2]
    domain = getattr(domains, mapping)(num_elements=num_elements[:2], degree=degree[:2])
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs, polar_splines=True)
    derham = Derham(grid, derham_opts, comm=MPI.COMM_WORLD, domain=domain)
    mass = WeightedMassOperators(derham, domain)

    with pytest.raises(NotImplementedError):
        StiffnessPreconditioner(mass, "div", sigma=1.0)

    cases = [
        ("grad", 0.0, mass.M0, None, None),
        ("grad", 1.0, mass.M0, None, None),
        ("grad", 1.0, mass.M0, "Id", None),
        ("grad", 10.0, mass.M0, None, "eta3"),
        ("grad", 10.0, mass.M0, "Id", "eta3"),
        ("grad", 10.0, mass.M0, None, "eta2 eta3"),
        ("grad", 10.0, mass.M0, "Id", "eta2 eta3"),
        ("grad", 10.0, mass.M0, None, "eta2"),
        ("curl", 1.0, mass.M1, None, None),
        ("curl", 1e-3, mass.M1, None, None),
    ]
    for derivative, sigma, M, stab, stab_average in cases:
        # incomplete LU is not symmetric; for curl with small sigma the poloidal matrices are nearly
        # singular on the gradients, the asymmetry is large and PCG fails
        if poloidal_solver == "ilu" and derivative == "curl" and sigma < 1.0:
            continue
        d = getattr(derham, derivative)
        mid = mass.M1 if derivative == "grad" else mass.M2
        A = d.T @ mid @ d
        if sigma != 0.0:
            S = M if stab is None else IdentityOperator(M.domain)
            if stab_average is not None:
                avg = WeightedAverageProjection(derham, S, _parse_directions(stab_average))
                S = S @ (IdentityOperator(M.domain) - avg)
            A = A + sigma * S
        pc = StiffnessPreconditioner(
            mass, derivative, sigma=sigma, stab=stab, stab_average=stab_average, poloidal_solver=poloidal_solver
        )
        assert isinstance(pc, PolarStiffnessPreconditioner)
        assert pc.exact == (poloidal_solver != "ilu")

        x = M._V_boundary_op.dot(_rand_polar_vector(derham, M.domain, seed=4321))
        b = A.dot(x)

        # exact for the Laplacian on axisymmetric domains; with Dirichlet conditions in eta_3, the identity
        # stabilization and the averages over eta_3 are approximated
        periodic_3 = bcs[2] is None
        exact = pc.exact and derivative == "grad" and (periodic_3 or (stab is None and stab_average is None))
        if exact:
            assert _rel_err(pc.dot(b), x) < 1e-10, (derivative, sigma, stab, stab_average)

        solver = inverse(A, "pcg", pc=pc, tol=1e-10, maxiter=200)
        y = solver.dot(b)
        info = solver.get_info()
        logger.info(
            f"{mapping}, {bcs = }, {poloidal_solver = }, {derivative}, {sigma = }, {stab = }: {info['niter']} iterations"
        )
        assert info["success"], (derivative, sigma, stab)
        assert info["niter"] <= (2 if exact else 30), (derivative, sigma, stab, info["niter"])
        assert _rel_err(y, x) < 1e-8, (derivative, sigma, stab)


@pytest.mark.parametrize("space_id", ["H1", "Hcurl", "Hdiv", "L2", "H1vec"])
def test_restart_polar(space_id, tmp_path):
    """Write restart data (vector_stencil) like Simulation does and re-initialize a polar SplineFunction from it."""
    import cunumpy as xp
    import h5py
    from feectools.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry.domains import IGAPolarCylinder
    from struphy.io.options import DerhamOptions
    from struphy.polar.basic import PolarVector
    from struphy.topology.grids import TensorProductGrid

    num_elements, degree = [6, 9, 4], [2, 2, 1]
    domain = IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], Lz=1.0, a=1.0)
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "free"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, domain=domain)

    # random polar field
    rng = xp.random.default_rng(1234)
    f = derham.create_spline_function("f", space_id)
    assert isinstance(f.vector, PolarVector)
    f.vector.pol = [rng.random(a.shape) for a in f.vector.pol]
    tps = [f.vector.tp] if isinstance(f.vector.tp, StencilVector) else f.vector.tp.blocks
    for tp in tps:
        tp._data[:] = rng.random(tp._data.shape)
    f.vector.set_tp_coeffs_to_zero()
    f.vector.update_ghost_regions()
    f.extract_coeffs()

    # save restart data
    key = "restart/em_fields/f"
    with h5py.File(tmp_path / "data.hdf5", "w") as file:
        if isinstance(f.vector_stencil, StencilVector):
            file.create_dataset(key, data=f.vector_stencil._data[None])
        else:
            for n in range(3):
                file.create_dataset(key + "/" + str(n + 1), data=f.vector_stencil[n]._data[None])

    # restart
    g = derham.create_spline_function("g", space_id)
    with h5py.File(tmp_path / "data.hdf5", "r") as file:
        g.initialize_coeffs_from_restart_file(file, key)

    assert xp.allclose(g.vector.toarray(), f.vector.toarray(), atol=1e-12)


@pytest.mark.parametrize("mapping", ["IGAPolarCylinder", "IGAPolarTorus"])
@pytest.mark.parametrize("stab", ["M0", "Id"])
@pytest.mark.parametrize("directions", [(2,), (1, 2), (1,)])
def test_polar_weighted_average_projection(mapping, stab, directions):
    """The weighted average on polar 0-forms is a projection onto functions constant in the averaged
    directions, and S (I - P) is symmetric."""
    import numpy as np
    from feectools.linalg.basic import IdentityOperator
    from maybempi import MPI

    from struphy.feec.mass import PolarWeightedAverageProjection, WeightedAverageProjection, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.polar.kron_2dx1d import _interior
    from struphy.topology.grids import TensorProductGrid

    num_elements, degree = _pole_safe([8, 12, 6]), [3, 3, 2]
    domain = getattr(domains, mapping)(num_elements=num_elements[:2], degree=degree[:2])
    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "dirichlet"), None, None), polar_splines=True)
    derham = Derham(grid, derham_opts, comm=MPI.COMM_WORLD, domain=domain)
    mass = WeightedMassOperators(derham, domain)
    V = mass.M0.domain
    S = mass.M0 if stab == "M0" else IdentityOperator(V)

    with pytest.raises(NotImplementedError):
        WeightedAverageProjection(derham, S, (0, 2))

    P = WeightedAverageProjection(derham, S, directions)
    assert isinstance(P, PolarWeightedAverageProjection)

    x = mass.M0._V_boundary_op.dot(_rand_polar_vector(derham, V, seed=3))
    y = mass.M0._V_boundary_op.dot(_rand_polar_vector(derham, V, seed=5))
    Px = P.dot(x)

    # projection, onto functions constant in the averaged directions (local tensor-product part)
    assert _rel_err(P.dot(Px), Px) < 1e-12
    tp = np.asarray(_interior(Px.tp))
    if 2 in directions:
        assert np.allclose(Px.pol[0], Px.pol[0][:, :1])
        assert np.allclose(tp, tp[:, :, :1])
    if 1 in directions:
        assert np.allclose(tp, tp[:, :1, :])

    # S (I - P) is symmetric and vanishes on the averages
    A = S @ (IdentityOperator(V) - P)
    assert abs(A.dot(x).dot(y) - x.dot(A.dot(y))) < 1e-12 * abs(x.dot(A.dot(y)))
    r = A.dot(Px)
    s = S.dot(Px)
    assert r.dot(r) ** 0.5 < 1e-12 * s.dot(s) ** 0.5


if __name__ == "__main__":
    # test_spaces([6, 9, 4], [2, 2, 2], [False, True, False])
    # test_extraction_ops_and_derivatives([8, 12, 6], [2, 2, 3], [False, True, False])
    test_projectors([8, 12, 4], [3, 3, 2], (("free", "dirichlet"), None, None))
