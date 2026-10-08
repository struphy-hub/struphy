import pytest

FORMS = {"grad": "0", "curl": "1", "div": "2"}


def _blocks(v):
    from feectools.linalg.block import BlockVector

    return v.blocks if isinstance(v, BlockVector) else (v,)


def _max_abs(arrays, comm):
    import cunumpy as xp
    from maybempi import MPI

    m = max(float(xp.max(xp.abs(a))) for a in arrays)
    return comm.allreduce(m, op=MPI.MAX) if comm is not None and comm.Get_size() > 1 else m


@pytest.mark.parametrize("derivative", ["grad", "curl", "div"])
@pytest.mark.parametrize("sigma", [0.0, 0.7])
def test_stiffness_approximation_unit_cube(derivative, sigma):
    """On the unit cube (unit weights), the diagonal blocks of the Kronecker approximation equal those of
    the stiffness operator, the solver inverts the approximation, and probing gives its exact diagonal."""

    from maybempi import MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner, _local_diagonal, _probe_diagonal
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD
    derham = Derham(
        TensorProductGrid(num_elements=[6, 5, 4]), DerhamOptions(degree=[2, 3, 2], bcs=(None, None, None)), comm=comm
    )
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())

    P = StiffnessPreconditioner(
        mass_ops, derivative, sigma=sigma, apply_bc=False, diagonal_scaling=False, kernel_correction=False
    )
    A = P.core_operator
    x = create_equal_random_arrays(derham.fem_spaces[FORMS[derivative]], seed=1)[1]

    # diagonal blocks: apply to one component at a time
    for c in range(len(_blocks(x))):
        xc = x.copy()
        for k, b in enumerate(_blocks(xc)):
            if k != c:
                b._data[:] = 0.0
        r = A.dot(xc) - P.matrix.dot(xc)
        ref = A.dot(xc)
        assert _max_abs([_blocks(r)[c].toarray()], comm) < 1e-12 * _max_abs([_blocks(ref)[c].toarray()], comm)

    if sigma > 0:
        r = P.solver.dot(P.matrix.dot(x)) - x
        assert _max_abs([b.toarray() for b in _blocks(r)], comm) < 1e-12

    # probing the stiffness operator gives its diagonal, which equals the one of the approximation
    diffs = [a - b for a, b in zip(_probe_diagonal(A, P.matrix.domain), _local_diagonal(P.matrix))]
    assert _max_abs(diffs, comm) < 1e-12 * _max_abs(_local_diagonal(P.matrix), comm)


@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder"])
@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_stiffness_preconditioner_grad(mapping, sigma):
    """PCG for G^T M1 G + sigma M0: exact on the unit cube, fewer iterations than without preconditioner otherwise."""

    from feectools.linalg.solvers import inverse
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=[8, 10, 4]),
        DerhamOptions(degree=[2, 3, 1], bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
    )
    domain = domains.Cuboid() if mapping == "Cuboid" else domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    A = derham.grad.T @ mass_ops.M1 @ derham.grad
    if sigma:
        A = A + sigma * mass_ops.M0
    b = derham.boundary_ops["0"].dot(create_equal_random_arrays(derham.V0fem, seed=2, flattened=True)[1])

    niter = {}
    for label, P in (
        ("none", None),
        ("unit", StiffnessPreconditioner(mass_ops, "grad", sigma=sigma, diagonal_scaling=False)),
        ("scaled", StiffnessPreconditioner(mass_ops, "grad", sigma=sigma)),
    ):
        inv = inverse(A, "pcg", pc=P, tol=1e-10, maxiter=2000) if P else inverse(A, "cg", tol=1e-10, maxiter=2000)
        inv.dot(b)
        assert inv._info["success"]
        niter[label] = inv._info["niter"]

    if mapping == "Cuboid":
        assert niter["unit"] <= 2
        assert niter["scaled"] <= 2
    else:
        assert niter["unit"] < niter["none"]
        assert niter["scaled"] <= niter["unit"]


@pytest.mark.parametrize("derivative", ["curl", "div"])
@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder"])
def test_stiffness_preconditioner_kernel_correction(derivative, mapping):
    """PCG for C^T M2 C + sigma M1 and D^T M3 D + sigma M2: with kernel correction, fewer iterations
    than with block Jacobi alone and than with the mass-matrix preconditioner."""

    from feectools.linalg.solvers import inverse
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixPreconditioner, StiffnessPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=[8, 10, 4]),
        DerhamOptions(degree=[2, 3, 1], bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
    )
    domain = domains.Cuboid() if mapping == "Cuboid" else domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    k = int(FORMS[derivative])
    d = getattr(derham, derivative)
    M_k, M_k1 = getattr(mass_ops, f"M{k}"), getattr(mass_ops, f"M{k + 1}")
    sigma = 1.0
    A = d.T @ M_k1 @ d + sigma * M_k
    b = derham.boundary_ops[FORMS[derivative]].dot(
        create_equal_random_arrays(derham.fem_spaces[FORMS[derivative]], seed=2, flattened=True)[1]
    )

    P = StiffnessPreconditioner(mass_ops, derivative, sigma=sigma)
    assert P.kernel_correction is not None
    niter = {}
    for label, pc in (
        ("mass", MassMatrixPreconditioner(M_k, diagonal_scaling=True)),
        ("block Jacobi", StiffnessPreconditioner(mass_ops, derivative, sigma=sigma, kernel_correction=False)),
        ("kernel correction", P),
    ):
        inv = inverse(A, "pcg", pc=pc, tol=1e-10, maxiter=3000)
        inv.dot(b)
        assert inv._info["success"]
        niter[label] = inv._info["niter"]

    assert niter["kernel correction"] < niter["block Jacobi"]
    assert niter["kernel correction"] < niter["mass"]

    with pytest.raises(ValueError):
        StiffnessPreconditioner(mass_ops, derivative, sigma=0.0)


@pytest.mark.parametrize("derivative", ["grad", "curl", "div"])
def test_stiffness_preconditioner_mpi(derivative):
    """The preconditioner must not depend on the MPI decomposition."""

    import cunumpy as xp
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    out = []
    for comm in (MPI.COMM_WORLD, None):
        derham = Derham(
            TensorProductGrid(num_elements=[7, 5, 4]),
            DerhamOptions(degree=[2, 2, 1], bcs=(("dirichlet", "free"), None, None)),
            comm=comm,
        )
        mass_ops = WeightedMassOperators(derham, domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0))
        _, v = create_equal_random_arrays(derham.fem_spaces[FORMS[derivative]], seed=1234)
        out += [StiffnessPreconditioner(mass_ops, derivative, sigma=1.0).dot(v)]

    for a, b in zip(_blocks(out[0]), _blocks(out[1])):
        sl = tuple(slice(si, ei + 1) for si, ei in zip(a.space.starts, a.space.ends))
        assert xp.allclose(a[sl], b[sl], rtol=1e-12, atol=1e-14)


def test_stiffness_preconditioner_options():
    """Option checks, transpose and polar splines."""

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(TensorProductGrid(num_elements=[4, 4, 2]), DerhamOptions(degree=[1, 1, 1], bcs=(None, None, None)))
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())
    P = StiffnessPreconditioner(mass_ops, "grad", sigma=0.5)
    assert P.transpose() is P
    assert P.derivative == "grad" and P.sigma == 0.5 and P.diagonal_scaling
    with pytest.raises(AssertionError):
        StiffnessPreconditioner(mass_ops, "rot")
    with pytest.raises(AssertionError):
        StiffnessPreconditioner(mass_ops, "grad", sigma=-1.0)
