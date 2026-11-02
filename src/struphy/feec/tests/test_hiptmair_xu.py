import pytest


def _setup(num_elements, comm, mapping="Cuboid", degree=(2, 2, 2)):
    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=num_elements),
        DerhamOptions(degree=degree, bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=comm,
    )
    domain = domains.Cuboid() if mapping == "Cuboid" else domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    return derham, WeightedMassOperators(derham, domain)


def _niter(A, pc, b):
    from feectools.linalg.solvers import inverse

    inv = inverse(A, "pcg", pc=pc, tol=1e-8, maxiter=2000)
    inv.dot(b)
    assert inv._info["success"]
    return inv._info["niter"]


@pytest.mark.parametrize("smoother", ["jacobi", "block_jacobi"])
def test_hiptmair_xu_robust(smoother):
    """On the unit cube (isotropic mesh), the PCG iterations hardly depend on the mesh size and on sigma."""
    from maybempi import MPI

    from struphy.feec.preconditioner import HiptmairXuPreconditioner
    from struphy.feec.utilities import create_equal_random_arrays

    niter = {}
    for N in (8, 16):
        derham, mass_ops = _setup([N, N, N], MPI.COMM_WORLD)
        b = derham.boundary_ops["1"].dot(create_equal_random_arrays(derham.fem_spaces["1"], seed=2, flattened=True)[1])
        for sigma in (1.0, 1e-4):
            A = derham.curl.T @ mass_ops.M2 @ derham.curl + sigma * mass_ops.M1
            niter[N, sigma] = _niter(A, HiptmairXuPreconditioner(A, mass_ops, sigma, smoother=smoother), b)

    for sigma in (1.0, 1e-4):
        assert niter[16, sigma] <= 1.2 * niter[8, sigma] + 2
    assert max(niter.values()) <= 1.5 * min(niter.values()) + 2
    if smoother == "block_jacobi":
        assert max(niter.values()) <= 25


@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder"])
def test_hiptmair_xu_vs_mass(mapping):
    """HX needs far fewer iterations than the mass-matrix preconditioner, in particular for small sigma."""
    from maybempi import MPI

    from struphy.feec.preconditioner import HiptmairXuPreconditioner, MassMatrixPreconditioner
    from struphy.feec.utilities import create_equal_random_arrays

    derham, mass_ops = _setup([8, 10, 4], MPI.COMM_WORLD, mapping=mapping, degree=(2, 3, 1))
    b = derham.boundary_ops["1"].dot(create_equal_random_arrays(derham.fem_spaces["1"], seed=2, flattened=True)[1])
    sigma = 1e-3
    A = derham.curl.T @ mass_ops.M2 @ derham.curl + sigma * mass_ops.M1
    n_hx = _niter(A, HiptmairXuPreconditioner(A, mass_ops, sigma), b)
    n_mass = _niter(A, MassMatrixPreconditioner(mass_ops.M1), b)
    assert 3 * n_hx < n_mass


@pytest.mark.parametrize("smoother", ["jacobi", "block_jacobi"])
def test_hiptmair_xu_mpi(smoother):
    """The preconditioner must not depend on the MPI decomposition."""
    import cunumpy as xp
    from maybempi import MPI

    from struphy.feec.preconditioner import HiptmairXuPreconditioner
    from struphy.feec.utilities import create_equal_random_arrays

    out = []
    for comm in (MPI.COMM_WORLD, None):
        derham, mass_ops = _setup([8, 6, 4], comm, mapping="HollowCylinder", degree=(2, 2, 1))
        A = derham.curl.T @ mass_ops.M2 @ derham.curl + 0.3 * mass_ops.M1
        _, v = create_equal_random_arrays(derham.fem_spaces["1"], seed=1234)
        out += [HiptmairXuPreconditioner(A, mass_ops, 0.3, smoother=smoother).dot(v)]

    for a, b in zip(out[0].blocks, out[1].blocks):
        sl = tuple(slice(si, ei + 1) for si, ei in zip(a.space.starts, a.space.ends))
        assert xp.allclose(a[sl], b[sl], rtol=1e-10, atol=1e-12)


def test_hiptmair_xu_options():
    """Symmetry, option checks."""
    from maybempi import MPI

    from struphy.feec.preconditioner import HiptmairXuPreconditioner
    from struphy.feec.utilities import create_equal_random_arrays

    derham, mass_ops = _setup([6, 5, 4], MPI.COMM_WORLD, degree=(2, 2, 1))
    A = derham.curl.T @ mass_ops.M2 @ derham.curl + 0.5 * mass_ops.M1
    P = HiptmairXuPreconditioner(A, mass_ops, 0.5)
    assert P.transpose() is P
    assert P.smoother == "block_jacobi" and P.sigma == 0.5 and P.aux_weight == 1.0

    # symmetric: u^T P v = v^T P u
    _, u = create_equal_random_arrays(derham.fem_spaces["1"], seed=1)
    _, v = create_equal_random_arrays(derham.fem_spaces["1"], seed=2)
    u, v = derham.boundary_ops["1"].dot(u), derham.boundary_ops["1"].dot(v)
    assert abs(u.inner(P.dot(v)) - v.inner(P.dot(u))) < 1e-10 * abs(u.inner(P.dot(v)))

    with pytest.raises(ValueError):
        HiptmairXuPreconditioner(A, mass_ops, 0.0)
    with pytest.raises(AssertionError):
        HiptmairXuPreconditioner(A, mass_ops, 0.5, smoother="gauss_seidel")
    with pytest.raises(AssertionError):
        HiptmairXuPreconditioner(None, mass_ops, 0.5, smoother="jacobi")
