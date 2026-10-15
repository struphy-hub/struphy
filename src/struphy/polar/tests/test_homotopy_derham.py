"""Tests of the high-order (homotopy) polar splines in :class:`~struphy.feec.psydac_derham.Derham`."""

import numpy as np
import pytest


def _derham(domain, num_elements=(8, 12, 1), degree=(3, 3, 1), regularity=2, bcs=(("free", "free"), None, None), **kw):
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    opts = DerhamOptions(
        degree=degree, bcs=bcs, polar_splines=True, polar_method="homotopy", polar_regularity=regularity, **kw
    )
    # eta1 is not decomposed: the process at the pole holds all polar rings
    grid = TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(False, True, True))
    return Derham(grid, opts, comm=MPI.COMM_WORLD, domain=domain)


def _domain(name):
    from struphy.geometry import domains

    if name == "PerturbedDiskCylinder":
        return domains.PerturbedDiskCylinder(R1=1.0, eps=0.2, k=3, Lz=1.0)
    if name == "HollowCylinder":
        return domains.HollowCylinder(a1=0.0, a2=1.0, Lz=1.0)
    if name == "HollowTorus":
        return domains.HollowTorus(a1=0.0, a2=0.3, R0=1.0, tor_period=1)
    return getattr(domains, name)(num_elements=(8, 12), degree=(3, 3))


@pytest.mark.parametrize("domain_name", ["PerturbedDiskCylinder", "HollowCylinder", "IGAPolarCylinder"])
@pytest.mark.parametrize("regularity", [1, 2, 3])
def test_dimensions_match_total_complex(domain_name, regularity):
    derham = _derham(_domain(domain_name), regularity=regularity, num_elements=(8, 12, 4), degree=(3, 3, 2))
    c = derham.ck_blocks.complex2d
    n3B, n3M = derham.ck_blocks.n3["B"], derham.ck_blocks.n3["M"]
    d0, d1, d2 = (c.dimension(k) for k in range(3))
    expected = {"H1": d0 * n3B, "Hcurl": d1 * n3B + d0 * n3M, "Hdiv": d1 * n3M + d2 * n3B, "L2": d2 * n3M}
    for space_id, dim in expected.items():
        assert derham.polar_coeff_spaces[derham.space_to_form[space_id]].dimension == dim, space_id


@pytest.mark.parametrize("domain_name", ["PerturbedDiskCylinder", "IGAPolarCylinder"])
@pytest.mark.parametrize("regularity", [1, 3])
def test_poloidal_blocks_equal_core_derivatives(domain_name, regularity):
    """With one cell in eta3 the poloidal parts of grad/curl are the core matrices D_bar = F~^T D E."""
    derham = _derham(_domain(domain_name), regularity=regularity)
    blocks = derham.ck_blocks
    c = blocks.complex2d
    sets = blocks._sets
    D0, D1 = c.derivative(0).toarray(), c.derivative(1).toarray()
    s0, s1x, s1y, s2 = sets["0"], sets["1x"], sets["1y"], sets["2"]
    # grad: Hcurl x <- H1, Hcurl y <- H1
    assert np.allclose(blocks.grad_pol_to_pol[0][0].toarray(), D0[np.ix_(s1x, s0)], atol=1e-12)
    assert np.allclose(blocks.grad_pol_to_pol[1][0].toarray(), D0[np.ix_(s1y, s0)], atol=1e-12)
    # curl: Hdiv z <- Hcurl x, y
    assert np.allclose(blocks.curl_pol_to_pol[2][0].toarray(), D1[np.ix_(s2, s1x)], atol=1e-12)
    assert np.allclose(blocks.curl_pol_to_pol[2][1].toarray(), D1[np.ix_(s2, s1y)], atol=1e-12)
    # div: the polar Hdiv basis functions are Lambda^{1,1} E_l (B_x = a_y, B_y = -a_x), whose divergence is the 2D
    # curl of E_l
    assert np.allclose(blocks.div_pol_to_pol[0][0].toarray(), D1[np.ix_(s2, s1y)], atol=1e-12)
    assert np.allclose(blocks.div_pol_to_pol[0][1].toarray(), D1[np.ix_(s2, s1x)], atol=1e-12)


@pytest.mark.parametrize("domain_name", ["PerturbedDiskCylinder", "HollowTorus"])
def test_exactness(domain_name):
    """d o d = 0 for the polar operators (polar and tensor-product parts)."""
    from struphy.polar.tests.test_polar import _rand_polar_vector

    derham = _derham(_domain(domain_name), regularity=2, num_elements=(6, 12, 4), degree=(3, 3, 1))
    spaces = [derham.polar_coeff_spaces[f] for f in ("0", "1", "2", "3")]
    ops = [derham.grad_bcfree, derham.curl_bcfree, derham.div_bcfree]
    for k in range(2):
        x = _rand_polar_vector(derham, spaces[k], seed=7)
        y = ops[k + 1].dot(ops[k].dot(x))
        assert np.sqrt(y.inner(y)) < 1e-12 * np.sqrt(x.inner(x))


def _logical_function(a, b, c):
    return (
        np.sin(2 * np.pi * b) * a**2 * np.cos(2 * np.pi * c) + a**3 * np.cos(4 * np.pi * b) + a * np.sin(2 * np.pi * b)
    )


@pytest.mark.parametrize("domain_name", ["PerturbedDiskCylinder", "HollowTorus"])
@pytest.mark.parametrize("regularity", [1, 3])
@pytest.mark.parametrize("eta3", [(4, 2), (1, 1)])
def test_commuting_projectors(domain_name, regularity, eta3):
    """grad P0 = P1 grad, curl P1 = P2 curl, div P2 = P3 div (up to the quadrature of the GDOFs); also with a single
    periodic cell in eta3 (the usual setting of 2D runs)."""
    derham = _derham(
        _domain(domain_name),
        regularity=regularity,
        num_elements=(8, 12, eta3[0]),
        degree=(3, 3, eta3[1]),
        nquads_proj=(8, 10, 10),
    )
    h = 1e-6

    def d(fun, i):
        def g(a, b, c):
            ep, em = [a, b, c], [a, b, c]
            ep[i] = ep[i] + h
            em[i] = em[i] - h
            return (fun(*ep) - fun(*em)) / (2 * h)

        return g

    f = _logical_function
    u, g = derham.P0(f), derham.P1([d(f, 0), d(f, 1), d(f, 2)])
    r = derham.grad.dot(u) - g
    assert np.sqrt(r.inner(r)) < 1e-9 * max(np.sqrt(g.inner(g)), 1.0)

    A = [
        lambda a, b, c: np.cos(2 * np.pi * b) * a + 0 * c,
        lambda a, b, c: a**2 * np.sin(2 * np.pi * b) * np.cos(2 * np.pi * c),
        lambda a, b, c: a**3 * np.sin(2 * np.pi * b),
    ]
    curlA = [
        lambda a, b, c: d(A[2], 1)(a, b, c) - d(A[1], 2)(a, b, c),
        lambda a, b, c: d(A[0], 2)(a, b, c) - d(A[2], 0)(a, b, c),
        lambda a, b, c: d(A[1], 0)(a, b, c) - d(A[0], 1)(a, b, c),
    ]
    a1, b2 = derham.P1(A), derham.P2(curlA)
    r = derham.curl.dot(a1) - b2
    assert np.sqrt(r.inner(r)) < 1e-9 * max(np.sqrt(b2.inner(b2)), 1.0)
    B = [lambda a, b, c: a * np.cos(2 * np.pi * b) * np.cos(2 * np.pi * c), lambda a, b, c: a**2 + 0 * b + 0 * c, A[2]]

    def divB(a, b, c):
        return d(B[0], 0)(a, b, c) + d(B[1], 1)(a, b, c) + d(B[2], 2)(a, b, c)

    b2, r3 = derham.P2(B), derham.P3(divB)
    r = derham.div.dot(b2) - r3
    assert np.sqrt(r.inner(r)) < 1e-9 * max(np.sqrt(r3.inner(r3)), 1.0)


@pytest.mark.parametrize("domain_name", ["HollowCylinder", "IGAPolarTorus"])
@pytest.mark.parametrize(
    "bcs", [(("free", "free"), None, None), (("free", "dirichlet"), None, ("dirichlet", "dirichlet"))]
)
def test_mass_matrices_and_preconditioner(domain_name, bcs):
    from feectools.linalg.solvers import inverse

    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import PolarMassMatrixPreconditioner
    from struphy.polar.tests.test_polar import _rand_polar_vector, _rel_err

    domain = _domain(domain_name)
    derham = _derham(domain, num_elements=(8, 12, 4), degree=(3, 3, 2), regularity=3, bcs=bcs)
    mass = WeightedMassOperators(derham, domain)
    for name in ("M0", "M1", "M2", "M3", "Mv"):
        M = getattr(mass, name)
        pc = PolarMassMatrixPreconditioner(M)
        x = M._V_boundary_op.dot(_rand_polar_vector(derham, M.domain, seed=4321))
        b = M.dot(x)
        solver = inverse(M, "pcg", pc=pc, tol=1e-10, maxiter=300)
        y = solver.dot(b)
        assert solver.get_info()["success"], name
        assert _rel_err(y, x) < 1e-8, name
