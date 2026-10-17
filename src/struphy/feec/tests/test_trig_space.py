"""Trigonometric functions in eta2 of the Derham complex (``DerhamOptions(angular_basis="trig")``)."""

import numpy as np
import pytest


def _derham(n=9, polar=False, regularity=2, domain=None, num_elements=(6, None, 1), degree=(3, 3, 1)):
    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    domain = domains.HollowCylinder(a1=0.0, a2=1.0, Lz=1.0) if domain is None else domain
    options = DerhamOptions(
        degree=degree,
        bcs=(("free", "free"), None, None),
        angular_basis="trig",
        polar_splines=polar,
        polar_regularity=regularity,
        nquads=(degree[0] + 1, n + 2, 2),
    )
    grid = TensorProductGrid(num_elements=(num_elements[0], n, num_elements[2]), mpi_dims_mask=(True, False, False))
    return Derham(grid, options, domain=domain), domain


def test_nodal_bases():
    from struphy.feec.trig_space import cardinal_values, histopolant_values

    n = 9
    nodes = np.arange(n) / n
    assert np.allclose(cardinal_values(nodes, n)[0], np.eye(n))
    u, w = np.polynomial.legendre.leggauss(20)
    integrals = np.array([(w / (2 * n)) @ histopolant_values((j + 0.5 + 0.5 * u) / n, n)[0] for j in range(n)])
    assert np.allclose(integrals, np.eye(n), atol=1e-13)
    x = np.linspace(0, 1, 37)
    dL, H = cardinal_values(x, n, 1)[1], histopolant_values(x, n)[0]
    assert np.allclose(dL, np.roll(H, 1, axis=1) - H)
    assert np.allclose(cardinal_values(x, n)[0].sum(axis=1), 1.0)


def test_masses_equal_a_dense_assembly():
    """Mass matrices assembled by Struphy's kernels (all functions active on every cell, duplicate stencil offsets)
    equal a dense assembly with the 1D bases."""
    from struphy.feec.mass import WeightedMassOperators
    from struphy.polar.homotopy.derham_spaces import basis_values_1d

    derham, domain = _derham(n=7, num_elements=(4, None, 1))
    mass = WeightedMassOperators(derham, domain)
    M0 = mass.M0.matrix.tosparse().toarray()
    # dense: int B_i B_j det(DF) on Gauss points of the cells
    V = derham.V0fem.spaces
    x1, w1 = _gauss_on(V[0].breaks, 6)
    x2, w2 = _gauss_on(V[1].breaks, 12)
    B1, B2 = basis_values_1d(V[0], x1), basis_values_1d(V[1], x2)
    det = np.asarray(domain.jacobian_det(x1, x2, np.array([0.5])))[:, :, 0]
    weights = np.outer(w1, w2) * det
    dense = np.einsum("ab,ai,bj,ak,bl->ijkl", weights, B1, B2, B1, B2).reshape(B1.shape[1] * B2.shape[1], -1)
    assert np.allclose(M0, dense, atol=1e-12 * np.abs(dense).max())
    assert np.allclose(mass.M0.matrix.transpose().tosparse().toarray(), dense.T, atol=1e-12 * np.abs(dense).max())


def _gauss_on(breaks, q):
    u, w = np.polynomial.legendre.leggauss(q)
    a, b = breaks[:-1, None], breaks[1:, None]
    return ((a + b) / 2 + (b - a) / 2 * u).ravel(), ((b - a) / 2 * w).ravel()


def test_projections_derivatives_and_evaluation():
    derham, domain = _derham()

    def f(a, b, c):
        return (1 + a**2) * np.cos(2 * np.pi * b) + 0.3 * a * np.sin(4 * np.pi * b) + 0 * c

    def grad_f(a, b, c):
        return (
            2 * a * np.cos(2 * np.pi * b) + 0.3 * np.sin(4 * np.pi * b) + 0 * c,
            -2 * np.pi * (1 + a**2) * np.sin(2 * np.pi * b) + 1.2 * np.pi * a * np.cos(4 * np.pi * b) + 0 * c,
            0 * a,
        )

    u = derham.P0(f)
    g = derham.P1([lambda a, b, c, i=i: grad_f(a, b, c)[i] for i in range(3)])
    r = derham.grad.dot(u) - g
    assert np.sqrt(r.inner(r) / g.inner(g)) < 1e-7
    # trigonometric polynomials of degree <= 4 in eta2 are interpolated exactly along eta2
    F = derham.create_spline_function("u", "H1")
    F.vector = u
    e1, e2 = np.array(derham.V0fem.spaces[0].greville), np.linspace(0, 1, 13)
    values = np.asarray(F(e1, e2, np.array([0.5])))[:, :, 0]
    assert np.allclose(values, f(*np.meshgrid(e1, e2, indexing="ij"), 0), atol=1e-12)
    markers = np.column_stack([np.full(13, e1[2]), e2, np.full(13, 0.5)])
    assert np.allclose(np.asarray(F(markers)), values[2])


def test_polar_splines_with_trigonometric_functions_equal_the_reference():
    """The polar complex on the Derham's trigonometric spaces spans the spaces of the reference construction
    (PolarFEEC.jl's modal trigonometric spaces) in every degree. (The polar duals are pseudo-inverses in coefficient
    space, so the commuting projections depend on the angular coordinates for r >= 2.)"""
    from struphy.polar.homotopy import PolarComplex2D
    from struphy.polar.homotopy.demos import common
    from struphy.polar.tests.reference_core import ClampedBSplineSpace, TrigonometricSpace

    e1, e2 = np.linspace(0.02, 1, 23), np.linspace(0, 1, 41)
    for r in (1, 2, 3):
        struphy = common.polar_complex(r, 5, (6, 13), domain=common.star_disk(), angular="trig")
        reference = PolarComplex2D(
            ClampedBSplineSpace(5, 6), TrigonometricSpace(6), common.cross_section(common.star_disk()), r
        )
        assert [struphy.dimension(k) for k in range(3)] == [reference.dimension(k) for k in range(3)]
        for k in range(3):
            values = []
            for c in (struphy, reference):
                E = c.extraction(k).toarray()
                columns = [c.spaces[k].evaluate(E[:, j], e1, e2) for j in range(E.shape[1])]
                values.append(np.array([np.ravel(v) for v in columns]))
            A, B = values
            ranks = [np.linalg.matrix_rank(M, tol=1e-8 * np.abs(M).max()) for M in (A, B, np.vstack([A, B]))]
            assert ranks == [A.shape[0]] * 3, (r, k, ranks)
        assert struphy.cohomology_dimensions() == [1, 0, 0]


def test_trigonometric_eta2_limits():
    derham, _ = _derham()
    with pytest.raises(NotImplementedError, match="particles"):
        derham.args_derham
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    with pytest.raises(AssertionError, match="odd"):
        Derham(
            TensorProductGrid(num_elements=(4, 8, 1), mpi_dims_mask=(True, False, False)),
            DerhamOptions(degree=(2, 2, 1), bcs=(("free", "free"), None, None), angular_basis="trig"),
        )
