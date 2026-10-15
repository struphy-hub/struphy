"""Tests of the serial high-order polar FEEC core, ported from ``test/test_paper_theorems.jl``,
``test/test_boundary_homotopy.jl`` and ``test/test_curl_curl.jl`` of PolarFEEC.jl (same parameters and tolerances)."""

import numpy as np
import pytest
import scipy.sparse as sp

from struphy.polar.homotopy import (
    ClampedBSplineSpace,
    CoordinateTransformation2D,
    DiskMapping,
    PeriodicBSplineSpace,
    PolarComplex2D,
    TrigonometricSpace,
    monomial_basis,
    perturbed_disk,
    zernike_basis,
)
from struphy.polar.homotopy.linalg import as_dense
from struphy.polar.homotopy.mappings import complex_step
from struphy.polar.homotopy.solvers import curl_curl_free_lunch, range_projection

DISC = DiskMapping()


def gauss01(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return (t + 1) / 2, w / 2


def integrate(f, a, b, n=40):
    t, w = gauss01(n)
    return (b - a) * sum(wi * f(a + (b - a) * ti) for ti, wi in zip(t, w))


def range_coordinates(form, A):
    """Coordinates of the columns of ``A`` in the extraction, or ``None`` if not in its range."""
    A = as_dense(A)
    coordinates = form.duals.T @ A
    if np.max(np.abs(form.extraction @ coordinates - A)) <= 1e-10 * max(1.0, np.max(np.abs(A))):
        return coordinates
    return None


def continuous_zero_domain(c):
    """Coefficient vectors of continuous 0-forms (constant first ring)."""
    nx, ny = c.radial.n, c.angular.n
    centre = np.zeros(nx * ny)
    centre[::nx] = c.angular.constant_coefficients()
    others = [i + j * nx for j in range(ny) for i in range(1, nx)]
    return np.column_stack([centre, np.eye(nx * ny)[:, others]])


def poincare_gradient(A):
    t, w = gauss01(20)

    def h(x, y):
        return sum(wq * (A(tq * x, tq * y)[0] * x + A(tq * x, tq * y)[1] * y) for tq, wq in zip(t, w))

    return h


def poincare_curl(B):
    t, w = gauss01(20)

    def h(x, y):
        s = sum(wq * tq * B(tq * x, tq * y) for tq, wq in zip(t, w))
        return (-y * s, x * s)

    return h


# ------------------------------------------------------------------------------------------------- 1D lemmas
@pytest.mark.parametrize(
    "space",
    [
        ClampedBSplineSpace(1, 3),
        ClampedBSplineSpace(4, 6),
        PeriodicBSplineSpace(1, 5),
        PeriodicBSplineSpace(4, 8),
        PeriodicBSplineSpace(3, 6),
        TrigonometricSpace(3),
    ],
)
def test_unisolvence_of_1d_gdofs(space):
    for s in (space, space.scaled):
        assert np.linalg.matrix_rank(s.histopolation_matrix.toarray()) == s.n


def test_vanishing_physical_derivative():
    radial = ClampedBSplineSpace(4, 6)
    for i in range(1, 5):
        value = radial.basis_matrix(np.array([1e-6, 5e-7])).toarray()[:, i]
        assert np.log2(value[0] / value[1]) == pytest.approx(i, abs=1e-3)


def test_derivative_identities_1d():
    for space in (ClampedBSplineSpace(3, 6), PeriodicBSplineSpace(3, 6), TrigonometricSpace(3)):
        x = np.linspace(0.013, 0.987, 23)
        u = np.cos(np.arange(space.n) + 0.3)
        h = 1e-6
        derivative = (space.basis_matrix(x + h) @ u - space.basis_matrix(x - h) @ u) / (2 * h)
        assert np.allclose(space.scaled.basis_matrix(x) @ (space.derivative_matrix() @ u), derivative, atol=1e-6)


# ------------------------------------------------------------------------------------------------- polar subspaces
CASES = [(r, kind) for r in (1, 2, 3) for kind in ("trig", "bspline")]


def _angular(r, kind):
    return TrigonometricSpace(r + 1) if kind == "trig" else PeriodicBSplineSpace(3, 2 * r + 3)


@pytest.fixture(scope="module", params=CASES, ids=[f"r{r}-{k}" for r, k in CASES])
def polar(request):
    r, kind = request.param
    return PolarComplex2D(ClampedBSplineSpace(3, 7), _angular(r, kind), DISC, regularity=r)


def test_dimensions(polar):
    r = polar.regularity
    nx0, ny0, nx1, ny1 = polar._shapes
    n0, n1G, n1C, n2 = (r + 1) * (r + 2) // 2, (r + 3) * r // 2, (r - 1) * r // 2, (r - 1) * r // 2
    assert polar.central_counts() == (n0, n1G + n1C, n2)
    assert polar.dimension(0) == n0 + (nx0 - r - 1) * ny0
    assert polar.dimension(1) == n1G + n1C + (nx1 - r) * ny0 + (nx0 - r - 1) * ny1
    assert polar.dimension(2) == n2 + (nx1 - r) * ny1


def test_conformity_conditions(polar):
    nx0, ny0, nx1, ny1 = polar._shapes
    E1, E2 = polar.extraction(1).toarray(), polar.extraction(2).toarray()
    first = nx1 * ny0
    a_x_first = E1[0:first:nx1]
    a_y = E1[first:]
    assert np.max(np.abs(a_y[0::nx0])) == 0
    assert np.max(np.abs(a_y[1::nx0] - polar.angular.derivative_matrix() @ a_x_first)) <= 1e-13
    assert np.max(np.abs(E2[0::nx1])) == 0


def test_conforming_two_form_projector(polar):
    r = polar.regularity
    nx1, ny1 = polar._shapes[2:]
    Pt = polar.reduced_two_form_projector(r)
    assert (Pt @ Pt != Pt).nnz == 0
    image = Pt @ np.sin(np.arange(1, nx1 * ny1 + 1))
    assert np.all(image[[i + j * nx1 for j in range(ny1) for i in range(r)]] == 0)
    w = polar.two_form_integral_vector
    assert np.allclose(Pt.T @ w, w)


def test_cochain_and_chain_complex(polar):
    G, C = polar.spaces[0].derivative_matrix, polar.spaces[1].derivative_matrix
    E0, E1, E2 = (polar.extraction(k) for k in range(3))
    assert range_coordinates(polar[1], G @ E0) is not None
    assert range_coordinates(polar[2], C @ E1) is not None
    assert range_coordinates(polar[0], polar.gradient_homotopy_matrix @ E1) is not None
    assert range_coordinates(polar[1], polar.curl_homotopy_matrix @ E2) is not None
    assert np.max(np.abs(as_dense(polar.homotopy(1) @ polar.homotopy(2)))) <= 1e-12
    ranks = np.linalg.matrix_rank(as_dense(polar.derivative(0))) + np.linalg.matrix_rank(as_dense(polar.homotopy(2)))
    assert ranks == polar.dimension(1)


def test_cohomology(polar):
    assert polar.cohomology_dimensions() == [1, 0, 0]
    dirichlet = PolarComplex2D(polar.radial, polar.angular, DISC, regularity=polar.regularity, boundary="dirichlet")
    assert dirichlet.cohomology_dimensions() == [0, 0, 1]


def test_discrete_homotopy_operators(polar):
    for k in range(3):
        n = polar.dimension(k)
        contraction = np.zeros((n, n))
        if k < 2:
            contraction += as_dense(polar.homotopy(k + 1) @ polar.derivative(k))
        if k > 0:
            contraction += as_dense(polar.derivative(k - 1) @ polar.homotopy(k))
        assert np.allclose(contraction, np.eye(n) - as_dense(polar.cohomology_projection(k)), atol=1e-11)


def test_one_form_projection_fixes_external_inputs(polar):
    E1 = polar.extraction(1)
    central = polar[1].central_count
    external = [E1[:, c].nonzero()[0].min() for c in range(central, E1.shape[1])]
    P1 = polar[1].projector.toarray()
    assert np.allclose(P1[:, external], np.eye(E1.shape[0])[:, external], atol=1e-12)


def test_commuting_projection_matrices(polar):
    P = [polar[k].projector.toarray() for k in range(3)]
    G, C = polar.spaces[0].derivative_matrix.toarray(), polar.spaces[1].derivative_matrix.toarray()
    HG, HC = polar.gradient_homotopy_matrix.toarray(), polar.curl_homotopy_matrix.toarray()
    continuous = continuous_zero_domain(polar)
    assert np.allclose(P[1] @ G @ continuous, G @ P[0] @ continuous, atol=1e-11)
    assert np.allclose(C @ P[1], P[2] @ C, atol=1e-11)
    assert np.allclose(HG @ P[1], P[0] @ HG, atol=1e-11)
    assert np.allclose(HC @ P[2], P[1] @ HC, atol=1e-11)
    assert np.allclose(P[1], G @ P[0] @ HG + HC @ P[2] @ C, atol=1e-11)


def f0(x, y):
    return np.exp(x) * np.cos(y) + x * y**2


def grad_f0(x, y):
    return (np.exp(x) * np.cos(y) + y**2, -np.exp(x) * np.sin(y) + 2 * x * y)


def A1(x, y):
    return (np.sin(y) + x**2, x * y + np.cos(x))


def curl_A1(x, y):
    return y - np.sin(x) - np.cos(y)


def test_commuting_projections_on_fields(polar):
    G, C = polar.spaces[0].derivative_matrix, polar.spaces[1].derivative_matrix
    o = dict(additional_order=6)
    assert np.allclose(G @ polar[0](f0, **o), polar[1](grad_f0, **o), atol=1e-9)
    assert np.allclose(C @ polar[1](A1, **o), polar[2](curl_A1, **o), atol=1e-9)
    assert np.allclose(
        polar.gradient_homotopy_matrix @ polar[1](A1, **o), polar[0](poincare_gradient(A1), **o), atol=1e-8
    )
    assert np.allclose(
        polar.curl_homotopy_matrix @ polar[2](curl_A1, **o), polar[1](poincare_curl(curl_A1), **o), atol=1e-8
    )


def test_conservation_properties(polar):
    def B(x, y):
        return np.exp(x - y) + x**2

    projected = polar[2](B, additional_order=6)
    w = polar.two_form_integral_vector
    exact = (
        2
        * np.pi
        * integrate(
            lambda rho: rho * integrate(lambda t: B(rho * np.cos(2 * np.pi * t), rho * np.sin(2 * np.pi * t)), 0, 1),
            0,
            1,
        )
    )
    assert w @ projected == pytest.approx(exact, rel=1e-9)

    def A(x, y):
        return (-y * np.exp(x), x + y**3)

    nx0, ny0, nx1, ny1 = polar._shapes
    circulation = np.zeros(polar.spaces[1].n)
    first = nx1 * ny0
    integrals = polar.angular.scaled.coefficient_integrals
    for j in range(ny1):
        circulation[first + nx0 - 1 + j * nx0] = integrals[j]

    def tangential(t):
        a = A(np.cos(2 * np.pi * t), np.sin(2 * np.pi * t))
        return a[0] * (-2 * np.pi * np.sin(2 * np.pi * t)) + a[1] * 2 * np.pi * np.cos(2 * np.pi * t)

    boundary = integrate(tangential, 0, 1)
    assert circulation @ polar[1](A, additional_order=6) == pytest.approx(boundary, rel=1e-9)


def test_polynomial_reproduction_under_compatibility(polar):
    if not isinstance(polar.angular, TrigonometricSpace):
        pytest.skip("r-compatibility needs trigonometric angular functions")
    r = polar.regularity
    points = [(rho, t) for rho in (0.0, 1e-3, 0.05, 0.3, 0.9) for t in (0.0, 0.21, 0.66)]
    for T in zernike_basis(r):
        projected = polar[0](T)
        for rho, t in points:
            X, Y = DISC(rho, t)
            assert polar.spaces[0].evaluate(projected, rho, t)[0, 0] == pytest.approx(T(X, Y), abs=1e-11)
        assert np.allclose(projected, polar.spaces[0].commuting_projection(T), atol=1e-11)
    if r >= 2:
        for T in zernike_basis(r - 2):
            projected = polar[2](T)
            for rho, t in points:
                if rho > 0:
                    X, Y = DISC(rho, t)
                    assert polar.spaces[2].evaluate(projected, rho, t)[0, 0] == pytest.approx(T(X, Y), abs=1e-10)
    G = polar.spaces[0].derivative_matrix
    for T in zernike_basis(r):

        def gradient(x, y, T=T):
            return (complex_step(T, x, y, argnum=0), complex_step(T, x, y, argnum=1))

        assert np.allclose(polar[1](gradient), G @ polar[0](T), atol=1e-9)


# ------------------------------------------------------------------------------------------------- boundary
def test_boundary_compatible_homotopy_identities():
    polar = PolarComplex2D(
        ClampedBSplineSpace(3, 6), TrigonometricSpace(3), perturbed_disk(), regularity=3, boundary="dirichlet"
    )
    hG, hC, Bm = (as_dense(a) for a in (polar.homotopy(1), polar.homotopy(2), polar.cohomology_projection(2)))
    D0, D1 = polar.derivative(0).toarray(), polar.derivative(1).toarray()
    assert np.allclose(hG @ D0, np.eye(D0.shape[1]), atol=1e-10)
    assert np.allclose(D0 @ hG + hC @ D1, np.eye(D0.shape[0]), atol=1e-10)
    assert np.allclose(D1 @ hC, np.eye(D1.shape[0]) - Bm, atol=1e-11)
    assert np.max(np.abs(hG @ hC)) <= 1e-11
    M2 = polar.mass_matrix(2).toarray()
    assert np.allclose(Bm.T @ M2, M2 @ Bm, atol=1e-11)
    w = polar.extraction(2).T @ polar.two_form_integral_vector
    assert np.allclose(w @ Bm, w, atol=1e-12)


def generic_disc_map():
    return CoordinateTransformation2D(
        lambda r, t: (r * np.cos(2 * np.pi * t), r * np.sin(2 * np.pi * t)), has_polar_axis=True
    )


def scaled_residual(actual, expected):
    actual, expected = as_dense(actual), as_dense(expected)
    return np.max(np.abs(actual - expected)) / max(1.0, np.max(np.abs(expected)))


@pytest.mark.parametrize("regularity", [1, 4])
@pytest.mark.parametrize("radial_cells", [2, 4])
@pytest.mark.parametrize("geometry", ["disc", "perturbed disc"])
@pytest.mark.parametrize("angular_kind", ["trig", "bspline"])
def test_boundary_homotopy_regression(regularity, radial_cells, geometry, angular_kind):
    transformation = generic_disc_map() if geometry == "disc" else perturbed_disk()
    angular = (
        TrigonometricSpace(regularity)
        if angular_kind == "trig"
        else PeriodicBSplineSpace(regularity, 4 if regularity == 1 else 10)
    )
    radial = ClampedBSplineSpace(regularity, radial_cells)
    polar = PolarComplex2D(
        radial, angular, transformation, regularity, basis=monomial_basis(regularity), boundary="dirichlet"
    )
    D0, D1 = polar.derivative(0), polar.derivative(1)
    hG, hC, Bm = polar.homotopy(1), polar.homotopy(2), polar.cohomology_projection(2)
    I0, I1, I2 = (np.eye(polar.dimension(k)) for k in range(3))
    assert scaled_residual(hG @ D0, I0) < 8e-11
    assert scaled_residual(as_dense(D0) @ as_dense(hG) + as_dense(hC) @ as_dense(D1), I1) < 8e-11
    assert scaled_residual(as_dense(D1) @ as_dense(hC), I2 - as_dense(Bm)) < 3e-11
    norm_inf = lambda A: np.max(np.sum(np.abs(as_dense(A)), axis=1))  # noqa: E731
    assert norm_inf(as_dense(hG) @ as_dense(hC)) / max(1.0, norm_inf(hG) * norm_inf(hC)) < 1e-11
    full_curl = polar.spaces[1].derivative_matrix
    E1, E2 = polar.extraction(1), polar.extraction(2)
    assert scaled_residual(full_curl @ E1 @ as_dense(hC), E2 @ (I2 - as_dense(Bm))) < 3e-11
    assert np.all((polar.extraction(0) @ as_dense(hG))[polar.zero_form_dirichlet_rows()] == 0)
    assert np.all((E1 @ as_dense(hC))[polar.one_form_dirichlet_rows()] == 0)
    B = as_dense(Bm)
    assert scaled_residual(B @ B, B) < 3e-12
    integral = E2.T @ polar.two_form_integral_vector
    assert scaled_residual(integral @ B, integral) < 3e-12
    M2 = polar.mass_matrix(2).toarray()
    assert scaled_residual(B.T @ M2, M2 @ B) < 3e-11
    # the projected constant, computed independently through the L2 projection
    projected_constant = polar.l2_projection(2, lambda x, y: np.ones_like(x))
    constant_reduced = polar.duals(2).T @ projected_constant
    constant_integral = integral @ constant_reduced
    assert constant_integral > 0
    assert constant_integral == pytest.approx(constant_reduced @ M2 @ constant_reduced, abs=3e-11)
    unit = constant_reduced / constant_integral
    assert np.max(np.abs(B @ unit - unit)) < 3e-11
    # lifted fields have vanishing tangential trace on the boundary
    data = polar.duals(2).T @ np.kron(
        polar.angular.constant_one_form_coefficients(), np.eye(radial.scaled.n)[regularity]
    )
    mean_free = data - B @ data
    lifted = as_dense(hC) @ data
    assert np.max(np.abs(as_dense(D1) @ lifted - mean_free)) / max(1.0, np.max(np.abs(mean_free))) < 3e-11
    field = E1 @ lifted
    assert np.all(field[polar.one_form_dirichlet_rows()] == 0)
    for t in (0.0, 0.137, 0.41, 0.79):
        values = polar.spaces[1].logical.evaluate(field, [1.0], [t])
        tangent = transformation.jacobian_matrix(1.0, t)[:, 1]
        # logical 1-form proxy: A . dF/dy = (J^T A)_y
        assert abs(values[1][0, 0]) < 3e-11 * max(1.0, np.linalg.norm(tangent))
    potential = polar.extraction(0) @ (as_dense(hG) @ np.cos(np.arange(1, polar.dimension(1) + 1)))
    assert np.all(potential[polar.zero_form_dirichlet_rows()] == 0)


# ------------------------------------------------------------------------------------------------- 2D curl-curl
def test_curl_curl_free_lunch():
    """C1: one mass solve ``a = h_2 (I - B) M_2^{-1} l`` equals the direct range projection."""
    petals = DiskMapping(radius=lambda r, t: 0.3 * r * (1 + 0.2 * r * np.cos(6 * np.pi * t)))
    section = PolarComplex2D(
        ClampedBSplineSpace(3, 4), PeriodicBSplineSpace(3, 8), petals, regularity=2, boundary="dirichlet"
    )
    D = section.derivative(1)
    M2 = section.mass_matrix(2)
    load = M2 @ np.cos(np.arange(1, section.dimension(2) + 1))
    mean = as_dense(section.cohomology_projection(2))
    x = section.mass_solver(2).solve(load)
    a = as_dense(section.homotopy(2)) @ (x - mean @ x)
    b = range_projection(section, 1, load)
    assert np.linalg.norm(D @ a - b) <= 1e-10 * np.linalg.norm(b)
    assert np.linalg.norm(D.T @ (load - M2 @ (D @ a))) <= 1e-10 * np.linalg.norm(D.T @ load)
    x = np.sin(np.arange(1, section.dimension(2) + 1))
    assert np.linalg.norm(D @ (as_dense(section.homotopy(2)) @ x) - (x - mean @ x)) <= 1e-12 * np.linalg.norm(x)
    # the written-out algorithm of the paper, for a 1-form load
    load1 = section.mass_matrix(1) @ np.cos(np.arange(1, section.dimension(1) + 1))
    expected = range_projection(section, 1, section.homotopy(2).T @ load1)
    assert np.allclose(D @ curl_curl_free_lunch(section, load1), expected, atol=1e-10 * np.max(np.abs(expected)))


def test_mass_and_l2_projection_reproduce_extraction():
    c = PolarComplex2D(ClampedBSplineSpace(3, 8), TrigonometricSpace(3), perturbed_disk(), regularity=3)
    for k in range(3):
        E = c.extraction(k)
        assert np.allclose((c.duals(k).T @ E).toarray(), np.eye(E.shape[1]), atol=5e-12)
        P = c[k].projector
        assert abs((P @ P - P)).max() < 5e-12
        assert abs((P @ E - E)).max() < 5e-12
        M = c.mass_matrix(k).toarray()
        assert np.allclose(M, (E.T @ c.full_mass_matrix(k) @ E).toarray(), atol=1e-12)
    assert sp.issparse(c.derivative(0))
