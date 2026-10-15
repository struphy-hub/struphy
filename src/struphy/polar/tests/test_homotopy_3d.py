"""Tests of the 3D polar complex (total complex of the 2D polar complex and the toroidal complex), ported from
``test/test_polar3d.jl`` and ``test/test_complexes.jl`` of PolarFEEC.jl."""

import numpy as np
import pytest

from struphy.polar.homotopy import (
    ClampedBSplineSpace,
    DiskMapping,
    PeriodicBSplineSpace,
    PolarComplex2D,
    TrigonometricSpace,
)
from struphy.polar.homotopy.complex3d import PeriodicComplex, PolarComplex3D
from struphy.polar.homotopy.linalg import as_dense

SECTION = DiskMapping(radius=lambda r, t: 0.5 * r * (1 + 0.1 * r * np.cos(4 * np.pi * t)))


@pytest.mark.parametrize("space", [PeriodicBSplineSpace(3, 8), TrigonometricSpace(3)])
def test_periodic_complex(space):
    c = PeriodicComplex(space)
    d, h = c.derivative(0).toarray(), as_dense(c.homotopy(1))
    P0, P1 = as_dense(c.cohomology_projection(0)), as_dense(c.cohomology_projection(1))
    n = space.n
    assert np.allclose(h @ d, np.eye(n) - P0, atol=1e-12)
    assert np.allclose(d @ h, np.eye(n) - P1, atol=1e-12)
    assert np.allclose(h @ P1, 0, atol=1e-12) and np.allclose(P0 @ h, 0, atol=1e-12)


@pytest.mark.parametrize("boundary, cohomology", [("free", [1, 1, 0, 0]), ("dirichlet", [0, 0, 1, 1])])
@pytest.mark.parametrize("toroidal", [PeriodicBSplineSpace(3, 8), TrigonometricSpace(3)])
def test_polar_complex_3d(boundary, cohomology, toroidal):
    c2 = PolarComplex2D(ClampedBSplineSpace(3, 3), PeriodicBSplineSpace(3, 8), SECTION, regularity=2, boundary=boundary)
    c3 = PolarComplex3D(c2, toroidal)
    nz = toroidal.n
    for m in range(4):
        expected = sum(c2.dimension(p) * nz for p, q in c3.bidegree_blocks(m))
        assert c3.dimension(m) == expected
        E, F = c3.extraction(m), c3.duals(m)
        assert np.allclose((F.T @ E).toarray(), np.eye(c3.dimension(m)), atol=1e-12)
    assert c3.cohomology_dimensions() == cohomology
    for m in range(2):
        assert np.max(np.abs(as_dense(c3.derivative(m + 1) @ c3.derivative(m)))) < 1e-13
    for m in range(4):
        n = c3.dimension(m)
        contraction = np.zeros((n, n))
        if m < 3:
            contraction += as_dense(c3.homotopy(m + 1) @ c3.derivative(m))
        if m > 0:
            contraction += as_dense(c3.derivative(m - 1) @ c3.homotopy(m))
        assert np.allclose(contraction, np.eye(n) - as_dense(c3.cohomology_projection(m)), atol=1e-11)
