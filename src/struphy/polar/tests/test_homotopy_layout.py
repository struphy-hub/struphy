"""The polar complex built from the 1D spaces of a Struphy Derham complex in Struphy's coefficient ordering equals the
reference construction (the ported PolarFEEC.jl spaces in PolarFEEC.jl's ordering).

The bases differ only by the ordering of the tensor-product coefficients (Struphy's ``i * ny + j`` with periodic
splines shifted by the degree, against ``i + j * nx``) and of the external polar basis functions, so the comparison
uses the full-space operators ``E F^T`` (commuting projections), ``E D F^T`` (derivatives) and ``E h F^T``
(homotopies)."""

import numpy as np
import pytest
import scipy.sparse as sp


def _permutation(shapes, shift):
    """Matrix mapping reference (PolarFEEC.jl ordering) to Struphy coefficients of stacked components."""
    blocks = []
    for nx, ny in shapes:
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        struphy = (i * ny + j).ravel()
        reference = (i + ((j - shift) % ny) * nx).ravel()
        blocks.append(sp.csr_matrix((np.ones(nx * ny), (struphy, reference)), shape=(nx * ny, nx * ny)))
    return sp.block_diag(blocks).tocsr()


def _dense(A):
    from struphy.polar.homotopy.linalg import as_dense

    return as_dense(A)


def _complexes(degree, cells, regularity, boundary, enrichment="cochain_chain"):
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.polar.homotopy import ClampedBSplineSpace, PeriodicBSplineSpace, PolarComplex2D, perturbed_disk
    from struphy.polar.homotopy.derham_spaces import derham_factors
    from struphy.polar.homotopy.layout import STRUPHY
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=(*cells, 1)),
        DerhamOptions(degree=(degree, degree, 1), bcs=(("free", "free"), None, None)),
    )
    radial, angular, _ = derham_factors(derham)
    mapping = perturbed_disk()
    kwargs = dict(boundary=boundary, enrichment=enrichment)
    struphy = PolarComplex2D(radial, angular, mapping, regularity, layout=STRUPHY, **kwargs)
    reference = PolarComplex2D(
        ClampedBSplineSpace(degree, cells[0]), PeriodicBSplineSpace(degree, cells[1]), mapping, regularity, **kwargs
    )
    return struphy, reference


@pytest.mark.parametrize("degree, cells", [(3, (6, 12)), (4, (5, 9))])
@pytest.mark.parametrize("regularity", [1, 2, 3])
@pytest.mark.parametrize("boundary", ["free", "dirichlet"])
def test_struphy_layout_equals_reference(degree, cells, regularity, boundary):
    struphy, reference = _complexes(degree, cells, regularity, boundary)
    nx0, ny0, nx1, ny1 = reference._shapes
    shapes = {0: [(nx0, ny0)], 1: [(nx1, ny0), (nx0, ny1)], 2: [(nx1, ny1)]}
    P = {k: _permutation(shapes[k], degree) for k in range(3)}
    assert [struphy.dimension(k) for k in range(3)] == [reference.dimension(k) for k in range(3)]
    assert struphy.central_counts() == reference.central_counts()

    def full(c, k_out, k_in, A):
        return _dense(c.extraction(k_out)) @ _dense(A) @ _dense(c.duals(k_in)).T

    for k in range(3):
        a = full(struphy, k, k, sp.identity(struphy.dimension(k)))
        b = P[k] @ full(reference, k, k, sp.identity(reference.dimension(k))) @ P[k].T
        assert np.max(np.abs(a - b)) < 1e-12 * max(1.0, np.max(np.abs(b))), f"projector of degree {k}"
    for k in range(2):
        a = full(struphy, k + 1, k, struphy.derivative(k))
        b = P[k + 1] @ full(reference, k + 1, k, reference.derivative(k)) @ P[k].T
        assert np.max(np.abs(a - b)) < 1e-11 * max(1.0, np.max(np.abs(b))), f"derivative of degree {k}"
    for k in (1, 2):
        a = full(struphy, k - 1, k, struphy.homotopy(k))
        b = P[k - 1] @ full(reference, k - 1, k, reference.homotopy(k)) @ P[k].T
        assert np.max(np.abs(a - b)) < 1e-10 * max(1.0, np.max(np.abs(b))), f"homotopy of degree {k}"
    assert struphy.cohomology_dimensions() == reference.cohomology_dimensions()


def test_derivatives_equal_the_derham_derivatives():
    """The 2D derivatives of the Struphy-ordered forms are the poloidal blocks of Derham's grad and curl."""
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.polar.homotopy.derham_spaces import derham_factors
    from struphy.polar.homotopy.forms2d import FormSpace2D
    from struphy.polar.homotopy.layout import STRUPHY
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=(6, 9, 1)),
        DerhamOptions(degree=(3, 2, 1), bcs=(("free", "free"), None, None)),
    )
    radial, angular, _ = derham_factors(derham)
    G = FormSpace2D(0, radial, angular, STRUPHY).derivative_matrix
    C = FormSpace2D(1, radial, angular, STRUPHY).derivative_matrix
    n3 = derham.H1_1d_serial[2].spaces[0].nbasis
    assert n3 == 1
    grad = derham.grad_tp.tosparse()
    curl = derham.curl_tp.tosparse()
    n0 = radial.n * angular.n
    n1x, n1y = radial.scaled.n * angular.n, radial.n * angular.scaled.n
    assert np.allclose(grad[: n1x + n1y, :n0].toarray(), G.toarray())
    n2z = radial.scaled.n * angular.scaled.n
    offset = curl.shape[0] - n2z
    assert np.allclose(curl[offset:, : n1x + n1y].toarray(), C.toarray())


def test_cross_sections():
    """The reference cross-sections of the polar splines: the domain at eta3 = 0 (one tensor-grid evaluation of the
    domain), the unit disk, and a warning for regularity >= 2 when the cross-sections change shape along eta3."""
    import warnings

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.io.options import DerhamOptions
    from struphy.polar.homotopy.derham_blocks import DomainCrossSection, UnitDiskCrossSection
    from struphy.topology.grids import TensorProductGrid

    disk = DomainCrossSection(domains.HollowCylinder(a1=0.0, a2=1.0, Lz=1.0))
    e1, e2 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 7), indexing="ij")
    X, Y = disk(e1, e2)
    X_ref, Y_ref = UnitDiskCrossSection()(e1, e2)
    assert np.allclose(X, X_ref) and np.allclose(Y, Y_ref)
    assert np.allclose(disk.jacobian_matrix(e1, e2), UnitDiskCrossSection().jacobian_matrix(e1, e2))
    # scattered points give the same values as the tensor grid
    Xs, Ys = disk(e1.ravel()[::-1], e2.ravel()[::-1])
    assert np.allclose(Xs, X.ravel()[::-1]) and np.allclose(Ys, Y.ravel()[::-1])

    def build(domain, regularity, **kwargs):
        options = DerhamOptions(
            degree=(3, 3, 2),
            bcs=(("free", "free"), None, None),
            polar_splines=True,
            polar_regularity=regularity,
            **kwargs,
        )
        return Derham(TensorProductGrid(num_elements=(6, 12, 4)), options, domain=domain)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(domains.HollowTorus(a1=0.0, a2=0.3, R0=1.0), 2)  # the same cross-section everywhere
        build(domains.HelicalTube("islands", num_elements=(6, 24, 32), degree=(3, 3, 3)), 1)
    with pytest.warns(UserWarning, match="cross-sections eta3 = const of the domain differ"):
        build(domains.HelicalTube("islands", num_elements=(6, 24, 32), degree=(3, 3, 3)), 2)
    unit = build(
        domains.HelicalTube("islands", num_elements=(6, 24, 32), degree=(3, 3, 3)), 2, polar_cross_section="unit_disk"
    )
    assert unit.ck_blocks.regularity == 2
