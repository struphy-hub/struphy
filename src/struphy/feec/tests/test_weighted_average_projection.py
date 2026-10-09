import pytest


@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder", "HollowTorus"])
@pytest.mark.parametrize("stab", ["M0", "M0ad_withT"])
def test_weighted_average_projection(mapping, stab):
    """S (I - P) is symmetric, P is a projection, and P is the plain average (AverageOperator) if the
    weight of S does not depend on the averaged directions (Cuboid, HollowCylinder averaged over eta3)."""

    from feectools.linalg.basic import IdentityOperator
    from maybempi import MPI

    from struphy import domains, equils
    from struphy.feec.mass import AverageOperator, WeightedAverageProjection, WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    if mapping == "Cuboid":
        domain, dirs = domains.Cuboid(), (1, 2)
    elif mapping == "HollowCylinder":
        domain, dirs = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=4.0), (2,)
    else:
        domain, dirs = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=1), (1, 2)
    equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0)
    equil.domain = domain

    derham = Derham(
        TensorProductGrid(num_elements=[8, 8, 6]),
        DerhamOptions(degree=[2, 2, 2], bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
    )
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)
    S0 = getattr(mass_ops, stab)
    P = WeightedAverageProjection(derham, S0, dirs)
    assert P.directions == dirs
    S = S0 @ (IdentityOperator(derham.V0) - P)

    bc = derham.boundary_ops["0"]
    x = bc.dot(create_equal_random_arrays(derham.V0fem, seed=1, flattened=True)[1])
    y = bc.dot(create_equal_random_arrays(derham.V0fem, seed=2, flattened=True)[1])

    xSy, ySx = float(x.inner(S.dot(y))), float(y.inner(S.dot(x)))
    assert abs(xSy - ySx) < 1e-12 * abs(xSy)

    Px = P.dot(x)
    dP = P.dot(Px) - Px
    assert float(dP.inner(dP)) ** 0.5 < 1e-12 * float(Px.inner(Px)) ** 0.5

    if mapping != "HollowTorus":
        avg = AverageOperator(derham, "H1", dirs[0])
        for d in dirs[1:]:
            avg = avg @ AverageOperator(derham, "H1", d)
        diff = avg.dot(x) - Px
        assert float(diff.inner(diff)) ** 0.5 < 1e-12 * float(Px.inner(Px)) ** 0.5
