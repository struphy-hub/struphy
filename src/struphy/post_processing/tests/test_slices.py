"""The logical surfaces of the physical planes in struphy.post_processing.slices."""

import numpy as np
import pytest

from struphy import domains
from struphy.post_processing.slices import physical_planes

DOMAINS = [
    domains.Cuboid(r1=2.0, l2=-1.0, r2=1.0, r3=4.0),
    domains.HollowCylinder(a1=0.1, a2=1.0, Lz=4.0),
    domains.IGAPolarCylinder(num_elements=(8, 9), degree=(3, 3), a=1.0, Lz=4.0),
    domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=1),
    domains.IGAPolarTorus(num_elements=(8, 9), degree=(3, 3), a=1.0, R0=3.0, tor_period=1),
]


def surface(domain, plane, value, n=7):
    """Cartesian coordinates of the logical surface ``plane.cut = value``."""
    etas = {e: np.linspace(0.0, 1.0, n) for e in ("eta1", "eta2", "eta3")}
    etas[plane.cut] = np.array([value])
    x, y, z = domain(etas["eta1"], etas["eta2"], etas["eta3"])
    return {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z)}


@pytest.mark.mpi_skip
@pytest.mark.parametrize("domain", DOMAINS, ids=lambda d: type(d).__name__)
@pytest.mark.parametrize("phi", [np.pi, 0.5])
def test_planes_lie_in_their_physical_plane(domain, phi):
    planes = physical_planes(domain, phi=phi)
    # the spline mappings only approximate the poloidal angle 2 pi eta2
    tol = 1e-2 if type(domain).__name__.startswith("IGAPolar") else 1e-12
    for name, plane in planes.items():
        for value in plane.values:
            xyz = surface(domain, plane, value)
            if name == "rz":
                # the half plane at the angle phi: (x, y) points in the direction (cos phi, sin phi)
                normal = -np.sin(phi) * xyz["x"] + np.cos(phi) * xyz["y"]
                along = np.cos(phi) * xyz["x"] + np.sin(phi) * xyz["y"]
                assert np.max(np.abs(normal)) < tol * np.max(np.abs(along))
                assert np.all(along >= -tol)
            else:
                # the coordinate that is not on an axis of the plot is constant
                (const,) = {"x", "y", "z"} - {plane.hlabel, plane.vlabel}
                spread = np.ptp(xyz[const])
                assert spread < tol * max(np.ptp(xyz[c]) for c in "xyz")


@pytest.mark.mpi_skip
def test_cuboid_has_no_poloidal_plane():
    assert set(physical_planes(DOMAINS[0])) == {"xy", "xz", "yz"}


@pytest.mark.mpi_skip
def test_unknown_domain_raises():
    with pytest.raises(NotImplementedError):
        physical_planes(domains.Colella())


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "domain",
    [domains.HollowTorus(tor_period=3), domains.HollowCylinder(poc=2), domains.HollowTorus(tor_period=1, pol_period=2)],
    ids=["tor_period", "poc", "pol_period"],
)
def test_partial_angles_raise(domain):
    with pytest.raises(ValueError):
        physical_planes(domain)
