"""The logical surfaces of the physical planes in struphy.post_processing.slices."""

import numpy as np
import pytest

from struphy import domains
from struphy.post_processing.slices import Plane, evaluate_plane, physical_planes, plot_plane, rotate

DOMAINS = [
    domains.Cuboid(r1=2.0, l2=-1.0, r2=1.0, r3=4.0),
    domains.HollowCylinder(a1=0.1, a2=1.0, Lz=4.0),
    domains.IGAPolarCylinder(num_elements=(8, 9), degree=(3, 3), a=1.0, Lz=4.0),
    domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=1),
    domains.IGAPolarTorus(num_elements=(8, 9), degree=(3, 3), a=1.0, R0=3.0, tor_period=1),
    domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=6),
    domains.IGAPolarTorus(num_elements=(8, 9), degree=(3, 3), a=1.0, R0=3.0, tor_period=3),
]


def surface(domain, plane, i, n=7):
    """Cartesian coordinates of the i-th logical surface of ``plane``, rotated into place."""
    etas = {e: np.linspace(0.0, 1.0, n) for e in ("eta1", "eta2", "eta3")}
    etas[plane.cut] = np.array([plane.values[i]])
    x, y, z = (np.asarray(c) for c in domain(etas["eta1"], etas["eta2"], etas["eta3"]))
    if plane.rotations is not None:
        x, y = rotate(x, y, plane.rotations[i])
    return {"x": x, "y": y, "z": z}


@pytest.mark.mpi_skip
@pytest.mark.parametrize("domain", DOMAINS, ids=lambda d: f"{type(d).__name__}-{d.params.get('tor_period', 1)}")
@pytest.mark.parametrize("phi", [np.pi, 0.5])
def test_planes_lie_in_their_physical_plane(domain, phi):
    planes = physical_planes(domain, phi=phi, all_planes=True)
    # the spline mappings only approximate the poloidal angle 2 pi eta2
    tol = 1e-2 if type(domain).__name__.startswith("IGAPolar") else 1e-12
    for name, plane in planes.items():
        for i in range(len(plane.values)):
            xyz = surface(domain, plane, i)
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
def test_tori_have_two_planes_by_default():
    torus = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=6)
    assert set(physical_planes(torus)) == {"xy", "rz"}
    assert set(physical_planes(torus, all_planes=True)) == {"xy", "xz", "yz", "rz"}
    # other domains keep all their planes
    assert set(physical_planes(DOMAINS[1])) == {"xy", "xz", "yz", "rz"}


@pytest.mark.mpi_skip
def test_unknown_domain_raises():
    with pytest.raises(NotImplementedError):
        physical_planes(domains.Colella())


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "domain",
    [domains.HollowCylinder(poc=2), domains.HollowTorus(tor_period=1, pol_period=2)],
    ids=["poc", "pol_period"],
)
def test_partial_poloidal_angles_raise(domain):
    with pytest.raises(ValueError):
        physical_planes(domain)


class FakeOutput:
    """Stands in for Output in evaluate_plane: a field that is a function of the Cartesian coordinates of
    ``domain``, given on the requested logical grid."""

    def __init__(self, domain, fun, form="v"):
        self.domain, self.fun, self.form = domain, fun, form

    def field_form(self, name):
        return self.form

    def evaluate(self, name, t=None, representation=None, **etas):
        import xarray as xr

        grids = [np.atleast_1d(np.asarray(etas[e], dtype=float)) for e in ("eta1", "eta2", "eta3")]
        x, y, z = (np.asarray(c) for c in self.domain(*grids))
        coords = {e: g for e, g in zip(("eta1", "eta2", "eta3"), grids)}
        dims = ("eta1", "eta2", "eta3")
        coords.update({c: (dims, v) for c, v in zip(("X", "Y", "Z"), (x, y, z))})
        values = self.fun(x, y, z)
        if isinstance(values, tuple):  # a vector field
            return xr.DataArray(np.stack(values), dims=("component",) + dims, coords=coords)
        return xr.DataArray(values, dims=dims, coords=coords)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name", ["xy", "xz", "yz", "rz"])
def test_torus_sector_gives_the_full_torus_planes(name):
    """A field with the toroidal period of the sector (here 6 phi) gives the same planes on a sixth of the torus as
    on the full torus: the sector planes are rotated into place, and the plane z = 0 is repeated around the axis."""

    def fun(x, y, z):
        return np.cos(6 * np.arctan2(y, x)) * (1 + z) + np.hypot(x, y)

    def exact(e1, e2, e3, x, y, z):
        return fun(x, y, z)

    for n in (1, 6):
        domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=n)
        plane = physical_planes(domain, phi=2.0, all_planes=True)[name]
        num, exact_values, coords = evaluate_plane(FakeOutput(domain, fun), domain, "f", 0, exact, plane, 25)
        # the values follow the rotated coordinates, so they equal the field at those points
        assert np.allclose(num, exact_values)
        if name == "xy":
            # the copies cover the whole angle around the axis
            angles = np.arctan2(coords["y"], coords["x"])
            assert np.ptp(angles) > 1.9 * np.pi
            # and continue each other along eta3: no jumps between neighbouring points
            steps = np.hypot(np.diff(coords["x"], axis=2), np.diff(coords["y"], axis=2))
            assert np.max(steps) < 2 * np.pi * 5.0 / 25 * 1.5
        else:
            # the same physical plane as on the full torus
            full_domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=1)
            full = physical_planes(full_domain, phi=2.0, all_planes=True)[name]
            full_coords = evaluate_plane(FakeOutput(full_domain, fun), full_domain, "f", 0, exact, full, 25)[2]
            assert len(plane.values) == len(full.values)
            for c in (plane.hlabel, plane.vlabel):
                assert np.allclose(np.sort(coords[c].ravel()), np.sort(full_coords[c].ravel()))


@pytest.mark.mpi_skip
def test_plane_without_exact_solution_has_one_panel():
    plane = Plane("x", "y", "z at the center", "eta3", (0.5,))
    grid = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), [0.5], indexing="ij")
    coords = {"x": grid[0], "y": grid[1], "z": grid[2]}
    fig = plot_plane(grid[0] + grid[1], None, coords, plane, "f")
    assert len(fig.axes) == 2  # one panel and its colorbar
    fig = plot_plane(grid[0] + grid[1], grid[0], coords, plane, "f")
    assert len(fig.axes) == 4


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name", ["xy", "xz", "yz", "rz"])
def test_cartesian_components_are_rotated_with_the_torus_sector(name, monkeypatch):
    """Cartesian components of a field that turns with the torus (here (-y, x, z)) are rotated with the sector:
    they equal the field at the rotated points. The push-forward is replaced by the identity, so that the fake
    output gives the Cartesian components directly."""

    def fun(x, y, z):
        return (-y, x, z)

    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=6)
    monkeypatch.setattr(type(domain), "push", lambda self, values, *grids, kind: values)
    plane = physical_planes(domain, phi=2.0, all_planes=True)[name]
    num, exact_values, coords = evaluate_plane(
        FakeOutput(domain, fun), domain, "f", 0, None, plane, 13, kind="physical", components=True
    )
    assert exact_values is None
    assert num.shape[0] == 3
    assert np.allclose(num, np.stack(fun(coords["x"], coords["y"], coords["z"])))


@pytest.mark.mpi_skip
def test_logical_components_are_repeated_around_the_torus():
    def fun(x, y, z):
        return (x, 2 * y, 3 * z)

    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=4.0, tor_period=6)
    plane = physical_planes(domain)["xy"]
    num, _, coords = evaluate_plane(FakeOutput(domain, fun), domain, "f", 0, None, plane, 13, kind="2", components=True)
    # the logical components do not change with the copies of the sector
    assert num.shape == (3,) + coords["x"].shape
    n = num.shape[-1] // 6
    assert np.allclose(num[..., :n], num[..., n : 2 * n])


@pytest.mark.mpi_skip
def test_component_figures(monkeypatch):
    from struphy.post_processing import slices

    def fun(x, y, z):
        return (x, y, z)

    domain = domains.Cuboid(r1=2.0, l2=-1.0, r2=1.0, r3=4.0)
    out = FakeOutput(domain, fun)
    figs = slices.plot_physical_slices(out, domain, (4, 4, 4), [("u", "U", "f", 0, None, "2")], components=True)
    assert set(figs) == {f"u{k}_slices_{plane}" for k in (1, 2, 3) for plane in ("xy", "xz", "yz")}
    assert "component 1 (2-form)" in figs["u1_slices_xy"].axes[0].get_title()
