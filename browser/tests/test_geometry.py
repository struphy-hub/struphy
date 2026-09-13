"""Numerical regression tests shared by native Python and browser Pyodide."""

import io
import json
import sys

import numpy as np
import pytest

from struphy_geometry import (
    CircularFlux,
    EQDSKFlux,
    Spline,
    domains,
    interp_mapping,
    load_geometry,
    sample_surface,
    save_geometry,
)

ANALYTIC = [
    "Cuboid",
    "Orthogonal",
    "Colella",
    "HollowCylinder",
    "PoweredEllipticCylinder",
    "HollowTorus",
    "ShafranovShiftCylinder",
    "ShafranovSqrtCylinder",
    "ShafranovDshapedCylinder",
]
MAPPINGS = ANALYTIC + ["IGAPolarCylinder", "IGAPolarTorus"]


@pytest.mark.parametrize("name", MAPPINGS)
def test_derivatives_and_shapes(name):
    d = getattr(domains, name)()
    q = np.array([0.37, 0.23, 0.41])
    J = d.jacobian(*q, squeeze_out=True)
    h = 1e-6
    fd = np.column_stack(
        [
            (d(*(q + np.eye(3)[i] * h), squeeze_out=True) - d(*(q - np.eye(3)[i] * h), squeeze_out=True)) / (2 * h)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(J, fd, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(d.metric(*q, squeeze_out=True), J.T @ J, atol=1e-10)
    np.testing.assert_allclose(d.jacobian_inv(*q, squeeze_out=True) @ J, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(d.jacobian_det(*q, squeeze_out=True), np.linalg.det(J), rtol=1e-10)
    np.testing.assert_allclose(
        d.metric_inv(*q, squeeze_out=True) @ d.metric(*q, squeeze_out=True), np.eye(3), atol=1e-10
    )
    assert d(np.array([q, q])).shape == (3, 2)
    assert d(np.array([0.2, 0.4]), np.array([0.2, 0.4]), np.array([0.2, 0.4])).shape == (3, 2, 2, 2)


@pytest.mark.parametrize("name", MAPPINGS)
def test_archive_roundtrip(name):
    d = getattr(domains, name)()
    restored = load_geometry(save_geometry(d))
    q = np.random.default_rng(17).uniform(0.1, 0.9, (8, 3))
    np.testing.assert_allclose(restored(q), d(q), atol=1e-12)
    np.testing.assert_allclose(restored.jacobian(q), d.jacobian(q), atol=1e-12)
    if name in ANALYTIC:
        np.testing.assert_allclose(d.from_dict(d.to_dict())(q), d(q))


@pytest.mark.parametrize("kind", ["0", "1", "2", "3", "v"])
def test_pull_push(kind):
    d = domains.HollowTorus(tor_period=1)
    q = (0.31, 0.22, 0.43)
    a = 2.0 if kind in ("0", "3") else [1.0, 2.0, 3.0]
    pulled = d.pull(a, *q, kind=kind, squeeze_out=True)
    pushed = d.push(pulled, *q, kind=kind, squeeze_out=True)
    np.testing.assert_allclose(pushed, a, atol=1e-10)


@pytest.mark.parametrize(
    "forward,backward,a",
    [
        ("0_to_3", "3_to_0", 2.0),
        ("1_to_2", "2_to_1", [1.0, 2.0, 3.0]),
        ("v_to_1", "1_to_v", [1.0, 2.0, 3.0]),
        ("v_to_2", "2_to_v", [1.0, 2.0, 3.0]),
    ],
)
def test_transform(forward, backward, a):
    d = domains.IGAPolarTorus()
    q = (0.31, 0.22, 0.43)
    result = d.transform(a, *q, kind=forward, squeeze_out=True)
    np.testing.assert_allclose(d.transform(result, *q, kind=backward, squeeze_out=True), a, atol=1e-10)


def test_spline_3d_construction():
    counts, degree, periodic = (3, 4, 3), (2, 2, 2), (False, False, False)

    def component(i):
        def f(*etas):
            return np.meshgrid(*etas, indexing="ij")[i]

        return f

    coefficients = interp_mapping(counts, degree, periodic, *[component(i) for i in range(3)])
    d = Spline(counts, degree, periodic, *coefficients)
    q = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    np.testing.assert_allclose(d(q), q.T, atol=1e-12)
    np.testing.assert_allclose(load_geometry(save_geometry(d))(q), q.T, atol=1e-12)


@pytest.mark.parametrize("mode", ["equal_angle", "equal_arc_length", "sfl", "equal_area", "equal_volume"])
def test_flux_tracing(mode):
    flux = CircularFlux()
    d = domains.Tokamak(
        equilibrium=flux,
        num_elements=(4, 12),
        degree=(2, 3),
        xi_param=mode,
        num_elements_pre=(6, 24),
        p_pre=(3, 3),
        psi_power=0.5,
        psi_shifts=(4.0, 10.0),
        tor_period=1,
    )
    q = np.array([[0.2, 0.13, 0.2], [0.7, 0.73, 0.4]])
    xyz = d(q)
    expected = 0.04 + q[:, 0] ** 2 * (0.9 - 0.04)
    np.testing.assert_allclose(flux.psi(np.hypot(xyz[0], xyz[1]), xyz[2]), expected, rtol=0.04, atol=0.002)
    restored = load_geometry(save_geometry(d))
    np.testing.assert_allclose(restored(q), xyz, atol=1e-12)
    np.testing.assert_allclose(restored.jacobian(q), d.jacobian(q), atol=1e-12)


def eqdsk_text(sign=1):
    n = 9
    R, Z = np.linspace(1.0, 5.0, n), np.linspace(-2.0, 2.0, n)
    flux = sign * ((R[None, :] - 3.0) ** 2 + Z[:, None] ** 2)
    header = [4.0, 4.0, 3.0, 1.0, 0.0, 3.0, 0.0, 0.0, sign, 1.0, *([0.0] * 10)]
    values = [*header, *([0.0] * (4 * n)), *flux.ravel(), *([0.0] * n)]
    return f"Synthetic circular flux 0 {n} {n}\n" + "\n".join(
        "".join(f"{x:16.9E}".replace("E", "D") for x in values[i : i + 5]) for i in range(0, len(values), 5)
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_eqdsk(sign):
    flux = EQDSKFlux.from_text(eqdsk_text(sign))
    np.testing.assert_allclose(flux.psi_axis_RZ, [3.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(flux.psi([3.2, 3.4], 0.3), sign * np.array([0.13, 0.25]), atol=1e-10)
    np.testing.assert_allclose(flux.psi(3.2, 0.3, dR=1), sign * 0.4, atol=1e-10)
    d = domains.Tokamak(equilibrium=flux, num_elements=(4, 12))
    assert np.isfinite(d(0.5, 0.3, 0.1)).all()


def test_seams_poles_and_outside():
    d = domains.IGAPolarTorus(tor_period=1)
    np.testing.assert_allclose(d(0.5, 0.0, 0.3), d(0.5, 1.0, 0.3), atol=1e-12)
    np.testing.assert_allclose(d(0.5, 0.3, 0.0), d(0.5, 0.3, 1.0), atol=1e-12)
    np.testing.assert_allclose(d(0.0, 0.2, 0.3), d(0.0, 0.7, 0.3), atol=1e-12)
    q = np.array([[0.2, 0.3, 0.4], [-0.1, 0.3, 0.4], [1.1, 0.3, 0.4]])
    assert d(q.copy(), remove_outside=True).shape == (3, 1)
    outside = d(q.copy(), remove_outside=False)
    np.testing.assert_array_equal(outside[:, 1:], -1.0)
    assert sample_surface(d, 8)[0].shape == (8, 8)


def test_invalid_inputs_and_excluded_adapters():
    with pytest.raises(ValueError, match="header"):
        EQDSKFlux.from_text("bad")
    with pytest.raises(ValueError, match="Incomplete"):
        EQDSKFlux.from_text("bad 0 9 9\n")
    with pytest.raises(ValueError):
        CircularFlux(a=-1)
    with pytest.raises(ValueError, match="explicit"):
        Spline()
    with pytest.raises(ValueError, match="equilibrium"):
        domains.Tokamak()
    assert not hasattr(domains, "GVECunit")
    assert not hasattr(domains, "DESCunit")
    stream = io.BytesIO()
    np.savez(stream, metadata=json.dumps({"format": "struphy-geometry", "version": 99}))
    with pytest.raises(ValueError, match="version"):
        load_geometry(stream.getvalue())


def test_no_native_runtime_dependencies():
    forbidden = {"struphy", "pyccel", "feectools", "mpi4py", "cupy", "vtk", "pyvista", "h5py", "scope_profiler"}
    assert not forbidden.intersection(name.split(".")[0] for name in sys.modules)
