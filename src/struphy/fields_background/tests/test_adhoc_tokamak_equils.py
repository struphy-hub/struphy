import cunumpy as xp
import pytest

from struphy.fields_background.equils import AdhocTorus, AdhocTorusQPsi, CircularTokamak


@pytest.mark.parametrize("equil_class", [AdhocTorus, AdhocTorusQPsi, CircularTokamak])
def test_psi_mixed_derivative(equil_class):
    """Mixed derivative d^2 psi/(dR dZ) against finite differences of d psi/dR."""

    equil = equil_class()

    r = xp.linspace(0.1, 0.9, 5) * equil.params["a"]
    th = xp.linspace(0.1, 2 * xp.pi - 0.1, 7)
    rr, tt = xp.meshgrid(r, th, indexing="ij")
    R = equil.params["R0"] + rr * xp.cos(tt)
    Z = rr * xp.sin(tt)

    h = 1e-5
    fd = (equil.psi(R, Z + h, dR=1) - equil.psi(R, Z - h, dR=1)) / (2 * h)

    assert xp.allclose(equil.psi(R, Z, dR=1, dZ=1), fd, rtol=1e-4, atol=1e-8)


def test_adhoc_torus_pressure_q_kind_2():
    """Cylindrical-limit pressure (p_kind=0) for the q_kind=2 profile; for l=0 it equals the q_kind=0 one."""

    r = xp.linspace(0.0, 1.0, 11)

    p2 = AdhocTorus(p_kind=0, q_kind=2, l=0.0).p_r(r)
    p0 = AdhocTorus(p_kind=0, q_kind=0).p_r(r)

    assert xp.allclose(p2, p0, atol=1e-8)


def test_circular_tokamak_psi_range():
    """psi_range must match psi() on the magnetic axis and at the plasma boundary."""

    equil = CircularTokamak()

    R0, a = equil.params["R0"], equil.params["a"]

    assert xp.isclose(equil.psi_range[0], equil.psi(*equil.psi_axis_RZ))
    assert xp.isclose(equil.psi_range[1], equil.psi(R0 + a, 0.0))
    assert xp.isclose(equil.psi_range[1], equil.psi(R0, a))
