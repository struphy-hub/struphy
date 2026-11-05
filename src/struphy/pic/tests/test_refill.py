"""Refilling of markers leaving a hollow torus through its walls (``bc="refill"``, or ``"remove"`` with
``bc_refill``)."""

import math

import pytest

pytestmark = pytest.mark.mpi_pic


def _make_particles(boundary_params, tor_period, sfl=False):
    """Particles5D in a hollow torus (1/tor_period of the torus) with the ad hoc tokamak equilibrium."""
    from maybempi import MPI

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background import equils
    from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
    from struphy.io.options import DerhamOptions
    from struphy.kinetic_background import maxwellians
    from struphy.pic.particles import Particles5D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    domain = domains.HollowTorus(a1=0.1, a2=0.9, R0=3.0, tor_period=tor_period, sfl=sfl)
    equil = equils.AdhocTorus(a=1.0, R0=3.0, B0=1.0, q0=1.2, q1=2.0)
    equil.domain = domain

    grid = TensorProductGrid(num_elements=[8, 8, 4])
    derham = Derham(grid, DerhamOptions(degree=[1, 1, 1], bcs=[("free", "free"), None, None]), comm=comm, domain=domain)
    projected_equil = ProjectedMHDequilibrium(equil, derham)

    particles = Particles5D(
        comm_world=comm if comm.Get_size() > 1 else None,
        domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
        loading_params=LoadingParameters(ppc=4, seed=1234, moments=(0.0, 0.0, 1.0, 1.0)),
        boundary_params=boundary_params,
        domain=domain,
        equil=equil,
        projected_equil=projected_equil,
        background=maxwellians.GyroMaxwellian2D(n=(1.0, None)),
    )
    particles.draw_markers()
    return particles, equil, domain


def _field_line_shift(equil, domain, r, eta2):
    """Shift of eta3 along the field line from eta2 to 1 - eta2 at the radius r, by integrating its pitch. With
    sfl=True, eta2 is the straight field line angle, with sfl=False the geometric angle."""
    from scipy.integrate import quad

    q = float(equil.q_r(r))
    tor_period = domain.params["tor_period"]
    if domain.params["sfl"]:
        return -2 * tor_period * q * eta2

    R0 = equil.params["R0"]
    pitch = lambda theta: q * (1 - (r / R0) ** 2) ** 0.5 * R0 / (R0 + r * math.cos(theta))
    theta = 2 * math.pi * eta2
    return tor_period * quad(pitch, theta, -theta)[0] / (2 * math.pi)


@pytest.mark.parametrize("sfl", [False, True])
@pytest.mark.parametrize("tor_period", [1, 6])
@pytest.mark.parametrize("bc", ["refill", "remove_with_bc_refill"])
def test_refill_on_flux_surface(bc, tor_period, sfl):
    """A marker leaving through a wall is put back just inside that wall at the up-down mirrored poloidal angle
    (eta2 -> 1 - eta2), shifted along the field line, with its velocity unchanged; no marker is lost."""
    import cunumpy as xp

    from struphy import BoundaryParameters

    if bc == "refill":
        boundary_params = BoundaryParameters(bc=("refill", "periodic", "periodic"))
    else:
        boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"), bc_refill=("inner", "outer"))
    particles, equil, domain = _make_particles(boundary_params, tor_period, sfl=sfl)
    assert particles.bc_refill == ("inner", "outer")

    valid = xp.nonzero(particles.valid_mks)[0]
    assert len(valid) >= 3
    n_valid = len(valid)
    inner, outer, inside = valid[:3]

    # the marker outside the outer wall, the one outside the inner wall, and one inside (unchanged)
    eta = {inner: (-0.01, 0.2, 0.5), outer: (1.01, 0.3, 0.4), inside: (0.5, 0.3, 0.4)}
    for ip, (e1, e2, e3) in eta.items():
        particles.markers[:3, ip] = (e1, e2, e3)
    velocities = particles.markers[3:5, [inner, outer, inside]].copy()

    particles.apply_kinetic_bc()

    assert particles.n_lost_markers == 0
    assert xp.count_nonzero(particles.valid_mks) == n_valid
    assert xp.allclose(particles.markers[3:5, [inner, outer, inside]], velocities)

    a1, a2 = domain.params["a1"], domain.params["a2"]
    for ip, eta1_new, r_wall in ((inner, 1e-4, a1), (outer, 1 - 1e-4, a2)):
        _, e2, e3 = eta[ip]
        assert particles.markers[0, ip] == pytest.approx(eta1_new)
        assert particles.markers[1, ip] == pytest.approx(1.0 - e2)
        eta3_new = (e3 + _field_line_shift(equil, domain, r_wall, e2)) % 1.0
        assert particles.markers[2, ip] == pytest.approx(eta3_new)
    assert xp.allclose(particles.markers[:3, inside], eta[inside])


def test_refill_only_for_eta1():
    """The refill boundary condition is for the radial direction only."""
    from struphy import BoundaryParameters

    with pytest.raises(AssertionError):
        _make_particles(BoundaryParameters(bc=("periodic", "refill", "periodic")), 1)
