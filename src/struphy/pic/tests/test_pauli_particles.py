import pytest


def _particles(cls, domain, equil):
    import numpy as np

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, maxwellians
    from struphy.initial.base import GenericPerturbation

    def pert(*etas):
        # called with (eta1, eta2, eta3) or with one (3, n) array of positions
        e = etas[0] if len(etas) == 1 else etas
        return 0.1 * np.sin(2 * np.pi * e[1])

    background = maxwellians.Maxwellian3D(n=(1.0, None))
    initial = maxwellians.Maxwellian3D(n=(1.0, GenericPerturbation(pert)))
    particles = cls(
        loading_params=LoadingParameters(Np=2000, seed=1234),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        domain=domain,
        equil=equil,
        background=background,
        initial_condition=initial,
    )
    particles.draw_markers(sort=False)
    particles.initialize_weights()
    return particles


def _configurations():
    from struphy import domains, equils

    # tilted uniform field
    domain = domains.Cuboid(r1=2.0, r2=3.0, r3=4.0)
    equil = equils.HomogenSlab(B0x=0.3, B0y=-0.4, B0z=1.0)
    equil.domain = domain
    yield domain, equil

    # screw pinch: |B0| and b0 depend on the position
    domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=10.0)
    equil = equils.ScrewPinch(a=1.0, R0=10.0 / (2 * 3.141592653589793), B0=1.0)
    equil.domain = domain
    yield domain, equil


@pytest.mark.mpi_skip
@pytest.mark.parametrize("config", [0, 1])
def test_pauli_particles_slow_manifold(config):
    """PauliParticles6D draws the same markers as Particles6D, then sets mu = |v_perp|^2 / (2|B0|) and v = v_par b0;
    the sampling densities and the (control-variate) weights are those of Particles6D."""
    import numpy as np

    from struphy.pic.particles import Particles6D, PauliParticles6D

    domain, equil = list(_configurations())[config]
    ref = _particles(Particles6D, domain, equil)
    pauli = _particles(PauliParticles6D, domain, equil)
    assert pauli.mu_idx == pauli.first_diagnostics_idx
    assert pauli.n_cols == ref.n_cols + 1

    b, absB = pauli._unit_b_and_absB(ref.positions)
    v_ref = ref.velocities
    v_par = np.sum(v_ref * b, axis=0)
    v_perp2 = np.sum(v_ref**2, axis=0) - v_par**2

    assert np.allclose(pauli.positions, ref.positions, rtol=0.0, atol=0.0)
    assert np.allclose(pauli.velocities, v_par * b, rtol=1e-12, atol=1e-14)
    assert np.allclose(pauli.magnetic_moments, v_perp2 / (2 * absB), rtol=1e-12, atol=1e-14)
    assert np.allclose(pauli.mu_prime(), 0.0, atol=1e-14)

    # the gyrotropic velocity of f_coords has the drawn speed and parallel velocity
    v_f = pauli.f_coords[3:6]
    assert np.allclose(np.sum(v_f**2, axis=0), np.sum(v_ref**2, axis=0), rtol=1e-12)
    assert np.allclose(np.sum(v_f * b, axis=0), v_par, rtol=1e-12, atol=1e-14)

    assert np.allclose(pauli.sampling_density_values, ref.sampling_density_values, rtol=1e-14)
    assert np.allclose(pauli.weights0, ref.weights0, rtol=1e-14)
    assert np.allclose(pauli.weights, ref.weights, rtol=1e-10, atol=1e-16)


@pytest.mark.mpi_skip
def test_pauli_particles_need_gyrotropic_background():
    """Anisotropic thermal velocities are rejected."""
    from struphy import BoundaryParameters, LoadingParameters, domains, equils, maxwellians
    from struphy.pic.particles import PauliParticles6D

    domain = domains.Cuboid()
    equil = equils.HomogenSlab()
    equil.domain = domain
    with pytest.raises(AssertionError):
        PauliParticles6D(
            loading_params=LoadingParameters(Np=100, seed=1234),
            boundary_params=BoundaryParameters(),
            domain=domain,
            equil=equil,
            background=maxwellians.Maxwellian3D(vth1=(2.0, None)),
        )
