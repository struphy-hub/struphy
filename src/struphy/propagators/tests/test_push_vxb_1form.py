import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


@pytest.mark.parametrize("algo", ["analytic", "implicit"])
def test_push_vxb_1form_equals_2form(algo: str):
    """PushVxB1form with the equilibrium field as a 1-form rotates the velocities like PushVxB with the 2-form, up to
    the projection errors of the equilibrium field (Colella domain: non-polynomial form components), and conserves
    |v|."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians
    from struphy.models import LinearMHDVlasovPC
    from struphy.propagators.push_vxb import PushVxB
    from struphy.propagators.push_vxb_1form import PushVxB1form

    velocities = {}
    for b_space in ("Hdiv", "Hcurl"):
        model = LinearMHDVlasovPC(b_space=b_space)
        prop = model.propagators.push_vxb
        assert type(prop) is (PushVxB if b_space == "Hdiv" else PushVxB1form)
        prop.options = prop.Options(algo=algo)
        model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=500, seed=1234))
        model.energetic_ions.var.add_background(maxwellians.Maxwellian3D())
        sim = Simulation(
            model=model,
            domain=domains.Colella(alpha=0.1),
            equil=equils.HomogenSlab(B0x=0.3, B0y=0.2, B0z=1.0),
            grid=grids.TensorProductGrid(num_elements=(16, 16, 4)),
            derham_opts=DerhamOptions(degree=(3, 3, 2)),
        )
        sim.allocate()
        particles = model.energetic_ions.var.particles
        v0 = particles.markers[3:6, particles.valid_mks].T.copy()
        prop(0.05)
        v1 = particles.markers[3:6, particles.valid_mks].T.copy()
        assert xp.allclose(xp.linalg.norm(v1, axis=1), xp.linalg.norm(v0, axis=1), rtol=1e-12)
        assert xp.linalg.norm(v1 - v0) > 1e-3 * xp.linalg.norm(v0)
        velocities[b_space] = (v0, v1)

    (v0_2, v1_2), (v0_1, v1_1) = velocities["Hdiv"], velocities["Hcurl"]
    assert xp.allclose(v0_1, v0_2)
    # the velocity change agrees up to the (small) projection errors of the two equilibrium fields
    assert xp.linalg.norm((v1_1 - v0_1) - (v1_2 - v0_2)) < 1e-2 * xp.linalg.norm(v1_2 - v0_2)
