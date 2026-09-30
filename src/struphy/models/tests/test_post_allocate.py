from types import SimpleNamespace

import cunumpy as xp
import pytest

from struphy.models import cold_plasma_vlasov, vlasov_ampere_one_species


class _Particles:
    """Minimal stand-in for Particles: `update_weights` turns full-f weights into delta-f weights."""

    def __init__(self):
        self.weights0 = xp.array([1.0, 2.0, 3.0])
        self.weights = self.weights0.copy()

    def update_weights(self):
        self.weights = self.weights0 - 0.5


@pytest.mark.parametrize(
    "module, cls, species",
    [
        (cold_plasma_vlasov, "ColdPlasmaVlasov", "hot_elec"),
        (vlasov_ampere_one_species, "VlasovAmpereOneSpecies", "kinetic_ions"),
    ],
)
def test_post_allocate_resets_weights(monkeypatch, module, cls, species):
    """After the initial Poisson solve the marker weights must be reset to the initial weights w0."""
    particles = _Particles()
    weights_in_solve = []

    def poisson(dt):
        weights_in_solve.append(particles.weights.copy())

    poisson.allocate = lambda: None
    poisson.variables = SimpleNamespace(phi=SimpleNamespace(spline=SimpleNamespace(vector=xp.zeros(1))))

    model = SimpleNamespace(
        initial_poisson=poisson,
        em_fields=SimpleNamespace(e_field=SimpleNamespace(spline=SimpleNamespace(vector=None))),
    )
    setattr(model, species, SimpleNamespace(var=SimpleNamespace(particles=particles)))

    grad = SimpleNamespace(dot=lambda v, out=None: out)
    monkeypatch.setattr(module, "Propagator", SimpleNamespace(derham=SimpleNamespace(grad=grad)))

    getattr(module, cls).post_allocate(model)

    # the Poisson source uses the control-variate weights ...
    assert xp.allclose(weights_in_solve[0], particles.weights0 - 0.5)
    # ... but afterwards the weights are back to w0
    assert xp.allclose(particles.weights, particles.weights0)
