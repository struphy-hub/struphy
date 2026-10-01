import pytest


@pytest.mark.parametrize("mapping", [["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 4.0}]])
def test_energy_utilities(mapping):
    """get_kinetic_energy_particles and get_electron_thermal_energy run with the current kernel arguments."""

    from types import SimpleNamespace

    import cunumpy as xp

    from struphy import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.pic.utilities import get_electron_thermal_energy, get_kinetic_energy_particles
    from struphy.topology.grids import TensorProductGrid

    num_elements = [4, 3, 2]
    domain = getattr(domains, mapping[0])(**mapping[1])
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=[2, 2, 1]), comm=None)
    volume = 2.0 * 3.0 * 4.0

    # kinetic energy with zero vector potential: 0.5 * sum_p w_p |v_p|^2
    rng = xp.random.default_rng(0)
    markers = xp.zeros((5, 7))
    markers[:, :3] = rng.uniform(0.0, 1.0, (5, 3))
    markers[:, 3:6] = rng.normal(size=(5, 3))
    markers[:, 6] = rng.uniform(0.5, 1.5, 5)
    markers[-1, 0] = -1.0  # hole
    a = derham.V1.zeros()
    res = get_kinetic_energy_particles(a, derham, domain, SimpleNamespace(markers=markers))
    ref = 0.5 * xp.sum(markers[:-1, 6] * xp.sum(markers[:-1, 3:6] ** 2, axis=1))
    assert xp.isclose(res[0], ref)

    # thermal energy of a constant density n = e: int n ln(n) sqrt(g) = e * volume
    nqs = derham.nquads
    pads = derham.V0fem.coeff_space.pads
    data = xp.full([n + 2 * p for n, p in zip(num_elements, pads)] + list(nqs), xp.e)
    density = SimpleNamespace(_operators=[SimpleNamespace(matrix=SimpleNamespace(_data=data))])
    res = get_electron_thermal_energy(density, derham, domain, *num_elements, *nqs)
    assert xp.isclose(res[0], xp.e * volume)


if __name__ == "__main__":
    test_energy_utilities(["Cuboid", {"l1": 0.0, "r1": 2.0, "l2": 0.0, "r2": 3.0, "l3": 0.0, "r3": 4.0}])
