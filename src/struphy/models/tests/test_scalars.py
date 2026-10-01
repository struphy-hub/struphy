import cunumpy as xp

from struphy.models.scalars import KineticEnergyPIC, KineticEnergySPH, Scalar, Scalars
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.particles import Particles5D


class _Constant(Scalar):
    """A scalar whose value is set from outside, to follow updates without a simulation."""

    def __init__(self, value):
        super().__init__()
        self.current = value

    def _local_update(self):
        self.local_value[0] = self.current

    def _mpi_sum(self):
        self.value[0] = self.local_value[0]


def test_nested_sum_follows_its_summands():
    """`a + b + c` is SumOfScalars(SumOfScalars(a, b), c); the inner sum must be recomputed at every update."""
    a, b, c = _Constant(1.0), _Constant(2.0), _Constant(3.0)
    scalars = Scalars(a=a, b=b, c=c, total=a + b + c)

    scalars.update()
    assert scalars.dct["total"].value[0] == 6.0

    a.current, b.current, c.current = 10.0, 20.0, 30.0
    scalars.update()
    assert scalars.dct["total"].value[0] == 60.0


class _DriftKineticMarkers:
    """Particles5D-like markers: weight in column 5, mu * |B_0| in column 8, plus holes and ghosts."""

    def __init__(self, Np):
        self.Np = Np
        self.markers = xp.zeros((Np + 4, 10), dtype=float)
        self.markers[:Np, 5] = 2.0 / Np  # density 2; the weights carry the 1/Np of the Monte-Carlo estimate
        self.markers[:Np, 8] = 3.0
        self.markers[Np:, 5:9] = 1.0  # holes and ghosts, which must not be counted
        self.valid_mks = xp.zeros(Np + 4, dtype=bool)
        self.valid_mks[:Np] = True

    @property
    def markers_wo_holes_and_ghost(self):
        return self.markers[self.valid_mks]

    def save_magnetic_background_energy(self):
        pass


def test_drift_kinetic_magnetic_energy_is_weighted():
    """The magnetic energy sum_p w_p mu_p |B_0(eta_p)| must not depend on Np, since the weights include 1/Np."""
    from types import SimpleNamespace

    from struphy.models.drift_kinetic_electrostatic_adiabatic import DriftKineticElectrostaticAdiabatic

    for Np in (4, 400):
        particles = _DriftKineticMarkers(Np)
        model = SimpleNamespace(kinetic_ions=SimpleNamespace(var=SimpleNamespace(particles=particles)))
        energy = DriftKineticElectrostaticAdiabatic._compute_en_particle_magnetic(model)
        assert xp.isclose(energy, 6.0), f"magnetic energy = {energy} for Np = {Np}, expected 6.0"


class _MovingParticles:
    """Markers whose accessors return copies, as `Particles.velocities` does (it is fancy-indexed)."""

    def __init__(self):
        self._markers = xp.ones((4, 3), dtype=float)
        self._weights = xp.ones(4, dtype=float)
        self._valid = xp.ones(4, dtype=bool)
        self.Np = 4

    @property
    def velocities(self):
        return self._markers[self._valid]  # boolean indexing, as Particles does: a copy, not a view

    @property
    def weights(self):
        return self._weights[self._valid]

    def accelerate(self, factor):
        self._markers *= factor


class _FakePICVariable(PICVariable):
    def __init__(self, particles):
        self._particles = particles


class _FakeSPHVariable(SPHVariable):
    def __init__(self, particles):
        self._particles = particles


def test_kinetic_energy_follows_the_markers():
    """The marker arrays are copies, so a cached one would report the first step's energy forever."""
    for variable_class, scalar_class in ((_FakePICVariable, KineticEnergyPIC), (_FakeSPHVariable, KineticEnergySPH)):
        particles = _MovingParticles()
        scalar = scalar_class(variable_class(particles))

        scalar._local_update()
        before = float(scalar.local_value[0])

        particles.accelerate(2.0)  # four times the kinetic energy
        scalar._local_update()
        after = float(scalar.local_value[0])

        # Four markers of weight 1 and |v|^2 = 3: the energy is 0.5 * sum(w |v|^2) = 6. The weights already
        # carry the 1/Np of the Monte-Carlo estimate, so the sum must not be divided by Np a second time.
        assert before == 6.0, f"{scalar_class.__name__} is normalized wrongly: {before} instead of 6.0"
        assert after == 4.0 * before, f"{scalar_class.__name__} did not follow the markers: {before} -> {after}"


class _Fake5DParticles(Particles5D):
    """Particles5D markers with velocity columns (v_par, mu), without the equilibrium needed to set them up."""

    def __init__(self, v_par, mu, weights):
        self._velocities = xp.stack((v_par, mu), axis=1)
        self._weights = weights

    @property
    def velocities(self):
        return self._velocities

    @property
    def weights(self):
        return self._weights


def test_kinetic_energy_ignores_magnetic_moment_of_5d_markers():
    """For Particles5D the second velocity column is mu, which must not enter 0.5 * sum(w v^2)."""
    v_par = xp.array([1.0, -2.0, 3.0])
    mu = xp.array([5.0, 7.0, 11.0])
    weights = xp.array([0.5, 1.0, 2.0])
    scalar = KineticEnergyPIC(_FakePICVariable(_Fake5DParticles(v_par, mu, weights)))

    scalar._local_update()

    expected = 0.5 * float(xp.sum(weights * v_par**2))
    assert float(scalar.local_value[0]) == expected


class _GuidingCenterMarkers:
    """Particles5D-like markers: weight in column 5, mu * |B_0| in column 8 (the first diagnostics column)."""

    def __init__(self, Np):
        self.Np = Np
        self.markers = xp.zeros((Np + 2, 10), dtype=float)
        self.markers[:Np, 5] = 1.0 / Np  # the weights carry the 1/Np of the Monte-Carlo estimate
        self.markers[:Np, 8] = 3.0
        self.markers[Np:, 5:9] = -1.0  # holes
        self.holes = xp.zeros(Np + 2, dtype=bool)
        self.holes[Np:] = True

    def save_magnetic_background_energy(self):
        pass


def test_guiding_center_en_fB_does_not_divide_by_Np():
    """en_fB = sum_p w_p mu_p |B_0(eta_p)| must not depend on Np, since the weights already include 1/Np."""
    from types import SimpleNamespace

    from struphy.models.guiding_center import GuidingCenter

    for Np in (4, 400):
        model = SimpleNamespace(
            kinetic_ions=SimpleNamespace(var=SimpleNamespace(particles=_GuidingCenterMarkers(Np))),
        )
        energy = GuidingCenter._compute_en_fB(model)
        assert xp.isclose(energy, 3.0), f"en_fB = {energy} for Np = {Np}, expected 3.0"
