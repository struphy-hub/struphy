import cunumpy as xp

from struphy.models.scalars import KineticEnergyPIC, KineticEnergySPH, Scalar, Scalars
from struphy.models.variables import PICVariable, SPHVariable


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

        assert before > 0.0
        assert after == 4.0 * before, f"{scalar_class.__name__} did not follow the markers: {before} -> {after}"
