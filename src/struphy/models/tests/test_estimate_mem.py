from types import SimpleNamespace
from unittest.mock import patch

from struphy.models.variables import PICVariable
from struphy.simulation import sim as sim_module
from struphy.simulation.sim import Simulation


def test_simulation_estimate_mem_returns_total():
    class DummyDerham:
        def __init__(self, grid, derham_opts, comm=None, domain=None):
            self.grid = grid
            self.derham_opts = derham_opts
            self.comm = comm
            self.domain = domain
            # derivative operators are matrix-free (no data), see linop_nbytes
            self.grad = None
            self.curl = None
            self.div = None

    class DummyMassOperators:
        def __init__(self, derham, domain, eq_mhd=None):
            assert isinstance(derham, DummyDerham)

        def estimate_mem(self):
            return {"M1": 7}

    class DummyFEECVariable:
        def estimate_mem(self, derham):
            assert isinstance(derham, DummyDerham)
            return 10

    class DummyPICVariable:
        def estimate_mem(self, clone_config=None, derham=None, domain=None, equil=None):
            assert isinstance(derham, DummyDerham)
            return 20

    class DummySPHVariable:
        def estimate_mem(self, derham=None, domain=None, equil=None):
            assert isinstance(derham, DummyDerham)
            return 30

    sim = Simulation.__new__(Simulation)
    sim._grid = object()
    sim._derham_opts = object()
    sim._clone_config = None
    sim._domain = object()
    sim._equil = object()
    sim.comm = None
    sim.rank = 0
    sim.comm_size = 1
    sim._model = SimpleNamespace(
        field_species={"f": SimpleNamespace(variables={"u": DummyFEECVariable()})},
        fluid_species={},
        particle_species={"p": SimpleNamespace(variables={"v": DummyPICVariable(), "w": DummySPHVariable()})},
        diagnostic_species={"d": SimpleNamespace(variables={"z": DummyFEECVariable()})},
    )

    with (
        patch.object(sim_module, "Derham", DummyDerham),
        patch.object(sim_module, "WeightedMassOperators", DummyMassOperators),
        patch.object(sim_module, "FEECVariable", DummyFEECVariable),
        patch.object(sim_module, "PICVariable", DummyPICVariable),
        patch.object(sim_module, "SPHVariable", DummySPHVariable),
    ):
        mem = sim.estimate_mem(print_report=False)

    assert mem["f.u"] == 10
    assert mem["p.v"] == 20
    assert mem["p.w"] == 30
    assert mem["d.z"] == 10
    assert mem["matrices.derivatives"] == 0
    assert mem["matrices.M1"] == 7
    assert mem["total"] == 77
    assert mem["total"] >= 0


def test_picvariable_estimate_mem_uses_dry_run_particles():
    class DummyKineticBackground:
        pass

    class DummyParticles:
        last_instance = None
        last_kwargs = None

        def __init__(self, **kwargs):
            DummyParticles.last_kwargs = kwargs
            DummyParticles.last_instance = self
            self.Np = 16
            self.n_cols = 12
            self.nbytes_local = 128
            if not kwargs.get("dry_run", False):
                self.markers = object()

    var = PICVariable(space="Particles6D")
    var._species = SimpleNamespace(
        loading_params=SimpleNamespace(),
        weights_params=SimpleNamespace(),
        boundary_params=SimpleNamespace(),
        sorting_params=SimpleNamespace(),
        bufsize=0.25,
        equation_params={},
        saving_params=SimpleNamespace(n_markers=0),
    )
    var._backgrounds = DummyKineticBackground()
    var._initial_condition = var._backgrounds
    var._n_as_volume_form = False

    with (
        patch("struphy.models.variables.KineticBackground", DummyKineticBackground),
        patch("struphy.models.variables.particles.Particles6D", DummyParticles),
    ):
        nbytes = var.estimate_mem()

    assert DummyParticles.last_kwargs["dry_run"] is True
    assert not hasattr(DummyParticles.last_instance, "markers")
    assert nbytes == 128
