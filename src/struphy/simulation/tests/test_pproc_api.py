"""Unit tests for the Simulation post-processing convenience behavior."""

import importlib

from struphy import Simulation


def test_pproc_can_load_and_return_plotting_data(monkeypatch):
    calls = []
    expected = object()

    class FakePostProcessor:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))

        def process(self, **kwargs):
            calls.append(("process", kwargs))

    class FakeSimulation:
        rank = 0

        @property
        def post_processor(self):
            return self._post_processor

        def load_plotting_data(self):
            calls.append(("load", {}))
            return expected

    simulation_module = importlib.import_module("struphy.simulation.sim")
    monkeypatch.setattr(simulation_module, "PostProcessor", FakePostProcessor)

    sim = FakeSimulation()
    data = Simulation.pproc(sim, physical=True, create_vtk=False, force=False, load=True)

    assert data is expected
    assert calls == [
        ("construct", {"sim": sim, "parallel_pproc": False}),
        (
            "process",
            {
                "step": 1,
                "celldivide": 1,
                "physical": True,
                "guiding_center": False,
                "classify": False,
                "create_vtk": False,
                "force": False,
            },
        ),
        ("load", {}),
    ]


def test_pproc_does_not_load_by_default(monkeypatch):
    class FakePostProcessor:
        def __init__(self, **kwargs):
            pass

        def process(self, **kwargs):
            pass

    class FakeSimulation:
        rank = 0

        @property
        def post_processor(self):
            return self._post_processor

        def load_plotting_data(self):
            raise AssertionError("plotting data should not be loaded")

    simulation_module = importlib.import_module("struphy.simulation.sim")
    monkeypatch.setattr(simulation_module, "PostProcessor", FakePostProcessor)

    assert Simulation.pproc(FakeSimulation()) is None
