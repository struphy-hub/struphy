"""Tests for the public post-processing convenience API."""

from struphy.api import post_processing


def test_post_process_processes_then_loads(monkeypatch):
    calls = []
    expected = object()

    class FakeSimulation:
        def pproc(self, **kwargs):
            calls.append(kwargs)
            return expected

    sim = FakeSimulation()

    data = post_processing.post_process(
        sim=sim,
        step=2,
        celldivide=(2, 3, 4),
        physical=True,
        guiding_center=True,
        classify=True,
        create_vtk=False,
        force=False,
    )

    assert data is expected
    assert calls == [
        {
            "step": 2,
            "celldivide": (2, 3, 4),
            "physical": True,
            "guiding_center": True,
            "classify": True,
            "create_vtk": False,
            "force": False,
            "load": True,
        }
    ]


def test_post_process_accepts_an_output_path(monkeypatch):
    seen = []

    class FakePostProcessor:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def process(self, **kwargs):
            pass

    class FakePlottingData:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def load(self):
            pass

    monkeypatch.setattr(post_processing, "PostProcessor", FakePostProcessor)
    monkeypatch.setattr(post_processing, "PlottingData", FakePlottingData)

    post_processing.post_process(path_out="sim_1")

    assert seen == [
        {"sim": None, "path_out": "sim_1"},
        {"sim": None, "path_out": "sim_1"},
    ]
