"""Space-charge defocusing moves the lens waist downstream (coarse, fast version)."""

from dataclasses import replace

from slit_lens_space_charge import LAMINAR_BEAM, run, steady_beam, waist


def test_space_charge_moves_waist_downstream(tmp_path):
    beam = replace(LAMINAR_BEAM, markers_per_ns=40.0)
    waists = {}
    for current in (0.0, 400.0):
        sim = run(tmp_path, current, beam=beam, num_elements=(60, 10), dt=0.08, end_time=34.0)
        result = steady_beam(sim)
        assert result["transmission"] > 0.99
        waists[current] = waist(result)[0]
    assert waists[400.0] > waists[0.0] + 4.0
