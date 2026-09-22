"""Continuous injection: exact charge bookkeeping and losses booked per boundary part."""

from dataclasses import replace

import numpy as np
from slit_lens_injection import BEAM, TAGS, build_simulation, load_charges


def test_injection_reaches_steady_state_with_exact_charge_balance(tmp_path):
    beam = replace(BEAM, markers_per_ns=20.0)
    sim = build_simulation(tmp_path, num_elements=(80, 10), dt=0.08, end_time=30.0, beam=beam)
    sim.run()
    model = sim.model
    model.update_ledger()
    time, charges = load_charges(sim)
    ledger = model.ledger
    source = model.propagators.inject.source

    lost = sum(charges[f"lost_charge_{name}"] for name in (*TAGS, "other"))
    np.testing.assert_allclose(charges["injected_charge"] - charges["live_charge"] - lost, 0.0, atol=1e-10)
    # Injection follows the emission rate to within one marker.
    np.testing.assert_allclose(ledger.injected_markers, source.rate * time[-1], atol=1.0)
    # Every removed marker is attributed to a named boundary part.
    assert ledger.lost_markers["other"] == 0
    assert ledger.lost_markers["inlet"] == 0
    assert ledger.lost_markers["electrode 1"] > 0 and ledger.lost_markers["outlet"] > 0
    # Once the beam has crossed the domain, the in-flight charge is constant up to noise.
    late = time > 22.0
    live = charges["live_charge"][late]
    assert np.ptp(live) < 0.1 * live.mean()
    # Outlet records hold physical exit points.
    exits = ledger.records["outlet"]
    assert len(exits) == ledger.lost_markers["outlet"]
    np.testing.assert_allclose(exits[:, 0], 80.0, atol=0.5)
