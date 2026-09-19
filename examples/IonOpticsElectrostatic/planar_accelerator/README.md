# Planar electrostatic accelerator

From the repository root, using the installed Struphy environment:

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/planar_accelerator/planar_accelerator.py
```

This serial example launches 17 ions as a single, slightly divergent bunch in
the prescribed potential `phi = 0.4 (1 - x)`, with `epsilon = 1`.
All quantities are normalized; the potential amplitude is not a voltage in SI units.
There is no continuous injection, self-consistent space charge, or electrode solve yet.
The integration ends before the bunch reaches a boundary.

Strang splitting advances the native Struphy model. The saved particle histories
are checked against `vx = vx0 + 0.4 t` and `x = x0 + vx0 t + 0.2 t²`,
and kinetic-energy gain is checked against potential-energy loss.
For this constant force, Strang splitting reproduces the analytic motion to roundoff.

The script saves `planar_accelerator.png`: projected potential, numerical beam
trajectories, work–energy comparison, and initial/final transverse phase space.
The reduction of `vy/vx` is acceleration-induced angular compression; there is
no transverse focusing force in this benchmark. Simulation data are written to
`output/planar_accelerator/`; rerunning replaces this example's output.

```sh
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/planar_accelerator/test_planar_accelerator.py
```

The tests also check two charge-to-mass normalization factors.
