# Planar diode and the Child–Langmuir law (space charge)

![Child–Langmuir diode](child_langmuir_diode.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/child_langmuir_diode/child_langmuir_diode.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/child_langmuir_diode
```

This is the first benchmark of the self-consistent field, run with
`IonOpticsElectrostatic(space_charge=True)`. Protons leave an emitter at U = 10 kV almost at
rest, and are collected at a grounded plate d = 10 mm away. In the normalization
`IonOpticsUnits(length=d, voltage=U)`:
- the charge unit is ε₀Ud, so Poisson's equation reads −φ'' = ρ, and
- the space-charge-limited current density is exactly J_CL = 4√2/9. That is 54.5 mA/cm²
  for this diode.

Before every time step the model:
1. deposits the marker charges into the H1 dual space (`charge_density_0form`),
2. solves Poisson's equation with the same electrode constraints as the vacuum solve, and
3. pushes the markers.

Current is injected continuously with `InjectMarkers`.

## Results (64 elements, p = 3, dt = 0.01, 2000 markers per time unit, averaged over t ≥ 6)

| Injected J | Collector current | Max \|φ − φ_ref\| | Reflected to emitter |
|------------|-------------------|--------------------|----------------------|
| 0.25 J_CL  | 0.250 J_CL        | 3.0e-6             | 0 %                  |
| 0.5 J_CL   | 0.500 J_CL        | 8.3e-6             | 0 %                  |
| 0.9 J_CL   | 0.899 J_CL        | 1.3e-4             | 0 %                  |
| 2 J_CL     | 1.02 J_CL         | —                  | 50 %                 |

- φ_ref comes from `solve_bvp` applied to the steady-state equation −φ'' = J / √(v₀² + 2(1 − φ)).
- Below the limit, all of the injected current reaches the collector, and the
  potential agrees with the steady state to 1e-4 or better. At 0.9 J_CL, the remaining error
  is mostly the slow approach to the steady state.
- At twice the limit, a virtual anode forms in front of the emitter and reflects
  half of the beam. The transmitted current saturates at the Child–Langmuir value.

## Limitations

- The field is recomputed once per time step, before the Strang-split push, so the
  coupling is first-order in time. This does not matter for steady states.
- Each space-charge Poisson solve uses CG with a relative tolerance of 1e-12.
