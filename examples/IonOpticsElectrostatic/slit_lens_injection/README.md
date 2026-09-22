# Continuous injection through the slit lens (zero current)

![Continuous injection](slit_lens_injection.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/slit_lens_injection/slit_lens_injection.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/slit_lens_injection
```

This example uses the same lens as [`slit_immersion_lens`](../slit_immersion_lens): 5 keV
protons, V₂ = −10 kV. The beam is emitted continuously from the plane x = 1 mm:

- **Source.** Uniform over |y| ≤ 4.5 mm, with a 40 mrad rms angular spread, at 80 markers/ns.
  `PlaneSource` provides the source and the `InjectMarkers` propagator emits it.
  Emission times are staggered within each time step, so the beam is not bunched at the step frequency.
- **Charges.** Each marker's weight is `current / rate`. Charges are therefore
  fractions of the injected current multiplied by the time unit.
- **Loss accounting.** `Particles` now records every marker that a `"remove"`
  boundary deletes. `CurrentLedger` books these records to the `LossTag` parts:
  electrode 1, electrode 2, outlet and inlet. The model saves the injected, in-flight
  and per-part lost charge as scalars, and keeps the outlet records to give the exit phase space.

## Results (160 × 20 elements, p = 3, dt = 0.04, 145 ns)

- **Bookkeeping.** Injected = in-flight + Σ lost holds to 1e-12 at every step.
- **Steady state.** The in-flight charge is constant after about 60 ns, the transit time
  of the slowest ions. Over the steady window it changes by only 0.2 % of the injected
  charge.
- **Currents vs exact-field Monte Carlo.** The reference is 4000 rays traced by RK4 in the
  analytic field. The errors are binomial standard errors.

  | Boundary part | Struphy       | Reference     |
  |---------------|---------------|---------------|
  | Electrode 1   | 3.9 ± 0.3 %   | 3.7 ± 0.3 %   |
  | Electrode 2   | 1.8 ± 0.2 %   | 1.4 ± 0.2 %   |
  | Outlet        | 94.5 ± 0.3 %  | 94.9 ± 0.4 %  |

  All three agree within about 1.5σ of the combined statistics.
- **Outlet emittance.** 54 mm·mrad (rms, trace space).

## Limitations

- Serial only: `InjectMarkers` raises on more than one MPI rank.
- The marker array is sized through `bufsize` from a single placeholder marker.
  `SavingParameters(n_markers=0)` is not supported by Struphy, so one marker is saved.
- Zero current: space charge (plan phase 9) is the next step.
