# Slit lens at increasing beam current (space charge)

![Slit lens with space charge](slit_lens_space_charge.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/slit_lens_space_charge/slit_lens_space_charge.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/slit_lens_space_charge
```

This is the current ramp of plan phase 9, run on the lens of
[`slit_immersion_lens`](../slit_immersion_lens). The beam is 5 keV protons,
V₂ = −10 kV, ±3 mm wide with a 3 mrad rms angular spread, injected continuously as in
[`slit_lens_injection`](../slit_lens_injection).

With `space_charge=True`, the model deposits the ion charge before every step and
re-solves Poisson's equation with the electrode voltages. Currents are given per mm of
slit length, because the simulation domain is 1 mm thick in the invariant direction z.
The normalized current is `I / IonOpticsUnits.current`, where the current unit is 2.74 mA for 1 mm and 1 kV.

## Results (160 × 20 elements, p = 3, dt = 0.04, 80 markers/ns, 145 ns)

| Current (µA per mm) | Waist x (mm) | Waist rms (mm) | Outlet rms emittance (mm·mrad) | Transmission |
|---------------------|--------------|----------------|--------------------------------|--------------|
| 0                   | 54.9         | 0.07           | 3.28                           | 100 %        |
| 100                 | 58.4         | 0.08           | 3.71                           | 100 %        |
| 200                 | 64.3         | 0.10           | 4.48                           | 100 %        |
| 400                 | 77.3         | 0.20           | 7.01                           | 100 %        |

- Space-charge defocusing weakens the lens, so the waist moves downstream and grows.
- The nonlinear space-charge field near the waist bends the outlet phase space into an
  S-shape, which increases the rms emittance.
- At 400 µA/mm the waist has almost left the 80 mm domain.

## Limitations

- These results are not converged. The mesh, marker-count and time-step studies of plan
  phase 10 are still to do for these currents, so the numbers above show trends
  rather than validated design values.
- There is no electron space-charge compensation downstream (plan phase 9), so the
  defocusing is an upper bound for a real beamline with residual gas.
