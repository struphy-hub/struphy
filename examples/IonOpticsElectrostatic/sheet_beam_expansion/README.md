# Sheet-beam expansion by space charge

![Sheet-beam expansion](sheet_beam_expansion.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/sheet_beam_expansion/sheet_beam_expansion.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/sheet_beam_expansion
```

This is the 2D counterpart of the plan's "uniform beam in a drift space" benchmark.

A laminar, uniform proton sheet beam is injected continuously between grounded plates:
5 keV, a₀ = 2 mm, 55 µA per mm of slit length. By Gauss's law, the field at the edge of a
uniform slab is E = I′/(2v), independent of the plates. Every ray therefore follows
y = y₀ (1 + I′(x − x₀)²/(4a₀v³)). The beam stays uniform, and its rms size is a(x)/√3.

## Results (160 × 20 elements, p = 3, dt = 0.04, 250 markers per time unit)

- **Edge growth.** The analytic edge grows by 49.5 % over 80 mm. The rms size of the
  simulated steady beam follows the analytic envelope.
- **Growth coefficient.** A least-squares fit to all 6000 markers in flight gives
  7.81e-5 ± 0.32e-5 per mm², against the analytic 7.93e-5 (−1.6 %). The error bar
  is from a bootstrap.
- **Profiles.** Transverse profiles stay uniform (flat-top) at both x = 10 mm and x = 70 mm.
- **Losses.** No marker reaches the plates.

The 2 mm slices contain only about 150 markers each, so single slices scatter by a few
percent. The global fit is the meaningful comparison.
