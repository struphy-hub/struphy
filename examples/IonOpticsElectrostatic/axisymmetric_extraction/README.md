# Axisymmetric plasma extraction (round apertures, IBSimu-style)

![Axisymmetric extraction](axisymmetric_extraction.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/axisymmetric_extraction/axisymmetric_extraction.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/axisymmetric_extraction   # coarse, about 30 s
```

This is the generic extraction of [`plasma_extraction`](../plasma_extraction), with the same wall
profile, voltages and plasma, revolved about the beam axis. The apertures are round, which is
the most common IBSimu configuration.

- **Geometry and solver.** `AxisymmetricElectrodeChannel` (see
  [`axisymmetric_lens`](../axisymmetric_lens)), with the plasma electrode (0 V) and the puller
  (−1.5 kV) as segments on the outer wall. The steady state comes from `SteadyStateIteration`:
  ray tracing, then the Poisson–Boltzmann Newton solve on the convex energy. The meniscus is not
  prescribed.
- **Source.** `RayBundle.axisymmetric_disk`: a uniform disk of current density e·n₀·v_B at the back of
  the chamber, with radii ∝ √u so that every ray carries equal current. The rays are scrambled
  Sobol points, with an isotropic transverse temperature of 0.5 eV.
- **Exit phase space.** The radial phase space (r, r′) is shown. ε_x is computed from it with
  ⟨x²⟩ = ⟨r²⟩/2 and similar relations (no beam rotation).

The extracted fraction is about the aperture-to-chamber area ratio, (1/3)² ≈ 11 %, as expected
for uniform emission over the chamber. The defaults use the adaptive damping and the residual
criterion developed in [`plasma_extraction`](../plasma_extraction) (see its README for what it takes to
converge). Only about 1/9 of the rays pass the round aperture, so the noise-limited regime is reached
sooner than in the slit case.

The production check uses 1024 Sobol rays (a power of two) on 48 × 18 elements for 70 rounds. It
was run on 2026-09-22 and took 131 s. The last-ten-round extracted fraction was
0.0951 ± 0.0030 and current balance closed to roundoff; the potential change was 2.2e-5.
The alpha-independent charge residual remained noise limited at 0.27, however, so this is a stable
statistical result rather than strict fixed-point convergence. Run the opt-in regression with:

```sh
STRUPHY_RUN_DENSE_ION_OPTICS=1 .venv/bin/python -m pytest -q test_axisymmetric_extraction.py -k dense
```

This result closes the previously untested large-ray check and shows that importance sampling near
the round-aperture separatrix is still required for deterministic residual convergence.
