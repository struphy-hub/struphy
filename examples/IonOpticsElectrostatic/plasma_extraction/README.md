# Slit plasma extraction (IBSimu-style)

![Plasma extraction](plasma_extraction.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/plasma_extraction/plasma_extraction.py   # about 2 min
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/plasma_extraction                           # coarse, about 30 s
```

This is a generic 2D positive-ion slit extraction, set up the way IBSimu's
plasma-extraction examples are. It is not a specific device; all parameters are in the
`Extraction` dataclass.

| Part | Setup |
|------|-------|
| Geometry | `SegmentedElectrodeChannel`, a conforming spline mapping. The plasma chamber is 6 mm high; the plasma electrode (0 V) has a 2 mm slit; there is a gap; the puller (−1.5 kV) has a 3 mm slit, followed by a drift region. The electrode voltages are Dirichlet data on wall segments. The gap wall has the natural boundary condition. |
| Plasma | `BoltzmannElectrons`: T_e = 5 eV, n₀ = 2.8e16 m⁻³ (λ_D = 0.1 mm), φ_P = 17 V (Kalvas eq. 2.28 for hydrogen). |
| Ions | H⁺ from the back of the chamber at the Bohm speed, with the quasi-neutral flux n₀v_B and T_t = 0.5 eV (900 scrambled-Sobol rays). |
| Solver | `SteadyStateIteration` with adaptive damping (α₀ = 0.2): ray tracing, trajectory deposit, then the Poisson–Boltzmann Newton solve on the convex energy. The meniscus is not prescribed. |

## What the figure shows

- **Top:** electrodes, equipotentials (every 100 V), and the meniscus, plotted as the
  contour φ = φ_P − 2T_e. Extracted rays are red; a sample of the lost ones is grey.
- **Bottom:** the Boltzmann electron density (the plasma sits behind the aperture, with a
  concave meniscus inside it), the convergence history (fixed-point residual and the damping α),
  and the exit phase space.

With the default settings the iteration converges in 48 rounds (about 1.5 min): 30 % of the
emitted current is extracted, 63 % ends on the plasma electrode and 7 % on the puller. The
extracted beam is focused by the concave meniscus to a waist in the gap and diverges through the
puller; its exit phase space is nearly a line (a laminar beam), ε_rms = 7.8 mm·mrad.
Most ions strike the chamber walls because they are emitted over the whole 6 mm chamber height
while the slit is only 2 mm wide.

## The outer iteration: what it takes to converge

The steady state is a fixed point of ρ → solve Poisson–Boltzmann → trace rays → ρ*. Findings from
runs on this case (48 × 18 elements):

- **A constant damping α = 0.5 never converges** (a limit cycle: more ion charge deforms the
  potential so that the beam is deflected away from the same region, and the charge alternates).
  α = 0.2 converges, but slowly; Anderson acceleration did not help.
- **`relaxation="adaptive"`** (`AdaptiveRelaxation`) grows α while successive residuals agree and shrinks
  it when they alternate. With a smooth map α climbs to 1 and the residual falls geometrically (900 rays:
  0.2 → 7e-5 in 60 rounds).
- **Ray count.** With few rays the charge map is not smooth: a ray flips between "extracted" and
  "intercepted at the electrode edge", so the residual stalls at a noise floor. At 300 and 512 rays
  the iteration only scatters around the solution (extracted current 0.25 – 0.33 between rounds); at 900 it
  locks into an exactly self-consistent state and converges. The damping then averages the noise: a safeguard
  halves the cap on α when the residual sets no new minimum for 10 rounds.
- **Reporting.** `iteration.converged` uses the undamped residual (`criterion="residual"`), so it does not
  claim convergence in the noise-limited regime; `iteration.averaged(10)` gives mean ± std over the last rounds
  for those runs. The figure title switches to that form when not converged.
- **Guidance:** at least about 900 rays for this geometry (roughly 100 per mm² of chamber cross-section). For an
  unconverged run, quote the averaged values with their scatter.

## Limitations

- **Gap boundary.** The gap between electrodes is a natural (zero normal field) boundary.
  IBSimu treats open domain boundaries the same way, but a real gap opens into a larger vacuum.
- **Statistical error.** The extracted current differs by about 10 % between ray sets (0.33 at 300 rays,
  0.37 at 900), so the ray-count dependence should be checked before quoting a number.
- **2D slit only** here; round apertures are in [`axisymmetric_extraction`](../axisymmetric_extraction).
- **Serial only.**
