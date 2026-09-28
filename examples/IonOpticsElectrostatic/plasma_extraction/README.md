# Slit plasma extraction (IBSimu-style)

![Plasma extraction](plasma_extraction.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/plasma_extraction/plasma_extraction.py   # 65 rounds, 3-6 min depending on load
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/plasma_extraction                           # coarse, a few seconds
```

This is a generic 2D positive-ion slit extraction, set up the way IBSimu's
plasma-extraction examples are. It is not a specific device; all parameters are in the
`Extraction` dataclass.

| Part | Setup |
|------|-------|
| Geometry | `SegmentedElectrodeChannel`, a conforming spline mapping. The plasma chamber is 6 mm high; the plasma electrode (0 V) has a 2 mm slit; there is a gap; the puller (−1.5 kV) has a 3 mm slit, followed by a drift region. The walls are piecewise linear with sharp corners, and the mapping is of degree 1 along the channel on 64 elements, so that every corner is an element boundary: the mapped walls are the profile exactly (`domain.wall_error` ≈ 1e-16 mm). A cubic mapping on the former 48 elements rounded the lips and overshot them, which narrowed the plasma-electrode slit from 2.0 to 1.70 mm. The electrode voltages are Dirichlet data on wall segments. The gap wall has the natural boundary condition. |
| Plasma | `BoltzmannElectrons`: T_e = 5 eV, n₀ = 2.8e16 m⁻³ (λ_D = 0.1 mm), φ_P = 17 V (Kalvas eq. 2.28 for hydrogen). |
| Ions | H⁺ from the back of the chamber at the Bohm speed, with the quasi-neutral flux n₀v_B and T_t = 0.5 eV (1800 scrambled-Sobol rays). |
| Solver | `SteadyStateIteration` with adaptive damping (α₀ = 0.2): fused fixed-step ray tracing with trajectory deposit (`FixedStepRayTracer`, one kernel call per step), then the Poisson–Boltzmann Newton solve on the convex energy. The meniscus is not prescribed. |

## What the figure shows

- **Top:** electrodes (drawn from the mapped walls of the simulated domain), equipotentials (every
  100 V), and the meniscus, plotted as the contour φ = φ_P − 2T_e. Extracted rays are red; a sample of the lost ones is grey.
- **Bottom:** the Boltzmann electron density (the plasma sits behind the aperture, with a
  concave meniscus inside it), the convergence history (fixed-point residual and the damping α),
  and the exit phase space.

With the default settings (1800 rays, 64 × 18 elements) the iteration converges in 65 rounds (3–6 min depending on
machine load): 27.8 % of the emitted current is extracted, 71.2 % ends on the plasma electrode and 1.0 % on the
puller, and the exit emittance is ε_rms = 7.30 mm·mrad. (With the rounded walls of the former cubic mapping it was
29.9 %, 63.4 %, 6.6 % and 7.64 mm·mrad: the overshooting puller lip intercepted part of the beam.) The extracted beam is focused by the concave meniscus to a waist in the gap and diverges through the
puller; its exit phase space is nearly a line (a laminar beam).
Most ions strike the chamber walls because they are emitted over the whole 6 mm chamber height
while the slit is only 2 mm wide.

## The outer iteration: what it takes to converge

The steady state is a fixed point of ρ → solve Poisson–Boltzmann → trace rays → ρ*. Findings from
runs on this case (48 × 18 elements, on the former cubic mapping with rounded lips; the numbers below were not
rerun with the exact walls, the behaviour of the iteration should carry over):

- **A constant damping α = 0.5 never converges** (a limit cycle: more ion charge deforms the
  potential so that the beam is deflected away from the same region, and the charge alternates).
  α = 0.2 converges, but slowly; Anderson acceleration did not help.
- **`relaxation="adaptive"`** (`AdaptiveRelaxation`) grows α while successive residuals agree and shrinks
  it when they alternate. With a smooth map α climbs to 1 and the residual falls geometrically (900 rays:
  0.2 → 7e-5 in 60 rounds).
- **Ray count.** With few rays the charge map is not smooth: a ray flips between "extracted" and
  "intercepted at the electrode edge", so the residual stalls at a noise floor. At 300 and 512 rays
  the iteration only scatters around the solution (extracted current 0.25 – 0.33 between rounds). At 900 it
  locks into an exactly self-consistent state, but *when* depends on round-off: the propagator and the fused
  tracer (identical to 1e-10 per round) converged after 48 and 97 rounds to the same state (transmission 0.300,
  ε = 7.75 mm·mrad). At 1800 rays it takes 47 rounds and at 2700 rays 38 rounds with α climbing to 1, all to
  transmission 0.299 – 0.300 and ε = 7.6 – 7.8 mm·mrad. The damping averages the noise: a safeguard
  halves the cap on α when the residual sets no new minimum for 10 rounds.
- **Reporting.** `iteration.converged` uses the undamped residual (`criterion="residual"`), so it does not
  claim convergence in the noise-limited regime; `iteration.averaged(10)` gives mean ± std over the last rounds
  for those runs. The figure title switches to that form when not converged.
- **Guidance:** at least about 1800 rays for this geometry (roughly 200 per mm² of chamber cross-section); with
  the fused tracer, rays are cheap. For an unconverged run, quote the averaged values with their scatter.

## Limitations

- **Gap boundary.** The gap between electrodes is a natural (zero normal field) boundary.
  IBSimu treats open domain boundaries the same way, but a real gap opens into a larger vacuum.
- **Statistical error.** Between 900, 1800 and 2700 rays the converged transmission is 0.299 – 0.300 and the
  emittance 7.6 – 7.8 mm·mrad; below 900 rays the iteration does not lock in and the extracted current scatters by
  about 10 %. Check the ray-count dependence before quoting a number for a new geometry.
- **2D slit only** here; round apertures are in [`axisymmetric_extraction`](../axisymmetric_extraction).
- **Serial only.**
