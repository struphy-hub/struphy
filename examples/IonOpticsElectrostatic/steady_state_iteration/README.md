# Steady-state Vlasov–Poisson iteration (IBSimu cycle)

![Steady-state iteration](steady_state_iteration.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/steady_state_iteration/steady_state_iteration.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/steady_state_iteration
```

`struphy.models.ion_optics_steady_state.SteadyStateIteration` implements the iteration
cycle of Kalvas (2013, §5.1 and §5.6–5.8) on top of `IonOpticsElectrostatic`:

1. It starts from the vacuum potential of the electrode solve.
2. It traces a fixed `RayBundle` through the frozen field until all rays have left the domain.
   The rays are scrambled Sobol samples of a `PlaneSource`, and each carries a current `I`.
   After every step, each ray deposits the charge `I·dt` at its position
   (`charge_density_0form`, a Galerkin deposit, so no boundary-node correction is needed).
3. It under-relaxes the charge, ρ_k = α ρ* + (1 − α) ρ_{k−1}, and solves Poisson's
   equation (sparse LU, factorized once).
4. It stops when the mean of the last 5 changes of the exit rms emittance falls below
   `tol` (default 1e-3) for two consecutive rounds (`emittance_converged`).

Optional diagnostic planes record the first crossing of every ray (`planes=(axis, eta_values)`).

## Results

**Planar diode** (compare [`child_langmuir_diode`](../child_langmuir_diode); 400 rays, dt = 0.005, α = 1):

| J        | Rounds          | Max \|φ − φ_ref\| | Time-dependent PIC |
|----------|-----------------|--------------------|--------------------|
| 0.25 J_CL | 9              | 1.6e-6             | 3.0e-6             |
| 0.5 J_CL  | 9              | 3.4e-6             | 8.3e-6             |
| 0.9 J_CL  | 11             | 7.8e-6             | 1.3e-4             |
| 2 J_CL    | no convergence | —                  | limits the current to 1.02 J_CL |

- **Below the limit:** the iteration takes seconds instead of about a minute, and it is more
  accurate, because the rays are deterministic and it involves no time-dependent transients.
- **Above the limit:** each round alternates between a fully transmitted and a fully reflected
  beam. This is exactly the non-convergence the thesis describes for space-charge-limited
  emission. Time-dependent PIC remains the tool for that regime.

**Slit lens at 400 µA per mm of slit** (compare [`slit_lens_space_charge`](../slit_lens_space_charge);
2000 rays, 160 × 20 elements, p = 3, dt = 0.04, α = 1):

| Metric              | Steady-state iteration | Time-dependent PIC (steady window by emittance criterion) |
|---------------------|------------------------|------------------------------------------------------------|
| Rounds / cost        | 10 rounds, about 30 s  | 145 ns of beam time, about 25 s                            |
| Exit rms size        | 0.277 mm               | 0.278 mm                                                   |
| Exit rms emittance   | 6.74 mm·mrad           | 7.08 mm·mrad                                               |

- The exit emittance converges to 1e-6 relative in 6 rounds.
- The two envelopes agree; the PIC envelope carries marker noise.
- The PIC emittance is 5 % higher. This fits the grid- and noise-induced emittance
  growth of time-stepped PIC (Kalvas §5.9.1). The convergence study (`../convergence_study`)
  quantifies it.

## Limitations

- Serial only.
- The ray tracing uses a fixed time step (Strang: v/2, x, v/2); IBSimu uses an adaptive
  Cash–Karp integrator. The dt sweep of the convergence study sets the step. A per-ray adaptive step is available
  as an option (`diode_iteration(..., step_control=StepControl())`, see `struphy.pic.ray_tracing`): on the diode
  it matches the finest fixed step with about half the iterations. It gives no gain on the plasma extraction,
  so the fixed step is the default.
- There is no plasma model yet: the Boltzmann-electron term and the Newton iteration for
  plasma extraction are still to be added (plan phase 8).
