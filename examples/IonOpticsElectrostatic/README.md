# IonOpticsElectrostatic examples

These examples track the progress of the ion-optics roadmap in
[`ION_OPTICS_SIMULATION_PLAN.md`](../../ION_OPTICS_SIMULATION_PLAN.md).
Each folder contains a script that writes its figure, a README with the results
and limitations, and pytest checks.

| Example | Plan phase | What it shows |
|---------|------------|---------------|
| [`planar_accelerator`](planar_accelerator) | 6 (pusher) | Constant-field acceleration matches the analytic motion to roundoff |
| [`vacuum_field_convergence`](vacuum_field_convergence) | 5 | Laplace solve against analytic potentials (coaxial, slit lens) under h-/p-refinement |
| [`slit_immersion_lens`](slit_immersion_lens) | 2, 5, 6 | **Ion beam plot**: 5 keV protons focused by a −10 kV slit lens (SI units), compared with exact-field rays |
| [`slit_lens_injection`](slit_lens_injection) | 4 (losses), 7 | **Steady-state beam** from continuous injection; current booked per electrode, compared with an exact-field Monte Carlo |
| [`child_langmuir_diode`](child_langmuir_diode) | 9 | **Space charge**: planar diode potential vs steady-state ODE; current saturates at the Child–Langmuir limit |
| [`sheet_beam_expansion`](sheet_beam_expansion) | 9 | **Space charge in 2D**: a uniform sheet beam expands as the analytic slab envelope |
| [`axisymmetric_beam_expansion`](axisymmetric_beam_expansion) | 9 | **Axisymmetric space charge**: a uniform round beam expands as the analytic cylindrical envelope |
| [`slit_lens_space_charge`](slit_lens_space_charge) | 9 | **Lens at increasing current**: space-charge defocusing moves the waist and grows the emittance |
| [`steady_state_iteration`](steady_state_iteration) | 7, 9 | **IBSimu-style steady-state iteration**: converges in about 10 rounds below the Child–Langmuir limit and oscillates above it; lens ray tracing compared with time-dependent PIC |
| [`convergence_study`](convergence_study) | 10 | Mesh, degree, ray-count and time-step convergence of the lens metrics at 0 and 300 µA/mm |
| [`bohm_sheath`](bohm_sheath) | 8 | **Plasma extraction building block**: ray-traced ions + Boltzmann electrons (convex Poisson–Boltzmann Newton) reproduce the Bohm sheath |
| [`plasma_extraction`](plasma_extraction) | 4, 8, 9 | **IBSimu-style slit extraction**: a plasma chamber with the plasma electrode and puller as mapped wall segments; the meniscus and beam come out of the Poisson–Boltzmann iteration |
| [`axisymmetric_lens`](axisymmetric_lens) | 3 | **Cylindrical symmetry**: an (r, z) wedge of revolution checked against the Bessel-series two-tube lens (potential and ray focus) |
| [`axisymmetric_extraction`](axisymmetric_extraction) | 3, 8 | **Round-aperture extraction** (IBSimu-style, cylindrical): meniscus and beam from the Poisson–Boltzmann iteration |

Run all checks:

```sh
.venv/bin/python -m pytest examples/IonOpticsElectrostatic src/struphy/models/tests/test_ion_optics_electrostatic.py
```
