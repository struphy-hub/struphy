# Vacuum field convergence

![Vacuum field convergence](vacuum_field_convergence.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/vacuum_field_convergence/vacuum_field_convergence.py
```

This example checks `IonOpticsElectrostatic.solve_vacuum_potential` against
analytic potentials under h-refinement for p = 2, 3, 4. It prints tables of the
maximum errors in φ and in E = −∇φ; the field is taken from the model's `e_field`.

- **Coaxial capacitor** on the curved `HollowCylinder` mapping, with the
  electrodes on the inner and outer radial faces. The errors converge at the
  optimal rates: order p + 1 for φ and order p for E. This confirms the
  electrode constraint, the stiffness matrix on a curved mapping, and the
  gradient used by the particle pusher.
- **Slit immersion lens**, with the exact potential imposed on all four faces
  of a 30 mm × 10 mm window. The errors are measured in the beam region
  |y| ≤ h/2. They converge at roughly second order overall, with an irregular
  sequence and almost independently of p. The likely cause is the kinks of the
  linear-gap electrode trace at the gap edges: interpolating kinked Dirichlet
  data, plus the corner singularities, limit convergence. Rounded electrodes or
  a smooth gap profile should restore the higher-order rates (plan phases 4–5).

`test_vacuum_potential_converges_on_curved_mapping` in
`src/struphy/models/tests/test_ion_optics_electrostatic.py` is a fast regression
test of the coaxial case.
