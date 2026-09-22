# Planar Bohm sheath (plasma extraction building block)

![Bohm sheath](bohm_sheath.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/bohm_sheath/bohm_sheath.py   # about 40 s
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/bohm_sheath
```

Plasma extraction needs a compensating plasma in the field solve. Here that is thermal
electrons with the Boltzmann density ρ_e = −ρ_e0 exp((φ − φ_P)/T_e), given by
`struphy.physics.plasma_models.BoltzmannElectrons`. Pass it as
`IonOpticsElectrostatic(plasma=...)`. The ions are traced as before, entering from the
plasma at or above the Bohm velocity.

## Formulation

The electron term makes Poisson's equation nonlinear. Instead of a finite-difference
Newton with lumped nodes and damping heuristics (Kalvas 2013, §5.4), Struphy uses the
variational form. For a fixed ion charge q, the potential minimizes the strictly convex energy

E(φ) = ½∫|∇φ|² − Σᵢ qᵢφᵢ + ∫ ρ_e0 T_e e^{(φ−φ_P)/T_e} dx

over the spline coefficients, with the electrode values fixed:

- **Exact Galerkin terms.** The electron terms are integrated by Gauss quadrature of φ_h
  (`L2Projector.get_dofs`), not lumped.
- **Hessian.** It is the stiffness matrix plus the weighted mass matrix
  M⁰[ρ_e0/T_e · e^{(φ_h−φ_P)/T_e}] (`create_weighted_mass`), assembled into a sparse matrix
  by probing.
- **Newton with a line search on E.** This converges globally to the unique solution because E is
  convex. No initial guess of the plasma region or meniscus is needed. In practice each solve takes 2–6
  quadratically converging iterations.

The outer loop is `SteadyStateIteration` with `criterion="potential"`: it stops when the
relative change of φ between rounds is below `tol`. Two further improvements to the ray deposit:

- A ray leaving during a step contributes the trapezoidal charge of the exact time it
  spent inside, so the deposit stays continuous in the field.
- Exit records are moved to the exact exit point.

## Test case

The domain spans 20 Debye lengths. The plasma edge (x = 0) is at φ = 0, and the wall is at
φ_W = −10 kT_e/e. The units are natural (λ_D, kT_e/e, n₀), so ρ_e0 = T_e = v_B = 1. The ions
enter at v₀ with the quasi-neutral flux n₀v₀. The reference is `solve_bvp` applied to Kalvas
eq. 2.23, φ'' = −(v₀/√(v₀² − 2φ) − e^φ).

| v₀     | Rounds to 1e-6 | Max \|φ − φ_ref\| (kT_e/e) |
|--------|----------------|------------------------------|
| 1.0 v_B | 49            | 9.7e-5                       |
| 1.3 v_B | 26            | 2.4e-4                       |

(80 elements, p = 3, 200 rays, dt = 0.02, α = 1.)

- Injection exactly at the Bohm velocity is the marginal case of the sheath criterion: the
  outer iteration converges linearly and slowly, with transient bumps.
- Anderson acceleration of the charge (`anderson=5`, experimental) halves the rounds to reach
  1e-2, but then stalls around 1e-4.
- Nothing is reflected: all the ion current reaches the wall.

## Not yet included

- **2D/3D extraction:** a plasma electrode with an aperture and a meniscus. The solver parts are
  in place; this needs aperture geometry (plan phase 4), because the plasma must sit behind
  a plasma electrode inside the domain.
- **Negative-ion extraction models** (Kalvas §5.2.2).
- **MPI:** the Poisson–Boltzmann solve uses the serial direct solver.
