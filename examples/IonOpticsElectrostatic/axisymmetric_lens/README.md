# Axisymmetric two-tube lens (cylindrical symmetry)

![Axisymmetric lens](axisymmetric_lens.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/axisymmetric_lens/axisymmetric_lens.py   # about 75 s
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/axisymmetric_lens
```

## How cylindrical symmetry is represented

`struphy.geometry.axisymmetric.AxisymmetricElectrodeChannel` is a thin wedge of revolution:
a `PoloidalSplineTorus` whose poloidal plane is the meridional (r, z) half-plane of the
electrode system, revolved about the beam axis by 2π/N (N = 360 here, a 1° wedge).

- **Field.** The FEEC volume element is r dr dz dθ, so the ordinary weak Poisson problem on the
  wedge *is* the axisymmetric one. One element across the wedge represents the θ-independent
  potential exactly. No separate r–z solver is needed, and the direct solver, the Poisson–Boltzmann
  Newton solve and the charge deposition all work unchanged.
- **Symmetry boundaries.** The natural boundary on the wedge faces (η3) and on a thin axis
  cylinder r₀ = 0.01 mm (η2 = 0) gives ∂φ/∂θ = 0 and ∂φ/∂r = 0. The electrodes are
  `ElectrodeSegment`s on the outer wall (η2 = 1).
- **Particles.** Markers reflect on the axis cylinder and the wedge faces, and are removed
  at the wall and the ends: `BoundaryParameters(bc=("remove", ("reflect", "remove"), "reflect"))`.
  Per-side `(left, right)` boundary conditions are new in `Particles`. For an axisymmetric
  ensemble, reflection is exact because it is a symmetry of the field. A ray at radius r
  stands for the wedge fraction 1/N of its ring.

## Results (tubes at 0 / −30 kV, R = 5 mm, 2 mm gap, 5 keV protons)

- **Vacuum potential against the Bessel series.** Max error in r ≤ 0.8R: 0.37 → 0.091 → 0.022 kV
  (10 → 20 → 40 radial elements). This is second order, the same as the slit lens, and limited
  by the kinked linear-gap wall trace.
- **Focus against rays in the exact field.** The rays launched at r₀ = 0.5–3 mm cross the axis
  within 0.03 mm of the reference (45–47 mm, with visible spherical aberration). The innermost
  ray (r₀ = 0.25 mm) is off by 0.5 mm. That near-paraxial ray probes the region next to the
  axis cylinder, where the natural boundary only approximates the regularity condition at r = 0.
