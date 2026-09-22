# Slit immersion lens (zero current)

![Slit immersion lens](slit_immersion_lens.png)

From the repository root, using the installed Struphy environment:

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/slit_immersion_lens/slit_immersion_lens.py
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/slit_immersion_lens
```

The design is given in SI units in `Design`: 5 keV protons, plates at
y = ±5 mm, a 2 mm electrode gap at x = 35 mm, V₁ = 0 and V₂ = −10 kV. The design is
invariant in z. `IonOpticsUnits(length=1 mm, voltage=1 kV)` converts it to Struphy's
normalization. In that normalization, `phi` is in kV, lengths are in mm, and
`epsilon = 1` for the protons.

**Field.** The plates are electrode faces (`electrode_faces`). Their potential
is `PiecewiseLinearPotential`: V₁ upstream, V₂ downstream, and a linear ramp
across the gap. The model's `solve_vacuum_potential` solves Laplace's equation.
The domain ends (x = 0 and x = 80 mm) have the natural zero-normal-field
condition, so they act as mirrors.

**Beam.** 31 parallel rays (|y| ≤ 3 mm) are traced with `PushVinForceField`
and `PushEta` (Strang splitting). A ray is removed when it leaves the domain.

**Reference.** `analytic.py` gives the exact potential of this lens between
infinite plates, in closed form. The linear gap is averaged over analytically.
Rays integrated in this exact field with `DOP853` are the reference.

## Results (default settings: 320 × 40 elements, p = 3, dt = 0.02)

- The maximum ray deviation from the exact-field reference is 1.7 µm.
- The paraxial focus is at 56.09 mm; the reference gives 56.10 mm.
- Spherical aberration is clearly visible. The outer rays (|y| = 3 mm) cross
  the axis 2.4 mm upstream of the paraxial focus.
- The trace-space emittance grows from 0 to 1.75 mm·mrad through the lens,
  entirely due to aberration. Inside the accelerating gap, `y' = v_y/v_x` is not
  a conserved phase-space coordinate, so the emittance curve has a transient spike there.

Convergence with the number of elements (p = 3, dt = 0.02). Halving dt changes
the results by less than 5 %, so the error is set by the field, not the pusher:

| Elements (x × y) | Max. ray deviation | Max. axis-crossing error |
|------------------|--------------------|--------------------------|
| 160 × 20         | 6.1 µm             | 35 µm                    |
| 320 × 40         | 1.7 µm             | 11 µm                    |
| 640 × 80         | 0.6 µm             | 4.3 µm                   |

**Domain-size lesson (plan phase 3).** With the lens only 25 mm from the
upstream end, the mirror boundary at x = 0 shifted the entrance potential by
about 2.5 V. That capped the focus accuracy at about 60 µm, independent of the mesh.
Moving the lens to 35 mm (7 half-widths) removed this error floor.

## Limitations

- Linear-gap model: the electrode edges are not resolved geometrically.
  Resolving them needs face-segment Dirichlet conditions or a conforming
  mapping (plan phase 4).
- Zero current, and all rays are launched at once. There is no continuous
  injection (plan phase 7) and no space charge yet (plan phase 9).
- Rays that hit a plate are removed, but not yet counted per electrode.
