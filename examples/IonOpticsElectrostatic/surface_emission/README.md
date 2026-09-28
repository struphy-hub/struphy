# Surface emission from the aperture lips

![Surface-produced ion trajectories](surface_emission_beam.png)

![Surface emission](surface_emission.png)

```sh
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/surface_emission/surface_emission.py   # a few minutes
.venv/bin/python -m pytest examples/IonOpticsElectrostatic/surface_emission                           # coarse, about 10 s
```

The slit extraction of [`plasma_extraction`](../plasma_extraction) with a second ion source: ions emitted
from the plasma-electrode surface, the way surface-produced H⁻ ions are in cesiated negative-ion sources
(Kalvas 2013, §2.3). `RayBundle.from_wall_surface` places quasi-random rays on a chosen part of a channel wall
(here the sloped and the flat part of both aperture lips, x = 2–3 mm), uniform in arc length, and launches them
along the inward wall normal with a fixed energy and the cosine (Lambert) angular law; every ray carries the same
current, `j_s × arc length × width / n`. The bundle is concatenated to the plasma-volume rays (`RayBundle.__add__`)
and the whole set is traced in the steady-state iteration, so both populations share the space charge and the
Poisson–Boltzmann field. Exit records now carry the ray id, which is how the outlet current is split by origin.

| Part | Setup |
|------|-------|
| Volume source | as in `plasma_extraction`: Bohm-speed protons with the quasi-neutral flux over the chamber height, 1800 rays |
| Surface source | 600 rays on both lips, j_s = 20 A/m² (2 mA/cm²; `surface_current_density`), 1 eV, cosine law |
| Species | **the same protons for both sources** (a start; see the limitations) |

## Result

The first figure (`surface_emission_beam.png`) is the beam picture: the whole channel on top and the aperture region
below, with every surface-produced (SP) trajectory coloured by where the ion ends, and its launch point as a dot.
SP ions born on the plasma-facing slope (blue) are turned back by the sheath within a few micrometres, so only their
launch dots are visible; ions from the aperture edge (purple) cross the axis in the gap and land on the puller; the
few red ones from the innermost edge are extracted. The second figure adds the exit phase space by origin.

With j_s = 20 A/m² the iteration converges in 52 rounds (about 2 min). The surface ions leave the lips at 0 V
into the extraction field, converge strongly (they start on the two lip surfaces and cross the axis in the gap),
and almost all of them end on the puller: only 0.2 % of the total emitted current reaches the outlet from the
surface, against 25.9 % from the plasma volume; of the SP current itself, 65 % returns to the plasma electrode
(the slope), 33 % ends on the puller and 2 % is extracted. Their space charge in the aperture is what matters: it pushes the
meniscus back and lowers the volume transmission from 29.9 % (`plasma_extraction`) to 25.9 %.

At j_s = 100 A/m² (10 mA/cm², a typical cesiated-surface H⁻ yield) the surface current is comparable to the whole
plasma flux and, being slow ions, deposits a large charge at the lips: the volume transmission drops to about 17 %
(0.8 % from the surface), and the iteration is noise-limited (residual 0.1–0.2 with the damping at its floor,
potential change 5e-6) because the fate of lip rays at the edge flips between rounds. Quote such runs with
`iteration.averaged(10)`.

## Limitations

- **Species and sign.** Surface-produced ions are negative in a real source; here both populations are protons,
  so the surface ions born on the plasma-facing slope see the sheath as a barrier (they return to the wall)
  instead of being pulled into the plasma, and the electrons are still the Boltzmann electrons of a positive-ion
  plasma. A negative-ion version needs the charge sign in the Poisson coupling and the plasma model of Kalvas
  §2.4 (positive ions plus electrons compensating the H⁻ space charge); the source itself does not change.
- **Emission model.** Fixed energy and a cosine angular law; no energy spread, no dependence of the yield on the
  local field or on the angle of the surface.
- **Gap boundary, 2D slit, serial only**, as in `plasma_extraction`.
