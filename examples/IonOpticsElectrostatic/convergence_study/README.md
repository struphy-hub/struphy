# Convergence study (plan phase 10)

![Convergence study](convergence_study.png)

```sh
# quick check: 7 steady runs at 300 µA/mm, about 4 min
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/convergence_study/convergence_study.py
# full study: all sweeps at 0 and 300 µA/mm plus 4 time-dependent runs, about 30 min
# (results already cached in convergence_study.json; add --replot to only redraw)
MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/convergence_study/convergence_study.py --full
```

This is the slit lens of [`slit_lens_space_charge`](../slit_lens_space_charge), solved with the
steady-state iteration. At 400 µA/mm the waist leaves the 80 mm domain, so the study uses
300 µA/mm. The base case is 160 × 20 elements (20 across the 10 mm slit), p = 3, 2000 rays,
dt = 0.04 and α = 1. One parameter is refined at a time.

## Results at 300 µA per mm of slit (relative to the finest value of each sweep)

| Parameter              | Values                | Exit emittance | Exit rms size | Waist position | Waist size |
|------------------------|-----------------------|----------------|---------------|----------------|------------|
| Elements across slit    | 10 / **20** / 40 / 80 | +7.4 / **+0.9** / +0.05 % | +8.9 / **+1.5** / +0.3 % | −0.75 / **−0.25** / 0 mm | +4.7 / **+0.4** / 0 % |
| Spline degree p         | 2 / **3** / 4         | 0.2 % spread   | ±2 %          | ±0.25 mm       | < 0.1 %    |
| Rays                    | 500 … 8000            | < 0.2 %        | < 0.2 %       | ±0.25 mm       | < 0.4 %    |
| Time step dt            | 0.08 … 0.01           | < 0.2 %        | < 1.3 %       | 0              | < 0.1 %    |

- **Converged values:** exit emittance 5.18 mm·mrad, exit rms size 0.299 mm, waist at
  72.25 mm with an rms size of 0.168 mm.
- **At zero current,** every metric changes by less than 0.4 % over all sweeps.
- **The mesh dominates the error,** as Kalvas (§5.9.3) also found. With 20 elements across
  the slit (4 across the 3 mm beam half-width), the error is about 1 %; with 40 elements it
  is below 0.3 %. Rays (500 is already enough) and dt barely matter at these settings.
- **Time-dependent PIC** (4 runs: base, finer mesh, 2× markers, dt/2) gives an exit
  emittance of 5.40–5.73 mm·mrad, 4–11 % above the converged steady state. Its waist
  scatters between 70 and 77 mm, and its waist sizes (0.10–0.16 mm) come from sparse
  slices of live markers. Its results are dominated by marker noise, consistent with the
  noise-driven emittance growth of Kalvas §5.9.1.

## Recommended settings

- For design runs of this kind of lens: use the steady-state iteration with at least 40
  elements across the slit, p = 3, 2000 rays, dt = 0.04 and α = 1. The computational error is
  then below 0.3 %, well under the "few percent" error budget of the thesis.
- For quick parameter scans, 20 elements across the slit give errors of about 1–1.5 %.
- Use time-dependent PIC only where the steady-state iteration cannot run: above the
  space-charge limit, and for transients.
