---
name: postprocess-simulation
description: Use when reading, post-processing, plotting or analysing the output of a Struphy simulation (a folder with run_metadata.json and data/) - writing pproc_*.py scripts or notebooks with the Output API, evaluating fields, scalars, distributions or orbits as xarray objects, or using `struphy output`. Triggers on requests like "plot the results", "post-process the run", "load the simulation output", "compare two runs".
---

# Post-processing a Struphy simulation

Every run writes an output folder (`sim.env.path_out`, by default `sim_1/` next to where it was
launched) with `run_metadata.json` and the raw data in `data/`. The `Output` class is a lazy
handle to that folder: it reads metadata on construction and loads data only on request.
All products are labelled xarray objects.

This skill ships with the installed Struphy package, so it matches the installed version.

## Getting an `Output`

```python
out = sim.run()                 # in the params script: run() returns the Output
out = sim.output                # same object, after a run

from struphy import Output      # in a separate script or notebook, from the folder alone
out = Output("sim_1")
```

`Output` never creates or holds a `Simulation` (there is no `out.sim`), and post-processing
scripts must not import the params file: model, domain, grid and options are rebuilt from
`run_metadata.json` (`out.model`, `out.domain`, `out.grid`, `out.derham_opts`, `out.time_opts`).
In a `pproc_<name>.py` next to the params file, build the default path relative to `__file__`
(`Path(__file__).resolve().parent / "sim_1"`) and keep a command-line override. In notebooks, use an
explicit path.

## Finding what is there

```bash
struphy output info sim_1       # evaluable products with descriptions
struphy output keys sim_1       # one key per line
struphy output report sim_1     # markdown (or --format html) data report
```

In Python: `out.info()`, `out.keys()`, `out.info("species/variable")` (particle datasets for one name).

## Evaluating products

```python
out.scalars                                    # xarray.Dataset of all scalar time series, read from raw data
out.evaluate("scalars", variables="en_tot")

phi = out.evaluate("em_fields/phi", t=-1)      # raw FEEC field on the grid's cell centres, last snapshot
line = out.evaluate("em_fields/phi", eta1=np.linspace(0, 1, 100), eta2=0.5, eta3=0.5, t=-1)
plane = out.evaluate("em_fields/phi", t=-1, eta3=0)          # the (eta1, eta2) plane
b_norm = out.evaluate("em_fields/b_field", representation="norm")

f = out.evaluate("kinetic_ions/f")             # first binned distribution of that species
f = out.evaluate("kinetic_ions/e1_v1_density/f")             # a specific dataset (full key from keys())

out.fields.<species>.<variable>                # attribute access to the same products
out.distributions, out.densities, out.orbits
```

- Names are `"species/variable"`, `"species/dataset/variable"` or `"scalars"`; take them from `out.keys()`
  instead of guessing.
- `t`: an int or float selects one snapshot and drops the `t` dimension; lists and slices keep it;
  omitted means all saved steps. `out.time` holds the time grid, `out.standard_snapshots()` the
  first/middle/last indices used for standard figures.
- For raw FEEC fields, `eta1`/`eta2`/`eta3` evaluate the spline at those logical coordinates (omitted
  directions stay on the cell centres). For all other products they select saved coordinates.
- Arrays carry the physical coordinates `X, Y, Z`, so `phi.plot(x="X", y="Y")` plots in physical space.
- Use xarray for everything else: `.isel(t=-1, component=0)`, `.sel(t=0.35, method="nearest")`,
  `.plot()`, `.plot.line(x="t")`, `.to_numpy()`.

## Explicit post-processing and MPI

Products other than scalars and raw FEEC fields are post-processed on first use with default options.
Call `out.pproc(...)` only to choose other options (all keyword arguments, see `help(out.pproc)`):

```python
out.pproc(step=2, celldivide=2, physical=True, create_vtk=True, vtk_steps=[0, -1])
```

or from the shell: `struphy output pproc sim_1 [--physical]`.

Under MPI, processing is collective: call `out.pproc()` and every `out.evaluate()` on **all** ranks,
not inside `if rank == 0:`; restrict only printing and plotting to rank 0. Processing runs in parallel when
the job has as many ranks as the saved run (`out.mpi_ranks`), otherwise serially on rank 0.

## Plots and diagnostics

With `pip install "struphy[pproc]"` (package plasma-plots), creating an `Output` adds
`out.plot`, `out.analysis` (e.g. `out.plot.energies()`, `out.analysis.time_fft("em_fields/phi")`) and
`.plasma.plot` / `.plasma.analysis` on every product (e.g. `phi.plasma.plot.slice(x="eta1", y="eta2", t=-1)`).
`import plasma_plots; help(plasma_plots)` gives an overview.

To compare runs: `Output.compare(out_a, out_b, "em_fields/phi")` returns an `xarray.Dataset`.

## Videos of fields

`struphy.post_processing.frames` writes the frames of a video of a raw FEEC field (one PNG per saved time step,
at most `max_frames`, evenly spaced; the same color scale, view and image size in all frames), evaluated from the
saved spline coefficients (no `pproc` needed):

```python
from struphy.post_processing.frames import frames_to_video, write_3d_frames, write_slice_frames, write_videos

# radial component (normalized basis) in the poloidal plane of a torus or cylinder
write_slice_frames(out, "mhd/velocity", "frames/u_r_rz", component=0, label="u_r")
# 3D with a cutaway; view="cut" (camera on the cut face) or "domain" (whole domain)
write_3d_frames(out, "mhd/velocity", "frames/u_r_3d", component=0, view="domain", label="u_r")
frames_to_video("frames/u_r_rz")  # frames/u_r_rz.mp4, needs ffmpeg
# or all standard videos at once: frames/u_r_rz (or u_r_xy, u_r_xz, u_r_yz for a slab), frames/u_r_3d_cut, frames/u_r_3d
write_videos(out, "mhd/velocity", "frames", component=0, prefix="u_r", label="u_r")
```

Options: `component=None` for the magnitude of a vector field, `basis` (`"norm"` default, `"1"`, `"2"`, `"v"`,
`"physical"`), `plane` (`"rz"` for tori and cylinders, `"xy"`, `"xz"`, `"yz"`; for other domains than Cuboid,
Colella, Orthogonal, cylinders and tori the logical mid-surfaces `"eta3"`, `"eta2"`; `"rz"` for other tori such
as Tokamak), `clim` (None: one
scale over all frames, `"frame"`, or `(vmin, vmax)`), `steps`, `max_frames` (default 500). The titles show the time
as `t = <t>/<end time>`. The frames go to `frame_0000.png`, ... with `frames.txt` (step and time per frame); they are
written on rank 0 only.

## Frequencies and mode structures

`struphy.post_processing.mode_analysis` analyses waves and eigenmodes in tori and cylinders (eta1 radial, eta2
poloidal, eta3 toroidal or axial): Fourier coefficients in the angles at all saved times, a windowed FFT in time at
every radius, the peak frequency and the mode structure filtered at that frequency.

```python
from struphy.post_processing import mode_analysis as ma

coeffs = ma.mode_coefficients(out, "mhd/velocity", ms=(0, 1, 2, 3), n=-1, component=0)  # dims (t, eta1, m), coord r
spectrum = ma.time_spectrum(coeffs)                    # dims (omega, eta1, m), attrs resolution = 2 pi / T
omega = ma.peak_frequency(spectrum, m=1, r=0.5)
profiles = ma.filtered_profile(spectrum, omega=omega)  # or band=(lo, hi)
ma.plot_omega_r(spectrum, out=out, expected=0.067).fig.savefig("omega_r.png")  # with shear Alfven continua
```

`n` is the logical mode number along eta3 (physical n = n * tor_period). The default `representation="norm"` gives
the physical radial component for circular cross-sections (component 0). Compare the mode structure of runs with
different resolution with `filtered_profile`, not with snapshots: snapshots also contain continuum oscillations
whose radial scale shrinks in time. The continua need an equilibrium with `q_r` (e.g. AdhocTorus, ScrewPinch).
The spectrum needs at least 4 saved times; the frequency resolution is 2 pi / T.

## Common pitfalls

- Don't re-run or re-import the simulation to get results; open the output folder.
- Raw output must exist (`data/data_proc0.hdf5`), otherwise `evaluate` raises `FileNotFoundError`.
- Time is normalized by default; `out.with_time_units("physical")` gives a view with `t` in seconds, and
  `out.to_si(product, unit)` converts values (`unit` one of `x`, `B`, `n`, `v`, `t`, `p`, `rho`, `j`, `kBT`;
  `out.units` holds the model's units).
