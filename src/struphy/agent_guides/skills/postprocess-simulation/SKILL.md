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

## Common pitfalls

- Don't re-run or re-import the simulation to get results; open the output folder.
- Raw output must exist (`data/data_proc0.hdf5`), otherwise `evaluate` raises `FileNotFoundError`.
- Time is normalized by default; `out.with_time_units("physical")` gives a view with `t` in seconds, and
  `out.to_si(product, unit)` converts values (`unit` one of `x`, `B`, `n`, `v`, `t`, `p`, `rho`, `j`, `kBT`;
  `out.units` holds the model's units).
