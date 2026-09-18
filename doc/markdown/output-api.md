# Working with simulation output

`Output` is a lightweight handle to one completed Struphy run. It reads saved metadata when it
is created, but does not load field or particle data until requested.

```python
from struphy import Output

out = Output("path/to/run")
```

When working directly after a simulation, `Simulation.run()` returns the same kind of object:

```python
out = sim.run()
```

## Discover available data

Use `keys()` to list the names accepted by `evaluate()`, or `info()` for the same names with
short descriptions.

```python
print(out.info())

for key in out.keys():
    print(key)

# Machine-readable product metadata
catalog = out.catalog(details=True)
```

`keys()` and `info()` do not load product arrays. If post-processed products do not exist yet,
they materialize them using default post-processing options. Call `pproc()` first when those
options matter.

## Materialize post-processing products

Use `pproc()` explicitly to choose how fields and particle diagnostics are generated.

```python
out.pproc(
    physical=True,       # also create physical field components and coordinates
    step=1,              # use every saved time step
    celldivide=1,
)
```

Matching existing products are reused. `evaluate()` also calls `pproc()` automatically for a
missing non-scalar product when running serially.

## Evaluate data

`evaluate()` returns an ordinary `xarray.DataArray`. Use xarray for selections, arithmetic,
reductions, and interoperability with other scientific Python packages.

```python
rho = out.evaluate("diagnostics/rho_xyz")
phi = out.evaluate("phi_integral")

# Last saved time and one vector component, selected by integer position
electric_field = out.evaluate("em_fields/E", isel={"t": -1, "component": 2})

# Select the nearest logical-coordinate plane
midplane = out.evaluate(
    "diagnostics/rho_xyz",
    sel={"e3": 0.5},
    method="nearest",
    drop=True,
)

# Select a time range
history = out.evaluate("phi_integral", isel={"t": slice(100, None)})
```

`isel` uses integer positions and `sel` uses named dimension-coordinate values. Selections are
applied in that order. Physical `X`, `Y`, and `Z` coordinates describe the mapped logical grid;
evaluating at an arbitrary physical point requires interpolation or an inverse-coordinate map.

To return only values, without xarray coordinates and attributes, use `as_numpy=True`.

```python
rho_values = out.evaluate("diagnostics/rho_xyz", isel={"t": -1}, as_numpy=True)
```

For domains with an analytical inverse map, evaluate a field at a physical point directly:

```python
value = out.evaluate(
    "em_fields/phi_xyz",
    physical={"X": 1.0, "Y": 0.0, "Z": 0.2},
)
```

## Analyze and report data

Numerical helpers stay on `Output` and return values or xarray arrays rather than figures.

```python
fit = out.growth_rate("phi_integral", window=(20.0, 60.0), amplitude=True)
energy_error = out.relative_error("en_tot")
energy_drift = out.drift("en_tot")
```

Write a compact data report with metadata and the product catalog. Add selected products to
record their dimensions and units.

```python
report = out.report("report", products=["en_tot", "diagnostics/rho_xyz"])
html_report = out.report("report", format="html")
```

`out.xarray` provides the complete lazy xarray `DataTree` when access to the grouped product
store is useful. Prefer `evaluate(key)` for normal single-product work.

## Plot data

Struphy-aware plotting is performed by `Output`, not by modifying xarray arrays. Rendering
methods return `(fig, ax)` (or `(fig, axes)` for panels), so normal Matplotlib controls display,
saving, and further customization.

```python
from matplotlib import pyplot as plt

fig, ax = out.timeseries("phi_integral", fit=(0.0, None), fit_amplitude=True)
ax.set_title("Potential growth")

fig, ax = out.viewer(
    "diagnostics/rho_xyz",
    x="e1",
    y="e2",
    coords="physical",
    plane="RZ",
)

fig, axes = out.panels("kinetic_ions/e1_v1_density/f", x="e1", y="v1")
fig, ax = out.trajectories("kinetic_ions", max_markers=1000)

plt.show()
```

Plotting methods also accept a derived `DataArray` instead of a saved-product name.

```python
rho_last = out.evaluate("diagnostics/rho_xyz", isel={"t": -1})
fig, ax = out.slice(rho_last, x="e1", y="e2", coords="physical", plane="RZ")
```

The available product plotting methods are `timeseries`, `slice`, `panels`, `viewer`,
`animation`, and `trajectories`. `view` creates a reusable view configuration, while `frames`
writes PNG files and returns their paths. Whole-run plots are `plot_scalars` and `equilibrium`.

## MPI post-processing

`Output` always uses `MPI.COMM_WORLD`; no communicator is passed to its constructor.

For serial post-processing under MPI, call `pproc()` on every rank. Rank 0 does the work and the
other ranks wait at the synchronization barrier.

```python
out.pproc(physical=True)
```

For parallel post-processing, also call it on every rank and pass `parallel=True`. The current
world communicator must have the same number of ranks as the run that wrote the raw output.

```python
out.pproc(parallel=True, physical=True)
```

Automatic materialization through `evaluate()` is intentionally disabled when more than one MPI
rank is active. Call `pproc()` explicitly first in that case.
