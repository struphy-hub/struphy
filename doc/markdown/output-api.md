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

Use `keys()` to list discovered products, or `info()` for their short descriptions. Pass fields
and particle data to `evaluate()` as `species/variable`; use the reserved `"scalars"` name for
scalar histories.

```python
out.info()

for key in out.keys():
    print(key)
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

`evaluate()` always returns an xarray object: an `xarray.DataArray` for one field or particle
product, and an `xarray.Dataset` for `"scalars"`. Use xarray for selections, arithmetic,
reductions, plotting, and interoperability with other scientific Python packages.

```python
rho = out.evaluate("diagnostics/rho_xyz")
scalars = out.evaluate("scalars")
electric_energy = out.evaluate("scalars", variables="electric_energy")

# Last saved time and one vector component, selected by integer position
electric_field = out.evaluate("em_fields/E", t=-1, component=2)

# Select the nearest logical-coordinate plane
midplane = out.evaluate("diagnostics/rho_xyz", e3=0.5, method="nearest", drop=True)

# Select a time range
history = out.evaluate("scalars", variables="electric_energy", t=slice(100, None)).electric_energy
```

Integer `t` values select saved snapshot positions; a floating-point `t` selects a saved time
coordinate. Other keyword arguments select named xarray coordinates. Physical `X`, `Y`, and `Z`
coordinates describe the mapped logical grid; evaluating at an arbitrary physical point requires
interpolation or an inverse-coordinate map.

For raw FEEC fields, use the `species/variable` name. With no `eta` coordinates, evaluation uses
the full simulation grid at cell centres and includes `X`, `Y`, and `Z` coordinates. Providing
one or two eta coordinates makes a line or plane cut; unspecified directions use `0.5`.

```python
import numpy as np

phi = out.evaluate("em_fields/phi", t=-1)  # full 3-D grid
line = out.evaluate("em_fields/phi", eta1=np.linspace(0, 1, 200), t=-1)
```

Particle products use the same `species/variable` form. The default is the first matching binned
result, followed by a density/KDE result and then orbits. `info()` exposes the available choices;
`dataset=` selects one explicitly.

```python
out.info("kinetic_ions/f")
distribution = out.evaluate("kinetic_ions/f")
delta_f = out.evaluate("kinetic_ions/f", dataset="e1_v1_density/delta_f")
```

To make a figure, select the dimensions to show and call xarray's native `.plot()` methods (see
[Plot data](#plot-data)).

## Analyze and report data

`Output` has no fitting or error helpers: products are xarray objects, so such analysis is a few
lines of xarray and NumPy. Relative energy error, and an exponential growth or damping rate
fitted over a time window:

```python
energy = out.evaluate("scalars", variables="en_tot").en_tot
relative_error = (energy - energy.isel(t=0)) / energy.isel(t=0)

window = out.evaluate("scalars", variables="phi_integral").phi_integral.sel(t=slice(20.0, 60.0))
rate, log_amplitude = np.polyfit(window.t, np.log(np.abs(window)), 1)
```

For an oscillating signal such as the field energy in Landau damping, fit the local maxima rather
than the raw series. A field reduces to a time series with an xarray reduction over every
dimension except `t`.

```python
from scipy.signal import find_peaks

electric_energy = out.evaluate("scalars", variables="electric_energy").electric_energy
peaks, _ = find_peaks(electric_energy.values)
damping_rate, _ = np.polyfit(electric_energy.t[peaks], np.log(electric_energy[peaks]), 1)

rho = out.evaluate("diagnostics/rho")
rho_squared = (rho**2).mean([dim for dim in rho.dims if dim != "t"])
```

Fields carry mapped `X`, `Y`, `Z` coordinates; binned products (such as `e1_e2_density`) do not.
`with_physical_coords` attaches them by evaluating the run's domain on the array's logical grid.

```python
density = out.with_physical_coords("kinetic_ions/e1_e2_density/f").isel(t=-1)
radius = np.hypot(density.X, density.Y)
```

Write a compact data report with metadata. Add selected products to record their dimensions and
units.

```python
report = out.report("report", products=["en_tot", "diagnostics/rho_xyz"])
html_report = out.report("report", format="html")
```

`out.xarray` provides the complete lazy xarray `DataTree` when access to the grouped product
store is useful. Prefer `evaluate(key)` for normal single-product work.

## Reduce a distribution function

A binned distribution usually has more dimensions than a question needs. Averaging over the
logical space dimensions it has turns an `e1_v1` product into f(v1, t). A binned product keeps
only the dimensions of its slice, so select them from `data.dims`. The mean is uniform in the
logical coordinates, which is the volume average on a Cartesian domain; on a mapped domain it is
not weighted by the Jacobian.

Velocity moments are xarray reductions over a velocity dimension: the density, the mean velocity
and the variance, as functions of the remaining dimensions. In normalized units the variance is
the temperature divided by the mass. Mean and variance are NaN where the density is not positive;
for a `delta_f` product only the density (its perturbation) is meaningful.

```python
data = out.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
space = [dim for dim in ("e1", "e2", "e3") if dim in data.dims]
f_of_v = data.mean(space)

bin_width = data.v1.differentiate("v1")
density = (data * bin_width).sum("v1")
mean_v1 = (data * data.v1 * bin_width).sum("v1") / density
temperature_over_mass = ((data * (data.v1 - mean_v1) ** 2 * bin_width).sum("v1") / density).mean(space)
```


## Convert to SI units

Products are in the normalization of the model, whose units are `out.units`. `to_si` converts
the coordinates of a product whenever they are present: time `t` to seconds, the mapped `X`, `Y`,
`Z` to meters and the velocities `v1`, `v2`, `v3` to m/s. Values are converted only when `unit`
names the unit that the variable was normalized with, because a product does not record it:
`"x"`, `"B"`, `"n"`, `"v"`, `"t"`, `"p"`, `"rho"`, `"j"` or `"kBT"`. For a composite unit, pass a
number and its `label`. The original product is not modified, and converting coordinates twice
changes nothing.

```python
print(out.units.x, out.units.v, out.units.t)   # meters, m/s and seconds per unit

f_si = out.to_si("kinetic_ions/e1_v1_density/f")       # coordinates only: v1 in m/s, t in s

# values, for a variable that the model normalizes with the unit B
b_si = out.to_si("em_fields/b_field", "B")             # in tesla

# values, for a quantity normalized with a product of units
flux_si = out.to_si(flux, out.units.n * out.units.v, label="m^-2 s^-1")
```

## Profile a run

A run started with `sim.run(profiling_activated=True)` writes `profiling_data.h5`, and
`out.profile` reads it. Regions are the setup steps, every propagator (`prop: ...`), pusher,
accumulation, compiled kernel (`kernel: ...`) and linear solve. Regions nest, so the time of a
region includes the regions it calls; times of different regions must not be added up.

`summary()` returns an xarray dataset along `region` with `calls` and `total_time` (per rank),
`mean_time`, `min_time` and `max_time` (per call, in seconds) and `fraction`, the share of the run.
It is sorted by `total_time` by default. `table()` prints the same as text.

```python
print(out.profile.table(top=10))

kernels = out.profile.summary(prefix="kernel:", sort_by="total_time")
kernels.total_time.to_series()
```

`compare()` puts one statistic of several runs side by side, with NaN for a region that a run
does not have. It accepts `Output` objects or `Profile` objects.

```python
times = out.profile.compare(other_out, metric="total_time", prefix="prop:")
ratio = times.isel(run=1) / times.isel(run=0)
```

For anything not covered here, `out.profile.results` is the full `scope_profiler` result.

## Plot data

Use the native xarray plotting methods after making the intended selection.

```python
out.evaluate("scalars", variables="electric_energy").electric_energy.plot.line(x="t")

rho = out.evaluate("diagnostics/rho_xyz", t=-1)
rho.isel(e3=rho.sizes["e3"] // 2).plot(x="e1", y="e2")
```

xarray squeezes size-one dimensions before plotting, so an array that is 2-D on a grid with one
cell in some direction plots as a line. Select until the array has the dimensions the plot needs.

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
