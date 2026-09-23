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

For oscillating signals such as the field energy in Landau damping, `damping_rate` fits the
exponential to the local maxima (`envelope`) instead of the raw series. `norm` reduces a field
to a time series (by default over every dimension except `t`), which can then be fitted.

```python
damping = out.damping_rate("electric_energy", window=(0.0, 8.0), amplitude=True)
peaks = out.envelope("electric_energy")

growth = out.growth_rate(out.norm("diagnostics/rho", squared=True), amplitude=True)
```

Fields carry mapped `X`, `Y`, `Z` coordinates; binned products (such as `e1_e2_density`) do not.
`with_physical_coords` attaches them by evaluating the run's domain on the array's logical grid.

```python
density = out.with_physical_coords("kinetic_ions/e1_e2_density/f").isel(t=-1)
radius = np.hypot(density.X, density.Y)
```

Write a compact data report with metadata and the product catalog. Add selected products to
record their dimensions and units.

```python
report = out.report("report", products=["en_tot", "diagnostics/rho_xyz"])
html_report = out.report("report", format="html")
```

`out.xarray` provides the complete lazy xarray `DataTree` when access to the grouped product
store is useful. Prefer `evaluate(key)` for normal single-product work.

## Fourier spectra and temporal filtering

Fourier diagnostics operate on a saved product name or a selected `DataArray`.
They run on demand; no FFT option on `pproc()` and no legacy pickle files are
needed. Select a component, slice or time interval before transforming when you
do not need the full field: the selected values are loaded into memory.

```python
velocity = out.evaluate("mhd/velocity_xyz").isel(e3=0)
spectrum = out.time_fft(velocity)
coefficients = spectrum.coefficients       # complex, t replaced by omega
power_at_each_point = spectrum.power       # one-sided mean-square power per bin
plane_power = spectrum.power.mean(("e1", "e2"))

# Optional mean subtraction and a periodic Hann taper for spectral inspection
tapered = out.time_fft(velocity, detrend=True, window="hann")

# A two-sided FFT along any uniform numeric coordinate, including complex data
initial = out.evaluate("mhd/velocity").isel(t=0, component=0, e1=4, e3=0)
initial = initial.isel(e2=slice(None, -1))  # if the grid includes both periodic endpoints
modes = out.fft(initial, dim="e2")         # complex coefficients, coordinate k_e2
m = modes.k_e2 / (2 * np.pi)              # eta2 has period 1
```

All transforms divide coefficients by the sample count `N`. `time_fft` doubles
positive-frequency **power**, except DC and the even-length Nyquist bin. Thus
`spectrum.power.sum("omega")` equals the time mean square of the input. With a
window or mean subtraction it equals the mean square of that processed signal;
no window-amplitude correction is applied. Power is per bin, not a density per
unit frequency, and a sum over spatial samples is not a volume-weighted energy.

Frequencies are angular: `omega = 2*pi*f`, with units inverse to the supplied
time coordinate (`rad / s` when time is in seconds). Sampling comes from the
saved times, including `save_step`/post-processing downsampling, not the original
solver time step. `frequency_resolution`, `nyquist_frequency`, `n_samples` and
`sample_spacing` are recorded in the result attributes. At least two finite,
strictly increasing, uniformly spaced samples are required. Irregular grids
(including an off-cadence final sample), NaNs and complex input to the one-sided
time transform are rejected. The general `fft` supports complex data. Duplicate
spatial endpoints must be removed explicitly; the temporal last sample is retained.

To reproduce the dominant-frequency filtering workflow from the TAE example:

```python
result = out.filter_time(velocity, dims=("e1", "e2"), omega_min=1e-8, pad_bins=0)
filtered_velocity = result.filtered       # original dimensions, coordinates and units
bands = result.spectrum                   # reduced power, dominant_frequency, omega_lo/hi
print(bands[["dominant_frequency", "omega_lo", "omega_hi", "has_peak"]])

# Wider band, recomputed in memory without rerunning the simulation or pproc
wider = out.filter_time(velocity, pad_bins=2)
```

Power is summed over `dims` to choose a contiguous full-width-at-half-maximum
band for each remaining dimension. The default reduces all dimensions except
`t` and `component`, so every component has one band shared across space.
`dims=()` selects a band independently at every point. Padding adds bins on
either side but never includes frequencies below the positive `omega_min`.
Filtering uses an untapered transform and preserves neither the DC offset nor
unselected modes. A zero or constant component returns zero filtered values,
`has_peak=False`, NaN frequency bounds and indices of −1. No peak is invented
for it. Short records, off-bin frequencies and abrupt band edges can produce
leakage and ringing: a strongest bin does not by itself identify an eigenfrequency.

`inverse_time_fft(spectrum.coefficients, velocity)` from
`struphy.post_processing.time_fft` reconstructs a time series, using the template
to retain the original coordinates and distinguish odd/even lengths. Windowing
and mean subtraction are not undone. The same diagnostics are available as
`array.struphy.analysis.fft(...)`, `.time_fft(...)` and `.filter_time(...)`.

## Reduce a distribution function

A binned distribution usually has more dimensions than a question needs. `spatial_average`
averages over the logical space dimensions `e1`, `e2` and `e3` (or the ones passed as `dims`),
so an `e1_v1` product becomes f(v1, t). The mean is uniform in the logical coordinates, which is
the volume average on a Cartesian domain; on a mapped domain it is not weighted by the Jacobian.

`velocity_moments` integrates over the velocity dimensions instead and returns a dataset with the
`density`, and the mean `mean_v1` and variance `variance_v1` along every velocity direction, as
functions of the remaining dimensions. In normalized units the variance is the temperature divided
by the mass. Mean and variance are NaN where the density is not positive, and a `delta_f` product
has only the density (its perturbation).

```python
f = "kinetic_ions/e1_v1_density/f"

f_of_v = out.spatial_average(f)              # dimensions (t, v1)
moments = out.velocity_moments(f)            # density, mean_v1, variance_v1 over (t, e1)
temperature_over_mass = out.spatial_average(moments.variance_v1)
```

Both are also available on any product as `array.struphy.analysis.spatial_average()` and
`array.struphy.analysis.velocity_moments()`.

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
