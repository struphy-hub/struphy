# Struphy geometry in the browser

`struphy-web` is a pure-Python wheel built from this repository's shared
geometry sources. Import it as `struphy_web`. It can coexist with full
Struphy and retains cuNumPy on its NumPy backend. No Pyccel compilation, MPI,
FEEC installation, GPU, or computation server is required.

## Run the viewer

From the repository root (Python >= 3.11 and Node.js are build tools only):

```sh
npm ci --prefix browser
python3 browser/prepare.py
python3 -m http.server 8765 --directory browser/site
```

Open **http://localhost:8765**. Choose a domain, edit its parameters, and click
**Construct domain**. Drag to rotate the sampled surface. The probe displays
physical coordinates, the Jacobian, determinant, and metric. Choose `Tokamak`
to construct flux-aligned geometry from circular flux or an uploaded G-EQDSK
file. **Save geometry** and **Open geometry** exchange `.npz` files.

The preparation step downloads the pinned Pyodide runtime and dependencies,
verifies downloaded package hashes, and copies them into `site/vendor`.
After preparation, the site uses only same-origin static assets. All geometry
calculations and file parsing run in a browser Web Worker. Files are not
uploaded to a computation server. Any static HTTP(S) host can serve `site/`;
opening `index.html` directly through `file://` is not supported. Vendored
assets are generated and ignored by Git. A fresh load still needs access to
the static host; this is not a service-worker offline-installable application.

## Use the Python API

Build just the wheel with `python3 browser/build.py`. Install the resulting
`browser/dist/struphy_web-<version>-py3-none-any.whl` in Python, or install
its hosted URL with `await micropip.install(...)` in Pyodide. The runtime
requires NumPy, SciPy, and cuNumPy; the viewer pins their tested distribution.

```python
import cunumpy as xp
from struphy_web import domains, CircularFlux, EQDSKFlux
from struphy_web import save_geometry, load_geometry, sample_surface

xp.set_backend("numpy")
domain = domains.IGAPolarTorus(a=0.7, R0=3.0, tor_period=1)
eta = xp.linspace(0.0, 1.0, 24)
xyz = domain(0.8, eta, eta)                # a logical coordinate surface
J = domain.jacobian(0.4, 0.2, 0.3, squeeze_out=True)
g = domain.metric(0.4, 0.2, 0.3, squeeze_out=True)

flux = CircularFlux(R0=3.0, a=1.0)
# For an uploaded file: flux = EQDSKFlux.from_text(uploaded_text)
tokamak = domains.Tokamak(
    equilibrium=flux,
    num_elements=(8, 24), degree=(2, 3),
    xi_param="equal_angle", tor_period=1,
)
archive = save_geometry(tokamak)           # bytes, no pickle
restored = load_geometry(archive)          # needs no equilibrium object
x, y, z = sample_surface(restored, resolution=24)
```

Keep existing evaluation conventions: three coordinate arrays produce tensor
products; a single `(N, 3)` array evaluates a list of points. Use
`squeeze_out=True` for scalar probes. `sample_surface` fixes eta1; it does not
assemble all boundary faces of a general domain. Cartesian coordinates are
in mapping/input-file units; the browser package does not rescale EQDSK data.

## Included functionality

- All nine analytic domains in `geometry/domains.py`.
- `IGAPolarCylinder`, `IGAPolarTorus`, generic `Spline`,
  `PoloidalSplineStraight`, and `PoloidalSplineTorus`.
- Knot generation, B-spline evaluation, 2D/3D mapping interpolation, and
  `spline_interpolation_nd`.
- Jacobians, determinants, inverses, metrics, pullbacks, pushforwards, and
  field transformations.
- Tokamak flux tracing with `equal_angle`, `equal_arc_length`, `sfl`,
  `equal_area`, and `equal_volume` parametrizations. Custom flux providers
  must expose `psi(R, Z, dR=0, dZ=0)`, `psi_axis_RZ`, and `psi_range`.
- EQDSK rectangular poloidal flux interpolation, including first/second
  derivatives. The reader accepts scientific E/D notation. It refines the
  interpolated magnetic axis for either flux sign and performs no smoothing.
  It does not implement the pressure/current profiles of `EQDSKequilibrium`.
- Portable archives of analytic definitions and spline control points.
- Existing Matplotlib `show()` can be used if Matplotlib is installed and
  an appropriate display backend is configured; the viewer uses canvas.

## Deliberate differences from full Struphy

- The import name is `struphy_web`, and the wheel does not depend on or
  initialize full Struphy. The full `struphy` distribution remains unchanged
  in scope and dependencies.
- Browser `Tokamak()` requires an explicit flux provider. The viewer supplies
  `CircularFlux()` when no file is selected; it is a geometry example, not a
  force-balanced equilibrium. Full Struphy retains its EQDSK default.
- Browser `Spline()` requires explicit `cx`, `cy`, and `cz` control points.
- `GVECunit` and `DESCunit` are omitted. Create these domains with full
  Struphy, then use `struphy.geometry.portable.save_geometry(domain)` to export
  them. Browser loading reconstructs the generic spline and preserves mapping
  evaluation and derivatives; it does not run those equilibrium solvers.
- PyVista/VTK methods (`create_geometry_mesh`, `show_3d`, `export_geometry`)
  are omitted. Mesh arrays and NPZ archives are available instead.
- Analytic `to_dict/from_dict` remains available. Use the archive functions
  for spline/equilibrium-derived mappings; these save numerical coefficients
  rather than trying to serialize/reconstruct arbitrary equilibrium objects.
- GPU execution and runtime compilation are not browser features. Kernels
  run as Python functions with NumPy; cuNumPy's `PyccelKernel` wrapper remains
  valid because it accepts ordinary callables.

Dense sampling and fine flux tracing can be expensive in interpreted Python.
The worker keeps the UI responsive, but does not make the kernels faster.
Start with small grids. Polar axes are coordinate singularities; inspect
inverse metrics away from the axis. Flux tracing retains Struphy's assumptions
about nested surfaces and convergence, so not every EQDSK topology is suitable.

## How the shared-source build works

`build.py` has an explicit module allowlist. It generates an isolated namespace,
removes the `pure`/`stack_array` compilation decorators, excludes the native
adapters listed above, and writes a deterministic wheel with metadata, hashes,
and the license. It never copies `.so` files or modifies generated native
kernels. There is no separately maintained copy of the numerical algorithms.
Native compiler imports remain intact in the original sources.

Small shared-source changes decouple geometry imports from plotting, option
and class-helper infrastructure, allow lightweight flux providers, and fix
the missing exponent factor in `powered_ellipse_df`. Existing compiled
Struphy installations must rebuild their kernels to pick up that fix.

## Verification

```sh
python3 browser/build.py
# In an environment with NumPy, SciPy, cuNumPy and pytest:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q browser/tests

# After preparing the site:
npm exec --prefix browser -- playwright install chromium
npm test --prefix browser
```

The same numerical tests run on native Python and in browser Pyodide, covering
all analytic/IGA domains, 3D spline construction, finite-difference Jacobians,
metric identities, transformations, all five flux parametrizations, EQDSK of
both signs, periodic seams, poles, out-of-domain points, and archive roundtrips.
The browser tests block external requests to check that static hosting is
sufficient, and exercise editing, error recovery, and saving/reopening geometry.
