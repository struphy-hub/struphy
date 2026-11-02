# Struphy examples

Ready-to-run simulations for [Struphy](https://github.com/struphy-hub/struphy), the Python package for
structure-preserving plasma physics PDEs. Each example is a physics scenario or a numerical test case with
a parameter file that runs the simulation and a post-processing script that plots and checks the results.

- Website: <https://struphy-hub.github.io/> (models, numerics and an examples gallery)
- Documentation: <https://struphy-hub.github.io/struphy/>
- Source code: <https://github.com/struphy-hub/struphy>

The examples are developed in the [`examples/`](https://github.com/struphy-hub/struphy/tree/devel/examples)
folder of the Struphy repository and copied to
[struphy-examples](https://github.com/struphy-hub/struphy-examples) with every Struphy release.

## Versions

The tag `vX.Y.Z` holds the examples of Struphy release `X.Y.Z`; the `main` branch holds those of the latest release.
Use the tag that matches your installed Struphy (`struphy --version`), because parameter files change with the API:

```bash
pip install -U struphy
struphy compile
git clone --branch vX.Y.Z https://github.com/struphy-hub/struphy-examples.git   # X.Y.Z from struphy --version
```

## Running an example

The folders are sorted by model name and then by case: `<Model>/<case>/` contains

- `params_*.py`: the parameter file, which sets up and runs the simulation;
- `pproc_*.py`: the post-processing script, which reads the output and creates the figures.

Some cases are also used for performance profiling in Struphy; their `submit_*.py` scripts need the profiling
driver of the Struphy repository and are therefore not included here.

For example, the two-stream instability of the model `VlasovAmpereOneSpecies`:

```bash
cd struphy-examples/VlasovAmpereOneSpecies/two_stream
python params_two_stream.py                  # serial run
mpirun -n 4 python params_two_stream.py      # or in parallel (needs `pip install mpi4py`)
python pproc_two_stream.py                   # figures of the run
```

The output folder (`sim_1/` or `sim_data/`) is created beside the scripts, and the post-processing script finds it
there. Both scripts accept `--help`; the parameter files take `--out-folder` and `--sim-folder` to change the output
location, and most of them `--test-run` for a short, cheap run.
The physics and the expected results are described in the docstring at the top of each parameter file.

For plots and diagnostics of the output, install [plasma-plots](https://struphy-hub.github.io/plasma-plots)
with `pip install "struphy[pproc]"`.

## Available examples

| Model | Cases |
| --- | --- |
| `CurlCurl` | `gaussian_cube`, `gaussian_hollow_cylinder`, `gaussian_hollow_torus`, `gaussian_on_axis_cylinder`, `gaussian_on_axis_torus` |
| `DriftKineticElectrostaticAdiabatic` | `cyclone` (Cyclone base case ITG), `itg_cylindre` (ITG in a cylinder), `slab_drift_wave` |
| `GuidingCenter` | `screw_pinch_equilibrium`, `tokamak_equilibrium` |
| `GyrokineticPoisson` | `gaussian_hollow_cylinder`, `gaussian_hollow_torus`, `gaussian_on_axis_cylinder`, `gaussian_on_axis_torus` |
| `LinearEuler` | `standing_wave_colella` (standing sound wave in the curvilinear Colella domain) |
| `LinearMHD` | `itpa_tae_benchmark`, `lowmode_tae` (toroidal Alfvén eigenmodes), `slab_pulse_2d` |
| `LinearMHDDriftkineticCC` | `lowmode_tae` (toroidal Alfvén eigenmode driven by energetic ions) |
| `Poisson` | `gaussian_cube`, `gaussian_hollow_cylinder`, `gaussian_hollow_torus`, `gaussian_on_axis_cylinder`, `gaussian_on_axis_torus` |
| `ShearAlfven` | `itpa_tae_benchmark`, `lowmode_tae` (toroidal Alfvén eigenmodes) |
| `ToyDrift` | `diocotron_instability` |
| `VariationalMHD` | `orszag_tang` (Orszag-Tang vortex), `slab_dispersion` (Alfvén, slow and fast waves) |
| `VariationalPerturbativeLinearMHD` | `with_p/lowmode_tae`, `with_q/lowmode_tae` (the `LinearMHD` low-mode TAE with `VariationalPerturbativeMHD` and `VariationalPerturbativeMHD_with_q`, `linear=True`) |
| `VariationalPerturbativeMHD_with_q` | `slab_dispersion` (Alfvén, slow and fast waves), `slab_dispersion_linear` (the same with `linear=True`) |
| `Vlasov` | `maxwellian_uniform_field` |
| `VlasovAmpereOneSpecies` | `bump_on` (bump-on-tail), `strong_Landau_damping`, `two_stream`, `weak_Landau_damping` |
| `VlasovMaxwellOneSpecies` | `weak_Landau_damping`, `weibel_instability` |
| `VlasovPoissonElectrostaticAdiabatic` | `slab_drift_wave` |

The `gaussian_*` cases solve elliptic problems with a manufactured solution on different geometries
(cube, cylinder, torus, with and without a hole) and compare polar and tensor-product splines and preconditioners.
The model equations are documented in the [list of models](https://struphy-hub.github.io/struphy/sections/models.html).

## Starting your own simulation

Copy the closest example and adapt it, or write a default parameter file for any model with

```bash
struphy params <Model>
```

See the [quickstart](https://struphy-hub.github.io/struphy/sections/quickstart.html) and the
[user guide](https://struphy-hub.github.io/struphy/sections/userguide.html).
If you work with an AI coding agent (Claude Code, Codex, Copilot, Cursor, ...), run `struphy agents` for a
usage guide that matches your installed Struphy version.

## Contributing

This repository is overwritten with every Struphy release. To add or change an example, open a pull request
into the `devel` branch of [Struphy](https://github.com/struphy-hub/struphy) with your changes in its
`examples/` folder.
