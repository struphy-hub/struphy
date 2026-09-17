---
name: setup-simulation
description: Use when creating or editing a Struphy simulation parameter file (params_*.py) - choosing a model, configuring domain/grid/time-stepping, species/backgrounds/perturbations and propagator options, then running the simulation and post-processing results. Triggers on requests like "set up a simulation", "create a params file for a model", "configure a model run", or "how do I run struphy".
---

# Setting up a Struphy simulation

Struphy simulations are configured as plain Python scripts (`params_<name>.py`) that
build a `Simulation` object from the Struphy API, then call `sim.run()`. The params
file can use real Python (loops, conditionals, computed values) to derive parameters.
Each run writes `run_metadata.json`, containing a configuration snapshot and runtime
facts such as the MPI rank count. Post-processing reads this saved metadata directly.

## Workflow

1. **Confirm Struphy is installed and compiled** (`struphy compile`) if this is a fresh
   checkout — see `doc/sections/install.rst`. Skip if the user is clearly iterating on
   an existing working setup.

2. **Pick a starting point, in this order of preference:**
   - Find the closest existing example under `examples/<ModelName>/<case>/params_*.py`
     (list with `ls examples/`) and adapt it. This repo's examples are the most
     reliable reference because model-specific options (which propagators exist, what
     each species needs) vary per model and are easiest to get right by copying a
     working case.
   - If no example matches, scaffold a default with the CLI:
     `struphy params <ModelName>` — writes `params_<ModelName>.py` in the cwd with
     every option pre-filled to defaults and commented.
   - List available models with `struphy params --help` (shows all model names grouped
     as Fluid/Kinetic/Hybrid) or `ls src/struphy/models/*.py`.
   - Read the model's docstring (`src/struphy/models/<file>.py`) for its species,
     variables, and propagators before wiring options — don't guess propagator names.

3. **Edit the params file** following the structure below.

4. **Run it:**
   - `python params_<name>.py`
   - `mpirun -n <N> python params_<name>.py` for multiple MPI processes (all Struphy
     data structures are MPI-aware, so this works unchanged).

5. **Post-process** with a sibling `pproc_<name>.py` script (see pattern below) if the
   user wants plots or derived quantities beyond what's dumped live.

## Anatomy of a params file

Based on `examples/VlasovAmpereOneSpecies/two_stream/params_two_stream.py` and
`doc/sections/quickstart.rst`. Sections, in order:

```python
# 1. Free-text description (printed at run start, keeps runs traceable)
description = """..."""

# 2. Imports from the top-level struphy API
from pathlib import Path

from struphy import (
    BaseUnits, DerhamOptions, EnvironmentOptions, FieldsBackground,
    Simulation, Time, domains, equils, grids, perturbations,
)
# Kinetic species only:
from struphy import (
    BinningPlot, BoundaryParameters, KernelDensityPlot, LoadingParameters,
    SavingParameters, SortingParameters, WeightsParameters, maxwellians,
)
from struphy.models import <ModelName>

# 3. Model instance (constructor kwargs are model-specific, e.g. alpha/epsilon/with_B0)
model = <ModelName>(...)
model.<species>.<variable>.save_data = True   # opt in/out of saving each variable

# 4. Simulation-level config (all optional, sensible defaults exist)
base_units = BaseUnits()                                   # x, B, n, kBT -> derived units
env = EnvironmentOptions(
    out_folders=str(Path(__file__).resolve().parent),
    sim_folder="sim_data",
)  # output beside the script, independent of the working directory
time_opts = Time(dt=0.1, Tend=50.0, split_algo="LieTrotter")
domain = domains.Cuboid(r1=31.42)                           # see src/struphy/geometry/domains.py
equil = None                                                # or e.g. equils.HomogenSlab()
grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
derham_opts = DerhamOptions(degree=(3, 1, 1))               # spline degree, bcs, quadrature

# 5. Build the Simulation (this is what sim.run() acts on)
sim = Simulation(
    model=model, params_path=__file__, env=env, time_opts=time_opts,
    domain=domain, equil=equil, grid=grid, derham_opts=derham_opts,
)

# 6. Kinetic species only: markers (loading/weights/boundary/sorting/saving)
loading_params = LoadingParameters(ppc=1000, moments=(...))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)
saving_params = SavingParameters(binning_plots=(BinningPlot(...),))
model.<kinetic_species>.set_markers(
    loading_params=loading_params, weights_params=weights_params,
    boundary_params=boundary_params, sorting_params=sorting_params,
    saving_params=saving_params,
)

# 7. Propagator options — wire each propagator listed in the model to its variables.
#    Names/required args are model-specific; check the model source or an example.
model.propagators.<name>.options = model.propagators.<name>.Options(...)

# 8. Backgrounds, perturbations, initial conditions
#    - Fields: model.<species>.<var>.add_background(FieldsBackground())
#              model.<species>.<var>.add_perturbation(perturbations.ModesCos(...))
#    - Kinetic: background is mandatory; if add_initial_condition() is never called,
#      the background doubles as the initial condition. Perturbations for kinetic
#      species are added to distribution moments, not the variable directly:
maxwellian = maxwellians.Maxwellian3D(n=(0.5, None), u1=(3.0, None))
model.<kinetic_species>.var.add_background(maxwellian)
perturbation = perturbations.ModesCos(amps=(0.001,), ls=(1,))
init = maxwellians.Maxwellian3D(n=(0.5, perturbation), u1=(3.0, None))
model.<kinetic_species>.var.add_initial_condition(init)

# 9. Run only when the parameter script is executed directly.
if __name__ == "__main__":
    sim.run()
```

Key building blocks and where to look them up:

| Piece                                                                                                | Source                                                         |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Domains (`Cuboid`, `HollowTorus`, `Tokamak`, ...)                                                    | `src/struphy/geometry/domains.py`                              |
| Grid                                                                                                 | `src/struphy/topology/grids.py` (`TensorProductGrid`)          |
| Fluid equilibria                                                                                     | `src/struphy/fields_background/equils.py`                      |
| Perturbations                                                                                        | `src/struphy/initial/perturbations.py`                         |
| Kinetic backgrounds                                                                                  | `src/struphy/kinetic_background/maxwellians.py`                |
| Options dataclasses (`Time`, `BaseUnits`, `DerhamOptions`, `EnvironmentOptions`, `FieldsBackground`) | `src/struphy/io/options.py` (full docstrings with every field) |
| Models and their propagators/species                                                                 | `src/struphy/models/<model>.py`                                |

## Post-processing pattern

```python
from pathlib import Path
from struphy import Output

path_out = Path(__file__).resolve().parent / "sim_data"
out = Output(path_out)
out.process(physical=True)       # optional; products are otherwise processed with defaults on first access

out.scalars.<name>                                     # xarray time series, no post-processing needed
out.fields.<species>.<variable>                    # dims (t, [component,] e1, e2, e3)
out.distributions.<species>.<binning_name>.f    # dims (t, <slice dims>)
out.orbits.<species>                                   # dims (t, marker, quantity)
out.model.units                                        # model reconstructed from metadata
out.domain, out.grid, out.time_opts, out.derham_opts     # reconstructed lazily with from_dict()
```

Use the same folder name as the parameter script (`sim_data` or `sim_1` in the
examples). Keep a command-line path override when useful; only the default is
relative to `__file__`. In notebooks, where `__file__` is unavailable, choose an
explicit output path.

`Output` does not retain or construct a `Simulation`; there is no `out.sim`. Do not
import the parameter module for post-processing. Metadata comes from
`run_metadata.json`, with legacy `config.json` supported only as a fallback.
`sim.pproc()` remains supported for existing callers. Under MPI, call
`out.process()` on every rank; `parallel=True` requires the saved run's rank count.

Products sit under their species and plot themselves, no imports needed:

```python
out.<species>.<binning_name>.f.struphy.plot.slice(x="e1", y="v1", t="last")   # also .panels/.viewer/.animation/.frames
out.<species>.orbits.struphy.plot.trajectories()
out.scalars.<name>.struphy.plot.timeseries(fit=(t0, t1))                             # also .growth_rate/.drift/.relative_error
out.plot.scalars(), out.plot.equilibrium(), out.save_report()                   # whole-run plots
```

Name any dimension to select it: `t="last"`, `t=-1` (position), `t=0.35` (nearest value), `component=0`.
See `examples/VlasovAmpereOneSpecies/two_stream/pproc_two_stream.py` for a complete script.

## Common pitfalls

- Don't invent propagator or option names — every model has a different propagator
  set; read the model file or copy from a matching example instead of guessing.
- For kinetic species, a background is required even if only using perturbations;
  perturbations attach to distribution moments (via `maxwellians.*`), not directly to
  the species variable like field perturbations do.
- Keep `sim.run()` behind `if __name__ == "__main__":` to avoid starting a run on
  import. Post-processing scripts should open the saved output folder directly.
