---
name: setup-simulation
description: Use when creating or editing a Struphy simulation parameter file (params_*.py) - choosing a model, configuring domain/grid/time-stepping, species/backgrounds/perturbations and propagator options, then running the simulation. Triggers on requests like "set up a simulation", "create a params file for a model", "configure a model run", or "how do I run struphy". For reading and plotting results, use the postprocess-simulation skill.
---

# Setting up a Struphy simulation

Struphy simulations are configured as plain Python scripts (`params_<name>.py`) that
build a `Simulation` object from the Struphy API, then call `sim.run()`. The params
file can use real Python (loops, conditionals, computed values) to derive parameters.
Each run writes `run_metadata.json`, containing a configuration snapshot and runtime
facts such as the MPI rank count. Post-processing reads this saved metadata directly.

This skill ships with the installed Struphy package, so it matches the installed version.
`struphy agents` prints the general usage guide and the examples URL for that version.

## Workflow

1. **Confirm Struphy is installed and compiled** if this is a fresh environment:
   `struphy compile --status` must report all kernels compiled; otherwise run `struphy compile`.
   Install help: https://struphy-hub.github.io/struphy/sections/install.html.
   Skip this if the user is clearly iterating on an existing working setup.

2. **Pick a starting point, in this order of preference:**
   - Find the closest existing example and adapt it. Examples live in `<ModelName>/<case>/params_*.py` of
     https://github.com/struphy-hub/struphy-examples (not in the pip package; `struphy agents` prints the URL
     of the tag for the installed version, clone it with `git clone --branch vX.Y.Z ...`). In a Struphy source
     checkout, they are in `examples/`. They are the most reliable reference because model-specific
     options (which propagators exist, what each species needs) vary per model and are easiest to get right
     by copying a working case.
   - If no example matches, scaffold a default with the CLI:
     `struphy params <ModelName>` writes `params_<ModelName>.py` in the cwd with
     every option pre-filled to defaults and commented (`-y` skips the prompt).
   - List available models with `struphy params --help` (all model names), or
     `struphy --fluid`, `struphy --kinetic`, `struphy --hybrid` for one group.
     Model descriptions: https://struphy-hub.github.io/struphy/sections/models.html.
   - Read the model's docstring for its species, variables and propagators before wiring options,
     e.g. `python -c "from struphy.models import Maxwell; help(Maxwell)"`. Don't guess propagator names.

3. **Edit the params file** following the structure below.

4. **Check it** with `struphy params <ModelName> --check-file params_<name>.py`.

5. **Run it:**
   - `python params_<name>.py`
   - `mpirun -n <N> python params_<name>.py` for multiple MPI processes (all Struphy
     data structures are MPI-aware, so this works unchanged; requires `pip install mpi4py`).

6. **Post-process** with a sibling `pproc_<name>.py` script if the user wants plots or derived
   quantities beyond what's dumped live. See the postprocess-simulation skill.

## Anatomy of a params file

Based on the example `VlasovAmpereOneSpecies/two_stream/params_two_stream.py` and the
quickstart (https://struphy-hub.github.io/struphy/sections/quickstart.html). Sections, in order:

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
domain = domains.Cuboid(r1=31.42)                           # see help(domains.Cuboid)
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
#    Names/required args are model-specific; check help(<ModelName>), `struphy params <ModelName>` or an example.
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

Key building blocks and where to look them up. Every class has a full docstring
(`help(...)` in Python); the module's file is printed by e.g.
`python -c "import struphy.geometry.domains as m; print(m.__file__)"`.

| Piece                                                                                                | Module                                                    |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Domains (`Cuboid`, `HollowTorus`, `Tokamak`, ...)                                                    | `struphy.domains` (`struphy.geometry.domains`)            |
| Grid (`TensorProductGrid`)                                                                           | `struphy.grids` (`struphy.topology.grids`)                |
| Fluid equilibria                                                                                     | `struphy.equils` (`struphy.fields_background.equils`)     |
| Perturbations                                                                                        | `struphy.perturbations` (`struphy.initial.perturbations`) |
| Kinetic backgrounds                                                                                  | `struphy.maxwellians` (`struphy.kinetic_background.maxwellians`) |
| Options dataclasses (`Time`, `BaseUnits`, `DerhamOptions`, `EnvironmentOptions`, `FieldsBackground`) | `struphy.io.options` (docstrings list every field)        |
| Models and their propagators/species                                                                 | `struphy.models.<Model>`                                  |

Lists of what is available, with parameters, under https://struphy-hub.github.io/struphy/sections/:
`domains.html`, `fluid-equils.html`, `perturbations.html`, `kinetic-equils.html` and
`models.html` (models and their propagators).

## Common pitfalls

- Don't invent propagator or option names — every model has a different propagator
  set; read `help(<ModelName>)`, generate `struphy params <ModelName>` or copy from a matching example
  instead of guessing.
- The output folder is `EnvironmentOptions.sim_folder` (default `sim_1`) inside `out_folders`; the
  post-processing script must open the same folder.
- For kinetic species, a background is required even if only using perturbations;
  perturbations attach to distribution moments (via `maxwellians.*`), not directly to
  the species variable like field perturbations do.
- Keep `sim.run()` behind `if __name__ == "__main__":` to avoid starting a run on
  import. Post-processing scripts should open the saved output folder directly.
