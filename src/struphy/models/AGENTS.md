# Adding a new model

Read this before creating or renaming a model in `src/struphy/models/`. The general rules in the root `AGENTS.md` apply too.

## Start from an existing model

- One model per file, snake_case file name, CamelCase class name (e.g. `curl_curl.py` → `CurlCurl`).
- Copy the structure of a clean, recent model rather than starting from scratch:
  `maxwell.py` (pure fields, smallest), `curl_curl.py`, `vlasov_ampere_one_species.py` (particles + fields).
- Reuse existing propagators where possible. For a new propagator, follow `src/struphy/propagators/AGENTS.md`.

## New model or not? (a model is defined by its equations)

A Struphy model is defined by the PDE system in its `doc_pde`, not by its discretization.
Before adding a model, search the `doc_pde` of the existing models for the same equations.
- **Same equations, new discretization** (other spaces, propagator, solver, time stepping): don't create a new
  model. Add the discretization to the existing model through model flags (constructor arguments) or propagator
  options.
- **Existing equations plus a new term** (e.g. a dissipative term added to the MHD equations): create a new model
  that **inherits** from the existing one, adds the propagator(s) for the new term (override `Propagators`, keeping
  the parent's propagators) and shows the extended system, including the new term, in its own `doc_pde`.
- **No flags or options that switch propagators (i.e. terms of the equations) on or off.** A model with a term
  switched off is a different model; express it by inheritance as above.
See "Models are defined by their equations" in `doc/sections/dev_guide.rst`.

## Required structure (`StruphyModel` in `base.py`)

- **Species** as inner classes subclassing `FieldSpecies`, `FluidSpecies`, `ParticleSpecies` or `DiagnosticSpecies`
  (`species.py`). Their `__init__` declares the variables (`FEECVariable(space="H1"|"Hcurl"|"Hdiv"|"L2"|"H1vec")`,
  `PICVariable`, `SPHVariable` from `variables.py`) and then calls `self.init_variables(...)` with charge and mass
  numbers (no arguments for field species). A particle species holds exactly one variable.
- **`Propagators`** inner class. Its `__init__` creates propagator instances as attributes. **The attribute order is
  the splitting order** used by Lie-Trotter/Strang, unless `Propagators` defines a tuple `splitting_order` with the
  names of all propagators (class attribute, or set in `__init__`); `apply_splitting_order` in `base.py` reorders them.
- **`__init__`**, with the numbered steps:
  0. `self._store_params(locals())`. All constructor arguments need defaults, and only
     bool/int/float/str/tuple/list/None/`BaseUnits` values survive serialization;
  1.-5. `self._setup(base_units)`, which runs the hooks `_create_species()` (species as attributes, constructor
     arguments in `self.params`), `setup_equation_params` (units and `alpha`, `epsilon`, `lam`, `kappa`),
     `self.propagators = self.Propagators(**self._propagator_args())`, `_link_propagator_variables()`
     (`self.propagators.<prop>.variables.<var> = self.<species>.<var>`) and
     `self.scalars = Scalars(**self._create_scalars())` (`scalars.py`, e.g. `BilinearEnergyFEEC`, `KineticEnergyPIC`,
     summed with `+`).
  Older models write the steps out in `__init__` instead of `_setup`; both work.
  Keep `__init__` light-weight: no allocation. Work that needs `derham`, `mass_ops` or `projected_equil` goes into
  `post_allocate()`.
- **Inheritance** (a parent model plus new terms, e.g. `Vlasov` → `VlasovAmpereOneSpecies` → `VlasovMaxwellOneSpecies`):
  - `__init__` calls `self._store_params(locals())` **before** `super().__init__(...)` (the first call wins, so the
    hooks see the child's arguments, e.g. `alpha`), then adds what only the child has (e.g. the initial Poisson solve).
  - The hooks call `super()` and extend: `_create_species` adds species, `_propagator_args` adds constructor
    arguments of `Propagators`, `_link_propagator_variables` links the new propagators, `_create_scalars` adds
    scalars (and recomputes `en_tot`).
  - `class Propagators(Parent.Propagators)` passes its constructor arguments on with `super().__init__(**kwargs)` and
    adds the new propagators, with `splitting_order` if the order changes.
  - Species inherit too: `class EMFields(Parent.EMFields)` declares its new variables and then calls
    `super().__init__()` (whose `init_variables()` includes them); `KineticIons` of the Vlasov and guiding-center
    models only set `space`. Use the same variable names along the chain (`e_field`, `b_field`, `phi`, `var`) and
    the scalar names `en_f`, `en_E`, `en_B`, `en_phi`, `en_mu`, `en_tot`.
  - A model that renames a propagator adds `_renamed_propagators = {"old": "new"}` (class attribute), so that
    `from_dict` restores the options of old run metadata; unmatched options are skipped with a warning.
  - After the inner classes, annotate the attributes whose class the model defines or changes, e.g.
    `em_fields: EMFields` and `propagators: Propagators`. Species and propagators are created in the parent's hooks
    and `propagators` is a property, so without these annotations IDEs (Pylance) and type checkers see the parent's
    classes or none (`model.propagators.maxwell`, `model.em_fields.b_field` unknown in the params file).
- **`model_type()`** classmethod: `"Toy"`, `"Fluid"`, `"Kinetic"` or `"Hybrid"`. It selects the test group and marker.
- **`bulk_species`** and **`velocity_scale`** (`"alfvén"`, `"cyclotron"`, `"light"`, `"thermal"`) properties.
  They define the normalization; all scales except `"light"` need a bulk species.
  See `doc/sections/subsections/models-normalization.rst`.
- Propagator options are **not** set in the model. Users set them in the params file
  (`model.propagators.<prop>.options = model.propagators.<prop>.Options(...)`).
  New option literals go in `LiteralOptions` in `src/struphy/io/options.py`.

## Documentation (enforced by tests and CI)

- Class docstring: one-line summary plus a numpy-style `Parameters` section for the constructor arguments.
- All 8 doc classmethods, written as reST docstrings with math in `.. math::` / `:math:`:
  `doc_pde`, `doc_normalization`, `doc_scalar_quantities`, `doc_long_description`, `doc_examples`,
  `doc_use_cases`, `doc_cannot_be_used_for`, and `doc_discretization`.
  Put `doc_normalization` directly below `doc_pde` (also in subclasses that change the normalization), since the
  normalization defines the parameters appearing in the PDEs.
  `doc_discretization` must **return** a string. The default in `StruphyModel` generates it from `Propagators` (the
  propagators in splitting order with their `__doc__`; the Sphinx docs list them via a hook in `doc/conf.py`), so
  `Propagators()` must be constructible with its default arguments. Override it only to describe the time stepping in
  its docstring, returning `cls.propagators_doc()` (see `vlasov_poisson_adiabatic.py`).
  `test_all_models_expose_new_doc_api` fails if any is missing or empty.
- `doc_pde` is written like a physics textbook: always the full PDE system in strong form, with the unknowns and
  parameters named, but **no** solution spaces, weak formulations or discrete operators (those belong in the
  propagator docstrings, which `doc_discretization` embeds).
- Conserved quantities (energy, momentum, helicity, mass, charge, ...), whenever the system has them, are always
  given as formulas in `doc_pde` below the equations under a heading **Conserved quantities**, with the conditions
  under which they hold (boundary conditions, no sources, ...). `doc_scalar_quantities` lists the tracked scalars.
  See "Docstrings of models and propagators" in `doc/sections/dev_guide.rst`.
- Add an `.. autoclass:: struphy.models.<Name>` block (same options as its neighbours) under the matching category in
  `doc/sections/subsections/models-all.rst`. `doc/check_models_doc.py` fails in CI otherwise.

## Registration

- Run `struphy build-init-files` to regenerate `src/struphy/models/__init__.py`. Don't edit that file by hand.
- Add the model by hand to `src/struphy/api/models/__init__.py` (import and `__all__`). That file is not generated.

## Tests and examples

- The generic model tests (`src/struphy/models/tests/regression/test_models.py`, marker `models` plus
  `toy`/`fluid`/`kinetic`/`hybrid`) pick up every model automatically. They build `model()` with default
  arguments, generate the default params file (Cuboid domain, default `Options()` of every propagator),
  check serialization round-trips, run a simulation and a restart. So **the model must run with every propagator's
  default options**. Run one model with:
  ```bash
  pytest -m single --model-name <Name> src/struphy/models/tests/regression/
  mpirun -n 2 python -m pytest -p no:xdist --with-mpi -m single --model-name <Name> src/struphy/models/tests/regression/
  ```
- These are regression tests: the run (before the restart) is compared with `rtol=1e-6` against the baseline
  `src/struphy/models/tests/regression/baselines/<case>.json` (scalar time series, norms of the FE coefficients,
  marker sums), with one entry for 1 and one for 2 MPI ranks. Random marker loading and noise are seeded in the test.
  A new model, or a new model variant tested with its own `case`, needs a baseline; a change of the numbers
  (bug fix, new default) needs an update of the baselines. Create or update them with `--update-baselines`:
  ```bash
  pytest -m single --model-name <Name> --update-baselines src/struphy/models/tests/regression/
  mpirun -n 2 python -m pytest -p no:xdist --with-mpi -m single --model-name <Name> --update-baselines src/struphy/models/tests/regression/
  ```
  (or `-m models` for all models). Commit the JSON files, and explain in the PR description why the numbers changed.
  Never update a baseline only to make a failing test pass without understanding the change.
  Use `-n 4` instead if the model has particles.
- `struphy params <Name>` must produce a valid `params_<Name>.py`.
- Add a verification test against an analytic or reference result where possible, and where the test is not too expensive. Name it:
  `src/struphy/models/tests/verification/test_verif_<Name>.py` (e.g. `test_verif_Maxwell.py`). It is collected by folder.
- Add at least one example in `examples/<Name>/<case>/params_<case>.py`, with optional `pproc_<case>.py` and
  `regress_<case>.py`. Follow an existing example: module docstring, `name`, a `description` (summary, Physics, Setup,
  Expected result, References) and the argparse options (e.g. `--test-run`).
  `test_examples.py` runs every example with `--test-run`, so keep that mode small and fast.
- Don't edit `CHANGELOG.md` in the PR; it is written at release time.
