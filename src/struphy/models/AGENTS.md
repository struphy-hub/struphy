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
- **Inheritance** (a parent model plus new terms, e.g. `Vlasov` → `VlasovAmpereOneSpecies` → `VlasovMaxwellOneSpecies`):
  `class Propagators(Parent.Propagators)` calls `super().__init__()` and adds the new propagators (an attribute of
  the same name replaces a parent propagator), with `splitting_order` if the order changes. Species classes inherit
  too (`class KineticIons(Parent.KineticIons)`). Link the variables in `_link_propagator_variables()`, which calls
  `super()` first. The model `__init__` keeps the numbered steps below (no `super().__init__()`).
- **`__init__`**, keeping the numbered steps of the existing models:
  0. `self.params = copy.deepcopy(locals())`. All constructor arguments need defaults, and only
     bool/int/float/str/tuple/list/None/`BaseUnits` values survive serialization.
  1. instantiate the species as attributes;
  2. `self.setup_equation_params(base_units=base_units)` (derives units and `alpha`, `epsilon`, `lam`, `kappa`);
  3. `self.propagators = self.Propagators()`;
  4. link variables: `self.propagators.<prop>.variables.<var> = self.<species>.<var>`;
  5. `self.scalars = Scalars(...)` with the tracked quantities (`scalars.py`, e.g. `BilinearEnergyFEEC`,
     `KineticEnergyPIC`, summed with `+`).
  Keep `__init__` light-weight: no allocation. Work that needs `derham`, `mass_ops` or `projected_equil` goes into
  `post_allocate()`.
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

- The generic model tests (`src/struphy/models/tests/default_params/test_models.py`, marker `models` plus
  `toy`/`fluid`/`kinetic`/`hybrid`) pick up every model automatically. They build `model()` with default
  arguments, generate the default params file (Cuboid domain, default `Options()` of every propagator),
  check serialization round-trips, run a simulation and a restart. So **the model must run with every propagator's
  default options**. Run one model with:
  ```bash
  pytest -m single --model-name <Name> src/struphy/models/tests/default_params/
  mpirun -n 2 python -m pytest -p no:xdist --with-mpi -m single --model-name <Name> src/struphy/models/tests/default_params/
  ```
  Use `-n 4` instead if the model has particles.
- `struphy params <Name>` must produce a valid `params_<Name>.py`.
- Add a verification test against an analytic or reference result where possible, and where the test is not too expensive. Name it:
  `src/struphy/models/tests/verification/test_verif_<Name>.py` (e.g. `test_verif_Maxwell.py`). It is collected by folder.
- Add at least one example in `examples/<Name>/<case>/params_<case>.py`, with optional `pproc_<case>.py` and
  `regress_<case>.py`. Follow an existing example: module docstring, `name`, a `description` (summary, Physics, Setup,
  Expected result, References) and the argparse options (e.g. `--test-run`).
  `test_examples.py` runs every example with `--test-run`, so keep that mode small and fast.
- Don't edit `CHANGELOG.md` in the PR; it is written at release time.
