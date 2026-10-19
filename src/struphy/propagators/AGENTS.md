# Adding a new propagator

Read this before creating or renaming a propagator in `src/struphy/propagators/`. The general rules in the root `AGENTS.md` apply too.
A propagator advances some variables of a model from t to t + dt; models chain propagators by operator splitting.

## Start from an existing propagator

- One module per propagator, snake_case module name, CamelCase class name (e.g. `push_vxb.py` → `PushVxB`).
  Closely related variants may share a module (e.g. `shear_alfven_curl_curl_propagator.py`).
  Particle pushers are named `Push...`, variational propagators `Variational...`.
- Copy the structure of a clean, recent propagator: `push_vxb.py` (particles), `shear_alfven_propagator.py` or
  `shear_alfven_curl_curl_propagator.py` (FEEC with linear solvers and preconditioners).
- Check first whether an existing propagator already does the job with a new option.
- Don't copy the template in `base.py` literally: its `opt1` default is a tuple.

## Required structure (`Propagator` in `base.py`)

All of the following are abstract; the class can't be instantiated without them.

- **`Variables`** inner class: one private slot per variable that the propagator **updates**, with a typed property
  and a setter that asserts the type (`FEECVariable`, `PICVariable`, `SPHVariable`) and the space
  (e.g. `assert new.space in ("Hcurl", "Hdiv")`). The model links its species' variables here.
- **`__init__`**: sets `self.variables = self.Variables()`. Variables that are only **read** (not updated) are
  constructor arguments with default `None` (e.g. `PushVxB(b2_var=...)`), not options.
- **`Options`**: `@dataclass(repr=False)` subclass of `OptionsBase`.
  - Every field has a default, and **the default options must give a working propagator**, because the generic
    model tests run every model with default `Options()`.
  - Allowed values are `Literal` types, defined in the class (`OptsAlgo = Literal[...]`) or shared in
    `LiteralOptions` in `src/struphy/io/options.py`. Validate them in `__post_init__` with `check_option`, add
    cross-option asserts there, and fill `None` defaults (e.g. `SolverParameters()`, `ButcherTableau()`).
  - Document every option in a numpy-style `Parameters` section.
- **`options`** property and setter: copy the boilerplate from an existing propagator unchanged.
- **`allocate()`**: build operators, solvers, pushers and temporary vectors here, never in `__init__`. The shared
  objects are available as class attributes: `self.derham`, `self.domain`, `self.mass_ops`, `self.basis_ops`,
  `self.projected_equil`. Species parameters come from `self.variables.<var>.species.equation_params`.
- **`__call__(self, dt)`**: advance from t to t + dt without allocating. Write FEEC results back with
  `self.update_feec_variables(<var>=new_coeffs)`, which also updates ghost regions. Particles are updated in place
  (e.g. by a `Pusher`). The current time is `self.time_state[0]`.

Linear solves: configure with `SolverParameters` (`src/struphy/linear_algebra/solver.py`) and choose solver and
preconditioner through `Literal` options. Never put `KroneckerStencilMatrix.dot` (pure Python) in a Krylov loop.

## Docstring

The class docstring is a raw string (`r"""..."""`) in reST:
- the equations solved (strong or weak form) in `.. math::`, with the function spaces;
- the discrete system and the time discretization (implicit/explicit, Butcher tableau, ...);
- the available algorithms/options and the solver used, with `:class:` cross-references.

It is embedded verbatim into the docs of every model that uses the propagator (`doc_discretization`).

## Registration and docs

- Run `struphy build-init-files` to regenerate `src/struphy/propagators/__init__.py`. Don't edit that file by hand.
- Add an `.. automodule:: struphy.propagators.<module>` block (same options as its neighbours, including
  `:exclude-members: options, allocate`) under the matching heading in
  `doc/sections/subsections/propagators-avail.rst`. `doc/check_propagator_doc.py` fails in CI otherwise.
- New option literals go in `LiteralOptions` in `src/struphy/io/options.py`.
- Use the propagator in at least one model. Model rules are in `src/struphy/models/AGENTS.md`.

## Particle kernels

- Each pusher or accumulation kernel lives in its own folder: `src/struphy/pic/pushing/kernels/<name>/` or
  `src/struphy/pic/accumulation/kernels/<name>/`, containing `<name>_kernels.py` (Pyccel), an `__init__.py` that
  creates the kernel with `Kernel.from_folder(...)` and declares `OUTPUTS`, and optionally `<name>_cuda.cu`.
  Copy an existing folder, e.g. `pic/pushing/kernels/push_vxb_analytic/`. See `CUDA_STRATEGY.md` for the CUDA port
  and the parity tests in `pic/tests/`.
- Pusher kernels take `(dt, stage, args_markers, args_domain, args_derham, *extra)`. Keep the module import of
  `struphy.kernel_arguments.pusher_args_kernels`, which is needed for compile dependencies.
- `Pusher(..., pushes_eta=...)` must say whether the kernel moves the markers in eta. Only then are markers sorted
  across MPI ranks afterwards.
- Recompile with `struphy compile -y` and check `struphy compile --status`.

## Tests

- Put tests in `src/struphy/propagators/tests/test_<module>.py`. That folder is already registered in CI.
- Test the physics, not just that it runs: conservation (energy via `Propagator.mass_ops` inner products),
  convergence in dt or grid size against an analytic solution, agreement with an existing propagator for the same
  equations, and rejection of invalid options (`pytest.raises(AssertionError)`).
- Two styles exist: through a model and `Simulation` (`sim.allocate()`, then call the propagator; see
  `test_shear_alfven_curl_curl.py`), or standalone by setting `Propagator.derham`, `.domain` and `.mass_ops`
  yourself (see `test_curl_curl.py`).
- Run FEEC propagator tests under `mpirun -n 2` and particle propagator tests under `mpirun -n 4`, with
  `-p no:xdist --with-mpi`.
