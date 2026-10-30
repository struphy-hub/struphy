---
name: new-model
description: Use when a user wants to implement, add or create a new Struphy model (a new PDE system in src/struphy/models/), or to add terms to an existing model. Checks first whether the model belongs to an existing model family (Vlasov, Guiding-center, LinearEuler, ...) and asks the user whether to add it there. Triggers on requests like "implement a new model", "add a model for ...", "create model X with propagators Y and Z".
---

# Implementing a new Struphy model

1. **Read `src/struphy/models/AGENTS.md`** (required structure, docs, registration, tests) and, for new
   propagators, `src/struphy/propagators/AGENTS.md`.

2. **Check the model families first** (section "Model families" in `src/struphy/models/AGENTS.md` and the family
   sections of `doc/sections/subsections/models-all.rst`). Write down the PDE system of the requested model and
   compare it with the `doc_pde` of the family members:
   - **same equations** as an existing model: no new model; add the discretization as a flag or propagator option;
   - **equations of a family member plus new terms**: the model fits that family. **Ask the user** whether to add
     it to the family as a child of that member (inheriting species, propagators, scalars via `super()`), naming
     the parent and the new propagators;
   - several possible parents: present the options with a recommendation;
   - no family fits: say so, and ask whether the new model should become the parent of a new family (then it must be
     well tested, see below).
   Do not start coding before the user has answered.

3. **Implement** following the inheritance pattern of the family (e.g. `vlasov.py` → `vlasov_ampere_one_species.py`,
   `linear_euler.py` → `linear_mhd.py`): `_store_params(locals())` before `super().__init__()`, hooks
   `_create_species`, `_propagator_args`, `_link_propagator_variables`, `_create_scalars` extended with `super()`,
   `class Propagators(Parent.Propagators)` with `splitting_order` (and `strang_groups`, inherited) if needed, type annotations of the redefined
   attributes, no flags that switch terms on or off.

4. **Tests**: the generic model tests pick up the model automatically. Parent models must be well tested
   (verification against analytic results, a curvilinear geometry such as `Colella`, conservation laws, time
   convergence order); for children, test at least what the new terms add.

5. **Register**: `models-all.rst` (in the family section), `struphy build-init-files`, `src/struphy/api/models/__init__.py`,
   and update the family list in `src/struphy/models/AGENTS.md` if the family changes.
