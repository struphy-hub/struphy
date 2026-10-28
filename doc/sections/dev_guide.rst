.. _dev_guide:

Developer's guide
=================

This is a collection of tutorials for developers. They are meant to be a practical guide to how to use Struphy 
for development purposes, such as implementing new features, propagators or models.

The tutorials give an overview of the most important classes used in writing new Propagators.
These include classes for FEEC (finite element exterior calculus), particles and the coupling between them.

All notebooks are available at https://github.com/struphy-hub/struphy-tutorials (with a link to a predefined environment on binder) 
or within the Struphy source under ``tutorials/`` (and in the built documentation under ``_collections/tutorials/``). 
They can be run with Jupyter notebooks or Jupyter lab. 
It is recommended to use the same Python environment as for Struphy, e.g., by installing the Jupyter packages in the same environment.


Code formatting and import sorting
----------------------------------

Struphy uses `ruff <https://docs.astral.sh/ruff/>`_ for formatting, linting and sorting imports
(ruff's ``I`` rules replace isort). The settings are in the ``[tool.ruff]`` section of ``pyproject.toml``,
and ruff is installed together with Struphy.
Run the following from the repository root before pushing::

    ruff check --fix    # lint and sort imports
    ruff format         # format code

Both commands also handle the Jupyter notebooks in ``tutorials/``.
The CI fails if either of the following reports a problem::

    ruff format --check
    ruff check

To limit ruff to certain files or folders, pass their paths, e.g. ``ruff format src/struphy/models``.
Most editors can run ruff on save, e.g. the `ruff extension for VS Code <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_.


Auto-generated ``__init__.py`` files
------------------------------------

The files ``struphy/models/__init__.py``, ``struphy/propagators/__init__.py`` and ``struphy/geometry/domains/__init__.py``
are generated from the classes defined in these packages. After adding, renaming or removing a model, propagator or domain, regenerate them with::

    struphy build-init-files

This command is only available when Struphy is installed in editable mode, and it formats the generated files with ruff.


.. _docstrings_models_propagators:

Docstrings of models and propagators
------------------------------------

Models and propagators are documented at two levels of detail. A model states the physics,
and its propagators state the mathematics and numerics. The propagator docstrings are embedded in the
model documentation (``doc_discretization``), so together they describe the model from the PDE down to the time stepping.

**Models** (``doc_pde`` and ``doc_scalar_quantities``) are written in the style of a physics textbook:

* ``doc_pde`` always shows the PDE system solved by the model, in strong form and with the usual physics notation
  (e.g. :math:`\nabla \times`, :math:`\nabla \cdot`, :math:`\mathbf v \times \mathbf B`).
  It names the unknowns and the parameters (as defined in ``doc_normalization``), but contains
  **no** solution spaces, weak formulations or discrete operators. These belong to the propagators.
* If the system has conserved quantities, such as energy, momentum, magnetic helicity, mass or charge,
  they are always stated as formulas (e.g. :math:`\frac{d}{dt} \frac{1}{2} \int (|\mathbf E|^2 + |\mathbf B|^2)\,\textrm d \mathbf x = 0`),
  in ``doc_pde`` below the equations under a heading **Conserved quantities**, together with the conditions under which they hold
  (e.g. boundary conditions, absence of sources). ``doc_scalar_quantities`` lists the scalars tracked during a simulation.

**Propagators** are more mathematical. The class docstring always contains three sections:

1. **Continuous problem:** the substep solved by the propagator, stated for the unknowns that it updates,
   with their solution spaces (e.g. :math:`\mathbf E \in H(\textrm{curl})`, :math:`\mathbf B \in H(\textrm{div})`)
   and the weak formulation where one is used (find :math:`\mathbf E \in H(\textrm{curl})` such that ... for all
   :math:`\mathbf F \in H(\textrm{curl})`).
2. **Semi-discrete problem:** the space-discrete version of the substep, as a system for the FE coefficients and/or the
   markers. Every operator that appears (mass matrices, discrete derivatives, projectors, basis operators,
   accumulation matrices, ...) is defined or referenced, e.g. with ``:class:`` or ``:ref:`` links into the documentation.
3. **Time discretization:** all available time-stepping options (e.g. implicit midpoint/Crank-Nicolson, explicit
   Runge-Kutta with Butcher tableau, discrete gradient, analytic solution), each with its update formula,
   the option that selects it and the linear or nonlinear solver it requires.


.. toctree::
   :maxdepth: 1
   :caption: FEEC tutorials:

   ../_collections/tutorials/dev_tutorial_feec_basics
   ../_collections/tutorials/dev_tutorial_feec_bcs
   ../_collections/tutorials/dev_tutorial_data_structs


.. toctree::
   :maxdepth: 1
   :caption: SPH tutorials:

   ../_collections/tutorials/dev_tutorial_sph_eval_kernels