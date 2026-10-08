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
and ruff is installed with the ``dev`` extra (``pip install -e .[dev]``).
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