.. _overview:

Tl;dr
=====

**Struphy is a Python package for solving partial differential equations (PDEs) of plasma physics, from fluid and
MHD models to kinetic and hybrid fluid-kinetic models, with structure-preserving numerics and on HPC systems.**

Main features:

* **Model library**: about 50 ready-to-run :ref:`models` (Maxwell, MHD, Vlasov, drift-kinetic, gyrokinetic,
  MHD-kinetic hybrids, SPH fluids, ...), each defined by its equations, with physical normalization and
  :ref:`propagators` that can be reused to build new models (see :ref:`model_anatomy`).
* **Structure-preserving numerics**: geometric finite elements (FEEC), particle-in-cell (PIC) and smoothed particle
  hydrodynamics (SPH), coupled such that conservation laws, :math:`\nabla \cdot \mathbf B = 0` and the
  Hamiltonian structure of the models carry over to the discrete level; see :ref:`gempic`.
* **Geometry and equilibria**: simulations on mapped domains (:ref:`avail_mappings`), from slabs to tori and polar
  domains with a magnetic axis, with analytic and numerical :ref:`MHD equilibria <fluid_equils>` (EQDSK, GVEC, DESC)
  and :ref:`kinetic backgrounds <kinetic_equils>`.
* **Performance**: compute kernels transpiled to Fortran with `Pyccel <https://github.com/pyccel/pyccel>`_,
  MPI domain decomposition (optional, Struphy also runs without MPI) and particle clones, and an optional GPU backend
  (CuPy).
* **Usability**: simulations are set up in plain Python parameter files, which are run with ``python`` (or
  ``mpirun``) or from a notebook; results are read as xarray objects with the post-processing API (see :ref:`userguide`).
  Ready-to-run examples are in `struphy-examples <https://github.com/struphy-hub/struphy-examples>`_, and
  ``struphy agents`` prints a usage guide for AI coding agents (``--install`` installs their skills).

Start with :ref:`install` and the :ref:`quickstart`.
Questions, bug reports and ideas are welcome on `GitHub <https://github.com/struphy-hub/struphy#get-in-touch>`_.
