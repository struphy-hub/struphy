.. _overview:

Tl;dr
=====

**Struphy provides easy access to partial differential equations (PDEs) in plasma physics.
The package combines** *performance* **(for HPC),** *flexibility* **(models and physics features)
and** *usability* **(Python).**  

**Performance** in Struphy is achieved from three building blocks:

* `numpy <https://numpy.org/>`_ (vectorization)
* `mpi4py <https://pypi.org/project/mpi4py/>`_ (parallelization)
* `pyccel <https://github.com/pyccel/pyccel>`_ (compilation)

Heavy compute kernels are transpiled using the Python accelerator `pyccel <https://github.com/pyccel/pyccel>`_.
You will thus enjoy the speed of Fortran or C while working in the familiar Python environment.

**Flexibility** comes through the possibility of applying different :ref:`models` to a plasma physics problem.
Each model can be run on different :ref:`avail_mappings` and can load a variety of :ref:`equils`,
as well as other relevant physial inputs (such as initial conditions).

**Usability** is guaranteed by Python. 
Struphy code can be handled in the familiar object-oriented way of Python.

Struphy is modular and allows you to add your own model, benefitting from the abstraction
provided by the Struphy classes. Check out :ref:`add_model` to learn more.

At present, abstractions for the following numerical methods are available:

* finite element exterior calculus (FEEC), through `Psydac <https://github.com/pyccel/psydac>`_
* particle-in-cell (PIC)
* smoothed-particle hydrodynamics

See :ref:`gempic` for more details on the numerical methods.

If you are passionate about some of the above topics, `get in touch <https://github.com/struphy-hub/struphy/tree/32-publish-struphy-pages-on-github-clean?tab=readme-ov-file#get-in-touch>`_!
