.. _overview:

Struphy abstract
================

**Struphy provides easy access to partial differential equations (PDEs) in plasma physics.
The package combines** *performance* **(for HPC),** *flexibility* **(models and physics features)
and** *usabilty* **(documentation).**  

*Performance* in Struphy is achieved using three building blocks:

   * `numpy <https://numpy.org/>`_ (vectorization)
   * `mpi4py <https://pypi.org/project/mpi4py/>`_ (parallelization)
   * `pyccel <https://github.com/pyccel/pyccel>`_ (compilation)

Heavy computational kernels are pre-compiled using the Python accelerator `pyccel <https://github.com/pyccel/pyccel>`_,
which on average shows `better performance <https://github.com/pyccel/pyccel-benchmarks>`_ than *Pythran* or *Numba*.

*Flexibility* comes through the possibility of applying different :ref:`models` to a plasma physics problem.
Each model can be run on different :ref:`avail_mappings` and can load a variety of :ref:`mhd_equil_avail`.
You can add your model, mapping or equilibrium via the :ref:`base_classes`.

*Usability* is guaranteed by Struphy's extensive documentation. For instance, you can learn Struphy
through a series of Jupyter notebook :ref:`tutorials`.

Struphy is an object-oriented code. The concept of `inheritance <https://www.w3schools.com/python/python_inheritance.asp>`_ 
is heavily used in its basic design. 
It is simple to add new plasma models to Struphy. 
The base class :class:`struphy.models.base.StruphyModel` is the designed framework to do this.
The basic principle follows the "Lego"-approach, where basic operators/modules are provided to the developer
through the :ref:`base_classes`. A more detailed explanation can be found in :ref:`add_model`.

Model discretization is based on finite element exterior calculus (FEEC) for the fluid/field quantities
and particle-in-cell (PIC) methods for the kinetic species. An overview of these methods is given in :ref:`gempic`.
For the FEEC spaces Struphy uses the open source library 

   * `psydac <https://github.com/pyccel/psydac>`_ (FEEC spaces)

You can visit the :ref:`gallery` to get some impressions of Struphy simulation results.
