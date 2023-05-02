Struphy documentation
=====================

.. toctree::
   :maxdepth: 1
   :caption: Contents

   sections/install
   sections/quickstart
   sections/userguide
   sections/models
   sections/domains
   sections/mhd_equils
   sections/kinetic_backgrounds
   sections/diagnostics
   sections/developers
   sections/discretization
   sections/propagators
   sections/notebooks
   sections/examples

*Struphy* (STRUcture-Preserving HYbrid codes) is a plasma physics library for solving partial differential equations based
on finite element and particle-in-cell methods. Its main objective is to provide easy access to MHD-kinetic hybrid models
for the description of nonlinear wave-particle resonances.

*Struphy* is open source *and* open development, and thus thrives on the contributions of its users. Codes are entirely
written in ``Python 3`` to maintain a low entry barrier for new contributors. All models are implemented on 
:ref:`avail_mappings` and can load a variety of :ref:`mhd_equil`.

**How to contribute:** The *Struphy* repository lies on the Gitlab instance of Max Planck Computing and Data Facility (MPCDF) https://gitlab.mpcdf.mpg.de/struphy/struphy.
To add code, you need an MPCDF Gitlab account. For the necessary invitation, please contact a Max Planck member, e.g. one of

   * florian.holderied@ipp.mpg.de
   * stefan.possanner@ipp.mpg.de
   * eric.sonnendruecker@ipp.mpg.de

Performance is important in *Struphy*. It is achieved using three building blocks:

   * `numpy <https://numpy.org/>`_ (vectorization)
   * `mpi4py <https://pypi.org/project/mpi4py/>`_ (parallelization)
   * `pyccel <https://github.com/pyccel/pyccel>`_ (compilation)

Heavy computational kernels are pre-compiled using the Python accelerator `pyccel <https://github.com/pyccel/pyccel>`_,
which on average shows `better performance <https://github.com/pyccel/pyccel-benchmarks>`_ than *Pythran* or *Numba*.

After :ref:`install`, you can quickly run *Struphy* models by following the :ref:`quickstart`.

*Struphy* is an object-oriented code. The concept of `inheritance <https://www.w3schools.com/python/python_inheritance.asp>`_ 
is heavily used in its basic design. 
Check out :ref:`base_classes` for an overview over the various predefined types that developers can build on.
*Struphy* :ref:`tutorials` in the form of `Jupyter notebooks <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/tree/devel/notebooks>`_
ensure a quick learning process for handling *Struphy* classes.

It is simple to add new model PDEs to the *Struphy* package. 
The base class :class:`struphy.models.base.StruphyModel` is the designed framework to do this.
The basic principle follows the "Lego"-approach, where basic operators/modules are provided to the developer
through the :ref:`base_classes`. A more detailed explanation can be found in :ref:`add_model`.

*Struphy* model discretization is based on finite element exterior calculus (FEEC) for the fluid/field quantities
and particle-in-cell (PIC) methods for the kinetic species. An overview of these methods is given in :ref:`gempic`.
For the FEEC spaces *Struphy* uses the open source library 

   * `psydac <https://github.com/pyccel/psydac>`_ (FEEC spaces)

Finally, you can visit the :ref:`gallery` to get some impressions of *Struphy* simulation results.


Contact
-------

The *Struphy* code base is constantly maintained. Please contact 

   * florian.holderied@ipp.mpg.de
   * stefan.possanner@ipp.mpg.de

for questions.




   


