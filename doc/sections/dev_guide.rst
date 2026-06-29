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