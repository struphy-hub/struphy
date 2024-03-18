.. _developers:

Developer's guide
=================

Developers can contribute to Struphy in multiple ways. A common approach is by :ref:`add_model`,
but also writing :ref:`diagnostics` or physics features, or modifying core routines is possible.

The main line of communication between developers is the `Struphy-developers channel <https://chat.gwdg.de/channel/struphy-developers>`_ 
on the GWDG RocketChat. 
Besides, the Struphy developer community meets regularly at `Struphy Hackathons <https://gitlab.mpcdf.mpg.de/struphy/struphy-hackathons>`_.

Struphy is an object-oriented code that provides base classes (templates)
which facilitate the implementation of new features. 
A developer can use the Struphy base classes to construct new instances of operators/algorithms,
suitable for his PDE model. The available core routines provide the MPI/OpenMP hybrid parallelization
necessary for HPC applications.
**The Struphy approach enables to focus on the physical and mathematical aspects of a model.**

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    subsections/git_workflow
    subsections/adding_model
    subsections/coding_conventions
    subsections/data_structures
    subsections/change_doc











































