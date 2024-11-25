.. _data_structures:

Struphy data structures
-----------------------

Check out `Tutorial 07 - Struphy data structures <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_07_data_structures.html>`_
for a hands-on introduction.


FEEC variables
^^^^^^^^^^^^^^

Struphy uses the FEEC data structures provided by the open source package `Psydac <https://github.com/pyccel/psydac>`_ for its
fluid/EM-fields variables. FE coefficients are stores as

* a :class:`StencilVector <psydac.linalg.stencil.StencilVector>` for scalar-valued variables (:code:`H1` or :code:`L2`)
* a :class:`BlockVector <psydac.linalg.block.BlockVector>` for vector-valued variables (:code:`Hcurl`, :code:`Hdiv` or :code:`H1vec`)

A BlockVector is just a 3-list of StencilVectors. 

.. autoclass:: psydac.linalg.stencil.StencilVector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: psydac.linalg.block.BlockVector
    :members:
    :undoc-members:
    :show-inheritance:

Kinetic variables
^^^^^^^^^^^^^^^^^

All information pertaining to markers in Struphy is stored in the :ref:`particle_base`. 
In particular, the data structure holding the values of each marker is under :meth:`struphy.pic.base.Particles.markers`.
