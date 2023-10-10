.. _accums:

Accumulators
============

This page lists `currently available accumulation kernels <https://struphy.pages.mpcdf.de/struphy/sections/STUBDIR/struphy.pic.accumulation.accum_kernels.html>`_ (for charge and current deposition to the grid).
Accumulation routines are an integral of PIC algorithms, as they are the kernels for computing :ref:`monte_carlo`.

Accumulators are implemented within the following sub-modules:

.. currentmodule:: ''

.. autosummary::
    :nosignatures:
    :toctree: STUBDIR

    struphy.pic.accumulation.particles_to_grid
    struphy.pic.accumulation.accum_kernels
    struphy.pic.accumulation.accum_kernels_gc

.. toctree::
    :caption: Lists of available accumulators:

    STUBDIR/struphy.pic.accumulation.particles_to_grid
    STUBDIR/struphy.pic.accumulation.accum_kernels
    STUBDIR/struphy.pic.accumulation.accum_kernels_gc


.. _accumulator:

Base classes
------------

.. automodule:: struphy.pic.accumulation.particles_to_grid
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


.. _accum_kernels:

6D accumulation kernels
-----------------------

.. automodule:: struphy.pic.accumulation.accum_kernels
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


.. _accum_kernels_gc:

5D accumulation kernels
-----------------------

.. automodule:: struphy.pic.accumulation.accum_kernels_gc
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:

