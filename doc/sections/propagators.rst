.. _propagators:

Propagators
===========

Documented modules:

.. currentmodule:: ''

.. autosummary::
    :nosignatures:
    :toctree: STUBDIR

    struphy.propagators.base
    struphy.propagators.propagators_markers
    struphy.propagators.propagators_fields
    struphy.propagators.propagators_coupling
    struphy.pic.pusher
    struphy.pic.pusher_kernels
    struphy.pic.pusher_kernels_gc

.. toctree::
    :caption: Lists of available propagators (and particle pushers therein):

    STUBDIR/struphy.propagators.base
    STUBDIR/struphy.propagators.propagators_markers
    STUBDIR/struphy.propagators.propagators_fields
    STUBDIR/struphy.propagators.propagators_coupling
    STUBDIR/struphy.pic.pusher
    STUBDIR/struphy.pic.pusher_kernels
    STUBDIR/struphy.pic.pusher_kernels_gc

Notation:

================= ============================================ ==========================================================
Symbol            Example                                      Meaning
================= ============================================ ==========================================================
bold letter       :math:`\mathbf{e},\,\mathbf{b}`              Vector in :math:`\mathbb R^N`
upper index *n*   :math:`\mathbf{e}^{n}`                       Vector in :math:`\mathbb R^N` at time :math:`t^n=n \Delta t`
blackboard bold   :math:`\mathbb G,\,\mathbb C,\,\mathbb D`    grad, curl, div matrices
blackboard bold M :math:`\mathbb M_1`                          Mass matrices
calligaphic       :math:`\mathcal T,\,\mathcal U,\,\mathcal K` Basis projection operator, see :ref:`_mhd_ops`
================= ============================================ ==========================================================

See :ref:`gempic` for more details on the used symbols.


.. _prop_base:

Propagator base class
---------------------

.. automodule:: struphy.propagators.base
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Particle propagators
--------------------

.. automodule:: struphy.propagators.propagators_markers
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Field propagators
-----------------

.. automodule:: struphy.propagators.propagators_fields
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Particle-field propagators
--------------------------

.. automodule:: struphy.propagators.propagators_coupling
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


.. _pushers:

Pusher base classes
-------------------

.. automodule:: struphy.pic.pusher
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Pusher kernels
--------------

.. automodule:: struphy.pic.pusher_kernels
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:

.. automodule:: struphy.pic.pusher_kernels_gc
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance: