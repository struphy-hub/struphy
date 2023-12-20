.. _propagators:

Propagators
-----------

This page lists currently available Struphy propagators (for time stepping :math:`t \to t + \Delta t`):

- `field propagators <https://struphy.pages.mpcdf.de/struphy/sections/STUBDIR/struphy.propagators.propagators_fields.html>`_
- `marker propagators <https://struphy.pages.mpcdf.de/struphy/sections/STUBDIR/struphy.propagators.propagators_markers.html>`_
- `hybrid propagators <https://struphy.pages.mpcdf.de/struphy/sections/STUBDIR/struphy.propagators.propagators_coupling.html>`_

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of every algorithm.
Check out :ref:`add_prop` for a manual on writing new propagators.

Propagators are implemented within the following sub-modules:

.. currentmodule:: ''

.. autosummary::
    :nosignatures:
    :toctree: STUBDIR

    struphy.propagators.base
    struphy.propagators.propagators_markers
    struphy.propagators.propagators_fields
    struphy.propagators.propagators_coupling
    struphy.pic.pushing.pusher
    struphy.pic.pushing.pusher_kernels
    struphy.pic.pushing.pusher_kernels_gc
    struphy.pic.pushing.eval_kernels_gc

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
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.propagators.base
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Particle propagators
^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.propagators.propagators_markers
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Field propagators
^^^^^^^^^^^^^^^^^

.. automodule:: struphy.propagators.propagators_fields
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Particle-field propagators
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.propagators.propagators_coupling
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


.. _pushers:

Pusher base classes
^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.pushing.pusher
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Pusher kernels
^^^^^^^^^^^^^^

.. automodule:: struphy.pic.pushing.pusher_kernels
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Pusher kernels guiding center
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.pushing.pusher_kernels_gc
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:

.. automodule:: struphy.pic.pushing.eval_kernels_gc
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:
