.. _propagators:

Propagators
-----------

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of every algorithm.
Check out :ref:`disc_example` for how propagators are used in Struphy.

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

.. inheritance-diagram:: struphy.propagators.propagators_markers
    :parts: 1

.. automodule:: struphy.propagators.propagators_markers
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Field propagators
^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: struphy.propagators.propagators_fields
    :parts: 1

.. automodule:: struphy.propagators.propagators_fields
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:


Particle-field propagators
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: struphy.propagators.propagators_coupling
    :parts: 1

.. automodule:: struphy.propagators.propagators_coupling
    :members:
    :undoc-members:
    :exclude-members: variables
    :show-inheritance:

