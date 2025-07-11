.. _equils_avail:

Available fluid equilibria
^^^^^^^^^^^^^^^^^^^^^^^^^^

Aside form the classes listed below, the fluid background ``LogicalConst``
is available for simple testing; it has the following input structure::

    LogicalConst :
        values : 1.3 

or, for vector-valued variables::

    LogicalConst :
        values : [.3, .15, null] 

.. automodule:: struphy.fields_background.equils
    :members:
    :undoc-members:
    :exclude-members: set_defaults
    :show-inheritance:


.. _projected_equils_avail:

Projected fluid equilibria
^^^^^^^^^^^^^^^^^^^^^^^^^^

These classes provide discrete representations of fluid equilibria in De Rham spaces.

.. automodule:: struphy.fields_background.projected_equils
    :members:
    :undoc-members:
    :exclude-members: set_defaults
    :show-inheritance:


.. _generic_equils:

Generic fluid equilibria
^^^^^^^^^^^^^^^^^^^^^^^^

These classes can be used to quickly pass callables to Struphy objects 
(for instance in Jupyter notebooks).

.. automodule:: struphy.fields_background.generic
    :members:
    :undoc-members:
    :exclude-members: set_defaults
    :show-inheritance:


.. _mhd_base:

Base classes
^^^^^^^^^^^^

.. automodule:: struphy.fields_background.base
    :members:
    :undoc-members: 
    :exclude-members: 
    :show-inheritance: