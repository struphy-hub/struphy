.. _avail_mappings:

3D mapped domains
-----------------

Struphy models are implemented in curvilinear coordinates and can be run
on a variaty of mapped domains. 
Besides analytical mappings, there are also discrete spline mappings available (IGA approach).
The following inheritance diagram shows the existing mapped domains:


.. inheritance-diagram:: struphy.geometry.domains
    :parts: 1


Base classes
^^^^^^^^^^^^

.. automodule:: struphy.geometry.base
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__


Available domains
^^^^^^^^^^^^^^^^^

.. automodule:: struphy.geometry.domains
    :members:
    :exclude-members: kind_map, params_map, params_numpy, pole, periodic_eta3
    :show-inheritance:


.. _field_tracing:

Utilities
^^^^^^^^^

.. automodule:: struphy.geometry.utilities
    :members:
    :show-inheritance: