.. _avail_inits:

Initial conditions
==================

Initial conditions can be specified via the parameters file, as explained in :ref:`params_yml`.
Besides noise, the following pertubations can be applied to all :ref:`backgrounds`:

.. automodule:: struphy.initial.perturbations
    :members:
    :special-members:
    :exclude-members: __weakref__
    
Besides this, model specific analytical initial conditions can be applied:

.. automodule:: struphy.initial.analytic
    :members:
    :special-members:
    :exclude-members: __weakref__