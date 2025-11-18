.. _propagators:

Propagators
===========

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of every algorithm. A propagator is used to advance a subset
of a model's variables by one time step, :math:`t \to t + \Delta t`.

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    subsections/propagators-fields
    subsections/propagators-markers
    subsections/propagators-coupling
    subsections/propagators-base_class



