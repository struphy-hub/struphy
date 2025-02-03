.. _Tutorial 1 - Particles: ../tutorials/tutorial_01_test_particles.ipynb
.. _Tutorial 5 - Poisson: ../tutorials/tutorial_02_poisson.ipynb
.. _Tutorial 6 - Heat equation: ../tutorials/tutorial_03_heat_equation.ipynb
.. _Tutorial 7 - Maxwell equations: ../tutorials/tutorial_04_maxwell.ipynb
.. _Tutorial 8 - Vlasov-Maxwell: ../tutorials/tutorial_05_vlasov_maxwell.ipynb
.. _Tutorial 9 - Linear MHD equations: ../tutorials/tutorial_06_linear_mhd.ipynb

.. _propagators:

Propagators
===========

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of every algorithm. A propagator is used to advance a subset
of a model's variables by one time step, :math:`t \to t + \Delta t`.

Check out the following tutorials for how to use propagators in Struphy:

* `Tutorial 1 - Particles`_
* `Tutorial 5 - Poisson`_
* `Tutorial 6 - Heat equation`_
* `Tutorial 7 - Maxwell equations`_
* `Tutorial 8 - Vlasov-Maxwell`_
* `Tutorial 9 - Linear MHD equations`_

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    subsections/propagators_fields
    subsections/propagators_markers
    subsections/propagators_coupling
    subsections/propagator_base_class



