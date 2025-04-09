.. _Tutorial 1 - Kinetic particles: ../tutorials/tutorial_01_kinetic_particles.ipynb
.. _Tutorial 2 - Fluid particles: ../tutorials/tutorial_02_fluid_particles.ipynb
.. _Tutorial 6 - Poisson: ../tutorials/tutorial_06_poisson.ipynb
.. _Tutorial 7 - Heat equation: ../tutorials/tutorial_07_heat_equation.ipynb
.. _Tutorial 8 - Maxwell equations: ../tutorials/tutorial_08_maxwell.ipynb
.. _Tutorial 9 - Vlasov-Maxwell: ../tutorials/tutorial_09_vlasov_maxwell.ipynb
.. _Tutorial 10 - Linear MHD equations: ../tutorials/tutorial_10_linear_mhd.ipynb

.. _propagators:

Propagators
===========

Propagators are the main building blocks of :ref:`models`, as they define the 
time splitting scheme of every algorithm. A propagator is used to advance a subset
of a model's variables by one time step, :math:`t \to t + \Delta t`.

Check out the following tutorials for how to use propagators in Struphy:

* `Tutorial 1 - Kinetic particles`_
* `Tutorial 2 - Fluid particles`_
* `Tutorial 6 - Poisson`_
* `Tutorial 7 - Heat equation`_
* `Tutorial 8 - Maxwell equations`_
* `Tutorial 9 - Vlasov-Maxwell`_
* `Tutorial 10 - Linear MHD equations`_

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    subsections/propagators_fields
    subsections/propagators_markers
    subsections/propagators_coupling
    subsections/propagator_base_class



