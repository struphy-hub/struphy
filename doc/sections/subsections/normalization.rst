.. _normalization:

Normalization
-------------

In Struphy, models are implemented in normalized variables. 
Each particular normalization is stated with the model equations in the docstring (see below).
For any dependent or independent variable :math:`X`, we write 

.. math::

    X = \hat X\, X'\,,

where :math:`\hat X` is the unit (for instance 0.1 meter) and :math:`X'` is the numerical value
appearing in the code. Choosing a normalization means that units of different variables
are not independent anymore. For instance, choosing

.. math::
    :label: timescale

    \hat t = \frac{\hat x}{\hat v} \,,

means that the time unit :math:`\hat t` is fixed once the units of velocity :math:`\hat v`
and the unit of length :math:`\hat x` have been chosen. In this case we say that
:math:`\hat t` is a "derived unit" and :math:`\hat v` and :math:`\hat x` are "basic units".

In Struphy, all units can be fixed by means of four basic units. Three of these basic units 
can be prescribed in the :code:`parameter.yml` file by the user, namely:

1. the unit of length :math:`\hat x`, expressed in Meter,

2. the unit of the magnetic field strength :math:`\hat B`, expressed in Tesla,

3. the unit of number density :math:`\hat n`, expressed in :math:`10^{20}\,m^{-3}`.

Additionally, the fourth basic unit is

4. the unit of veloctiy :math:`\hat v`.

This unit is hard coded for each model
under the attribute `velocity_scale <https://struphy.pages.mpcdf.de/struphy/sections/developers.html#struphy.models.base.StruphyModel.velocity_scale>`_.
There are currently three possibilities for ``velocity_scale``:

* speed of light, :math:`\hat v = c`,

* Alfvén speed of the bulk species, 

.. math::
    
    \hat v = v_\textnormal{A, bulk} := \sqrt{\hat B^2 / (m_\textnormal{bulk} \hat n \mu_0)}\,,

* Cyclotron speed of the bulk species, 

.. math::
    
    \hat v = v_\textnormal{c, bulk} := \hat x \Omega_\textnormal{c, bulk}/(2\pi) = \hat x\, q_\textnormal{bulk} \hat B /(m_\textnormal{bulk}2\pi)\,.

With the velocity scale set by the model developer, the time scale :math:`\hat t` in Struphy is always determined by :eq:`timescale`.
The numerical values of all derived units, for any :code:`MODEL` with a parameter file :code:`FILE`, can be inspected via::

    struphy units MODEL -i FILE

For the derivation of these units, several physics parameters are needed, of which 
the notation is summarized in the following table:

======================== =================================================== 
Symbol                   Meaning                                      
======================== ===================================================
:math:`\mu_0`            magnetic constant             
:math:`m_\textnormal{H}` proton mass     
:math:`A_\textnormal{b}` mass number of bulk species (in units of proton mass)   
:math:`e`                elementary charge (positive)
:math:`Z_\textnormal{b}` charge number of bulk species (in units of elementary charge)
:math:`c`                vacuum speed of light
======================== ===================================================  

We refer to :ref:`disc_example` for an example of how to derive a normalization for a physics model.
In addition, `Tutorial 01 <https://struphy.pages.mpcdf.de/struphy/tutorials/tutorial_01_units_run_main.html#Struphy-normalization-(units)>`_ 
provides some insights into the use of Struphy units.