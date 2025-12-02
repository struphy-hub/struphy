.. _normalization:

Normalization
-------------

Struphy models are implemented in normalized variables. 
In general, any dependent or independent variable :math:`X` is expressed as 

.. math::

    X = X'\,\hat X\,,

where :math:`\hat X` is the unit and :math:`X'` is the numerical value
appearing in the code. For example, the same length :math:`a` could be either
expressed as :math:`a = 2 \cdot 1\, \textrm{meter}` or as
:math:`a = 4 \cdot 0.5\, \textrm{meter}`, where in the second case the unit 
of length was chosen to be 0.5 meter.

The units :math:`\hat X` for a Struphy model
can be influenced by the user through :class:`~struphy.io.options.BaseUnits` in the launch file,
where the user can set

* the unit of **length** :math:`\hat x`, expressed in **Meter**,

* the unit of the **magnetic field strength** :math:`\hat B`, expressed in **Tesla**,

* the unit of **number density** :math:`\hat n`, expressed in :math:`\mathbf{10^{20}\,m^{-3}}`.

* optional: the unit of **thermal energy** :math:`k_\textnormal{B} \hat T`, expressed in **keV**.

This immediately gives meaning to the numerical values of
these quantities appearing in the parameter file. Additionally,

* the unit of **velocity** :math:`\hat v`, expressed in **Meters/Second**,

is hard coded for each model
under the attribute :attr:`~struphy.models.base.StruphyModel.velocity_scale`.
There are four possibilities:

1. speed of ``light``, :math:`\hat v = c`.

2. ``alfvén`` speed of the bulk species, 

.. math::
    
    \hat v = v_\textnormal{A, bulk} := \sqrt{\hat B^2 / (m_\textnormal{bulk} \hat n \mu_0)}\,.

3. ``cyclotron`` speed of the bulk species, 

.. math::
    
    \hat v = v_\textnormal{c, bulk} := \hat x \Omega_\textnormal{c, bulk}/(2\pi) = \hat x\, q_\textnormal{bulk} \hat B /(m_\textnormal{bulk}2\pi)\,.

4. ``thermal`` velocity of the bulk species,

.. math::

    \hat v = \sqrt{\frac{k_\textnormal{B} \hat T}{m_\textnormal{bulk}}}\,.

Several additional units are derived internally from the above basic units, 
in the class :class:`~struphy.io.options.Units`. In particular,

* the **time** unit in **Seconds**:

.. math::

    \hat t = \frac{\hat x}{\hat v} \,,

* the **pressure** unit in **Pascal**:

.. math::

    \hat p = m_\textnormal{bulk} \hat n \hat v^2\,,

which is equal to :math:`\hat B^2/\mu_0` if the velocity scale is ``Alfvén``,

* the **mass density** unit in :math:`\mathbf{kg/m^3}`:

.. math::

    \hat \rho = m_\textnormal{bulk} \hat n

* the current density unit in **Ampere**:math:`\boldsymbol{/m^2}`:

.. math::

    \hat \jmath = q_\textnormal{bulk} \hat n \hat v\,. 

We refer to :ref:`disc_example` for an example of how to derive a normalization for a physics model.


Units class
-----------

.. autoclass:: struphy.io.options.Units
    :members:
    :undoc-members:


.. _equation_params:

Equation parameters
-------------------

In Struphy models, the following equation parameters appear:

.. math::

        \alpha_\textrm{s} = \frac{\hat \Omega_\textnormal{ps}}{\hat \Omega_\textnormal{cs}}\,,\qquad \varepsilon_\textrm{s} = \frac{1}{\hat \Omega_\textnormal{cs} \hat t} \,,

featuring the plasma- and cyclotron frequency of species :math:`\textrm{s}`, respectively,

.. math::

        \hat\Omega_\textnormal{ps} = \sqrt{\frac{\hat n (Z_\textrm{s}e)^2}{\epsilon_0 (A_\textrm{s} m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{cs}} = \frac{(Z_\textrm{s}e) \hat B}{(A_\textrm{s} m_\textnormal{H})}\,,

where :math:`Z_\textrm{s}` and :math:`A_\textrm{s}` stand for the species' charge and mass number, respectively.
These equation parameters are defined in :class:`~struphy.models.species.Species.EquationParameters` and can be overridden
in the launch file via :func:`~struphy.models.species.Species.set_phys_params`.

.. autoclass:: struphy.models.species.Species.EquationParameters
    :members:
    :undoc-members:

