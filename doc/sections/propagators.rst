.. _propagators:

Propagators
===========

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


Particle propagators
--------------------

.. automodule:: struphy.propagators.propagators_markers
    :members:
    :undoc-members:
    :exclude-members: variables


Field propagators
-----------------

.. automodule:: struphy.propagators.propagators_fields
    :members:
    :undoc-members:
    :exclude-members: variables


Particle-field propagators
--------------------------

.. automodule:: struphy.propagators.propagators_coupling
    :members:
    :undoc-members:
    :exclude-members: variables