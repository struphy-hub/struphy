.. _avail_mappings:

Geometry
========

Struphy models are implemented in curvilinear coordinates and can be run
on a variaty of mapped domains. 
Besides analytical mappings, there are discrete spline mappings available (IGA approach).

The (physical) domain :math:`\Omega \subset \mathbb R^3` is an open subset of :math:`\mathbb R^3`,
defined by a diffeomorphism 

.. math::

    F:(0, 1)^3 \to \Omega\,,\qquad \boldsymbol{\eta} \mapsto F(\boldsymbol \eta) = \mathbf x\,,

mapping points :math:`\boldsymbol{\eta} \in (0, 1)^3 = \hat\Omega` of the (logical)
unit cube to physical points :math:`\mathbf x \in \Omega`.
The corresponding Jacobain matrix :math:`DF:\hat\Omega \to \mathbb R^{3\times 3}`, 
its volume element :math:`\sqrt g: \hat\Omega \to \mathbb R`
and the metric tensor :math:`G:\hat\Omega \to \mathbb R^{3\times 3}` are defined by

.. math::

    DF_{i,j} = \frac{\partial F_i}{\partial \eta_j}\,,\qquad \sqrt g = |\textnormal{det}(DF)|\,,\qquad G = DF^\top DF\,.

Only right-handed mappings (:math:`\textnormal{det}(DF) > 0`) are admitted.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    subsections/domains-avail
    subsections/domains-base
    subsections/domains-kernels
    subsections/domains-utils
