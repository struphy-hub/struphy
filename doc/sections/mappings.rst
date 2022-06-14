In :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` equations are discretized on the unit cube (*logical domain*)
:math:`\hat \Omega = [0,1]^3` with logical coordinates :math:`\boldsymbol \eta = (\eta_1, \eta_2,\eta_3) \in \hat \Omega`. 
The *physical domain* :math:`\Omega \subset \mathbb R^3` is the image of :math:`\hat\Omega` under the mapping

   .. math::
      F:\hat{\Omega}\rightarrow\Omega,\,\,(\eta_1, \eta_2,\eta_3) = \boldsymbol \eta \mapsto \mathbf x =  (x, y, z) = F(\eta_1, \eta_2,\eta_3).

Hence, :math:`\mathbf{x} := (x, y, z)` are global or "Cartesian" coordinates of :math:`\Omega` and
:math:`\mathbf \eta` are local, curvilinear coordinates of :math:`\Omega`.
The map *F* is assumed to be :math:`\mathcal{C}^1` everywhere except at one polar point (which can be treated with :ref:`polar_splines`).
The Jacobian matrix, its determinant and the metric tensor are denoted by

   .. math::
      DF_{i,j} = \frac{\partial F_i}{\partial \eta_j}\,,\qquad \sqrt g := |\det(DF)|\,,\qquad G := DF^\top DF\,.


The mapping *F* and corresponding metric coefficients are handled via the *Domain class* in :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)`.
This class has four important methods (described below):

   - ``evaluate`` for evaluation of metric coefficients
   - ``pull`` for the pullback of p-forms 
   - ``push`` for the pushforward of p-forms
   - ``transform`` for the transformation between p-forms

Summary of pull-back and push-forward transformations between generic scalar and vector-valued functions
:math:`f=f(\mathbf{x})` and :math:`\mathbf{V}=[V_x(\mathbf{x}),V_y(\mathbf{x}),V_z(\mathbf{x})]`, respectively, 
and differential *k*-forms (:math:`0\leq k\leq 3`) under the map :math:`F:\hat{\Omega}\rightarrow\Omega,\,\,\boldsymbol{\eta}\mapsto\mathbf{x}=F(\boldsymbol{\eta})`.

.. image:: ../pics/pforms_table.png

The list of availabe mapppings in :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` can be read from the Notes below.
The entry in ``parameters.yml`` to initialize the Domain class can be found in :ref:`params_geometry`.

.. autoclass:: struphy.geometry.domain_3d.Domain
   :members: 
   :undoc-members:

Input arguments of these routines are handled via

.. automodule:: struphy.geometry.domain_3d
   :members: prepare_args


