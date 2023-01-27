.. _models:

Available models
================

.. note::

    All models are implemented with a special STRUPHY normalization stated with the model equations. 
    The notation is such that hat quantities like :math:`\hat \omega` denote the unit of a quantity, i.e.
    :math:`1/\hat \omega` denotes the time unit, :math:`1/\hat k` is the unit of length and :math:`\hat B` 
    stands for the unit of the magnetic field.


Fluid models
------------

.. autoclass:: struphy.models.models.LinearMHD
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities


Fluid-kinetic hybrid models
---------------------------

.. autoclass:: struphy.models.models.HybridMHDVlasovCC
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities

.. autoclass:: struphy.models.models.HybridMHDVlasovPC
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities


Kinetic models
--------------

.. autoclass:: struphy.models.models.LinearVlasovMaxwell
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities


Toy models
----------

.. autoclass:: struphy.models.models.Maxwell
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities

.. autoclass:: struphy.models.models.Vlasov
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities

.. autoclass:: struphy.models.models.DriftKinetic
    :members:
    :undoc-members:
    :exclude-members: propagators, scalar_quantities, update_scalar_quantities


.. .. _cc_lin_mhd_6d:

.. MHD-kinetic current coupling (CC)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
.. Linear, ideal MHD equations and 6D full-orbit Vlasov equation, command line call ``struphy -r cc_lin_mhd_6d``.

.. .. math::
..     &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_\text{eq} \tilde{\mathbf{U}})=0\,, 

..     \rho_\text{eq}&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
..     =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_\text{eq} + \mathbf{J}_\text{eq}\times \tilde{\mathbf{B}}
..     \color{red}+ (\rho_\text{h} \tilde{\mathbf{U}}-\mathbf{j}_\text{h})\times (\mathbf{B}_\textnormal{eq}
..     + \tilde{\mathbf{B}})\,, \qquad
..     \color{black} \mathbf{J}_\textnormal{eq} = \nabla\times\mathbf{B}_\text{eq}\,,

..     &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_\text{eq} \tilde{\mathbf{U}}) 
..     + (\gamma-1)p_\text{eq}\nabla\cdot \tilde{\mathbf{U}}=0\,,
    
..     &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_\text{eq})
..     = 0\,,

..     &\color{red} \frac{\partial f_\text{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\text{h}
..     +\Big[(\mathbf{v} - \tilde{\mathbf{U}}) \times (\mathbf{B}_\text{eq} 
..     + \tilde{\mathbf{B}}) \Big]\cdot\nabla_\mathbf{v}f_\text{h}=0\,,
..     \qquad\rho_\text{h}=\int f_\text{h}\,\text{d}^3v,\qquad\mathbf{j}_\text{h}=\int\mathbf{v}f_\text{h}\,\text{d}^3v\,.


.. .. _pc_lin_mhd_6d:

.. MHD-kinetic pressure coupling (PC)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

.. Linear, ideal MHD equations and 6D full-orbit Vlasov equation, command line call ``struphy -r pc_lin_mhd_6d_perp``.

.. .. math::
..     &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_\text{eq} \tilde{\mathbf{U}})=0\,, 

..     \rho_\text{eq}&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
..     =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_\text{eq} + \mathbf{J}_\text{eq}\times \tilde{\mathbf{B}}
..     \color{red} - (\nabla \cdot \mathbb P_\text{h})_\perp\,, \qquad
..     \color{black} \mathbf{J}_\textnormal{eq} = \nabla\times\mathbf{B}_\text{eq}\,,

..     &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_\text{eq} \tilde{\mathbf{U}}) 
..     + (\gamma-1)p_\text{eq}\nabla\cdot \tilde{\mathbf{U}}=0\,,
    
..     &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_\text{eq})
..     = 0\,,

..     &\color{red} \frac{\partial f_\text{h}}{\partial t}+( \mathbf{v} - \tilde{\mathbf U})\cdot\nabla f_\text{h}
..     +\Big[\mathbf{v}  \times (\mathbf{B}_\text{eq} 
..     + \tilde{\mathbf{B}}) - \mathbf v_\perp \cdot \nabla \tilde{\mathbf U} \Big]\cdot\nabla_\mathbf{v}f_\text{h}=0\,,
..     \qquad (\nabla \cdot \mathbb P_\text{h})_\perp=\int\mathbf{v}_\perp \mathbf{v}^\top f_\text{h}\,\text{d}^3v\,.


.. .. _cold_plasma:

.. Electron cold plasma hybrid model
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Linear momentum conservation law and Vlasov-Maxwell, command line call ``struphy -r cold_plasma``.

.. .. math::
..     &\frac{\partial\tilde{\mathbf{j}}_\text{c}}{\partial t}=\epsilon_0\Omega_\text{pe}^2\tilde{\mathbf{E}}
..     + \tilde{\mathbf{j}}_\text{c}\times \frac{q_\text{e}}{m_\text{e}} \mathbf{B}_0(\mathbf{x})\,,

..     &\frac{\partial \tilde{\mathbf{B}}}{\partial t}=-\nabla\times\tilde{\mathbf{E}}\,,

..     &\frac{1}{c^2}\frac{\partial \tilde{\mathbf{E}}}{\partial t}=\nabla\times\tilde{\mathbf{B}}
..     - \mu_0\tilde{\mathbf{j}}_\text{c} \color{red} - \mu_0q_\text{e}\int\mathbf{v}\tilde{f}_\text{h}\,\text{d}^3\mathbf{v}\,,

..     &\color{red} \frac{\partial f_\text{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\text{h}+\frac{q_\text{e}}{m_\text{e}}(\mathbf{E}
..     + \mathbf{v}\times\mathbf{B})\cdot\nabla_\mathbf{v}f_\text{h}=0\,.


.. .. _kinetic_extended:

.. Kinetic thermal ions with extended Ohm's law
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. 6D Vlasov equation with mass-less fluid electrons, quasi-neutral (no EPs thus far), command line call ``struphy -r kinetic_extended``.

.. .. math::
..     &\frac{\partial f}{\partial t} + {\mathbf v} \cdot \frac{\partial f}{\partial \mathbf x} 
..     +  ({\mathbf E} + {\mathbf v} \times {\mathbf B}) \cdot \frac{\partial f}{\partial {\mathbf v}} = 0\,,

..     &\frac{\partial \mathbf B}{\partial t} = - \nabla \times {\mathbf E}\,,\qquad {\mathbf E} = -{\mathbf u} \times {\mathbf B} - \frac{\kappa T}{n}\nabla n 
..     + \frac{\nabla \times {\mathbf B}}{n} \times {\mathbf B}\,,

..     &n = \int f\, \text{d}^3v, \quad n{\mathbf u} = \int {\mathbf v} f \, \text{d}^3v\,.


.. .. _cc_lin_mhd_6d_axissymm:

.. Initial-value solvers for axis-symmetric systems (2D)
.. -----------------------------------------------------

.. Axis-symmetric CC
.. ^^^^^^^^^^^^^^^^^
    
.. Same model equations as in :ref:`cc_lin_mhd_6d`, with single toroidal mode number, command line call ``struphy -r cc_lin_mhd_6d_axissymm``.
.. The ansatz for the MHD variables is (with *k* fixed)

.. .. math::
..     \mathbf U(t;s,\theta,\varphi) = \mathbf U_\text{pol}(t;s,\theta)\,\text{e}^{-i k \varphi}\,.


.. .. _mhd_eig_axissymm:

.. Eigenvalue solvers
.. ------------------

.. Ideal MHD in axis-symmetric equilibrium
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Ideal MHD equations with single toroidal mode number, command line call ``struphy -r mhd_eig_axissymm``.
.. The eigenvalue problem reads

.. .. math::
..     -\omega^2\rho_\text{eq}\,\mathbf{U} &= \mathbf F(\mathbf{U})\,, 
    
..     \mathbf F(\mathbf{U}) &= \nabla p(\mathbf{U}) - \big[\nabla\times\mathbf{B}(\mathbf{U}) \big]\times\mathbf{B}_\text{eq} - \mathbf{J}_\text{eq}\times\mathbf{B}(\mathbf{U})\,,

..     p(\mathbf{U}) &= \nabla\cdot(p_\text{eq}\,\mathbf{U}) + (\gamma-1)\,p_\text{eq}\,\nabla\cdot\mathbf{U}\,,

..     \mathbf{B}(\mathbf{U}) &= \nabla\times(\mathbf{B}_\text{eq}\times \mathbf{U}) \,.