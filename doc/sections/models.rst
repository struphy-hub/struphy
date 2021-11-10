.. _models:

Model equations
===============

Currently implemented initial-value solvers on 3D mapped domains are (black: bulk plasma, red: EPs):

    * **Current coupling** between linearized, ideal MHD equations and 6D full-orbit Vlasov equation:

        (call option ``struphy -r cc_lin_mhd_6d``, set as default)

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_\text{eq} \tilde{\mathbf{U}})=0\,, 

        \rho_\text{eq}&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_\text{eq} + \mathbf{J}_\text{eq}\times \tilde{\mathbf{B}}
        \color{red}+ (\rho_\text{h} \tilde{\mathbf{U}}-\mathbf{j}_\text{h})\times (\mathbf{B}_\textnormal{eq}
        + \tilde{\mathbf{B}})\,, \qquad
        \color{black} \mathbf{J}_\textnormal{eq} = \nabla\times\mathbf{B}_\text{eq}\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_\text{eq} \tilde{\mathbf{U}}) 
        + (\gamma-1)p_\text{eq}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_\text{eq})
        = 0\,,

        &\color{red} \frac{\partial f_\text{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\text{h}
        +\Big[(\mathbf{v} - \tilde{\mathbf{U}}) \times (\mathbf{B}_\text{eq} 
        + \tilde{\mathbf{B}}) \Big]\cdot\nabla_\mathbf{v}f_\text{h}=0\,,
        \qquad\rho_\text{h}=\int f_\text{h}\,\text{d}^3v,\qquad\mathbf{j}_\text{h}=\int\mathbf{v}f_\text{h}\,\text{d}^3v\,.

    * **Pressure coupling** between linearized, ideal MHD equations and 6D full-orbit Vlasov equation:

        (call option ``struphy -r pc_lin_mhd_6d``)

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_\text{eq} \tilde{\mathbf{U}})=0\,, 

        \rho_\text{eq}&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_\text{eq} + \mathbf{J}_\text{eq}\times \tilde{\mathbf{B}}
        \color{red} - (\nabla \cdot \mathbb P_\text{h})_\perp\,, \qquad
        \color{black} \mathbf{J}_\textnormal{eq} = \nabla\times\mathbf{B}_\text{eq}\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_\text{eq} \tilde{\mathbf{U}}) 
        + (\gamma-1)p_\text{eq}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_\text{eq})
        = 0\,,

        &\color{red} \frac{\partial f_\text{h}}{\partial t}+( \mathbf{v} - \tilde{\mathbf U})\cdot\nabla f_\text{h}
        +\Big[\mathbf{v}  \times (\mathbf{B}_\text{eq} 
        + \tilde{\mathbf{B}}) - \mathbf v_\perp \cdot \nabla \tilde{\mathbf U} \Big]\cdot\nabla_\mathbf{v}f_\text{h}=0\,,
        \qquad (\nabla \cdot \mathbb P_\text{h})_\perp=\int\mathbf{v}_\perp \mathbf{v}^\top f_\text{h}\,\text{d}^3v\,.

    * **Electron cold plasma hybrid model** between linearized momentum conservation law and Vlasov-Maxwell:

        (call option ``struphy -r cold_electrons``)

    .. math::
        &\frac{\partial\tilde{\mathbf{j}}_\text{c}}{\partial t}=\epsilon_0\Omega_\text{pe}^2\tilde{\mathbf{E}}
        + \tilde{\mathbf{j}}_\text{c}\times \frac{q_\text{e}}{m_\text{e}} \mathbf{B}_0(\mathbf{x})\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t}=-\nabla\times\tilde{\mathbf{E}}\,,

        &\frac{1}{c^2}\frac{\partial \tilde{\mathbf{E}}}{\partial t}=\nabla\times\tilde{\mathbf{B}}
        - \mu_0\tilde{\mathbf{j}}_\text{c} \color{red} - \mu_0q_\text{e}\int\mathbf{v}\tilde{f}_\text{h}\,\text{d}^3\mathbf{v}\,,

        &\color{red} \frac{\partial f_\text{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\text{h}+\frac{q_\text{e}}{m_\text{e}}(\mathbf{E}
        + \mathbf{v}\times\mathbf{B})\cdot\nabla_\mathbf{v}f_\text{h}=0\,.

    * **6D kinetic thermal ions** with mass-less electrons and extended Ohm's law (no EPs thus far):

        (call option ``struphy -r kinetic_extended``)

    .. math::
        &\frac{\partial f}{\partial t} + {\mathbf v} \cdot \frac{\partial f}{\partial \mathbf x} 
        +  ({\mathbf E} + {\mathbf v} \times {\mathbf B}) \cdot \frac{\partial f}{\partial {\mathbf v}} = 0\,,

        &\frac{\partial \mathbf B}{\partial t} = - \nabla \times {\mathbf E}\,,\qquad {\mathbf E} = -{\mathbf u} \times {\mathbf B} - \frac{\kappa T}{n}\nabla n 
        + \frac{\nabla \times {\mathbf B}}{n} \times {\mathbf B}\,,

        &n = \int f\, \text{d}^3v, \quad n{\mathbf u} = \int {\mathbf v} f \, \text{d}^3v\,.

