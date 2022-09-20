
import numpy as np
from struphy.models.base import StruphyModel


class Maxwell(StruphyModel):
    r'''Maxwell's equations in vacuum. 
    
    Normalization:

    .. math::

        c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,,

    where :math:`c` is the vacuum speed of light. Implemented equations:

    .. math::
    
        &\frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0\,, 
        
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    Parameters
    ----------
        derham: struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        domain: struphy.geometry.base.Domain
            All things mapping.

        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, derham, domain, params):

        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.propagators.propagators import StepMaxwell

        super().__init__(derham, domain, params, e_field='Hcurl', b_field='Hdiv')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(derham, domain)
        self._mass_ops.assemble_M1()
        self._mass_ops.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = .5*self._e.dot(self._mass_ops.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = .5*self._b.dot(self._mass_ops.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + \
                                               self._scalar_quantities['en_B'][0]


class LinearMHD(StruphyModel):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). 
    
    Normalization: 

    .. math::

        \frac{\hat B}{\sqrt{\mu_0\,A\, m_\textnormal{p}\,\hat n}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,,
        
    where :math:`m_\textnormal{p}` is the proton mass, :math:`A` the mass number of the ion species and :math:`\mu_0` the vacuum permeability. Implemented equations:

    .. math::

        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 

        n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,
        
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        
    where the equilibrium quantities must satisfy the MHD equilibrium condition :math:`\nabla p_0=\mathbf J_0\times\mathbf B_0`.

    Parameters
    ----------
        derham: struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        domain: struphy.geometry.base.Domain
            All things mapping.

        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, derham, domain, params):
        
        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.psydac_api.mhd_ops_pure_psydac import MHDOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators
        
        self._u_space = params['fields']['mhd_u_space']
        
        if self._u_space == 'Hdiv':
            super().__init__(derham, domain, params, n3='L2', u2=self._u_space, p3='L2', b2='Hdiv')
        else:
            super().__init__(derham, domain, params, n3='L2', uv=self._u_space, p3='L2', b2='Hdiv')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        shearalfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']

        # Load MHD equilibrium
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(equil_params[equil_params['type']], domain)
        
        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(derham, domain, eq_mhd=mhd_equil)
        
        self._mass_ops.assemble_M2()
        self._mass_ops.assemble_M3()
        
        if self._u_space == 'Hdiv':
            self._mass_ops.assemble_M2n()
            self._mass_ops.assemble_M2J()
        else:
            self._mass_ops.assemble_Mvn()
            self._mass_ops.assemble_MvJ()
        
        # Assemble necessary linear MHD projection operators
        self._mhd_ops = MHDOperators(derham, domain, mhd_equil)
        
        if self._u_space == 'Hdiv':
            self._mhd_ops.assemble_K2()
            self._mhd_ops.assemble_Q2()
            self._mhd_ops.assemble_T2()
            self._mhd_ops.assemble_S2()
        else:
            self._mhd_ops.assemble_K0()
            self._mhd_ops.assemble_Q0()
            self._mhd_ops.assemble_T0()
            self._mhd_ops.assemble_S0()
            self._mhd_ops.assemble_J0()
        
        # Pointers to Stencil-/Blockvectors
        self._n = self.fields[0].vector
        self._u = self.fields[1].vector
        self._p = self.fields[2].vector
        self._b = self.fields[3].vector
        
        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        
        ShearAlfven = getattr(propagators, 'StepShearAlfvén' + str(self._u_space))
        Magnetosonic = getattr(propagators, 'StepMagnetosonic' + str(self._u_space)) 
            
        self._propagators += [ShearAlfven(self._u, self._b, derham, self._mass_ops, self._mhd_ops, shearalfven_solver)]
        self._propagators += [Magnetosonic(self._n, self._u, self._p, self._b, derham, self._mass_ops, self._mhd_ops, magnetosonic_solver)]
        
        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators

    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        
        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_tot'][0]  = self._scalar_quantities['en_U'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


