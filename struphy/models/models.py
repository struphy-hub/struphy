import numpy as np
from mpi4py import MPI

from struphy.models.base import StruphyModel
from struphy.pic.utilities import eval_field_at_particles


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
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.propagators.propagators import StepMaxwell

        super().__init__(params, comm, e_field='Hcurl', b_field='Hdiv')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(self.derham, self.domain)
        self._mass_ops.assemble_M1()
        self._mass_ops.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, self.derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators
 
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
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):
        
        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.psydac_api.mhd_ops_pure_psydac import MHDOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators
        
        self._u_space = params['fields']['mhd_u_space']
        
        if self._u_space == 'Hdiv':
            super().__init__(params, comm, n3='L2', u2=self._u_space, p3='L2', b2='Hdiv')
        else:
            super().__init__(params, comm, n3='L2', uv=self._u_space, p3='L2', b2='Hdiv')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        shearalfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']

        # Load MHD equilibrium and project fields
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(equil_params[equil_params['type']], self.domain)
        
        self._b_eq = self.derham.P2([mhd_equil.b2_1, mhd_equil.b2_2, mhd_equil.b2_3]).coeffs
        self._p_eq = self.derham.P3(mhd_equil.p3).coeffs
        
        self._ones = self.derham.V3.vector_space.zeros()
        self._ones[:] = 1.
        
        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(self.derham, self.domain, eq_mhd=mhd_equil)
        
        self._mass_ops.assemble_M2()
        self._mass_ops.assemble_M3()
        
        if self._u_space == 'Hdiv':
            self._mass_ops.assemble_M2n()
            self._mass_ops.assemble_M2J()
        else:
            self._mass_ops.assemble_Mvn()
            self._mass_ops.assemble_MvJ()
        
        # Assemble necessary linear MHD projection operators
        self._mhd_ops = MHDOperators(self.derham, self.domain, mhd_equil)
        
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
            
        self._propagators += [ShearAlfven(self._u, self._b, self.derham, self._mass_ops, self._mhd_ops, shearalfven_solver)]
        self._propagators += [Magnetosonic(self._n, self._u, self._p, self._b, self.derham, self._mass_ops, self._mhd_ops, magnetosonic_solver)]
        
        # Scalar variables to be saved during simulation
        
        # time
        self._scalar_quantities['time']     = np.empty(1, dtype=float)
        
        # energies
        self._scalar_quantities['en_U']     = np.empty(1, dtype=float)
        self._scalar_quantities['en_p']     = np.empty(1, dtype=float)
        self._scalar_quantities['en_B']     = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq']  = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq']  = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot']   = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators
    
    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        
        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.Mvn.dot(self._u))/2
            
        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b))/2
        
        self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(self._mass_ops.M2.dot(self._b_eq))/2
        
        self._scalar_quantities['en_B_tot'][0] = (self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b))/2

        self._scalar_quantities['en_tot'][0]  = self._scalar_quantities['en_U'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


class LinearVlasovMaxwell(StruphyModel):
    r'''The linearized Vlasov Maxwell system with a Maxwellian background distribution function
    is described by the following equations:

    .. math::

        \frac{\partial \mathbf{E}}{\partial t} & = \nabla \times \mathbf{B} -
        \sum_p w_p \sqrt{f_{0,p}(\mathbf{x}_p, \mathbf{v}_p)} \mathbf{v}_p \,,

        \frac{\partial \mathbf{B}}{\partial t} & = - \nabla \times \mathbf{E} \,,

        \frac{\text{d} \mathbf{x}_p}{\text{d} t} & = \mathbf{v}_p \,,

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = \mathbf{E}_0 + \mathbf{v}_p \times \mathbf{B}_0 \,,

        \frac{\text{d} w_p}{\text{d} t} & = \frac{1}{v_{\text{th},p}^2} \,
        \sqrt{f_{0,p}(\mathbf{x}_p, \mathbf{v}_p)} \, \mathbf{E} \cdot \mathbf{v}_p
    
    which form a Hamiltonian system with the energies:

    .. math::

        H_0(t) & = \sum_p \left( \frac{\mathbf{v}_p^2}{2} + \phi_0(\mathbf{x}_p) \right) \,,

        H_h(t) & = \sum_p \frac{v_{\text{th},p}^2 w_p^2}{2}
        + \frac{1}{2} \int_\Omega |\mathbf{E}|^2 \, \text{d}^3 \mathbf{x}
        + \frac{1}{2} \int_\Omega |\mathbf{B}|^2 \, \text{d}^3 \mathbf{x} \,.

    All natural constants are set equal to 1 and all particles are normalized
    to have unit mass and unit charge.

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.propagators.propagators import StepStaticEfield, StepStaticBfield, StepEfieldWeights, StepMaxwell
        from struphy.psydac_api.fields import Field
        from struphy.fields_background.mhd_equil import analytical

        super().__init__(params, comm, e_field='Hcurl', b_field='Hdiv', electrons='Particles6D')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(self.derham, self.domain)
        self._mass_ops.assemble_M1()
        self._mass_ops.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        self._electrons = self.kinetic_species[0]

        # ====================================================================================
        # Instantiate background electric field and potential
        self._background_fields = []
        self._background_fields += [Field('e_background', 'Hcurl', self.derham)]
        self._background_fields += [Field('phi_background', 'H1', self.derham)]

        self._background_fields[1].set_initial_conditions(self.domain, [True], params['fields']['init'])

        self._e_background = self._background_fields[0].vector
        self._phi_background = self._background_fields[1].vector

        self._e_background = self.derham.grad.dot(self._phi_background)

        # Initialize background magnetic field from MHD equilibrium
        self._background_fields += [Field('b_background', 'Hdiv', self.derham)]
        self._b_background = self._background_fields[2].vector
        
        # Create MHD equilibrium
        equil_params = params['fields']['mhd_equilibrium']
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(equil_params[equil_params['type']], self.domain)
        
        # self._b_background[0] = 
        self._b_background = self.derham.P2([mhd_equil.b_x, mhd_equil.b_y, mhd_equil.b_z]).coeffs
        # ====================================================================================

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepStaticEfield(self.domain, self.derham, self._electrons, self._e_background)]
        self._propagators += [StepStaticBfield(self.domain, self.derham, self._electrons, self._b_background)]
        self._propagators += [StepEfieldWeights(self.domain, self.derham, self._e, self._electrons, self._mass_ops,
                                                params['kinetic']['electrons']['background'], params['solvers']['solver_1'])]
        self._propagators += [StepMaxwell(self._e, self._b, self.derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time']       = np.empty(1, dtype=float)
        self._scalar_quantities['en_E']       = np.empty(1, dtype=float)
        self._scalar_quantities['en_B']       = np.empty(1, dtype=float)
        self._scalar_quantities['en_weights'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_all']     = np.empty(1, dtype=float)
        self._scalar_quantities['en_el_pot']  = np.empty(1, dtype=float)
        self._scalar_quantities['en_kin']     = np.empty(1, dtype=float)
        self._scalar_quantities['en_sing']    = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators    
    
    def _compute_electric_potential(self):
        ''' Compute the sum of the electric potential at all particle positions '''
        
        res = eval_field_at_particles(self._phi_background, self._derham, 'H1', self._electrons)

        return res

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = self._e.dot(self._mass_ops.M1.dot(self._e)) / 2.
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b)) / 2.
        self._scalar_quantities['en_weights'][0] = np.sum(self._electrons.markers[:, 8])**2
        self._scalar_quantities['en_all'][0] = self._scalar_quantities['en_weights'][0] + \
                                               self._scalar_quantities['en_E'][0] + \
                                               self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_el_pot'][0] = self._compute_electric_potential()
        self._scalar_quantities['en_kin'][0] = np.sum(np.sum(self._electrons.markers[:, 3:6], axis=1)**2)
        self._scalar_quantities['en_sing'][0] = self._scalar_quantities['en_el_pot'][0] + \
                                                self._scalar_quantities['en_kin'][0]


class PC_LinMHD_6d_full(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with full pressure coupling scheme. 
    
    Normalization: 

    .. math::

        \frac{\hat B^2}{\hat \rho \mu_0} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,.

    Implemented equations:

    .. math::

        &\\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p + \nabla\cdot \tilde{\mathbb{P}}_h 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,
        
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,

        &\frac{\partial f_h}{\partial t} + (\mathbf{v} + \tilde{\mathbf{U}})\cdot\frac{\partial f_h}{\partial \mathbf{x}}
        + (\frac{q_h}{m_h}\mathbf{v}\\times(\mathbf{B} + \tilde{\mathbf{B}}_0) - \nabla \tilde{\mathbf{U}}\cdot \mathbf{v})\cdot\frac{\partial f_h}{\partial \mathbf{v}}
        = 0\,,

        &\tilde{\mathbb{P}_h} = \int \mathbf{v}\mathbf{v}^\top f_h d\mathbf{v} \,.
        domain: struphy.geometry.domain_3d.Domain
        
    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):
        
        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.psydac_api.mhd_ops_pure_psydac import MHDOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators
        
        self._u_space = params['fields']['mhd_u_space']
        super().__init__(params, comm, n3='L2', u1=self._u_space, p3='L2', b2='Hdiv', energetic_ions='Particles6D')

        # extract necessary parameters
        equil_params = params['fields']['mhd_equilibrium']
        alfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']
        pressurecoupling_solver = params['solvers']['solver_2']
        nuh =  params['kinetic']['energetic_ions']['attributes']['nuh']
        self._nuh = nuh
        self._comm = self.derham.comm

        # Load MHD equilibrium and project fields
        mhd_equil_class = getattr(analytical, equil_params['type'])
        mhd_equil = mhd_equil_class(equil_params[equil_params['type']], self.domain)
        
        self._b_eq = self.derham.P2([mhd_equil.b2_1, mhd_equil.b2_2, mhd_equil.b2_3]).coeffs
        
        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(self.derham, self.domain, eq_mhd=mhd_equil)
        
        self._mass_ops.assemble_M1()
        self._mass_ops.assemble_M2()
        self._mass_ops.assemble_M3()
        
        if self._u_space == 'Hcurl':
            self._mass_ops.assemble_M1n()
            self._mass_ops.assemble_M1J()
        elif self._u_space == 'Hdiv':
            self._mass_ops.assemble_M2n()
            self._mass_ops.assemble_M2J()
        else:
            self._mass_ops.assemble_Mvn()
            self._mass_ops.assemble_MvJ()
        
        # Assemble necessary linear MHD projection operators
        self._mhd_ops = MHDOperators(self.derham, self.domain, mhd_equil)
        
        if self._u_space == 'Hcurl':
            self._mhd_ops.assemble_X1()
            self._mhd_ops.assemble_K1()
            self._mhd_ops.assemble_Q1()
            self._mhd_ops.assemble_T1()
            self._mhd_ops.assemble_S1()
            self._mhd_ops.assemble_U1()
        elif self._u_space == 'Hdiv':
            self._mhd_ops.assemble_X2()
            self._mhd_ops.assemble_K2()
            self._mhd_ops.assemble_Q2()
            self._mhd_ops.assemble_T2()
            self._mhd_ops.assemble_S2()
        else:
            self._mhd_ops.assemble_X0()
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
        if self._u_space == 'Hcurl':
            Alfven = getattr(propagators, 'StepShearAlfvénHcurl')
            Magnetosonic = getattr(propagators, 'StepMagnetosonicHcurl')
            Pressurecoupling = getattr(propagators, 'StepPressurecouplingHcurl')
            PushEta = getattr(propagators, 'StepPushEtaPC')
            PushVel = getattr(propagators, 'StepPushVxB')
        elif self._u_space == 'Hdiv':
            Alfven = getattr(propagators, 'StepShearAlfvénHdiv')
            Magnetosonic = getattr(propagators, 'StepMagnetosonicHdiv')
            Pressurecoupling = getattr(propagators, 'StepPressurecouplingHdiv')
            PushEta = getattr(propagators, 'StepPushEtaPC')
            PushVel = getattr(propagators, 'StepPushVxB')
        elif self._u_space == 'H1vec':
            Alfven = getattr(propagators, 'StepShearAlfvénH1vec')
            Magnetosonic = getattr(propagators, 'StepMagnetosonicH1vec') 
            Pressurecoupling = getattr(propagators, 'StepPressurecouplingH1vec')
            PushEta = getattr(propagators, 'StepPushEtaPC')
            PushVel = getattr(propagators, 'StepPushVxB')
            
        self._propagators = []
        self._propagators += [Alfven(self._u, self._b, self.derham, self._mass_ops, self._mhd_ops, alfven_solver)]
        self._propagators += [Magnetosonic(self._n, self._u, self._p, self._b, self.derham, self._mass_ops, self._mhd_ops, magnetosonic_solver)]
        for particles in self._kinetic_species:
            self._propagators += [PushEta(self._u, particles, self.derham, self.domain, self._u_space)]
            self._propagators += [Pressurecoupling(self._u, particles, self.derham, self.domain, self._mass_ops, self._mhd_ops, pressurecoupling_solver)]
            self._propagators += [PushVel(particles, self.derham, self._b, self._b_eq)]
            
        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        
        if self._u_space == 'Hcurl':
            self._en_U_loc = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)         
        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b))/2

        for particles in self._kinetic_species:
            self._en_f_loc = particles.markers[~particles.holes,8].dot(particles.markers[~particles.holes,3]**2
                                                                      +particles.markers[~particles.holes,4]**2
                                                                      +particles.markers[~particles.holes,5]**2
                                                                      )/(2. * particles.n_mks) 

            self._comm.Reduce(self._en_f_loc, self._scalar_quantities['en_f'], op=MPI.SUM, root=0)

        self._scalar_quantities['en_tot'][0]  = self._scalar_quantities['en_U'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
        
        
class Vlasov(StruphyModel):
    r'''TODO
    '''
    
    def __init__(self, params, comm):
        
        from struphy.propagators.propagators import StepPushVxB, StepPushEtaRk4
        from struphy.fields_background.mhd_equil import analytical
        
        super().__init__(params, comm, ions='Particles6D')
        
        print(f'rank : {self.derham.comm.Get_rank()}, Np : {self.kinetic_species[0].n_mks}, markers shape : {self.kinetic_species[0].markers.shape}')
        
        # Load and project magnetic field
        equil_params = params['fields']['mhd_equilibrium']
        mhd_equil_class = getattr(analytical, equil_params['type'])
        self._mhd_equil = mhd_equil_class(equil_params[equil_params['type']], self.domain)
        
        if self.derham.comm.Get_rank() == 0: print('Start of background magnetic field projection ...')
        self._b = self.derham.P2([self._mhd_equil.b2_1, 
                                  self._mhd_equil.b2_2, 
                                  self._mhd_equil.b2_3]).coeffs
        if self.derham.comm.Get_rank() == 0: print('Background magnetic field projection done ...')
        
        # Pointer to ions
        self._ions = self.kinetic_species[0]
        
        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepPushVxB(self._ions, self.derham, self._b)]
        self._propagators += [StepPushEtaRk4(self._ions, self.derham)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        
    @property
    def propagators(self):
        return self._propagators
    
    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        