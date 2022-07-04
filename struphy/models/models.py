
import numpy as np
from struphy.models.base import StruphyModel


class Maxwell(StruphyModel):
    '''Maxwell's equations in vacuum. 
    
    Normalization:

    .. math::

        c = \\frac{\hat \omega}{\hat k} = \\frac{\hat E}{\hat B}\,,

    where :math:`c` is the vacuum speed of light. Implemented equations:

    .. math::
    
        &\\frac{\partial \mathbf E}{\partial t} - \\nabla\\times\mathbf B = 0\,, 
        
        &\\frac{\partial \mathbf B}{\partial t} + \\nabla\\times\mathbf E = 0\,.

    Parameters
    ----------
        DR: Derham obj
            From struphy/psydac_api/psydac_derham.Derham_build.

        DOMAIN: Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.
    '''

    def __init__(self, DR, DOMAIN, *solver_params):

        from struphy.propagators.propagators import StepMaxwell

        super().__init__(DR, DOMAIN, *solver_params, e_field='Hcurl', b_field='Hdiv')

        # Assemble necessary mass matrices
        self.DR.assemble_M1()
        self.DR.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, DR, self.solver_params[0])]

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
        self._scalar_quantities['en_E'][0] = .5*self._e.dot(self.DR.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = .5*self._b.dot(self.DR.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + \
                                               self._scalar_quantities['en_B'][0]


class LinearMHD(StruphyModel):
    '''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). 
    
    Normalization:

    .. math::

        TODO.

    Implemented equations:

    .. math::

        &\\frac{\partial \\tilde \\rho}{\partial t}+\\nabla\cdot(\\rho_0 \\tilde{\mathbf{U}})=0\,, 

        \\rho_0&\\frac{\partial \\tilde{\mathbf{U}}}{\partial t} + \\nabla \\tilde p
        =(\\nabla\\times \\tilde{\mathbf{B}})\\times\mathbf{B}_0 + \mathbf{J}_0\\times \\tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \\nabla\\times\mathbf{B}_0\,,

        &\\frac{\partial \\tilde p}{\partial t} + \\nabla\cdot(p_0 \\tilde{\mathbf{U}}) 
        + (\gamma-1)p_0\\nabla\cdot \\tilde{\mathbf{U}}=0\,,
        
        &\\frac{\partial \\tilde{\mathbf{B}}}{\partial t} - \\nabla\\times(\\tilde{\mathbf{U}} \\times \mathbf{B}_0)
        = 0\,.

    Parameters
    ----------
        derham : Derham
            From struphy/psydac_api/psydac_derham.

        domain : Domain
            From struphy/geometry/domain_3d.
            
        params_mhd_equil : dict
            Parameters for MHD equilibrium.
            
        formulation : str
            Numerical representation of (U, p)

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters correpsonding to one solver, obtained from paramaters.yml.
    '''

    def __init__(self, derham, domain, params_mhd_equil, formulation, *solver_params):
        
        from struphy.psydac_api.mass_psydac import WeightedMass
        from struphy.psydac_api.mhd_ops_pure_psydac import MHDOperators
        
        # Choose model formulation
        self._formulation = formulation
        
        if   formulation == 'Hdiv':
            super().__init__(derham, domain, *solver_params, n='L2', U='Hdiv', p='L2', B='Hdiv')
        elif formulation == 'H1vec':
            super().__init__(derham, domain, *solver_params, n='L2', U='H1vec', p='L2', B='Hdiv')
        else:
            raise NotImplementedError('Chosen formulation does not exist!')
        
        # Load MHD equilibrium
        if   params_mhd_equil['type'] == 'homogeneous':
            from struphy.fields_equil.mhd_equil.analytical import EquilibriumMHDSlab
            self._eq_mhd = EquilibriumMHDSlab(params_mhd_equil['homogeneous'], domain)
            
        elif params_mhd_equil['type'] == 'sheared_slab':
            from struphy.fields_equil.mhd_equil.analytical import EquilibriumMHDShearedSlab
            self._eq_mhd = EquilibriumMHDShearedSlab(params_mhd_equil['sheared_slab'], domain)
            
        else:
            raise NotImplementedError('Chosen MHD equilibrium does not exist!')
        
        # Assemble necessary mass matrices
        self._mass_ops = WeightedMass(derham, derham.F.get_callable_mapping(), eq_mhd=self._eq_mhd)
        
        self._mass_ops.assemble_M2()
        self._mass_ops.assemble_M3()
        
        if self._formulation == 'Hdiv':
            self._mass_ops.assemble_M2n()
            self._mass_ops.assemble_M2J()
        else:
            self._mass_ops.assemble_Mvn()
            self._mass_ops.assemble_MvJ()
        
        # Assemble necessary linear MHD projection operators
        self._mhd_ops = MHDOperators(derham, derham.F.get_callable_mapping(), self._eq_mhd)
        
        if self._formulation == 'Hdiv':
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
        
        if self._formulation == 'Hdiv':
            
            from struphy.propagators.propagators import StepShearAlfven2
            from struphy.propagators.propagators import StepMagnetosonic2
            
            self._propagators += [StepShearAlfven2(self._u, self._b, derham, self._mass_ops, self._mhd_ops, self.solver_params[0])]
            self._propagators += [StepMagnetosonic2(self._n, self._u, self._p, self._b, derham, self._mass_ops, self._mhd_ops, self.solver_params[1])]
        
        elif self._formulation == 'H1vec':
            
            from struphy.propagators.propagators import StepShearAlfven3
            from struphy.propagators.propagators import StepMagnetosonic3
            
            self._propagators += [StepShearAlfven3(self._u, self._b, derham, self._mass_ops, self._mhd_ops, self.solver_params[0])]
            self._propagators += [StepMagnetosonic3(self._n, self._u, self._p, self._b, derham, self._mass_ops, self._mhd_ops, self.solver_params[1])]
        
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
        
        if self._formulation == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_tot'][0]  = self._scalar_quantities['en_U'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0] 
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]



class LinearVlasovMaxwell( StruphyModel ):
    """
    Linearized Vlasov Maxwell model, has electric and magnetic fields, and electrons as particles

    Parameters
    ----------
        DR : Derham obj
            From struphy/feec/psydac_derham.Derham_build.

        DOMAIN : Domain obj
            From struphy/geometry/domain_3d.Domain.

        solver_params : list
            Each entry corresponds to one linear solver used in the model. 
            An entry is a dict with the solver parameters corresponding to one solver, obtained from parameters.yml.
    """

    def __init__(self, DR, DOMAIN, *solver_params, electron_markers, f_0_params):

        from struphy.kinetic_background.kinetic_equil_6d import MaxwellHomogenSlab
        from struphy.propagators.propagators import StepMaxwell

        super().__init__(DR, DOMAIN, *solver_params, efield='Hcurl',
                         bfield='Hdiv', electrons=electron_markers)

        # set kinetic equilibrium/background distribution function and set it for the electrons
        if f_0_params['type'] == 'Maxwell_homogen_slab':
            EQ_KINETIC = MaxwellHomogenSlab(f_0_params, DOMAIN)
            self.EQ_Kinetic = EQ_KINETIC
            self._kinetic_species[0].set_kinetic_equil(self.EQ_Kinetic)
        else:
            raise ValueError('Equilibrium not implemented!')

        # Assemble necessary mass matrices
        self.DR.assemble_M1()
        self.DR.assemble_M2()

        # Pointers to Stencil-/Blockvectors
        self._e = self.fields[0].vector
        self._b = self.fields[1].vector

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [StepMaxwell(self._e,
                                          self._b, DR, self.solver_params[0])]
        # TODO: self._propagators += [StepLinearVlasovMaxwell]

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

    @property
    def KIN(self):
        """Dictionary with all the kinetic objects in them, keys are the names"""
        return self._KIN
