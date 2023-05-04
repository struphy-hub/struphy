import numpy as np
from struphy.models.base import StruphyModel

#############################
# Fluid models
#############################
class LinearMHD(StruphyModel):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,.

    Implemented equations:

    .. math::

        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 

        n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    
    @classmethod
    def bulk_species(cls):
        return 'mhd'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'

    def __init__(self, params, comm):
        
        # choose MHD veclocity space
        u_space = params['fluid']['mhd']['mhd_u_space']
         
        assert u_space in ['Hdiv', 'H1vec'], f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'
        
        u_name = 'u2' if u_space == 'Hdiv' else 'uv'
            
        self._u_space = u_space

        # initialize base class
        super().__init__(params, comm, 
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: u_space, 'p3': 'L2'})
            
        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # extract necessary parameters
        alfven_solver = params['solvers']['solver_1']
        sonic_solver = params['solvers']['solver_2']

        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops
        Propagator.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(
            self._u,
            self._b,
            u_space=self._u_space,
            **alfven_solver)]
        self._propagators += [propagators_fields.Magnetosonic(
            self._n,
            self._u,
            self._p,
            u_space=self._u_space,
            b=self._b,
            **sonic_solver)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
        # temporary vectors for scalar quantities
        if self._u_space == 'Hdiv':
            self._tmp_u1 = self.derham.Vh['2'].zeros()
        else:
            self._tmp_u1 = self.derham.Vh['v'].zeros()
            
        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities      

    def update_scalar_quantities(self):

        # perturbed fields
        if self._u_space == 'Hdiv':
            self._mass_ops.M2n.dot(self._u, out=self._tmp_u1)
        else:
            self._mass_ops.Mvn.dot(self._u, out=self._tmp_u1)
            
        self._mass_ops.M2.dot(self._b, out=self._tmp_b1)
        
        en_U = self._u.dot(self._tmp_u1)/2
        en_B = self._b.dot(self._tmp_b1)/2
        en_p = self._p.dot(self._ones)/(5/3 - 1)
        
        self._scalar_quantities['en_U'][0] = en_U
        self._scalar_quantities['en_B'][0] = en_B
        self._scalar_quantities['en_p'][0] = en_p
        
        self._scalar_quantities['en_tot'][0]  = en_U
        self._scalar_quantities['en_tot'][0] += en_B
        self._scalar_quantities['en_tot'][0] += en_p

        # background fields
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        
        en_B0 = self._b_eq.dot(self._tmp_b1)/2
        en_p0 = self._p_eq.dot(self._ones)/(5/3 - 1)
        
        self._scalar_quantities['en_B_eq'][0] = en_B0
        self._scalar_quantities['en_p_eq'][0] = en_p0
        
        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self._b
        
        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        
        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self._scalar_quantities['en_B_tot'][0] = en_Btot
        
        
class LinearExtendedMHD(StruphyModel):
    r'''Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,.

    Implemented equations:

    .. math::

        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 

        n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    
    @classmethod
    def bulk_species(cls):
        return 'mhd'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm, 
                         b1='Hcurl',
                         mhd={'n3': 'L2', 'u2': 'Hdiv', 'pi3': 'L2', 'pe3': 'L2'})
            
        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators

        # pointers to em-field variables
        self._b = self.em_fields['b1']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd']['u2']['obj'].vector
        self._p_i = self.fluid['mhd']['pi3']['obj'].vector
        self._p_e = self.fluid['mhd']['pe3']['obj'].vector

        # extract necessary parameters
        alfven_solver = params['solvers']['solver_1']
        sonic_solver = params['solvers']['solver_2']

        # project background magnetic field (1-form) and pressure (3-form)
        self._b_eq = self.derham.P['1']([self.mhd_equil.b1_1,
                                         self.mhd_equil.b1_2,
                                         self.mhd_equil.b1_3])
        self._p_i_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._p_e_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_i_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops
        Propagator.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvénB1(
            self._u,
            self._b,
            **alfven_solver)]
        #self._propagators += [propagators_fields.Magnetosonic(
            #self._n,
            #self._u,
            #self._p,
            #u_space=self._u_space,
            #b=self._b,
            #**sonic_solver)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_i'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_e'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_i_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_e_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()
            
        self._tmp_b1 = self.derham.Vh['1'].zeros()
        self._tmp_b2 = self.derham.Vh['1'].zeros()

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities      

    def update_scalar_quantities(self):

        # perturbed fields
        self._mass_ops.M2n.dot(self._u, out=self._tmp_u1)
        
            
        self._mass_ops.M1.dot(self._b, out=self._tmp_b1)
        
        en_U = self._u.dot(self._tmp_u1)/2
        en_B = self._b.dot(self._tmp_b1)/2
        en_p_i = self._p_i.dot(self._ones)/(5/3 - 1)
        en_p_e = self._p_e.dot(self._ones)/(5/3 - 1)
        
        self._scalar_quantities['en_U'][0] = en_U
        self._scalar_quantities['en_B'][0] = en_B
        self._scalar_quantities['en_p_i'][0] = en_p_i
        self._scalar_quantities['en_p_e'][0] = en_p_e
        
        self._scalar_quantities['en_tot'][0]  = en_U
        self._scalar_quantities['en_tot'][0] += en_B
        self._scalar_quantities['en_tot'][0] += en_p_i
        self._scalar_quantities['en_tot'][0] += en_p_e

        # background fields
        self._mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        
        en_B0 = self._b_eq.dot(self._tmp_b1)/2
        en_p0_i = self._p_i_eq.dot(self._ones)/(5/3 - 1)
        en_p0_e = self._p_e_eq.dot(self._ones)/(5/3 - 1)
        
        self._scalar_quantities['en_B_eq'][0] = en_B0
        self._scalar_quantities['en_p_i_eq'][0] = en_p0_i
        self._scalar_quantities['en_p_e_eq'][0] = en_p0_e
        
        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self._b
        
        self._mass_ops.M1.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        
        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self._scalar_quantities['en_B_tot'][0] = en_Btot
        

# class ColdPlasma(StruphyModel):
#     r'''Cold plasma model

#     Normalization:

#     .. math::

#         &c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,,

#         &\hat \omega = \Omega_{ce}\,,

#         &\alpha = \frac{\Omega_{pe}}{\Omega_{ce}}\,,

#         &\hat j_c = \varepsilon_0 \Omega_{pe} \hat E\,,

#     where :math:`c` is the vacuum speed of light, :math:`\Omega_{ce}` the electron cyclotron frequency,
#     :math:`\Omega_{pe}` the plasma frequency and :math:`\varepsilon_0` the vacuum dielectric constant.
#     Implemented equations:

#     .. math::

#         &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,

#         &-\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
#         \alpha \mathbf j_c \,,

#         &\frac{\partial \mathbf j_c}{\partial t} = \alpha \mathbf E + \mathbf j_c \times \mathbf B\,.


#     Parameters
#     ----------
#         params : dict
#             Simulation parameters, see from :ref:`params_yml`.
#     '''

#     def __init__(self, params, comm):

#         from struphy.psydac_api.mass import WeightedMassOperators
#         from struphy.propagators import propagators_fields

#         super().__init__(params, comm, e1='Hcurl', b2='Hdiv',
#                          electron={'j1': 'Hcurl'}, hot_electrons='Particles6D')

#         # pointers to em-fields variables
#         self._e = self.em_fields['e1']['obj'].vector
#         self._b = self.em_fields['b2']['obj'].vector

#         # pointers to  fluid variables
#         self._j = self.fluid['electrons']['j1']['obj'].vector

#         # extract necessary parameters
#         maxwell_solver = params['solvers']['solver_1']
#         cold_solver = params['solvers']['solver_2']

#         # Define callable for weighted mass matrices
#         proton_mass = 1.6726219237e-27
#         electron_mass = self.fluid['electrons']['plasma_params']['M'] * proton_mass
#         vacuum_permittivity = 8.854187813e-12
#         prefactor = (electron_mass / vacuum_permittivity)**0.5

#         def call_alpha(e1, e2, e3):
#             return prefactor * self.mhd_equil.n0(e1, e2, e3, sqeez_out=False)**0.5 / self.mhd_equil.absB0(e1, e2, e3, sqeez_out=False)

#         def call_M1alpha(e1, e2, e3, m, n):
#             return self.domain.Ginv(e1, e2, e3)[:, :, :, m, n] * self.domain.sqrt_g(e1, e2, e3)*call_alpha(e1, e2, e3)

#         # Assemble necessary mass matrices
#         self._mass_ops = WeightedMassOperators(
#             self.derham, self.domain, alpha=call_M1alpha)

#         # Initialize propagators/integrators used in splitting substeps
#         self._propagators = []
#         self._propagators += [propagators_fields.Maxwell(
#             self._e, self._b, self.derham, self._mass_ops, maxwell_solver)]
#         self._propagators += [propagators_fields.OhmCold(
#             self._j, self._e, self._mass_ops, cold_solver)]

#         # Scalar variables to be saved during simulation
#         self._scalar_quantities['time'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

#     @property
#     def propagators(self):
#         return self._propagators

#     def update_scalar_quantities(self, time):
#         self._scalar_quantities['time'][0] = time
#         self._scalar_quantities['en_E'][0] = .5 * \
#             self._e.dot(self._mass_ops.M1.dot(self._e))
#         self._scalar_quantities['en_B'][0] = .5 * \
#             self._b.dot(self._mass_ops.M2.dot(self._b))
#         self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0]
#         self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


#############################
# Fluid-kinetic hybrid models
#############################
class LinearMHDVlasovCC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **current coupling scheme**.

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat n_\textnormal{h} = \hat n\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD}\,\, &\left\{\,\,
        \begin{aligned}
        &\frac{\partial \tilde{n}}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}}\kappa\left(n_\textnormal{h}\tilde{\mathbf{U}}-n_\textnormal{h}\mathbf{u}_\textnormal{h}\right)\times(\mathbf{B}_0+\tilde{\mathbf{B}}) \color{black}\,,
        \\
        &\frac{\partial \tilde p}{\partial t} + (\gamma-1)\nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)\,,\qquad \nabla\cdot\tilde{\mathbf{B}}=0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\textnormal{h} + \kappa\left[\color{blue} (\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \color{black} + \mathbf{v}\times(\mathbf{B}_0+\tilde{\mathbf{B}})\right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\
        &\quad\,\,n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3v\,,\qquad n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3v\,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0` and
    
    .. math::
    
        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    """

    @classmethod
    def bulk_species(cls):
        return 'mhd'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'
    
    def __init__(self, params, comm):

        # choose MHD veclocity space
        u_space = params['fluid']['mhd']['mhd_u_space']
         
        assert u_space in ['Hdiv', 'H1vec'], f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'
        
        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'
            
        self._u_space = u_space

        # initialize base class
        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles6D')
            
        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from mpi4py.MPI import SUM, IN_PLACE

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to energetic ions
        self._e_ions = self.kinetic['energetic_ions']['obj']
        e_ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']
        solver_params_4 = params['solvers']['solver_4']

        # compute coupling parameter kappa
        units_basic, units_der, units_dimless = self.model_units(params, verbose=False)
        
        ee = 1.602176634e-19 # elementary charge (C)
        mH = 1.67262192369e-27 # proton mass (kg)
        
        Ab = params['fluid']['mhd']['phys_params']['A']
        Zb = params['fluid']['mhd']['phys_params']['Z']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        Zh = params['kinetic']['energetic_ions']['phys_params']['Z']
        
        omega_ch = (Zh*ee*units_basic['B'])/(Ah*mH)
        kappa = omega_ch*units_basic['t']
        
        if abs(kappa - 1) < 1e-6:
            kappa = 1.
        
        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa
        
        # background distribution function used as control variate
        if params['kinetic']['energetic_ions']['markers']['type'] == 'control_variate':
            assert 'background' in params['kinetic']['energetic_ions'], 'no background distribution function for control variate specified'
            control = True
            f0_name = params['kinetic']['energetic_ions']['background']['type']
            f0 = getattr(kin_ana, f0_name)(
                **params['kinetic']['energetic_ions']['background'][f0_name])
        else:
            control = False
            f0 = None

        # project background magnetic field (2-form) and background pressure (3-form)
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops
        Propagator.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        # updates u
        self._propagators += [propagators_fields.CurrentCoupling6DDensity(
            self._u,
            particles=self._e_ions,
            u_space=self._u_space,
            b_eq=self._b_eq,
            b_tilde=self._b,
            f0=f0,
            **solver_params_1,
            **self._coupling_params)]

        # updates u and b
        self._propagators += [propagators_fields.ShearAlfvén(
            self._u,
            self._b,
            u_space=self._u_space,
            **solver_params_2)]

        # updates u and v (and weights for control variate)
        self._propagators += [propagators_coupling.CurrentCoupling6DCurrent(
            self._e_ions,
            self._u,
            u_space=self._u_space,
            b_eq=self._b_eq,
            b_tilde=self._b,
            f0=f0,
            **solver_params_3,
            **self._coupling_params)]

        # updates eta (and weights for control variate)
        self._propagators += [propagators_markers.PushEta(
            self._e_ions,
            algo=e_ions_params['push_algos']['eta'],
            bc_type=e_ions_params['markers']['bc_type'],
            f0=f0)]

        # updates v (and weights for control variate)
        self._propagators += [propagators_markers.PushVxB(
            self._e_ions,
            algo=e_ions_params['push_algos']['vxb'],
            scale_fac=self._coupling_params['kappa'],
            b_eq=self._b_eq,
            b_tilde=self._b,
            f0=f0)]

        # updates u and p
        self._propagators += [propagators_fields.Magnetosonic(
            self._n,
            self._u,
            self._p,
            u_space=self._u_space,
            b=self._b,
            **solver_params_4)]

        # Scalar variables to be saved during simulation:
        self._scalar_quantities = {}

        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        # self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        # self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):

        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2

        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        # self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
        #     self._ones)/(5/3 - 1)
        # self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
        #     self._mass_ops.M2.dot(self._b_eq, apply_bc=False))/2

        self._scalar_quantities['en_B_tot'][0] = (
            self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b, apply_bc=False))/2

        self._scalar_quantities['en_f'][0] = self._coupling_params['Ah']/self._coupling_params['Ab']*self._e_ions.markers_wo_holes[:, 6].dot(
            self._e_ions.markers_wo_holes[:, 3]**2 +
            self._e_ions.markers_wo_holes[:, 4]**2 +
            self._e_ions.markers_wo_holes[:, 5]**2)/(2*self._e_ions.n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._scalar_quantities['en_f'], op=self._mpi_sum)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


class LinearMHDVlasovPC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with **pressure coupling scheme**. 

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat{\mathbb{P}}_\textnormal{h} = A_\textnormal{h}m_\textnormal{H}\hat n \hat v_\textnormal{A}^2\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla\cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp} \color{black} 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} \color{blue} + \tilde{\mathbf{U}}_\perp \color{black})\cdot \nabla f_\textnormal{h}
        + \left[\kappa\, \mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{blue}- \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \color{black} \right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\quad\,\,\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_h d\mathbf{v} \,,
        \end{aligned}
        \right.
        \end{align}
        
    where 
    
    .. math::
    
        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    There is also a version of this model without the :math:`\perp` subscript (can be selected in :code:`parameters.yml`).

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return 'mhd'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'
    
    def __init__(self, params, comm):

        # choose MHD veclocity space
        u_space = params['fluid']['mhd']['mhd_u_space']
         
        assert u_space in ['Hdiv', 'H1vec'], f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'
        
        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'
            
        self._u_space = u_space

        # initialize base class
        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles6D')
            
        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.models.utilities import derive_units
        from mpi4py.MPI import SUM, IN_PLACE

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to kinetic variables
        self._e_ions = self.kinetic['energetic_ions']['obj']
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']

        # compute coupling parameter kappa
        units_basic, units_der, units_dimless = self.model_units(params, verbose=False)
        
        ee = 1.602176634e-19 # elementary charge (C)
        mH = 1.67262192369e-27 # proton mass (kg)
        
        Ab = params['fluid']['mhd']['phys_params']['A']
        Zb = params['fluid']['mhd']['phys_params']['Z']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        Zh = params['kinetic']['energetic_ions']['phys_params']['Z']
        
        omega_ch = (Zh*ee*units_basic['B'])/(Ah*mH)
        kappa = omega_ch*units_basic['t']
        
        if abs(kappa - 1) < 1e-6:
            kappa = 1.
        
        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa
        
        # background distribution function used as control variate
        if params['kinetic']['energetic_ions']['markers']['type'] == 'control_variate':
            assert 'background' in params['kinetic']['energetic_ions'], 'no background distribution function for control variate specified'
            control = True
            f0_name = params['kinetic']['energetic_ions']['background']['type']
            f0 = getattr(kin_ana, f0_name)(
                **params['kinetic']['energetic_ions']['background'][f0_name])
        else:
            control = False
            f0 = None

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops
        Propagator.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        # updates u and b
        self._propagators += [propagators_fields.ShearAlfvén(
            self._u, 
            self._b, 
            u_space=self._u_space, 
          **solver_params_1,)]
        
        self._propagators += [propagators_coupling.PressureCoupling6D(
            self._e_ions,
            self._u, 
            u_space=self._u_space, 
            use_perp_model=ions_params['use_perp_model'], 
          **solver_params_2,
          **self._coupling_params)]
        
        # updates eta
        self._propagators += [propagators_markers.PushEtaPC(
            self._e_ions, 
            u_mhd=self._u, 
            u_space=self._u_space, 
            bc_type=ions_params['markers']['bc_type'],
            use_perp_model=ions_params['use_perp_model'])]
        
        # updates v
        self._propagators += [propagators_markers.PushVxB(
            self._e_ions, 
            algo=ions_params['push_algos']['vxb'], 
            scale_fac=self._coupling_params['kappa'], 
            b_eq=self._b_eq,
            b_tilde=self._b,
            f0=f0)]
        
        # updates u and p
        self._propagators += [propagators_fields.Magnetosonic(
            self._n, 
            self._u, 
            self._p,
            u_space=self._u_space,
            b=self._b,
            **solver_params_3)]

        # Scalar variables to be saved during simulation:
        self._scalar_quantities = {}

        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities
    
    def update_scalar_quantities(self):

        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2

        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
            self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
            self._mass_ops.M2.dot(self._b_eq, apply_bc=False))/2

        self._scalar_quantities['en_B_tot'][0] = (
            self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b, apply_bc=False))/2

        self._scalar_quantities['en_f'][0] = self._coupling_params['Ah']/self._coupling_params['Ab']*self._e_ions.markers_wo_holes[:, 6].dot(
            self._e_ions.markers_wo_holes[:, 3]**2 +
            self._e_ions.markers_wo_holes[:, 4]**2 +
            self._e_ions.markers_wo_holes[:, 5]**2)/(2*self._e_ions.n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._scalar_quantities['en_f'], op=self._mpi_sum)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


class LinearMHDDriftkineticCC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Driftkinetic) equations with **current coupling scheme**. 

    :ref:`normalization`: 

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0} \, \,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat n_\textnormal{h} = \hat n\,.

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf J_0 \times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}}\kappa \,(n_\textnormal{h} \tilde{\mathbf{U}} - \mathbf{J}_\textnormal{gc} - \frac{1}{\kappa}\nabla \times \mathbf{M}_\textnormal{gc}) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{black}\,,
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        \quad &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*)\cdot\nabla f_\textnormal{h}
        + \kappa \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel}
        = 0\,,
        \\
        & n_\textnormal{h} = \int f_\textnormal{h} B^*_\parallel \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{J}_\textnormal{gc} = q \int f_\textnormal{h} (v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{M}_\textnormal{gc} = - \int f_\textnormal{h} \mu \mathbf{b}_0 B^*_\parallel \,\textnormal dv_\parallel \textnormal d\mu \,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf J_0 = \nabla \times \mathbf B_0` and

    .. math::

        \begin{align}
        \mathbf{B}^* &= \mathbf{B} + \frac{1}{\kappa} v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,,
        \\
        \mathbf{E}^* &= \color{blue} - \tilde{\mathbf{U}} \times \mathbf{B} \color{black} - \frac{1}{\kappa} \mu \nabla B_\parallel \,.
        \end{align}
        
    Moreover,
    
    .. math::
    
        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.    
        
    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return 'mhd'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'
    
    def __init__(self, params, comm):

        # choose MHD veclocity space
        u_space = params['fluid']['mhd']['mhd_u_space']
         
        assert u_space in ['Hdiv', 'H1vec'], f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'
        
        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'
            
        self._u_space = u_space

        # initialize base class
        super().__init__(params, comm, 
                         b2='Hdiv', 
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles5D')
            
        from struphy.polar.basic import PolarVector
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from mpi4py.MPI import SUM

        # guiding center asymptotic parameter (rhostar)
        epsilon = 1. # TODO

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to kinetic variables
        self._e_ions = self.kinetic['energetic_ions']['obj']
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']
        solver_params_4 = params['solvers']['solver_4']
        solver_params_5 = params['solvers']['solver_5']

        # compute coupling parameter kappa
        units_basic, units_der, units_dimless = self.model_units(params, verbose=False)
        
        ee = 1.602176634e-19 # elementary charge (C)
        mH = 1.67262192369e-27 # proton mass (kg)
        
        Ab = params['fluid']['mhd']['phys_params']['A']
        Zb = params['fluid']['mhd']['phys_params']['Z']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        Zh = params['kinetic']['energetic_ions']['phys_params']['Z']
        
        omega_ch = (Zh*ee*units_basic['B'])/(Ah*mH)
        kappa = omega_ch*units_basic['t']
        
        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        print('kappa', kappa)
        
        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa

        # background distribution function used as control variate
        if params['kinetic']['energetic_ions']['markers']['type'] == 'control_variate':
            assert 'background' in params['kinetic']['energetic_ions'], 'no background distribution function for control variate specified'
            control = True
            f0_name = params['kinetic']['energetic_ions']['background']['type']
            f0 = getattr(kin_ana, f0_name)(
                **params['kinetic']['energetic_ions']['background'][f0_name])
        else:
            control = False
            f0 = None

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        self._abs_b = self.derham.P['0'](self.mhd_equil.absB0)

        self._unit_b1 = self.derham.P['1']([self.mhd_equil.unit_b1_1,
                                           self.mhd_equil.unit_b1_2,
                                           self.mhd_equil.unit_b1_3])

        self._unit_b2 = self.derham.P['2']([self.mhd_equil.unit_b2_1,
                                           self.mhd_equil.unit_b2_2,
                                           self.mhd_equil.unit_b2_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        self._b_cart = self.derham.P['v']([self.mhd_equil.b_cart_1,
                                                self.mhd_equil.b_cart_2,
                                                self.mhd_equil.b_cart_3])                   

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops
        Propagator.basis_ops = BasisProjectionOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        # updates u and b
        self._propagators += [propagators_fields.ShearAlfvénCurrentCoupling5D(
            self._u,
            self._b,
            particles=self._e_ions,
            b_eq=self._b_eq,
            f0=f0,
            u_space=self._u_space,
            **solver_params_1,
            **self._coupling_params)]
        # # updates u and p
        # self._propagators += [propagators_fields.MagnetosonicCurrentCoupling5D(
        #     self._n,
        #     self._u,
        #     self._p,
        #     b=self._b,
        #     particles=self._e_ions,
        #     unit_b1=self._unit_b1,
        #     f0=f0,
        #     u_space=self._u_space,
        #     **solver_params_2,
        #     **self._coupling_params)] 
        # # update H       
        # self._propagators += [propagators_markers.StepPushDriftKinetic1(
        #     self._e_ions,
        #     kappa=kappa,
        #     b=self._b,
        #     b_eq=self._b_eq, 
        #     unit_b1=self._unit_b1, 
        #     unit_b2=self._unit_b2, 
        #     abs_b=self._abs_b,
        #     integrator=ions_params['push_algos']['integrator'],
        #     method=ions_params['push_algos']['method'],
        #     maxiter=ions_params['push_algos']['maxiter'],
        #     tol=ions_params['push_algos']['tol'])]
        # # update H and v parallel
        # self._propagators += [propagators_markers.StepPushDriftKinetic2(
        #     self._e_ions,
        #     kappa=kappa,
        #     b=self._b,
        #     b_eq=self._b_eq, 
        #     unit_b1=self._unit_b1, 
        #     unit_b2=self._unit_b2, 
        #     abs_b=self._abs_b,
        #     method=ions_params['push_algos']['method'],
        #     integrator=ions_params['push_algos']['integrator'],
        #     maxiter=ions_params['push_algos']['maxiter'],
        #     tol=ions_params['push_algos']['tol'])]        
        # self._propagators += [propagators_coupling.CurrentCoupling5DCurrent1(
        #     self._e_ions,
        #     self._u, 
        #     b=self._b,
        #     b_eq=self._b_eq,
        #     unit_b1=self._unit_b1,
        #     f0=f0, 
        #     u_space=self._u_space, 
        #   **solver_params_3,
        #   **self._coupling_params)]
        # self._propagators += [propagators_coupling.CurrentCoupling5DCurrent2(
        #     self._e_ions,
        #     self._u,
        #     b=self._b, 
        #     b_eq=self._b_eq,
        #     unit_b1=self._unit_b1,
        #     unit_b2=self._unit_b2,
        #     abs_b=self._abs_b,
        #     f0=f0, 
        #     u_space=self._u_space, 
        #   **solver_params_4,
        #   **self._coupling_params)]
        # self._propagators += [propagators_fields.CurrentCoupling6DDensity(
        #     self._u,
        #     particles=self._e_ions,
        #     u_space=self._u_space,
        #     b_eq=self._b_eq,
        #     b_tilde=self._b,
        #     f0=f0,
        #     **solver_params_5,
        #     **self._coupling_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_fv_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fv'] = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fB'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # things needed in update_scalar_quantities
        self._mpi_sum = SUM
        self._prop = Propagator

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):
        
        if self._u_space == 'Hcurl':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M1n.dot(self._u))/2

        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.M2n.dot(self._u))/2

        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(self._mass_ops.Mvn.dot(self._u))/2


        self._scalar_quantities['en_p'][0] = self._p.toarray().sum()/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(self._mass_ops.M2.dot(self._b))/2

        self._en_fv_loc = self._e_ions.markers[~self._e_ions.holes, 5].dot(self._e_ions.markers[~self._e_ions.holes, 3]**2) / (2*self._e_ions.n_mks)
        self.derham.comm.Reduce(self._en_fv_loc, self._scalar_quantities['en_fv'], op=self._mpi_sum, root=0)

        # calculate particle magnetic energy
        self._e_ions.save_magnetic_energy(self._derham, self._prop.basis_ops.PB.dot(self._b + self._b_eq))

        self._en_fB_loc = self._e_ions.markers[~self._e_ions.holes, 5].dot(self._e_ions.markers[~self._e_ions.holes, 8])/self._e_ions.n_mks
        self.derham.comm.Reduce(self._en_fB_loc, self._scalar_quantities['en_fB'], op=self._mpi_sum, root=0)


        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fv'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fB'][0]


# class ColdPlasmaVlasov(StruphyModel):
#     r'''Cold plasma model

#     Normalization:

#     .. math::

#         &c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,,

#         &\hat \omega = \Omega_{ce}\,,

#         &\alpha = \frac{\Omega_{pe}}{\Omega_{ce}}\,,

#         &\hat j_c = \varepsilon_0 \Omega_{pe} \hat E\,,

#         &\hat v = \frac{\Omega_{ce}}{\hat k} = c\,,

#     where :math:`c` is the vacuum speed of light, :math:`\Omega_{ce}` the electron cyclotron frequency,
#     :math:`\Omega_{pe}` the plasma frequency and :math:`\varepsilon_0` the vacuum dielectric constant.
#     Implemented equations:

#     .. math::

#         &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,

#         &-\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
#         \alpha\left(\mathbf j_c + \frac{\hat n_h}{n_0 \left(x\right)} \alpha \mathbf j_h\right)\,,

#         &\frac{\partial \mathbf j_c}{\partial t} = \alpha \mathbf E + \mathbf j_c \times \mathbf B\,,

#         &\frac{\partial f_h}{\partial t} + v \cdot \nabla f_h
#         + \left(\mathbf E + \mathbf v \times B\right) \cdot \nabla_v f_h = 0\,.


#     Parameters
#     ----------
#         params : dict
#             Simulation parameters, see from :ref:`params_yml`.
#     '''

#     def __init__(self, params, comm):

#         from struphy.propagators import propagators_fields
#         from struphy.psydac_api.mass import WeightedMassOperators
#         from struphy.propagators import propagators_fields

#         super().__init__(params, comm, e1='Hcurl', b2='Hdiv',
#                          electron={'j1': 'Hcurl'}, hot_electrons='Particles6D')

#         # pointers to em-fields variables
#         self._e = self.em_fields['e1']['obj'].vector
#         self._b = self.em_fields['b2']['obj'].vector

#         # pointers to  fluid variables
#         self._j = self.fluid['electrons']['j1']['obj'].vector

#         # extract necessary parameters
#         maxwell_solver = params['solvers']['solver_1']
#         cold_solver = params['solvers']['solver_2']

#         # Define callable for weighted mass matrices
#         proton_mass = 1.6726219237e-27
#         electron_mass = self.fluid['electrons']['plasma_params']['M'] * proton_mass
#         vacuum_permittivity = 8.854187813e-12
#         prefactor = (electron_mass / vacuum_permittivity)**0.5

#         def call_alpha(e1, e2, e3):
#             return prefactor * self.mhd_equil.n0(e1, e2, e3, sqeez_out=False)**0.5 / self.mhd_equil.absB0(e1, e2, e3, sqeez_out=False)

#         def call_M1alpha(e1, e2, e3, m, n):
#             return self.domain.Ginv(e1, e2, e3)[:, :, :, m, n] * self.domain.sqrt_g(e1, e2, e3)*call_alpha(e1, e2, e3)

#         # Assemble necessary mass matrices
#         self._mass_ops = WeightedMassOperators(
#             self.derham, self.domain, alpha=call_M1alpha)

#         # Initialize propagators/integrators used in splitting substeps
#         self._propagators = []
#         self._propagators += [propagators_fields.Maxwell(
#             self._e, self._b, self.derham, self._mass_ops, maxwell_solver)]
#         self._propagators += [propagators_fields.OhmCold(
#             self._j, self._e, self._mass_ops, cold_solver)]

#         # Scalar variables to be saved during simulation
#         self._scalar_quantities['time'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

#     @property
#     def propagators(self):
#         return self._propagators

#     def update_scalar_quantities(self, time):
#         self._scalar_quantities['time'][0] = time
#         self._scalar_quantities['en_E'][0] = .5 * \
#             self._e.dot(self._mass_ops.M1.dot(self._e))
#         self._scalar_quantities['en_B'][0] = .5 * \
#             self._b.dot(self._mass_ops.M2.dot(self._b))
#         self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0]
#         self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


#############################
# Kinetic models
#############################
class LinearVlasovMaxwell(StruphyModel):
    r'''Linearized Vlasov Maxwell equations with Maxwellian background.

    :ref:`normalization`:

    .. math::

        \begin{align}
            c & = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B} = \hat v = \hat u \,, \qquad  \hat h = \frac{\hat n}{\hat v^3} \,.
        \end{align}

    Implemented equations:

    .. math::

        \begin{align}
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \kappa\left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \sqrt{f_0} \, \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \alpha^2 \kappa \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where
    
    .. math::
    
        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \kappa = 2\pi \frac{\Omega_\textnormal{c}}{\hat \omega} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.
    
    Moreover, :math:`f_0` is a Maxwellian background distribution function with constant velocity shift :math:`\mathbf{u}`
    and thermal velocity matrix :math:`\mathbb{1}_{\text{th}} = \text{diag} \left( \frac{1}{v_{\text{th},1}^2}, \frac{1}{v_{\text{th},2}^2}, \frac{1}{v_{\text{th},3}^2} \right)`
    and :math:`h = \frac{\delta f}{\sqrt{f_0}}`.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return 'electrons'

    @classmethod
    def timescale(cls):
        return 'light'

    def __init__(self, params, comm):

        super().__init__(params, comm, 
                         e_field='Hcurl', b_field='Hdiv',
                         electrons='Particles6D')

        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # pointers to em-field variables
        self._e = self.em_fields['e_field']['obj'].vector
        self._b = self.em_fields['b_field']['obj'].vector

        # pointer to electrons
        self._electrons = self.kinetic['electrons']['obj']
        electron_params = params['kinetic']['electrons']

        # kinetic background
        assert electron_params['background']['type'] == 'Maxwellian6DUniform', \
            "The background distribution function must be a uniform Maxwellian!"

        self._maxwellian_params = electron_params['background']['Maxwellian6DUniform']
        self._f0 = getattr(kin_ana, 'Maxwellian6DUniform')(
            **self._maxwellian_params)

        # Get coupling strength
        self.alpha = 1. # TODO

        # Get Poisson solver params
        self._poisson_params = params['solvers']['solver_poisson']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # Create pointers to background electric potential and field
        self._phi_background = self.derham.P['0'](self.electric_equil.phi0)
        self._e_background = self.derham.grad.dot(self._phi_background)
        # ====================================================================================

        # set propagators base class attributes (available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        # Only add StepStaticEfield if efield is non-zero, otherwise it is more expensive
        if np.all(self._e_background[0]._data < 1e-14) and np.all(self._e_background[1]._data < 1e-14) and np.all(self._e_background[2]._data < 1e-14):
            self._propagators += [propagators_markers.PushEta(
                self._electrons,
                algo=electron_params['push_algos']['eta'],
                bc_type=electron_params['markers']['bc_type'],
                f0=None)]  # no conventional weights update here, thus f0=None
        else:
            self._propagators += [propagators_markers.StepStaticEfield(
                self._electrons,
                e_eq=self._e_background)]

        self._propagators += [propagators_markers.PushVxB(
            self._electrons,
            algo=electron_params['push_algos']['vxb'],
            scale_fac=1.,
            b_eq=self._b_background,
            b_tilde=None,
            f0=None)]  # no conventional weights update here, thus f0=None

        self._propagators += [propagators_coupling.EfieldWeights(
            self._e,
            self._electrons,
            alpha=self.alpha,
            f0=self._f0,
            **params['solvers']['solver_ew']
        )]

        self._propagators += [propagators_fields.Maxwell(
            self._e,
            self._b,
            **params['solvers']['solver_eb'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_e'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_b'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_w'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    @property
    def propagators(self):
        return self._propagators  

    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def initialize_from_params(self):
        from struphy.propagators import solvers
        from struphy.pic.particles_to_grid import AccumulatorVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Correct initialization weights by dividing by N*sqrt(f_0)
        self._electrons.markers[~self._electrons.holes, 6] /= (self._electrons.n_mks *
                                                               np.sqrt(self._f0(*self._electrons.markers_wo_holes[:, :6].T)))

        # evaluate f0
        f0_values = self._f0(self._electrons.markers[:, 0],
                             self._electrons.markers[:, 1],
                             self._electrons.markers[:, 2],
                             self._electrons.markers[:, 3],
                             self._electrons.markers[:, 4],
                             self._electrons.markers[:, 5])

        # Accumulate charge density
        charge_accum = AccumulatorVector(self.derham, self.domain, "H1", "linear_vlasov_maxwell_poisson")
        charge_accum.accumulate(self._electrons, f0_values)

        # compute ion charge to have charge neutral rhs of Poisson solver
        charge = np.zeros(1)
        self.derham.comm.Allreduce(np.sum(charge_accum.vectors[0].toarray()), charge, op=self._mpi_sum)

        # Subtract the total charge local to each process
        charge_accum._vectors[0][:] -= np.sum(charge_accum.vectors[0].toarray())

        # Then solve Poisson equation
        poisson_solver = solvers.PoissonSolver(rho=charge_accum.vectors[0], **self._poisson_params)
        poisson_solver(0.)
        self.derham.grad.dot(-poisson_solver._phi, out=self._e)

    def update_scalar_quantities(self):

        # e^T * M_1 * e
        self._scalar_quantities['en_e'][0] = self._e.dot(
            self._mass_ops.M1.dot(self._e)) / 2.

        # b^T * M_2 * b
        self._scalar_quantities['en_b'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b)) / 2.

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * N/2 * sum_p s_0 * w_p^2
        self._scalar_quantities['en_w'][0] = \
            self.alpha**2 * self._electrons.n_mks / 2. * \
            self._maxwellian_params['vthx']**2 * \
            self._maxwellian_params['vthy']**2 * \
            self._maxwellian_params['vthz']**2 * \
            np.dot(self._electrons.markers_wo_holes[:, 6]**2,
                   self._electrons.markers_wo_holes[:, 7])

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._scalar_quantities['en_w'], op=self._mpi_sum)

        # en_tot = en_w + en_e + en_b
        self._scalar_quantities['en_tot'][0] = \
            self._scalar_quantities['en_w'][0] + \
            self._scalar_quantities['en_e'][0] + \
            self._scalar_quantities['en_b'][0]


#############################
# Toy models
#############################
class Maxwell(StruphyModel):
    r'''Maxwell's equations in vacuum.

    :ref:`normalization`:

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

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return None
    
    @classmethod
    def timescale(cls):
        return 'light'
    
    def __init__(self, params, comm):

        super().__init__(params, comm, e1='Hcurl', b2='Hdiv')
            
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields

        # Pointers to em-field variables
        self._e = self.em_fields['e1']['obj'].vector
        self._b = self.em_fields['b2']['obj'].vector

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # set propagators base class attributes
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.Maxwell(
            self._e,
            self._b,
            **solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)
        
        # temporary vectors for scalar quantities
        self._tmp_e = self.derham.Vh['1'].zeros()
        self._tmp_b = self.derham.Vh['2'].zeros()

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):
        self._mass_ops.M1.dot(self._e, out=self._tmp_e)
        self._mass_ops.M2.dot(self._b, out=self._tmp_b)
        
        en_E = self._e.dot(self._tmp_e)/2
        en_B = self._b.dot(self._tmp_b)/2
        
        self._scalar_quantities['en_E'][0] = en_E
        self._scalar_quantities['en_B'][0] = en_B
        
        self._scalar_quantities['en_tot'][0] = en_E + en_B


class Vlasov(StruphyModel):
    r'''Vlasov equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat \omega = \hat \Omega_\textnormal{c} := \frac{Ze \hat B}{A m_\textnormal{H}}\,,\qquad \frac{\hat \omega}{\hat k} = \hat v\,,
        
    Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left(\mathbf{v}\times\mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,,

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return 'ions'
    
    @classmethod
    def timescale(cls):
        return 'cyclotron'
    
    def __init__(self, params, comm):

        super().__init__(params, comm, ions='Particles6D')
            
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_markers

        # pointer to ions
        ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        print(
            f'Total number of markers : {ions.n_mks}, shape of markers array on rank {self.derham.comm.Get_rank()} : {ions.markers.shape}')

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # set propagators base class attributes
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        self._propagators += [propagators_markers.PushVxB(
            ions,
            algo=ions_params['push_algos']['vxb'],
            scale_fac=1.,
            b_eq=self._b_eq,
            b_tilde=None,
            f0=None)]

        self._propagators += [propagators_markers.PushEta(
            ions,
            algo=ions_params['push_algos']['eta'],
            bc_type=ions_params['markers']['bc_type'],
            f0=None)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}

        self._en_fv_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fv'] = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fB'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):
        pass


class DriftKinetic(StruphyModel):
    r'''Drift-kinetic equation in static background magnetic field (guiding-center motion). 

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k}
    
    Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[ \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and 

    .. math::

        \mathbf{E}^* = - \frac{\mu}{\kappa} \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \frac{1}{\kappa} v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Moreover, 
    
    .. math::
    
        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{c}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def bulk_species(cls):
        return 'ions'
    
    @classmethod
    def timescale(cls):
        return 'alfvén'
    
    def __init__(self, params, comm):

        super().__init__(params, comm, ions='Particles5D')
            
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_markers
        from mpi4py.MPI import SUM, IN_PLACE

        # pointer to ions
        self._ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        self._abs_b = self.derham.P['0'](self.mhd_equil.absB0)

        self._unit_b1 = self.derham.P['1']([self.mhd_equil.unit_b1_1,
                                            self.mhd_equil.unit_b1_2,
                                            self.mhd_equil.unit_b1_3])

        self._unit_b2 = self.derham.P['2']([self.mhd_equil.unit_b2_1,
                                            self.mhd_equil.unit_b2_2,
                                            self.mhd_equil.unit_b2_3])

        self._b_cart = self.derham.P['v']([self.mhd_equil.b_cart_1,
                                           self.mhd_equil.b_cart_2,
                                           self.mhd_equil.b_cart_3])

        self._E0T = self.derham.E['0'].transpose()
        self._EvT = self.derham.E['v'].transpose()
        self._b_cart = self._EvT.dot(self._b_cart)     
        
        ee = 1.602176634e-19 # elementary charge (C)
        mH = 1.67262192369e-27 # proton mass (kg)
        
        Ah = params['kinetic']['ions']['phys_params']['A']
        Zh = params['kinetic']['ions']['phys_params']['Z']
        
        omega_ch = (Zh*ee*self.units_basic['B'])/(Ah*mH)
        kappa = omega_ch*self.units_basic['t']
        
        if abs(kappa - 1) < 1e-6:
            kappa = 1.              

        # set propagators base class attributes
        Propagator.derham = self.derham
        Propagator.domain = self.domain

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepPushGuidingCenter1(
            self._ions,
            kappa=kappa,
            b_eq=self._b_eq, 
            unit_b1=self._unit_b1, 
            unit_b2=self._unit_b2, 
            abs_b=self._abs_b,
            integrator=ions_params['push_algos']['integrator'],
            method=ions_params['push_algos']['method'],
            maxiter=ions_params['push_algos']['maxiter'],
            tol=ions_params['push_algos']['tol'])]
        self._propagators += [propagators_markers.StepPushGuidingCenter2(
            self._ions,
            kappa=kappa,
            b_eq=self._b_eq, 
            unit_b1=self._unit_b1, 
            unit_b2=self._unit_b2, 
            abs_b=self._abs_b,
            method=ions_params['push_algos']['method'],
            integrator=ions_params['push_algos']['integrator'],
            maxiter=ions_params['push_algos']['maxiter'],
            tol=ions_params['push_algos']['tol'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._en_fv_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fv'] = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fB'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):

        self._en_fv_loc = self._ions.markers[~self._ions.holes, 5].dot(self._ions.markers[~self._ions.holes, 3]**2) / (2*self._ions.n_mks)
        self.derham.comm.Reduce(self._en_fv_loc, self._scalar_quantities['en_fv'], op=self._mpi_sum, root=0)

        # calculate particle magnetic energy
        self._ions.save_magnetic_energy(self._derham, self._E0T.dot(self.derham.P['0'](self.mhd_equil.absB0)))

        self._en_fB_loc = self._ions.markers[~self._ions.holes, 5].dot(self._ions.markers[~self._ions.holes, 8]) / self._ions.n_mks
        self.derham.comm.Reduce(self._en_fB_loc, self._scalar_quantities['en_fB'], op=self._mpi_sum, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_fv'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fB'][0]


########################################################
# Hybrid models with kinetic ions and massless electrons
########################################################
class VlasovMasslessElectrons(StruphyModel):
    r'''Hybrid (kinetic ions + massless electrons) equations with quasi-neutrality condition. 
    Unknowns: distribution function for ions, and vector potential.

    Normalization: 

    .. math::
            t, x, p, A, f...


    Implemented equations:

    Hyrid model with kinetic ions and massless electrons.

    .. math::

        \begin{align}
        \textnormal{Vlasov}\qquad& \frac{\partial f}{\partial t} + (\mathbf{p} - \mathbf{A}) \cdot \frac{\partial f}{\partial \mathbf{x}}
        - \left[ T_e \frac{\nabla n}{n} - \left( \frac{\partial{\mathbf A}}{\partial {\mathbf x}} \right)^\top ({\mathbf A} - {\mathbf p} )  \right] \cdot \frac{\partial f}{\partial \mathbf{p}}
        = 0\,,
        \\
        \textnormal{Faraday's law}\qquad& \frac{\partial {\mathbf A}}{\partial t} = - \frac{\nabla \times \nabla \times A}{n} \times \nabla \times {\mathbf A} - \frac{\int ({\mathbf A} - {\mathbf p}f \mathrm{d}{\mathbf p})}{n} \times \nabla \times {\mathbf A}, \quad n = \int f \mathrm{d}{\mathbf p}.
        \end{align}

    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    @classmethod
    def bulk_species(cls):
        return 'ions'
    
    @classmethod
    def timescale(cls):
        return 'cyclotron'
    
    def __init__(self, params, comm):

        from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_markers
        from mpi4py.MPI import SUM, IN_PLACE
        from struphy.pic.particles_to_grid import Accumulator

        super().__init__(params, comm, a1='Hcurl', ions='Particles6D')

        # pointers to em-field variables
        self._a = self.em_fields['a1']['obj'].vector

        # pointer to kinetic variables
        self._ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        # extract necessary parameters
        shape_params = params['kinetic']['ions']['ionsshape'] # shape function info, degree and support size
        self.thermal = params['kinetic']['electrons']['temperature'] # electron temperature 

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        
        # Project background magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1, 
                                         self.mhd_equil.b2_2, 
                                         self.mhd_equil.b2_3])

        # set propagators base class attributes
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        Propagator.mass_ops = self.mass_ops

        self._accum_density = Accumulator(self.derham, 
                                          self.domain, 
                                          'H1', 
                                          'hybrid_fA_density', 
                                          add_vector=False)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        
        self._propagators += [propagators_markers.StepHybridXPSymplectic(
                              self._ions, 
                              a = self._a, 
                              particle_bc = ions_params['markers']['bc_type'], 
                              quad_number = params['grid']['nq_el'], 
                              shape_degree = np.array(shape_params['degree']), 
                              shape_size = np.array(shape_params['size']), 
                              electron_temperature = self.thermal, 
                              accumulate_density = self._accum_density)]

        self._propagators += [propagators_markers.StepPushpxBHybrid(
                              self._ions, 
                              method = ions_params['push_algos']['pxb'], 
                              a = self._a, 
                              b_eq = self._b_eq)]
        
        self._propagators += [propagators_fields.FaradayExtended(
                              self._a, 
                              a_space = 'Hcurl', 
                              beq = self._b_eq, 
                              particles = self._ions, 
                              quad_number = params['grid']['nq_el'], 
                              shape_degree = np.array(shape_params['degree']), 
                              shape_size = np.array(shape_params['size']), 
                              solver_params = solver_params_1, 
                              accumulate_density = self._accum_density)]
        
        # Scalar variables to be saved during simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._en_thermal_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_thermal'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM

    @property
    def propagators(self):
        return self._propagators  
    
    @property
    def scalar_quantities(self):
        return self._scalar_quantities

    def update_scalar_quantities(self):
        import struphy.pic.utilities as pic_util

        rank = self._derham.comm.Get_rank()

        self._curla = self._derham.curl.dot(self._a)

        self._scalar_quantities['en_B'][0] = (self._curla + self._b_eq).dot(
            self._mass_ops.M2.dot(self._curla + self._b_eq))/2

        self._en_f_loc = pic_util.get_kinetic_energy_particles(self._a, self._derham, self._domain, self._ions)/self._ions.n_mks

        self.derham.comm.Reduce(self._en_f_loc, self._scalar_quantities['en_f'], op=self._mpi_sum, root=0)

        self._en_thermal_loc = pic_util.get_electron_thermal_energy(self._accum_density, self._derham, self._domain, int(self._derham.domain_array[int(rank), 2]), int(self._derham.domain_array[int(rank), 5]), int(self._derham.domain_array[int(rank), 8]), int(self._derham.quad_order[0]+1), int(self._derham.quad_order[1]+1), int(self._derham.quad_order[2]+1) )

        self.derham.comm.Reduce(self.thermal*self._en_thermal_loc, self._scalar_quantities['en_thermal'], op=self._mpi_sum, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_thermal'][0]
