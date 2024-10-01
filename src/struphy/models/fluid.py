import numpy as np

from struphy.models.base import StruphyModel
from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers


class LinearMHD(StruphyModel):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::
    
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        = (\nabla \times \tilde{\mathbf{B}})\times \mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
    2. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b_field'] = 'Hdiv'
        dct['fluid']['mhd'] = {'density': 'L2', 'velocity': 'Hdiv', 'pressure': 'L2'}
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.ShearAlfven: ['mhd_velocity', 'b_field'],
                propagators_fields.Magnetosonic: ['mhd_density', 'mhd_velocity', 'mhd_pressure']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    # add special options
    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(species=['fluid', 'mhd'], key='u_space',
                       option='Hdiv', dct=dct)
        return dct

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        u_space = params['fluid']['mhd']['options']['u_space']
        alfven_solver = params['fluid']['mhd']['options']['ShearAlfven']['solver']
        sonic_solver = params['fluid']['mhd']['options']['Magnetosonic']['solver']

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

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.ShearAlfven] = {'u_space': u_space,
                                                        'solver': alfven_solver}

        self._kwargs[propagators_fields.Magnetosonic] = {'b': self.pointer['b_field'],
                                                         'u_space': u_space,
                                                         'solver': sonic_solver}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_p_eq')
        self.add_scalar('en_B_eq')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()
        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_velocity'], out=self._tmp_u1)
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._tmp_b1)

        en_U = self.pointer['mhd_velocity'] .dot(self._tmp_u1)/2
        en_B = self.pointer['b_field'] .dot(self._tmp_b1)/2
        en_p = self.pointer['mhd_pressure'] .dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p)
        self.update_scalar('en_tot', en_U + en_B + en_p)

        # background fields
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2
        en_p0 = self._p_eq.dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_B_eq', en_B0)
        self.update_scalar('en_p_eq', en_p0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b_field']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)


class LinearExtendedMHDuniform(StruphyModel):
    r'''Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). 
    For uniform background conditions only.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \frac{5}{3}\,p_{0}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 - \frac{1}{\varepsilon} \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right)
        = 0\,.

    where

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfvenB1`
    2. :class:`~struphy.propagators.propagators_fields.Hall`
    3. :class:`~struphy.propagators.propagators_fields.MagnetosonicUniform`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b_field'] = 'Hcurl'
        dct['fluid']['mhd'] = {'rho': 'L2', 
                               'u': 'Hdiv', 
                               'p': 'L2',
                               }
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'
    
    @staticmethod
    def propagators_dct():
        return {propagators_fields.ShearAlfvenB1: ['mhd_u', 'b_field'],
                propagators_fields.Hall: ['b_field'],
                propagators_fields.MagnetosonicUniform: ['mhd_rho', 'mhd_u', 'mhd_p']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        alfven_solver = params['fluid']['mhd']['options']['ShearAlfvenB1']['solver']
        M1_inv = params['fluid']['mhd']['options']['ShearAlfvenB1']['solver_M1']
        hall_solver = params['em_fields']['options']['Hall']['solver']
        sonic_solver = params['fluid']['mhd']['options']['MagnetosonicUniform']['solver']

        # project background magnetic field (1-form) and pressure (3-form)
        self._b_eq = self.derham.P['1']([self.mhd_equil.b1_1,
                                         self.mhd_equil.b1_2,
                                         self.mhd_equil.b1_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self.pointer['mhd_p'].space.zeros()
        # project background vector potential (1-form)
        self._a_eq = self.derham.P['1']([self.mhd_equil.a1_1,
                                         self.mhd_equil.a1_2,
                                         self.mhd_equil.a1_3])

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # compute coupling parameters
        epsilon = self.equation_params['mhd']['epsilon']

        if abs(epsilon - 1) < 1e-6:
            epsilon = 1.

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.ShearAlfvenB1] = {'solver': alfven_solver,
                                                          'solver_M1': M1_inv}

        self._kwargs[propagators_fields.Hall] = {'solver': hall_solver,
                                                 'epsilon': epsilon}
        
        self._kwargs[propagators_fields.MagnetosonicUniform] = {'solver': sonic_solver}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_p_eq')
        self.add_scalar('en_B_eq')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')
        self.add_scalar('helicity')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()
        self._tmp_b1 = self.derham.Vh['1'].zeros()
        self._tmp_b2 = self.derham.Vh['1'].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_u'], out=self._tmp_u1)

        self._mass_ops.M1.dot(self.pointer['b_field'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u'].dot(self._tmp_u1)/2.0
        en_B = self.pointer['b_field'].dot(self._tmp_b1)/2.0
        helicity = self._a_eq.dot(self._tmp_b1)*2.0
        en_p_i = self.pointer['mhd_p'].dot(self._ones)/(5.0/3.0 - 1.0)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p_i)
        self.update_scalar('helicity', helicity)
        self.update_scalar('en_tot', en_U + en_B + en_p_i)

        # background fields
        self._mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2.0
        en_p0 = self._p_eq.dot(self._ones)/(5.0/3.0 - 1.0)

        self.update_scalar('en_B_eq', en_B0)
        self.update_scalar('en_p_eq', en_p0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b_field']

        self._mass_ops.M1.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2.0

        self.update_scalar('en_B_tot', en_Btot)


class ColdPlasma(StruphyModel):
    r'''Cold plasma model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,. 

    :ref:`Equations <gempic>`:

    .. math::

        \frac{1}{n_0} &\frac{\partial \mathbf j}{\partial t} = \frac{1}{\varepsilon} \mathbf E + \frac{1}{\varepsilon n_0} \mathbf j \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and
    
    .. math::
    
        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,, \qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['em_fields']['b_field'] = 'Hdiv'
        dct['fluid']['electrons'] = {'j': 'Hcurl'}
        return dct

    @staticmethod
    def bulk_species():
        return 'electrons'

    @staticmethod
    def velocity_scale():
        return 'light'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.Maxwell: ['e_field', 'b_field'],
                propagators_fields.OhmCold: ['electrons_j', 'e_field'],
                propagators_fields.JxBCold: ['electrons_j']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        # model parameters
        self._alpha = self.equation_params['electrons']['alpha']
        self._epsilon = self.equation_params['electrons']['epsilon']

        # solver parameters
        params_maxwell = params['em_fields']['options']['Maxwell']['solver']
        params_ohmcold = params['fluid']['electrons']['options']['OhmCold']['solver']
        params_jxbcold = params['fluid']['electrons']['options']['JxBCold']['solver']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {'solver': params_maxwell}
        
        self._kwargs[propagators_fields.OhmCold] = {'alpha': self._alpha,
                                                    'epsilon': self._epsilon,
                                                    'solver': params_ohmcold}
        
        self._kwargs[propagators_fields.JxBCold] = {'epsilon': self._epsilon,
                                                    'solver': params_jxbcold}
        
        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('electric energy')
        self.add_scalar('magnetic energy')
        self.add_scalar('kinetic energy')
        self.add_scalar('total energy')

        # temporaries
        self._tmp1 = self.pointer['e_field'].space.zeros()
        self._tmp2 = self.pointer['b_field'].space.zeros()

    def update_scalar_quantities(self):

        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._tmp2)
        en_E = .5 * self.pointer['e_field'].dot(self._tmp1)
        en_B = .5 * self.pointer['b_field'].dot(self._tmp2)

        self._mass_ops.M1ninv.dot(self.pointer['electrons_j'], out=self._tmp1)
        en_J = .5 * self._alpha**2 * \
            self.pointer['electrons_j'].dot(self._tmp1)

        self.update_scalar('electric energy', en_E)
        self.update_scalar('magnetic energy', en_B)
        self.update_scalar('kinetic energy', en_J)
        self.update_scalar('total energy', en_E + en_B + en_J)


class VariationalMHD(StruphyModel):
    r'''Full (non-linear) MHD equations discretized with a variational method 
    (see https://www.arxiv.org/abs/2402.02905 for more details about the scheme).
    
    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho,s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho,s))}{\partial s} + \mathbf B \times \nabla \times \mathbf B = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) = 0 \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalMagFieldEvolve`

    :ref:`Model info <add_model>`:
    '''
    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.VariationalDensityEvolve: ['mhd_rho3', 'mhd_uv'],
                propagators_fields.VariationalMomentumAdvection: ['mhd_uv'],
                propagators_fields.VariationalEntropyEvolve: ['mhd_s3', 'mhd_uv'],
                propagators_fields.VariationalMagFieldEvolve: ['b2', 'mhd_uv']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator
        import numpy as np
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'],
            self.derham.Vh_fem['v'],
            V_extraction_op=self.derham.extraction_ops['v'],
            W_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            W_boundary_op=self.derham.boundary_ops['v'])

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params['fluid']['mhd']['options']['VariationalMomentumAdvection']['lin_solver']
        nonlin_solver_momentum = params['fluid']['mhd']['options']['VariationalMomentumAdvection']['nonlin_solver']
        lin_solver_density = params['fluid']['mhd']['options']['VariationalDensityEvolve']['lin_solver']
        nonlin_solver_density = params['fluid']['mhd']['options']['VariationalDensityEvolve']['nonlin_solver']
        lin_solver_entropy = params['fluid']['mhd']['options']['VariationalEntropyEvolve']['lin_solver']
        nonlin_solver_entropy = params['fluid']['mhd']['options']['VariationalEntropyEvolve']['nonlin_solver']
        lin_solver_magfield = params['em_fields']['options']['VariationalMagFieldEvolve']['lin_solver']
        nonlin_solver_magfield = params['em_fields']['options']['VariationalMagFieldEvolve']['nonlin_solver']

        self._gamma = params['fluid']['mhd']['options']['VariationalDensityEvolve']['physics']['gamma']
        model = 'full'

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {'model': model,
                                                                     's': self.pointer['mhd_s3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_density,
                                                                     'nonlin_solver': nonlin_solver_density}

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {'mass_ops': self.WMM,
                                                                         'lin_solver': lin_solver_momentum,
                                                                         'nonlin_solver': nonlin_solver_momentum}

        self._kwargs[propagators_fields.VariationalEntropyEvolve] = {'model': model,
                                                                     'rho': self.pointer['mhd_rho3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_entropy,
                                                                     'nonlin_solver': nonlin_solver_entropy}
        
        self._kwargs[propagators_fields.VariationalMagFieldEvolve] = {'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_magfield,
                                                                     'nonlin_solver': nonlin_solver_magfield}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_mag')
        self.add_scalar('en_tot')
        self.add_scalar('dens_tot')
        self.add_scalar('entr_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh_pol['v'].zeros()
        self._tmp_wb2 = self.derham.Vh_pol['2'].zeros()
        tmp_dof = self.derham.Vh_pol['3'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol['3'].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

    def update_scalar_quantities(self):

        # Update mass matrix
        rhon = self.pointer['mhd_rho3']
        self._propagators[0].rhof1.vector = rhon

        self._propagators[0]._update_weighted_MM()

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['mhd_uv'], out=self._tmp_m1)

        en_U = self.pointer['mhd_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        wb2 = self._mass_ops.M2.dot(self.pointer['b2'])
        en_mag = wb2.dot(self.pointer['b2'])/2
        self.update_scalar('en_mag', en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar('en_tot', en_tot)

        dens_tot = self._ones.dot(self.pointer['mhd_rho3'])
        self.update_scalar('dens_tot', dens_tot)
        entr_tot = self._ones.dot(self.pointer['mhd_s3'])
        self.update_scalar('entr_tot', entr_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        '''
        en_prop = self._propagators[0]
        en_prop.sf.vector = self.pointer['mhd_s3']
        en_prop.rhof.vector = self.pointer['mhd_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_drho)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        return np.power(rho, gam)*np.exp(s/rho)


class ViscoresistiveMHD(StruphyModel):
    r'''Full (non-linear) visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = \frac{1}{T}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`, 
    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalMagFieldEvolve`
    5. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    6. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    '''
    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'
    
    @staticmethod
    def propagators_dct():
        return {propagators_fields.VariationalDensityEvolve: ['mhd_rho3', 'mhd_uv'],
                propagators_fields.VariationalMomentumAdvection: ['mhd_uv'],
                propagators_fields.VariationalEntropyEvolve: ['mhd_s3', 'mhd_uv'],
                propagators_fields.VariationalMagFieldEvolve: ['b2', 'mhd_uv'],
                propagators_fields.VariationalViscosity: ['mhd_s3', 'mhd_uv'],
                propagators_fields.VariationalResistivity: ['mhd_s3', 'b2'],}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator
        import numpy as np
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'],
            self.derham.Vh_fem['v'],
            V_extraction_op=self.derham.extraction_ops['v'],
            W_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            W_boundary_op=self.derham.boundary_ops['v'])

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params['fluid']['mhd']['options']['VariationalMomentumAdvection']['lin_solver']
        nonlin_solver_momentum = params['fluid']['mhd']['options']['VariationalMomentumAdvection']['nonlin_solver']
        lin_solver_density = params['fluid']['mhd']['options']['VariationalDensityEvolve']['lin_solver']
        nonlin_solver_density = params['fluid']['mhd']['options']['VariationalDensityEvolve']['nonlin_solver']
        lin_solver_entropy = params['fluid']['mhd']['options']['VariationalEntropyEvolve']['lin_solver']
        nonlin_solver_entropy = params['fluid']['mhd']['options']['VariationalEntropyEvolve']['nonlin_solver']
        lin_solver_magfield = params['em_fields']['options']['VariationalMagFieldEvolve']['lin_solver']
        nonlin_solver_magfield = params['em_fields']['options']['VariationalMagFieldEvolve']['nonlin_solver']
        lin_solver_viscosity = params['fluid']['mhd']['options']['VariationalViscosity']['lin_solver']
        nonlin_solver_viscosity = params['fluid']['mhd']['options']['VariationalViscosity']['nonlin_solver']
        lin_solver_resistivity = params['fluid']['mhd']['options']['VariationalResistivity']['lin_solver']
        nonlin_solver_resistivity = params['fluid']['mhd']['options']['VariationalResistivity']['nonlin_solver']
        if 'linearize_current' in params['fluid']['mhd']['options']['VariationalResistivity'].keys():
            self._linearize_current = params['fluid']['mhd']['options']['VariationalResistivity']['linearize_current']
        else :
            self._linearize_current = False
        self._gamma = params['fluid']['mhd']['options']['VariationalDensityEvolve']['physics']['gamma']
        self._mu = params['fluid']['mhd']['options']['VariationalViscosity']['physics']['mu']
        self._mu_a = params['fluid']['mhd']['options']['VariationalViscosity']['physics']['mu_a']
        self._eta = params['fluid']['mhd']['options']['VariationalResistivity']['physics']['eta']
        self._eta_a = params['fluid']['mhd']['options']['VariationalResistivity']['physics']['eta_a']
        model = 'full'

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {'model': model,
                                                                     's': self.pointer['mhd_s3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_density,
                                                                     'nonlin_solver': nonlin_solver_density}

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {'mass_ops': self.WMM,
                                                                         'lin_solver': lin_solver_momentum,
                                                                         'nonlin_solver': nonlin_solver_momentum}

        self._kwargs[propagators_fields.VariationalEntropyEvolve] = {'model': model,
                                                                     'rho': self.pointer['mhd_rho3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_entropy,
                                                                     'nonlin_solver': nonlin_solver_entropy}
        
        self._kwargs[propagators_fields.VariationalMagFieldEvolve] = {'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_magfield,
                                                                     'nonlin_solver': nonlin_solver_magfield}
        
        self._kwargs[propagators_fields.VariationalViscosity] = {'model': model,
                                                                 'rho': self.pointer['mhd_rho3'],
                                                                 'gamma': self._gamma,
                                                                 'mu': self._mu,
                                                                 'mu_a': self._mu_a,
                                                                 'mass_ops': self.WMM,
                                                                 'lin_solver': lin_solver_viscosity,
                                                                 'nonlin_solver': nonlin_solver_viscosity}
        
        self._kwargs[propagators_fields.VariationalResistivity] = {'model': model,
                                                                   'rho': self.pointer['mhd_rho3'],
                                                                   'gamma': self._gamma,
                                                                   'eta': self._eta,
                                                                   'eta_a': self._eta_a,
                                                                   'lin_solver': lin_solver_resistivity,
                                                                   'nonlin_solver': nonlin_solver_resistivity,
                                                                   'linearize_current' :self._linearize_current}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_mag')
        self.add_scalar('en_tot')
        self.add_scalar('dens_tot')
        self.add_scalar('entr_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh_pol['v'].zeros()
        self._tmp_wb2 = self.derham.Vh_pol['2'].zeros()
        tmp_dof = self.derham.Vh_pol['3'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol['3'].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

    def update_scalar_quantities(self):

        # Update mass matrix
        rhon = self.pointer['mhd_rho3']
        self._propagators[0].rhof1.vector = rhon

        self._propagators[0]._update_weighted_MM()

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['mhd_uv'], out=self._tmp_m1)

        en_U = self.pointer['mhd_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        wb2 = self._mass_ops.M2.dot(self.pointer['b2'])
        en_mag = wb2.dot(self.pointer['b2'])/2
        self.update_scalar('en_mag', en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar('en_tot', en_tot)

        dens_tot = self._ones.dot(self.pointer['mhd_rho3'])
        self.update_scalar('dens_tot', dens_tot)
        entr_tot = self._ones.dot(self.pointer['mhd_s3'])
        self.update_scalar('entr_tot', entr_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        '''
        en_prop = self._propagators[0]
        en_prop.sf.vector = self.pointer['mhd_s3']
        en_prop.rhof.vector = self.pointer['mhd_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_drho)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        return np.power(rho, gam)*np.exp(s/rho)


class ViscousFluid(StruphyModel):
    r'''Full (non-linear) viscous Navier-Stokes equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} - \nabla \cdot \left((\mu +\mu_a(\mathbf x)) \nabla \mathbf u\right) = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = \frac{1}{T}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 \right) \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.
    and :math:`\mu_a(\mathbf x)` is an artificial viscosity coefficient.
    
    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`

    :ref:`Model info <add_model>`:
    '''
    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
        return dct

    @staticmethod
    def bulk_species():
        return 'fluid'

    @staticmethod
    def velocity_scale():
        return 'alfvén'
    
    @staticmethod
    def propagators_dct():
        return {propagators_fields.VariationalDensityEvolve: ['fluid_rho3', 'fluid_uv'],
                propagators_fields.VariationalMomentumAdvection: ['fluid_uv'],
                propagators_fields.VariationalEntropyEvolve: ['fluid_s3', 'fluid_uv'],
                propagators_fields.VariationalViscosity: ['fluid_s3', 'fluid_uv']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]
    
    def __init__(self, params, comm, inter_comm = None):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator
        import numpy as np
        from struphy.polar.basic import PolarVector

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)
        
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'],
            self.derham.Vh_fem['v'],
            V_extraction_op=self.derham.extraction_ops['v'],
            W_extraction_op=self.derham.extraction_ops['v'],
            V_boundary_op=self.derham.boundary_ops['v'],
            W_boundary_op=self.derham.boundary_ops['v'])

        # Initialize propagators/integrators used in splitting substeps
        lin_solver_momentum = params['fluid']['fluid']['options']['VariationalMomentumAdvection']['lin_solver']
        nonlin_solver_momentum = params['fluid']['fluid']['options']['VariationalMomentumAdvection']['nonlin_solver']
        lin_solver_density = params['fluid']['fluid']['options']['VariationalDensityEvolve']['lin_solver']
        nonlin_solver_density = params['fluid']['fluid']['options']['VariationalDensityEvolve']['nonlin_solver']
        lin_solver_entropy = params['fluid']['fluid']['options']['VariationalEntropyEvolve']['lin_solver']
        nonlin_solver_entropy = params['fluid']['fluid']['options']['VariationalEntropyEvolve']['nonlin_solver']
        lin_solver_viscosity = params['fluid']['fluid']['options']['VariationalViscosity']['lin_solver']
        nonlin_solver_viscosity = params['fluid']['fluid']['options']['VariationalViscosity']['nonlin_solver']

        self._gamma = params['fluid']['fluid']['options']['VariationalDensityEvolve']['physics']['gamma']
        self._mu = params['fluid']['fluid']['options']['VariationalViscosity']['physics']['mu']
        self._mu_a = params['fluid']['fluid']['options']['VariationalViscosity']['physics']['mu_a']
        model = 'full'


        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {'model': model,
                                                                     's': self.pointer['fluid_s3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_density,
                                                                     'nonlin_solver': nonlin_solver_density}

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {'mass_ops': self.WMM,
                                                                         'lin_solver': lin_solver_momentum,
                                                                         'nonlin_solver': nonlin_solver_momentum}

        self._kwargs[propagators_fields.VariationalEntropyEvolve] = {'model': model,
                                                                     'rho': self.pointer['fluid_rho3'],
                                                                     'gamma': self._gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_entropy,
                                                                     'nonlin_solver': nonlin_solver_entropy}
        
        self._kwargs[propagators_fields.VariationalViscosity] = {'model': model,
                                                                 'gamma': self._gamma,
                                                                 'rho': self.pointer['fluid_rho3'],
                                                                 'mu': self._mu,
                                                                 'mu_a': self._mu_a,
                                                                 'mass_ops': self.WMM,
                                                                 'lin_solver': lin_solver_viscosity,
                                                                 'nonlin_solver': nonlin_solver_viscosity}
        
        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_tot')
        self.add_scalar('dens_tot')
        self.add_scalar('entr_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh_pol['v'].zeros()
        self._tmp_wb2 = self.derham.Vh_pol['2'].zeros()
        tmp_dof = self.derham.Vh_pol['3'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f, dofs=tmp_dof)

        self._ones = self.derham.Vh_pol['3'].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

    def update_scalar_quantities(self):

        # Update mass matrix
        rhon = self.pointer['fluid_rho3']
        self._propagators[0].rhof1.vector = rhon

        self._propagators[0]._update_weighted_MM()

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['fluid_uv'], out=self._tmp_m1)

        en_U = self.pointer['fluid_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo
        self.update_scalar('en_tot', en_tot)

        dens_tot = self._ones.dot(self.pointer['fluid_rho3'])
        self.update_scalar('dens_tot', dens_tot)
        entr_tot = self._ones.dot(self.pointer['fluid_s3'])
        self.update_scalar('entr_tot', entr_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        '''
        en_prop = self._propagators[0]
        en_prop.sf.vector = self.pointer['fluid_s3']
        en_prop.rhof.vector = self.pointer['fluid_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_drho)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._gamma
        return np.power(rho, gam)*np.exp(s/rho)
