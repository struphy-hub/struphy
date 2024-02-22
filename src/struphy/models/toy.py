import numpy as np
from struphy.models.base import StruphyModel


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
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e1'] = 'Hcurl'
        dct['em_fields']['b2'] = 'Hdiv'
        return dct

    @classmethod
    def bulk_species(cls):
        return None

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import Maxwell

        dct = {}
        cls.add_option(species='em_fields', key='solver',
                       option=Maxwell.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        # extract necessary parameters
        solver_params = params['em_fields']['options']['solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **solver_params))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_E')
        self.add_scalar('en_B')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_e = self.derham.Vh['1'].zeros()
        self._tmp_b = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):
        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp_e)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b)

        en_E = self.pointer['e1'].dot(self._tmp_e)/2
        en_B = self.pointer['b2'].dot(self._tmp_b)/2

        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_tot', en_E + en_B)


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
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['ions'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'ions'

    @classmethod
    def velocity_scale(cls):
        return 'cyclotron'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_markers import PushEta, PushVxB

        dct = {}
        cls.add_option(species=['kinetic', 'ions'], key='push_eta',
                       option=PushEta.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'ions'], key='push_vxb',
                       option=PushVxB.options()['algo'], dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['ions']['params']

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['ions'],
            algo=ions_params['options']['push_vxb'],
            scale_fac=1.,
            b_eq=self._b_eq,
            b_tilde=None))
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['ions'],
            algo=ions_params['options']['push_eta'],
            bc_type=ions_params['markers']['bc']['type']))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_f')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):

        self._tmp[0] = self.pointer['ions'].markers_wo_holes[:, 6].dot(
            self.pointer['ions'].markers_wo_holes[:, 3]**2 +
            self.pointer['ions'].markers_wo_holes[:, 4]**2 +
            self.pointer['ions'].markers_wo_holes[:, 5]**2) / (2*self.pointer['ions'].n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_f', self._tmp[0])


class DriftKinetic(StruphyModel):
    r'''Drift-kinetic equation in static background magnetic field (guiding-center motion). 

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k}

    Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\epsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and 

    .. math::

        \mathbf{E}^* = - -\epsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Moreover, 

    .. math::

        \epsilon = \frac{\hat \omega }{2\pi \hat \Omega_{\textnormal{c}}}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['ions'] = 'Particles5D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'ions'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_markers import PushGuidingCenterbxEstar, PushGuidingCenterBstar

        dct = {}
        cls.add_option(species=['kinetic', 'ions'], key='push_bxEstar',
                       option=PushGuidingCenterbxEstar.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'ions'], key='push_Bstar',
                       option=PushGuidingCenterBstar.options()['algo'], dct=dct)

        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
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

        self._gradB1 = self.derham.P['1']([self.mhd_equil.gradB1_1,
                                           self.mhd_equil.gradB1_2,
                                           self.mhd_equil.gradB1_3])

        self._curl_unit_b2 = self.derham.P['2']([self.mhd_equil.curl_unit_b2_1,
                                                 self.mhd_equil.curl_unit_b2_2,
                                                 self.mhd_equil.curl_unit_b2_3])

        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._EvT = self.derham.extraction_ops['v'].transpose()

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushGuidingCenterbxEstar(
            self.pointer['ions'],
            epsilon=self.equation_params['ions']['epsilon_unit'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            gradB1=self._gradB1,
            curl_unit_b2=self._curl_unit_b2,
            **ions_params['options']['push_bxEstar']))
        self.add_propagator(self.prop_markers.PushGuidingCenterBstar(
            self.pointer['ions'],
            epsilon=self.equation_params['ions']['epsilon_unit'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            gradB1=self._gradB1,
            curl_unit_b2=self._curl_unit_b2,
            **ions_params['options']['push_Bstar']))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_fv')
        self.add_scalar('en_fB')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._en_fv = np.empty(1, dtype=float)
        self._en_fB = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # particles' kinetic energy
        self._en_fv[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 3]**2) / (2.*self.pointer['ions'].n_mks)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv, op=self._mpi_sum)

        # particles' magnetic energy
        self.pointer['ions'].save_magnetic_energy(self._abs_b)

        self._en_fB[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 8]) / self.pointer['ions'].n_mks
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB, op=self._mpi_sum)

        self.update_scalar('en_fv', self._en_fv[0])
        self.update_scalar('en_fB', self._en_fB[0])
        self.update_scalar('en_tot', self._en_fv[0] + self._en_fB[0])


class ShearAlfven(StruphyModel):
    r'''Taking only the ShearAlfven propagator from Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,.

    Implemented equations:

    .. math::

        n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

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
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'u2': 'Hdiv'}
        return dct

    @classmethod
    def bulk_species(cls):
        return 'mhd'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ShearAlfvén

        dct = {}
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'shear_alfven'],
                       option=ShearAlfvén.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        alfven_solver = params['fluid']['mhd']['options']['solvers']['shear_alfven']

        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.ShearAlfvén(
            self.pointer['mhd_u2'],
            self.pointer['b2'],
            **alfven_solver))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_B')
        self.add_scalar('en_B_eq')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()
        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_u2'], out=self._tmp_u1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u2'] .dot(self._tmp_u1)/2
        en_B = self.pointer['b2'] .dot(self._tmp_b1)/2

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_tot', en_U + en_B)

        # background fields
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2

        self.update_scalar('en_B_eq', en_B0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b2']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)


class VariationalBurgers(StruphyModel):
    r'''Burgers's equation discretized with a variational method.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t \mathbf u \cdot \mathbf v \, \textnormal d^3 \mathbf x - \frac{1}{3} \int_{\Omega} \mathbf u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x = 0 ~ ,

    where :

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'uv': 'H1vec'}
        return dct

    @classmethod
    def bulk_species(cls):
        return 'fluid'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import VariationalVelocityAdvection
        dct = {}

        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalVelocityAdvection.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.VariationalVelocityAdvection(
            self.pointer['fluid_uv']))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['v'].zeros()

    def update_scalar_quantities(self):
        mu1 = self._mass_ops.Mv.dot(self.pointer['fluid_uv'], out=self._tmp_u1)

        en_U = self.pointer['fluid_uv'] .dot(mu1)/2
        self.update_scalar('en_U', en_U)


class VariationalPressurelessFluid(StruphyModel):
    r'''Pressure-less fluid equations discretized with a variational method.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t (\rho \mathbf u) \cdot \mathbf v \, \textnormal d^3 \mathbf x 
        - \int_{\Omega} \rho \mathbf u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x 
        + \int_{\Omega} \frac{| \mathbf u |^2}{2} \nabla \cdot (\rho \mathbf v) \, \textnormal d^3 \mathbf x = 0 ~ ,

        \partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 ~ ,

    where :

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 'uv': 'H1vec'}
        return dct

    @classmethod
    def bulk_species(cls):
        return 'fluid'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import VariationalMomentumAdvection, VariationalDensityEvolve
        dct = {}

        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalMomentumAdvection.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalDensityEvolve.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        from struphy.feec.mass import WeightedMassOperator

        # initialize base class
        super().__init__(params, comm)
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(self.derham.Vh_fem['v'], self.derham.Vh_fem['v'], matrix_free=True)        
        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.VariationalDensityEvolve(
            self.pointer['fluid_rho3'], self.pointer['fluid_uv'], model='pressureless', mass_ops=self.WMM))
        self.add_propagator(self.prop_fields.VariationalMomentumAdvection(
            self.pointer['fluid_uv'], mass_ops=self.WMM))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['v'].zeros()

    def update_scalar_quantities(self):
        WMM = self.WMM
        m1 = WMM.dot(self.pointer['fluid_uv'], out=self._tmp_u1)

        en_U = self.pointer['fluid_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)


class VariationalBarotropicFluid(StruphyModel):
    r'''Barotropic fluid equations discretized with a variational method.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t (\rho \mathbf u) \cdot \mathbf v \, \textnormal d^3 \mathbf x 
        - \int_{\Omega}\rho \mathbf u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x 
        + \int_{\Omega} \big( \frac{| \mathbf u |^2}{2} - \frac{\partial \rho e}{\partial \rho} \big) \nabla \cdot (\rho \mathbf v) \, \textnormal d^3 \mathbf x = 0 ~ ,

        \partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 ~ ,

    where

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    and

    .. math::
        e = \frac{\rho}{2} ~ .

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 'uv': 'H1vec'}
        return dct

    @classmethod
    def bulk_species(cls):
        return 'fluid'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import VariationalMomentumAdvection, VariationalDensityEvolve
        dct = {}

        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalMomentumAdvection.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalDensityEvolve.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        from struphy.feec.mass import WeightedMassOperator

        # initialize base class
        super().__init__(params, comm)
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'], self.derham.Vh_fem['v'], matrix_free=True)
        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.VariationalDensityEvolve(
            self.pointer['fluid_rho3'], self.pointer['fluid_uv'], model='barotropic', mass_ops=self.WMM))
        self.add_propagator(self.prop_fields.VariationalMomentumAdvection(
            self.pointer['fluid_uv'], mass_ops=self.WMM))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh['v'].zeros()
        self._tmp_rho1 = self.derham.Vh['3'].zeros()

    def update_scalar_quantities(self):
        WMM = self.WMM
        m1 = WMM.dot(self.pointer['fluid_uv'], out=self._tmp_m1)

        en_U = self.pointer['fluid_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        rho1 = self.mass_ops.M3.dot(
            self.pointer['fluid_rho3'], out=self._tmp_rho1)
        en_thermo = self.pointer['fluid_rho3'] .dot(rho1)/2
        self.update_scalar('en_thermo', en_thermo)

        en_tot = en_U + en_thermo
        self.update_scalar('en_tot', en_tot)


class VariationalCompressibleFluid(StruphyModel):
    r'''Fully compressible fluid equations discretized with a variational method.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t (\rho \mathbf u) \cdot \mathbf v \, \textnormal d^3 \mathbf x 
        - \int_{\Omega}\rho \mathbf u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x 
        + \int_{\Omega} \big( \frac{| \mathbf u |^2}{2} - \frac{\partial \rho e}{\partial \rho} \big) \nabla \cdot (\rho \mathbf v) \, \textnormal d^3 \mathbf x
        + \int_{\Omega} \big( - \frac{\partial \rho e}{\partial s} \big) \nabla \cdot (s \mathbf v) \, \textnormal d^3 \mathbf x = 0 ~ ,

        \partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 ~ ,

        \partial_t s + \nabla \cdot ( s \mathbf u ) = 0 ~ ,

    where

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    and

    .. math::
        e = \rho^{\gamma-1} \exp(s / \rho) ~ .

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''
    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
        return dct

    @classmethod
    def bulk_species(cls):
        return 'fluid'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import VariationalMomentumAdvection, VariationalDensityEvolve, VariationalEntropyEvolve
        dct = {}

        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalMomentumAdvection.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalDensityEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'fluid'], key=['solvers'],
                       option=VariationalEntropyEvolve.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator

        # initialize base class
        super().__init__(params, comm)
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'], self.derham.Vh_fem['v'], matrix_free=True)

        # Initialize propagators/integrators used in splitting substeps
        gamma = params['fluid']['fluid']['options']['solvers']['gamma']

        self.add_propagator(self.prop_fields.VariationalDensityEvolve(
            self.pointer['fluid_rho3'], self.pointer['fluid_uv'], model='full', s=self.pointer['fluid_s3'], gamma=gamma, mass_ops=self.WMM))
        self.add_propagator(self.prop_fields.VariationalMomentumAdvection(
            self.pointer['fluid_uv'], mass_ops=self.WMM))
        self.add_propagator(self.prop_fields.VariationalEntropyEvolve(
            self.pointer['fluid_s3'], self.pointer['fluid_uv'], model='full', rho=self.pointer['fluid_rho3'], gamma=gamma, mass_ops=self.WMM))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh['v'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f)

    def update_scalar_quantities(self):

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['fluid_uv'], out=self._tmp_m1)

        en_U = self.pointer['fluid_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo
        self.update_scalar('en_tot', en_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.
        
        :meta private:
        '''
        en_prop = self._propagators[2]
        en_prop.sf.vector = self.pointer['fluid_s3']
        en_prop.rhof.vector = self.pointer['fluid_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_V3_spans, en_prop.integration_grid_V3_bd, out=en_prop._sf_values_V3)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_V3_spans, en_prop.integration_grid_V3_bd, out=en_prop._rhof_values_V3)
        e = en_prop._ener
        ener_values = en_prop._proj_ener_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_ds)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo


class Poisson(StruphyModel):
    r'''Poisson's equations .

    .. math::   - \Delta \Phi = \rho 

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_Poisson.yml`.

    comm : mpi4py.MPI.Intracomm

    '''

    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['phi'] = 'H1'
        return dct

    @classmethod
    def bulk_species(cls):
        return None

    @classmethod
    def velocity_scale(cls):
        return None

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ImplicitDiffusion

        dct = {}
        cls.add_option(species=['em_fields'], key=['solvers', 'poisson'],
                       option=ImplicitDiffusion.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        phi_n = self.derham.Vh['0'].zeros()
        x0 = self.derham.Vh['0'].zeros()

        # extract necessary parameters
        solver_params = params['em_fields']['options']['solvers']['poisson']

        # Initialize propagator
        self.add_propagator(self.prop_fields.ImplicitDiffusion(
            self.pointer['phi'],
            A_mat='M1',
            sigma=0.,
            phi_n=phi_n,
            x0=x0,
            **solver_params))

        # assert dt=1 for implicit diffusion to solve Poisson.
        assert params['time'][
            'dt'] == 1., f"Time step must be 1.0 in the Poisson model, but is {params['time']['dt']}"

        # Scalar variables to be saved during simulation
        self.add_scalar('en_E')

    def update_scalar_quantities(self):
        pass

    # make dt=1 in parameter file
    @classmethod
    def generate_default_parameter_file(cls, file=None, save=True, prompt=True):
        ''':meta private:'''

        params = super(Poisson, cls).generate_default_parameter_file(
            file=file, save=False, prompt=False)
        params['time']['dt'] = 1.0

        Poisson.write_parameters_to_file(
            parameters=params, file=file, save=save, prompt=prompt)

        return params
