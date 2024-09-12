import numpy as np

from struphy.models.base import StruphyModel
from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers


class Maxwell(StruphyModel):
    r'''Maxwell's equations in vacuum.

    :ref:`normalization`:

    .. math::

        \hat E = c \hat B\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0\,,

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['em_fields']['b_field'] = 'Hdiv'
        return dct

    @staticmethod
    def bulk_species():
        return None

    @staticmethod
    def velocity_scale():
        return 'light'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.Maxwell: ['e_field', 'b_field']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        # extract necessary parameters
        solver = params['em_fields']['options']['Maxwell']['solver']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {'solver': solver}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('electric energy')
        self.add_scalar('magnetic energy')
        self.add_scalar('total energy')

        # temporary vectors for scalar quantities
        self._tmp_e = self.derham.Vh['1'].zeros()
        self._tmp_b = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._tmp_e)
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._tmp_b)

        en_E = self.pointer['e_field'].dot(self._tmp_e)/2
        en_B = self.pointer['b_field'].dot(self._tmp_b)/2

        self.update_scalar('electric energy', en_E)
        self.update_scalar('magnetic energy', en_B)
        self.update_scalar('total energy', en_E + en_B)


class Vlasov(StruphyModel):
    r'''Vlasov equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat v = \hat \Omega_\textnormal{c} \hat x\,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left(\mathbf{v}\times\mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushVxB`
    2. :class:`~struphy.propagators.propagators_markers.PushEta`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['ions'] = 'Particles6D'
        return dct

    @staticmethod
    def bulk_species():
        return 'ions'

    @staticmethod
    def velocity_scale():
        return 'cyclotron'

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushVxB: ['ions'],
                propagators_markers.PushEta: ['ions']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['ions']['params']

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushVxB] = {'algo': ions_params['options']['PushVxB']['algo'],
                                                     'scale_fac': 1.,
                                                     'b_eq': self._b_eq,
                                                     'b_tilde': None}

        self._kwargs[propagators_markers.PushEta] = {'algo': ions_params['options']['PushEta']['algo'],
                                                     'bc_type': ions_params['markers']['bc']['type']}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_f',compute = 'from_particles', species = 'ions')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):

        self._tmp[0] = self.pointer['ions'].markers_wo_holes[:, 6].dot(
            self.pointer['ions'].markers_wo_holes[:, 3]**2 +
            self.pointer['ions'].markers_wo_holes[:, 4]**2 +
            self.pointer['ions'].markers_wo_holes[:, 5]**2) / (2*self.pointer['ions'].n_mks)

        # self.derham.comm.Allreduce(
        #     self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_f', self._tmp[0])


class GuidingCenter(StruphyModel):
    r'''Guiding-center equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat v = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\epsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::

        \mathbf{E}^* = -\epsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Moreover,

    .. math::

        \epsilon = \frac{1 }{ \hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['ions'] = 'Particles5D'
        return dct

    @staticmethod
    def bulk_species():
        return 'ions'

    @staticmethod
    def velocity_scale():
        return 'alfvén'

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushGuidingCenterBxEstar: ['ions'],
                propagators_markers.PushGuidingCenterParallel: ['ions']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        # initialize base class
        super().__init__(params, comm = comm, inter_comm = inter_comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['ions']['params']
        epsilon = self.equation_params['ions']['epsilon']

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushGuidingCenterBxEstar] = {'epsilon': epsilon,
                                                                      'algo': ions_params['options']['PushGuidingCenterBxEstar']['algo']}

        self._kwargs[propagators_markers.PushGuidingCenterParallel] = {'epsilon': epsilon,
                                                                       'algo': ions_params['options']['PushGuidingCenterParallel']['algo']}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_fv')
        self.add_scalar('en_fB')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._en_fv = np.empty(1, dtype=float)
        self._en_fB = np.empty(1, dtype=float)
        self._en_tot = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # particles' kinetic energy
        self._en_fv[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 3]**2) / (2.*self.pointer['ions'].n_mks)
        

        self._en_tot[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 8]) / self.pointer['ions'].n_mks
        

        self._en_fB[0] = self._en_tot[0] - self._en_fv[0]
        

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv, op=self._mpi_sum)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_tot, op=self._mpi_sum)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB, op=self._mpi_sum)

        self.update_scalar('en_fv', self._en_fv[0])
        self.update_scalar('en_fB', self._en_fB[0])
        self.update_scalar('en_tot', self._en_tot[0])

        self._n_lost_particles[0] = self.pointer['ions'].n_lost_markers
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._n_lost_particles, op=self._mpi_sum)


class ShearAlfven(StruphyModel):
    r'''ShearAlfven propagator from :class:`~struphy.models.fluid.LinearMHD` with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'u2': 'Hdiv'}
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.ShearAlfven: ['mhd_u2', 'b2']}

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
        alfven_solver = params['fluid']['mhd']['options']['ShearAlfven']['solver']

        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.ShearAlfven] = {'u_space': 'Hdiv',
                                                        'solver': alfven_solver}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        # self.add_scalar('en_U')
        # self.add_scalar('en_B')
        # self.add_scalar('en_B_eq')
        # self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        self.add_scalar('en_U', compute = 'from_field')
        self.add_scalar('en_B', compute = 'from_field')
        self.add_scalar('en_B_eq', compute = 'from_field')
        self.add_scalar('en_B_tot', compute = 'from_field')
        self.add_scalar('en_tot2',summands=['en_U','en_B','en_B_eq'])

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


class VariationalPressurelessFluid(StruphyModel):
    r'''Pressure-less fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`

    :ref:`Model info <add_model>`:
    '''
    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 'uv': 'H1vec'}
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
                propagators_fields.VariationalMomentumAdvection: ['fluid_uv']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        

        from struphy.feec.mass import WeightedMassOperator

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

        gamma = params['fluid']['fluid']['options']['VariationalDensityEvolve']['physics']['gamma']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {'model': 'pressureless',
                                                                     'gamma': gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_density,
                                                                     'nonlin_solver': nonlin_solver_density}

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {'mass_ops': self.WMM,
                                                                         'lin_solver': lin_solver_momentum,
                                                                         'nonlin_solver': nonlin_solver_momentum}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

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

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \qquad \hat{\mathcal U} = \frac{\hat \rho}{2} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho))}{\partial \rho} = 0 \,.

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho/2`.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`

    :ref:`Model info <add_model>`:
    '''
    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}
        dct['fluid']['fluid'] = {'rho3': 'L2', 'uv': 'H1vec'}
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
                propagators_fields.VariationalMomentumAdvection: ['fluid_uv']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        from struphy.feec.mass import WeightedMassOperator

        # initialize base class
        super().__init__(params, comm, inter_comm = inter_comm)

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

        gamma = params['fluid']['fluid']['options']['VariationalDensityEvolve']['physics']['gamma']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.VariationalDensityEvolve] = {'model': 'barotropic',
                                                                     'gamma': gamma,
                                                                     'mass_ops': self.WMM,
                                                                     'lin_solver': lin_solver_density,
                                                                     'nonlin_solver': nonlin_solver_density}

        self._kwargs[propagators_fields.VariationalMomentumAdvection] = {'mass_ops': self.WMM,
                                                                         'lin_solver': lin_solver_momentum,
                                                                         'nonlin_solver': nonlin_solver_momentum}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

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

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = K\,,\qquad \hat s = \hat \rho C_v \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`

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
                propagators_fields.VariationalEntropyEvolve: ['fluid_s3', 'fluid_uv']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator

        # initialize base class
        super().__init__(params, comm, inter_comm = inter_comm)

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

        self._gamma = params['fluid']['fluid']['options']['VariationalDensityEvolve']['physics']['gamma']
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

        # Initialize propagators used in splitting substeps
        self.init_propagators()

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
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_ds)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        return np.power(rho, self._gamma)*np.exp(s/rho)


class Poisson(StruphyModel):
    r'''Weak discretization of Poisson's equation with diffusion matrix, stabilization 
    and time-depedent right-hand side.

    :ref:`normalization`:

    .. math::

        \hat D = \frac{\hat n}{\hat x^2}\,,\qquad \hat \rho = \hat n \,.

    :ref:`Equations <gempic>`: Find :math:`\phi \in H^1` such that

    .. math::

        - \nabla \cdot D_0(\mathbf x) \nabla \phi + n_0(\mathbf x) \phi =  \rho(t, \mathbf x)\,,

    where :math:`n_0, \rho(t):\Omega \to \mathbb R` are real-valued functions, :math:`\rho(t)` parametrized with time :math:`t`,
    and :math:`D_0:\Omega \to \mathbb R^{3\times 3}` is a positive matrix. 
    Boundary terms from integration by parts are assumed to vanish.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.TimeDependentSource`
    2. :class:`~struphy.propagators.propagators_fields.ImplicitDiffusion`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['phi'] = 'H1'
        dct['em_fields']['source'] = 'H1'
        return dct

    @staticmethod
    def bulk_species():
        return None

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {propagators_fields.TimeDependentSource: ['source'],
                propagators_fields.ImplicitDiffusion: ['phi']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        super().__init__(params, comm, inter_comm = inter_comm)

        # extract necessary parameters
        model_params = params['em_fields']['options']['ImplicitDiffusion']['model']
        solver_params = params['em_fields']['options']['ImplicitDiffusion']['solver']
        omega = params['em_fields']['options']['TimeDependentSource']['omega']
        hfun = params['em_fields']['options']['TimeDependentSource']['hfun']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.TimeDependentSource] = {'omega': omega,
                                                                'hfun': hfun}

        self._kwargs[propagators_fields.ImplicitDiffusion] = {'sigma_1': model_params['sigma_1'],
                                                              'stab_mat': model_params['stab_mat'],
                                                              'diffusion_mat': model_params['diffusion_mat'],
                                                              'rho': self.pointer['source'],
                                                              'solver': solver_params}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

    def update_scalar_quantities(self):
        pass


class DeterministicParticleDiffusion(StruphyModel):
    r'''Diffusion equation discretized with a deterministic particle method; 
    the solution is :math:`L^2`-projected onto :math:`V^0 \subset H^1` to compute the flux.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} +  \nabla \cdot\left(\mathbf F(u) u\right) = 0\,, \qquad \mathbf F(u) = -\mathbb D\,\frac{\nabla u}{u}\,, 

    where :math:`\mathbb D: \Omega\to \mathbb R^{3\times 3 }` is a positive diffusion matrix. 
    At the moment only matrices of the form :math:`D*Id` are implemented, where :math:`D > 0` is a positive diffusion coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushDeterministicDiffusion`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['species1'] = 'Particles3D'
        return dct

    @staticmethod
    def bulk_species():
        return 'species1'

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushDeterministicDiffusion: ['species1']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        super().__init__(params, comm, inter_comm = inter_comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        params = self.kinetic['species1']['params']
        algo = params['options']['PushDeterministicDiffusion']['algo']
        diffusion_coefficient = params['options']['PushDeterministicDiffusion']['diffusion_coefficient']

        # # project magnetic background
        # self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
        #                                  self.mhd_equil.b2_2,
        #                                  self.mhd_equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushDeterministicDiffusion] = {'algo': algo,
                                                                        'bc_type': params['markers']['bc']['type'],
                                                                        'diffusion_coefficient': diffusion_coefficient}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_f')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        pass


class RandomParticleDiffusion(StruphyModel):
    r'''Diffusion equation discretized with a (random) particle method;
    the diffusion is computed through a Wiener process.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} -  D \, \Delta u = 0\,,

    where :math:`D > 0` is a positive diffusion coefficient. 

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushRandomDiffusion`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['kinetic']['species1'] = 'Particles3D'
        return dct

    @staticmethod
    def bulk_species():
        return 'species1'

    @staticmethod
    def velocity_scale():
        return None

    @staticmethod
    def propagators_dct():
        return {propagators_markers.PushRandomDiffusion: ['species1']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm, inter_comm = None):

        super().__init__(params, comm, inter_comm = inter_comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        species1_params = self.kinetic['species1']['params']
        algo = species1_params['options']['PushRandomDiffusion']['algo']
        diffusion_coefficient = species1_params['options']['PushRandomDiffusion']['diffusion_coefficient']

        # # project magnetic background
        # self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
        #                                  self.mhd_equil.b2_2,
        #                                  self.mhd_equil.b2_3])

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushRandomDiffusion] = {'algo': algo,
                                                                 'bc_type': species1_params['markers']['bc']['type'],
                                                                 'diffusion_coefficient': diffusion_coefficient}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar('en_f')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        pass
