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
    def bulk_species(cls):
        return None

    @classmethod
    def timescale(cls):
        return 'light'

    def __init__(self, params, comm):

        super().__init__(params, comm, e1='Hcurl', b2='Hdiv')

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

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
    def bulk_species(cls):
        return 'ions'

    @classmethod
    def timescale(cls):
        return 'cyclotron'

    def __init__(self, params, comm):

        super().__init__(params, comm, ions='Particles6D')

        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['ions']['params']
        print(
            f'Total number of markers : {self.pointer["ions"].n_mks}, shape of markers array on rank {self.derham.comm.Get_rank()} : {self.pointer["ions"].markers.shape}')

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['ions'],
            algo=ions_params['push_algos']['vxb'],
            scale_fac=1.,
            b_eq=self._b_eq,
            b_tilde=None,
            f0=None))
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['ions'],
            algo=ions_params['push_algos']['eta'],
            bc_type=ions_params['markers']['bc_type'],
            f0=None))

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

        self._E0T = self.derham.E['0'].transpose()
        self._EvT = self.derham.E['v'].transpose()

        kappa = 1. / self.eq_params['ions']['epsilon_unit']
        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.StepPushGuidingCenter1(
            self.pointer['ions'],
            kappa=kappa,
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            integrator=ions_params['push_algos1']['integrator'],
            method=ions_params['push_algos1']['method'],
            maxiter=ions_params['push_algos1']['maxiter'],
            tol=ions_params['push_algos1']['tol']))
        self.add_propagator(self.prop_markers.StepPushGuidingCenter2(
            self.pointer['ions'],
            kappa=kappa,
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            integrator=ions_params['push_algos2']['integrator'],
            method=ions_params['push_algos2']['method'],
            maxiter=ions_params['push_algos2']['maxiter'],
            tol=ions_params['push_algos2']['tol']))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_fv')
        self.add_scalar('en_fB')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._en_fv_loc = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # particles' kinetic energy
        self._en_fv_loc[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 3]**2) / (2.*self.pointer['ions'].n_mks)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv_loc, op=self._mpi_sum)

        # particles' magnetic energy
        self.pointer['ions'].save_magnetic_energy(self._derham,
                                        self._E0T.dot(self.derham.P['0'](self.mhd_equil.absB0)))

        self._en_fB_loc[0] = self.pointer['ions'].markers[~self.pointer['ions'].holes, 5].dot(
            self.pointer['ions'].markers[~self.pointer['ions'].holes, 8]) / self.pointer['ions'].n_mks
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB_loc, op=self._mpi_sum)

        self.update_scalar('en_fv', self._en_fv_loc[0])
        self.update_scalar('en_fB', self._en_fB_loc[0])
        self.update_scalar('en_tot', self._en_fv_loc[0] + self._en_fB_loc[0])
