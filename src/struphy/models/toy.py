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

        self._E0T = self.derham.E['0'].transpose()
        self._EvT = self.derham.E['v'].transpose()

        ee = 1.602176634e-19  # elementary charge (C)
        mH = 1.67262192369e-27  # proton mass (kg)

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

        self._en_fv_loc = self._ions.markers[~self._ions.holes, 5].dot(
            self._ions.markers[~self._ions.holes, 3]**2) / (2*self._ions.n_mks)
        self.derham.comm.Reduce(
            self._en_fv_loc, self._scalar_quantities['en_fv'], op=self._mpi_sum, root=0)

        # calculate particle magnetic energy
        self._ions.save_magnetic_energy(self._derham, self._E0T.dot(
            self.derham.P['0'](self.mhd_equil.absB0)))

        self._en_fB_loc = self._ions.markers[~self._ions.holes, 5].dot(
            self._ions.markers[~self._ions.holes, 8]) / self._ions.n_mks
        self.derham.comm.Reduce(
            self._en_fB_loc, self._scalar_quantities['en_fB'], op=self._mpi_sum, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_fv'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fB'][0]
