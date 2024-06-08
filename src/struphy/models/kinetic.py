import numpy as np
from struphy.models.base import StruphyModel
from struphy.kinetic_background.base import KineticBackground


class VlasovAmpereOneSpecies(StruphyModel):
    r'''Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = \frac{(A m_\textnormal{H})\hat v^2}{(Z e) \hat x} \,, \qquad  \hat \phi = \hat E \hat x \,.

    Implemented equations: find :math:`(\mathbf E, f) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x = 
            \kappa^2 \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl})\,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \mathbf{E}
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \kappa = \hat \Omega_\textnormal{p}\hat t\,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. 
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \kappa^2 \left(\frac{Z_0}{Z}\int_\Omega \psi\, n_0\,\textrm d \mathbf x + \int_\Omega \int_{\mathbb{R}^3} \psi\, f(t=0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \right) \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,,
            \end{align}

    where :math:`Z_0 \in \mathbb Z` and :math:`n_0:\Omega \to \mathbb R^+` denote the charge number and the number density 
    of the neutralizing background, respectively, such that

    .. math::

        \frac{Z_0}{Z} n_0 = - \int_{\mathbb{R}^3} f_0 \, \text{d}^3 \mathbf{v} < 0\,,

    where :math:`f_0` is the kinetic background distribution (static). 
    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0\,.

    Notes
    -----

    * The Poisson equation is solved with the :ref:`control_var`.

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved: 
    Find :math:`(\mathbf E, f) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x = 
            \kappa^2 \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \mathbf{E} \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,.
        \end{align}


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
        dct['kinetic']['species1'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'species1'

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ImplicitDiffusion
        from struphy.propagators.propagators_markers import PushEta
        from struphy.propagators.propagators_coupling import VlasovAmpere

        dct = {}
        cls.add_option(species='em_fields', key=['solvers', 'poisson'],
                       option=ImplicitDiffusion.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key=['algos', 'push_eta'],
                       option=PushEta.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='coupling_solver',
                       option=VlasovAmpere.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='verification',
                       option={'use': False, 'kappa': 1.}, dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='Z0',
                       option=-1., dct=dct)

        return dct

    def __init__(self, params, comm=None):

        super().__init__(params, comm=comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # get rank and size
        if self.comm is None:
            self._rank = 0
        else:
            self._rank = self.comm.Get_rank()

        # get species paramaters
        spec_params = params['kinetic']['species1']

        # Get coupling strength
        if spec_params['options']['verification']['use']:
            self.kappa = spec_params['options']['verification']['kappa']
            print(
                f'\n!!! Verification run: equation parameters set to {self.kappa = }.')
        else:
            self.kappa = self.equation_params['species1']['kappa']

        # Check if it is control-variate method
        self._control_variate = (
            spec_params['markers']['type'] == 'control_variate')

        # set background density factor
        Z0 = spec_params['options']['Z0']
        Z = spec_params['phys_params']['Z']
        assert Z0 * \
            Z < 0, f'Neutralizing background has wrong polarity {Z0 = } to {Z = }.'

        # multiply background to get quasi neutrality
        self.pointer['species1']._f0 = - Z0/Z * self.pointer['species1'].f0

        # check mean velocity
        # TODO: assert f0.params[] == 0.

        # propagator parameters
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['species1']['options']['algos']['push_eta']
        params_coupling = params['kinetic']['species1']['options']['coupling_solver']

        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['species1'],
            algo=algo_eta,
            bc_type=spec_params['markers']['bc']['type']))

        self.add_propagator(self.prop_coupling.VlasovAmpere(
            self.pointer['e1'],
            self.pointer['species1'],
            c1=self.kappa**2,
            **params_coupling))

        # Scalar variables to be saved during the simulation
        self.add_scalar('en_E')
        self.add_scalar('en_f')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._tmp1 = self.derham.Vh['1'].zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        '''Solve initial Poisson equation.

        :meta private:
        '''

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # initialize fields and particles
        super().initialize_from_params()

        if self._rank == 0:
            print('\nINITIAL POISSON SOLVE:')

        # use control variate method
        if self._control_variate:
            self.pointer['species1'].update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [np.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "charge_density_0form")
        charge_accum.accumulate(self.pointer['species1'])

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        # Instantiate Poisson solver
        _phi = self.derham.Vh['0'].zeros()
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.,
            sigma_2=0.,
            sigma_3=1.,
            rho=self.kappa**2 * charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        if self._rank == 0:
            print('\nSolving initial Poisson problem...')
        poisson_solver(1.)

        self.derham.grad.dot(-_phi, out=self.pointer['e1'])
        if self._rank == 0:
            print('Done.')

    def update_scalar_quantities(self):

        # e*M1*e/2
        self.mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        en_E = self.pointer['e1'].dot(self._tmp1) / 2.
        self.update_scalar('en_E', en_E)

        # kappa^2 / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self.kappa**2 / (2 * self.pointer['species1'].n_mks) * \
            np.dot(self.pointer['species1'].markers_wo_holes[:, 3]**2 +
                   self.pointer['species1'].markers_wo_holes[:, 4]**2 +
                   self.pointer['species1'].markers_wo_holes[:, 5]**2,
                   self.pointer['species1'].markers_wo_holes[:, 6])
        if self.comm is not None:
            self.comm.Allreduce(
                self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_f', self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar('en_tot', en_E + self._tmp[0])


class VlasovMaxwellOneSpecies(StruphyModel):
    r'''Vlasov-Maxwell equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    Implemented equations: find :math:`(\mathbf E, \mathbf B, f) \in H(\textnormal{curl}) \times H(\textnormal{div}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \times \mathbf{F} \cdot \mathbf B\,\textrm d \mathbf x = 
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl})\,,
            \\[2mm]
            &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\Big[ \mathbf{E} + \mathbf{v} \times \mathbf{B} \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \end{align}

    with the normalization parameters

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. 
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \left( \frac{Z_0}{Z}\int_\Omega \psi\, n_0\,\textrm d \mathbf x + \int_\Omega \int_{\mathbb{R}^3} \psi\, f(t=0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \right) \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,,
            \end{align}

    where :math:`Z_0 \in \mathbb Z` and :math:`n_0:\Omega \to \mathbb R^+` denote the charge number and the number density 
    of the neutralizing background, respectively, such that

    .. math::

        \frac{Z_0}{Z} n_0 = - \int_{\mathbb{R}^3} f_0 \, \text{d}^3 \mathbf{v} < 0\,,

    where :math:`f_0` is the kinetic background distribution (static). 
    Moreover, it is assumed that

    .. math::

        \nabla \times \mathbf B_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}\,,

    where :math:`\mathbf B_0` is the static equilibirum magnetic field.

    Notes
    -----

    * The Poisson equation is solved with the :ref:`control_var`.

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved: 
    Find :math:`(\mathbf E, \tilde{\mathbf B}, f) \in H(\textnormal{curl}) \times H(\textnormal{div}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \times \mathbf{F} \cdot \tilde{\mathbf B}\,\textrm d \mathbf x = 
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\Big[ \mathbf{E} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf B}) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \end{align}

    where :math:`\tilde{\mathbf B} = \mathbf B - \mathbf B_0` denotes the magnetic perturbation.


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
        dct['kinetic']['species1'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'species1'

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import Maxwell, ImplicitDiffusion
        from struphy.propagators.propagators_markers import PushEta, PushVxB
        from struphy.propagators.propagators_coupling import VlasovAmpere

        dct = {}
        cls.add_option(species='em_fields', key=['solvers', 'maxwell'],
                       option=Maxwell.options()['solver'], dct=dct)
        cls.add_option(species='em_fields', key=['solvers', 'poisson'],
                       option=ImplicitDiffusion.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key=['algos', 'push_eta'],
                       option=PushEta.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key=['algos', 'push_vxb'],
                       option=PushVxB.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='coupling_solver',
                       option=VlasovAmpere.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='verification',
                       option=False, dct=dct)
        cls.add_option(species=['kinetic', 'species1'], key='Z0',
                       option=-1., dct=dct)

        return dct

    def __init__(self, params, comm=None):

        super().__init__(params, comm=comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # get rank and size
        if self.comm is None:
            self._rank = 0
        else:
            self._rank = self.comm.Get_rank()

        # get species paramaters
        spec_params = params['kinetic']['species1']

        # equation parameters
        if spec_params['options']['verification']:
            self._alpha = 1.
            self._epsilon = -1.
            print(
                f'\n!!! Verification run: equation parameters set to {self._alpha = } and {self._epsilon = }.')
        else:
            self._alpha = self.equation_params['species1']['alpha']
            self._epsilon = self.equation_params['species1']['epsilon']

        # set background density and mean velocity factors
        Z0 = spec_params['options']['Z0']
        Z = spec_params['phys_params']['Z']
        assert Z0 * \
            Z < 0, f'Neutralizing background has wrong polarity {Z0 = } to {Z = }.'

        self.pointer['species1'].f0.moment_factors['n'] = - Z0/Z
        self.pointer['species1'].f0.moment_factors['u'] = [
            self._epsilon/self._alpha**2]*3

        # Initialize background magnetic field from MHD equilibrium
        b_backgr = self.derham.P['2']([self.mhd_equil.b2_1,
                                       self.mhd_equil.b2_2,
                                       self.mhd_equil.b2_3])

        # propagator parameters
        params_maxwell = params['em_fields']['options']['solvers']['maxwell']
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['species1']['options']['algos']['push_eta']
        algo_vxb = params['kinetic']['species1']['options']['algos']['push_vxb']
        params_coupling = params['kinetic']['species1']['options']['coupling_solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params_maxwell))

        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['species1'],
            algo=algo_eta,
            bc_type=spec_params['markers']['bc']['type']))

        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['species1'],
            algo=algo_vxb,
            scale_fac=1/self._epsilon,
            b_eq=b_backgr,
            b_tilde=self.pointer['b2']))

        self.add_propagator(self.prop_coupling.VlasovAmpere(
            self.pointer['e1'],
            self.pointer['species1'],
            c1=self._alpha**2/self._epsilon,
            c2=1/self._epsilon,
            **params_coupling))

        # Scalar variables to be saved during the simulation
        self.add_scalar('en_E')
        self.add_scalar('en_B')
        self.add_scalar('en_f')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._tmp1 = self.derham.Vh['1'].zeros()
        self._tmp2 = self.derham.Vh['2'].zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        ''':meta private:'''

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # initialize fields and particles
        super().initialize_from_params()

        if self._rank == 0:
            print('\nINITIAL POISSON SOLVE:')

        # use control variate method
        self.pointer['species1'].update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [np.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "charge_density_0form")
        charge_accum.accumulate(self.pointer['species1'])

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        # Instantiate Poisson solver
        _phi = self.derham.Vh['0'].zeros()
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.,
            sigma_2=0.,
            sigma_3=1.,
            rho=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        if self._rank == 0:
            print('\nSolving initial Poisson problem...')
        poisson_solver(1.)

        self.derham.grad.dot(-_phi, out=self.pointer['e1'])
        if self._rank == 0:
            print('Done.')

    def update_scalar_quantities(self):

        # e*M1*e and b*M2*b
        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = self.pointer['e1'].dot(self._tmp1) / 2.
        en_B = self.pointer['b2'].dot(self._tmp2) / 2.
        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self._alpha**2 / (2 * self.pointer['species1'].n_mks) * \
            np.dot(self.pointer['species1'].markers_wo_holes[:, 3]**2 +
                   self.pointer['species1'].markers_wo_holes[:, 4] ** 2 +
                   self.pointer['species1'].markers_wo_holes[:, 5]**2,
                   self.pointer['species1'].markers_wo_holes[:, 6])
        if self.comm is not None:
            self.comm.Allreduce(
                self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_f', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', en_E + en_B + self._tmp[0])


class LinearVlasovAmpereOneSpecies(StruphyModel):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = \frac{(A m_\textnormal{H})\hat v^2}{(Z e) \hat x} \,, \qquad  \hat \phi = \hat E \hat x \,.

    Implemented equations: find :math:`(\mathbf{E}, f_1) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x = 
            \kappa^2 \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} f_1 \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl})\,,
            \\[2mm]
            &\frac{\partial f_1}{\partial t} + \mathbf{v} \cdot \, \nabla f_1
            + \mathbf{E}_0 \cdot \frac{\partial f_1}{\partial \mathbf{v}}
            = \frac{\kappa}{v_{\text{th}}^2} \, \mathbf{E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \kappa = \hat \Omega_\textnormal{p}\hat t\,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right)

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{\kappa}{v_{\text{th}}^2} \mathbf{E}_0

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \kappa^2 \left(\frac{Z_0}{Z}\int_\Omega \psi\, n_0\,\textrm d \mathbf x + \int_\Omega \int_{\mathbb{R}^3} \psi\, f(t=0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \right) \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,,
            \end{align}

    where :math:`Z_0 \in \mathbb Z` and :math:`n_0:\Omega \to \mathbb R^+` denote the charge number and the number density 
    of the neutralizing background, respectively, such that

    .. math::

        \frac{Z_0}{Z} n_0 = - \int_{\mathbb{R}^3} f_i \, \text{d}^3 \mathbf{v} < 0\,,

    where :math:`f_i` is the kinetic background distribution (static). 
    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_i \, \text{d}^3 \mathbf{v} = 0\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    """

    @classmethod
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['kinetic']['species1'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'species1'

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ImplicitDiffusion
        from struphy.propagators.propagators_markers import PushEta
        from struphy.propagators.propagators_coupling import EfieldWeights
        dct = {}
        cls.add_option(['em_fields'], ['solvers', 'poisson'],
                       ImplicitDiffusion.options()['solver'], dct)
        cls.add_option(['kinetic', 'species1'], ['algos', 'push_eta'],
                       PushEta.options()['algo'], dct)
        cls.add_option(['kinetic', 'species1'], ['coupling_solver'],
                       EfieldWeights.options()['solver'], dct)
        cls.add_option(species=['kinetic', 'species1'], key='verification',
                       option={'use': False, 'kappa': 1., 'alpha': 1.}, dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        self._species_params = params['kinetic']['species1']

        # Assert Maxwellian background
        assert self._species_params['background']['type'] == 'Maxwellian3D', \
            "The background distribution function must be a uniform Maxwellian!"

        # Assert uniformity of the Maxwellian background
        self._f0 = self.pointer['species1'].f0
        assert self._f0.maxw_params['u1'] == 0., "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params['u2'] == 0., "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params['u3'] == 0., "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params['vth1'] == self._f0.maxw_params['vth2'] == self._f0.maxw_params['vth3'], \
            "The background Maxwellian must be isotropic in velocity space!"
        self.vth = self._f0.maxw_params['vth1']

        # get species paramaters
        spec_params = params['kinetic']['species1']

        # Get coupling strength
        if spec_params['options']['verification']['use']:
            self.kappa = spec_params['options']['verification']['kappa']
            self.alpha = spec_params['options']['verification']['alpha']
            if self._rank == 0:
                print(
                    f"\n!!! Verification run: equation parameters set to {self.kappa = }, {self.alpha = }.\n")
        else:
            self.kappa = self.equation_params['species1']['kappa']
            self.alpha = self.equation_params['species1']['alpha']

        # ====================================================================================
        # Create pointers to background electric potential and field
        self._phi_background = self.derham.Vh['0'].zeros()
        self._e_background = self.derham.grad.dot(self._phi_background)
        # ====================================================================================

        # propagator params
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['species1']['options']['algos']['push_eta']
        params_coupling = params['kinetic']['species1']['options']['coupling_solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['species1'],
            algo=algo_eta,
            bc_type=self._species_params['markers']['bc']['type']))
        if self._rank == 0:
            print("Added Step PushEta\n")

        # Only add StepVinEfield if e-field is non-zero, otherwise it is more expensive
        if not np.all(self._e_background[0]._data < 1e-14) or not np.all(self._e_background[1]._data < 1e-14) or not np.all(self._e_background[2]._data < 1e-14):
            self.add_propagator(self.prop_markers.StepVinEfield(
                self.pointer['species1'],
                e_field=self._e_background,
                kappa=self.kappa))
            if self._rank == 0:
                print("Added Step VinEfield\n")

        self.add_propagator(self.prop_coupling.EfieldWeights(
            self.pointer['e_field'],
            self.pointer['species1'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params_coupling
        ))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        # Scalar variables to be saved during the simulation
        self.add_scalar('en_e')
        self.add_scalar('en_w')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._en_e_tmp = self.pointer['e_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        ''':meta private:'''
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain,
            "H1", "charge_density_0form"
        )
        charge_accum.accumulate(self.pointer['species1'])

        # Instantiate Poisson solver
        _phi = self.derham.Vh['0'].zeros()
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.,
            sigma_2=0.,
            sigma_3=1.,
            rho=self.alpha**2 * self.kappa * charge_accum.vectors[0],
            **self._poisson_params
        )

        # Solve with dt=1. and compute electric field
        if self._rank == 0:
            print('\nSolving initial Poisson problem...')
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e_field'])
        if self._rank == 0:
            print('Done.')

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._en_e_tmp)
        en_E = self.pointer['e_field'].dot(self._en_e_tmp) / 2.
        self.update_scalar('en_e', en_E)

        # evaluate f0
        f0_values = self._f0(
            self.pointer['species1'].markers_wo_holes[:, 0],
            self.pointer['species1'].markers_wo_holes[:, 1],
            self.pointer['species1'].markers_wo_holes[:, 2],
            self.pointer['species1'].markers_wo_holes[:, 3],
            self.pointer['species1'].markers_wo_holes[:, 4],
            self.pointer['species1'].markers_wo_holes[:, 5],
        )

        # alpha^2 * v_th^2 / (2*N) * sum_p s_0 * w_p^2 / f_{0,p}
        self._tmp[0] = \
            self.alpha**2 * self.vth**2 / (2 * self.pointer['species1'].n_mks) * \
            np.dot(
                self.pointer['species1'].markers_wo_holes[:, 6]**2,  # w_p^2
                self.pointer['species1'].markers_wo_holes[:, 7] / \
                f0_values  # s_{0,p} / f_{0,p}
        )

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum
        )

        self.update_scalar('en_w', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', self._tmp[0] + en_E)


class LinearVlasovMaxwell(StruphyModel):
    r'''Linearized Vlasov Maxwell equations with a Maxwellian background distribution function :math:`f_0`.

    :ref:`normalization`:

    .. math::

        \begin{align}
            c & = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B} = \hat v = \hat u \,, \qquad  \hat h = \frac{\hat n}{\hat v^3} \,.
        \end{align}

    Implemented equations:

    .. math::

        \begin{align}
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \frac{1}{\varepsilon}\left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \frac{1}{\varepsilon} \sqrt{f_0} \, \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{\hat \omega}{2\pi \, \Omega_\textnormal{c}} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    At initial time the Poisson equation is solved once to satisfy the Gauss law

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \text{d}^3 \mathbf{v}
        \end{align}

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
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['em_fields']['b_field'] = 'Hdiv'
        dct['kinetic']['electrons'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'electrons'

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import Maxwell, ImplicitDiffusion
        from struphy.propagators.propagators_markers import PushEta, PushVxB
        from struphy.propagators.propagators_coupling import EfieldWeightsImplicit
        dct = {}
        cls.add_option(['em_fields'], ['solvers', 'maxwell'],
                       Maxwell.options()['solver'], dct)
        cls.add_option(['em_fields'], ['solvers', 'poisson'],
                       ImplicitDiffusion.options()['solver'], dct)
        cls.add_option(['kinetic', 'electrons'], ['algos', 'push_eta'],
                       PushEta.options()['algo'], dct)
        cls.add_option(['kinetic', 'electrons'], ['algos', 'push_vxb'],
                       PushVxB.options()['algo'], dct)
        cls.add_option(['kinetic', 'electrons'], ['solver'],
                       EfieldWeightsImplicit.options()['solver'], dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        self._electron_params = params['kinetic']['electrons']

        # kinetic background
        assert self._electron_params['background']['type'] == 'Maxwellian3D', \
            AssertionError(
                "The background distribution function must be a uniform Maxwellian!")

        self._maxwellian_params = self._electron_params['background']['Maxwellian3D']
        self.pointer['electrons']._f0 = getattr(
            kin_ana, 'Maxwellian3D')(maxw_params=self._maxwellian_params)
        self._f0 = self.pointer['electrons'].f0

        assert self._f0.maxw_params['u1'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['u2'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['u3'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['vth1'] == self._f0.maxw_params['vth2'] == self._f0.maxw_params['vth3'], \
            "Background Maxwellian must be isotropic in velocity space!"

        # Get coupling strength
        self.alpha = self.equation_params['electrons']['alpha']
        self.kappa = 1. / self.equation_params['electrons']['epsilon']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # TODO: must be set from model options
        self._phi_background = self.derham.Vh['0'].zeros()
        self._e_background = self.derham.grad.dot(self._phi_background)
        # ====================================================================================

        # propagator params
        params_maxwell = params['em_fields']['options']['solvers']['maxwell']
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['electrons']['options']['algos']['push_eta']
        algo_vxb = params['kinetic']['electrons']['options']['algos']['push_vxb']
        params_coupling = params['kinetic']['electrons']['options']['solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=algo_eta,
            bc_type=self._electron_params['markers']['bc']['type']))
        if self._rank == 0:
            print("Added Step PushEta\n")

        # Only add StepVinEfield if e-field is non-zero, otherwise it is more expensive
        if np.all(self._e_background[0]._data < 1e-14) and np.all(self._e_background[1]._data < 1e-14) and np.all(self._e_background[2]._data < 1e-14):
            self.add_propagator(self.prop_markers.StepVinEfield(
                self.pointer['electrons'],
                e_field=self._e_background,
                kappa=self.kappa))
            if self._rank == 0:
                print("Added Step VinEfield\n")

        # Only add VxB Step if b-field is non-zero, otherwise it is more expensive
        e1 = np.linspace(0, 1, 20)
        e2 = np.linspace(0, 1, 20)
        e3 = np.linspace(0, 1, 20)
        b_bckgr_strength = np.max(self.mhd_equil.absB0(e1, e2, e3))
        if b_bckgr_strength > 1e-6:
            self.add_propagator(self.prop_markers.PushVxB(
                self.pointer['electrons'],
                algo=algo_vxb,
                scale_fac=1.,
                b_eq=self._b_background,
                b_tilde=None))
            if self._rank == 0:
                print("Added Step VxB\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsImplicit(
            self.pointer['e_field'],
            self.pointer['electrons'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            model='linear_vlasov_maxwell',
            **params_coupling))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e_field'],
            self.pointer['b_field'],
            **params_maxwell))
        if self._rank == 0:
            print("\nAdded Step Maxwell\n")

        # Scalar variables to be saved during the simulation
        self.add_scalar('en_e')
        self.add_scalar('en_b')
        self.add_scalar('en_w')
        # self.add_scalar('en_e1')
        # self.add_scalar('en_e2')
        # self.add_scalar('en_e3')
        # self.add_scalar('en_b1')
        # self.add_scalar('en_b2')
        # self.add_scalar('en_b3')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._en_e_tmp = self.pointer['e_field'].space.zeros()
        self._en_b_tmp = self.pointer['b_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        ''':meta private:'''
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
        from struphy.pic.base import Particles
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        # evaluate f0
        f0_values = self._f0(self.pointer['electrons'].markers[:, 0],
                             self.pointer['electrons'].markers[:, 1],
                             self.pointer['electrons'].markers[:, 2],
                             self.pointer['electrons'].markers[:, 3],
                             self.pointer['electrons'].markers[:, 4],
                             self.pointer['electrons'].markers[:, 5])

        self.pointer['electrons']._f0 = self._f0

        # edges = self.kinetic['electrons']['bin_edges']['e1']
        # components = [False] * 6
        # components[0] = True

        # self.pointer['electrons'].show_distribution_function(components, edges, self.domain)

        # overwrite binning function to always bin marker data for f_1, not h
        def new_binning(self, components, bin_edges):
            """
            Overwrite the binning method of the parent class to correctly bin data from f_1
            and not from f_1/sqrt(f_0).

            Parameters & Info
            -----------------
            see struphy.pic.particles.base.Particles.binning
            """
            f0_values = self._f0(self.markers[:, 0],
                                 self.markers[:, 1],
                                 self.markers[:, 2],
                                 self.markers[:, 3],
                                 self.markers[:, 4],
                                 self.markers[:, 5])[~self.holes]
            self.markers[~self.holes, 6] *= np.sqrt(f0_values)
            res, res_df = Particles.binning(
                self, components, bin_edges)
            self.markers[~self.holes, 6] /= np.sqrt(f0_values)
            return res, res_df

        func_type = type(self.pointer['electrons'].binning)

        self.pointer['electrons'].binning = func_type(
            new_binning, self.pointer['electrons'])

        # Correct initialization of weights by dividing by sqrt(f_0)
        self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 6] /= \
            np.sqrt(f0_values[~self.pointer['electrons'].holes])

        # edges = self.kinetic['electrons']['bin_edges']['e1']
        # components = [False] * 6
        # components[0] = True

        # self.pointer['electrons'].show_distribution_function(components, edges, self.domain)

        # Accumulate charge density
        charge_accum = AccumulatorVector(self.derham,
                                         self.domain,
                                         "H1",
                                         "linear_vlasov_maxwell_poisson")
        charge_accum.accumulate(self.pointer['electrons'], f0_values,
                                np.array(
                                    list(self._maxwellian_params.values())),
                                self.alpha, self.kappa)

        # Locally subtract mean charge for solvability with periodic bc
        if np.all(charge_accum.vectors[0].space.periods):
            charge_accum._vectors[0][:] -= np.mean(charge_accum.vectors[0].toarray()[
                                                   charge_accum.vectors[0].toarray() != 0])

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh['0'])
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.,
            rho=charge_accum.vectors[0],
            x0=charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e_field'])

        self.pointer['e_field'].blocks[1]._data *= 0.
        self.pointer['e_field'].blocks[2]._data *= 0.

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._en_e_tmp)
        en_E = self.pointer['e_field'].dot(self._en_e_tmp) / 2.
        self.update_scalar('en_e', en_E)

        # # 0.5 * |e_1|^2
        # self.update_scalar('en_e1', self.pointer['e_field']._blocks[0].dot(
        #     self._en_e_tmp._blocks[0]) / 2.)

        # # 0.5 * |e_2|^2
        # self.update_scalar('en_e2', self.pointer['e_field']._blocks[1].dot(
        #     self._en_e_tmp._blocks[1]) / 2.)

        # # 0.5 * |e_3|^2
        # self.update_scalar('en_e3', self.pointer['e_field']._blocks[2].dot(
        #     self._en_e_tmp._blocks[2]) / 2.)

        # 0.5 * b^T * M_2 * b
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._en_b_tmp)
        en_B = self.pointer['b_field'].dot(self._en_b_tmp) / 2.
        self.update_scalar('en_b', en_B)

        # # 0.5 * |b_1|^2
        # self.update_scalar('en_b1', self.pointer['b_field']._blocks[0].dot(
        #     self._en_b_tmp._blocks[0]) / 2.)

        # # 0.5 * |b_2|^2
        # self.update_scalar('en_b2', self.pointer['b_field']._blocks[1].dot(
        #     self._en_b_tmp._blocks[1]) / 2.)

        # # 0.5 * |b_3|^2
        # self.update_scalar('en_b3', self.pointer['b_field']._blocks[2].dot(
        #     self._en_b_tmp._blocks[2]) / 2.)

        # alpha^2 / (2N) * (v_th_1 * v_th_2 * v_th_3)^(2/3) * sum_p s_0 * w_p^2
        self._tmp[0] = \
            self.alpha**2 / (2 * self.pointer['electrons'].n_mks) * \
            (self._f0.maxw_params['vth1'] *
             self._f0.maxw_params['vth2'] *
             self._f0.maxw_params['vth3'])**(2/3) * \
            np.dot(self.pointer['electrons'].markers_wo_holes[:, 6]**2,  # w_p^2
                   self.pointer['electrons'].markers_wo_holes[:, 7])  # s_{0,p}

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_w', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', self._tmp[0] + en_E + en_B)


class DeltaFVlasovAmpereOneSpecies(StruphyModel):
    r'''Vlasov Maxwell with Maxwellian background and delta-f method.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = \frac{(A m_\textnormal{H})\hat v^2}{(Z e) \hat x} \,, \qquad  \hat \phi = \hat E \hat x \,.

    Implemented equations: find :math:`(\mathbf{E_1}, f_1) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            &\partial_t f_1 + \mathbf{v} \cdot \, \nabla_\mathbf{x} f_1 + \kappa \left( \mathbf{E}_0 + \mathbf{E_1} + \mathbf{v} \times (\mathbf{B}_0 + \mathbf{B_1}) \right)
            \cdot \nabla_\mathbf{v} f_1 = \frac{\kappa}{v_\text{th}^2} \mathbf{E_1} \cdot \mathbf{v} f_0 \,,
            \\[2mm]
            &\frac{\partial \mathbf{E_1}}{\partial t} = \nabla \times \mathbf{B_1} -
            \alpha^2 \kappa \int_{\mathbb{R}^3} f_1 \mathbf{v} \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B_1}}{\partial t} = - \nabla \times \mathbf{E_1} \,,
        \end{align}

    where

    where :math:`Z_0 \in \mathbb Z` and :math:`n_0:\Omega \to \mathbb R^+` denote the charge number and the number density 
    of the neutralizing background, respectively, such that

    .. math::

        \frac{Z_0}{Z} n_0 = - \int_{\mathbb{R}^3} f_i \, \text{d}^3 \mathbf{v} < 0\,,

    where :math:`f_i` is the kinetic background distribution (static). 
    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_i \, \text{d}^3 \mathbf{v} = 0\,.

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

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['kinetic']['species1'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'species1'

    @classmethod
    def velocity_scale(cls):
        return 'light'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ImplicitDiffusion
        from struphy.propagators.propagators_markers import PushEta
        from struphy.propagators.propagators_coupling import EfieldWeights

        dct = {}
        cls.add_option(['em_fields'], ['solvers', 'poisson'],
                       ImplicitDiffusion.options()['solver'], dct)
        cls.add_option(['kinetic', 'species1'], ['algos', 'push_eta'],
                       PushEta.options()['algo'], dct)
        cls.add_option(['kinetic', 'species1'], ['coupling_solver'],
                       EfieldWeights.options()['solver'], dct)
        cls.add_option(species=['kinetic', 'species1'], key='verification',
                       option={'use': False, 'kappa': 1., 'alpha': 1.}, dct=dct)

        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        self._species_params = params['kinetic']['species1']

        # Assert Maxwellian background
        assert self._species_params['background']['type'] == 'Maxwellian3D', \
            "The background distribution function must be a uniform Maxwellian!"

        # Assert uniformity of the Maxwellian background
        self._f0 = self.pointer['species1'].f0
        assert self._f0.maxw_params['u1'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['u2'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['u3'] == 0., "No shifts in velocity space possible!"
        assert self._f0.maxw_params['vth1'] == self._f0.maxw_params['vth2'] == self._f0.maxw_params['vth3'], \
            "Background Maxwellian must be isotropic in velocity space!"
        self.vth = self._f0.maxw_params['vth1']

        # get species paramaters
        spec_params = params['kinetic']['species1']

        # Get coupling strength
        if spec_params['options']['verification']['use']:
            self.kappa = spec_params['options']['verification']['kappa']
            self.alpha = spec_params['options']['verification']['alpha']
            if self._rank == 0:
                print(
                    f"\n!!! Verification run: equation parameters set to {self.kappa = }, {self.alpha = }.\n")
        else:
            self.kappa = self.equation_params['species1']['kappa']
            self.alpha = self.equation_params['species1']['alpha']

        # ====================================================================================
        # Create pointers to background electric potential and field
        self._phi_background = self.derham.Vh['0'].zeros()
        self._e_background = self.derham.grad.dot(self._phi_background)
        # ====================================================================================

        # propagator params
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['species1']['options']['algos']['push_eta']
        params_coupling = params['kinetic']['species1']['options']['coupling_solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['species1'],
            algo=algo_eta,
            bc_type=self._species_params['markers']['bc']['type']))
        if self._rank == 0:
            print("Added Step PushEta\n")

        self.add_propagator(self.prop_markers.StepVinEfield(
            self.pointer['species1'],
            e_field=self._e_background + self.pointer['e_field'],
            kappa=self.kappa))
        if self._rank == 0:
            print("Added Step VinEfield\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsAnalytic(
            self.pointer['e_field'],
            self.pointer['species1'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params_coupling))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights Explicit\n")

        # self.add_propagator(self.prop_coupling.EfieldWeightsDiscreteGradient(
        #     self.pointer['e_field'],
        #     self.pointer['species1'],
        #     alpha=self.alpha,
        #     kappa=self.kappa,
        #     f0=self._f0,
        #     **params['solvers']['solver_ew']))
        # if self._rank == 0:
        #     print("\nAdded Step EfieldWeights Discrete Gradient\n")

        # self.add_propagator(self.prop_coupling.EfieldWeightsImplicit(
        #     self.pointer['e_field'],
        #     self.pointer['species1'],
        #     alpha=self.alpha,
        #     kappa=self.kappa,
        #     f0=self._f0,
        #     model='delta_f_vlasov_maxwell',
        #     **params_implicit))
        # if self._rank == 0:
        #     print("\nAdded Step EfieldWeights Semi-Crank-Nicolson\n")

        # Scalar variables to be saved during simulation
        self.add_scalar('en_e')
        self.add_scalar('en_w')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._en_e_tmp = self.pointer['e_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        ''':meta private:'''
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
        from struphy.pic.particles import Particles
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        # evaluate f0
        f0_values = self._f0(
            *self.pointer['species1'].markers_wo_holes[:, :6].T
        )

        ln_f0_values = np.log(f0_values)

        # overwrite binning function to always bin marker data for f_1, not h
        def new_binning(self, components, bin_edges):
            """
            Overwrite the binning method of the parent class to correctly bin data from f_1
            and not from f_0 - (f_0 - f_1) ln(f_0).

            Parameters & Info
            -----------------
            see struphy.pic.particles.base.Particles.binning
            """

            # values of f0 and their logarithm
            f0_values = self._f0(
                *self.markers_wo_holes[:, :6].T
            )
            ln_f0_values = np.log(f0_values)

            # convert weights from h to f1
            # w_p^f = f_{0,p} / s_{0,p} * (1 / ln(f_{0,p}) - 1) - w_p^h / ln(f_{0,p})
            self.markers[~self.holes, 6] -= f0_values * \
                (1 - ln_f0_values) / self.markers[~self.holes, 7]
            self.markers[~self.holes, 6] /= (-1) * ln_f0_values

            # do the particle binning
            res, res_df = Particles.binning(
                self, components, bin_edges
            )

            # convert weights from f1 to h
            # w_p^h = f_0 * (1 - ln(f_0)) / s_0 - w_p^f * ln(f_0)
            self.markers[~self.holes, 6] *= (-1) * ln_f0_values
            self.markers[~self.holes, 6] += f0_values * \
                (1 - ln_f0_values) / self.markers[~self.holes, 7]

            return res, res_df

        func_type = type(self.pointer['species1'].binning)

        self.pointer['species1'].binning = func_type(
            new_binning, self.pointer['species1'])

        # Accumulate charge density before converting f1 to h
        charge_accum = AccumulatorVector(
            self.derham, self.domain,
            "H1", "charge_density_0form"
        )
        charge_accum.accumulate(self.pointer['species1'])

        # Instantiate Poisson solver
        _phi = self.derham.Vh['0'].zeros()
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.,
            sigma_2=0.,
            sigma_3=1.,
            rho=self.alpha**2 * self.kappa * charge_accum.vectors[0],
            **self._poisson_params
        )

        # Solve with dt=1. and compute electric field
        if self._rank == 0:
            print('\nSolving initial Poisson problem...')
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e_field'])
        if self._rank == 0:
            print('Done.')

        # Correct initialization of weights fomr f1 to h
        # w_p^h = f_0 * (1 - ln(f_0)) / s_0 - w_p^f * ln(f_0)
        self.pointer['species1'].markers[~self.pointer['species1'].holes, 6] *= \
            (-1) * ln_f0_values
        self.pointer['species1'].markers[~self.pointer['species1'].holes, 6] += \
            f0_values * (1 - ln_f0_values) / \
            self.pointer['species1'].markers_wo_holes[:, 7]

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._en_e_tmp)
        en_E = self.pointer['e_field'].dot(self._en_e_tmp) / 2.
        self.update_scalar('en_e', en_E)

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * sum_p w_p
        self._tmp[0] = \
            self.alpha**2 * self.vth**2 * \
            np.sum(self.pointer['species1'].markers_wo_holes[:, 6]) / \
            self.pointer['species1'].n_mks

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum
        )

        self.update_scalar('en_w', self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar('en_tot', self._tmp[0] + en_E)


# class VlasovMasslessElectrons(StruphyModel):
#     r'''Hybrid (kinetic ions + massless electrons) equations with quasi-neutrality condition.
#     Unknowns: distribution function for ions, and vector potential.

#     Normalization:

#     .. math::
#             t, x, p, A, f...


#     Implemented equations:

#     Hyrid model with kinetic ions and massless electrons.

#     .. math::

#         \begin{align}
#         \textnormal{Vlasov}\qquad& \frac{\partial f}{\partial t} + (\mathbf{p} - \mathbf{A}) \cdot \frac{\partial f}{\partial \mathbf{x}}
#         - \left[ T_e \frac{\nabla n}{n} - \left( \frac{\partial{\mathbf A}}{\partial {\mathbf x}} \right)^\top ({\mathbf A} - {\mathbf p} )  \right] \cdot \frac{\partial f}{\partial \mathbf{p}}
#         = 0\,,
#         \\
#         \textnormal{Faraday's law}\qquad& \frac{\partial {\mathbf A}}{\partial t} = - \frac{\nabla \times \nabla \times A}{n} \times \nabla \times {\mathbf A} - \frac{\int ({\mathbf A} - {\mathbf p}f \mathrm{d}{\mathbf p})}{n} \times \nabla \times {\mathbf A}, \quad n = \int f \mathrm{d}{\mathbf p}.
#         \end{align}

#     Parameters
#     ----------
#         params : dict
#             Simulation parameters, see from :ref:`params_yml`.
#     '''

#     @classmethod
#     def bulk_species(cls):
#         return 'ions'

#     @classmethod
#     def velocity_scale(cls):
#         return 'cyclotron'

#     def __init__(self, params, comm):

#         from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
#         from struphy.propagators.base import Propagator
#         from struphy.propagators import propagators_fields, propagators_markers
#         from mpi4py.MPI import SUM, IN_PLACE
#         from struphy.pic.accumulation.particles_to_grid import Accumulator

#         super().__init__(params, comm, a1='Hcurl', ions='Particles6D')

#         # pointers to em-field variables
#         self._a = self.em_fields['a1']['obj'].vector

#         # pointer to kinetic variables
#         self._ions = self.kinetic['ions']['obj']
#         ions_params = self.kinetic['ions']['params']

#         # extract necessary parameters
#         # shape function info, degree and support size
#         shape_params = params['kinetic']['ions']['ionsshape']
#         # electron temperature
#         self.thermal = params['kinetic']['electrons']['temperature']

#         # extract necessary parameters
#         solver_params_1 = params['solvers']['solver_1']

#         # Project background magnetic field
#         self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
#                                          self.mhd_equil.b2_2,
#                                          self.mhd_equil.b2_3])

#         # set propagators base class attributes
#         Propagator.derham = self.derham
#         Propagator.domain = self.domain
#         Propagator.mass_ops = self.mass_ops

#         self._accum_density = Accumulator(self.derham,
#                                           self.domain,
#                                           'H1',
#                                           'hybrid_fA_density',
#                                           add_vector=False)

#         # Initialize propagators/integrators used in splitting substeps
#         self._propagators = []

#         self._propagators += [propagators_markers.StepHybridXPSymplectic(
#                               self._ions,
#                               a=self._a,
#                               particle_bc=ions_params['markers']['bc']['type'],
#                               quad_number=params['grid']['nq_el'],
#                               shape_degree=np.array(shape_params['degree']),
#                               shape_size=np.array(shape_params['size']),
#                               electron_temperature=self.thermal,
#                               accumulate_density=self._accum_density)]

#         self._propagators += [propagators_markers.StepPushpxBHybrid(
#                               self._ions,
#                               method=ions_params['push_algos']['pxb'],
#                               a=self._a,
#                               b_eq=self._b_eq)]

#         self._propagators += [propagators_fields.FaradayExtended(
#                               self._a,
#                               a_space='Hcurl',
#                               beq=self._b_eq,
#                               particles=self._ions,
#                               quad_number=params['grid']['nq_el'],
#                               shape_degree=np.array(shape_params['degree']),
#                               shape_size=np.array(shape_params['size']),
#                               solver_params=solver_params_1,
#                               accumulate_density=self._accum_density)]

#         # Scalar variables to be saved during simulation
#         self._scalar_quantities = {}
#         self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
#         self._en_f_loc = np.empty(1, dtype=float)
#         self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
#         self._en_thermal_loc = np.empty(1, dtype=float)
#         self._scalar_quantities['en_thermal'] = np.empty(1, dtype=float)
#         self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

#         # MPI operations needed for scalar variables
#         self._mpi_sum = SUM

#     @property
#     def propagators(self):
#         return self._propagators

#     @property
#     def scalar_quantities(self):
#         return self._scalar_quantities

#     def update_scalar_quantities(self):
#         import struphy.pic.utilities as pic_util

#         rank = self._derham.comm.Get_rank()

#         self._curla = self._derham.curl.dot(self._a)

#         self._scalar_quantities['en_B'][0] = (self._curla + self._b_eq).dot(
#             self._mass_ops.M2.dot(self._curla + self._b_eq))/2

#         self._en_f_loc = pic_util.get_kinetic_energy_particles(
#             self._a, self._derham, self._domain, self._ions)/self._ions.n_mks

#         self.derham.comm.Reduce(
#             self._en_f_loc, self._scalar_quantities['en_f'], op=self._mpi_sum, root=0)

#         self._en_thermal_loc = pic_util.get_electron_thermal_energy(self._accum_density, self._derham, self._domain, int(self._derham.domain_array[int(rank), 2]), int(
#             self._derham.domain_array[int(rank), 5]), int(self._derham.domain_array[int(rank), 8]), int(self._derham.nquads[0]+1), int(self._derham.nquads[1]+1), int(self._derham.nquads[2]+1))

#         self.derham.comm.Reduce(self.thermal*self._en_thermal_loc,
#                                 self._scalar_quantities['en_thermal'], op=self._mpi_sum, root=0)

#         self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_B'][0]
#         self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
#         self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_thermal'][0]


class DriftKineticElectrostaticAdiabatic(StruphyModel):
    r'''Drift-kinetic equation for one ion species in static background magnetic field,
    coupled to quasi-neutrality equation with adiabatic electrons. 

    :ref:`normalization`:

    .. math::

       \hat v = \hat v_\textrm{i} = \sqrt{\frac{k_B \hat T_\textrm{i}}{m_\textrm{i}}}\,,\qquad  \hat E = \hat v_\textrm{i}\hat B\,,\qquad \hat \phi = \hat E \hat x \,.

    Implemented equations:

    .. math::

        &\frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\varepsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.
        \\[2mm]
        &\int \frac{n_0}{|B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi\,\textrm d \mathbf x + \frac{1}{\varepsilon} \int n_0 \psi \left(1 + \frac{1}{Z \varepsilon} \frac{1}{T_{0}} \phi \right) \,\textrm d \mathbf x  = \frac 1 \varepsilon \int \int \psi \, f B^*_\parallel \,\textrm d \mathbf x\,\textnormal d v_\parallel \textnormal d \mu \qquad \forall \ \psi \in H^1\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and 

    .. math::
        \mathbf{E}^* = - \nabla \phi - \varepsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,,

    and with the normalization parameters

    .. math::

        \varepsilon := \frac{\hat v_\textrm{i}}{\hat \Omega_\textrm{i} \hat x}\,,\qquad \hat \Omega_\textrm{i} = \frac{Ze \hat B}{m_\textrm{i}} \,.

    Notes
    -----

    * The :ref:`control_var` in the Poisson equation is optional; in case it is enabled via the parameter file, the following Poisson equation is solved: 
    Find :math:`\phi \in H^1` such that

    .. math::

        \int \frac{n_0}{|B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi\,\textrm d \mathbf x + \frac{1}{Z\varepsilon^2} \int  \frac{n_0}{T_{0}} \psi \phi \,\textrm d \mathbf x  = \frac 1 \varepsilon \int \int \psi \, (f - f_0) B^*_\parallel \,\textrm d \mathbf x\,\textnormal d v_\parallel \textnormal d \mu \qquad \forall \ \psi \in H^1\,.

    * The polarization density can be turned off by choosing ``AdiabaticPhi`` as the potential solver; in case it is enabled via the parameter file, the potential is determined from

    .. math::

        \frac{1}{Z\varepsilon} \int  \frac{n_0}{T_{0}} \psi \phi \,\textrm d \mathbf x  = \int \int \psi \, (f - f_0) B^*_\parallel \,\textrm d \mathbf x\,\textnormal d v_\parallel \textnormal d \mu \qquad \forall \ \psi \in H^1\,.


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

        dct['em_fields']['phi'] = 'H1'
        dct['kinetic']['ions'] = 'Particles5D'

        return dct

    @classmethod
    def bulk_species(cls):
        return 'ions'

    @classmethod
    def velocity_scale(cls):
        return 'thermal'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ImplicitDiffusion, AdiabaticPhi
        from struphy.propagators.propagators_markers import PushDriftKineticBxEstar, PushDriftKineticParallel

        dct = {}

        cls.add_option(species=['kinetic', 'ions'], key='push_bxEstarWithPhi',
                       option=PushDriftKineticBxEstar.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'ions'], key='push_BstarWithPhi',
                       option=PushDriftKineticParallel.options()['algo'], dct=dct)
        cls.add_option(species=['em_fields'], key=['phi'],
                       option=['ImplicitDiffusion'], dct=dct)
        cls.add_option(species=['em_fields'], key=['ImplicitDiffusion', 'solver'],
                       option=ImplicitDiffusion.options()['solver'], dct=dct)
        cls.add_option(species=['em_fields'], key=['AdiabaticPhi', 'solver'],
                       option=AdiabaticPhi.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'ions'], key='verification',
                       option={'use': False, 'epsilon': 1.}, dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE
        from struphy.feec.projectors import L2Projector
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # prelim
        ions_params = self.kinetic['ions']['params']
        phi_method = params['em_fields']['options']['phi']
        solver_params = params['em_fields']['options'][phi_method]['solver']
        spec_params = params['kinetic']['ions']

        Z = spec_params['phys_params']['Z']
        assert Z > 0  # must be ions

        # magnetic background
        if 'braginskii_equilibrium' in params:
            magn_bckgr = self.braginskii_equil
            self.mass_ops.selected_weight = 'eq_braginskii'
        else:
            magn_bckgr = self.mhd_equil

        # Poisson right-hand side
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "gc_density_0form")

        rho = (charge_accum, self.pointer['ions'])

        if 'full_f' in ions_params['markers']['type']:
            print(f'{phi_method = }')
            try:
                assert phi_method == 'ImplicitDiffusion'
            except:
                exit(
                    f'full_f requires phi_method to be "ImplicitDiffusion", but it is "{phi_method}". Exiting ...')
            l2_proj = L2Projector('H1', self.mass_ops)
            f0e = Z * self.pointer['ions'].f0
            assert isinstance(f0e, KineticBackground)
            rho_eh = l2_proj.get_dofs(f0e.n)
            rho = [rho]
            rho += [rho_eh]

        # Get coupling strength
        if spec_params['options']['verification']['use']:
            self.epsilon = spec_params['options']['verification']['epsilon']
            print(
                f'\n!!! Verification run: equation parameters set to {self.epsilon = }.')
        else:
            self.epsilon = self.equation_params['ions']['epsilon']

        # Initialize propagators/integrators used in splitting substeps
        if phi_method == 'ImplicitDiffusion':
            self.add_propagator(self.prop_fields.ImplicitDiffusion(
                self.pointer['phi'],
                sigma_1=1. / self.epsilon**2 / Z,  # set to zero for Landau damping test
                sigma_2=0.,
                sigma_3=1. / self.epsilon,
                stab_mat='M0ad',
                diffusion_mat='M1gyro',
                rho=rho,
                **solver_params
            ))
        elif phi_method == 'AdiabaticPhi':
            self.add_propagator(self.prop_fields.AdiabaticPhi(
                self.pointer['phi'],
                A_mat='M0ad',
                rho=rho,
                **solver_params
            ))
        else:
            raise ValueError(f'{phi_method = } not allowed.')

        self.add_propagator(self.prop_markers.PushDriftKineticBxEstar(
            self.pointer['ions'],
            phi0=self.pointer['phi'],
            magn_bckgr=magn_bckgr,
            epsilon=self.equation_params['ions']['epsilon'],
            Z=Z,
            **ions_params['options']['push_bxEstarWithPhi']))

        self.add_propagator(self.prop_markers.PushDriftKineticParallel(
            self.pointer['ions'],
            phi0=self.pointer['phi'],
            magn_bckgr=magn_bckgr,
            epsilon=self.epsilon,
            Z=Z,
            **ions_params['options']['push_BstarWithPhi']))

        self._phi_method = phi_method

        self.add_scalar('en_phi')
        self.add_scalar('en_particles')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE
        self._tmp1 = self.derham.Vh['1'].zeros()
        self._tmp2 = self.derham.Vh['0'].zeros()
        self._tmp3 = np.empty(1, dtype=float)
        self._e_field = self.derham.Vh['1'].zeros()

    def update_scalar_quantities(self):

        # energy from polarization
        if self._phi_method == 'ImplicitDiffusion':
            e1 = self.derham.grad.dot(-self.pointer['phi'], out=self._e_field)
            M1_e1 = self.mass_ops.M1gyro.dot(e1, out=self._tmp1)
            en_phi1 = e1.dot(M1_e1) / 2.
        else:
            en_phi1 = 0.

        # energy from adiabatic electrons
        self.mass_ops.M0ad.dot(self.pointer['phi'], out=self._tmp2)
        en_phi0 = self.pointer['phi'].dot(
            self._tmp2) / (2. * self.epsilon**2)

        # for Landau damping test
        # en_phi0 = 0.

        # mu_p * |B0(eta_p)|
        self.pointer['ions'].save_magnetic_background_energy()

        # 1/N sum_p (w_p v_p^2/2 + mu_p |B0|_p)
        self._tmp3[0] = 1/self.pointer['ions'].n_mks * np.sum(
            self.pointer['ions'].weights * self.pointer['ions'].velocities[:, 0]**2 / 2. + self.pointer['ions'].markers[~self.pointer['ions'].holes, 8])

        if self.comm is not None:
            self.comm.Allreduce(
                self._mpi_in_place, self._tmp3, op=self._mpi_sum)

        self.update_scalar('en_phi', en_phi0 + en_phi1)
        self.update_scalar('en_particles', self._tmp3[0])
        self.update_scalar('en_tot', en_phi0 + en_phi1 + self._tmp3[0])
