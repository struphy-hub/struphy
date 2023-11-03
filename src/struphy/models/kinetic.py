'The bulk plasma is kinetic.'


import numpy as np
from struphy.models.base import StruphyModel


class VlasovMaxwell(StruphyModel):
    r'''Vlasov Maxwell equations with Poisson splitting.

    :ref:`normalization`:

    .. math::

        \begin{align}
            c & = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B} = \hat v \,, \qquad  \hat f = \frac{\hat n}{c^3} \,.
        \end{align}

    Implemented equations:

    .. math::

        \begin{align}
            &\frac{\partial \mathbf{E}}{\partial t} - \nabla \times \mathbf{B} = -
            \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \,,
            \\[1mm]
            &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
            \\[1mm]
            &\partial_t f + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\Big[ \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,.
        \end{align}

    where :math:`\mathbf B_0` is an equilibrium magnetic field and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{\hat \omega}{\hat \Omega_\textnormal{c}} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    At initial time the Poisson equation is solved once to weakly satisfy Gauss' law

    .. math::

            \nabla \cdot \mathbf{E} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}\,.

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
        from struphy.propagators.propagators_coupling import VlasovMaxwell

        dct = {}
        cls.add_option(species='em_fields', key=['solvers', 'maxwell'],
                       option=Maxwell.options()['solver'], dct=dct)
        cls.add_option(species='em_fields', key=['solvers', 'poisson'],
                       option=ImplicitDiffusion.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'electrons'], key=['algos', 'push_eta'],
                       option=PushEta.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'electrons'], key=['algos', 'push_vxb'],
                       option=PushVxB.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'electrons'], key='solver',
                       option=VlasovMaxwell.options()['solver'], dct=dct)

        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        electron_params = params['kinetic']['electrons']

        self._marker_type = electron_params['markers']['type']
        assert self._marker_type in ['full_f', 'control_variate']
        if self._marker_type == 'full_f':
            f0 = None
        else:
            f0 = self.pointer['electrons'].f_backgr

        # model parameters
        self._alpha = self.equation_params['electrons']['alpha_unit']
        self._epsilon = self.equation_params['electrons']['epsilon_unit']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # propagator params
        params_maxwell = params['em_fields']['options']['solvers']['maxwell']
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['electrons']['options']['algos']['push_eta']
        algo_vxb = params['kinetic']['electrons']['options']['algos']['push_vxb']
        params_coupling = params['kinetic']['electrons']['options']['solver']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params_maxwell))

        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=algo_eta,
            bc_type=electron_params['markers']['bc']['type']))

        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['electrons'],
            algo=algo_vxb,
            scale_fac=1/self._epsilon,
            b_eq=self._b_background,
            b_tilde=self.pointer['b2']))

        self.add_propagator(self.prop_coupling.VlasovMaxwell(
            self.pointer['e1'],
            self.pointer['electrons'],
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

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
        from struphy.feec.projectors import L2_Projector
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        if self._rank == 0:
            print('\nINITIAL POISSON SOLVE:')

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "vlasov_maxwell_poisson")
        charge_accum.accumulate(self.pointer['electrons'])

        # add contribution from background in control variate method
        if self._marker_type == 'control_variate':
            _proj = L2_Projector(self._mass_ops.M0, space='H1', derham=self.derham)
            _phi_bckgr = _proj(self.pointer['electrons'].f_backgr.n)
            # TODO: what to do with this?

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh['0'])
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma=1e-11,
            phi_n=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            x0=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        if self._rank == 0:
            print('Solving ...')
        poisson_solver(1.)

        self.derham.grad.dot(-_phi, out=self.pointer['e1'])
        if self._rank == 0:
            print('Done.')

    def update_scalar_quantities(self):
        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = self.pointer['e1'].dot(self._tmp1) / 2.
        en_B = self.pointer['b2'].dot(self._tmp2) / 2.
        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self._alpha**2 / (2 * self.pointer['electrons'].n_mks) * \
            np.dot(self.pointer['electrons'].markers_wo_holes[:, 3]**2 +
                   self.pointer['electrons'].markers_wo_holes[:, 4] ** 2 +
                   self.pointer['electrons'].markers_wo_holes[:, 5]**2,
                   self.pointer['electrons'].markers_wo_holes[:, 6])
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_f', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', en_E + en_B + self._tmp[0])


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
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \frac{1}{\epsilon}\left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \frac{1}{\epsilon} \sqrt{f_0} \, \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \frac{\alpha^2}{\epsilon} \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \epsilon = \frac{\hat \omega}{2\pi \, \Omega_\textnormal{c}} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    At initial time the Poisson equation is solved once to satisfy the Gauss law

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \frac{\alpha^2}{\epsilon} \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \text{d}^3 \mathbf{v}
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
        assert self._electron_params['background']['type'] == 'Maxwellian6DUniform', \
            AssertionError(
                "The background distribution function must be a uniform Maxwellian!")

        self._maxwellian_params = self._electron_params['background']['Maxwellian6DUniform']
        self.pointer['electrons']._f_backgr = getattr(
            kin_ana, 'Maxwellian6DUniform')(**self._maxwellian_params)
        self._f0 = self.pointer['electrons'].f_backgr

        assert self._f0.params['u1'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['u2'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['u3'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['vth1'] == self._f0.params['vth2'] == self._f0.params['vth3'], \
            "Background Maxwellian must be isotropic in velocity space!"

        # Get coupling strength
        self.alpha = self.equation_params['electrons']['alpha_unit']
        self.kappa = 1. / self.equation_params['electrons']['epsilon_unit']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # Create pointers to background electric potential and field
        self._phi_background = self.derham.P['0'](self.electric_equil.phi0)
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
        def new_binning(self, components, bin_edges, pforms=['0','0']):
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
            res = Particles.binning(
                self, components, bin_edges, pforms)
            self.markers[~self.holes, 6] /= np.sqrt(f0_values)
            return res

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
            sigma=0.,
            phi_n=charge_accum.vectors[0],
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
            (self._f0.params['vth1'] *
             self._f0.params['vth2'] *
             self._f0.params['vth3'])**(2/3) * \
            np.dot(self.pointer['electrons'].markers_wo_holes[:, 6]**2,  # w_p^2
                   self.pointer['electrons'].markers_wo_holes[:, 7])  # s_{0,p}

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_w', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', self._tmp[0] + en_E + en_B)


class DeltaFVlasovMaxwell(StruphyModel):
    r'''Vlasov Maxwell with Maxwellian background and delta-f method.

    :ref:`normalization`:

    .. math::

        \begin{align}
            c & = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B} = \hat v = \hat u \,, \qquad  \hat h = \frac{\hat n}{\hat v^3} \,.
        \end{align}

    Implemented equations:

    .. math::

        \begin{align}
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \frac{1}{\epsilon}\left( \mathbf{E}_0 + \mathbf{E} + \mathbf{v} \times (\mathbf{B}_0 + \mathbf{B}) \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \frac{1}{\epsilon} \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right)
            \, \left( \frac{f_0 - h}{\ln(f_0)} - f_0 \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \alpha^2 \frac{1}{\epsilon} \int_{\mathbb{R}^3} \left( \frac{f_0 - h}{\ln(f_0)} - f_0 \right) \left( \mathbb{1}_{\text{th}}^2
            (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \epsilon = \frac{\hat \omega}{2\pi \, \Omega_\textnormal{c}} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    Moreover, :math:`f_0` is a Maxwellian background distribution function with constant velocity shift :math:`\mathbf{u}`
    and thermal velocity matrix :math:`\mathbb{1}_{\text{th}} = \text{diag} \left( \frac{1}{v_{\text{th},1}^2}, \frac{1}{v_{\text{th},2}^2}, \frac{1}{v_{\text{th},3}^2} \right)`
    and :math:`h = f_0 - (f_0 + f_1) \, \ln(f_0)`.

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
        from struphy.propagators.propagators_coupling import EfieldWeightsImplicit, EfieldWeightsAnalytic

        dct = {}
        cls.add_option(['em_fields'], ['solvers', 'maxwell'],
                       Maxwell.options()['solver'], dct)
        cls.add_option(['em_fields'], ['solvers', 'poisson'],
                       ImplicitDiffusion.options()['solver'], dct)
        cls.add_option(['kinetic', 'electrons'], ['algos', 'push_eta'],
                       PushEta.options()['algo'], dct)
        cls.add_option(['kinetic', 'electrons'], ['algos', 'push_vxb'],
                       PushVxB.options()['algo'], dct)
        cls.add_option(['kinetic', 'electrons'], ['solvers', 'implicit'],
                       EfieldWeightsImplicit.options()['solver'], dct)
        cls.add_option(['kinetic', 'electrons'], ['solvers', 'analytic'],
                       EfieldWeightsAnalytic.options()['solver'], dct)

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
        assert self._electron_params['background']['type'] == 'Maxwellian6DUniform', \
            AssertionError(
                "The background distribution function must be a uniform Maxwellian!")

        self.pointer['electrons']._f_backgr = getattr(
            kin_ana, 'Maxwellian6DUniform')(**self._electron_params['background']['Maxwellian6DUniform'])
        self._f0 = self.pointer['electrons'].f_backgr
        self._maxwellian_params = self._f0.params

        assert self._f0.params['u1'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['u2'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['u3'] == 0., "No shifts in velocity space possible!"
        assert self._f0.params['vth1'] == self._f0.params['vth2'] == self._f0.params['vth3'], \
            "Background Maxwellian must be isotropic in velocity space!"

        # Get coupling strength
        self.alpha = self.equation_params['electrons']['alpha_unit']
        self.kappa = 1. / self.equation_params['electrons']['epsilon_unit']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # Create pointers to background electric potential and field
        self._phi_background = self.derham.P['0'](self.electric_equil.phi0)
        self._e_background = self.derham.grad.dot(self._phi_background)
        # ====================================================================================

        # propagator params
        params_maxwell = params['em_fields']['options']['solvers']['maxwell']
        self._poisson_params = params['em_fields']['options']['solvers']['poisson']
        algo_eta = params['kinetic']['electrons']['options']['algos']['push_eta']
        algo_vxb = params['kinetic']['electrons']['options']['algos']['push_vxb']
        params_analytic = params['kinetic']['electrons']['options']['solvers']['analytic']
        params_implicit = params['kinetic']['electrons']['options']['solvers']['implicit']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=algo_eta,
            bc_type=self._electron_params['markers']['bc']['type']))  
        if self._rank == 0:
            print("Added Step PushEta\n")

        self.add_propagator(self.prop_markers.StepVinEfield(
            self.pointer['electrons'],
            e_field=self._e_background + self.pointer['e_field'],
            kappa=self.kappa))
        if self._rank == 0:
            print("Added Step VinEfield\n")

        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['electrons'],
            algo=algo_vxb,
            scale_fac=1.,
            b_eq=self._b_background + self.pointer['b_field'],
            b_tilde=None))  
        if self._rank == 0:
            print("\nAdded Step VxB\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsAnalytic(
            self.pointer['e_field'],
            self.pointer['electrons'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params_analytic))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights Explicit\n")

        # self.add_propagator(self.prop_coupling.EfieldWeightsDiscreteGradient(
        #     self.pointer['e_field'],
        #     self.pointer['electrons'],
        #     alpha=self.alpha,
        #     kappa=self.kappa,
        #     f0=self._f0,
        #     **params['solvers']['solver_ew']))
        # if self._rank == 0:
        #     print("\nAdded Step EfieldWeights Discrete Gradient\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsImplicit(
            self.pointer['e_field'],
            self.pointer['electrons'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            model='delta_f_vlasov_maxwell',
            **params_implicit))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights Semi-Crank-Nicolson\n")

        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e_field'],
            self.pointer['b_field'],
            **params_maxwell))
        if self._rank == 0:
            print("\nAdded Step Maxwell\n")

        # Scalar variables to be saved during simulation
        self.add_scalar('en_e')
        self.add_scalar('en_b')
        self.add_scalar('en_w')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._en_e_tmp = self.pointer['e_field'].space.zeros()
        self._en_b_tmp = self.pointer['b_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
        from struphy.pic.particles import Particles
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        f0_values = self._f0(
            *self.pointer['electrons'].markers_wo_holes[:, :6].T)

        # evaluate f0
        f0_values = self._f0(self.pointer['electrons'].markers[:, 0],
                             self.pointer['electrons'].markers[:, 1],
                             self.pointer['electrons'].markers[:, 2],
                             self.pointer['electrons'].markers[:, 3],
                             self.pointer['electrons'].markers[:, 4],
                             self.pointer['electrons'].markers[:, 5])[~self.pointer['electrons'].holes]
        ln_f0_values = np.log(f0_values)

        self.pointer['electrons']._f0 = self._f0

        # overwrite binning function to always bin marker data for f_1, not h
        def new_binning(self, components, bin_edges, pforms=['0','0']):
            """
            Overwrite the binning method of the parent class to correctly bin data from f_1
            and not from f_0 - (f_0 - f_1) ln(f_0).

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

            ln_f0_values = np.log(f0_values)

            # w_p^h = f_0 * (1 - ln(f_0)) / s_0 - w_p^f * ln(f_0)
            self.markers[~self.holes, 6] *= (-1) * ln_f0_values
            self.markers[~self.holes, 6] += f0_values * \
                (1 - ln_f0_values) / self.markers[~self.holes, 7]

            res = Particles.binning(
                self, components, bin_edges, pforms)
            self.markers[~self.holes, 6] -= f0_values * \
                (1 - ln_f0_values) / self.markers[~self.holes, 7]
            self.markers[~self.holes, 6] /= (-1) * ln_f0_values

            return res

        func_type = type(self.pointer['electrons'].binning)

        self.pointer['electrons'].binning = func_type(
            new_binning, self.pointer['electrons'])

        # Correct initialization of weights
        self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 6] -= \
            f0_values * (1 - ln_f0_values) / \
            self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 7]
        self.pointer['electrons'].markers[~self.pointer['electrons'].holes,
                                          6] /= (-1) * ln_f0_values

        # evaluate f0
        f0_values = self._f0(self.pointer['electrons'].markers[:, 0],
                             self.pointer['electrons'].markers[:, 1],
                             self.pointer['electrons'].markers[:, 2],
                             self.pointer['electrons'].markers[:, 3],
                             self.pointer['electrons'].markers[:, 4],
                             self.pointer['electrons'].markers[:, 5])

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "delta_f_vlasov_maxwell_poisson")

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
            sigma=1e-11,
            phi_n=charge_accum.vectors[0],
            x0=charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e_field'])

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._en_e_tmp)
        en_E = self.pointer['e_field'].dot(self._en_e_tmp) / 2.
        self.update_scalar('en_e', en_E)

        # 0.5 * b^T * M_2 * b
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._en_b_tmp)
        en_B = self.pointer['b_field'].dot(self._en_b_tmp) / 2.
        self.update_scalar('en_b', en_B)

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * sum_p w_p
        self._tmp[0] = \
            self.alpha**2 * \
            (self._f0.params['vth1'] *
             self._f0.params['vth2'] *
             self._f0.params['vth3'])**(2/3) * \
            np.sum(self.pointer['electrons'].markers_wo_holes[:, 6]) / \
            self.pointer['electrons'].n_mks

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_w', self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar('en_tot', self._tmp[0] + en_E + en_B)


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
    def velocity_scale(cls):
        return 'cyclotron'

    def __init__(self, params, comm):

        from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_markers
        from mpi4py.MPI import SUM, IN_PLACE
        from struphy.pic.accumulation.particles_to_grid import Accumulator

        super().__init__(params, comm, a1='Hcurl', ions='Particles6D')

        # pointers to em-field variables
        self._a = self.em_fields['a1']['obj'].vector

        # pointer to kinetic variables
        self._ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        # extract necessary parameters
        # shape function info, degree and support size
        shape_params = params['kinetic']['ions']['ionsshape']
        # electron temperature
        self.thermal = params['kinetic']['electrons']['temperature']

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
                              a=self._a,
                              particle_bc=ions_params['markers']['bc']['type'],
                              quad_number=params['grid']['nq_el'],
                              shape_degree=np.array(shape_params['degree']),
                              shape_size=np.array(shape_params['size']),
                              electron_temperature=self.thermal,
                              accumulate_density=self._accum_density)]

        self._propagators += [propagators_markers.StepPushpxBHybrid(
                              self._ions,
                              method=ions_params['push_algos']['pxb'],
                              a=self._a,
                              b_eq=self._b_eq)]

        self._propagators += [propagators_fields.FaradayExtended(
                              self._a,
                              a_space='Hcurl',
                              beq=self._b_eq,
                              particles=self._ions,
                              quad_number=params['grid']['nq_el'],
                              shape_degree=np.array(shape_params['degree']),
                              shape_size=np.array(shape_params['size']),
                              solver_params=solver_params_1,
                              accumulate_density=self._accum_density)]

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

        self._en_f_loc = pic_util.get_kinetic_energy_particles(
            self._a, self._derham, self._domain, self._ions)/self._ions.n_mks

        self.derham.comm.Reduce(
            self._en_f_loc, self._scalar_quantities['en_f'], op=self._mpi_sum, root=0)

        self._en_thermal_loc = pic_util.get_electron_thermal_energy(self._accum_density, self._derham, self._domain, int(self._derham.domain_array[int(rank), 2]), int(
            self._derham.domain_array[int(rank), 5]), int(self._derham.domain_array[int(rank), 8]), int(self._derham.nquads[0]+1), int(self._derham.nquads[1]+1), int(self._derham.nquads[2]+1))

        self.derham.comm.Reduce(self.thermal*self._en_thermal_loc,
                                self._scalar_quantities['en_thermal'], op=self._mpi_sum, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_thermal'][0]
