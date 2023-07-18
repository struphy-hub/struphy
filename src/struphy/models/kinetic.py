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
            &\partial_t f + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right)
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{\hat \omega}{\hat \Omega_\textnormal{c}} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}
        \end{align}

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
                         e1='Hcurl', b2='Hdiv',
                         electrons='Particles6D')

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        electron_params = params['kinetic']['electrons']

        # model parameters
        self._alpha = self.eq_params['electrons']['alpha_unit']
        self._epsilon = self.eq_params['electrons']['epsilon_unit']

        # Get Poisson solver params parameters
        self._poisson_params = params['solvers']['solver_poisson']

        # ====================================================================================
        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params['solvers']['solver_maxwell']))
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc_type'],
            f0=None))
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['electrons'],
            algo=electron_params['push_algos']['vxb'],
            scale_fac=1/self._epsilon,
            b_eq=self._b_background,
            b_tilde=self.pointer['b2'],
            f0=None))
        self.add_propagator(self.prop_coupling.VlasovMaxwell(
            self.pointer['e1'],
            self.pointer['electrons'],
            c1=self._alpha**2/self._epsilon,
            c2=1/self._epsilon,
            **params['solvers']['solver_vlasovmaxwell']))

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

        from struphy.pic.particles_to_grid import AccumulatorVector
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "vlasov_maxwell_poisson")
        charge_accum.accumulate(self.pointer['electrons'])

        # Locally subtract mean charge for solvability with periodic bc
        if np.all(charge_accum.vectors[0].space.periods):
            charge_accum._vectors[0][:] -= np.mean(charge_accum.vectors[0].toarray()[
                                                   charge_accum.vectors[0].toarray() != 0])

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh['0'])
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma=1e-11,
            phi_n=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            x0=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e1'])

    def update_scalar_quantities(self):
        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = self.pointer['e1'].dot(self._tmp1) / 2.
        en_B = self.pointer['b2'].dot(self._tmp2) / 2.
        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self._alpha**2 / (2 * self.pointer['electrons'].n_mks) * \
            np.dot(self.pointer['electrons'].markers_wo_holes[:, 3]**2 + self.pointer['electrons'].markers_wo_holes[:, 4] ** 2 +
                   self.pointer['electrons'].markers_wo_holes[:, 5]**2, self.pointer['electrons'].markers_wo_holes[:, 6])
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
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \kappa\left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \kappa \sqrt{f_0} \, \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \alpha^2 \kappa \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \kappa = 2\pi \frac{\Omega_\textnormal{c}}{\hat \omega} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    At initial time the Poisson equation is solved once to satisfy the Gauss law

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \alpha^2 \kappa \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \text{d}^3 \mathbf{v}
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
    def bulk_species(cls):
        return 'electrons'

    @classmethod
    def timescale(cls):
        return 'light'

    def __init__(self, params, comm):

        super().__init__(params, comm,
                         e_field='Hcurl', b_field='Hdiv',
                         electrons='Particles6D')

        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        electron_params = params['kinetic']['electrons']

        # kinetic background
        assert electron_params['background']['type'] == 'Maxwellian6DUniform', \
            AssertionError(
                "The background distribution function must be a uniform Maxwellian!")

        self._maxwellian_params = electron_params['background']['Maxwellian6DUniform']
        assert self._maxwellian_params['u1'] == 0., "No shifts in velocity space possible!"
        assert self._maxwellian_params['u2'] == 0., "No shifts in velocity space possible!"
        assert self._maxwellian_params['u3'] == 0., "No shifts in velocity space possible!"
        self.pointer['electrons']._f_backgr = getattr(
            kin_ana, 'Maxwellian6DUniform')(**self._maxwellian_params)
        self._f0 = self.pointer['electrons'].f_backgr

        # Get coupling strength
        self.alpha = self.eq_params['electrons']['alpha_unit']
        self.kappa = 1. / self.eq_params['electrons']['epsilon_unit']

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

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc_type'],
            f0=None))  # no conventional weights update here, thus f0=None
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
        b_bckgr_params = params['mhd_equilibrium'][params['mhd_equilibrium']['type']]
        if (b_bckgr_params['B0x'] != 0.) or (b_bckgr_params['B0y'] != 0.) or (b_bckgr_params['B0z'] != 0.):
            self.add_propagator(self.prop_markers.PushVxB(
                self.pointer['electrons'],
                algo=electron_params['push_algos']['vxb'],
                scale_fac=1.,
                b_eq=self._b_background,
                b_tilde=None,
                f0=None))  # no conventional weights update here, thus f0=None
            if self._rank == 0:
                print("Added Step VxB\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsImplicit(
            self.pointer['e_field'],
            self.pointer['electrons'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params['solvers']['solver_ew']))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e_field'],
            self.pointer['b_field'],
            **params['solvers']['solver_eb']))
        if self._rank == 0:
            print("\nAdded Step Maxwell\n")

        # Scalar variables to be saved during the simulation
        self.add_scalar('en_e')
        self.add_scalar('en_b')
        self.add_scalar('en_w')
        self.add_scalar('en_e1')
        self.add_scalar('en_e2')
        self.add_scalar('en_b3')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._en_e_tmp = self.pointer['e_field'].space.zeros()
        self._en_b_tmp = self.pointer['b_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):

        from struphy.pic.particles_to_grid import AccumulatorVector
        from psydac.linalg.stencil import StencilVector

        # Get physical properties of the Maxwellian
        init_type = self.kinetic['electrons']['params']['init']['type']
        if init_type == 'Maxwellian6DUniform':
            sigma1 = self.kinetic['electrons']['params']['init'][init_type]['vth1']
            sigma2 = self.kinetic['electrons']['params']['init'][init_type]['vth2']
            sigma3 = self.kinetic['electrons']['params']['init'][init_type]['vth3']
        elif init_type == 'Maxwellian6DPerturbed':
            sigma1 = self.kinetic['electrons']['params']['init'][init_type]['vth1']['vth01']
            sigma2 = self.kinetic['electrons']['params']['init'][init_type]['vth2']['vth02']
            sigma3 = self.kinetic['electrons']['params']['init'][init_type]['vth3']['vth03']
        else:
            raise NotImplementedError('Unknown initialization function!')

        # Compute determinant of the diagonal matrix holding the thermal velocities
        det_one_th = 1 / (sigma1 * sigma2 * sigma3)

        # Compute scaled velocities
        vth1 = sigma1**3 * det_one_th**(2/3)
        vth2 = sigma2**3 * det_one_th**(2/3)
        vth3 = sigma3**3 * det_one_th**(2/3)

        # Set scaled velocities in params for initialization
        if init_type == 'Maxwellian6DUniform':
            self.kinetic['electrons']['params']['init'][init_type]['vth1'] = vth1
            self.kinetic['electrons']['params']['init'][init_type]['vth2'] = vth2
            self.kinetic['electrons']['params']['init'][init_type]['vth3'] = vth3
        elif init_type == 'Maxwellian6DPerturbed':
            self.kinetic['electrons']['params']['init'][init_type]['vth1']['vth01'] = vth1
            self.kinetic['electrons']['params']['init'][init_type]['vth2']['vth02'] = vth2
            self.kinetic['electrons']['params']['init'][init_type]['vth3']['vth03'] = vth3

        # Take smaller width of the two gaussians for markers drawing in order to avoid
        # small values for f0 and hence division by zero
        self.kinetic['electrons']['params']['markers']['loading']['moments'][3] = \
            min(vth1, sigma1)
        self.kinetic['electrons']['params']['markers']['loading']['moments'][4] = \
            min(vth2, sigma2)
        self.kinetic['electrons']['params']['markers']['loading']['moments'][5] = \
            min(vth3, sigma3)

        # Initialize fields and particles
        super().initialize_from_params()

        # edges = self.kinetic['electrons']['bin_edges']['e1']
        # # edges = [self.kinetic['electrons']['bin_edges']['v1_v2'][1]]
        # components = [False] * 6
        # components[0] = True

        # self.pointer['electrons'].show_distribution_function(components, edges, self.domain)

        # Correct initialization of weights by dividing by sqrt(f_0)
        self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 6] /= \
            (np.sqrt(
                self._f0(*self.pointer['electrons'].markers_wo_holes[:, :6].T)))

        # # Set v3 = 0
        # self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 5] = 0.

        # self.pointer['electrons'].show_distribution_function(components, edges, self.domain)
        # exit()

        # evaluate f0
        f0_values = self._f0(self.pointer['electrons'].markers[:, 0],
                             self.pointer['electrons'].markers[:, 1],
                             self.pointer['electrons'].markers[:, 2],
                             self.pointer['electrons'].markers[:, 3],
                             self.pointer['electrons'].markers[:, 4],
                             self.pointer['electrons'].markers[:, 5])

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

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._en_e_tmp)
        en_E = self.pointer['e_field'].dot(self._en_e_tmp) / 2.
        self.update_scalar('en_e', en_E)

        # 0.5 * |e_1|^2
        self.update_scalar('en_e1', self.pointer['e_field']._blocks[0].dot(
            self._en_e_tmp._blocks[0]) / 2.)

        # 0.5 * |e_2|^2
        self.update_scalar('en_e2', self.pointer['e_field']._blocks[1].dot(
            self._en_e_tmp._blocks[1]) / 2.)

        # 0.5 * b^T * M_2 * b
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._en_b_tmp)
        en_B = self.pointer['b_field'].dot(self._en_b_tmp) / 2.
        self.update_scalar('en_b', en_B)

        # 0.5 * |b_3|^2
        self.update_scalar('en_b3', self.pointer['b_field']._blocks[2].dot(
            self._en_b_tmp._blocks[2]) / 2.)

        # alpha^2 / (2N) * (v_th_1 * v_th_2 * v_th_3)^(2/3) * sum_p s_0 * w_p^2
        self._tmp[0] = \
            self.alpha**2 / (2 * self.pointer['electrons'].n_mks) * \
            (self._maxwellian_params['vth1'] *
             self._maxwellian_params['vth2'] *
             self._maxwellian_params['vth3'])**(2/3) * \
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
            &\partial_t h + \mathbf{v} \cdot \, \nabla h + \kappa\left( \mathbf{E}_0 + \mathbf{E} + \mathbf{v} \times (\mathbf{B}_0 + \mathbf{B}) \right)
            \cdot \frac{\partial h}{\partial \mathbf{v}} = \kappa \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right)
            \, \left( \frac{f_0 - h}{\ln(f_0)} - f_0 \right) \,,
            \\[2mm]
            &\frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{B} -
            \alpha^2 \kappa \int_{\mathbb{R}^3} \left( \frac{f_0 - h}{\ln(f_0)} - f_0 \right) \left( \mathbb{1}_{\text{th}}^2
            (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            &\frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E} \,,
        \end{align}

    where

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \kappa = 2\pi \frac{\Omega_\textnormal{c}}{\hat \omega} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 A m_\textnormal{H}}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

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
    def bulk_species(cls):
        return 'electrons'

    @classmethod
    def timescale(cls):
        return 'light'

    def __init__(self, params, comm):

        super().__init__(params, comm,
                         e_field='Hcurl', b_field='Hdiv',
                         electrons='Particles6D')

        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        electron_params = params['kinetic']['electrons']

        # kinetic background
        assert electron_params['background']['type'] == 'Maxwellian6DUniform', \
            "The background distribution function must be a uniform Maxwellian!"

        self._maxwellian_params = electron_params['background']['Maxwellian6DUniform']
        self.pointer['electrons']._f_backgr = getattr(
            kin_ana, 'Maxwellian6DUniform')(**self._maxwellian_params)
        self._f0 = self.pointer['electrons'].f_backgr

        # Get coupling strength
        self.alpha = self.eq_params['electrons']['alpha_unit']
        self.kappa = 1. / self.eq_params['electrons']['epsilon_unit']

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

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['electrons'],
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc_type'],
            f0=None))  # no conventional weights update here, thus f0=None
        if self._rank == 0:
            print("Added Step PushEta\n")

        # Only add StepVinEfield if e-field is non-zero, otherwise it is more expensive
        self.add_propagator(self.prop_markers.StepVinEfield(
            self.pointer['electrons'],
            e_field=self._e_background + self.pointer['e_field'],
            kappa=self.kappa))
        if self._rank == 0:
            print("Added Step VinEfield\n")

        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['electrons'],
            algo=electron_params['push_algos']['vxb'],
            scale_fac=1.,
            b_eq=self._b_background + self.pointer['b_field'],
            b_tilde=None,
            f0=None))  # no conventional weights update here, thus f0=None
        if self._rank == 0:
            print("\nAdded Step VxB\n")

        self.add_propagator(self.prop_coupling.EfieldWeightsExplicit(
            self.pointer['e_field'],
            self.pointer['electrons'],
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params['solvers']['solver_ew']))
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e_field'],
            self.pointer['b_field'],
            **params['solvers']['solver_eb']))
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

        from struphy.pic.particles_to_grid import AccumulatorVector
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        f0_values = self._f0(
            *self.pointer['electrons'].markers_wo_holes[:, :6].T)

        # Correct initialization of weights: w_p = f_0 * (1 - log(f_0)) / (N * s_0) - w_p^0 * log(f_0) / N
        self.pointer['electrons'].markers[~self.pointer['electrons'].holes, 6] = \
            f0_values * (1 - np.log(f0_values)) / (self.pointer['electrons'].n_mks * self.pointer['electrons'].markers_wo_holes[:, 7]) - \
            self.pointer['electrons'].markers_wo_holes[:, 8] * \
            np.log(f0_values) / self.pointer['electrons'].n_mks

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
            sigma=0.,
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
            self._maxwellian_params['vth1']**2 * \
            self._maxwellian_params['vth2']**2 * \
            self._maxwellian_params['vth3']**2 * \
            np.sum(self.pointer['electrons'].markers_wo_holes[:, 6])

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
                              particle_bc=ions_params['markers']['bc_type'],
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
            self._derham.domain_array[int(rank), 5]), int(self._derham.domain_array[int(rank), 8]), int(self._derham.quad_order[0]+1), int(self._derham.quad_order[1]+1), int(self._derham.quad_order[2]+1))

        self.derham.comm.Reduce(self.thermal*self._en_thermal_loc,
                                self._scalar_quantities['en_thermal'], op=self._mpi_sum, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_thermal'][0]
