import numpy as np
from struphy.models.base import StruphyModel


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

        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # pointers to em-field variables
        self._e = self.em_fields['e_field']['obj'].vector
        self._b = self.em_fields['b_field']['obj'].vector

        # Get rank and size
        self._rank = comm.Get_rank()

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
        self.alpha = self.units_dimless['alpha']
        self.kappa = self.units_dimless['kappa']

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

        self._propagators += [propagators_markers.PushEta(
            self._electrons,
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc_type'],
            f0=None)]  # no conventional weights update here, thus f0=None
        if self._rank == 0:
            print("Added Step PushEta\n")

        # Only add StepVinEfield if e-field is non-zero, otherwise it is more expensive
        if np.all(self._e_background[0]._data < 1e-14) and np.all(self._e_background[1]._data < 1e-14) and np.all(self._e_background[2]._data < 1e-14):
            self._propagators += [propagators_markers.StepVinEfield(
                self._electrons,
                e_field=self._e_background,
                kappa=self.kappa)]
            if self._rank == 0:
                print("Added Step VinEfield\n")

        # Only add VxB Step if b-field is non-zero, otherwise it is more expensive
        b_bckgr_params = params['mhd_equilibrium'][params['mhd_equilibrium']['type']]
        if (b_bckgr_params['B0x'] != 0.) or (b_bckgr_params['B0y'] != 0.) or (b_bckgr_params['B0z'] != 0.):
            self._propagators += [propagators_markers.PushVxB(
                self._electrons,
                algo=electron_params['push_algos']['vxb'],
                scale_fac=1.,
                b_eq=self._b_background,
                b_tilde=None,
                f0=None)]  # no conventional weights update here, thus f0=None
            if self._rank == 0:
                print("Added Step VxB\n")

        self._propagators += [propagators_coupling.EfieldWeightsImplicit(
            self._e,
            self._electrons,
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params['solvers']['solver_ew']
        )]
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        self._propagators += [propagators_fields.Maxwell(
            self._e,
            self._b,
            **params['solvers']['solver_eb'])]
        if self._rank == 0:
            print("\nAdded Step Maxwell\n")

        # Scalar variables to be saved during the simulation
        self._scalar_quantities = {}
        self._scalar_quantities['en_e'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_b'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_w'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_e1'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_e2'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_b3'] = np.empty(1, dtype=float)
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

        # Correct initialization of weights by dividing by N*sqrt(f_0)
        self._electrons.markers[~self._electrons.holes, 6] /= \
            (self._electrons.n_mks *
             np.sqrt(self._f0(*self._electrons.markers_wo_holes[:, :6].T)))

        # evaluate f0
        f0_values = self._f0(self._electrons.markers[:, 0],
                             self._electrons.markers[:, 1],
                             self._electrons.markers[:, 2],
                             self._electrons.markers[:, 3],
                             self._electrons.markers[:, 4],
                             self._electrons.markers[:, 5])

        # Accumulate charge density
        charge_accum = AccumulatorVector(self.derham,
                                         self.domain,
                                         "H1",
                                         "linear_vlasov_maxwell_poisson")
        charge_accum.accumulate(self._electrons, f0_values,
                                np.array(
                                    list(self._maxwellian_params.values())),
                                self.alpha, self.kappa)

        # Subtract the charge local to each process
        charge_accum._vectors[0][:, :, :] -= \
            np.sum(charge_accum.vectors[0].toarray()) / \
            charge_accum.vectors[0].toarray().size

        # Then solve Poisson equation
        poisson_solver = solvers.PoissonSolver(
            rho=charge_accum.vectors[0],
            **self._poisson_params)
        poisson_solver(0.)
        self.derham.grad.dot(-poisson_solver._phi, out=self._e)

    def update_scalar_quantities(self):

        # e^T * M_1 * e
        self._scalar_quantities['en_e'][0] = self._e.dot(
            self._mass_ops.M1.dot(self._e)) / 2.

        # 0.5 * |e_1|^2
        self._scalar_quantities['en_e1'][0] = self._e._blocks[0].dot(
            self._e._blocks[0]) / 2.

        # 0.5 * |e_2|^2
        self._scalar_quantities['en_e2'][0] = self._e._blocks[1].dot(
            self._e._blocks[1]) / 2.

        # b^T * M_2 * b
        self._scalar_quantities['en_b'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b)) / 2.

        # 0.5 * |b_3|^2
        self._scalar_quantities['en_b3'][0] = self._b._blocks[2].dot(
            self._b._blocks[2]) / 2.

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * N/2 * sum_p s_0 * w_p^2
        self._scalar_quantities['en_w'][0] = \
            self.alpha**2 * self._electrons.n_mks / 2. * \
            self._maxwellian_params['vth1']**2 * \
            self._maxwellian_params['vth2']**2 * \
            self._maxwellian_params['vth3']**2 * \
            np.dot(self._electrons.markers_wo_holes[:, 6]**2,  # w_p^2
                   self._electrons.markers_wo_holes[:, 7])  # s_{0,p}

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._scalar_quantities['en_w'], op=self._mpi_sum)

        # en_tot = en_w + en_e + en_b
        self._scalar_quantities['en_tot'][0] = \
            self._scalar_quantities['en_w'][0] + \
            self._scalar_quantities['en_e'][0] + \
            self._scalar_quantities['en_b'][0]


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

        from struphy.propagators.base import Propagator
        from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers
        from struphy.kinetic_background import analytical as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # pointers to em-field variables
        self._e = self.em_fields['e_field']['obj'].vector
        self._b = self.em_fields['b_field']['obj'].vector

        # Get rank and size
        self._rank = comm.Get_rank()

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
        self.alpha = self.units_dimless['alpha']
        self.kappa = self.units_dimless['kappa']

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

        self._propagators += [propagators_markers.PushEta(
            self._electrons,
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc_type'],
            f0=None)]  # no conventional weights update here, thus f0=None
        if self._rank == 0:
            print("Added Step PushEta\n")

        # Only add StepVinEfield if e-field is non-zero, otherwise it is more expensive
        self._propagators += [propagators_markers.StepVinEfield(
            self._electrons,
            e_field=self._e_background + self._e,
            kappa=self.kappa)]
        if self._rank == 0:
            print("Added Step VinEfield\n")

        self._propagators += [propagators_markers.PushVxB(
            self._electrons,
            algo=electron_params['push_algos']['vxb'],
            scale_fac=1.,
            b_eq=self._b_background + self._b,
            b_tilde=None,
            f0=None)]  # no conventional weights update here, thus f0=None
        if self._rank == 0:
            print("\nAdded Step VxB\n")

        self._propagators += [propagators_coupling.EfieldWeightsExplicit(
            self._e,
            self._electrons,
            alpha=self.alpha,
            kappa=self.kappa,
            f0=self._f0,
            **params['solvers']['solver_ew']
        )]
        if self._rank == 0:
            print("\nAdded Step EfieldWeights\n")

        self._propagators += [propagators_fields.Maxwell(
            self._e,
            self._b,
            **params['solvers']['solver_eb'])]
        if self._rank == 0:
            print("\nAdded Step Maxwell\n")

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

        f0_values = self._f0(*self._electrons.markers_wo_holes[:, :6].T)

        # Correct initialization of weights: w_p = f_0 * (1 - log(f_0)) / (N * s_0) - w_p^0 * log(f_0) / N
        self._electrons.markers[~self._electrons.holes, 6] = \
            f0_values * (1 - np.log(f0_values)) / (self._electrons.n_mks * self._electrons.markers_wo_holes[:, 7]) - \
            self._electrons.markers_wo_holes[:, 8] * \
            np.log(f0_values) / self._electrons.n_mks

        # evaluate f0
        f0_values = self._f0(self._electrons.markers[:, 0],
                             self._electrons.markers[:, 1],
                             self._electrons.markers[:, 2],
                             self._electrons.markers[:, 3],
                             self._electrons.markers[:, 4],
                             self._electrons.markers[:, 5])

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "delta_f_vlasov_maxwell_poisson")
        charge_accum.accumulate(self._electrons, f0_values,
                                np.array(
                                    list(self._maxwellian_params.values())),
                                self.alpha, self.kappa)

        # Subtract the charge local to each process
        charge_accum._vectors[0][:, :, :] -= \
            np.sum(charge_accum.vectors[0].toarray()) / \
            charge_accum.vectors[0].toarray().size

        # Then solve Poisson equation
        poisson_solver = solvers.PoissonSolver(
            rho=charge_accum.vectors[0], **self._poisson_params)
        poisson_solver(0.)
        self.derham.grad.dot(-poisson_solver._phi, out=self._e)

    def update_scalar_quantities(self):

        # e^T * M_1 * e
        self._scalar_quantities['en_e'][0] = self._e.dot(
            self._mass_ops.M1.dot(self._e)) / 2.

        # b^T * M_2 * b
        self._scalar_quantities['en_b'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b)) / 2.

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * sum_p w_p
        self._scalar_quantities['en_w'][0] = \
            self.alpha**2 * \
            self._maxwellian_params['vth1']**2 * \
            self._maxwellian_params['vth2']**2 * \
            self._maxwellian_params['vth3']**2 * \
            np.sum(self._electrons.markers_wo_holes[:, 6])

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._scalar_quantities['en_w'], op=self._mpi_sum)

        # en_tot = en_w + en_e + en_b
        self._scalar_quantities['en_tot'][0] = \
            self._scalar_quantities['en_w'][0] + \
            self._scalar_quantities['en_e'][0] + \
            self._scalar_quantities['en_b'][0]


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
