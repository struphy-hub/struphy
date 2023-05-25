import numpy as np
from struphy.models.base import StruphyModel

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

        assert u_space in [
            'Hdiv', 'H1vec'], f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'

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

        self._scalar_quantities['en_tot'][0] = en_U
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
    r'''Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`). For homogenous background conditions.

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}  \,.

    Implemented equations:

    .. math::

        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 

        n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla (\tilde p_i + \tilde p_e)
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 \,,

        &\frac{\partial \tilde p_e}{\partial t} + \frac{5}{3}\,p_{e,0}\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde p_i}{\partial t} + \frac{5}{3}\,p_{i,0}\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 - \kappa \frac{\nabla\times \tilde{\mathbf{B}}}{n_0}\times \mathbf{B}_0 \right)
        = 0\,.

    where

    .. math::

        \kappa = \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.


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
        Hall_solver = params['solvers']['solver_2']
        SonicIon_solver = params['solvers']['solver_3']
        SonicElectron_solver = params['solvers']['solver_4']

        # project background magnetic field (1-form) and pressure (3-form)
        self._b_eq = self.derham.P['1']([self.mhd_equil.b1_1,
                                         self.mhd_equil.b1_2,
                                         self.mhd_equil.b1_3])
        self._p_i_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._p_e_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_i.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # compute coupling parameter kappa
        units_basic, units_der, units_dimless = self.model_units(
            params, verbose=False)

        ee = 1.602176634e-19  # elementary charge (C)
        mH = 1.67262192369e-27  # proton mass (kg)

        Ab = params['fluid']['mhd']['phys_params']['A']
        Zb = params['fluid']['mhd']['phys_params']['Z']

        omega_ch = (Zb*ee*units_basic['B'])/(Ab*mH)
        kappa = omega_ch*units_basic['t']/(2.0 * 3.141592654)

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        self._coupling_params = {}
        self._coupling_params['kappa'] = kappa

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
        self._propagators += [propagators_fields.Hall(
            self._b,
            **Hall_solver,
            **self._coupling_params)]
        self._propagators += [propagators_fields.SonicIon(
            self._n,
            self._u,
            self._p_i,
            **SonicIon_solver)]
        self._propagators += [propagators_fields.SonicElectron(
            self._n,
            self._u,
            self._p_e,
            **SonicElectron_solver)]

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

        en_U = self._u.dot(self._tmp_u1)/2.0
        en_B = self._b.dot(self._tmp_b1)/2.0
        en_p_i = self._p_i.dot(self._ones)/(5.0/3.0 - 1.0)
        en_p_e = self._p_e.dot(self._ones)/(5.0/3.0 - 1.0)

        self._scalar_quantities['en_U'][0] = en_U
        self._scalar_quantities['en_B'][0] = en_B
        self._scalar_quantities['en_p_i'][0] = en_p_i
        self._scalar_quantities['en_p_e'][0] = en_p_e

        self._scalar_quantities['en_tot'][0] = en_U
        self._scalar_quantities['en_tot'][0] += en_B
        self._scalar_quantities['en_tot'][0] += en_p_i
        self._scalar_quantities['en_tot'][0] += en_p_e

        # background fields
        self._mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2.0
        en_p0_i = self._p_i_eq.dot(self._ones)/(5.0/3.0 - 1.0)
        en_p0_e = self._p_e_eq.dot(self._ones)/(5.0/3.0 - 1.0)

        self._scalar_quantities['en_B_eq'][0] = en_B0
        self._scalar_quantities['en_p_i_eq'][0] = en_p0_i
        self._scalar_quantities['en_p_e_eq'][0] = en_p0_e

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self._b

        self._mass_ops.M1.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2.0

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