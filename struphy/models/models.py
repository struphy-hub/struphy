import numpy as np
from mpi4py import MPI

from struphy.models.base import StruphyModel
from struphy.polar.basic import PolarVector
from struphy.models.pre_processing import plasma_params

#############################
# Fluid models
#############################
class LinearMHD(StruphyModel):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    Normalization:

    .. math::

        \frac{\hat B}{\sqrt{\hat \rho \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,,

    where :math:`\mu_0` the vacuum permeability. Implemented equations:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,

    where the equilibrium quantities must satisfy the MHD equilibrium condition :math:`\nabla p_0=\mathbf J_0\times\mathbf B_0`.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.propagators import propagators_fields

        self._u_space = params['fluid']['mhd']['mhd_u_space']

        if self._u_space == 'Hdiv':
            u_name = 'u2'
        elif self._u_space == 'H1vec':
            u_name = 'uv'
        else:
            raise ValueError(
                f'MHD velocity must be in Hdiv or in H1vec, but has been specified in {self._u_space}.')

        super().__init__(params, comm, b2='Hdiv', mhd={
            'n3': 'L2', u_name: self._u_space, 'p3': 'L2'})

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # extract necessary parameters
        shearalfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']

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

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(
            self.derham, self.domain, self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(self._u, self._b, self._u_space, self.derham,
                                                             self._mass_ops, self._basis_ops, shearalfven_solver)]
        self._propagators += [propagators_fields.Magnetosonic(self._n, self._u, self._p, self._b, self._u_space,
                                                              self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]

        # Scalar variables to be saved during simulation

        # time
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

        # energies
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @staticmethod
    def print_units(model_units_params):

        # pressure unit in Pascal
        pressure_unit = model_units_params['B']**2/1.25663706212e-6

        # beta 2*mu0*p/B^2 (always 2)
        beta = 2.

        # temperature unit kBT = p/n in keV
        temperature_unit = pressure_unit / \
            (model_units_params['n']*1e20)/(1000*1.602176634e-19)

        size_params = {'B_abs [T]': model_units_params['B'],
                       'transit k [1/m]': 2*np.pi/model_units_params['L']}

        pparams = plasma_params(
            1, model_units_params['A'], temperature_unit, 2, size_params)

        print()
        print()
        print('------- MODEL UNITS (PRESCRIBED) -------')
        print('x [m]        : ', model_units_params['L'])
        print('B [T]        : ', model_units_params['B'])
        print('n [10²⁰ m⁻³] : ', model_units_params['n'])
        print('A            : ', model_units_params['A'])
        print()
        print('------- MODEL UNITS (DERIVED) ----------')
        print('rho [10⁷ kg/m³] : ', model_units_params['n'] *
              1e20*model_units_params['A']*1.67262192369e-27*1e7)
        print('p   [bar]       : ', pressure_unit*1e-5)
        print('t   [µs]        : ', model_units_params['L'] /
              pparams['v_A [10^6 m/s]'])
        print('v   [10⁶ m/s]   : ', pparams['v_A [10^6 m/s]'])
        print()
        print('------- OTHER QUANTITIES _--------------')

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2

        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
            self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
            self._mass_ops.M2.dot(self._b_eq, apply_bc=False))/2

        self._scalar_quantities['en_B_tot'][0] = (
            self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b, apply_bc=False))/2

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]


#############################
# Fluid-kinetic hybrid models
#############################
class LinearMHDVlasovCC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **current coupling scheme**.

    Normalization:

    .. math::

        &\mathbf{x}=\mathbf{x}^\prime\hat{L}\,,\quad\mathbf{B}=\mathbf{B}^\prime\hat{B}\,,\quad\rho_\textnormal{b}=n_\textnormal{b}^\prime A_\textnormal{b}m_\textnormal{p}\hat{n}_\textnormal{b}\,,\quad p=p^\prime\frac{\hat{B}^2}{\mu_0}\,,\quad\mathbf{U}=\mathbf{U}^\prime\hat{v}_\textnormal{A}\,,

        &t=t^\prime\hat{\tau}_{\textnormal{A}}\,,\quad\hat{\tau}_{\textnormal{A}}=\frac{\hat{L}}{\hat{v}_{\textnormal{A}}}\,,\quad\hat{v}_{\textnormal{A}}=\frac{\hat{B}}{\sqrt{\mu_0A_\textnormal{b}m_\textnormal{p}\hat{n}_\textnormal{b}}}\,,

        &f_\textnormal{h}=f_\textnormal{h}^\prime\frac{\hat{n}_\textnormal{h}}{\hat{v}_\textnormal{A}^3}\,,\quad \rho_\textnormal{h}=Z_\textnormal{h}en_\textnormal{h}^\prime\hat{n}_\textnormal{h}\,,\quad\mathbf{J}_\textnormal{h}=Z_\textnormal{h}en_\textnormal{h}^\prime \mathbf{U}_\textnormal{h}^\prime\hat{n}_\textnormal{h}\hat{v}_\textnormal{A}\,,

    where :math:`e` is the elementary charge, :math:`m_\textnormal{p}` the proton mass and :math:`\mu_0` the vaccuum permeability. 

    Implemented equations (dimensionless, primes are dropped):

    .. math::

        \begin{align}
        \textnormal{linear MHD}\,\, &\left\{\,\,
        \begin{aligned}
        &\frac{\partial \tilde{n}_\textnormal{b}}{\partial t}+\nabla\cdot(n_\textnormal{b0} \tilde{\mathbf{U}})=0\,, 
        \\
        n_\textnormal{b0} &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}+\nu_\textnormal{h}\frac{Z_\textnormal{h}}{A_\textnormal{b}}\kappa\left(n_\textnormal{h}\tilde{\mathbf{U}}-n_\textnormal{h}\mathbf{U}_\textnormal{h}\right)\times(\mathbf{B}_0+\tilde{\mathbf{B}})\,,
        \\
        &\frac{\partial \tilde p}{\partial t} + (\gamma-1)\nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)\,,\qquad \nabla\cdot\tilde{\mathbf{B}}=0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\textnormal{h}+\frac{Z_\textnormal{h}}{A_\textnormal{h}}\kappa\left[(\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}}+\mathbf{v}\times(\mathbf{B}_0+\tilde{\mathbf{B}})\right]\cdot\nabla_{\mathbf{v}}f_\textnormal{h}=0\,,
        \\
        &\quad\,\,n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3v\,,\qquad n_\textnormal{h}\mathbf{U}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3v\,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0` is the equilibrium current and subscripts "b" and "h" refer to bulk (MHD) and hot (energetic) species, respectively. Moreover, the dimensionless quantities :math:`\nu_\textnormal{h}=\hat{n}_\textnormal{h}/\hat{n}_\textnormal{b}` and :math:`\kappa=\hat{\Omega}_\textnormal{cp}\hat{\tau}_\textnormal{A}=e\hat{B}\,\hat{\tau}_\textnormal{A}/m_\textnormal{p}=e\hat{L}\sqrt{\mu_0A_\textnormal{b}\hat{n}_\textnormal{b}/m_\textnormal{p}}`.

    The characteristic quantities :math:`(\hat{L},\hat{B},\hat{n}_\textnormal{b})` as well as the dimensionless constants :math:`(\nu_\textnormal{h},A_\textnormal{b},A_\textnormal{h},Z_\textnormal{h})` need to be defined by the user in the parameter file.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    """

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling

        from struphy.kinetic_background import analytical as kin_ana

        self._u_space = params['fluid']['mhd']['mhd_u_space']

        if self._u_space == 'Hdiv':
            u_name = 'u2'
        elif self._u_space == 'H1vec':
            u_name = 'uv'
        else:
            raise ValueError(
                f'MHD velocity must be in Hdiv or in H1vec, but has been specified in {self._u_space}.')

        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles6D')

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to energetic ions
        self._e_ions = self.kinetic['energetic_ions']['obj']
        e_ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']
        solver_params_4 = params['solvers']['solver_4']

        # model units
        B = params['model_units']['B']
        L = params['model_units']['L']
        nb = params['model_units']['nb']
        nuh = params['model_units']['nuh']*0.01
        Ab = params['model_units']['Ab']
        Ah = params['model_units']['Ah']
        Zh = params['model_units']['Zh']

        kappa = 1.602176634e-19*L * \
            np.sqrt(1.25663706212e-6*Ab*nb*1e20/1.67262192369e-27)

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        coupling_params = {'nuh': nuh, 'Ab': Ab,
                           'Ah': Ah, 'Zh': Zh, 'kappa': kappa}

        if self.derham.comm.Get_rank() == 0:
            print('Bulk / EP coupling parameters : ', coupling_params)

        # background distribution function used as control variate
        if params['kinetic']['energetic_ions']['markers']['type'] == 'control_variate':
            assert 'background' in params['kinetic']['energetic_ions'], 'no background distribution function for control variate specified'
            control = True
            f0_name = params['kinetic']['energetic_ions']['background']['type']
            f0 = getattr(kin_ana, f0_name)(
                **params['kinetic']['energetic_ions']['background'][f0_name])
        else:
            control = False
            f0 = None

        # project background magnetic field (2-form) and background pressure (3-form)
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)

        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # mass and basis projection operators
        if control:
            self._mass_ops = WeightedMassOperators(
                self.derham, self.domain, eq_mhd=self.mhd_equil, kinetic_fun=f0)
        else:
            self._mass_ops = WeightedMassOperators(
                self.derham, self.domain, eq_mhd=self.mhd_equil)

        self._base_ops = BasisProjectionOperators(
            self.derham, self.domain, self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []

        # updates u
        self._propagators += [propagators_fields.CurrentCoupling6DDensity(
            self._e_ions, self.derham, self.domain, self._mass_ops, solver_params_1, coupling_params, self._u, self._u_space, self._b_eq, self._b, f0=f0)]

        # updates u and b
        self._propagators += [propagators_fields.ShearAlfvén(
            self._u, self._b, self._u_space, self.derham, self._mass_ops, self._base_ops, solver_params_2)]

        # updates u and v (and weights for control variate)
        self._propagators += [propagators_coupling.CurrentCoupling6DCurrent(
            self._e_ions, self.derham, self.domain, self._mass_ops, solver_params_3, coupling_params, self._u, self._u_space, self._b_eq, self._b, f0=f0)]

        # updates eta
        self._propagators += [propagators_markers.StepPushEta(
            self._e_ions, self.derham, self.domain, e_ions_params['push_algos']['eta'], e_ions_params['markers']['bc_type'], f0=f0)]

        # updates v
        self._propagators += [propagators_markers.StepPushVxB(
            self._e_ions, self.derham, self.domain, e_ions_params['push_algos']['vxb'], kappa*Zh/Ah, self._b_eq, self._b, f0=f0)]

        # updates u and p
        self._propagators += [propagators_fields.Magnetosonic(
            self._n, self._u, self._p, self._b, self._u_space, self.derham, self._mass_ops, self._base_ops, solver_params_4)]

        # Scalar variables to be saved during simulation:

        # 1. time
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

        # 2. energies
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_eq'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B_tot'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @staticmethod
    def print_units(model_units_params):

        # pressure unit in Pascal
        pressure_unit = model_units_params['B']**2/1.25663706212e-6

        # beta 2*mu0*p/B^2 (always 2)
        beta = 2.

        # bulk temperature unit kBT = p/nb in keV
        temperature_unit = pressure_unit / \
            (model_units_params['nb']*1e20)/(1000*1.602176634e-19)

        size_params = {'B_abs [T]': model_units_params['B'],
                       'transit k [1/m]': 2*np.pi/model_units_params['L']}

        pparams = plasma_params(
            1, model_units_params['Ab'], temperature_unit, 2, size_params)

        print()
        print()
        print('------- MODEL UNITS (PRESCRIBED) -------')
        print('x  [m]        : ', model_units_params['L'])
        print('B  [T]        : ', model_units_params['B'])
        print('nb [10²⁰ m⁻³] : ', model_units_params['nb'])
        print('Ab            : ', model_units_params['Ab'])
        print('nh [10²⁰ m⁻³] : ', model_units_params['nb']
              * model_units_params['nuh']*0.01)
        print('Ah            : ', model_units_params['Ah'])
        print('Zh            : ', model_units_params['Zh'])
        print()
        print('------- MODEL UNITS (DERIVED) ----------')
        print('rho_b [10⁷ kg/m⁻³] : ', model_units_params['nb']
              * 1e20*model_units_params['Ab']*1.67262192369e-27*1e7)
        print('p     [bar]        : ', pressure_unit*1e-5)
        print('t     [µs]         : ',
              model_units_params['L']/pparams['v_A [10^6 m/s]'])
        print('v     [10⁶ m/s]    : ', pparams['v_A [10^6 m/s]'])
        print()
        print('------- OTHER QUANTITIES ---------------')
        print('nuh                 : ', model_units_params['nuh']*0.01)
        print('kappa               : ', pparams['kappa'])
        print('EP gyro period [µs] : ', 2*np.pi*model_units_params['Ah']*1.67262192369e-27/(
            model_units_params['Zh']*1.602176634e-19*model_units_params['B'])*1e6)
        print('EP gyro radius [cm] : ', pparams['v_A [10^6 m/s]']*1e6*model_units_params['Ah'] *
              1.67262192369e-27/(model_units_params['Zh']*1.602176634e-19*model_units_params['B'])*100)

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2

        self._scalar_quantities['en_p'][0] = self._p.dot(self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
            self._ones)/(5/3 - 1)
        self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
            self._mass_ops.M2.dot(self._b_eq, apply_bc=False))/2

        self._scalar_quantities['en_B_tot'][0] = (
            self._b_eq + self._b).dot(self._mass_ops.M2.dot(self._b_eq + self._b, apply_bc=False))/2

        self._scalar_quantities['en_f'][0] = 0.05*self._e_ions.markers_wo_holes[:, 6].dot(
            self._e_ions.markers_wo_holes[:, 3]**2 +
            self._e_ions.markers_wo_holes[:, 4]**2 +
            self._e_ions.markers_wo_holes[:, 5]**2)/(2*self._e_ions.n_mks)

        self.derham.comm.Allreduce(
            MPI.IN_PLACE, self._scalar_quantities['en_f'], op=MPI.SUM)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


class LinearMHDVlasovPC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with **pressure coupling scheme**. 

    Normalization: 

    .. math::

        \frac{\hat B^2}{\hat \rho \mu_0} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,.

    Implemented equations:

    HybridMHDVlasovPC

    .. math::

        \begin{align}
        \textnormal{linear MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{red}+ \nabla\cdot \tilde{\mathbb{P}}_{h,\perp} \color{black} 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{Vlasov}\qquad& \frac{\partial f_h}{\partial t} + (\mathbf{v} \color{red} + \tilde{\mathbf{U}}_\perp \color{black})\cdot\frac{\partial f_h}{\partial \mathbf{x}}
        + \left[\frac{q_h}{m_h}\mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{red}- \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \color{black} \right]\cdot\frac{\partial f_h}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\color{red} \tilde{\mathbb{P}}_{h,\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_h d\mathbf{v} \color{black}\,.
        \end{align}

    HybridMHDVlasovPC_full (including the parallel pressure tensor)

    .. math::

        \begin{align}
        \textnormal{linear MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        &\rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{red}+ \nabla\cdot \tilde{\mathbb{P}}_h \color{black} 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{Vlasov}\qquad& \frac{\partial f_h}{\partial t} + (\mathbf{v} \color{red} + \tilde{\mathbf{U}} \color{black})\cdot\frac{\partial f_h}{\partial \mathbf{x}}
        + \left[\frac{q_h}{m_h}\mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{red}- \nabla \tilde{\mathbf{U}}\cdot \mathbf{v} \color{black} \right]\cdot\frac{\partial f_h}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\color{red} \tilde{\mathbb{P}}_h = \int \mathbf{v}\mathbf{v}^\top f_h d\mathbf{v} \color{black}\,.
        \end{align}

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling

        self._u_space = params['fluid']['mhd']['mhd_u_space']

        if self._u_space == 'Hdiv':
            u_name = 'u2'
        elif self._u_space == 'H1vec':
            u_name = 'uv'
        else:
            raise ValueError(
                f'MHD velocity must be in Hdiv or in H1vec, but has been specified in {self._u_space}.')

        super().__init__(params, comm, b2='Hdiv', mhd={
            'n3': 'L2', u_name: self._u_space, 'p3': 'L2'}, energetic_ions='Particles6D')

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to kinetic variables
        self._ions = self.kinetic['energetic_ions']['obj']
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        alfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']
        coupling_solver = params['solvers']['solver_3']
        coupling = ions_params['pc']
        self._nuh = self.kinetic['energetic_ions']['plasma_params']['n [10^20/m^3]'] / \
            self.fluid['mhd']['plasma_params']['n [10^20/m^3]']

        print('Coupling parameter nu_h = n_h/n = ' + str(self._nuh) + '\n')

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(
            self.derham, self.domain, self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.ShearAlfvén(
            self._u, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, alfven_solver)]
        self._propagators += [propagators_fields.Magnetosonic(
            self._n, self._u, self._p, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]
        self._propagators += [propagators_markers.StepPushEtaPC(
            self._ions, self.derham, self.domain, self._u, self._u_space, coupling, ions_params['markers']['bc_type'])]
        self._propagators += [propagators_coupling.StepPressurecoupling(
            self._u, self._u_space, coupling, self._ions, self.derham, self.domain, self._mass_ops, self._basis_ops, coupling_solver)]
        self._propagators += [propagators_markers.StepPushVxB(
            self._ions, self.derham, self.domain, ions_params['push_algos']['vxb'], 1., self._b, self._b_eq)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hcurl':
            self._en_U_loc = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)

        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._en_f_loc = self._ions.markers[~self._ions.holes, 8].dot(self._ions.markers[~self._ions.holes, 3]**2
                                                                      + self._ions.markers[~self._ions.holes, 4]**2
                                                                      + self._ions.markers[~self._ions.holes, 5]**2)/(2. * self._ions.n_mks)

        self.derham.comm.Reduce(
            self._en_f_loc, self._scalar_quantities['en_f'], op=MPI.SUM, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]


class LinearMHDDriftkineticCC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Driftkinetic) equations with **current coupling scheme**. 

    Normalization: 

    .. math::

        \frac{\hat B^2}{\hat \rho \mu_0} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = \hat \rho \, \hat v_\textnormal{A}^2\,.

    Implemented equations:

    CC_LinearMHD_Driftkinetic

    .. math::

        \begin{align}
        \textnormal{linear MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} + (\rho_h \tilde{\mathbf{U}} - \mathbf{J}_{gc} - \nabla \times \mathbf{M}_{gc}) \times (\mathbf{B}_0 + \tilde{\mathbf{B}})
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{Driftkinetic}\qquad& \frac{\partial F_h}{\partial t} + \frac{1}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*)\cdot\frac{\partial F_h}{\partial \mathbf{x}}
        + \frac{q_h}{m_h} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*)\cdot\frac{\partial F_h}{\partial v_\parallel}
        = 0\,,
        \\
        & \rho_h = \int F_h dv_\parallel d\mu \,,
        \\
        & \mathbf{J}_{gc} = q \int F_h \frac{1}{B_\parallel^*} (v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) dv_\parallel d\mu \,,
        \\
        & \mathbf{M}_{gc} = - \int F_h \mu \mathbf{b}_0 dv_\parallel d\mu \,,
        \end{align}

    where

    .. math::

        \begin{align}
        \mathbf{B}^* &= \mathbf{B} + \frac{m_h}{q_h} v_\parallel \nabla \times \mathbf{b}_0 \,,
        \\
        \mathbf{E}^* &= - \tilde{\mathbf{U}} \times \mathbf{B} - \frac{\mu}{q_h} \nabla B_\parallel \,,
        \\
        B_\parallel &= \mathbf{b}_0 \cdot \mathbf{B} \,,
        \\
        B^*_\parallel &= \mathbf{b}_0 \cdot \mathbf{B}^* \,.
        \end{align}
    Parameters
    ----------
        params : dict
            Simulation parameters, see from :ref:`params_yml`.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling

        self._u_space = params['fluid']['mhd']['mhd_u_space']

        if self._u_space == 'Hdiv':
            u_name = 'u2'
        elif self._u_space == 'H1vec':
            u_name = 'uv'
        else:
            raise ValueError(
                f'MHD velocity must be in Hdiv or in H1vec, but has been specified in {self._u_space}.')

        super().__init__(params, comm, b2='Hdiv', mhd={
            'n3': 'L2', u_name: self._u_space, 'p3': 'L2'}, energetic_ions='Particles5D')

        # guiding center asymptotic parameter (rhostar)
        epsilon = self.kinetic['energetic_ions']['plasma_params']['epsilon']

        # pointers to em-field variables
        self._b = self.em_fields['b2']['obj'].vector

        # pointers to fluid variables
        self._n = self.fluid['mhd']['n3']['obj'].vector
        self._u = self.fluid['mhd'][u_name]['obj'].vector
        self._p = self.fluid['mhd']['p3']['obj'].vector

        # pointer to kinetic variables
        self._ions = self.kinetic['energetic_ions']['obj']
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        alfven_solver = params['solvers']['solver_1']
        magnetosonic_solver = params['solvers']['solver_2']
        coupling_solver = params['solvers']['solver_3']
        self._nuh = self.kinetic['energetic_ions']['plasma_params']['n [10^20/m^3]'] / \
            self.fluid['mhd']['plasma_params']['n [10^20/m^3]']

        print('Coupling parameter nu_h = n_h/n = ' + str(self._nuh) + '\n')

        # Project magnetic field
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

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(
            self.derham, self.domain, self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        # self._propagators += [propagators_fields.ShearAlfvén(self._u, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, alfven_solver)]
        # self._propagators += [propagators_fields.Magnetosonic(self._n, self._u, self._p, self._b, self._u_space, self.derham, self._mass_ops, self._basis_ops, magnetosonic_solver)]
        self._propagators += [propagators_markers.StepPushDriftkinetic1(self._ions, self.derham, self.domain, self._basis_ops,
                                                                        epsilon, self._b,
                                                                        [self._b_eq, self._unit_b1,
                                                                            self._unit_b2, self._abs_b],
                                                                        ions_params['push_algos'],
                                                                        ions_params['markers']['bc_type'])]
        self._propagators += [propagators_markers.StepPushDriftkinetic2(self._ions, self.derham, self.domain, self._basis_ops,
                                                                        epsilon, self._b,
                                                                        [self._b_eq, self._unit_b1,
                                                                            self._unit_b2, self._abs_b],
                                                                        ions_params['push_algos'],
                                                                        ions_params['markers']['bc_type'])]
        self._propagators += [propagators_coupling.CurrentCoupling5DCurrent1(self._ions, self.derham, self.domain, self._mass_ops,
                                                                             epsilon, self._u, self._u_space, self._b, [
                                                                                 self._b_eq, self._unit_b1, self._unit_b2, self._abs_b],
                                                                             ions_params['markers']['bc_type'], coupling_solver)]
        # self._propagators += [propagators_coupling.CurrentCoupling5DCurrent2(self._ions, self.derham, self.domain, self._mass_ops, self._basis_ops,
        #                                                                      epsilon, self._u, self._u_space, self._b, self._b_eq, self._unit_b1, self._unit_b2, self._abs_b,
        #                                                                      ions_params['push_algos']['method'],
        #                                                                      ions_params['push_algos']['integrator'],
        #                                                                      ions_params['markers']['bc_type'],
        #                                                                      ions_params['push_algos']['maxiter'],
        #                                                                      ions_params['push_algos']['tol'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_U'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_p'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_fv_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fv'] = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_fB'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        if self._u_space == 'Hcurl':
            self._en_U_loc = self._u.dot(self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M1n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        elif self._u_space == 'Hdiv':
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.M2n.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)
        else:
            self._scalar_quantities['en_U'][0] = self._u.dot(
                self._mass_ops.Mvn.dot(self._u))/2
            self._scalar_quantities['en_p'][0] = self._p.toarray(
            ).sum()/(5/3 - 1)

        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b))/2

        self._en_fv_loc = self._ions.markers[~self._ions.holes, 8].dot(
            self._ions.markers[~self._ions.holes, 3]**2)/(self._ions.n_mks)
        self.derham.comm.Reduce(
            self._en_fv_loc, self._scalar_quantities['en_fv'], op=MPI.SUM, root=0)

        # self._en_fB_loc =  self._ions

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_U'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_p'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fv'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_fB'][0]


#############################
# Kinetic models
#############################
class LinearVlasovMaxwell(StruphyModel):
    r'''The linearized Vlasov Maxwell system is described by the following equations:

    .. math::

        \begin{align}
            \partial_t h + \mathbf{v} \cdot \, & \nabla_\mathbf{x} h + \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \nabla_\mathbf{v} h = \sqrt{f_0} \, \mathbf{E} \cdot \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \,,
            \\
            \frac{\partial \mathbf{E}}{\partial t} & = \nabla \times \mathbf{B} -
            \alpha^2 \int_{\mathbb{R}^3} \sqrt{f_0} \, h \, \left( \mathbb{1}_{\text{th}}^2 (\mathbf{v} - \mathbf{u}) \right) \, \text{d}^3 \mathbf{v} \,,
            \\
            \frac{\partial \mathbf{B}}{\partial t} & = - \nabla \times \mathbf{E} \,,
        \end{align}

    where :math:`f_0` is a Maxwellian background distribution function with constant velocity shift :math:`\mathbf{u}`
    and thermal velocity matrix :math:`\mathbb{1}_{\text{th}} = \text{diag} \left( \frac{1}{v_{\text{th},1}^2}, \frac{1}{v_{\text{th},2}^2}, \frac{1}{v_{\text{th},3}^2} \right)`
    and :math:`h = \frac{\delta f}{\sqrt{f_0}}`.
    These equations form a Hamiltonian system with the energy:

    .. math::

        H(t) = \frac{\alpha^2}{2 \det^2(\mathbb{1}_{\text{th}})} \int_\Omega h^2 \, \text{d}^3 \mathbf{x} \, \text{d}^3 \mathbf{v}
        + \frac{1}{2} \int_\Omega |\mathbf{E}|^2 \, \text{d}^3 \mathbf{x}
        + \frac{1}{2} \int_\Omega |\mathbf{B}|^2 \, \text{d}^3 \mathbf{x} \,.

    Normalization:

    .. math::

        \begin{align}
            c & = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B} \,, &
            \omega & = \Omega_c = \frac{q \hat B}{m} \,, &
            \alpha & = \frac{\Omega_p}{\Omega_c} \,,
        \end{align}

    where :math:`c` is the vacuum speed of light.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling

        super().__init__(params, comm, e_field='Hcurl',
                         b_field='Hdiv', electrons='Particles6D')

        # pointers to em-field variables
        self._e = self.em_fields['e_field']['obj'].vector
        self._b = self.em_fields['b_field']['obj'].vector

        # pointer to electrons
        self._electrons = self.kinetic['electrons']['obj']
        self.electron_params = params['kinetic']['electrons']

        assert self.electron_params['background']['type'] == 'Maxwellian6DUniform', \
            "The background distribution function must be a uniform Maxwellian!"
        self._maxwellian_params = self.electron_params['background']['Maxwellian6DUniform']

        # Get coupling strength
        self.alpha = self.kinetic['electrons']['plasma_params']['alpha']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain)

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
        self._propagators = []

        # Only add StepStaticEfield if efield is non-zero, otherwise it is more expensive
        if np.all(self._e_background[0]._data < 1e-14) and np.all(self._e_background[1]._data < 1e-14) and np.all(self._e_background[2]._data < 1e-14):
            self._propagators += [propagators_markers.StepPushEta(
                self._electrons, self.derham, self.domain,
                self.electron_params['push_algos']['eta'],
                self.electron_params['markers']['bc_type'])]

        else:
            self._propagators += [propagators_markers.StepStaticEfield(
                self._electrons, self.derham, self.domain, self._e_background)]

        self._propagators += [propagators_markers.StepPushVxB(
            self._electrons, self.derham, self.domain, self.electron_params['push_algos']['vxb'], 1., self._b_background)]

        self._propagators += [propagators_coupling.StepEfieldWeights(self.domain, self.derham,
                                                                     self._e, self._electrons, self._mass_ops,
                                                                     self.electron_params['background'], params['solvers']['solver_ew'],
                                                                     self.alpha)]

        self._propagators += [propagators_fields.Maxwell(self._e, self._b,
                                                         self.derham, self._mass_ops, params['solvers']['solver_eb'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_weights'] = np.empty(1, dtype=float)
        self._scalar_quantities['energy'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def set_initial_conditions(self):
        from struphy.kinetic_background import analytical

        super().set_initial_conditions()

        fun_name = 'Maxwellian6DUniform'

        if fun_name in self._maxwellian_params:
            f_bckgr = getattr(analytical, fun_name)(
                **self._maxwellian_params[fun_name])
        else:
            f_bckgr = getattr(analytical, fun_name)()

        # Correct initialization weights by dividing by N*sqrt(f_0)
        self._electrons.markers[~self._electrons.holes, 6] /= \
            (self._electrons.n_mks *
             np.sqrt(f_bckgr(*self._electrons.markers_wo_holes[:, :6].T)))

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        # e^T * M_1 * e
        self._scalar_quantities['en_E'][0] = self._e.dot(
            self._mass_ops.M1.dot(self._e)) / 2.

        # b^T * M_2 * b
        self._scalar_quantities['en_B'][0] = self._b.dot(
            self._mass_ops.M2.dot(self._b)) / 2.

        # alpha^2 * v_th_1^2 * v_th_2^2 * v_th_3^2 * N/2 * sum_p s_0 * w_p^2
        self._scalar_quantities['en_weights'][0] = \
            self.alpha**2 * self._electrons.n_mks / 2. * \
            self._maxwellian_params['vthx']**2 * \
            self._maxwellian_params['vthy']**2 * \
            self._maxwellian_params['vthz']**2 * \
            np.dot(self._electrons.markers_wo_holes[:, 6]**2,
                   self._electrons.markers_wo_holes[:, 7])

        self.derham.comm.Allreduce(
            MPI.IN_PLACE, self._scalar_quantities['en_weights'], op=MPI.SUM)

        # en_tot = en_w + en_e + en_b
        self._scalar_quantities['energy'][0] = \
            self._scalar_quantities['en_weights'][0] + \
            self._scalar_quantities['en_E'][0] + \
            self._scalar_quantities['en_B'][0]


#############################
# Toy models
#############################
class Maxwell(StruphyModel):
    r'''Maxwell's equations in vacuum.

    Normalization:

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

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.propagators import propagators_fields

        super().__init__(params, comm, e_field='Hcurl', b_field='Hdiv')

        # Pointers to em-field variables
        self._e = self.em_fields['e_field']['obj'].vector
        self._b = self.em_fields['b_field']['obj'].vector

        # extract necessary parameters
        solver_params = params['solvers']['solver_1']

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(self.derham, self.domain)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_fields.Maxwell(
            self._e, self._b, self.derham, self._mass_ops, solver_params)]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_E'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time
        self._scalar_quantities['en_E'][0] = .5 * \
            self._e.dot(self._mass_ops.M1.dot(self._e))
        self._scalar_quantities['en_B'][0] = .5 * \
            self._b.dot(self._mass_ops.M2.dot(self._b))
        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0] + \
            self._scalar_quantities['en_B'][0]


class Vlasov(StruphyModel):
    r'''Vlasov equation in static background magnetic field.

    Normalization:

    .. math::

        \mathbf{x}=\mathbf{x}^\prime\hat{L}\,,\quad\mathbf{B}=\mathbf{B}^\prime\hat{B}\,,\quad t=t^\prime\hat{\tau}=t^\prime\left(\frac{e\hat{B}}{m_\textnormal{p}}\right)^{-1}=t^\prime\Omega_\textnormal{pc}^{-1}\,,\quad \mathbf{v}=\mathbf{v}^\prime\frac{\hat{L}}{\hat{\tau}}\,,

    where :math:`\Omega_\textnormal{pc}=e\hat B/m_\textnormal{p}` is the proton cyclotron frequency. Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \frac{\partial f}{\partial \mathbf{x}} + \frac{Z}{A}\left(\mathbf{v}\times\mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,,

    where :math:`Z` and :math:`A` are the charge and mass number, respectively, of the particles species. The characteristic quantities :math:`(\hat{L},\hat{B})` as well as :math:`(Z,A)` need to be defined by the user in the parameter file.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.propagators import propagators_markers

        super().__init__(params, comm, ions='Particles6D')

        # pointer to ions
        ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        print(
            f'Total number of markers : {ions.n_mks}, shape of markers array on rank {self.derham.comm.Get_rank()} : {ions.markers.shape}')

        A = params['model_units']['A']
        Z = params['model_units']['Z']

        # project magnetic background
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepPushVxB(
            ions, self.derham, self.domain, ions_params['push_algos']['vxb'], Z/A, self._b_eq)]
        self._propagators += [propagators_markers.StepPushEta(
            ions, self.derham, self.domain, ions_params['push_algos']['eta'], ions_params['markers']['bc_type'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    @staticmethod
    def print_units(model_units_params):
        print()
        print()
        print('----- MODEL UNITS -------')
        print('x [m]       : ', model_units_params['L'])
        print('B [T]       : ', model_units_params['B'])
        print('t [10⁻⁸ s]  : ', 1.67262192369e-27 /
              (model_units_params['B']*1.602176634e-19)*1e8)
        print('v [10⁷ m/s] : ', model_units_params['L'] /
              (1.67262192369e-27/(model_units_params['B']*1.602176634e-19))*1e-7)
        print()
        print('----- OTHER QUANTITIES ----')
        print('EP gyro period [10⁻⁸ s] : ', 2*np.pi*model_units_params['A']*1.67262192369e-27/(
            model_units_params['Z']*1.602176634e-19*model_units_params['B'])*1e8)

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time


class DriftKinetic(StruphyModel):
    r'''Drift-kinetic equation in static background magnetic field (guiding-center motion). 

    Normalization:

    .. math::

        \hat v = v_{\textnormal{th}}\,,\qquad \hat \omega = v_{\textnormal{th}} \hat k = \Omega_{\textnormal{th}} = \varepsilon\, \Omega_\textnormal{c} \,,

    where :math:`v_{\textnormal{th}} = \sqrt{k_\textnormal{B} T/m}` denotes the thermal velocity, :math:`\Omega_{c}` is the cyclotron frequency and 

    .. math::

        \varepsilon := \frac{\Omega_{\textnormal{th}}}{\Omega_\textnormal{c}} = \hat k \rho_\textnormal{L} \ll 1\,, 

    must be small for the model's validity; :math:`\rho_\textnormal{L} = v_\textnormal{th}/\Omega_\textnormal{c}` stands for the Larmor radius.
    Implemented equations:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \varepsilon \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[ \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, t)` is the guiding center distribution and 

    .. math::

        \mathbf{E}^* = - \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.propagators import propagators_markers

        super().__init__(params, comm, ions='Particles5D')  # TODO:particles.Particles5D

        # pointer to ions
        ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        # guiding center asymptotic parameter (rhostar)
        epsilon = self.kinetic['ions']['plasma_params']['epsilon']

        # project magnetic background
        b = self.derham.P['2']([self.mhd_equil.b2_1,
                                self.mhd_equil.b2_2,
                                self.mhd_equil.b2_3])

        abs_b = self.derham.P['0'](self.mhd_equil.absB0)

        unit_b1 = self.derham.P['1']([self.mhd_equil.unit_b1_1,
                                      self.mhd_equil.unit_b1_2,
                                      self.mhd_equil.unit_b1_3])

        unit_b2 = self.derham.P['2']([self.mhd_equil.unit_b2_1,
                                      self.mhd_equil.unit_b2_2,
                                      self.mhd_equil.unit_b2_3])

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepPushGuidingCenter1(ions, self.derham, self.domain,
                                                                         epsilon, b, unit_b1, unit_b2, abs_b,
                                                                         ions_params['push_algos']['method'],
                                                                         ions_params['push_algos']['integrator'],
                                                                         ions_params['markers']['bc_type'],
                                                                         ions_params['push_algos']['maxiter'],
                                                                         ions_params['push_algos']['tol'])]
        self._propagators += [propagators_markers.StepPushGuidingCenter2(ions, self.derham, self.domain,
                                                                         epsilon, b, unit_b1, unit_b2, abs_b,
                                                                         ions_params['push_algos']['method'],
                                                                         ions_params['push_algos']['integrator'],
                                                                         ions_params['markers']['bc_type'],
                                                                         ions_params['push_algos']['maxiter'],
                                                                         ions_params['push_algos']['tol'])]
        # self._propagators += [propagators_markers.StepPushGuidingCenter(ions, self.derham, self.domain,
        #                                                                 epsilon, b, unit_b1, unit_b2, abs_b,
        #                                                                 ions_params['push_algos']['method'],
        #                                                                 ions_params['push_algos']['integrator'],
        #                                                                 ions_params['markers']['bc_type'],
        #                                                                 ions_params['push_algos']['maxiter'],
        #                                                                 ions_params['push_algos']['tol'])]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time


#############################
# Fluid-kinetic hybrid models
#############################
class Hybrid_fA(StruphyModel):
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

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    '''

    def __init__(self, params, comm):

        from struphy.psydac_api.mass import WeightedMassOperators
        from struphy.psydac_api.basis_projection_ops import BasisProjectionOperators
        from struphy.fields_background.mhd_equil import analytical
        from struphy.propagators import propagators_fields, propagators_markers, propagators_coupling
        from psydac.linalg.stencil import StencilVector, StencilMatrix
        from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

        super().__init__(params, comm, a1='Hcurl', ions='Particles6D')

        # pointers to em-field variables
        self._a = self.em_fields['a1']['obj'].vector

        # pointer to kinetic variables
        self._ions = self.kinetic['ions']['obj']
        ions_params = self.kinetic['ions']['params']

        # extract necessary parameters
        shape_params = params['kinetic']['ions']['ionsshape']

        nqs = [quad_grid.num_quad_pts for quad_grid in self.derham.Vh_fem['0'].quad_grids]
        pts = [quad_grid.points for quad_grid in self.derham.Vh_fem['0'].quad_grids]
        wts = [quad_grid.weights for quad_grid in self.derham.Vh_fem['0'].quad_grids]
        el_indices = [
            quad_grid.indices for quad_grid in self.derham.Vh_fem['0'].quad_grids]
        #basis = [quad_grid.basis          for quad_grid in self.derham.Vh_fem['0'].quad_grids]

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        # Assemble necessary mass matrices
        self._mass_ops = WeightedMassOperators(
            self.derham, self.domain, eq_mhd=self.mhd_equil)

        # Assemble necessary linear basis projection operators
        self._basis_ops = BasisProjectionOperators(
            self.derham, self.domain, self.mhd_equil)

        # Initialize propagators/integrators used in splitting substeps
        self._propagators = []
        self._propagators += [propagators_markers.StepHybridXP(
            self._a, self._ions, self.derham, self.domain, ions_params['markers']['bc_type'])]
        self._propagators += [propagators_markers.StepPushpxB_hybrid(
            self._ions, self.derham, self.domain, ions_params['push_algos']['pxb'], self._a, self._b_eq)]
        self._propagators += [propagators_fields.Hybrid_potential(self._a, 'Hcurl', self._b_eq, self.derham, self._mass_ops,
                                                                  self.domain, self._ions, nqs, np.array(shape_params['degree']), np.array(shape_params['size']))]

        # Scalar variables to be saved during simulation
        self._scalar_quantities['time'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_B'] = np.empty(1, dtype=float)
        self._en_f_loc = np.empty(1, dtype=float)
        self._scalar_quantities['en_f'] = np.empty(1, dtype=float)
        self._scalar_quantities['en_tot'] = np.empty(1, dtype=float)

    @property
    def propagators(self):
        return self._propagators

    def update_scalar_quantities(self, time):
        self._scalar_quantities['time'][0] = time

        self._scalar_quantities['en_B'][0] = self._a.dot(
            self._mass_ops.M1.dot(self._a))/2

        self._en_f_loc = self._ions.markers[~self._ions.holes, 8].dot(self._ions.markers[~self._ions.holes, 3]**2
                                                                      + self._ions.markers[~self._ions.holes, 4]**2
                                                                      + self._ions.markers[~self._ions.holes, 5]**2)/(2. * self._ions.n_mks)

        self.derham.comm.Reduce(
            self._en_f_loc, self._scalar_quantities['en_f'], op=MPI.SUM, root=0)

        self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_B'][0]
        self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_f'][0]
