import numpy as np
from struphy.models.base import StruphyModel


class LinearMHDVlasovCC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **current coupling scheme**.

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat n_\textnormal{h} = \hat n\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD}\,\, &\left\{\,\,
        \begin{aligned}
        &\frac{\partial \tilde{n}}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}}\kappa\left(n_\textnormal{h}\tilde{\mathbf{U}}-n_\textnormal{h}\mathbf{u}_\textnormal{h}\right)\times(\mathbf{B}_0+\tilde{\mathbf{B}}) \color{black}\,,
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
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\textnormal{h} + \kappa\left[\color{blue} (\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \color{black} + \mathbf{v}\times(\mathbf{B}_0+\tilde{\mathbf{B}})\right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\
        &\quad\,\,n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3v\,,\qquad n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3v\,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0` and

    .. math::

        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.
    """

    @classmethod
    def bulk_species(cls):
        return 'mhd'

    @classmethod
    def timescale(cls):
        return 'alfvén'

    def __init__(self, params, comm):

        # choose MHD veclocity space
        u_space = params['fluid']['mhd']['mhd_u_space']

        assert u_space in ['Hdiv', 'H1vec'], \
            f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'

        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'

        self._u_space = u_space
        self._un = 'mhd_' + u_name

        # initialize base class
        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles6D')

        from struphy.polar.basic import PolarVector
        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        e_ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']
        solver_params_4 = params['solvers']['solver_4']

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        kappa = 1. / self.eq_params['energetic_ions']['epsilon_unit']

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa

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

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.CurrentCoupling6DDensity(
            self.pointer[self._un],
            particles=self.pointer['energetic_ions'],
            u_space=self._u_space,
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2'],
            f0=f0,
            **solver_params_1,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.ShearAlfvén(
            self.pointer[self._un],
            self.pointer['b2'],
            u_space=self._u_space,
            **solver_params_2))
        self.add_propagator(self.prop_coupling.CurrentCoupling6DCurrent(
            self.pointer['energetic_ions'],
            self.pointer[self._un],
            u_space=self._u_space,
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2'],
            f0=f0,
            **solver_params_3,
            **self._coupling_params))
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['energetic_ions'],
            algo=e_ions_params['push_algos']['eta'],
            bc_type=e_ions_params['markers']['bc']['type'],
            f0=f0))
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['energetic_ions'],
            algo=e_ions_params['push_algos']['vxb'],
            scale_fac=self._coupling_params['kappa'],
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2'],
            f0=f0))
        self.add_propagator(self.prop_fields.Magnetosonic(
            self.pointer['mhd_n3'],
            self.pointer[self._un],
            self.pointer['mhd_p3'],
            u_space=self._u_space,
            b=self.pointer['b2'],
            **solver_params_4))

        # Scalar variables to be saved during simulation:
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_f')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        if self._u_space == 'Hdiv':
            self._tmp_u = self.derham.Vh['2'].zeros()
        else:
            self._tmp_u = self.derham.Vh['v'].zeros()

        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()
        self._tmp = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    def update_scalar_quantities(self):

        # perturbed fields
        if self._u_space == 'Hdiv':
            self._mass_ops.M2n.dot(self.pointer[self._un], out=self._tmp_u)
        else:
            self._mass_ops.Mvn.dot(self.pointer[self._un], out=self._tmp_u)

        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b1)

        en_U = self.pointer[self._un].dot(self._tmp_u)/2
        en_B = self.pointer['b2'].dot(self._tmp_b1)/2
        en_p = self.pointer['mhd_p3'].dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p)

        # total magnetic field
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b2']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)

        # particles
        self._tmp[0] = self._coupling_params['Ah']/self._coupling_params['Ab']*self.pointer['energetic_ions'].markers_wo_holes[:, 6].dot(
            self.pointer['energetic_ions'].markers_wo_holes[:, 3]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 4]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 5]**2)/(2*self.pointer['energetic_ions'].n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_f', self._tmp[0])
        self.update_scalar('en_tot', en_U + en_B + en_p + self._tmp[0])


class LinearMHDVlasovPC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Full-orbit Vlasov) equations with **pressure coupling scheme**. 

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat{\mathbb{P}}_\textnormal{h} = A_\textnormal{h}m_\textnormal{H}\hat n \hat v_\textnormal{A}^2\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla\cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp} \color{black} 
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
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} \color{blue} + \tilde{\mathbf{U}}_\perp \color{black})\cdot \nabla f_\textnormal{h}
        + \left[\kappa\, \mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{blue}- \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \color{black} \right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\quad\,\,\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_h d\mathbf{v} \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    There is also a version of this model without the :math:`\perp` subscript (can be selected in :code:`parameters.yml`).

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

        assert u_space in ['Hdiv', 'H1vec'], \
            f'MHD velocity must be Hdiv or H1vec, but was specified {self._u_space}.'

        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'

        self._u_space = u_space
        self._un = 'mhd_' + u_name

        # initialize base class
        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles6D')

        from struphy.polar.basic import PolarVector
        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        kappa = 1. / self.eq_params['energetic_ions']['epsilon_unit']

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa

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

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.ShearAlfvén(
            self.pointer[self._un],
            self.pointer['b2'],
            u_space=self._u_space,
            **solver_params_1,))
        self.add_propagator(self.prop_coupling.PressureCoupling6D(
            self.pointer['energetic_ions'],
            self.pointer[self._un],
            u_space=self._u_space,
            use_perp_model=ions_params['use_perp_model'],
            **solver_params_2,
            **self._coupling_params))
        self.add_propagator(self.prop_markers.PushEtaPC(
            self.pointer['energetic_ions'],
            u_mhd=self.pointer[self._un],
            u_space=self._u_space,
            bc_type=ions_params['markers']['bc']['type'],
            use_perp_model=ions_params['use_perp_model']))
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['energetic_ions'],
            algo=ions_params['push_algos']['vxb'],
            scale_fac=self._coupling_params['kappa'],
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2'],
            f0=f0))
        self.add_propagator(self.prop_fields.Magnetosonic(
            self.pointer['mhd_n3'],
            self.pointer[self._un],
            self.pointer['mhd_p3'],
            u_space=self._u_space,
            b=self.pointer['b2'],
            **solver_params_3))

        # Scalar variables to be saved during simulation:
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_f')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        if self._u_space == 'Hdiv':
            self._tmp_u = self.derham.Vh['2'].zeros()
        else:
            self._tmp_u = self.derham.Vh['v'].zeros()

        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()
        self._tmp = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    def update_scalar_quantities(self):

        # perturbed fields
        if self._u_space == 'Hdiv':
            self._mass_ops.M2n.dot(self.pointer[self._un], out=self._tmp_u)
        else:
            self._mass_ops.Mvn.dot(self.pointer[self._un], out=self._tmp_u)

        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b1)

        en_U = self.pointer[self._un].dot(self._tmp_u)/2
        en_B = self.pointer['b2'].dot(self._tmp_b1)/2
        en_p = self.pointer['mhd_p3'].dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p)

        # total magnetic field
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b2']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)

        # particles
        self._tmp[0] = self._coupling_params['Ah']/self._coupling_params['Ab']*self.pointer['energetic_ions'].markers_wo_holes[:, 6].dot(
            self.pointer['energetic_ions'].markers_wo_holes[:, 3]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 4]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 5]**2)/(2*self.pointer['energetic_ions'].n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_f', self._tmp[0])
        self.update_scalar('en_tot', en_U + en_B + en_p + self._tmp[0])


class LinearMHDDriftkineticCC(StruphyModel):
    r'''Hybrid (Linear ideal MHD + Driftkinetic) equations with **current coupling scheme**. 

    :ref:`normalization`: 

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat n \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U = \hat v = \hat u_\textnormal{h} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0} \, \,,\qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad \hat n_\textnormal{h} = \hat n\,.

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 
        \\
        n_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf J_0 \times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}}\kappa \,(n_\textnormal{h} \tilde{\mathbf{U}} - \mathbf{J}_\textnormal{gc} - \frac{1}{\kappa}\nabla \times \mathbf{M}_\textnormal{gc}) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{black}\,,
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        \quad &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*)\cdot\nabla f_\textnormal{h}
        + \kappa \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel}
        = 0\,,
        \\
        & n_\textnormal{h} = \int f_\textnormal{h} B^*_\parallel \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{J}_\textnormal{gc} = q \int f_\textnormal{h} (v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{M}_\textnormal{gc} = - \int f_\textnormal{h} \mu \mathbf{b}_0 B^*_\parallel \,\textnormal dv_\parallel \textnormal d\mu \,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf J_0 = \nabla \times \mathbf B_0` and

    .. math::

        \begin{align}
        \mathbf{B}^* &= \mathbf{B} + \frac{1}{\kappa} v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,,
        \\
        \mathbf{E}^* &= \color{blue} - \tilde{\mathbf{U}} \times \mathbf{B} \color{black} - \frac{1}{\kappa} \mu \nabla B_\parallel \,.
        \end{align}

    Moreover,

    .. math::

        \kappa = 2 \pi \frac{\hat \Omega_{\textnormal{ch}}}{\hat \omega}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.    

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

        if u_space == 'Hdiv':
            u_name = 'u2'
        else:
            u_name = 'uv'

        self._u_space = u_space
        self._un = 'mhd_' + u_name

        # initialize base class
        super().__init__(params, comm,
                         b2='Hdiv',
                         mhd={'n3': 'L2', u_name: self._u_space, 'p3': 'L2'},
                         energetic_ions='Particles5D')

        from struphy.polar.basic import PolarVector
        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        solver_params_1 = params['solvers']['solver_1']
        solver_params_2 = params['solvers']['solver_2']
        solver_params_3 = params['solvers']['solver_3']
        solver_params_4 = params['solvers']['solver_4']
        solver_params_5 = params['solvers']['solver_5']

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        kappa = 1. / self.eq_params['energetic_ions']['epsilon_unit']

        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa

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
        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.E['0'].transpose()
        self._EvT = self.derham.E['v'].transpose()
        self._E2T = self.derham.E['2'].transpose()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # add control variate to mass_ops object
        if control:
            self.mass_ops.weights['f0'] = f0

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.StepPushDriftKinetic1(
            self.pointer['energetic_ions'],
            kappa=kappa,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            integrator=ions_params['push_algos1']['integrator'],
            method=ions_params['push_algos1']['method'],
            maxiter=ions_params['push_algos1']['maxiter'],
            tol=ions_params['push_algos1']['tol']))
        self.add_propagator(self.prop_markers.StepPushDriftKinetic2(
            self.pointer['energetic_ions'],
            kappa=kappa,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            integrator=ions_params['push_algos2']['integrator'],
            method=ions_params['push_algos2']['method'],
            maxiter=ions_params['push_algos2']['maxiter'],
            tol=ions_params['push_algos2']['tol']))
        self.add_propagator(self.prop_coupling.CurrentCoupling5DCurrent2(
            self.pointer['energetic_ions'],
            self.pointer[self._un],
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._abs_b,
            f0=f0,
            u_space=self._u_space,
            **solver_params_4,
            **self._coupling_params,
            integrator='explicit',
            method='rk4'))
        self.add_propagator(self.prop_coupling.CurrentCoupling5DCurrent1(
            self.pointer['energetic_ions'],
            self.pointer[self._un],
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            f0=f0,
            u_space=self._u_space,
            **solver_params_3,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.ShearAlfvénCurrentCoupling5D(
            self.pointer[self._un],
            self.pointer['b2'],
            particles=self.pointer['energetic_ions'],
            b_eq=self._b_eq,
            f0=f0,
            u_space=self._u_space,
            **solver_params_1,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.MagnetosonicCurrentCoupling5D(
            self.pointer['mhd_n3'],
            self.pointer[self._un],
            self.pointer['mhd_p3'],
            b=self.pointer['b2'],
            particles=self.pointer['energetic_ions'],
            unit_b1=self._unit_b1,
            f0=f0,
            u_space=self._u_space,
            **solver_params_2,
            **self._coupling_params))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_fv')
        self.add_scalar('en_fB')
        self.add_scalar('en_fv_lost')
        self.add_scalar('en_fB_lost')
        self.add_scalar('en_tot')

        # things needed in update_scalar_quantities
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._PBb1 = self._abs_b.space.zeros()
        self._PBb2 = self._E0T.codomain.zeros()

        self._en_fv_loc = np.empty(1, dtype=float)
        self._en_fB_loc = np.empty(1, dtype=float)
        self._en_fv_loc_lost = np.empty(1, dtype=float)
        self._en_fB_loc_lost = np.empty(1, dtype=float)

        if self._u_space == 'Hcurl':
            self._tmp_u = self.derham.Vh['1'].zeros()
        elif self._u_space == 'Hdiv':
            self._tmp_u = self.derham.Vh['2'].zeros()
        else:
            self._tmp_u = self.derham.Vh['v'].zeros()

        self._tmp_b = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):

        if self._u_space == 'Hcurl':
            self._mass_ops.M1n.dot(self.pointer[self._un], out=self._tmp_u)
            en_U = self.pointer[self._un].dot(self._tmp_u)/2
        elif self._u_space == 'Hdiv':
            self._mass_ops.M2n.dot(self.pointer[self._un], out=self._tmp_u)
            en_U = self.pointer[self._un].dot(self._tmp_u)/2
        else:
            self._mass_ops.Mvn.dot(self.pointer[self._un], out=self._tmp_u)
            en_U = self.pointer[self._un].dot(self._tmp_u)/2

        en_p = self.pointer['mhd_p3'].toarray().sum()/(5/3 - 1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b)
        en_B = self.pointer['b2'].dot(self._tmp_b)/2

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_p', en_p)
        self.update_scalar('en_B', en_B)

        # self._scalar_quantities['en_p_eq'][0] = self._p_eq.dot(
        #     self._ones)/(5/3 - 1)
        # self._scalar_quantities['en_B_eq'][0] = self._b_eq.dot(
        #     self._mass_ops.M2.dot(self._b_eq, apply_bc=False))/2

        # calculate particle kinetic energy
        self._en_fv_loc[0] = self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 5].dot(
            self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 3]**2) / (2*self.pointer['energetic_ions'].n_mks)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv_loc, op=self._mpi_sum)

        self.update_scalar('en_fv', self._en_fv_loc[0])

        self._en_fv_loc_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
            self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 3]**2) / (2.*self.pointer['energetic_ions'].n_mks)
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv_loc_lost, op=self._mpi_sum)

        self.update_scalar('en_fv_lost', self._en_fv_loc_lost[0])

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)
        self._b_full1 += self.pointer['b2']
        self._b_full1.update_ghost_regions()

        # self._scalar_quantities['en_B_tot'][0] = (self._b_full1).dot(
        #     self._mass_ops.M2.dot(self._b_full1, apply_bc=False))/2.

        # absolute value of parallel magnetic field
        self._prop.basis_ops.PB.dot(self._b_full1, out=self._PBb1)
        self._E0T.dot(self._PBb1, out=self._PBb2)
        self._PBb2.update_ghost_regions()

        self.pointer['energetic_ions'].save_magnetic_energy(
            self._derham, self._PBb2)
        self._en_fB_loc[0] = self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 5].dot(
            self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 8])/self.pointer['energetic_ions'].n_mks
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB_loc, op=self._mpi_sum)

        self.update_scalar('en_fB', self._en_fB_loc[0])

        self._en_fB_loc_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
            self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 8]) / self.pointer['energetic_ions'].n_mks
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB_loc_lost, op=self._mpi_sum)

        self.update_scalar('en_fB_lost', self._en_fB_loc_lost[0])

        self.update_scalar('en_tot', en_U + en_p + en_B +
                           self._en_fv_loc[0] + self._en_fv_loc_lost[0] + self._en_fB_loc[0] + self._en_fB_loc_lost[0])


class ColdPlasmaVlasov(StruphyModel):
    r'''Cold plasma hybrid model

    Normalization:

    .. math::

        &c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,, \quad \alpha = \frac{\hat \Omega_\textnormal{pc}}{\hat \Omega_\textnormal{cc}}\,, \quad \varepsilon_\textnormal{c} = \frac{\hat{\omega}}{\hat \Omega_\textnormal{cc}}\,, \quad \varepsilon_\textnormal{h} = \frac{\hat{\omega}}{\hat \Omega_\textnormal{ch}}\,,

        &\hat j_\textnormal{c} = q_\textnormal{c} c \hat n_\textnormal{c}\,, \quad \hat j_\textnormal{h} = q_\textnormal{h} c \hat n_\textnormal{c}\,, \quad \hat f = \frac{\hat n_\textnormal{c}}{c^3} \,, \quad \nu = \frac{q_\textnormal{h}}{q_\textnormal{c}}\,,

    where :math:`c` is the vacuum speed of light, :math:`\hat \Omega_\textnormal{cc}` the cold electron cyclotron frequency,
    :math:`\hat \Omega_\textnormal{pc}` the cold electron plasma frequency,  :math:`\hat \Omega_\textnormal{ch}` the hot electron cyclotron frequency,
    and :math:`\hat \Omega_\textnormal{ph}` the hot electron plasma frequency.
    Implemented equations:

    .. math::

        &\partial_t f + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon_\textnormal{h}}\left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right)
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,

        &-\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon_\textnormal{c}} \left( \mathbf j_\textnormal{c} + \nu  \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right) \,,

        &\frac{1}{n_0} \frac{\partial \mathbf j_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf E + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf j_\textnormal{c} \times \mathbf B_0\,.

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \nu \frac{\alpha^2}{\varepsilon_\textnormal{c}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}
        \end{align}

    Parameters
    ----------
    params : dict
        Simulation parameters, see from :ref:`params_yml`.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.

    Note
    ----------
    If hot and cold particles are of the same species (:math:`Z_\textnormal{c} = Z_\textnormal{h} \,, A_\textnormal{c} = A_\textnormal{h}`) then :math:`\varepsilon_\textnormal{c} = \varepsilon_\textnormal{h}` and :math:`\nu = 1`.
    '''

    @classmethod
    def bulk_species(cls):
        return 'coldelectrons'

    @classmethod
    def timescale(cls):
        return 'light'

    def __init__(self, params, comm):

        super().__init__(params, comm, e1='Hcurl', b2='Hdiv',
                         coldelectrons={'j1': 'Hcurl'}, hotelectrons='Particles6D')

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        electron_params = params['kinetic']['hotelectrons']

        # get poisson solver parameters
        self._poisson_params = params['solvers']['solver_poisson']

        # model parameters
        self._alpha = np.abs(self.eq_params['coldelectrons']['alpha_unit'])
        self._epsilon_cold = self.eq_params['coldelectrons']['epsilon_unit']
        self._epsilon_hot = self.eq_params['hotelectrons']['epsilon_unit']

        self._nu = electron_params['phys_params']['Z'] / \
            params['fluid']['coldelectrons']['phys_params']['Z']

        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params['solvers']['solver_maxwell']))
        self.add_propagator(self.prop_fields.OhmCold(
            self.pointer['coldelectrons_j1'],
            self.pointer['e1'],
            **params['solvers']['solver_ohmcold'],
            alpha=self._alpha,
            epsilon=self._epsilon_cold))
        self.add_propagator(self.prop_fields.JxBCold(
            self.pointer['coldelectrons_j1'],
            **params['solvers']['solver_jxbcold'],
            alpha=self._alpha,
            epsilon=self._epsilon_cold))
        self.add_propagator(self.prop_markers.PushEta(
            self.pointer['hotelectrons'],
            algo=electron_params['push_algos']['eta'],
            bc_type=electron_params['markers']['bc']['type'],
            f0=None))
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['hotelectrons'],
            algo=electron_params['push_algos']['vxb'],
            scale_fac=1/self._epsilon_cold,
            b_eq=self._b_background,
            b_tilde=self.pointer['b2'],
            f0=None))
        self.add_propagator(self.prop_coupling.VlasovMaxwell(
            self.pointer['e1'],
            self.pointer['hotelectrons'],
            c1=self._nu * self._alpha**2/self._epsilon_cold,
            c2=1/self._epsilon_hot,
            **params['solvers']['solver_vlasovmaxwell']))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_E')
        self.add_scalar('en_B')
        self.add_scalar('en_J')
        self.add_scalar('en_f')
        self.add_scalar('en_tot')

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # temporaries
        self._tmp1 = self.pointer['e1'].space.zeros()
        self._tmp2 = self.pointer['b2'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):

        from struphy.pic.particles_to_grid import AccumulatorVector
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "vlasov_maxwell_poisson")
        charge_accum.accumulate(self.pointer['hotelectrons'])

        # Locally subtract mean charge for solvability with periodic bc
        if np.all(charge_accum.vectors[0].space.periods):
            charge_accum._vectors[0][:] -= np.mean(charge_accum.vectors[0].toarray()[
                                                   charge_accum.vectors[0].toarray() != 0])

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh['0'])
        poisson_solver = self.prop_fields.ImplicitDiffusion(
            _phi,
            sigma=0,
            phi_n=self._nu * self._alpha**2 /
            self._epsilon_cold * charge_accum.vectors[0],
            x0=self._nu * self._alpha**2 /
            self._epsilon_cold * charge_accum.vectors[0],
            **self._poisson_params)

        # Solve with dt=1. and compute electric field
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e1'])

    def update_scalar_quantities(self):

        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = .5 * self.pointer['e1'].dot(self._tmp1)
        en_B = .5 * self.pointer['b2'].dot(self._tmp2)
        self._mass_ops.M1ninv.dot(
            self.pointer['coldelectrons_j1'], out=self._tmp1)
        en_J = .5 * self._alpha**2 * \
            self.pointer['coldelectrons_j1'].dot(self._tmp1)
        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_J', en_J)

        # nu alpha^2 eps_h / eps_c / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self._nu * self._alpha**2 * self._epsilon_hot / self._epsilon_cold / \
            (2 * self.pointer['hotelectrons'].n_mks) * np.dot(self.pointer['hotelectrons'].markers_wo_holes[:, 3]**2 + self.pointer['hotelectrons'].markers_wo_holes[:, 4]
                                                              ** 2 + self.pointer['hotelectrons'].markers_wo_holes[:, 5]**2, self.pointer['hotelectrons'].markers_wo_holes[:, 6])
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_f', self._tmp[0])

        # en_tot = en_E + en_B + en_J + en_w
        self.update_scalar('en_tot', en_E + en_B + en_J + self._tmp[0])
