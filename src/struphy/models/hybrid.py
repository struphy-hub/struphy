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
            bc_type=e_ions_params['markers']['bc_type'],
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
            bc_type=ions_params['markers']['bc_type'],
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
        self.add_propagator(self.prop_fields.ShearAlfvénCurrentCoupling5D(
            self.pointer[self._un],
            self.pointer['b2'],
            particles=self.pointer['energetic_ions'],
            b_eq=self._b_eq,
            f0=f0,
            u_space=self._u_space,
            **solver_params_1,
            **self._coupling_params))
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

        self._en_fv_loc_lost = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
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

        self.pointer['energetic_ions'].save_magnetic_energy(self._derham, self._PBb2)
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

        # # calculate particle magnetic energy
        # self.pointer['energetic_ions'].save_magnetic_energy(self._derham, self._E0T.dot(
        #     self.derham.P['0'](self.mhd_equil.absB0)))

        # self._en_fB_loc = self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 5].dot(
        #     self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 8]) / self.pointer['energetic_ions'].n_mks
        # self.derham.comm.Reduce(
        #     self._en_fB_loc, self._scalar_quantities['en_fB'], op=self._mpi_sum, root=0)

        self.update_scalar('en_tot', en_U + en_p + en_B +
                           self._en_fv_loc[0] + self._en_fv_loc_lost[0] + self._en_fB_loc[0] + self._en_fB_loc_lost[0])

        print('Number of lost markers:', self.pointer['energetic_ions'].n_lost_markers)

# class ColdPlasmaVlasov(StruphyModel):
#     r'''Cold plasma model

#     Normalization:

#     .. math::

#         &c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,,

#         &\hat \omega = \Omega_{ce}\,,

#         &\alpha = \frac{\Omega_{pe}}{\Omega_{ce}}\,,

#         &\hat j_c = \varepsilon_0 \Omega_{pe} \hat E\,,

#         &\hat v = \frac{\Omega_{ce}}{\hat k} = c\,,

#     where :math:`c` is the vacuum speed of light, :math:`\Omega_{ce}` the electron cyclotron frequency,
#     :math:`\Omega_{pe}` the plasma frequency and :math:`\varepsilon_0` the vacuum dielectric constant.
#     Implemented equations:

#     .. math::

#         &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,

#         &-\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
#         \alpha\left(\mathbf j_c + \frac{\hat n_h}{n_0 \left(x\right)} \alpha \mathbf j_h\right)\,,

#         &\frac{\partial \mathbf j_c}{\partial t} = \alpha \mathbf E + \mathbf j_c \times \mathbf B\,,

#         &\frac{\partial f_h}{\partial t} + v \cdot \nabla f_h
#         + \left(\mathbf E + \mathbf v \times B\right) \cdot \nabla_v f_h = 0\,.


#     Parameters
#     ----------
#         params : dict
#             Simulation parameters, see from :ref:`params_yml`.
#     '''

#     def __init__(self, params, comm):

#         from struphy.propagators import propagators_fields
#         from struphy.psydac_api.mass import WeightedMassOperators
#         from struphy.propagators import propagators_fields

#         super().__init__(params, comm, e1='Hcurl', b2='Hdiv',
#                          electron={'j1': 'Hcurl'}, hot_electrons='Particles6D')

#         # pointers to em-fields variables
#         self._e = self.em_fields['e1']['obj'].vector
#         self.pointer['b2'] = self.em_fields['b2']['obj'].vector

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
#         self.add_propagator(self.prop_fields.Maxwell(
#             self._e, self.pointer['b2'], self.derham, self._mass_ops, maxwell_solver)]
#         self.add_propagator(self.prop_fields.OhmCold(
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
#             self.pointer['b2'].dot(self._mass_ops.M2.dot(self.pointer['b2']))
#         self._scalar_quantities['en_tot'][0] = self._scalar_quantities['en_E'][0]
#         self._scalar_quantities['en_tot'][0] += self._scalar_quantities['en_B'][0]
