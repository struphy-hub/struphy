import numpy as np
from struphy.models.base import StruphyModel

from struphy.propagators import propagators_fields, propagators_coupling, propagators_markers


class LinearMHDVlasovCC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **current coupling scheme**.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v = \hat v_\textnormal{A} \,, \qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
        \textnormal{MHD}\,\, &\left\{\,\,
        \begin{aligned}
        &\frac{\partial \tilde{\rho}}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\[2mm]
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p 
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \left(n_\textnormal{h}\tilde{\mathbf{U}}-n_\textnormal{h}\mathbf{u}_\textnormal{h}\right)\times(\mathbf{B}_0+\tilde{\mathbf{B}}) \color{black}\,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + (\gamma-1)\nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)\,,\qquad \nabla\cdot\tilde{\mathbf{B}}=0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\textnormal{h} + \frac{1}{\varepsilon} \left[\color{blue} (\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \color{black} + \mathbf{v}\times(\mathbf{B}_0+\tilde{\mathbf{B}})\right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &\quad\,\,n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,,\qquad n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3 \mathbf v\,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0` and

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c,hot}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c,hot}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.CurrentCoupling6DDensity`
    2. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
    3. :class:`~struphy.propagators.propagators_coupling.CurrentCoupling6DCurrent`
    4. :class:`~struphy.propagators.propagators_markers.PushEta`
    5. :class:`~struphy.propagators.propagators_markers.PushVxB`
    6. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b_field'] = 'Hdiv'
        dct['fluid']['mhd'] = {'rho': 'L2', 'u': 'Hdiv', 'p': 'L2'}
        dct['kinetic']['energetic_ions'] = 'Particles6D'
        return dct

    @staticmethod
    def bulk_species():
        return 'mhd'

    @staticmethod
    def velocity_scale():
        return 'alfvén'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.CurrentCoupling6DDensity: ['mhd_u'],
                propagators_fields.ShearAlfven: ['mhd_u', 'b_field'],
                propagators_coupling.CurrentCoupling6DCurrent: ['energetic_ions', 'mhd_u'],
                propagators_markers.PushEta: ['energetic_ions'],
                propagators_markers.PushVxB: ['energetic_ions'],
                propagators_fields.Magnetosonic: ['mhd_rho', 'mhd_u', 'mhd_p']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector
        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        e_ions_params = self.kinetic['energetic_ions']['params']
        u_space = 'Hdiv'

        # extract necessary parameters
        params_shear_alfven = params['fluid']['mhd']['options']['ShearAlfven']['solver']
        params_magnetosonic = params['fluid']['mhd']['options']['Magnetosonic']['solver']
        algo_eta = params['kinetic']['energetic_ions']['options']['PushEta']['algo']
        algo_vxb = params['kinetic']['energetic_ions']['options']['PushVxB']['algo']
        params_density = params['fluid']['mhd']['options']['CurrentCoupling6DDensity']['solver']
        params_current = params['kinetic']['energetic_ions']['options']['CurrentCoupling6DCurrent']['solver']

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        epsilon = self.equation_params['energetic_ions']['epsilon']

        if abs(epsilon - 1) < 1e-6:
            epsilon = 1.

        self._Ab = Ab
        self._Ah = Ah

        # add control variate to mass_ops object
        if self.pointer['energetic_ions'].control_variate:
            self.mass_ops.weights['f0'] = self.pointer['energetic_ions'].f0

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

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.CurrentCoupling6DDensity] = {'particles': self.pointer['energetic_ions'],
                                                                     'u_space': u_space,
                                                                     'b_eq': self._b_eq,
                                                                     'b_tilde': self.pointer['b_field'],
                                                                     'Ab': Ab,
                                                                     'Ah': Ah,
                                                                     'epsilon': epsilon,
                                                                     'solver': params_density}

        self._kwargs[propagators_fields.ShearAlfven] = {'u_space': u_space,
                                                        'solver': params_shear_alfven}

        self._kwargs[propagators_coupling.CurrentCoupling6DCurrent] = {'u_space': u_space,
                                                                       'b_eq': self._b_eq,
                                                                       'b_tilde': self.pointer['b_field'],
                                                                       'Ab': Ab,
                                                                       'Ah': Ah,
                                                                       'epsilon': epsilon,
                                                                       'solver': params_current}

        self._kwargs[propagators_markers.PushEta] = {'algo': algo_eta,
                                                     'bc_type': e_ions_params['markers']['bc']['type']}

        self._kwargs[propagators_markers.PushVxB] = {'algo': algo_vxb,
                                                     'scale_fac': 1./epsilon,
                                                     'b_eq': self._b_eq,
                                                     'b_tilde': self.pointer['b_field']}

        self._kwargs[propagators_fields.Magnetosonic] = {'u_space': u_space,
                                                         'b': self.pointer['b_field'],
                                                         'solver': params_magnetosonic}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation:
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_f')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities:
        self._tmp_u = self.derham.Vh['2'].zeros()

        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()
        self._tmp = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    def update_scalar_quantities(self):

        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_u'], out=self._tmp_u)
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u'].dot(self._tmp_u)/2
        en_B = self.pointer['b_field'].dot(self._tmp_b1)/2
        en_p = self.pointer['mhd_p'].dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p)

        # total magnetic field
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b_field']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)

        # particles
        self._tmp[0] = self._Ah/self._Ab*self.pointer['energetic_ions'].markers_wo_holes[:, 6].dot(
            self.pointer['energetic_ions'].markers_wo_holes[:, 3]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 4]**2 +
            self.pointer['energetic_ions'].markers_wo_holes[:, 5]**2)/(2*self.pointer['energetic_ions'].n_mks)

        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)

        self.update_scalar('en_f', self._tmp[0])
        self.update_scalar('en_tot', en_U + en_B + en_p + self._tmp[0])


class LinearMHDVlasovPC(StruphyModel):
    r'''
    Hybrid linear MHD + energetic ions (6D Vlasov) with **pressure coupling scheme**.

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
        + \left[\frac{1}{\epsilon}\, \mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{blue}- \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \color{black} \right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\quad\,\,\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_h d\mathbf{v} \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \epsilon = \frac{\hat \omega}{2 \pi \, \hat \Omega_{\textnormal{ch}}} \,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{ch}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    There is also a version of this model without the :math:`\perp` subscript (can be selected in :code:`parameters.yml`).

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

        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'n3': 'L2', 'u2': 'Hdiv', 'p3': 'L2'}
        dct['kinetic']['energetic_ions'] = 'Particles6D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'mhd'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ShearAlfven, Magnetosonic
        from struphy.propagators.propagators_markers import PushEtaPC, PushVxB
        from struphy.propagators.propagators_coupling import PressureCoupling6D

        dct = {}
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'shear_alfven'],
                       option=ShearAlfven.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'magnetosonic'],
                       option=Magnetosonic.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['use_perp_model'],
                       option=PushEtaPC.options()['use_perp_model'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['push_vxb'],
                       option=PushVxB.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['solver'],
                       option=PressureCoupling6D.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector
        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # prelim
        ions_params = self.kinetic['energetic_ions']['params']

        # extract necessary parameters
        params_shear_alfven = params['fluid']['mhd']['options']['solvers']['shear_alfven']
        params_magnetosonic = params['fluid']['mhd']['options']['solvers']['magnetosonic']
        use_perp_model = params['kinetic']['energetic_ions']['options']['use_perp_model']
        algo_vxb = params['kinetic']['energetic_ions']['options']['push_vxb']
        params_pc_solver = params['kinetic']['energetic_ions']['options']['solver']

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        kappa = 1. / self.equation_params['energetic_ions']['epsilon']

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah
        self._coupling_params['kappa'] = kappa

        # add control variate to mass_ops object
        if self.pointer['energetic_ions'].control_variate:
            self.mass_ops.weights['f0'] = self.pointer['energetic_ions'].f0

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

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.ShearAlfven(
            self.pointer['mhd_u2'],
            self.pointer['b2'],
            u_space='Hdiv',
            solver=params_shear_alfven,))
        self.add_propagator(self.prop_coupling.PressureCoupling6D(
            self.pointer['energetic_ions'],
            self.pointer['mhd_u2'],
            u_space='Hdiv',
            use_perp_model=use_perp_model,
            **params_pc_solver,
            **self._coupling_params))
        self.add_propagator(self.prop_markers.PushEtaPC(
            self.pointer['energetic_ions'],
            u_mhd=self.pointer['mhd_u2'],
            u_space='Hdiv',
            bc_type=ions_params['markers']['bc']['type'],
            use_perp_model=use_perp_model))
        self.add_propagator(self.prop_markers.PushVxB(
            self.pointer['energetic_ions'],
            algo=algo_vxb,
            scale_fac=self._coupling_params['kappa'],
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2']))
        self.add_propagator(self.prop_fields.Magnetosonic(
            self.pointer['mhd_n3'],
            self.pointer['mhd_u2'],
            self.pointer['mhd_p3'],
            u_space='Hdiv',
            b=self.pointer['b2'],
            solver=params_magnetosonic))

        # Scalar variables to be saved during simulation:
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_f')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_u = self.derham.Vh['2'].zeros()
        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()
        self._tmp = np.empty(1, dtype=float)

        # MPI operations needed for scalar variables
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    def update_scalar_quantities(self):

        # perturbed fields
        if 'Hdiv' == 'Hdiv':
            self._mass_ops.M2n.dot(self.pointer['mhd_u2'], out=self._tmp_u)
        else:
            self._mass_ops.Mvn.dot(self.pointer['mhd_u2'], out=self._tmp_u)

        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u2'].dot(self._tmp_u)/2
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
    r"""
    Hybrid linear MHD + energetic ions (5D Driftkinetic) with **current coupling scheme**. 

    :ref:`normalization`: 

    .. math::

        \hat U = \hat v_\textnormal{h} =: \hat v_\textnormal{A, bulk} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0} \, \qquad
        \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A,bulk} \hat \mu \hat B} \,,\qquad \hat \mu = \frac{A_\textnormal{h} m_\textnormal{H} \hat v_\textnormal{A,bulk}^2}{\hat B} \,.

    Implemented equations: find :math:`(\tilde n, \tilde{\mathbf{U}}, \tilde p, \tilde{\mathbf{B}}, f_\textnormal{h}) \in L^2 \times H(\textrm{div}) \times L^2 \times H(\textrm{div}) \times C^\infty` such that

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_{0} \tilde{\mathbf{U}})=0\,, 
        \\
        \int n_{0} &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \tilde{\mathbf V}\, \textnormal{d}\mathbf{x} - \int \tilde p\, \nabla \cdot \tilde{\mathbf{V}} \,\textrm d \mathbf x
         = \int \tilde{\mathbf{B}} \cdot \nabla \times(\mathbf{B}_0 \times \tilde{\mathbf V})\, \textnormal{d}\mathbf{x} + \int (\nabla \times \mathbf B_0) \times \tilde{\mathbf{B}} \cdot \tilde{\mathbf V} \, \textnormal{d}\mathbf{x}
        \\
        &\qquad \qquad \color{blue}+ \frac{A_\textnormal{h}}{A_\textnormal{b}} \int \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} \times \mathbf{B} \cdot \tilde{\mathbf V} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} \times \mathbf{B} \cdot \tilde{\mathbf V} -\mathbf{M}_\textnormal{gc} \cdot \nabla \times (\mathbf{B} \times \tilde{\mathbf V}) \right] \textnormal{d} \mathbf{x} \color{black} \,, \quad \forall \ \tilde{\mathbf V} \in H(\text{div})\,,
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
        + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel}
        = 0\,,
        \\
        & n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{J}_\textnormal{gc} = \int f_\textnormal{h}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{M}_\textnormal{gc} = - \int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \,\textnormal dv_\parallel \textnormal d\mu \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \begin{align}
        \mathbf{B}^* &= \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,,
        \\[2mm]
        \mathbf{E}^* &=  \color{blue}- \tilde{\mathbf{U}} \times \mathbf{B} \color{black} - \epsilon \mu \nabla B_\parallel \,,
        \end{align}

    and with the normalization parameter 

    .. math::

        \epsilon = \frac{1}{\hat \Omega_\textnormal{ch} \hat t} \,, \qquad \hat \Omega_\textnormal{ch} = \frac{Z_\textnormal{h} e \hat B}{A_\textnormal{h} m_\textnormal{H}} \,.

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

        dct['em_fields']['b2'] = 'Hdiv'
        dct['fluid']['mhd'] = {'n3': 'L2', 'u2': 'Hdiv', 'p3': 'L2'}
        dct['kinetic']['energetic_ions'] = 'Particles5D'
        return dct

    @classmethod
    def bulk_species(cls):
        return 'mhd'

    @classmethod
    def velocity_scale(cls):
        return 'alfvén'

    @classmethod
    def options(cls):
        # import propagator options
        from struphy.propagators.propagators_fields import ShearAlfvenCurrentCoupling5D, MagnetosonicCurrentCoupling5D, CurrentCoupling5DDensity
        from struphy.propagators.propagators_markers import PushDriftKineticbxGradB, PushDriftKineticParallelZeroEfield
        from struphy.propagators.propagators_coupling import CurrentCoupling5DCurlb, CurrentCoupling5DGradB

        dct = {}
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'shear_alfven'],
                       option=ShearAlfvenCurrentCoupling5D.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'magnetosonic'],
                       option=MagnetosonicCurrentCoupling5D.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['solvers', 'density'],
                       option=CurrentCoupling5DDensity.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['algos', 'push_bxgradb'],
                       option=PushDriftKineticbxGradB.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['algos', 'push_bstar'],
                       option=PushDriftKineticParallelZeroEfield.options()['algo'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['solvers', 'cc1'],
                       option=CurrentCoupling5DCurlb.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['solvers', 'cc2'],
                       option=CurrentCoupling5DGradB.options()['solver'], dct=dct)
        cls.add_option(species=['kinetic', 'energetic_ions'], key=['algos', 'push_cc2'],
                       option=CurrentCoupling5DGradB.options()['algo'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector
        from struphy.kinetic_background import maxwellians as kin_ana
        from mpi4py.MPI import SUM, IN_PLACE

        # compute coupling parameters
        Ab = params['fluid']['mhd']['phys_params']['A']
        Ah = params['kinetic']['energetic_ions']['phys_params']['A']
        epsilon = self.equation_params['energetic_ions']['epsilon']

        self._coupling_params = {}
        self._coupling_params['Ab'] = Ab
        self._coupling_params['Ah'] = Ah

        # add control variate to mass_ops object
        if self.pointer['energetic_ions'].control_variate:
            self.mass_ops.weights['f0'] = self.pointer['energetic_ions'].f0

        # Project magnetic field
        self._b_eq = self.derham.P['2']([self.mhd_equil.b2_1,
                                         self.mhd_equil.b2_2,
                                         self.mhd_equil.b2_3])

        self._absB0 = self.derham.P['0'](self.mhd_equil.absB0)

        self._unit_b1 = self.derham.P['1']([self.mhd_equil.unit_b1_1,
                                           self.mhd_equil.unit_b1_2,
                                           self.mhd_equil.unit_b1_3])

        self._unit_b2 = self.derham.P['2']([self.mhd_equil.unit_b2_1,
                                           self.mhd_equil.unit_b2_2,
                                           self.mhd_equil.unit_b2_3])

        self._gradB1 = self.derham.P['1']([self.mhd_equil.gradB1_1,
                                           self.mhd_equil.gradB1_2,
                                           self.mhd_equil.gradB1_3])

        self._curl_unit_b2 = self.derham.P['2']([self.mhd_equil.curl_unit_b2_1,
                                                 self.mhd_equil.curl_unit_b2_2,
                                                 self.mhd_equil.curl_unit_b2_3])

        self._p_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # propagator parameters
        params_shear_alfven = params['fluid']['mhd']['options']['solvers']['shear_alfven']
        params_magnetosonic = params['fluid']['mhd']['options']['solvers']['magnetosonic']
        params_density = params['kinetic']['energetic_ions']['options']['solvers']['density']
        algo_bxgradb = params['kinetic']['energetic_ions']['options']['algos']['push_bxgradb']
        algo_bstar = params['kinetic']['energetic_ions']['options']['algos']['push_bstar']
        params_cc1 = params['kinetic']['energetic_ions']['options']['solvers']['cc1']
        params_cc2 = params['kinetic']['energetic_ions']['options']['solvers']['cc2']
        algo_cc2 = params['kinetic']['energetic_ions']['options']['algos']['push_cc2']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_markers.PushDriftKineticbxGradB(
            self.pointer['energetic_ions'],
            epsilon=epsilon,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._absB0,
            gradB1=self._gradB1,
            curl_unit_b2=self._curl_unit_b2,
            **algo_bxgradb))
        self.add_propagator(self.prop_markers.PushDriftKineticParallelZeroEfield(
            self.pointer['energetic_ions'],
            epsilon=epsilon,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            abs_b=self._absB0,
            gradB1=self._gradB1,
            curl_unit_b2=self._curl_unit_b2,
            **algo_bstar))
        self.add_propagator(self.prop_coupling.CurrentCoupling5DGradB(
            self.pointer['energetic_ions'],
            self.pointer['mhd_u2'],
            epsilon=epsilon,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            unit_b1=self._unit_b1,
            unit_b2=self._unit_b2,
            absB0=self._absB0,
            gradB1=self._gradB1,
            curl_unit_b2=self._curl_unit_b2,
            u_space='Hdiv',
            **params_cc2,
            **self._coupling_params,
            method=algo_cc2))
        self.add_propagator(self.prop_coupling.CurrentCoupling5DCurlb(
            self.pointer['energetic_ions'],
            self.pointer['mhd_u2'],
            epsilon=epsilon,
            b=self.pointer['b2'],
            b_eq=self._b_eq,
            absB0=self._absB0,
            unit_b1=self._unit_b1,
            curl_unit_b2=self._curl_unit_b2,
            u_space='Hdiv',
            **params_cc1,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.CurrentCoupling5DDensity(
            self.pointer['mhd_u2'],
            particles=self.pointer['energetic_ions'],
            epsilon=epsilon,
            u_space='Hdiv',
            b_eq=self._b_eq,
            b_tilde=self.pointer['b2'],
            unit_b1=self._unit_b1,
            curl_unit_b2=self._curl_unit_b2,
            **params_density,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.ShearAlfvenCurrentCoupling5D(
            self.pointer['mhd_u2'],
            self.pointer['b2'],
            particles=self.pointer['energetic_ions'],
            unit_b1=self._unit_b1,
            absB0=self._absB0,
            u_space='Hdiv',
            **params_shear_alfven,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.MagnetosonicCurrentCoupling5D(
            self.pointer['mhd_n3'],
            self.pointer['mhd_u2'],
            self.pointer['mhd_p3'],
            b=self.pointer['b2'],
            particles=self.pointer['energetic_ions'],
            unit_b1=self._unit_b1,
            absB0=self._absB0,
            u_space='Hdiv',
            **params_magnetosonic,
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
        self._PBb = self._absB0.space.zeros()

        self._en_fv = np.empty(1, dtype=float)
        self._en_fB = np.empty(1, dtype=float)
        self._en_fv_lost = np.empty(1, dtype=float)
        self._en_fB_lost = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

        self._tmp_u = self.derham.Vh['2'].zeros()
        self._tmp_b = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):

        self._mass_ops.M2n.dot(self.pointer['mhd_u2'], out=self._tmp_u)
        en_U = self.pointer['mhd_u2'].dot(self._tmp_u)/2

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
        self._en_fv[0] = self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 5].dot(
            self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 3]**2) / (2*self.pointer['energetic_ions'].n_mks)*self._coupling_params['Ah']/self._coupling_params['Ab']
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv, op=self._mpi_sum)

        self.update_scalar('en_fv', self._en_fv[0])

        self._en_fv_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
            self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 3]**2) / (2.*self.pointer['energetic_ions'].n_mks)*self._coupling_params['Ah']/self._coupling_params['Ab']
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fv_lost, op=self._mpi_sum)

        self.update_scalar('en_fv_lost', self._en_fv_lost[0])

        # calculate particle magnetic energy
        self.pointer['energetic_ions'].save_magnetic_energy(self.pointer['b2'])

        self._en_fB[0] = self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 5].dot(
            self.pointer['energetic_ions'].markers[~self.pointer['energetic_ions'].holes, 8])/self.pointer['energetic_ions'].n_mks*self._coupling_params['Ah']/self._coupling_params['Ab']
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB, op=self._mpi_sum)

        self.update_scalar('en_fB', self._en_fB[0])

        self._en_fB_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
            self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 8]) / self.pointer['energetic_ions'].n_mks*self._coupling_params['Ah']/self._coupling_params['Ab']
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._en_fB_lost, op=self._mpi_sum)

        self.update_scalar('en_fB_lost', self._en_fB_lost[0])

        # self.update_scalar('en_tot', en_U + en_p + en_B +
        #                    self._en_fv[0] + self._en_fv_lost[0] + self._en_fB[0] + self._en_fB_lost[0])

        self.update_scalar('en_tot', en_U + en_B + en_p +
                           self._en_fv[0] + self._en_fB[0])

        self._n_lost_particles[0] = self.pointer['energetic_ions'].n_lost_markers
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._n_lost_particles, op=self._mpi_sum)

        if self.derham.comm.Get_rank() == 0:
            print('ratio of lost particles: ',
                  self._n_lost_particles[0]/self.pointer['energetic_ions'].n_mks*100, '%')


class ColdPlasmaVlasov(StruphyModel):
    r'''Cold plasma hybrid model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,,\qquad \hat f = \frac{\hat n}{c^3} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon_\textnormal{h}}\Big[ \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        \frac{1}{n_0} &\frac{\partial \mathbf j_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf E + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf j_\textnormal{c} \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon_\textnormal{c}} \left( \mathbf j_\textnormal{c} + \nu  \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right) \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p,cold}}{\hat \Omega_\textnormal{c,cold}}\,, \qquad \varepsilon_\textnormal{c} = \frac{1}{\hat \Omega_\textnormal{c,cold} \hat t}\,, \qquad \varepsilon_\textnormal{h} = \frac{1}{\hat \Omega_\textnormal{c,hot} \hat t} \,, \qquad \nu = \frac{Z_\textnormal{h}}{Z_\textnormal{c}}\,.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \nu \frac{\alpha^2}{\varepsilon_\textnormal{c}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}\,.
        \end{align}

    Note
    ----------
    If hot and cold particles are of the same species (:math:`Z_\textnormal{c} = Z_\textnormal{h} \,, A_\textnormal{c} = A_\textnormal{h}`) then :math:`\varepsilon_\textnormal{c} = \varepsilon_\textnormal{h}` and :math:`\nu = 1`.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_markers.PushEta` 
    6. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`

    :ref:`Model info <add_model>`:
    '''

    @staticmethod
    def species():
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['e_field'] = 'Hcurl'
        dct['em_fields']['b_field'] = 'Hdiv'
        dct['fluid']['cold_electrons'] = {'j': 'Hcurl'}
        dct['kinetic']['hot_electrons'] = 'Particles6D'
        return dct

    @staticmethod
    def bulk_species():
        return 'cold_electrons'

    @staticmethod
    def velocity_scale():
        return 'light'

    @staticmethod
    def propagators_dct():
        return {propagators_fields.Maxwell: ['e_field', 'b_field'],
                propagators_fields.OhmCold: ['cold_electrons_j', 'e_field'],
                propagators_fields.JxBCold: ['cold_electrons_j'],
                propagators_markers.PushEta: ['hot_electrons'],
                propagators_markers.PushVxB: ['hot_electrons'],
                propagators_coupling.VlasovAmpere: ['e_field', 'hot_electrons']}

    __em_fields__ = species()['em_fields']
    __fluid_species__ = species()['fluid']
    __kinetic_species__ = species()['kinetic']
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    # add special options
    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(species=['em_fields'],
                       option=propagators_fields.ImplicitDiffusion,
                       dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        from mpi4py.MPI import SUM, IN_PLACE

        # Get rank and size
        self._rank = comm.Get_rank()

        # prelim
        hot_params = params['kinetic']['hot_electrons']

        # model parameters
        self._alpha = np.abs(
            self.equation_params['cold_electrons']['alpha'])
        self._epsilon_cold = self.equation_params['cold_electrons']['epsilon']
        self._epsilon_hot = self.equation_params['hot_electrons']['epsilon']

        self._nu = hot_params['phys_params']['Z'] / \
            params['fluid']['cold_electrons']['phys_params']['Z']

        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P['2']([self.mhd_equil.b2_1,
                                                 self.mhd_equil.b2_2,
                                                 self.mhd_equil.b2_3])

        # propagator parameters
        params_maxwell = params['em_fields']['options']['Maxwell']['solver']
        params_ohmcold = params['fluid']['cold_electrons']['options']['OhmCold']['solver']
        params_jxbcold = params['fluid']['cold_electrons']['options']['JxBCold']['solver']
        algo_eta = params['kinetic']['hot_electrons']['options']['PushEta']['algo']
        algo_vxb = params['kinetic']['hot_electrons']['options']['PushVxB']['algo']
        params_coupling = params['em_fields']['options']['VlasovAmpere']['solver']
        self._poisson_params = params['em_fields']['options']['ImplicitDiffusion']['solver']

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {'solver': params_maxwell}

        self._kwargs[propagators_fields.OhmCold] = {'alpha': self._alpha,
                                                    'epsilon': self._epsilon_cold,
                                                    'solver': params_ohmcold}

        self._kwargs[propagators_fields.JxBCold] = {'epsilon': self._epsilon_cold,
                                                    'solver': params_jxbcold}

        self._kwargs[propagators_markers.PushEta] = {'algo': algo_eta,
                                                     'bc_type': hot_params['markers']['bc']['type']}

        self._kwargs[propagators_markers.PushVxB] = {'algo': algo_vxb,
                                                     'scale_fac': 1./self._epsilon_cold,
                                                     'b_eq': self._b_background,
                                                     'b_tilde': self.pointer['b_field']}

        self._kwargs[propagators_coupling.VlasovAmpere] = {'c1': self._nu * self._alpha**2/self._epsilon_cold,
                                                           'c2': 1./self._epsilon_hot,
                                                           'solver': params_coupling}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

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
        self._tmp1 = self.pointer['e_field'].space.zeros()
        self._tmp2 = self.pointer['b_field'].space.zeros()
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        ''':meta private:'''
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
        from psydac.linalg.stencil import StencilVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.derham, self.domain, "H1", "vlasov_maxwell_poisson")
        charge_accum.accumulate(self.pointer['hot_electrons'])

        # Locally subtract mean charge for solvability with periodic bc
        if np.all(charge_accum.vectors[0].space.periods):
            charge_accum._vectors[0][:] -= np.mean(charge_accum.vectors[0].toarray()[
                                                   charge_accum.vectors[0].toarray() != 0])

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh['0'])
        poisson_solver = propagators_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0,
            rho=self._nu * self._alpha**2 /
            self._epsilon_cold * charge_accum.vectors[0],
            x0=self._nu * self._alpha**2 /
            self._epsilon_cold * charge_accum.vectors[0],
            solver=self._poisson_params)

        # Solve with dt=1. and compute electric field
        poisson_solver(1.)
        self.derham.grad.dot(-_phi, out=self.pointer['e_field'])

    def update_scalar_quantities(self):

        self._mass_ops.M1.dot(self.pointer['e_field'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b_field'], out=self._tmp2)
        en_E = .5 * self.pointer['e_field'].dot(self._tmp1)
        en_B = .5 * self.pointer['b_field'].dot(self._tmp2)
        self._mass_ops.M1ninv.dot(
            self.pointer['cold_electrons_j'], out=self._tmp1)
        en_J = .5 * self._alpha**2 * \
            self.pointer['cold_electrons_j'].dot(self._tmp1)
        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_J', en_J)

        # nu alpha^2 eps_h / eps_c / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = self._nu * self._alpha**2 * self._epsilon_hot / self._epsilon_cold / \
            (2 * self.pointer['hot_electrons'].n_mks) * np.dot(self.pointer['hot_electrons'].markers_wo_holes[:, 3]**2 + self.pointer['hot_electrons'].markers_wo_holes[:, 4]
                                                               ** 2 + self.pointer['hot_electrons'].markers_wo_holes[:, 5]**2, self.pointer['hot_electrons'].markers_wo_holes[:, 6])
        self.derham.comm.Allreduce(
            self._mpi_in_place, self._tmp, op=self._mpi_sum)
        self.update_scalar('en_f', self._tmp[0])

        # en_tot = en_E + en_B + en_J + en_w
        self.update_scalar('en_tot', en_E + en_B + en_J + self._tmp[0])
