from struphy.models.base import StruphyModel
import numpy as np


class LinearMHD(StruphyModel):
    r'''Linear ideal MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    Find :math:`(\tilde n, \tilde{\mathbf{U}}, \tilde p, \tilde{\mathbf{B}}) \in L^2 \times H(\textrm{div}) \times L^2 \times H(\textrm{div})` such that

    .. math::
        &\frac{\partial \tilde n}{\partial t}+\nabla\cdot(n_0 \tilde{\mathbf{U}})=0\,, 

        \int n_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \tilde{\mathbf{V}}\,\textrm d \mathbf x  - \int \tilde p\, \nabla \cdot \tilde{\mathbf{V}} \,\textrm d \mathbf x
        =\int \tilde{\mathbf{B}}\cdot \nabla \times (\mathbf{B}_0 \times \tilde{\mathbf{V}})\,\textrm d \mathbf x + \int (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} \cdot \tilde{\mathbf{V}}\,\textrm d \mathbf x
        \qquad \forall \ \tilde{\mathbf{V}} \in H(\textrm{div})\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A, bulk} \,, \qquad \hat p = \frac{\hat B^2}{\mu_0}\,.

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
        from struphy.propagators.propagators_fields import ShearAlfvén, Magnetosonic

        dct = {}
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'shear_alfven'],
                       option=ShearAlfvén.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'magnetosonic'],
                       option=Magnetosonic.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        alfven_solver = params['fluid']['mhd']['options']['solvers']['shear_alfven']
        sonic_solver = params['fluid']['mhd']['options']['solvers']['magnetosonic']

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

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.ShearAlfvén(
            self.pointer['mhd_u2'],
            self.pointer['b2'],
            **alfven_solver))
        self.add_propagator(self.prop_fields.Magnetosonic(
            self.pointer['mhd_n3'],
            self.pointer['mhd_u2'],
            self.pointer['mhd_p3'],
            b=self.pointer['b2'],
            **sonic_solver))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_p')
        self.add_scalar('en_B')
        self.add_scalar('en_p_eq')
        self.add_scalar('en_B_eq')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()
        self._tmp_b1 = self.derham.Vh['2'].zeros()
        self._tmp_b2 = self.derham.Vh['2'].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_u2'], out=self._tmp_u1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u2'] .dot(self._tmp_u1)/2
        en_B = self.pointer['b2'] .dot(self._tmp_b1)/2
        en_p = self.pointer['mhd_p3'] .dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p', en_p)
        self.update_scalar('en_tot', en_U + en_B + en_p)

        # background fields
        self._mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2
        en_p0 = self._p_eq.dot(self._ones)/(5/3 - 1)

        self.update_scalar('en_B_eq', en_B0)
        self.update_scalar('en_p_eq', en_p0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b2']

        self._mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2

        self.update_scalar('en_B_tot', en_Btot)


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
    def species(cls):
        dct = {'em_fields': {}, 'fluid': {}, 'kinetic': {}}

        dct['em_fields']['b1'] = 'Hcurl'
        dct['fluid']['mhd'] = {'n3': 'L2',
                               'u2': 'Hdiv', 'pi3': 'L2', 'pe3': 'L2'}
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
        from struphy.propagators.propagators_fields import ShearAlfvénB1, Hall, SonicIon, SonicElectron

        dct = {}
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'shear_alfven'],
                       option=ShearAlfvénB1.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'M1_inv'],
                       option=ShearAlfvénB1.options()['M1_inv'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'hall'],
                       option=Hall.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'sonic_ion'],
                       option=SonicIon.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solvers', 'sonic_electron'],
                       option=SonicElectron.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        # initialize base class
        super().__init__(params, comm)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        alfven_solver = params['fluid']['mhd']['options']['solvers']['shear_alfven']
        M1_inv = params['fluid']['mhd']['options']['solvers']['M1_inv']
        Hall_solver = params['fluid']['mhd']['options']['solvers']['hall']
        SonicIon_solver = params['fluid']['mhd']['options']['solvers']['sonic_ion']
        SonicElectron_solver = params['fluid']['mhd']['options']['solvers']['sonic_electron']

        # project background magnetic field (1-form) and pressure (3-form)
        self._b_eq = self.derham.P['1']([self.mhd_equil.b1_1,
                                         self.mhd_equil.b1_2,
                                         self.mhd_equil.b1_3])
        self._p_i_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._p_e_eq = self.derham.P['3'](self.mhd_equil.p3)
        self._ones = self.pointer['mhd_pi3'].space.zeros()
        # project background vector potential (1-form)
        self._a_eq = self.derham.P['1']([self.mhd_equil.a1_1,
                                         self.mhd_equil.a1_2,
                                         self.mhd_equil.a1_3])

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.
        else:
            self._ones[:] = 1.

        # compute coupling parameters
        kappa = 1. / self.equation_params['mhd']['epsilon']

        if abs(kappa - 1) < 1e-6:
            kappa = 1.

        self._coupling_params = {}
        self._coupling_params['kappa'] = kappa

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.ShearAlfvénB1(
            self.pointer['mhd_u2'],
            self.pointer['b1'],
            **alfven_solver,
            **M1_inv))
        self.add_propagator(self.prop_fields.Hall(
            self.pointer['b1'],
            **Hall_solver,
            **self._coupling_params))
        self.add_propagator(self.prop_fields.SonicIon(
            self.pointer['mhd_n3'],
            self.pointer['mhd_u2'],
            self.pointer['mhd_pi3'],
            **SonicIon_solver))
        self.add_propagator(self.prop_fields.SonicElectron(
            self.pointer['mhd_n3'],
            self.pointer['mhd_u2'],
            self.pointer['mhd_pe3'],
            **SonicElectron_solver))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_p_i')
        self.add_scalar('en_p_e')
        self.add_scalar('en_B')
        self.add_scalar('en_p_i_eq')
        self.add_scalar('en_p_e_eq')
        self.add_scalar('en_B_eq')
        self.add_scalar('en_B_tot')
        self.add_scalar('en_tot')
        self.add_scalar('helicity')

        # temporary vectors for scalar quantities
        self._tmp_u1 = self.derham.Vh['2'].zeros()

        self._tmp_b1 = self.derham.Vh['1'].zeros()
        self._tmp_b2 = self.derham.Vh['1'].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        self._mass_ops.M2n.dot(self.pointer['mhd_u2'], out=self._tmp_u1)

        self._mass_ops.M1.dot(self.pointer['b1'], out=self._tmp_b1)

        en_U = self.pointer['mhd_u2'].dot(self._tmp_u1)/2.0
        en_B = self.pointer['b1'].dot(self._tmp_b1)/2.0
        helicity = self._a_eq.dot(self._tmp_b1)*2.0
        en_p_i = self.pointer['mhd_pi3'].dot(self._ones)/(5.0/3.0 - 1.0)
        en_p_e = self.pointer['mhd_pe3'].dot(self._ones)/(5.0/3.0 - 1.0)

        self.update_scalar('en_U', en_U)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_p_i', en_p_i)
        self.update_scalar('en_p_e', en_p_e)
        self.update_scalar('helicity', helicity)
        self.update_scalar('en_tot', en_U + en_B + en_p_i + en_p_e)

        # background fields
        self._mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)

        en_B0 = self._b_eq.dot(self._tmp_b1)/2.0
        en_p0_i = self._p_i_eq.dot(self._ones)/(5.0/3.0 - 1.0)
        en_p0_e = self._p_e_eq.dot(self._ones)/(5.0/3.0 - 1.0)

        self.update_scalar('en_B_eq', en_B0)
        self.update_scalar('en_p_i_eq', en_p0_i)
        self.update_scalar('en_p_e_eq', en_p0_e)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.pointer['b1']

        self._mass_ops.M1.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.dot(self._tmp_b2)/2.0

        self.update_scalar('en_B_tot', en_Btot)


class ColdPlasma(StruphyModel):
    r'''Cold plasma model

    Normalization:

    .. math::

        c = \frac{\hat \omega}{\hat k} = \frac{\hat E}{\hat B}\,, \qquad \alpha = \frac{\hat \Omega_\textnormal{pc}}{\hat \Omega_\textnormal{cc}}\,, \qquad \varepsilon_\textnormal{c} = \frac{\hat{\omega}}{\hat \Omega_\textnormal{cc}}\,, \qquad \hat j_\textnormal{c} = q_\textnormal{c} c \hat n_\textnormal{c}\,,

    where :math:`c` is the vacuum speed of light, :math:`\hat \Omega_\textnormal{cc}` the electron cyclotron frequency,
    and :math:`\hat \Omega_\textnormal{pc}` the plasma frequency.
    Implemented equations:

    .. math::

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,

        &-\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon_\textnormal{c}} \mathbf j_\textnormal{c} \,,

        &\frac{1}{n_0} \frac{\partial \mathbf j_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf E + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf j_\textnormal{c} \times \mathbf B_0\,.

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background.

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
        dct['fluid']['electrons'] = {'j1': 'Hcurl'}
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
        from struphy.propagators.propagators_fields import Maxwell, OhmCold, JxBCold

        dct = {}
        cls.add_option(species=['em_fields'], key=['solver', 'maxwell'],
                       option=Maxwell.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'electrons'], key=['solvers', 'ohmcold'],
                       option=OhmCold.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'electrons'], key=['solvers', 'jxbcold'],
                       option=JxBCold.options()['solver'], dct=dct)
        return dct

    def __init__(self, params, comm):

        super().__init__(params, comm)

        # model parameters
        self._alpha = self.equation_params['electrons']['alpha']
        self._epsilon = self.equation_params['electrons']['epsilon']

        # solver parameters
        params_maxwell = params['em_fields']['options']['solver']['maxwell']
        params_ohmcold = params['fluid']['electrons']['options']['solvers']['ohmcold']
        params_jxbcold = params['fluid']['electrons']['options']['solvers']['jxbcold']

        # Initialize propagators/integrators used in splitting substeps
        self.add_propagator(self.prop_fields.Maxwell(
            self.pointer['e1'],
            self.pointer['b2'],
            **params_maxwell))
        self.add_propagator(self.prop_fields.OhmCold(
            self.pointer['electrons_j1'],
            self.pointer['e1'],
            **params_ohmcold,
            alpha=self._alpha,
            epsilon=self._epsilon))
        self.add_propagator(self.prop_fields.JxBCold(
            self.pointer['electrons_j1'],
            **params_jxbcold,
            alpha=self._alpha,
            epsilon=self._epsilon))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_E')
        self.add_scalar('en_B')
        self.add_scalar('en_J')
        self.add_scalar('en_tot')

        # temporaries
        self._tmp1 = self.pointer['e1'].space.zeros()
        self._tmp2 = self.pointer['b2'].space.zeros()

    def update_scalar_quantities(self):

        self._mass_ops.M1.dot(self.pointer['e1'], out=self._tmp1)
        self._mass_ops.M2.dot(self.pointer['b2'], out=self._tmp2)
        en_E = .5 * self.pointer['e1'].dot(self._tmp1)
        en_B = .5 * self.pointer['b2'].dot(self._tmp2)

        self._mass_ops.M1ninv.dot(self.pointer['electrons_j1'], out=self._tmp1)
        en_J = .5 * self._alpha**2 * \
            self.pointer['electrons_j1'].dot(self._tmp1)

        self.update_scalar('en_E', en_E)
        self.update_scalar('en_B', en_B)
        self.update_scalar('en_J', en_J)
        self.update_scalar('en_tot', en_E + en_B + en_J)


class VariationalMHD(StruphyModel):
    r'''Full (non-linear) MHD systen discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat \rho \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = (\gamma - 1) \hat \rho^{\gamma} \exp(\hat s / \hat \rho) = \frac{\hat B^2}{\mu_0}\,.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t (\rho \mathbf u) \cdot \mathbf v \, \textnormal d^3 \mathbf x 
        - \int_{\Omega} \mathbf \rho u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x 
        + \int_{\Omega} \big( \frac{| \mathbf u |^2}{2} - \frac{\partial \rho e}{\partial \rho} \big) \nabla \cdot (\rho \mathbf v) \, \textnormal d^3 \mathbf x &

        - \int_{\Omega} \big( \frac{\partial \rho e}{\partial s} \big) \nabla \cdot (s \mathbf v) \, \textnormal d^3 \mathbf x 
        - \int_{\Omega} \mathbf B \cdot \nabla \times (\mathbf B \mathbf v) \, \textnormal d^3 \mathbf x& = 0 ~ , 

        \partial_t \rho + \nabla \cdot ( \rho \mathbf u ) & = 0 ~ , 

        \partial_t s + \nabla \cdot ( s \mathbf u ) & = 0 ~ , 

        \partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) & = 0 ~ , 

    where

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    and

    .. math::
        e = \rho^{\gamma-1} \exp(s / \rho) ~ .

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
        dct['fluid']['mhd'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
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
        from struphy.propagators.propagators_fields import VariationalMomentumAdvection, VariationalDensityEvolve, VariationalEntropyEvolve, VariationalMagFieldEvolve
        dct = {}

        cls.add_option(species=['fluid', 'mhd'], key=['solver_momentum'],
                       option=VariationalMomentumAdvection.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_density'],
                       option=VariationalDensityEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_entropy'],
                       option=VariationalEntropyEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_magnetic'],
                       option=VariationalMagFieldEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['physics'],
                       option=VariationalDensityEvolve.options()['physics'], dct=dct)

        return dct

    def __init__(self, params, comm):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator
        import numpy as np

        # initialize base class
        super().__init__(params, comm)
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'], 
            self.derham.Vh_fem['v'],)
            # V_extraction_op=self.derham.extraction_ops['v'],
            # W_extraction_op=self.derham.extraction_ops['v'],
            # V_boundary_op=self.derham.boundary_ops['v'],
            # W_boundary_op=self.derham.boundary_ops['v'])

        # Initialize propagators/integrators used in splitting substeps
        solver_momentum = params['fluid']['mhd']['options']['solver_momentum']
        solver_density = params['fluid']['mhd']['options']['solver_density']
        solver_entropy = params['fluid']['mhd']['options']['solver_entropy']
        solver_magnetic = params['fluid']['mhd']['options']['solver_magnetic']

        gamma = params['fluid']['mhd']['options']['physics']['gamma']

        self.add_propagator(self.prop_fields.VariationalDensityEvolve(
            self.pointer['mhd_rho3'], self.pointer['mhd_uv'],
            model='full',
            s=self.pointer['mhd_s3'],
            gamma=gamma,
            mass_ops=self.WMM,
            **solver_density))
        self.add_propagator(self.prop_fields.VariationalMomentumAdvection(
            self.pointer['mhd_uv'],
            mass_ops=self.WMM,
            **solver_momentum))
        self.add_propagator(self.prop_fields.VariationalEntropyEvolve(
            self.pointer['mhd_s3'], self.pointer['mhd_uv'],
            model='full',
            rho=self.pointer['mhd_rho3'],
            gamma=gamma,
            mass_ops=self.WMM,
            **solver_entropy))
        self.add_propagator(self.prop_fields.VariationalMagFieldEvolve(
            self.pointer['b2'], self.pointer['mhd_uv'],
            mass_ops=self.WMM,
            **solver_magnetic))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_mag')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh['v'].zeros()
        self._tmp_wb2 = self.derham.Vh['2'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f)

    def update_scalar_quantities(self):

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['mhd_uv'], out=self._tmp_m1)

        en_U = self.pointer['mhd_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        wb2 = self._mass_ops.M2.dot(self.pointer['b2'])
        en_mag = wb2.dot(self.pointer['b2'])/2
        self.update_scalar('en_mag', en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar('en_tot', en_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        '''
        en_prop = self._propagators[2]
        en_prop.sf.vector = self.pointer['mhd_s3']
        en_prop.rhof.vector = self.pointer['mhd_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_ds)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo
    
    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._params['fluid']['mhd']['options']['physics']['gamma']
        return np.power(rho, gam)*np.exp(s/rho)
    
    
class VariationalMHDwithExtField(StruphyModel):
    r'''Full (non-linear) MHD systen discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \frac{\hat B}{\sqrt{A_\textnormal{b} m_\textnormal{H} \hat \rho \mu_0}} =: \hat v_\textnormal{A} = \frac{\hat \omega}{\hat k} = \hat U \,, \qquad \hat p = (\gamma - 1) \hat \rho^{\gamma} \exp(\hat s / \hat \rho) = \frac{\hat B^2}{\mu_0}\,.

    Implemented equations:

    .. math::

        \int_{\Omega} \partial_t (\rho \mathbf u) \cdot \mathbf v \, \textnormal d^3 \mathbf x 
        - \int_{\Omega} \mathbf \rho u \cdot [\mathbf u, \mathbf v] \, \textnormal d^3 \mathbf x 
        + \int_{\Omega} \big( \frac{| \mathbf u |^2}{2} - \frac{\partial \rho e}{\partial \rho} \big) \nabla \cdot (\rho \mathbf v) \, \textnormal d^3 \mathbf x &

        - \int_{\Omega} \big( \frac{\partial \rho e}{\partial s} \big) \nabla \cdot (s \mathbf v) \, \textnormal d^3 \mathbf x 
        - \int_{\Omega} \mathbf B \cdot \nabla \times (\mathbf B \mathbf v) \, \textnormal d^3 \mathbf x& = 0 ~ , 

        \partial_t \rho + \nabla \cdot ( \rho \mathbf u ) & = 0 ~ , 

        \partial_t s + \nabla \cdot ( s \mathbf u ) & = 0 ~ , 

        \partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) & = 0 ~ , 

    where

    .. math::
        [\mathbf u,\mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u ~ .

    and

    .. math::
        e = \rho^{\gamma-1} \exp(s / \rho) ~ .

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
        dct['fluid']['mhd'] = {'rho3': 'L2', 's3': 'L2', 'uv': 'H1vec'}
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
        from struphy.propagators.propagators_fields import VariationalMomentumAdvection, VariationalDensityEvolve, VariationalEntropyEvolve, VariationalMagFieldEvolve
        dct = {}

        cls.add_option(species=['fluid', 'mhd'], key=['solver_momentum'],
                       option=VariationalMomentumAdvection.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_density'],
                       option=VariationalDensityEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_entropy'],
                       option=VariationalEntropyEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['solver_magnetic'],
                       option=VariationalMagFieldEvolve.options()['solver'], dct=dct)
        cls.add_option(species=['fluid', 'mhd'], key=['physics'],
                       option=VariationalDensityEvolve.options()['physics'], dct=dct)

        return dct

    def __init__(self, params, comm):

        from struphy.feec.projectors import L2Projector
        from struphy.feec.mass import WeightedMassOperator
        import numpy as np

        # initialize base class
        super().__init__(params, comm)
        # Initialize mass matrix
        self.WMM = WeightedMassOperator(
            self.derham.Vh_fem['v'], 
            self.derham.Vh_fem['v'],)
            # V_extraction_op=self.derham.extraction_ops['v'],
            # W_extraction_op=self.derham.extraction_ops['v'],
            # V_boundary_op=self.derham.boundary_ops['v'],
            # W_boundary_op=self.derham.boundary_ops['v'])

        # Initialize propagators/integrators used in splitting substeps
        solver_momentum = params['fluid']['mhd']['options']['solver_momentum']
        solver_density = params['fluid']['mhd']['options']['solver_density']
        solver_entropy = params['fluid']['mhd']['options']['solver_entropy']
        solver_magnetic = params['fluid']['mhd']['options']['solver_magnetic']

        gamma = params['fluid']['mhd']['options']['physics']['gamma']

        self.add_propagator(self.prop_fields.VariationalDensityEvolve(
            self.pointer['mhd_rho3'], self.pointer['mhd_uv'],
            model='full',
            s=self.pointer['mhd_s3'],
            gamma=gamma,
            mass_ops=self.WMM,
            **solver_density))
        self.add_propagator(self.prop_fields.VariationalMomentumAdvection(
            self.pointer['mhd_uv'],
            mass_ops=self.WMM,
            **solver_momentum))
        self.add_propagator(self.prop_fields.VariationalEntropyEvolve(
            self.pointer['mhd_s3'], self.pointer['mhd_uv'],
            model='full',
            rho=self.pointer['mhd_rho3'],
            gamma=gamma,
            mass_ops=self.WMM,
            **solver_entropy))
        self.add_propagator(self.prop_fields.VariationalMagFieldEvolve(
            self.pointer['b2'], self.pointer['mhd_uv'],
            mass_ops=self.WMM,
            **solver_magnetic))

        # Scalar variables to be saved during simulation
        self.add_scalar('en_U')
        self.add_scalar('en_thermo')
        self.add_scalar('en_mag')
        self.add_scalar('en_tot')

        # temporary vectors for scalar quantities
        self._tmp_m1 = self.derham.Vh['v'].zeros()
        self._tmp_wb2 = self.derham.Vh['2'].zeros()
        projV3 = L2Projector('L2', self._mass_ops)
        def f(e1, e2, e3): return 1
        f = np.vectorize(f)
        self._integrator = projV3(f)

    def update_scalar_quantities(self):

        WMM = self.WMM
        m1 = WMM.dot(self.pointer['mhd_uv'], out=self._tmp_m1)

        en_U = self.pointer['mhd_uv'] .dot(m1)/2
        self.update_scalar('en_U', en_U)

        wb2 = self._mass_ops.M2.dot(self.pointer['b2'])
        en_mag = wb2.dot(self.pointer['b2'])/2
        self.update_scalar('en_mag', en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar('en_tot', en_tot)

    def update_thermo_energy(self):
        '''Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        '''
        en_prop = self._propagators[2]
        en_prop.sf.vector = self.pointer['mhd_s3']
        en_prop.rhof.vector = self.pointer['mhd_rho3']
        sf_values = en_prop.sf.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._sf_values)
        rhof_values = en_prop.rhof.eval_tp_fixed_loc(
            en_prop.integration_grid_spans, en_prop.integration_grid_bd, out=en_prop._rhof_values)
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term*e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.dot(en_prop._linear_form_dl_ds)
        self.update_scalar('en_thermo', en_thermo)
        return en_thermo
    
    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        gam = self._params['fluid']['mhd']['options']['physics']['gamma']
        return np.power(rho, gam)*np.exp(s/rho)
