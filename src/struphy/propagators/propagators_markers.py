'Only particle variables are updated.'


from numpy import array, polynomial

from psydac.linalg.block import BlockVector

from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.pic.pushing.pusher import Pusher
from struphy.pic.pushing.pusher import ButcherTableau
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.kinetic_background.maxwellians import Maxwellian6DUniform


class PushEta(Propagator):
    r"""Solves

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant. Available algorithms:

        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'algo': 'rk4',
                          'bc_type': ['reflect', 'periodic', 'periodic'],
                          }

        params = set_defaults(params, params_default)

        self._bc_type = params['bc_type']

        # choose algorithm
        if params['algo'] == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif params['algo'] == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['algo'] == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['algo'] == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['algo'] == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_eta_stage', n_stages=self._butcher.n_stages)

    def __call__(self, dt):
        """
        TODO
        """

        # push markers
        self._pusher(self.particles[0], dt,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='last')

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = ['rk4', 'forward_euler', 'heun2', 'rk2', 'heun3']
        return dct


class PushVxB(Propagator):
    r"""Solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector. Available algorithms:

        * analytic
        * implicit

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'algo': 'analytic',
                          'scale_fac': 1.,
                          'b_eq': None,
                          'b_tilde': None, }

        params = set_defaults(params, params_default)

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        self._scale_fac = params['scale_fac']
        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']

        # load pusher
        kernel_name = 'push_vxb_' + params['algo']
        self._pusher = Pusher(self.derham, self.domain, kernel_name)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops['2'].transpose()

    def __call__(self, dt):
        """
        TODO
        """

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b_tilde is not None:
            b_full += self._b_tilde

        # extract coefficients to tensor product space
        b_full = self._E2T.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()

        # call pusher kernel
        self._pusher(self.particles[0], self._scale_fac*dt,
                     b_full[0]._data,
                     b_full[1]._data,
                     b_full[2]._data)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = ['analytic', 'implicit']
        return dct


class StepPushpxBHybrid(Propagator):
    r"""Solves

    .. math::

        \frac{\textnormal d \mathbf p_i(t)}{\textnormal d t} =  (\mathbf p_i(t) - {\mathbf A}({\mathbf x})) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`i` in markers array, with fixed rotation vector. Available algorithms:

        * analytic

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'b_eq': None,
                          'a': None,
                          'method': None
                          }

        params = set_defaults(params, params_default)

        self._b_eq = params['b_eq']
        self._a = params['a']
        self._algo = params['method']

        self._C = self.derham.curl

        # load pusher
        kernel_name = 'push_pxb_' + self._algo

        self._pusher = Pusher(self.derham, self.domain, kernel_name)

        assert isinstance(self._b_eq, (BlockVector, PolarVector))
        assert isinstance(self._a,    (BlockVector, PolarVector))

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops['2'].transpose()

    def __call__(self, dt):
        """
        TODO
        """

        # sum up total magnetic field
        b_full = self._b_eq.space.zeros()

        b_full += self._b_eq

        # extract coefficients to tensor product space
        b_full = self._E2T.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()
        self._a.update_ghost_regions()

        # call pusher kernel
        self._pusher(self.particles[0], dt,
                     b_full[0]._data,
                     b_full[1]._data,
                     b_full[2]._data,
                     self._a[0]._data,
                     self._a[1]._data,
                     self._a[2]._data)


class StepHybridXPSymplectic(Propagator):
    r'''Step for the update of particles' positions and canonical momentum with symplectic methods (only in Cartesian coordinates) which solve the following Hamiltonian system

    .. math::

        \frac{\mathrm{d} {\mathbf x}(t)}{\textnormal d t} = {\mathbf p} - {\mathbf A}, \quad \frac{\mathrm{d} {\mathbf p}(t)}{\textnormal d t} = - \left( \frac{\partial{\mathbf A}}{\partial {\mathbf x}} \right)^\top ({\mathbf A} - {\mathbf p} ) - T \frac{\nabla n}{n}. 

    for each marker in markers array.

    Parameters
    ----------
        particles : struphy.pic.particles.Particles6D

        **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'a': None,
                          'particle_bc': None,
                          'quad_number': None,
                          'shape_degree': None,
                          'shape_size': None,
                          'electron_temperature': None,
                          'accumulate_density': None
                          }

        params = set_defaults(params, params_default)

        self._a = params['a']
        self._bc = params['particle_bc']
        self._thermal = params['electron_temperature']

        assert isinstance(self._a, BlockVector)

        self._nqs = params['quad_number']
        self._p_shape = params['shape_degree']
        self._p_size = params['shape_size']
        self._accum_density = params['accumulate_density']

        # Initialize Accumulator object for getting density from particles
        self._pts_x = 1.0 / (2.0*self.derham.Nel[0]) * polynomial.legendre.leggauss(
            self._nqs[0])[0] + 1.0 / (2.0*self.derham.Nel[0])
        self._pts_y = 1.0 / (2.0*self.derham.Nel[1]) * polynomial.legendre.leggauss(
            self._nqs[1])[0] + 1.0 / (2.0*self.derham.Nel[1])
        self._pts_z = 1.0 / (2.0*self.derham.Nel[2]) * polynomial.legendre.leggauss(
            self._nqs[2])[0] + 1.0 / (2.0*self.derham.Nel[2])

        self._wts_x = 1.0 / \
            (2.0*self.derham.Nel[0]) * \
            polynomial.legendre.leggauss(self._nqs[0])[1]
        self._wts_y = 1.0 / \
            (2.0*self.derham.Nel[0]) * \
            polynomial.legendre.leggauss(self._nqs[1])[1]
        self._wts_z = 1.0 / \
            (2.0*self.derham.Nel[0]) * \
            polynomial.legendre.leggauss(self._nqs[2])[1]

        # set kernel function
        self._pusher_lnn = Pusher(
            self.derham, self.domain, 'push_hybrid_xp_lnn')
        self._pusher_ap = Pusher(self.derham, self.domain, 'push_hybrid_xp_ap')

        self._pusher_inputs = (
            self._a[0]._data, self._a[1]._data, self._a[2]._data)

    def __call__(self, dt):
        """
        TODO
        """
        # get density from particles
        self._accum_density.accumulate(self.particles[0], array(self.derham.Nel), array(self._nqs), array(
            self._pts_x), array(self._pts_y), array(self._pts_z), array(self._p_shape), array(self._p_size))
        if not self._accum_density._operators[0].matrix.ghost_regions_in_sync:
            self._accum_density._operators[0].matrix.update_ghost_regions()
        # print('++++++check_density+++++++++', self._accum_density._operators[0].matrix._data)
        self._pusher_lnn(self.particles[0], dt, array(self._p_shape), array(self._p_size), array(self.derham.Nel), array(self._pts_x), array(self._pts_y), array(
            self._pts_z), array(self._wts_x), array(self._wts_y), array(self._wts_z), self._accum_density._operators[0].matrix._data, self._thermal, array(self._nqs))

        if not self._a[0].ghost_regions_in_sync:
            self._a[0].update_ghost_regions()
        if not self._a[1].ghost_regions_in_sync:
            self._a[1].update_ghost_regions()
        if not self._a[2].ghost_regions_in_sync:
            self._a[2].update_ghost_regions()
        self._pusher_ap(
            self.particles[0], dt, self._a[0]._data, self._a[1]._data, self._a[2]._data, mpi_sort='last')


class PushEtaPC(Propagator):
    r'''Step for the update of particles' positions with the RK4 method which solves

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,, \qquad \textnormal{vec}( \hat{\mathbf U}) = \hat{\mathbf U}\,.

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    derham : struphy.feec.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    u : psydac.linalg.block.BlockVector
        FE coefficients of a discrete 0-form, 1-form or 2-form.

    u_space : dic
        params['fields']['mhd_u_space']

    bc : list[str]
        Kinetic boundary conditions in each direction.
    '''

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'u_mhd': None,
                          'u_space': 'Hdiv',
                          'bc_type': ['reflect', 'periodic', 'periodic'],
                          'use_perp_model': True
                          }

        params = set_defaults(params, params_default)

        assert isinstance(params['u_mhd'], (BlockVector, PolarVector))

        self._u = params['u_mhd']
        self._u_space = params['u_space']
        self._bc = params['bc_type']

        # call Pusher class
        pusher_ker = 'push_pc_eta_rk4_' + self._u_space
        if not params['use_perp_model']:
            pusher_ker += '_full'

        self._pusher = Pusher(
            self.derham, self.domain, pusher_ker, n_stages=4)

    def __call__(self, dt):
        """
        TODO
        """
        # push particles
        # check if ghost regions are synchronized
        if not self._u[0].ghost_regions_in_sync:
            self._u[0].update_ghost_regions()
        if not self._u[1].ghost_regions_in_sync:
            self._u[1].update_ghost_regions()
        if not self._u[2].ghost_regions_in_sync:
            self._u[2].update_ghost_regions()

        self._pusher(self.particles[0], dt,
                     self._u[0]._data, self._u[1]._data, self._u[2]._data,
                     mpi_sort='last')

    @classmethod
    def options(cls):
        dct = {}
        dct['use_perp_model'] = [True, False]
        dct['u_space'] = ['Hcurl', 'Hdiv', 'H1vec']
        return dct


class PushGuidingCenterbxEstar(Propagator):
    r"""Particle pushing step for the :math:`\mathbf b_ \times \mathbf E^*` guiding center drift part in `DriftKinetic <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.toy.DriftKinetic>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= - \frac{1}{B_\parallel^*} \mathbf b_0 \times \mathbf E^* \,,
                \\
                \dot v_\parallel &= 0 \,.
            \end{aligned}
        \right.

    where :math:`\mathbf E^* = - \epsilon \mu \nabla B_0`.

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \epsilon \mu_p \frac{1}{ B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  G^{-1}(\boldsymbol \eta_p) \hat{\mathbf b}^2_0(\boldsymbol \eta_p) \times G^{-1}(\boldsymbol \eta_p) \hat \nabla \hat{B}^0_0 (\boldsymbol \eta_p) \,,
            \\
            \dot v_{\parallel,\,p} &= 0 \,.
        \end{aligned}

    for each marker :math:`p` in markers array. Available algorithms:

        * forward_euler (1st order, explicit)
        * heun2 (2nd order, explicit)
        * rk2 (2nd order, explicit)
        * heun3 (3rd order, explicit)
        * rk4 (4th order, explicit)
        * discrete_gradient (2nd order, implicit)
        * discrete_gradient_faster (1st order, implicit)
        * discrete_gradient_Itoh_Newton (2nd order, implicit)

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'epsilon': 1.,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'gradB1': None,
                          'curl_unit_b2': None,
                          'method': 'discrete_gradient',
                          'maxiter': 20,
                          'tol': 1e-07,
                          'mpi_sort': 'each',
                          'verbose': False
                          }

        params = set_defaults(params, params_default)

        self._mpi_sort = params['mpi_sort']
        self._verbose = params['verbose']
        epsilon = params['epsilon']
        b_eq = params['b_eq']
        unit_b1 = params['unit_b1']
        unit_b2 = params['unit_b2']
        abs_b = params['abs_b']
        grad_abs_b = params['gradB1']
        curl_norm_b = params['curl_unit_b2']

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        E2T = self.derham.extraction_ops['2'].transpose()

        b_eq = E2T.dot(b_eq)
        unit_b1 = E1T.dot(unit_b1)
        unit_b2 = E2T.dot(unit_b2)
        abs_b = E0T.dot(abs_b)
        curl_norm_b = E2T.dot(curl_norm_b)
        grad_abs_b = E1T.dot(grad_abs_b)

        curl_norm_b.update_ghost_regions()
        grad_abs_b.update_ghost_regions()

        _eval_ker_names = []

        if params['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif params['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif params['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval_gradI']
        elif params['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval_gradI']

        elif params['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval1',
                                'gc_bxEstar_' +
                                params['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {params["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_bxEstar_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (epsilon, b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   butcher.a, butcher.b, butcher.c)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=params['maxiter'],
                                  tol=params['tol'])

            self._pusher_inputs = (epsilon, abs_b._data,
                                   b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   params['maxiter'], params['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',
                                  'discrete_gradient',
                                  'discrete_gradient_faster',
                                  'discrete_gradient_Itoh_Newton'],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        return dct


class PushGuidingCenterBstar(Propagator):
    r"""Particle pushing step for the :math:`\mathbf B^*` guiding center drift part in `DriftKinetic <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.toy.DriftKinetic>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= v_\parallel \frac{\mathbf B^*}{B^*_\parallel} \,,
                \\
                \dot v_\parallel &= \frac{1}{\epsilon} \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E^* \,,
            \end{aligned}
        \right.

    where :math:`\mathbf B^* = \mathbf B_0 + \epsilon v_\parallel \nabla \times \mathbf b_0`, :math:`B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*` and :math:`\mathbf E^* = - \epsilon \mu \nabla B_0`.

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p} \,,
            \\
            \dot v_{\parallel,\,p} &= - \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \cdot \mu_p  \hat \nabla \hat{B}^0_0(\boldsymbol \eta_p) \,.
        \end{aligned}

    for each marker :math:`p` in markers array. Available algorithms:

        * forward_euler (1st order, explicit)
        * heun2 (2nd order, explicit)
        * rk2 (2nd order, explicit)
        * heun3 (3rd order, explicit)
        * rk4 (4th order, explicit)
        * discrete_gradient (2nd order, implicit)
        * discrete_gradient_faster (2nd order, implicit)
        * discrete_gradient_Itoh_Newton (2nd order, implicit)

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'epsilon': 1.,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'gradB1': None,
                          'curl_unit_b2': None,
                          'method': 'discrete_gradient_faster',
                          'maxiter': 20,
                          'tol': 1e-07,
                          'mpi_sort': 'each',
                          'verbose': False
                          }

        params = set_defaults(params, params_default)

        self._mpi_sort = params['mpi_sort']
        self._verbose = params['verbose']
        epsilon = params['epsilon']
        b_eq = params['b_eq']
        unit_b1 = params['unit_b1']
        unit_b2 = params['unit_b2']
        abs_b = params['abs_b']
        grad_abs_b = params['gradB1']
        curl_norm_b = params['curl_unit_b2']

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        E2T = self.derham.extraction_ops['2'].transpose()

        b_eq = E2T.dot(b_eq)
        unit_b1 = E1T.dot(unit_b1)
        unit_b2 = E2T.dot(unit_b2)
        abs_b = E0T.dot(abs_b)
        curl_norm_b = E2T.dot(curl_norm_b)
        grad_abs_b = E1T.dot(grad_abs_b)

        curl_norm_b.update_ghost_regions()
        grad_abs_b.update_ghost_regions()

        _eval_ker_names = []

        if params['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif params['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif params['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval_gradI']

        elif params['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval_gradI']

        elif params['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval1',
                                'gc_Bstar_' +
                                params['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {params["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_Bstar_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (epsilon, b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   butcher.a, butcher.b, butcher.c)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=params['maxiter'],
                                  tol=params['tol'])

            self._pusher_inputs = (epsilon, abs_b._data,
                                   b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   params['maxiter'], params['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',
                                  'discrete_gradient',
                                  'discrete_gradient_faster',
                                  'discrete_gradient_Itoh_Newton'],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        return dct


class StepVinEfield(Propagator):
    r'''Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = \kappa \, DL^{-T} \mathbf{E}

    which is solved analytically.

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {
            'e_field': BlockVector(self.derham.Vh_fem['1'].vector_space),
            'method': 'analytical',
            'kappa': 1e2
        }

        params = set_defaults(params, params_default)
        self.kappa = params['kappa']
        method = params['method']

        assert isinstance(params['e_field'], (BlockVector, PolarVector))
        self._e_field = params['e_field']

        if method == 'analytical':
            self._pusher = Pusher(self.derham, self.domain,
                                  'push_v_with_efield')
        elif method == 'discrete_gradient':
            raise NotImplementedError('Not yet implemented.')
            # self._pusher = Pusher(self.derham, self.domain,
            #                     'push_v_in_static_efield_dg')
        else:
            raise ValueError(f'Method {method} not known.')

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     self._e_field.blocks[0]._data, self._e_field.blocks[1]._data, self._e_field.blocks[2]._data,
                     self.kappa)

    @classmethod
    def options(cls):
        pass


class StepStaticEfield(Propagator):
    r'''Solve the following system

    .. math::

        \frac{\text{d} \mathbf{\eta}_p}{\text{d} t} & = DL^{-1} \mathbf{v}_p \,,

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = \kappa \, DL^{-T} \mathbf{E}

    which is solved by an average discrete gradient method, implicitly iterating
    over :math:`k` (for every particle :math:`p`):

    .. math::

        \mathbf{\eta}^{n+1}_{k+1} = \mathbf{\eta}^n + \frac{\Delta t}{2} DL^{-1}
        \left( \frac{\mathbf{\eta}^{n+1}_k + \mathbf{\eta}^n }{2} \right) \left( \mathbf{v}^{n+1}_k + \mathbf{v}^n \right) \,,

        \mathbf{v}^{n+1}_{k+1} = \mathbf{v}^n + \Delta t \, \kappa \, DL^{-1}\left(\mathbf{\eta}^n\right)
        \int_0^1 \left[ \mathbb{\Lambda}\left( \eta^n + \tau (\mathbf{\eta}^{n+1}_k - \mathbf{\eta}^n) \right) \right]^T \mathbf{e} \, \text{d} \tau

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, particles, **params):

        from numpy import polynomial, floor

        super().__init__(particles)

        # parameters
        params_default = {
            'e_field': BlockVector(self.derham.Vh_fem['1'].vector_space),
            'kappa': 1e2
        }

        params = set_defaults(params, params_default)
        self.kappa = params['kappa']

        assert isinstance(params['e_field'], (BlockVector, PolarVector))
        self._e_field = params['e_field']

        pn1 = self.derham.p[0]
        pd1 = pn1 - 1
        pn2 = self.derham.p[1]
        pd2 = pn2 - 1
        pn3 = self.derham.p[2]
        pd3 = pn3 - 1

        # number of quadrature points in direction 1
        n_quad1 = int(floor(pd1 * pn2 * pn3 / 2 + 1))
        # number of quadrature points in direction 2
        n_quad2 = int(floor(pn1 * pd2 * pn3 / 2 + 1))
        # number of quadrature points in direction 3
        n_quad3 = int(floor(pn1 * pn2 * pd3 / 2 + 1))

        # get quadrature weights and locations
        self._loc1, self._weight1 = polynomial.legendre.leggauss(n_quad1)
        self._loc2, self._weight2 = polynomial.legendre.leggauss(n_quad2)
        self._loc3, self._weight3 = polynomial.legendre.leggauss(n_quad3)

        self._pusher = Pusher(self.derham, self.domain,
                              'push_x_v_static_efield')

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     self._loc1, self._loc2, self._loc3, self._weight1, self._weight2, self._weight3,
                     self._e_field.blocks[0]._data, self._e_field.blocks[1]._data, self._e_field.blocks[2]._data,
                     self.kappa,
                     array([1e-10, 1e-10]), 100)


class PushDriftKineticbxGradB(Propagator):
    r"""Particle pushing step for the :math:`\mathbf b_0 \times \nabla B_\parallel` driftkinetic part in `LinearMHDDriftkineticCC <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.hybrid.LinearMHDDriftkineticCC>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= \epsilon \frac{1}{B_\parallel^*} \mathbf b_0 \times \mu \nabla B_\parallel \,,
                \\
                \dot v_\parallel &= 0 \,.
            \end{aligned}
        \right.

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \epsilon \mu_p \frac{1}{ B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  G^{-1}(\boldsymbol \eta_p) \hat{\mathbf b}^2_0(\boldsymbol \eta_p) \times G^{-1}(\boldsymbol \eta_p) \hat \nabla \hat{B}^0_\parallel (\boldsymbol \eta_p) \,,
            \\
            \dot v_{\parallel,\,p} &= 0 \,.
        \end{aligned}

    for each marker :math:`p` in markers array. Available algorithms:

        Explicit:
        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)

        Implicit:
        * discrete_gradient
        * discrete_gradient_faster

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'epsilon': 1.,
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'gradB1': None,
                          'curl_unit_b2': None,
                          'method': 'discrete_gradient',
                          'maxiter': 20,
                          'tol': 1e-07,
                          'mpi_sort': 'each',
                          'verbose': False
                          }

        params = set_defaults(params, params_default)

        self._mpi_sort = params['mpi_sort']
        self._verbose = params['verbose']
        self._epsilon = params['epsilon']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']
        self._grad_abs_b = params['gradB1']
        self._curl_norm_b = params['curl_unit_b2']
        self._maxiter = params['maxiter']
        self._tol = params['tol']

        self._b_full = self._b_eq.space.zeros()

        # define gradient of absolute value of parallel magnetic field
        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(self._b)
        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._PBb += self._abs_b
        self._grad_PBb += self._grad_abs_b

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)

        self._curl_norm_b.update_ghost_regions()

        _eval_ker_names = []

        if params['method'] == 'forward_euler':
            self._method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif params['method'] == 'heun2':
            self._method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['method'] == 'rk2':
            self._method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['method'] == 'heun3':
            self._method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['method'] == 'rk4':
            self._method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif params['method'] == 'discrete_gradient':
            self._method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval_gradI']
        elif params['method'] == 'discrete_gradient_faster':
            self._method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval_gradI']
        elif params['method'] == 'discrete_gradient_Itoh_Newton':
            self._method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + params['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                params['method'] + '_eval1',
                                'gc_bxEstar_' +
                                params['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {params["method"]} is not implemented.')

        if self._method == 'explicit':
            self._butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_bxEstar_explicit_multistage',
                                  n_stages=self._butcher.n_stages)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=params['maxiter'],
                                  tol=params['tol'])

    def __call__(self, dt):
        """
        TODO
        """

        # sum up total magnetic field
        self._b_full = self._b_eq.copy()
        if self._b is not None:
            self._b_full += self._b

        # define gradient of absolute value of parallel magnetic field
        self._PBb = self._PB.dot(self._b)
        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._PBb += self._abs_b
        self._grad_PBb += self._grad_abs_b

        self._b_full = self._E2T.dot(self._b_full)
        self._PBb = self._E0T.dot(self._PBb)
        self._grad_PBb = self._E1T.dot(self._grad_PBb)

        self._b_full.update_ghost_regions()
        self._PBb.update_ghost_regions()
        self._grad_PBb.update_ghost_regions()

        if self._method == 'explicit':
            self._pusher_inputs = (self._epsilon,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)
        else:
            self._pusher_inputs = (self._epsilon, self._PBb._data,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._maxiter, self._tol)

        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',
                                  'discrete_gradient',
                                  'discrete_gradient_faster',
                                  'discrete_gradient_Itoh_Newton'],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        return dct


class PushDriftKineticBstar(Propagator):
    r"""Particle pushing step for the :math:`\mathbf B^*` driftkinetic part in `LinearMHDDriftkineticCC <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.hybrid.LinearMHDDriftkineticCC>`_ model,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= v_\parallel \frac{\mathbf B^*}{B^*_\parallel} \,,
                \\
                \dot v_\parallel &= - \frac{\mathbf B^*}{B^*_\parallel} \cdot \mu \nabla B_\parallel \,,
            \end{aligned}
        \right.

    where :math:`\mathbf B^* = \mathbf B + \epsilon v_\parallel \nabla \times \mathbf b_0` and :math:`B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*`.

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p} \,,
            \\
            \dot v_{\parallel,\,p} &= - \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \cdot \mu_p \hat \nabla \hat{B}^0_\parallel(\boldsymbol \eta_p) \,.
        \end{aligned}

    for each marker :math:`p` in markers array. Available algorithms:

        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, **params):

        super().__init__(particles)

        # parameters
        params_default = {'epsilon': 1.,
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'gradB1': None,
                          'curl_unit_b2': None,
                          'method': 'discrete_gradient',
                          'maxiter': 20,
                          'tol': 1e-07,
                          'mpi_sort': 'each',
                          'verbose': False
                          }

        params = set_defaults(params, params_default)

        self._mpi_sort = params['mpi_sort']
        self._verbose = params['verbose']
        self._epsilon = params['epsilon']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']
        self._grad_abs_b = params['gradB1']
        self._curl_norm_b = params['curl_unit_b2']
        self._maxiter = params['maxiter']
        self._tol = params['tol']

        self._b_full = self._b_eq.space.zeros()

        # define gradient of absolute value of parallel magnetic field
        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(self._b)
        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._PBb += self._abs_b
        self._grad_PBb += self._grad_abs_b

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.extraction_ops['0'].transpose()
        self._E1T = self.derham.extraction_ops['1'].transpose()
        self._E2T = self.derham.extraction_ops['2'].transpose()

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)

        self._curl_norm_b.update_ghost_regions()

        _eval_ker_names = []

        if params['method'] == 'forward_euler':
            self._method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif params['method'] == 'heun2':
            self._method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif params['method'] == 'rk2':
            self._method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif params['method'] == 'heun3':
            self._method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif params['method'] == 'rk4':
            self._method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif params['method'] == 'discrete_gradient':
            self._method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval_gradI']

        elif params['method'] == 'discrete_gradient_faster':
            self._method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval_gradI']

        elif params['method'] == 'discrete_gradient_Itoh_Newton':
            self._method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + params['method']
            _eval_ker_names += ['gc_Bstar_' +
                                params['method'] + '_eval1',
                                'gc_Bstar_' +
                                params['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {params["method"]} is not implemented.')

        if self._method == 'explicit':
            self._butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_Bstar_explicit_multistage',
                                  n_stages=self._butcher.n_stages)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=params['maxiter'],
                                  tol=params['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        # sum up total magnetic field
        self._b_full = self._b_eq.copy()
        if self._b is not None:
            self._b_full += self._b

        # define gradient of absolute value of parallel magnetic field
        self._PBb = self._PB.dot(self._b)
        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._PBb += self._abs_b
        self._grad_PBb += self._grad_abs_b

        self._b_full = self._E2T.dot(self._b_full)
        self._PBb = self._E0T.dot(self._PBb)
        self._grad_PBb = self._E1T.dot(self._grad_PBb)

        self._b_full.update_ghost_regions()
        self._PBb.update_ghost_regions()
        self._grad_PBb.update_ghost_regions()

        if self._method == 'explicit':
            self._pusher_inputs = (self._epsilon,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)
        else:
            self._pusher_inputs = (self._epsilon, self._PBb._data,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._maxiter, self._tol)

        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

    @classmethod
    def options(cls):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',
                                  'discrete_gradient',
                                  'discrete_gradient_faster',
                                  'discrete_gradient_Itoh_Newton'],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        return dct
