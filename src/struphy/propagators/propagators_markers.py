'Only particle variables are updated.'


from numpy import array, polynomial, random

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.pic.pushing.pusher import Pusher
from struphy.pic.pushing.pusher import ButcherTableau
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium
from struphy.pic.particles import Particles6D, Particles5D, Particles3D
from struphy.pic.base import Particles
from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.io.setup import descend_options_dict


class PushEta(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p\,,

    for constant :math:`\mathbf v_p` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p\,. 

    Available algorithms:

    * ``rk4`` (4th order, default)
    * ``forward_euler`` (1st order)
    * ``heun2`` (2nd order)
    * ``rk2`` (2nd order)
    * ``heun3`` (3rd order)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = ['rk4', 'forward_euler', 'heun2', 'rk2', 'heun3']
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 particles: Particles,
                 *,
                 algo: str = options(default=True)['algo'],
                 bc_type: list = ['reflect', 'periodic', 'periodic']):

        super().__init__(particles)

        self._bc_type = bc_type

        # choose algorithm
        if algo == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif algo == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_eta_stage', n_stages=self._butcher.n_stages)

    def __call__(self, dt):
        # push markers
        self._pusher(self.particles[0], dt,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='last')

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushVxB(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \mathbf B\,,

    for fixed rotation vector :math:`\mathbf B`, given as a 2-form:

    .. math::

        \mathbf B =  \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}\,.

    Available algorithms: ``analytic``, ``implicit``.
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = ['analytic', 'implicit']
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 particles: Particles6D,
                 *,
                 algo: str = options(default=True)['algo'],
                 scale_fac: float = 1.,
                 b_eq: BlockVector | PolarVector,
                 b_tilde: BlockVector | PolarVector):

        super().__init__(particles)

        self._scale_fac = scale_fac
        self._b_eq = b_eq
        self._b_tilde = b_tilde

        # load pusher
        self._pusher = Pusher(self.derham, self.domain, 'push_vxb_' + algo)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops['2'].transpose()

    def __call__(self, dt):
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


class PushGuidingCenterBxEstar(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} = \frac{\mathbf E^* \times \mathbf b_0}{B_\parallel^*} (\mathbf X_p(t))   \,,

    where 

    .. math::

        \mathbf E^* = - \varepsilon \mu_p \nabla B_0\,,\qquad B^*_\parallel = B_0 + \varepsilon v_{\parallel,p} (\nabla \times \mathbf b_0) \cdot \mathbf b_0\,,

    in logical space given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\hat B_\parallel^{*3}} (\boldsymbol \eta_p(t)) \,.

    Available algorithms:

    * ``discrete_gradient`` (2nd order, implicit, default)
    * ``discrete_gradient_faster`` (1st order, implicit)
    * ``discrete_gradient_Itoh_Newton`` (2nd order, implicit)
    * ``forward_euler`` (1st order, explicit)
    * ``heun2`` (2nd order, explicit)
    * ``rk2`` (2nd order, explicit)
    * ``heun3`` (3rd order, explicit)
    * ``rk4`` (4th order, explicit)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['discrete_gradient',
                                  'discrete_gradient_faster',
                                  'discrete_gradient_Itoh_Newton',
                                  'rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3'],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        if default:
            dct = descend_options_dict(dct, [])
        
        return dct

    def __init__(self,
                 particles: Particles5D,
                 *,
                 magn_bckgr: MHDequilibrium,
                 epsilon: float = 1.,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        self._mpi_sort = algo['mpi_sort']
        self._verbose = algo['verbose']
        self._epsilon = epsilon

        # magnetic field
        b_eq = self.derham.P['2']([magn_bckgr.b2_1,
                                   magn_bckgr.b2_2,
                                   magn_bckgr.b2_3])

        self._abs_b = self.derham.P['0'](magn_bckgr.absB0)

        unit_b1 = self.derham.P['1']([magn_bckgr.unit_b1_1,
                                      magn_bckgr.unit_b1_2,
                                      magn_bckgr.unit_b1_3])

        unit_b2 = self.derham.P['2']([magn_bckgr.unit_b2_1,
                                      magn_bckgr.unit_b2_2,
                                      magn_bckgr.unit_b2_3])

        grad_abs_b = self.derham.P['1']([magn_bckgr.gradB1_1,
                                         magn_bckgr.gradB1_2,
                                         magn_bckgr.gradB1_3])

        if hasattr(magn_bckgr, 'curl_unit_b2_1'):
            curl_norm_b = self.derham.P['2']([magn_bckgr.curl_unit_b2_1,
                                              magn_bckgr.curl_unit_b2_2,
                                              magn_bckgr.curl_unit_b2_3])
        else:
            curl_norm_b = self.derham.curl.dot(unit_b1)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        E2T = self.derham.extraction_ops['2'].transpose()

        # new array allocation
        b_eq = E2T.dot(b_eq)
        unit_b1 = E1T.dot(unit_b1)
        unit_b2 = E2T.dot(unit_b2)
        self._abs_b = E0T.dot(self._abs_b)
        curl_norm_b = E2T.dot(curl_norm_b)
        grad_abs_b = E1T.dot(grad_abs_b)

        curl_norm_b.update_ghost_regions()
        grad_abs_b.update_ghost_regions()

        _eval_ker_names = []

        if algo['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif algo['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif algo['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval_gradI']
        elif algo['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval_gradI']

        elif algo['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval1',
                                'gc_bxEstar_' +
                                algo['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {algo["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_bxEstar_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (self._epsilon, b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
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
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'])

            self._pusher_inputs = (self._epsilon, self._abs_b._data,
                                   b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   algo['maxiter'], algo['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

        # update_weights
        if self.particles[0].control_variate:

            if self.particles[0].f0.coords == 'constants_of_motion':
                self.particles[0].save_constants_of_motion(
                    epsilon=self._epsilon, abs_B0=self._abs_b)

            self.particles[0].update_weights()


class PushGuidingCenterParallel(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\mathbf B^*}{B^*_\parallel}(\mathbf X_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E^* (\mathbf X_p(t)) \,,
            \end{aligned}
        \right.

    where

    .. math::

        \mathbf E^* = - \varepsilon \mu_p \nabla B_0\,,\qquad \mathbf B^* = \mathbf B_0 + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad  B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    in logical space given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,.
            \end{aligned}
        \right.

    Available algorithms:

    * ``discrete_gradient`` (2nd order, implicit)
    * ``discrete_gradient_faster`` (2nd order, implicit, default)
    * ``discrete_gradient_Itoh_Newton`` (2nd order, implicit)
    * ``forward_euler`` (1st order, explicit)
    * ``heun2`` (2nd order, explicit)
    * ``rk2`` (2nd order, explicit)
    * ``heun3`` (3rd order, explicit)
    * ``rk4`` (4th order, explicit)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['discrete_gradient_faster',
                                  'discrete_gradient',
                                  'discrete_gradient_Itoh_Newton',
                                  'rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',
                                  ],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        if default:
            dct = descend_options_dict(dct, [])
            
        return dct

    def __init__(self,
                 particles: Particles5D,
                 *,
                 magn_bckgr: MHDequilibrium,
                 epsilon: float = 1.,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        self._mpi_sort = algo['mpi_sort']
        self._verbose = algo['verbose']
        self._epsilon = epsilon

        b_eq = self.derham.P['2']([magn_bckgr.b2_1,
                                   magn_bckgr.b2_2,
                                   magn_bckgr.b2_3])

        self._abs_b = self.derham.P['0'](magn_bckgr.absB0)

        unit_b1 = self.derham.P['1']([magn_bckgr.unit_b1_1,
                                      magn_bckgr.unit_b1_2,
                                      magn_bckgr.unit_b1_3])

        unit_b2 = self.derham.P['2']([magn_bckgr.unit_b2_1,
                                      magn_bckgr.unit_b2_2,
                                      magn_bckgr.unit_b2_3])

        grad_abs_b = self.derham.P['1']([magn_bckgr.gradB1_1,
                                         magn_bckgr.gradB1_2,
                                         magn_bckgr.gradB1_3])

        if hasattr(magn_bckgr, 'curl_unit_b2_1'):
            curl_norm_b = self.derham.P['2']([magn_bckgr.curl_unit_b2_1,
                                              magn_bckgr.curl_unit_b2_2,
                                              magn_bckgr.curl_unit_b2_3])
        else:
            curl_norm_b = self.derham.curl.dot(unit_b1)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        E2T = self.derham.extraction_ops['2'].transpose()

        b_eq = E2T.dot(b_eq)
        unit_b1 = E1T.dot(unit_b1)
        unit_b2 = E2T.dot(unit_b2)
        self._abs_b = E0T.dot(self._abs_b)
        curl_norm_b = E2T.dot(curl_norm_b)
        grad_abs_b = E1T.dot(grad_abs_b)

        curl_norm_b.update_ghost_regions()
        grad_abs_b.update_ghost_regions()

        _eval_ker_names = []

        if algo['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif algo['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif algo['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval_gradI']

        elif algo['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval_gradI']

        elif algo['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval1',
                                'gc_Bstar_' +
                                algo['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {algo["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_Bstar_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (self._epsilon, b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
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
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'])

            self._pusher_inputs = (self._epsilon, self._abs_b._data,
                                   b_eq[0]._data, b_eq[1]._data, b_eq[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   unit_b2[0]._data, unit_b2[1]._data, unit_b2[2]._data,
                                   curl_norm_b[0]._data, curl_norm_b[1]._data, curl_norm_b[2]._data,
                                   grad_abs_b[0]._data, grad_abs_b[1]._data, grad_abs_b[2]._data,
                                   algo['maxiter'], algo['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

        # update_weights
        if self.particles[0].control_variate:

            if self.particles[0].f0.coords == 'constants_of_motion':
                self.particles[0].save_constants_of_motion(
                    epsilon=self._epsilon, abs_B0=self._abs_b)

            self.particles[0].update_weights()


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
    r"""Particle pushing step for the :math:`\mathbf b_0 \times \nabla B_\parallel` driftkinetic part in :class:`~struphy.models.hybrid.LinearMHDDriftkineticCC`,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= \varepsilon \frac{1}{B_\parallel^*} \mathbf b_0 \times \mu \nabla B_\parallel \,,
                \\
                \dot v_\parallel &= 0 \,.
            \end{aligned}
        \right.

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \varepsilon \mu_p \frac{1}{ B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  G^{-1}(\boldsymbol \eta_p) \hat{\mathbf b}^2_0(\boldsymbol \eta_p) \times G^{-1}(\boldsymbol \eta_p) \hat \nabla \hat{B}^0_\parallel (\boldsymbol \eta_p) \,,
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
        self._abs_b = self._E0T.dot(self._abs_b)

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

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].save_constants_of_motion(
                epsilon=self._epsilon, abs_B0=self._abs_b)
            self.particles[0].update_weights()

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


class PushDriftKineticParallelZeroEfield(Propagator):
    r"""Particle pushing step for the :math:`\mathbf B^*` driftkinetic part in :class:`~struphy.models.hybrid.LinearMHDDriftkineticCC`,

    Equation:

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X} &= v_\parallel \frac{\mathbf B^*}{B^*_\parallel} \,,
                \\
                \dot v_\parallel &= - \frac{\mathbf B^*}{B^*_\parallel} \cdot \mu \nabla B_\parallel \,,
            \end{aligned}
        \right.

    where :math:`\mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0` and :math:`B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*`.

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
        self._abs_b = self._E0T.dot(self._abs_b)

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

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].save_constants_of_motion(
                epsilon=self._epsilon, abs_B0=self._abs_b)
            self.particles[0].update_weights()

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


class PushDriftKineticBxEstar(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \dot{\mathbf X}_p =  \frac{1}{B_\parallel^*} \mathbf E^* \times \mathbf b_0 \,(\mathbf X_p) \,,

    where :math:`v_{\parallel,p}` is constant and

    .. math::

        \mathbf{E}^* &=  - \nabla \phi - \frac{\varepsilon}{Z}\,\mu \, \nabla |B_0|\,,
        \\[2mm]
        B^*_\parallel &= |B_0| + \frac{\varepsilon}{Z} v_\parallel (\nabla \times \mathbf{b}_0) \cdot  \mathbf{b}_0 \,,

    in logical space given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \dot{\boldsymbol \eta}_p =  \frac{1}{ \sqrt g\, \hat B^{*}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \hat{\mathbf E}^{*1}(\boldsymbol \eta_p, \mu_p) \times  \hat{\mathbf b}^1_0(\boldsymbol \eta_p) \,.

    Available algorithms:

    * ``rk4`` (4th order, explicit, default)
    * ``forward_euler`` (1st order, explicit)
    * ``heun2`` (2nd order, explicit)
    * ``rk2`` (2nd order, explicit)
    * ``heun3`` (3rd order, explicit)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        if default:
            dct = descend_options_dict(dct, [])
            
        return dct

    def __init__(self,
                 particles,
                 *,
                 phi: StencilVector,
                 magn_bckgr: MHDequilibrium | BraginskiiEquilibrium = None,
                 epsilon: float = 1.,
                 Z: int = 1,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        self._mpi_sort = algo['mpi_sort']
        self._verbose = algo['verbose']

        # magnetic field
        absB0 = self.derham.P['0'](magn_bckgr.absB0)

        gradB1 = self.derham.P['1']([magn_bckgr.gradB1_1,
                                     magn_bckgr.gradB1_2,
                                     magn_bckgr.gradB1_3])

        unit_b1 = self.derham.P['1']([magn_bckgr.unit_b1_1,
                                      magn_bckgr.unit_b1_2,
                                      magn_bckgr.unit_b1_3])

        curl_unit_b1 = self.derham.curl.dot(unit_b1)

        # expose phi for use in __call__
        self._phi = phi
        self._e1 = self.derham.Vh['1'].zeros()

        _eval_ker_names = []

        if algo['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif algo['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif algo['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstarwithPhi_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval_gradI']
        elif algo['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval_gradI']
        elif algo['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_bxEstar_' + algo['method']
            _eval_ker_names += ['gc_bxEstar_' +
                                algo['method'] + '_eval1',
                                'gc_bxEstar_' +
                                algo['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {algo["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_bxEstarWithPhi_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (gradB1[0]._data, gradB1[1]._data, gradB1[2]._data,
                                   absB0._data,
                                   curl_unit_b1[0]._data, curl_unit_b1[1]._data, curl_unit_b1[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   epsilon,
                                   Z,
                                   butcher.a, butcher.b, butcher.c)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'])

            self._pusher_inputs = (gradB1[0]._data, gradB1[1]._data, gradB1[2]._data,
                                   absB0._data,
                                   curl_unit_b1[0]._data, curl_unit_b1[1]._data, curl_unit_b1[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   epsilon,
                                   Z,
                                   algo['maxiter'], algo['tol'])

    def __call__(self, dt):
        """
        TODO
        """

        # efield=self._grad_phi
        e1 = self.derham.grad.dot(-self._phi, out=self._e1)
        e1.update_ghost_regions()

        self._pusher(self.particles[0], dt,
                     e1[0]._data, e1[1]._data, e1[2]._data,
                     *self._pusher_inputs,
                     mpi_sort=self._mpi_sort, verbose=self._verbose)


class PushDriftKineticParallel(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \left\{ 
            \begin{aligned} 
                \dot{\mathbf X}_p &= v_{\parallel,p} \frac{\mathbf B^*}{B^*_\parallel}(\mathbf X_p, v_{\parallel,p}) \,,
                \\[2mm]
                \dot v_{\parallel,p} &=  \frac{Z}{\varepsilon} \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E^* (\mathbf X_p, v_{\parallel,p}) \,,
            \end{aligned}
        \right.

    where 

    .. math::

        \mathbf{E}^* &=  - \nabla \phi - \frac{\varepsilon}{Z} \,\mu \, \nabla |B_0|\,,
        \\[2mm]
        \mathbf B^* &=  \mathbf B_0 + \frac{\varepsilon}{Z} v_\parallel \nabla \times \mathbf b_0\,,
        \\[2mm]
        B^*_\parallel &= |B_0| + \frac{\varepsilon}{Z} v_\parallel (\nabla \times \mathbf{b}_0) \cdot  \mathbf{b}_0 \,,

    in logical space given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \frac{1}{ \sqrt g\, \hat B^{*}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p} \,,
            \\[2mm]
            \dot v_{\parallel,\,p} &= \frac{Z}{\varepsilon}\frac{1}{ \sqrt g\, \hat B^{*}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \hat{\mathbf E}^{*1} (\boldsymbol \eta_p, v_{\parallel,\,p})\cdot \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \,.
        \end{aligned}

    Available algorithms:

    * ``rk4`` (4th order, explicit, default)
    * ``forward_euler`` (1st order, explicit)
    * ``heun2`` (2nd order, explicit)
    * ``rk2`` (2nd order, explicit)
    * ``heun3`` (3rd order, explicit)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['rk4',
                                  'forward_euler',
                                  'heun2',
                                  'rk2',
                                  'heun3',],
                       'maxiter': 20,
                       'tol': 1e-7,
                       'mpi_sort': 'each',
                       'verbose': False}
        if default:
            dct = descend_options_dict(dct, [])
            
        return dct

    def __init__(self,
                 particles: Particles5D,
                 *,
                 phi: StencilVector,
                 magn_bckgr: MHDequilibrium | BraginskiiEquilibrium = None,
                 epsilon: float = 1.,
                 Z: int = 1,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        # expose phi for use in __call__
        self._mpi_sort = algo['mpi_sort']
        self._verbose = algo['verbose']
        self._phi = phi
        self._e1 = self.derham.Vh['1'].zeros()

        # magnetic field
        b2_eq = self.derham.P['2']([magn_bckgr.b2_1,
                                    magn_bckgr.b2_2,
                                    magn_bckgr.b2_3])

        absB0 = self.derham.P['0'](magn_bckgr.absB0)

        unit_b1 = self.derham.P['1']([magn_bckgr.unit_b1_1,
                                      magn_bckgr.unit_b1_2,
                                      magn_bckgr.unit_b1_3])

        curl_unit_b1 = self.derham.curl.dot(unit_b1)

        # grad phi and grad absB0
        grad_absB0 = self.derham.grad.dot(absB0)
        grad_absB0.update_ghost_regions()

        _eval_ker_names = []

        if algo['method'] == 'forward_euler':
            _method = 'explicit'
            a = []
            b = [1.]
            c = [0.]
        elif algo['method'] == 'heun2':
            _method = 'explicit'
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo['method'] == 'rk2':
            _method = 'explicit'
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo['method'] == 'heun3':
            _method = 'explicit'
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo['method'] == 'rk4':
            _method = 'explicit'
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        elif algo['method'] == 'discrete_gradient':
            _method = 'implicit'
            _kernel_name = 'push_gc_BstarWithPhi_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval_gradI']

        elif algo['method'] == 'discrete_gradient_faster':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval_gradI']

        elif algo['method'] == 'discrete_gradient_Itoh_Newton':
            _method = 'implicit'
            _kernel_name = 'push_gc_Bstar_' + algo['method']
            _eval_ker_names += ['gc_Bstar_' +
                                algo['method'] + '_eval1',
                                'gc_Bstar_' +
                                algo['method'] + '_eval2']
        else:
            raise NotImplementedError(
                f'Chosen method {algo["method"]} is not implemented.')

        if _method == 'explicit':
            butcher = ButcherTableau(a, b, c)

            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  'push_gc_BstarWithPhi_explicit_multistage',
                                  n_stages=butcher.n_stages)

            self._pusher_inputs = (b2_eq[0]._data, b2_eq[1]._data, b2_eq[2]._data,
                                   curl_unit_b1[0]._data, curl_unit_b1[1]._data, curl_unit_b1[2]._data,
                                   grad_absB0[0]._data, grad_absB0[1]._data, grad_absB0[2]._data,
                                   unit_b1[0]._data, unit_b1[0]._data, unit_b1[0]._data,
                                   absB0._data,
                                   epsilon,
                                   Z,
                                   butcher.a, butcher.b, butcher.c)
        else:
            self._pusher = Pusher(self.derham,
                                  self.domain,
                                  _kernel_name,
                                  init_kernel=True,
                                  eval_kernels_names=_eval_ker_names,
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'])

            self._pusher_inputs = (b2_eq[0]._data, b2_eq[1]._data, b2_eq[2]._data,
                                   curl_unit_b1[0]._data, curl_unit_b1[1]._data, curl_unit_b1[2]._data,
                                   grad_absB0[0]._data, grad_absB0[1]._data, grad_absB0[2]._data,
                                   unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                   absB0._data,
                                   epsilon,
                                   Z,
                                   algo['maxiter'], algo['tol'])

    def __call__(self, dt):
        """
        TODO
        """
        # efield=self._grad_phi
        e1 = self.derham.grad.dot(-self._phi, out=self._e1)
        e1.update_ghost_regions()

        self._pusher(self.particles[0], dt,
                     e1[0]._data, e1[1]._data, e1[2]._data,
                     *self._pusher_inputs, mpi_sort=self._mpi_sort, verbose=self._verbose)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushDeterministicDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = - D \, \frac{\nabla u}{ u}\mathbf (\mathbf x_p(t))\,,

    in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = - G\, D \, \frac{\nabla \Pi^0_{L^2}u_h}{\Pi^0_{L^2} u_h}\mathbf (\boldsymbol \eta_p(t))\,, 
        \qquad [\Pi^0_{L^2, ijk} u_h](\boldsymbol \eta_p) = \frac 1N \sum_{p} w_p \boldsymbol \Lambda^0_{ijk}(\boldsymbol \eta_p)\,,

    where :math:`D>0` is a positive diffusion coefficient. 
    Available algorithms:

    * ``rk4`` (4th order, default)
    * ``forward_euler`` (1st order)
    * ``heun2`` (2nd order)
    * ``rk2`` (2nd order)
    * ``heun3`` (3rd order)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = ['rk4', 'forward_euler', 'heun2', 'rk2', 'heun3']
        dct['diffusion_coefficient'] = 1.
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 particles: Particles3D,
                 *,
                 algo: str = options(default=True)['algo'],
                 bc_type: list = ['periodic', 'periodic', 'periodic'],
                 diffusion_coefficient: float = options()['diffusion_coefficient']):

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        super().__init__(particles)

        self._bc_type = bc_type
        self._diffusion = diffusion_coefficient

        # choose algorithm
        if algo == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif algo == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._u_on_grid = AccumulatorVector(
            self.derham, self.domain, "H1", "charge_density_0form")

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_deterministic_diffusion_stage', n_stages=self._butcher.n_stages)

        self._tmp = self.derham.Vh['1'].zeros()

    def __call__(self, dt):
        """
        TODO
        """

        self._u_on_grid.accumulate(self.particles[0], self.particles[0].vdim)

        pi_u = self._u_on_grid.vectors[0]
        grad_pi_u = self.derham.grad.dot(pi_u, out=self._tmp)
        grad_pi_u.update_ghost_regions()

        # push markers
        self._pusher(self.particles[0], dt,
                     pi_u._data,
                     grad_pi_u[0]._data, grad_pi_u[1]._data, grad_pi_u[2]._data,
                     self._diffusion,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='last')

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushRandomDiffusion(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \textnormal d \mathbf x_p(t) = \sqrt{2 D} \, \textnormal d \mathbf B_{t}\,,

    where :math:`D>0` is a positive diffusion coefficient and :math:`\textnormal d \mathbf B_{t}` is a Wiener process,

    .. math::

        \mathbf B_{t + \Delta t} - \mathbf B_{t} = \sqrt{\Delta t} \,\mathcal N(0;1)\,,

    with :math:`\mathcal N(0;1)` denoting the standard normal distribution with mean zero and variance one.

    Available algorithms:

    * ``forward_euler`` (1st order)
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = ['forward_euler']
        dct['diffusion_coefficient'] = 1.
        if default:
            dct = descend_options_dict(dct, [])
        return dct

    def __init__(self,
                 particles: Particles3D,
                 algo: str = options(default=True)['algo'],
                 bc_type: list = ['periodic', 'periodic', 'periodic'],
                 diffusion_coefficient: float = options()['diffusion_coefficient']):

        super().__init__(particles)

        self._bc_type = bc_type
        self._diffusion = diffusion_coefficient

        # choose algorithm
        if algo == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_random_diffusion_stage', n_stages=self._butcher.n_stages)

        # self._tmp = self.derham.Vh['1'].zeros()

        self._noise = array(self.particles[0].markers[0:3])

        self._mean = [0, 0, 0]
        self._cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def __call__(self, dt):
        """
        TODO
        """

        self._noise = random.multivariate_normal(
            self._mean, self._cov, len(self.particles[0].markers))

        # push markers
        self._pusher(self.particles[0], dt,
                     self._noise, self._diffusion,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='last')

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()
