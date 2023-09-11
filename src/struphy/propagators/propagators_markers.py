'Only particle variables are updated.'


from numpy import array, polynomial

from psydac.linalg.block import BlockVector

from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.pic.particles import Particles6D, Particles5D
from struphy.pic.pusher import Pusher, Pusher_iteration_Gonzalez, Pusher_iteration_Itoh
from struphy.pic.pusher import ButcherTableau
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
                          'f0': Maxwellian6DUniform()}

        params = set_defaults(params, params_default)

        if params['f0'] is not None:
            assert callable(params['f0'])

        self._bc_type = params['bc_type']
        self._f0 = params['f0']

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
                              'push_eta_stage', self._butcher.n_stages)

    def __call__(self, dt):
        """
        TODO
        """

        # push markers
        self._pusher(self.particles[0], dt,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     mpi_sort='last')

        # update_weights
        if self._f0 is not None:
            self.particles[0].update_weights(self._f0)


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
                          'b_tilde': None,
                          'f0': Maxwellian6DUniform()}

        params = set_defaults(params, params_default)

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        if params['f0'] is not None:
            assert callable(params['f0'])

        self._scale_fac = params['scale_fac']
        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']
        self._f0 = params['f0']

        # load pusher
        kernel_name = 'push_vxb_' + params['algo']
        self._pusher = Pusher(self.derham, self.domain, kernel_name)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.E['2'].transpose()

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
        if self._f0 is not None:
            self.particles[0].update_weights(self._f0)


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
        self._E2T = self.derham.E['2'].transpose()

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

    derham : struphy.psydac_api.psydac_derham.Derham
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


class StepPushGuidingCenter1(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{\mu}{\kappa B^*_\parallel}  G^{-1}(\eta_p(t)) \hat{\mathbb{b}}^2_0 \times G^{-1}(\eta_p(t)) \hat \nabla |\hat{B}^0_0| \,,

        \dot v_\parallel &= 0 \,.

    for each marker :math:`p` in markers array. Available algorithms:

        Explicit:
        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)

        Implicit:
        * discrete_gradients
        * discrete_gradients_faster

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
        params_default = {'kappa': 1.,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'integrator': 'implicit',
                          'method': 'discrete_gradient_faster',
                          'maxiter': 10,
                          'tol': 1e-12,
                          }

        params = set_defaults(params, params_default)

        self._kappa = params['kappa']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._grad_abs_b = self.derham.grad.dot(self._abs_b)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.E['0'].transpose()
        self._E1T = self.derham.E['1'].transpose()
        self._E2T = self.derham.E['2'].transpose()

        self._b_eq = self._E2T.dot(self._b_eq)
        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._abs_b = self._E0T.dot(self._abs_b)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)
        self._grad_abs_b = self._E1T.dot(self._grad_abs_b)

        self._curl_norm_b.update_ghost_regions()
        self._grad_abs_b.update_ghost_regions()

        if params['integrator'] == 'explicit':

            if params['method'] == 'forward_euler':
                a = []
                b = [1.]
                c = [0.]
            elif params['method'] == 'heun2':
                a = [1.]
                b = [1/2, 1/2]
                c = [0., 1.]
            elif params['method'] == 'rk2':
                a = [1/2]
                b = [0., 1.]
                c = [0., 1/2]
            elif params['method'] == 'heun3':
                a = [1/3, 2/3]
                b = [1/4, 0., 3/4]
                c = [0., 1/3, 2/3]
            elif params['method'] == 'rk4':
                a = [1/2, 1/2, 1.]
                b = [1/6, 1/3, 1/3, 1/6]
                c = [0., 1/2, 1/2, 1.]
            else:
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                self.derham, self.domain, 'push_gc1_explicit_stage', self._butcher.n_stages)

            self._pusher_inputs = (self._kappa, self._b_eq[0]._data, self._b_eq[1]._data, self._b_eq[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)

        elif params['integrator'] == 'implicit':

            if params['method'] == 'discrete_gradients':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc1_discrete_gradients', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_faster':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc1_discrete_gradients_faster', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_Itoh_Newton':
                self._pusher = Pusher_iteration_Itoh(
                    self.derham, self.domain, 'push_gc1_discrete_gradients_Itoh_Newton', params['maxiter'], params['tol'])

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

            self._pusher_inputs = (self._kappa, self._abs_b._data,
                                   self._b_eq[0]._data, self._b_eq[1]._data, self._b_eq[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data)

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort='each', verbose=True)

        # save magnetic field at each particles' position
        self.particles[0].save_magnetic_energy(self.derham, self._abs_b)


class StepPushGuidingCenter2(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{1}{B^*_\parallel} \frac{1}{\sqrt{g}(\eta_p(t))}\hat{\mathbf{B}}^{*2} v_\parallel \,,

        \dot v_\parallel &= - \frac{\mu}{B^*_\parallel} \frac{1}{\sqrt{g}(\eta_p(t))}\hat{\mathbf{B}}^{*2} \cdot \hat \nabla |\hat{B}^0_0| \,.

    for each marker :math:`p` in markers array. Available algorithms:

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
        params_default = {'kappa': 1.,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'integrator': 'implicit',
                          'method': 'discrete_gradient_faster',
                          'maxiter': 10,
                          'tol': 1e-12,
                          }

        params = set_defaults(params, params_default)

        self._kappa = params['kappa']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._grad_abs_b = self.derham.grad.dot(self._abs_b)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.E['0'].transpose()
        self._E1T = self.derham.E['1'].transpose()
        self._E2T = self.derham.E['2'].transpose()

        self._b_eq = self._E2T.dot(self._b_eq)
        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._abs_b = self._E0T.dot(self._abs_b)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)
        self._grad_abs_b = self._E1T.dot(self._grad_abs_b)

        self._curl_norm_b.update_ghost_regions()
        self._grad_abs_b.update_ghost_regions()

        if params['integrator'] == 'explicit':

            if params['method'] == 'forward_euler':
                a = []
                b = [1.]
                c = [0.]
            elif params['method'] == 'heun2':
                a = [1.]
                b = [1/2, 1/2]
                c = [0., 1.]
            elif params['method'] == 'rk2':
                a = [1/2]
                b = [0., 1.]
                c = [0., 1/2]
            elif params['method'] == 'heun3':
                a = [1/3, 2/3]
                b = [1/4, 0., 3/4]
                c = [0., 1/3, 2/3]
            elif params['method'] == 'rk4':
                a = [1/2, 1/2, 1.]
                b = [1/6, 1/3, 1/3, 1/6]
                c = [0., 1/2, 1/2, 1.]
            else:
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                self.derham, self.domain, 'push_gc2_explicit_stage', self._butcher.n_stages)

            self._pusher_inputs = (self._kappa, self._b_eq[0]._data, self._b_eq[1]._data, self._b_eq[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)

        elif params['integrator'] == 'implicit':

            if params['method'] == 'discrete_gradients':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc2_discrete_gradients', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_faster':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc2_discrete_gradients_faster', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_Itoh_Newton':
                self._pusher = Pusher_iteration_Itoh(
                    self.derham, self.domain, 'push_gc2_discrete_gradients_Itoh_Newton', params['maxiter'], params['tol'])

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

            self._pusher_inputs = (self._kappa, self._abs_b._data,
                                   self._b_eq[0]._data, self._b_eq[1]._data, self._b_eq[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data)

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort='each', verbose=True)

        # save magnetic field at each particles' position
        self.particles[0].save_magnetic_energy(self.derham, self._abs_b)


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


class StepPushDriftKinetic1(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{\mu}{\kappa B^*_\parallel}  G^{-1}(\eta_p(t)) \hat{\mathbf{b}}^2_0 \times G^{-1}(\eta_p(t)) \hat \nabla |\mathcal{P}_B \hat{\mathbf{B}}^2| \,,

        \dot v_\parallel &= 0 \,.

    for each marker :math:`p` in markers array. Available algorithms:

        Explicit:
        * forward_euler (1st order)
        * heun2 (2nd order)
        * rk2 (2nd order)
        * heun3 (3rd order)
        * rk4 (4th order)

        Implicit:
        * discrete_gradients
        * discrete_gradients_faster

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
        params_default = {'kappa': 1.,
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'integrator': 'implicit',
                          'method': 'discrete_gradient_faster',
                          'maxiter': 10,
                          'tol': 1e-12,
                          }

        params = set_defaults(params, params_default)

        self._kappa = params['kappa']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)

        # sum up total magnetic field
        self._b_full = self._b_eq.copy()

        # define gradient of absolute value of parallel magnetic field
        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(self._b_full)
        self._grad_PBb = self.derham.grad.dot(self._PBb)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.E['0'].transpose()
        self._E1T = self.derham.E['1'].transpose()
        self._E2T = self.derham.E['2'].transpose()

        self._b_eq = self._E2T.dot(self._b_eq)
        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)

        self._curl_norm_b.update_ghost_regions()

        self._integrator = params['integrator']

        if params['integrator'] == 'explicit':

            if params['method'] == 'forward_euler':
                a = []
                b = [1.]
                c = [0.]
            elif params['method'] == 'heun2':
                a = [1.]
                b = [1/2, 1/2]
                c = [0., 1.]
            elif params['method'] == 'rk2':
                a = [1/2]
                b = [0., 1.]
                c = [0., 1/2]
            elif params['method'] == 'heun3':
                a = [1/3, 2/3]
                b = [1/4, 0., 3/4]
                c = [0., 1/3, 2/3]
            elif params['method'] == 'rk4':
                a = [1/2, 1/2, 1.]
                b = [1/6, 1/3, 1/3, 1/6]
                c = [0., 1/2, 1/2, 1.]
            else:
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                self.derham, self.domain, 'push_gc1_explicit_stage', self._butcher.n_stages)

        elif params['integrator'] == 'implicit':

            if params['method'] == 'discrete_gradients':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc1_discrete_gradients', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_faster':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc1_discrete_gradients_faster', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_Itoh_Newton':
                self._pusher = Pusher_iteration_Itoh(
                    self.derham, self.domain, 'push_gc1_discrete_gradients_Itoh_Newton', params['maxiter'], params['tol'])

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    def __call__(self, dt):
        """
        TODO
        """

        # sum up total magnetic field
        self._b_full = self._b_eq.copy()
        if self._b is not None:
            self._b_full += self._b

        # define gradient of absolute value of parallel magnetic field
        self._PBb = self._PB.dot(self._b_full)
        self._grad_PBb = self.derham.grad.dot(self._PBb)

        self._b_full = self._E2T.dot(self._b_full)
        self._PBb = self._E0T.dot(self._PBb)
        self._grad_PBb = self._E1T.dot(self._grad_PBb)

        self._b_full.update_ghost_regions()
        self._PBb.update_ghost_regions()
        self._grad_PBb.update_ghost_regions()

        if self._integrator == 'explicit':
            self._pusher_inputs = (self._kappa,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)
        else:
            self._pusher_inputs = (self._kappa, self._PBb._data,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data)

        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort='each', verbose=True)


class StepPushDriftKinetic2(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{1}{B^*_\parallel} \frac{1}{\sqrt{g}(\eta_p(t))}\hat{\mathbf{B}}^{*2} v_\parallel \,,

        \dot v_\parallel &= - \frac{\mu}{B^*_\parallel} \frac{1}{\sqrt{g}(\eta_p(t))}\hat{\mathbf{B}}^{*2} \cdot \hat \nabla |\mathcal{P}_B \hat{\mathbf{B}}^2| \,.

    for each marker :math:`p` in markers array. Available algorithms:

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
        params_default = {'kappa': 1.,
                          'b': None,
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'integrator': 'implicit',
                          'method': 'discrete_gradient_faster',
                          'maxiter': 10,
                          'tol': 1e-12,
                          }

        params = set_defaults(params, params_default)

        self._kappa = params['kappa']
        self._b = params['b']
        self._b_eq = params['b_eq']
        self._unit_b1 = params['unit_b1']
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)

        # sum up total magnetic field
        self._b_full = self._b_eq.copy()

        # define gradient of absolute value of parallel magnetic field
        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(self._b_full)

        self._grad_PBb = self.derham.grad.dot(self._PBb)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E0T = self.derham.E['0'].transpose()
        self._E1T = self.derham.E['1'].transpose()
        self._E2T = self.derham.E['2'].transpose()

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)

        self._curl_norm_b.update_ghost_regions()

        self._integrator = params['integrator']

        if params['integrator'] == 'explicit':

            if params['method'] == 'forward_euler':
                a = []
                b = [1.]
                c = [0.]
            elif params['method'] == 'heun2':
                a = [1.]
                b = [1/2, 1/2]
                c = [0., 1.]
            elif params['method'] == 'rk2':
                a = [1/2]
                b = [0., 1.]
                c = [0., 1/2]
            elif params['method'] == 'heun3':
                a = [1/3, 2/3]
                b = [1/4, 0., 3/4]
                c = [0., 1/3, 2/3]
            elif params['method'] == 'rk4':
                a = [1/2, 1/2, 1.]
                b = [1/6, 1/3, 1/3, 1/6]
                c = [0., 1/2, 1/2, 1.]
            else:
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                self.derham, self.domain, 'push_gc2_explicit_stage', self._butcher.n_stages)

        elif params['integrator'] == 'implicit':

            if params['method'] == 'discrete_gradients':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc2_discrete_gradients', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_faster':
                self._pusher = Pusher_iteration_Gonzalez(
                    self.derham, self.domain, 'push_gc2_discrete_gradients_faster', params['maxiter'], params['tol'])

            elif params['method'] == 'discrete_gradients_Itoh_Newton':
                self._pusher = Pusher_iteration_Itoh(
                    self.derham, self.domain, 'push_gc2_discrete_gradients_Itoh_Newton', params['maxiter'], params['tol'])

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    def __call__(self, dt):
        """
        TODO
        """
        # sum up total magnetic field
        self._b_full = self._b_eq.copy()
        if self._b is not None:
            self._b_full += self._b

        # define gradient of absolute value of parallel magnetic field
        self._PBb = self._PB.dot(self._b_full)
        self._grad_PBb = self.derham.grad.dot(self._PBb)

        self._b_full = self._E2T.dot(self._b_full)
        self._PBb = self._E0T.dot(self._PBb)
        self._grad_PBb = self._E1T.dot(self._grad_PBb)

        self._b_full.update_ghost_regions()
        self._PBb.update_ghost_regions()
        self._grad_PBb.update_ghost_regions()

        if self._integrator == 'explicit':
            self._pusher_inputs = (self._kappa,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)
        else:
            self._pusher_inputs = (self._kappa, self._PBb._data,
                                   self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                                   self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                   self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data)

        self._pusher(self.particles[0], dt,
                     *self._pusher_inputs, mpi_sort='each', verbose=True)
