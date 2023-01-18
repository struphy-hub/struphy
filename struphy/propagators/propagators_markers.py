from numpy import array

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.polar.basic import PolarVector

from struphy.propagators.base import Propagator

from struphy.pic.pusher import Pusher, Pusher_iteration
from struphy.pic.pusher import ButcherTableau

class StepPushEta(Propagator):
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

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    algo : str
        The used algorithm.

    bc : list[str]
        Kinetic boundary conditions in each direction.
    """

    def __init__(self, particles, derham, domain, algo, bc):

        self._particles = particles
        self._bc = bc

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
        self._pusher = Pusher(derham, domain,
                              'push_eta_stage', self._butcher.n_stages)

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self._particles, dt,
                     self._butcher.a, self._butcher.b, self._butcher.c,
                     bc=self._bc, mpi_sort='last')


class StepPushVxB(Propagator):
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

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    algo : str
        The used algorithm.

    *b_vectors : psydac.linalg.block.BlockVector | struphy.polar.basic.PolarVector
        FE coefficients of several magnetic fields (2-form) (typically static and dynamical magnetic field).
    """

    def __init__(self, particles, derham, domain, algo, *b_vectors):

        self._particles = particles

        # load pusher
        kernel_name = 'push_vxb_' + algo

        self._pusher = Pusher(derham, domain, kernel_name)

        # magnetic field vectors
        for b in b_vectors:
            assert isinstance(b, (BlockVector, PolarVector))

        self._b_vectors = b_vectors

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = derham.E['2'].transpose()

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """

        # sum up total magnetic field
        b_full = self._b_vectors[0].space.zeros()

        for b in self._b_vectors:
            b_full += b

        # extract coefficients to tensor product space
        b_full = self._E2T.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()

        # call pusher kernel
        self._pusher(self._particles, dt,
                     b_full[0]._data,
                     b_full[1]._data,
                     b_full[2]._data)


class StepPushpxB_hybrid(Propagator):
    """
    TODO
    
    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.
    """

    def __init__(self, particles, derham, domain, algo, *field_vectors):

        self._C = derham.curl

        self._particles = particles

        # load pusher
        kernel_name = 'push_pxb_' + algo

        self._pusher = Pusher(derham, domain, kernel_name)

        # magnetic field vectors
        for b in field_vectors:
            assert isinstance(b, (BlockVector, PolarVector))

        self._field_vectors = field_vectors

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = derham.E['2'].transpose()

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """

        # calculate curl_A
        curl_A = self._C.dot(self._field_vectors[0])
        
        # check if ghost regions are synchronized
        for i in range(3):
            if not curl_A[i].ghost_regions_in_sync: curl_A[i].update_ghost_regions()

        # sum up total magnetic field
        b_full = self._field_vectors[1].space.zeros()

        for b in self._field_vectors:
            b_full = curl_A + self._field_vectors[1]

        # extract coefficients to tensor product space
        b_full = self._E2T.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()
        self._field_vectors[0].update_ghost_regions()

        # call pusher kernel
        self._pusher(self._particles, dt,
                     b_full[0]._data,
                     b_full[1]._data,
                     b_full[2]._data,
                     self._field_vectors[0][0]._data,
                     self._field_vectors[0][1]._data,
                     self._field_vectors[0][2]._data)


class StepPushEtaPC(Propagator):
    r'''Step for the update of particles' positions with the RK4 method which solves

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}})

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

    def __init__(self, particles, derham, domain, u, u_space, coupling, bc):

        assert isinstance(u, BlockVector)

        self._derham = derham
        self._domain = domain
        self._u = u
        self._particles = particles
        self._u_space = u_space
        self._bc = bc

        # call Pusher class
        if coupling == 'full':
            if self._u_space == 'Hcurl':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_Hcurl_full', n_stages=4)

            elif self._u_space == 'Hdiv':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_Hdiv_full', n_stages=4)

            elif self._u_space == 'H1vec':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_H1vec_full', n_stages=4)

            else:
                raise ValueError('Given u_space does not exist!')

        elif coupling == 'perp':
            if self._u_space == 'Hcurl':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_Hcurl', n_stages=4)

            elif self._u_space == 'Hdiv':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_Hdiv', n_stages=4)

            elif self._u_space == 'H1vec':
                self._pusher = Pusher(
                    self._derham, self._domain, 'push_pc_eta_rk4_H1vec', n_stages=4)

            else:
                raise ValueError('Given u_space does not exist!')

        else:
            raise NotImplementedError(
                'Given coupling scheme is not implemented!')

    @property
    def variables(self):
        return

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

        self._pusher(self._particles, dt,
                     self._u[0]._data, self._u[1]._data, self._u[2]._data,
                     bc=self._bc, mpi_sort='last')


class StepPushGuidingCenter1(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{\varepsilon \mu}{B^*_\parallel}  G^{-1} \mathbb{b}_{0, \otimes}G^{-1} \hat \nabla |\hat B_0|^0 \,,

        \dot v_\parallel &= 0 \,.

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

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    algo : str
        The used algorithm.

    bc : list[str]
        Kinetic boundary conditions in each direction.
    """

    def __init__(self, particles, derham, domain, epsilon, b, norm_b1, norm_b2, abs_b, algo, integrator, bc, maxiter=100, tol=1.e-8):

        self._particles = particles
        self._epsilon = epsilon
        self._bc = bc
        self._b = b
        self._norm_b1 = norm_b1
        self._norm_b2 = norm_b2
        self._abs_b = abs_b
        self._derham = derham

        self._curl_norm_b = derham.curl.dot(norm_b1)
        self._grad_abs_b = derham.grad.dot(abs_b)

        if not self._b[0].ghost_regions_in_sync:
            self._b[0].update_ghost_regions()
        if not self._b[1].ghost_regions_in_sync:
            self._b[1].update_ghost_regions()
        if not self._b[2].ghost_regions_in_sync:
            self._b[2].update_ghost_regions()
        if not self._abs_b.ghost_regions_in_sync:
            self._abs_b.update_ghost_regions()
        if not self._curl_norm_b[0].ghost_regions_in_sync:
            self._curl_norm_b[0].update_ghost_regions()
        if not self._curl_norm_b[1].ghost_regions_in_sync:
            self._curl_norm_b[1].update_ghost_regions()
        if not self._curl_norm_b[2].ghost_regions_in_sync:
            self._curl_norm_b[2].update_ghost_regions()
        if not self._grad_abs_b[0].ghost_regions_in_sync:
            self._grad_abs_b[0].update_ghost_regions()
        if not self._grad_abs_b[1].ghost_regions_in_sync:
            self._grad_abs_b[1].update_ghost_regions()
        if not self._grad_abs_b[2].ghost_regions_in_sync:
            self._grad_abs_b[2].update_ghost_regions()

        if integrator == 'explicit':

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
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                derham, domain, 'push_gc1_explicit_stage', self._butcher.n_stages)

            self._pusher_inputs = (self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)

        elif integrator == 'implicit':

            if algo == 'discrete_gradients':
                self._pusher = Pusher_iteration(
                    derham, domain, 'push_gc1_discrete_gradients_stage', maxiter, tol)

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

            self._pusher_inputs = (self._abs_b._data,
                                   self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data)

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self._particles, dt, self._epsilon,
                     *self._pusher_inputs,
                     bc=self._bc, mpi_sort='each', verbose=False)


class StepPushGuidingCenter2(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{1}{B^*_\parallel} \mathbf{B}^* v_\parallel \,,

        \dot v_\parallel &= - \mu \frac{1}{B^*_\parallel} \mathbf{B}^* \cdot \nabla |B_0| \,.

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

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    algo : str
        The used algorithm.

    bc : list[str]
        Kinetic boundary conditions in each direction.
    """

    def __init__(self, particles, derham, domain, epsilon, b, norm_b1, norm_b2, abs_b, algo, integrator, bc, maxiter=100, tol=1.e-8):

        self._particles = particles
        self._epsilon = epsilon
        self._bc = bc
        self._b = b
        self._norm_b1 = norm_b1
        self._norm_b2 = norm_b2
        self._abs_b = abs_b
        self._derham = derham

        self._curl_norm_b = derham.curl.dot(norm_b1)
        self._grad_abs_b = derham.grad.dot(abs_b)

        if not self._b[0].ghost_regions_in_sync:
            self._b[0].update_ghost_regions()
        if not self._b[1].ghost_regions_in_sync:
            self._b[1].update_ghost_regions()
        if not self._b[2].ghost_regions_in_sync:
            self._b[2].update_ghost_regions()
        if not self._abs_b.ghost_regions_in_sync:
            self._abs_b.update_ghost_regions()
        if not self._curl_norm_b[0].ghost_regions_in_sync:
            self._curl_norm_b[0].update_ghost_regions()
        if not self._curl_norm_b[1].ghost_regions_in_sync:
            self._curl_norm_b[1].update_ghost_regions()
        if not self._curl_norm_b[2].ghost_regions_in_sync:
            self._curl_norm_b[2].update_ghost_regions()
        if not self._grad_abs_b[0].ghost_regions_in_sync:
            self._grad_abs_b[0].update_ghost_regions()
        if not self._grad_abs_b[1].ghost_regions_in_sync:
            self._grad_abs_b[1].update_ghost_regions()
        if not self._grad_abs_b[2].ghost_regions_in_sync:
            self._grad_abs_b[2].update_ghost_regions()

        if integrator == 'explicit':

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
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                derham, domain, 'push_gc2_explicit_stage', self._butcher.n_stages)

            self._pusher_inputs = (self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)

        elif integrator == 'implicit':

            if algo == 'discrete_gradients':
                self._pusher = Pusher_iteration(
                    derham, domain, 'push_gc2_discrete_gradients_stage', maxiter, tol)

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

            self._pusher_inputs = (self._abs_b._data,
                                   self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data)

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self._particles, dt, self._epsilon,
                     *self._pusher_inputs,
                     bc=self._bc, mpi_sort='each', verbose=False)


class StepPushGuidingCenter(Propagator):
    r"""Solves

    .. math::

        \dot{\mathbf X} &= \frac{1}{B^*_\parallel} \mathbf{B}^* v_\parallel + \frac{\varepsilon \mu}{B^*_\parallel}  G^{-1} \mathbb{b}_{0, \otimes}G^{-1} \hat \nabla |\hat B_0|^0 \,,

        \dot v_\parallel &= - \mu \frac{1}{B^*_\parallel} \mathbf{B}^* \cdot \nabla |B_0| \,.

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

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.

    algo : str
        The used algorithm.

    bc : list[str]
        Kinetic boundary conditions in each direction.
    """

    def __init__(self, particles, derham, domain, epsilon, b, norm_b1, norm_b2, abs_b, algo, integrator, bc, maxiter=100, tol=1.e-8):

        self._particles = particles
        self._epsilon = epsilon
        self._bc = bc
        self._b = b
        self._norm_b1 = norm_b1
        self._norm_b2 = norm_b2
        self._abs_b = abs_b
        self._derham = derham

        self._curl_norm_b = derham.curl.dot(norm_b1)
        self._grad_abs_b = derham.grad.dot(abs_b)

        if not self._curl_norm_b[0].ghost_regions_in_sync:
            self._curl_norm_b[0].update_ghost_regions()
        if not self._curl_norm_b[1].ghost_regions_in_sync:
            self._curl_norm_b[1].update_ghost_regions()
        if not self._curl_norm_b[2].ghost_regions_in_sync:
            self._curl_norm_b[2].update_ghost_regions()
        if not self._grad_abs_b[0].ghost_regions_in_sync:
            self._grad_abs_b[0].update_ghost_regions()
        if not self._grad_abs_b[1].ghost_regions_in_sync:
            self._grad_abs_b[1].update_ghost_regions()
        if not self._grad_abs_b[2].ghost_regions_in_sync:
            self._grad_abs_b[2].update_ghost_regions()

        if integrator == 'explicit':
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
                raise NotImplementedError(
                    'Chosen algorithm is not implemented.')

            self._butcher = ButcherTableau(a, b, c)
            self._pusher = Pusher(
                derham, domain, 'push_gc_explicit_stage', self._butcher.n_stages)

            self._pusher_inputs = (self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data,
                                   self._butcher.a, self._butcher.b, self._butcher.c)

        elif integrator == 'implicit':

            if algo == 'discrete_gradients':
                self._pusher = Pusher_iteration(
                    derham, domain, 'push_gc_discrete_gradients_stage', maxiter, tol)

            else:
                raise NotImplementedError(
                    'Chosen implicit method is not implemented.')

            self._pusher_inputs = (self._abs_b._data,
                                   self._b[0]._data, self._b[1]._data, self._b[2]._data,
                                   self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                                   self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                                   self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                   self._grad_abs_b[0]._data, self._grad_abs_b[1]._data, self._grad_abs_b[2]._data)

        else:
            raise NotImplementedError('Chosen integrator is not implemented.')

    @property
    def variables(self):
        return self._particles

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self._particles, dt, self._epsilon,
                     *self._pusher_inputs,
                     bc=self._bc, mpi_sort='each', verbose=False)


class StepStaticEfield(Propagator):
    r'''Solve the following system

    .. math::

        \frac{\text{d} \mathbf{\eta}_p}{\text{d} t} & = DL^{-1} \mathbf{v}_p \,,

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} & = DL^{-T} \mathbf{E}_0

    which is solved by an average discrete gradient method, implicitly iterating
    over :math:`k` (for every particle :math:`p`):

    .. math::

        \mathbf{\eta}^{n+1}_{k+1} = \mathbf{\eta}^n + \frac{\Delta t}{2} DL^{-1}
        \left( \frac{\mathbf{\eta}^{n+1}_k + \mathbf{\eta}^n }{2} \right) \left( \mathbf{v}^{n+1}_k + \mathbf{v}^n \right) \,,

        \mathbf{v}^{n+1}_{k+1} = \mathbf{v}^n + \Delta t DL^{-1}\left(\mathbf{\eta}^n\right)
        \int_0^1 \left[ \mathbb{\Lambda}\left( \eta^n + \tau (\mathbf{\eta}^{n+1}_k - \mathbf{\eta}^n) \right) \right]^T \mathbf{e}_0 \, \text{d} \tau

    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete Derham complex.

    domain : struphy.geometry.domains
        Mapping info for evaluating metric coefficients.
    
    e_background : TODO
    '''

    def __init__(self, particles, derham, domain, e_background):
        
        from numpy import polynomial, floor

        self._domain = domain
        self._derham = derham
        self._particles = particles
        self._e_bg = e_background

        pn1 = derham.p[0]
        pd1 = pn1 - 1
        pn2 = derham.p[1]
        pd2 = pn2 - 1
        pn3 = derham.p[2]
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

        self._pusher = Pusher(derham, domain, 'push_x_v_static_efield')

    @property
    def variables(self):
        return [self._particles]

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(self._particles, dt,
                     self._loc1, self._loc2, self._loc3, self._weight1, self._weight2, self._weight3,
                     self._e_bg.blocks[0]._data, self._e_bg.blocks[1]._data, self._e_bg.blocks[2]._data,
                     array([1e-10, 1e-10]), 100)
