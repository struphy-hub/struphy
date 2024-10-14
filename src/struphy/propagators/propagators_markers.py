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
from struphy.pic.pushing import pusher_kernels, pusher_kernels_gc, eval_kernels_gc
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc


class PushEta(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p\,,

    for constant :math:`\mathbf v_p` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p\,. 

    Available algorithms:

    * Explicit from :class:`~struphy.pic.pushing.pusher.ButcherTableau`
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

        # base class constructor call
        super().__init__(particles)

        # parameters that need to be exposed
        self._bc_type = bc_type

        butcher = ButcherTableau(algo)

        # instantiate Pusher
        kernel = pusher_kernels.push_eta_stage

        args_kernel = (butcher.a,
                       butcher.b,
                       butcher.c)

        self._pusher = Pusher(particles,
                              kernel,
                              args_kernel,
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.,
                              n_stages=butcher.n_stages,
                              mpi_sort='last')

    def __call__(self, dt):
        # push markers
        self._pusher(dt)

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
                 b_tilde: BlockVector | PolarVector = None):

        # base class constructor call
        super().__init__(particles)

        # parameters that need to be exposed
        self._scale_fac = scale_fac
        self._b_eq = b_eq
        self._b_tilde = b_tilde
        self._tmp = self.derham.Vh['2'].zeros()
        self._b_full = self.derham.Vh['2'].zeros()

        # define pusher kernel
        if algo == 'analytic':
            kernel = pusher_kernels.push_vxb_analytic
        elif algo == 'implicit':
            kernel = pusher_kernels.push_vxb_implicit
        else:
            raise ValueError(f'{algo = } not supported.')

        # instantiate Pusher
        args_kernel = (self._b_full[0]._data,
                       self._b_full[1]._data,
                       self._b_full[2]._data)

        self._pusher = Pusher(particles,
                              kernel,
                              args_kernel,
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.)

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops['2'].transpose()

    def __call__(self, dt):
        # sum up total magnetic field
        tmp = self._b_eq.copy(out=self._tmp)
        if self._b_tilde is not None:
            tmp += self._b_tilde

        # extract coefficients to tensor product space
        b_full = self._E2T.dot(tmp, out=self._b_full)
        b_full.update_ghost_regions()

        # call pusher kernel
        self._pusher(self._scale_fac*dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()


class PushEtaPC(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf x_p(t)}{\textnormal d t} = \mathbf v_p + \mathbf U (\mathbf x_p(t))\,,

    for constant :math:`\mathbf v_p` and :math:`\mathbf U` in logical space given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p + \textnormal{vec}(\hat{\mathbf U}) \,, 

    where

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,, \qquad \textnormal{vec}( \hat{\mathbf U}) = \hat{\mathbf U}\,.

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
        dct['use_perp_model'] = [True, False]

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(self,
                 particles: Particles,
                 *,
                 u: BlockVector | PolarVector,
                 use_perp_model: bool = options(default=True)[
                     'use_perp_model'],
                 u_space: str):

        super().__init__(particles)

        assert isinstance(u, (BlockVector, PolarVector))

        self._u = u

        # call Pusher class
        if use_perp_model:
            if u_space == 'Hcurl':
                kernel = pusher_kernels.push_pc_eta_rk4_Hcurl
            elif u_space == 'Hdiv':
                kernel = pusher_kernels.push_pc_eta_rk4_Hdiv
            elif u_space == 'H1vec':
                kernel = pusher_kernels.push_pc_eta_rk4_H1vec
            else:
                raise ValueError(
                    f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.')
        else:
            if u_space == 'Hcurl':
                kernel = pusher_kernels.push_pc_eta_rk4_Hcurl_full
            elif u_space == 'Hdiv':
                kernel = pusher_kernels.push_pc_eta_rk4_Hdiv_full
            elif u_space == 'H1vec':
                kernel = pusher_kernels.push_pc_eta_rk4_H1vec_full
            else:
                raise ValueError(
                    f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.')

        self._pusher = Pusher(particles,
                              kernel,
                              (self._u[0]._data,
                               self._u[1]._data,
                               self._u[2]._data),
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.,
                              n_stages=4)

    def __call__(self, dt):

        # check if ghost regions are synchronized
        if not self._u[0].ghost_regions_in_sync:
            self._u[0].update_ghost_regions()
        if not self._u[1].ghost_regions_in_sync:
            self._u[1].update_ghost_regions()
        if not self._u[2].ghost_regions_in_sync:
            self._u[2].update_ghost_regions()

        self._pusher(dt)


class PushGuidingCenterBxEstar(Propagator):
    r"""For each marker :math:`p`, solves

    .. math::

        \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} = \frac{\mathbf E^* \times \mathbf b_0}{B_\parallel^*} (\mathbf X_p(t))   \,,

    where 

    .. math::

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad \mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad  B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` can be the full magnetic field (equilibrium + perturbation).
    The electric potential ``phi`` and/or the magnetic perturbation ``b_tilde`` 
    can be ignored by passing ``None``.
    In logical space this is given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\boldsymbol \eta_p(t)) \,.

    Available algorithms:

    * Explicit from :class:`~struphy.pic.pushing.pusher.ButcherTableau`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order_newton` 
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_2nd_order`  
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['discrete_gradient_2nd_order',
                                  'discrete_gradient_1st_order',
                                  'discrete_gradient_1st_order_newton',
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
                 phi: StencilVector = None,
                 evaluate_e_field: bool = False,
                 b_tilde: BlockVector = None,
                 epsilon: float = 1.,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        # magnetic equilibrium field
        unit_b1 = self.projected_mhd_equil.unit_b1
        self._gradB1 = self.projected_mhd_equil.gradB1
        self._absB0 = self.projected_mhd_equil.absB0
        curl_unit_b_dot_b0 = self.projected_mhd_equil.curl_unit_b_dot_b0

        # magnetic perturbation
        self._b_tilde = b_tilde
        if self._b_tilde is not None:
            self._B_dot_b = self.derham.Vh['0'].zeros()
            self._grad_b_full = self.derham.Vh['1'].zeros()

            self._PB = getattr(self.basis_ops, 'PB')

            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0
        else:
            self._grad_b_full = self._gradB1
            self._B_dot_b = self._absB0

        # allocate electric field
        if phi is None:
            phi = self.derham.Vh['0'].zeros()
            self._evaluate_e_field = False
        self._phi = phi
        self._evaluate_e_field = evaluate_e_field
        self._e_field = self.derham.Vh['1'].zeros()
        self._epsilon = epsilon

        # choose method
        if 'discrete_gradient' in algo['method']:

            # place for storing data during iteration
            first_free_idx = particles.args_markers.first_free_idx

            if '1st_order' in algo['method']:
                # init kernels
                self.add_init_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field))

                self.add_init_kernel(eval_kernels_gc.bstar_parallel_3form,
                                     first_free_idx + 1,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      curl_unit_b_dot_b0._data))

                self.add_init_kernel(eval_kernels_gc.unit_b_1form,
                                     first_free_idx + 2,
                                     (0, 1, 2),
                                     (unit_b1[0]._data,
                                      unit_b1[1]._data, unit_b1[2]._data))

                if 'newton' in algo['method']:
                    # eval kernels
                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 5,
                                         None,
                                         (self._epsilon,
                                          self._B_dot_b._data,
                                          self._phi._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 0., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 6,
                                         None,
                                         (self._epsilon,
                                          self._B_dot_b._data,
                                          self._phi._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 1., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.grad_driftkinetic_hamiltonian,
                                         first_free_idx + 7,
                                         (0,),
                                         (self._epsilon,
                                          self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                          self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                             self._evaluate_e_field),
                                         alpha=(1., 0., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.grad_driftkinetic_hamiltonian,
                                         first_free_idx + 8,
                                         (0, 1),
                                         (self._epsilon,
                                          self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                          self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                             self._evaluate_e_field),
                                         alpha=(1., 1., 0., 0.))

                    # pusher kernel
                    kernel = pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order_newton

                    alpha_in_kernel = 1.  # evaluate at eta^{n+1,k} and save
                    args_kernel = (self._epsilon,
                                   self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                   self._B_dot_b._data,
                                   self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                   self._phi._data,
                                   self._evaluate_e_field)

                else:
                    # eval kernels
                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 5,
                                         None,
                                         (self._epsilon,
                                          self._B_dot_b._data,
                                          self._phi._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 1., 1., 0.))  # evaluate at eta^{n+1,k} and save

                    # pusher kernel
                    kernel = pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_1st_order

                    alpha_in_kernel = .5  # evaluate at mid-point
                    args_kernel = (self._epsilon,
                                   self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                   self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                   self._evaluate_e_field)

            elif '2nd_order' in algo['method']:
                # init kernels (evaluate at eta^n and save)
                self.add_init_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field))

                # eval kernels
                self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx + 1,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field),
                                     alpha=(1., 1., 1., 0.))  # evaluate at eta^{n+1,k} and save)

                # pusher kernel
                kernel = pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_2nd_order

                alpha_in_kernel = .5  # evaluate at mid-point
                args_kernel = (self._epsilon,
                               unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                               self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                               self._B_dot_b._data,
                               curl_unit_b_dot_b0._data,
                               self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                               self._evaluate_e_field)

            else:
                raise NotImplementedError(
                    f'Chosen method {algo["method"]} is not implemented.')

            # Pusher instance
            self._pusher = Pusher(particles,
                                  kernel,
                                  args_kernel,
                                  self.derham.args_derham,
                                  self.domain.args_domain,
                                  alpha_in_kernel=alpha_in_kernel,
                                  init_kernels=self.init_kernels,
                                  eval_kernels=self.eval_kernels,
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'],
                                  mpi_sort=algo['mpi_sort'],
                                  verbose=algo['verbose'])

        else:
            butcher = ButcherTableau(algo['method'])

            kernel = pusher_kernels_gc.push_gc_bxEstar_explicit_multistage

            args_kernel = (self._epsilon,
                           unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                           self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                           self._B_dot_b._data,
                           curl_unit_b_dot_b0._data,
                           self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                           self._evaluate_e_field,
                           butcher.a, butcher.b, butcher.c)

            self._pusher = Pusher(particles,
                                  kernel,
                                  args_kernel,
                                  self.derham.args_derham,
                                  self.domain.args_domain,
                                  alpha_in_kernel=1.,
                                  n_stages=butcher.n_stages,
                                  mpi_sort=algo['mpi_sort'],
                                  verbose=algo['verbose'])

    def __call__(self, dt):

        # electric field
        # TODO: add out to __neg__ of StencilVector
        if self._evaluate_e_field:
            e_field = self.derham.grad.dot(-self._phi, out=self._e_field)
            e_field.update_ghost_regions()

        # magnetic perturbation
        if self._b_tilde is not None:
            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0

        # call pusher
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:

            if self.particles[0].f0.coords == 'constants_of_motion':
                self.particles[0].save_constants_of_motion(
                    epsilon=self._epsilon, abs_B0=self._absB0)

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

        \mathbf E^* = -\nabla \phi - \varepsilon \mu_p \nabla |\mathbf B|\,,\qquad \mathbf B^* = \mathbf B + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad  B^*_\parallel = \mathbf B^* \cdot \mathbf b_0\,,

    where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` can be the full magnetic field (equilibrium + perturbation).
    The electric potential ``phi`` and/or the magnetic perturbation ``b_tilde`` 
    can be ignored by passing ``None``.
    In logical space this is given by :math:`\mathbf X = F(\boldsymbol \eta)`:

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*2}}{\hat B^{*3}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,.
            \end{aligned}
        \right.

    Available algorithms:

    * Explicit from :class:`~struphy.pic.pushing.pusher.ButcherTableau`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order`
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order_newton` 
    * :func:`~struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_2nd_order`  
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct['algo'] = {'method': ['discrete_gradient_2nd_order',
                                  'discrete_gradient_1st_order',
                                  'discrete_gradient_1st_order_newton',
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
                 phi: StencilVector = None,
                 evaluate_e_field: bool = False,
                 b_tilde: BlockVector = None,
                 epsilon: float = 1.,
                 algo: dict = options(default=True)['algo']):

        super().__init__(particles)

        self._epsilon = epsilon

        # magnetic equilibrium field
        self._gradB1 = self.projected_mhd_equil.gradB1
        b2 = self.projected_mhd_equil.b2
        curl_unit_b2 = self.projected_mhd_equil.curl_unit_b2
        self._absB0 = self.projected_mhd_equil.absB0
        curl_unit_b_dot_b0 = self.projected_mhd_equil.curl_unit_b_dot_b0

        # magnetic perturbation
        self._b_tilde = b_tilde
        if self._b_tilde is not None:
            self._B_dot_b = self.derham.Vh['0'].zeros()
            self._grad_b_full = self.derham.Vh['1'].zeros()

            self._PB = getattr(self.basis_ops, 'PB')

            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0
        else:
            self._grad_b_full = self._gradB1
            self._B_dot_b = self._absB0

        # allocate electric field
        if phi is None:
            phi = self.derham.Vh['0'].zeros()
        self._phi = phi
        self._evaluate_e_field = evaluate_e_field
        self._e_field = self.derham.Vh['1'].zeros()
        self._epsilon = epsilon

        # choose method
        if 'discrete_gradient' in algo['method']:

            # place for storing data during iteration
            first_free_idx = particles.args_markers.first_free_idx

            if '1st_order' in algo['method']:

                # init kernels
                self.add_init_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field))

                self.add_init_kernel(eval_kernels_gc.bstar_parallel_3form,
                                     first_free_idx + 1,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      curl_unit_b_dot_b0._data))

                self.add_init_kernel(eval_kernels_gc.bstar_2form,
                                     first_free_idx + 2,
                                     (0, 1, 2),
                                     (self._epsilon,
                                      b2[0]._data, b2[1]._data, b2[2]._data,
                                         curl_unit_b2[0]._data, curl_unit_b2[1]._data, curl_unit_b2[2]._data))

                if 'newton' in algo['method']:
                    # eval kernels
                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 5,
                                         None,
                                         (self._epsilon,
                                          self._B_dot_b._data,
                                          self._phi._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 0., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 6,
                                         None,
                                         (self._epsilon,
                                          self._B_dot_b._data,
                                          self._phi._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 1., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.grad_driftkinetic_hamiltonian,
                                         first_free_idx + 7,
                                         (0,),
                                         (self._epsilon,
                                          self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                          self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                             self._evaluate_e_field),
                                         alpha=(1., 0., 0., 0.))

                    self.add_eval_kernel(eval_kernels_gc.grad_driftkinetic_hamiltonian,
                                         first_free_idx + 8,
                                         (0, 1),
                                         (self._epsilon,
                                          self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                          self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                          self._evaluate_e_field),
                                         alpha=(1., 1., 0., 0.))

                    # pusher kernel
                    kernel = pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order_newton

                    alpha_in_kernel = 1.  # evaluate at eta^{n+1,k} and save
                    args_kernel = (self._epsilon,
                                   self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                   self._B_dot_b._data,
                                   self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                   self._phi._data,
                                   self._evaluate_e_field)
                else:
                    # eval kernels
                    self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                         first_free_idx + 5,
                                         None,
                                         args_eval=(self._epsilon,
                                                    self._B_dot_b._data,
                                                    self._phi._data,
                                                    self._evaluate_e_field),
                                         alpha=1.)  # evaluate at Z^{n+1,k} and save

                    # pusher kernel
                    kernel = pusher_kernels_gc.push_gc_Bstar_discrete_gradient_1st_order

                    alpha_in_kernel = .5  # evaluate at mid-point
                    args_kernel = (self._epsilon,
                                   self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                                   self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                                   self._evaluate_e_field)

            elif '2nd_order' in algo['method']:
                # init kernels (evaluate at eta^n and save)
                self.add_init_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field))

                # eval kernels
                self.add_eval_kernel(eval_kernels_gc.driftkinetic_hamiltonian,
                                     first_free_idx + 1,
                                     None,
                                     (self._epsilon,
                                      self._B_dot_b._data,
                                      self._phi._data,
                                      self._evaluate_e_field),
                                     alpha=1.)  # evaluate at Z^{n+1,k} and save

                # pusher kernel
                kernel = pusher_kernels_gc.push_gc_Bstar_discrete_gradient_2nd_order

                alpha_in_kernel = .5  # evaluate at mid-point
                args_kernel = (self._epsilon,
                               self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                               b2[0]._data, b2[1]._data, b2[2]._data,
                               curl_unit_b2[0]._data, curl_unit_b2[1]._data, curl_unit_b2[2]._data,
                               self._B_dot_b._data,
                               curl_unit_b_dot_b0._data,
                               self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                               self._evaluate_e_field)

            else:
                raise NotImplementedError(
                    f'Chosen method {algo["method"]} is not implemented.')

            # Pusher instance
            self._pusher = Pusher(particles,
                                  kernel,
                                  args_kernel,
                                  self.derham.args_derham,
                                  self.domain.args_domain,
                                  alpha_in_kernel=alpha_in_kernel,
                                  init_kernels=self.init_kernels,
                                  eval_kernels=self.eval_kernels,
                                  maxiter=algo['maxiter'],
                                  tol=algo['tol'],
                                  mpi_sort=algo['mpi_sort'],
                                  verbose=algo['verbose'])

        else:
            butcher = ButcherTableau(algo['method'])

            kernel = pusher_kernels_gc.push_gc_Bstar_explicit_multistage

            args_kernel = (self._epsilon,
                           self._grad_b_full[0]._data, self._grad_b_full[1]._data, self._grad_b_full[2]._data,
                           b2[0]._data, b2[1]._data, b2[2]._data,
                           curl_unit_b2[0]._data, curl_unit_b2[1]._data, curl_unit_b2[2]._data,
                           self._B_dot_b._data,
                           curl_unit_b_dot_b0._data,
                           self._e_field[0]._data, self._e_field[1]._data, self._e_field[2]._data,
                           self._evaluate_e_field,
                           butcher.a, butcher.b, butcher.c)

            self._pusher = Pusher(particles,
                                  kernel,
                                  args_kernel,
                                  self.derham.args_derham,
                                  self.domain.args_domain,
                                  alpha_in_kernel=1.,
                                  n_stages=butcher.n_stages,
                                  mpi_sort=algo['mpi_sort'],
                                  verbose=algo['verbose'])

    def __call__(self, dt):

        # electric field
        # TODO: add out to __neg__ of StencilVector
        if self._evaluate_e_field:
            e_field = self.derham.grad.dot(-self._phi, out=self._e_field)
            e_field.update_ghost_regions()

        # magnetic perturbation
        if self._b_tilde is not None:
            B_dot_b = self._PB.dot(self._b_tilde, out=self._B_dot_b)
            B_dot_b.update_ghost_regions()

            grad_b_full = self.derham.grad.dot(B_dot_b, out=self._grad_b_full)
            grad_b_full.update_ghost_regions()

            grad_b_full += self._gradB1
            B_dot_b += self._absB0

        # call pusher
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:

            if self.particles[0].f0.coords == 'constants_of_motion':
                self.particles[0].save_constants_of_motion(
                    epsilon=self._epsilon, abs_B0=self._absB0)

            self.particles[0].update_weights()


class PushVinEfield(Propagator):
    r'''Push the velocities according to

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \kappa \, \mathbf{E}(\mathbf{x}_p) \,,

    and in logical coordinates given by :math:`\mathbf x = F(\boldsymbol \eta)`:

    .. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \kappa \, DF^{-T} \mathbf{E}(\boldsymbol \eta_p})  \,,

    which is solved analytically.
    '''

    @staticmethod
    def options():
        pass

    def __init__(self,
                 particles: Particles6D,
                 *,
                 e_field: BlockVector | PolarVector,
                 kappa: float = 1.):

        super().__init__(particles)

        self.kappa = kappa

        assert isinstance(e_field, (BlockVector, PolarVector))
        self._e_field = e_field

        self._pusher = Pusher(particles,
                              pusher_kernels.push_v_with_efield,
                              (self._e_field.blocks[0]._data,
                               self._e_field.blocks[1]._data,
                               self._e_field.blocks[2]._data,
                               self.kappa),
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.)

    def __call__(self, dt):
        """
        TODO
        """
        self._pusher(dt)


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

    * Explicit from :class:`~struphy.pic.pushing.pusher.ButcherTableau`
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
        
        self._tmp = self.derham.Vh['1'].zeros()

        # choose algorithm
        self._butcher = ButcherTableau(algo)

        self._u_on_grid = AccumulatorVector(particles,
                                            'H1',
                                            accum_kernels.charge_density_0form,
                                            self.derham,
                                            self.domain.args_domain)

        self._pusher = Pusher(particles,
                              pusher_kernels.push_deterministic_diffusion_stage,
                              (self._u_on_grid.vectors[0]._data,
                               self._tmp[0]._data, self._tmp[1]._data, self._tmp[2]._data,
                               self._diffusion,
                               self._butcher.a, self._butcher.b, self._butcher.c),
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.,
                              n_stages=self._butcher.n_stages)


    def __call__(self, dt):
        """
        TODO
        """

        # accumulate
        self._u_on_grid(self.particles[0].vdim)

        # take gradient
        pi_u = self._u_on_grid.vectors[0]
        grad_pi_u = self.derham.grad.dot(pi_u, out=self._tmp)
        grad_pi_u.update_ghost_regions()

        # push markers
        self._pusher(dt)

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

        self._noise = array(self.particles[0].markers[:, :3])

        # choose algorithm
        self._butcher = ButcherTableau('forward_euler')

        self._pusher = Pusher(particles,
                              pusher_kernels.push_random_diffusion_stage,
                              (self._noise,
                               self._diffusion,
                               self._butcher.a, self._butcher.b, self._butcher.c),
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.,
                              n_stages=self._butcher.n_stages)

        # self._tmp = self.derham.Vh['1'].zeros()
        self._mean = [0, 0, 0]
        self._cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def __call__(self, dt):
        """
        TODO
        """

        self._noise[:] = random.multivariate_normal(
            self._mean, self._cov, len(self.particles[0].markers))

        # push markers
        self._pusher(dt)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()
