import numpy as np

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles import Particles6D, Particles5D
from struphy.pic.particles_to_grid import Accumulator
from struphy.pic.pusher import Pusher
import struphy.pic.utilities_kernels as utilities
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.analytical import Maxwellian6D, Maxwellian6DUniform
from struphy.fields_background.mhd_equil.equils import set_defaults

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Inverse
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import LinOpWithTransp
from struphy.psydac_api.mass import WeightedMassOperator
import struphy.linear_algebra.iterative_solvers as it_solvers


class EfieldWeights(Propagator):
    r'''Solve the following Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - \mathbb{E} \\
            \mathbb{W} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        \begin{align}
        (\mathbb{W})_p & = \frac{1}{N \, s_{0, p}} \frac{1}{v_{\text{th}}^2} \sqrt{f_0}
            \left( DF^{-1} \mathbf{v}_p \right) \cdot \left( \mathbb{\Lambda}^1 \right)^T \,,
            \\
        (\mathbb{E})_p & = \alpha^2 \sqrt{f_0} \, \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right) \,.
        \end{align}

    make up the accumulation matrix :math:`\mathbb{E} \mathbb{W}` .

    Parameters
    ---------- 
    e : psydac.linalg.block.BlockVector
        FE coefficients of a 1-form.

    particles : struphy.pic.particles.Particles6D
        Particles object.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    '''

    def __init__(self, e, particles, **params):

        from struphy.kinetic_background.analytical import Maxwellian6DUniform

        # pointers to variables
        assert isinstance(e, (BlockVector, PolarVector))
        self._e = e

        assert isinstance(particles, Particles6D)
        self._particles = particles

        # parameters
        params_default = {'alpha': 1e2,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False}

        params = set_defaults(params, params_default)

        assert isinstance(params['f0'], Maxwellian6DUniform)

        self._alpha = params['alpha']
        self._f0 = params['f0']
        self._f0_params = np.array([self._f0.params['n'],
                                    self._f0.params['ux'],
                                    self._f0.params['uy'],
                                    self._f0.params['uz'],
                                    self._f0.params['vthx'],
                                    self._f0.params['vthy'],
                                    self._f0.params['vthz']])

        self._info = params['info']

        # Initialize Accumulator object
        self._accum = Accumulator(self.derham, self.domain, 'Hcurl', 'linear_vlasov_maxwell',
                                  add_vector=True, symmetry='symm')

        # Create buffers to store temporarily _e and its sum with old e
        self._e_temp = e.copy()
        self._e_sum = e.copy()

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params['pc'] == None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = - self._accum.operators[0].matrix / 4.

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC, pc=self._pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        # Instantiate particle pusher
        self._pusher = Pusher(self.derham, self.domain,
                              'push_weights_with_efield')

    @property
    def variables(self):
        return [self._e]

    def __call__(self, dt):
        """
        TODO
        """
        # evaluate f0 and accumulate
        f0_values = self._f0(self._particles.markers[:, 0],
                             self._particles.markers[:, 1],
                             self._particles.markers[:, 2],
                             self._particles.markers[:, 3],
                             self._particles.markers[:, 4],
                             self._particles.markers[:, 5])

        self._accum.accumulate(self._particles, f0_values,
                               self._f0_params, self._alpha)

        # Update Schur solver
        self._schur_solver.BC = - self._accum.operators[0].matrix / 4

        # allocate temporary BlockVector during solution
        self._e_temp, info = self._schur_solver(
            self._e, self._accum.vectors[0] / 2., dt)

        # Store old weights
        self._old_weights[~self._particles.holes] = self._particles.markers[~self._particles.holes, 6]

        self._e_sum = self._e_temp + self._e

        # Update weights
        self._pusher(self._particles, dt,
                     self._e_temp.blocks[0]._data + self._e.blocks[0]._data,
                     self._e_temp.blocks[1]._data + self._e.blocks[1]._data,
                     self._e_temp.blocks[2]._data + self._e.blocks[2]._data,
                     f0_values,
                     self._f0_params,
                     int(self._particles.n_mks))

        # write new coeffs into self.variables
        max_de, = self.in_place_update(self._e_temp)

        # Print out max differences for weights and efield
        if self._info:
            print('Status          for StepEfieldWeights:', info['success'])
            print('Iterations      for StepEfieldWeights:', info['niter'])
            print('Maxdiff    e1   for StepEfieldWeights:', max_de)
            max_diff = np.max(np.abs(self._old_weights[~self._particles.holes]
                                     - self._particles.markers[~self._particles.holes, 6]))
            print('Maxdiff weights for StepEfieldWeights:', max_diff)
            print()


class PressureCoupling6D(Propagator):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n)^{-1} V^\top (\bar {\mathcal X})^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X} V (\mathbb M^n)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        u_space : dict
                  params['fields']['mhd_u_space']

        coupling : dict
                   params['coupling']['scheme']

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
                 Infos regarding mapping.

        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
                   Weighted mass matrices from struphy.psydac_api.mass.

        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
                  Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        coupling_solver: dict
                         Solver parameters for this splitting step.
    '''

    def __init__(self, particles, u, **params):

        # pointers to variables
        assert isinstance(particles, Particles6D)
        self._particles = particles

        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'use_perp_model': True,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 5.,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        self._G = self.derham.grad
        self._GT = self.derham.grad.transpose()

        self._info = params['info']
        self._type = params['type']
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._verbose = params['verbose']

        self._rank = self.derham.comm.Get_rank()

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        if params['u_space'] == 'Hcurl':
            id_Mn = 'M1n'
            id_X = 'X1'
        elif params['u_space'] == 'Hdiv':
            id_Mn = 'M2n'
            id_X = 'X2'
        elif params['u_space'] == 'H1vec':
            id_Mn = 'Mvn'
            id_X = 'X0'

        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(getattr(self.mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        accum_ker = 'pc_lin_mhd_6d'
        pusher_ker = 'push_pc_GXu'
        if not params['use_perp_model']:
            accum_ker += '_full'
            pusher_ker +='_full'

        self._info = params['info']

        self._coupling_mat = params['nuh'] * params['kappa']**2 * \
            params['Zh']**2 / (params['Ab'] * params['Ah'])
        self._coupling_vec = params['nuh'] * \
            params['kappa'] * params['Zh'] / params['Ab']
        self._scale_push = params['kappa'] * params['Zh'] / params['Ah']

        print(self._coupling_mat, self._scale_push)

        self._ACC = Accumulator(self.derham, self.domain, 'Hcurl',
                                accum_ker, add_vector=True, 
                                symmetry='pressure')
        self._pusher = Pusher(self.derham, self.domain, pusher_ker)

        # Define operators
        self._A = getattr(self.mass_ops, id_Mn)
        self._X = getattr(self.basis_ops, id_X)
        self._XT = self._X.transpose()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles, self._coupling_mat, self._coupling_vec)

        MAT = [[self._ACC.operators[0].matrix, self._ACC.operators[1].matrix, self._ACC.operators[2].matrix],
               [self._ACC.operators[1].matrix, self._ACC.operators[3].matrix, self._ACC.operators[4].matrix],
               [self._ACC.operators[2].matrix, self._ACC.operators[4].matrix, self._ACC.operators[5].matrix]]
        VEC = [self._ACC.vectors[0], self._ACC.vectors[1], self._ACC.vectors[2]]

        GT_VEC = BlockVector(self.derham.Vh['v'],
                             blocks=[self._GT.dot(VEC[0]),
                                     self._GT.dot(VEC[1]),
                                     self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._XT, self.GT_MAT_G(self.derham, MAT), self._X))

        self._BV = Multiply(-1/2, self._XT).dot(GT_VEC)

        # call SchurSolver class
        self._schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_name=self._type,
                                   tol=self._tol, maxiter=self._maxiter,
                                   verbose=self._verbose)

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        un = self.variables[0]

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles, self._coupling_mat, self._coupling_vec)

        MAT = [[self._ACC.operators[0].matrix, self._ACC.operators[1].matrix, self._ACC.operators[2].matrix],
               [self._ACC.operators[1].matrix, self._ACC.operators[3].matrix, self._ACC.operators[4].matrix],
               [self._ACC.operators[2].matrix, self._ACC.operators[4].matrix, self._ACC.operators[5].matrix]]
        VEC = [self._ACC.vectors[0], self._ACC.vectors[1], self._ACC.vectors[2]]

        GT_VEC = BlockVector(self.derham.Vh['v'],
                             blocks=[self._GT.dot(VEC[0]),
                                     self._GT.dot(VEC[1]),
                                     self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._XT, self.GT_MAT_G(self.derham, MAT), self._X))

        BV = Multiply(-1/2, self._XT).dot(GT_VEC)

        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_name=self._type,
                                   tol=self._tol, maxiter=self._maxiter,
                                   verbose=self._verbose)

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt)

        # calculate GXu
        GXu_1 = self._G.dot(self._X.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._X.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._X.dot(un + _u)[2])

        GXu_1.update_ghost_regions()
        GXu_2.update_ghost_regions()
        GXu_3.update_ghost_regions()

        # push particles
        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max_du)
            print()

    class GT_MAT_G(LinOpWithTransp):
        r'''
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}` 
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.

        Parameters
        ----------
            derham : struphy.psydac_api.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        '''

        def __init__(self, derham, MAT, transposed=False):

            self._derham = derham
            self._G = derham.grad
            self._GT = derham.grad.transpose()

            self._domain = derham.Vh['v']
            self._codomain = derham.Vh['v']
            self._MAT = MAT

            v1 = StencilVector(derham.Vh['v'].spaces[0])
            v2 = StencilVector(derham.Vh['v'].spaces[1])
            v3 = StencilVector(derham.Vh['v'].spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.Vh['v'], blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self.derham.Vh['v'].dtype
        
        @property
        def tosparse(self):
            raise NotImplementedError()

        @property
        def toarray(self):
            raise NotImplementedError()

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self.derham, self._MAT, True)

        def dot(self, v, out=None):
            '''dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.vector_space.

            Returns
            -------
                A StencilVector or BlockVector from W.vector_space.'''

            assert v.space == self.domain

            v.update_ghost_regions()
            
            temp = [None, None, None]

            for i in range(3):
                for j in range(3):
                    temp[j] = self._MAT[i][j].dot(self._G.dot(v[j]))
                self._vector[i] = self._GT.dot(temp[0] + temp[1] + temp[2])

            self._vector.update_ghost_regions()

            del temp

            assert self._vector.space == self.codomain

            return self._vector


class CurrentCoupling6DCurrent(Propagator):
    """
    Parameters
    ----------
    particles : struphy.pic.particles.Particles6D
        Holdes the markers to push.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity.

    **params : dict
        Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, particles, u, **params):

        # pointers to variables
        assert isinstance(particles, Particles6D)
        self._particles = particles

        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'u_space': 'Hdiv',
                          'b_eq': None,
                          'b_tilde': None,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 5.,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.}

        params = set_defaults(params, params_default)

        # assert parameters and expose some quantities to self
        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))

        if params['b_tilde'] is not None:
            assert isinstance(params['b_tilde'], (BlockVector, PolarVector))

        self._b_eq = params['b_eq']
        self._b_tilde = params['b_tilde']
        self._f0 = params['f0']

        if self._f0 is not None:
            assert isinstance(self._f0, Maxwellian6D)

            # evaluate and save nh0 (0-form) * uh0 (2-form if H1vec or vector if Hdiv) at quadrature points for control variate
            quad_pts = [quad_grid.points.flatten()
                        for quad_grid in self.derham.Vh_fem['0'].quad_grids]

            uh0_cart = [self._f0.ux, self._f0.uy, self._f0.uz]

            if params['u_space'] == 'H1vec':
                self._nuh0_at_quad = self.domain.pull(
                    uh0_cart, *quad_pts, kind='2_form', squeeze_out=False, coordinates='logical')
            else:
                self._nuh0_at_quad = self.domain.pull(
                    uh0_cart, *quad_pts, kind='vector', squeeze_out=False, coordinates='logical')

            self._nuh0_at_quad[0] *= self.domain.pull(
                [self._f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[1] *= self.domain.pull(
                [self._f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[2] *= self.domain.pull(
                [self._f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')

        self._info = params['info']

        self._coupling_mat = params['nuh'] * params['kappa']**2 * \
            params['Zh']**2 / (params['Ab'] * params['Ah'])
        self._coupling_vec = params['nuh'] * \
            params['kappa'] * params['Zh'] / params['Ab']
        self._scale_push = params['kappa'] * params['Zh'] / params['Ah']

        # load accumulator
        self._accumulator = Accumulator(
            self.derham, self.domain, params['u_space'], 'cc_lin_mhd_6d_2', add_vector=True, symmetry='symm')

        # load particle pusher
        self._pusher = Pusher(self.derham, self.domain,
                              'push_bxu_' + params['u_space'])

        # FEM spaces and basis extraction operators for u and b
        u_id = self.derham.spaces_dict[params['u_space']]
        self._EuT = self.derham.E[u_id].transpose()
        self._EbT = self.derham.E['2'].transpose()

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(_A)

        _BC = Multiply(-1/4, self._accumulator.operators[0])

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        self._rank = self.derham.comm.Get_rank()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        """
        TODO
        """

        # old coefficients
        u_old = self.variables[0]

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b_tilde is not None:
            b_full += self._b_tilde

        # extract coefficients to tensor product space
        b_full = self._EbT.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        if self._f0 is not None:

            # evaluate total magnetic field at quadrature points
            b_quad = WeightedMassOperator.eval_quad(
                self.derham.Vh_fem['2'], b_full)

            control_vec_at_quad = [self._coupling_vec*(b_quad[1]*self._nuh0_at_quad[2] - b_quad[2]*self._nuh0_at_quad[1]),
                                   self._coupling_vec *
                                   (b_quad[2]*self._nuh0_at_quad[0] -
                                    b_quad[0]*self._nuh0_at_quad[2]),
                                   self._coupling_vec*(b_quad[0]*self._nuh0_at_quad[1] - b_quad[1]*self._nuh0_at_quad[0])]

            self._accumulator.accumulate(self._particles,
                                         b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                         self._space_key_int, self._coupling_mat, self._coupling_vec,
                                         control_vec=control_vec_at_quad)
        else:
            self._accumulator.accumulate(self._particles,
                                         b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                         self._space_key_int, self._coupling_mat, self._coupling_vec)

        # solve linear system for updated u coefficients
        u_new, info = self._schur_solver(
            u_old, -self._accumulator.vectors[0]/2, dt)

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        u_avg = self._EuT.dot((u_old + u_new)/2)

        u_avg.update_ghost_regions()

        # push particles
        self._pusher(self._particles, self._scale_push*dt,
                     b_full[0]._data, b_full[1]._data, b_full[2]._data,
                     u_avg[0]._data, u_avg[1]._data, u_avg[2]._data)

        # write new coeffs into Propagator.variables
        max_du = self.in_place_update(u_new)

        # update weights in case of control variate
        if self._f0 is not None:
            self._particles.update_weights(self._f0)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling6DCurrent:', info['success'])
            print('Iterations for CurrentCoupling6DCurrent:', info['niter'])
            print('Maxdiff up for CurrentCoupling6DCurrent:', max_du)
            print()


class CurrentCoupling5DCurrent1(Propagator):
    r'''
    TODO
    '''

    def __init__(self, particles, u, **params):

        from struphy.pic.particles import Particles5D
        assert isinstance(particles, Particles5D)
        self._particles = particles

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'epsilon': 0.01,
                          'u_space': 'Hdiv',
                          'b' : None,
                          'b_eq': None,
                          'unit_b1': None,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 0.05,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.}
        
        params = set_defaults(params, params_default)
        
        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}

        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        self._epsilon = params['epsilon']
        self._f0 = params['f0']

        assert isinstance(params['b'], (BlockVector, PolarVector))
        self._b = params['b']

        assert isinstance(params['b_eq'], (BlockVector, PolarVector))
        self._b_eq = params['b_eq']

        assert isinstance(params['unit_b1'], (BlockVector, PolarVector))
        self._unit_b1 = params['unit_b1']

        self._type = params['type']
        self._pc = params['pc']
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = params['nuh'] * params['kappa']**2 * \
            params['Zh']**2 / (params['Ab'] * params['Ah'])
        self._coupling_vec = params['nuh'] * \
            params['kappa'] * params['Zh'] / params['Ab']
        self._scale_push = params['kappa'] * params['Zh'] / params['Ah']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._curl_norm_b.update_ghost_regions()
        
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        u_id = self.derham.spaces_dict[params['u_space']]

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        self._A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(self._A)

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(self.derham, self.domain, params['u_space'],
                                'cc_lin_mhd_5d_J1', add_vector=True, symmetry='symm')
        self._pusher = Pusher(self.derham, self.domain, 'push_gc_cc_J1_' + params['u_space'])

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        un = self.variables[0]

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles, self._epsilon,
                             b_full[0]._data, b_full[1]._data, b_full[2]._data,
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        _BC = Multiply(-1/4, self._ACC.operators[0])

        # call SchurSolver class
        schur_solver = SchurSolver(self._A, _BC, pc=self._pc, solver_name=self._type,
                                   tol=self._tol, maxiter=self._maxiter,
                                   verbose=self._verbose)

        # solve linear system for updated u coefficients
        u_new, info = schur_solver(un, -self._ACC.vectors[0]/2, dt)

        # calculate average u
        u_avg = (un + u_new)/2
        u_avg.update_ghost_regions()

        self._pusher(self._particles, dt,
                     self._epsilon,
                     b_full[0]._data, b_full[1]._data, b_full[2]._data,
                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     u_avg[0]._data, u_avg[1]._data, u_avg[2]._data)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DCurrent1:', info['success'])
            print('Iterations for CurrentCoupling5DCurrent1:', info['niter'])
            print('Maxdiff u1 for CurrentCoupling5DCurrent1:', max_du)
            print()


class CurrentCoupling5DCurrent2( Propagator ):
    r'''
    TODO
    '''

    def __init__(self, particles, u, **params):

        from struphy.pic.pusher import ButcherTableau
        from struphy.pic.particles import Particles5D
        assert isinstance(particles, Particles5D)
        self._particles = particles

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'b': None,
                          'epsilon': 0.01,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 0.05,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.,
                          'integrator':'explicit',
                          'method':'rk4'}
        
        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        self._epsilon = params['epsilon']
        self._f0 = params['f0']
        assert isinstance(params['b'], (BlockVector, PolarVector))
        self._b = params['b']
        assert isinstance(params['b_eq'], (BlockVector, PolarVector))
        self._b_eq = params['b_eq']
        assert isinstance(params['unit_b1'], (BlockVector, PolarVector))
        self._unit_b1 = params['unit_b1']
        assert isinstance(params['unit_b2'], (BlockVector, PolarVector))
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._curl_norm_b.update_ghost_regions()

        self._type = params['type']
        self._pc = params['pc']
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = params['nuh'] * params['kappa']**2 * params['Zh']**2 / (params['Ab'] * params['Ah'])
        self._coupling_vec = params['nuh'] * params['kappa'] * params['Zh'] / params['Ab']
        self._scale_push = params['kappa'] * params['Zh'] / params['Ah']

        u_id = self.derham.spaces_dict[params['u_space']]

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(b_full)
        self._PBb.update_ghost_regions()

        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._grad_PBb.update_ghost_regions()

        _A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            pc = pc_class(_A)

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(self.derham, self.domain,  params['u_space'], 'cc_lin_mhd_5d_J2', add_vector=True, symmetry='symm')
        # self._pusher = Pusher(self.derham, self.domain, 'push_gc_cc_J2_' +  params['u_space'])

        # choose algorithm
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
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._butcher = ButcherTableau(a, b, c)
        self._pusher = Pusher(self.derham, self.domain,
                              'push_gc_cc_J2_stage_' +  params['u_space'], self._butcher.n_stages)

        _BC = Multiply(-1/4, self._ACC.operators[0])

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_name=params['type'],
                                         tol=params['tol'], maxiter=params['maxiter'],
                                         verbose=params['verbose'])

        self._rank = self.derham.comm.Get_rank()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
    
        un = self.variables[0]

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        self._PBb = self._PB.dot(b_full)
        self._PBb.update_ghost_regions()

        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._grad_PBb.update_ghost_regions()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles,
                             self._epsilon,
                             b_full[0]._data, b_full[1]._data, b_full[2]._data,
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # solve linear system for updated u coefficients
        u_new, info = self._schur_solver(un, -self._ACC.vectors[0]/2., dt)

        # calculate average u
        u_avg = (un + u_new)/2
        u_avg.update_ghost_regions()

        self._pusher(self._particles, dt,
                     self._epsilon,
                     b_full[0]._data, b_full[1]._data, b_full[2]._data,
                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                     self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     u_avg[0]._data, u_avg[1]._data, u_avg[2]._data,
                     self._butcher.a, self._butcher.b, self._butcher.c)
                     
        self._particles.save_magnetic_energy(self.derham, self._PBb)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(u_new)

        if self._info and self._rank == 0:
            print('Status     for CurrentCoupling5DCurrent2:', info['success'])
            print('Iterations for CurrentCoupling5DCurrent2:', info['niter'])
            print('Maxdiff u1 for CurrentCoupling5DCurrent2:', max_du)
            print()


class CurrentCoupling5DCurrent2_dg( Propagator ):
    r'''
    TODO
    '''

    def __init__(self, particles, u, **params):

        from struphy.pic.particles import Particles5D
        assert isinstance(particles, Particles5D)
        self._particles = particles

        # pointers to variables
        assert isinstance(u, (BlockVector, PolarVector))
        self._u = u

        # parameters
        params_default = {'b': None,
                          'epsilon': 0.01,
                          'u_space': 'Hdiv',
                          'b_eq': None,
                          'unit_b1': None,
                          'unit_b2': None,
                          'abs_b': None,
                          'f0': Maxwellian6DUniform(),
                          'type': 'PConjugateGradient',
                          'pc': 'MassMatrixPreconditioner',
                          'tol': 1e-8,
                          'maxiter': 3000,
                          'info': False,
                          'verbose': False,
                          'nuh': 0.05,
                          'Ab': 1,
                          'Ah': 1,
                          'Zh': 1,
                          'kappa': 1.,
                          'integrator':'explicit',
                          'method':'rk4'}
        
        params = set_defaults(params, params_default)

        assert params['u_space'] in {'Hcurl', 'Hdiv', 'H1vec'}
        if params['u_space'] == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.spaces_dict[params['u_space']])

        self._epsilon = params['epsilon']
        self._f0 = params['f0']
        assert isinstance(params['b'], (BlockVector, PolarVector))
        self._b = params['b']
        assert isinstance(params['b_eq'], (BlockVector, PolarVector))
        self._b_eq = params['b_eq']
        assert isinstance(params['unit_b1'], (BlockVector, PolarVector))
        self._unit_b1 = params['unit_b1']
        assert isinstance(params['unit_b2'], (BlockVector, PolarVector))
        self._unit_b2 = params['unit_b2']
        self._abs_b = params['abs_b']

        self._curl_norm_b = self.derham.curl.dot(self._unit_b1)
        self._curl_norm_b.update_ghost_regions()

        self._type = params['type']
        self._pc = params['pc']
        self._tol = params['tol']
        self._maxiter = params['maxiter']
        self._info = params['info']
        self._verbose = params['verbose']
        self._rank = self.derham.comm.Get_rank()

        self._coupling_vec = params['nuh'] * \
            params['kappa'] * params['Zh'] / params['Ab']
        self._scale_push = params['kappa'] * params['Zh'] / params['Ah']

        u_id = self.derham.spaces_dict[params['u_space']]

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        self._PB = getattr(self.basis_ops, 'PB')
        self._PBb = self._PB.dot(b_full)
        self._PBb.update_ghost_regions()
        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._grad_PBb.update_ghost_regions()

        self._A = getattr(self.mass_ops, 'M' + u_id + 'n')

        # preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(self._A)

        # Call the accumulation and Pusher class
        # self._ACC_prepare = Accumulator(self.derham, self.domain, params['u_space'], 'cc_lin_mhd_5d_J2_dg_prepare', add_vector=True, symmetry='symm')
        # self._ACC = Accumulator(self.derham, self.domain, params['u_space'], 'cc_lin_mhd_5d_J2_dg', add_vector=True, symmetry='symm')
        # self._pusher = Pusher(self.derham, self.domain, 'push_gc_cc_J2_dg_' + params['u_space'])

        self._ACC_prepare = Accumulator(self.derham, self.domain, params['u_space'], 'cc_lin_mhd_5d_J2_dg_prepare_faster', add_vector=True, symmetry='symm')
        self._ACC = Accumulator(self.derham, self.domain,  params['u_space'], 'cc_lin_mhd_5d_J2_dg_faster', add_vector=True, symmetry='symm')
        self._pusher = Pusher(self.derham, self.domain, 'push_gc_cc_J2_dg_faster_' +  params['u_space'])

        # linear solver 
        self._solver = getattr(it_solvers, params['type'])(self._A.domain)

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs1 = u.space.zeros()
        self._rhs2 = u.space.zeros()
        self._u_old = u.space.zeros()
        self._u_temp = u.space.zeros()
        self._u_pusher = u.space.zeros()
        self._u_diff = u.space.zeros()
        self._u_new = u.space.zeros()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        
        # save old u
        self.variables[0].copy(out=self._u_old)

        # sum up total magnetic field
        b_full = self._b_eq.copy()
        if self._b is not None:
            b_full += self._b

        b_full.update_ghost_regions()

        self._PBb = self._PB.dot(b_full)
        self._PBb.update_ghost_regions()

        self._grad_PBb = self.derham.grad.dot(self._PBb)
        self._grad_PBb.update_ghost_regions()

        # reorganize particles
        self._particles.mpi_sort_markers()

        #####################################
        #discrete gradient solver(mid point)#
        #####################################
        # eval initial particle energy
        self._particles.save_magnetic_energy(self.derham, self._PBb)
        en_fB_old = self._particles.markers[~self._particles.holes, 6].dot(
                    self._particles.markers[~self._particles.holes, 5])/self._particles.n_mks*self._coupling_vec 

        
        # ------------ initial guess of u ------------#
        # accumulate S*gradI
        self._ACC_prepare.accumulate(self._particles, self._epsilon,
                                     b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                     self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                     self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                     self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                     self._space_key_int, self._coupling_vec)

        # print('maximum accum result', np.max(np.unique(self._ACC_prepare.vectors[0].toarray_local())))

        # solve linear system A*u^0_n+1 = A*u_n + ACC_prepare.vector
        self._A.dot(self._u_old, out=self._rhs1)
        self._rhs1 += dt*self._ACC_prepare.vectors[0]

        self._rhs1.update_ghost_regions()

        info = self._solver.solve(self._A, self._rhs1, self._pc, 
                                  x0=self._u_old, tol=self._tol,
                                  maxiter=self._maxiter, verbose=self._verbose,
                                  out=self._u_new)[1]

        self._u_new.update_ghost_regions()

        # ------------ initial guess of H ------------#
        # save old etas in columns 9-11
        self._particles.markers[~self._particles.holes, 9:12] = self._particles.markers[~self._particles.holes, 0:3]

        # initial guess of eta is stored in columns 0:3
        self._pusher._pusher(self._particles.markers, dt, 0, *self._pusher._args_fem, *self.domain.args_map,
                             self._epsilon,
                             b_full[0]._data, b_full[1]._data, b_full[2]._data,
                             self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                             self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._u_old[0]._data, self._u_old[1]._data, self._u_old[2]._data)

        self._particles.mpi_sort_markers()

        # print('H initial guess', np.max(self._particles.markers[~self._particles.holes,0:3]))

        # ------------ fixed point iteration ------------#
        for stage in range(10):

            self._u_new.copy(out=self._u_temp)

            # save eta diff at markers[ip, 15:18]
            utilities.check_eta_diff(self._particles.markers)
            # self._particles.markers[~self._particles.holes, 15:18] = self._particles.markers[~self._particles.holes, 0:3] - self._particles.markers[~self._particles.holes, 9:12]
            
            sum_H_diff_loc = np.sum(self._particles.markers[~self._particles.holes, 15:18]**2)
            u_norm_loc = np.sum((self._u_new.toarray_local() - self._u_old.toarray_local())**2)
            denominator = sum_H_diff_loc + u_norm_loc

            # eval particle magnetic energy
            self._particles.save_magnetic_energy(self.derham, self._PBb)
            en_fB_loc = self._particles.markers[~self._particles.holes, 6].dot(self._particles.markers[~self._particles.holes, 5])/self._particles.n_mks*self._coupling_vec 

            # move particle to the mid point position and then the real position is saved at markers[ip, 12:15]
            utilities.check_eta_mid(self._particles.markers)
            self._particles.markers[~self._particles.holes, 0:3], self._particles.markers[~self._particles.holes, 12:15] = self._particles.markers[~self._particles.holes, 12:15].copy(), self._particles.markers[~self._particles.holes, 0:3].copy()
            self._particles.mpi_sort_markers()

            # Accumulate
            accum_gradI_const_loc = utilities.accum_gradI_const(self._particles.markers, self._particles.n_mks, *self._pusher._args_fem,
                                                                self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                                                self._coupling_vec)[0]

            # gradI_const = (en_u - en_u_old - u_diff.dot(self._A.dot(u_mid)) + np.sum(en_fB) - np.sum(en_fB_old) - np.sum(accum_gradI_const))/denominator
            gradI_const = (en_fB_loc - en_fB_old - accum_gradI_const_loc)/denominator

            # Accumulate
            self._ACC.accumulate(self._particles, self._epsilon,
                                 b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                 self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                 self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                 self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                 self._grad_PBb[0]._data, self._grad_PBb[1]._data, self._grad_PBb[2]._data,
                                 gradI_const,
                                 self._space_key_int, self._coupling_vec)

            # update u
            # solve linear system A*u^k_n+1 = A*u_n + ACC.vector
            self._A.dot(self._u_old, out=self._rhs1)
            self._rhs1 += dt*self._ACC.vectors[0]

            info = self._solver.solve(self._A, self._rhs1, self._pc, 
                                      x0=self._u_temp, tol=self._tol,
                                      maxiter=self._maxiter, verbose=self._verbose,
                                      out=self._u_new)[1]

            self._u_new.update_ghost_regions()

            # send particle back to the real position
            self._particles.markers[~self._particles.holes, 0:3], self._particles.markers[~self._particles.holes, 12:15] = self._particles.markers[~self._particles.holes, 12:15].copy(), self._particles.markers[~self._particles.holes, 0:3].copy()
            self._particles.mpi_sort_markers()

            self._particles.markers[~self._particles.holes, 12:15] = self._particles.markers[~self._particles.holes, 0:3]

            # update H (1 step ealiler u is needed, u_temp)
            # calculate average u
            self._u_old.copy(out=self._u_pusher)
            self._u_pusher += self._u_temp
            self._u_pusher /= 2

            self._u_temp.copy(out=self._u_diff)
            self._u_diff -= self._u_old
            self._u_diff *= gradI_const

            self._u_diff.update_ghost_regions()

            self._u_pusher += Inverse(self._A, pc=self._pc, tol=1e-18).dot(self._u_diff)

            self._u_pusher.update_ghost_regions()

            self._pusher._pusher(self._particles.markers, dt, 0, *self._pusher._args_fem, *self.domain.args_map,
                                 self._epsilon,
                                 b_full[0]._data, b_full[1]._data, b_full[2]._data,
                                 self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
                                 self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
                                 self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                                 self._u_pusher[0]._data, self._u_pusher[1]._data, self._u_pusher[2]._data)

            self._particles.mpi_sort_markers()

            print('stage', stage+1)

            u_norm_loc = np.sum((self._u_new.toarray_local() - self._u_temp.toarray_local())**2)
            print('u differences',  np.sqrt(u_norm_loc))
          
            sum_H_diff_loc = np.sum((self._particles.markers[~self._particles.holes, 0:3] - self._particles.markers[~self._particles.holes, 12:15])**2)
            print('H differences', np.sqrt(sum_H_diff_loc))

            diff = np.sqrt(u_norm_loc + sum_H_diff_loc)
            print('diff', diff)
            if diff < 1e-15:
                print('converged!')
                break

        self._particles.save_magnetic_energy(self.derham, self._PBb)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(self._u_new)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max_du)
            print()
