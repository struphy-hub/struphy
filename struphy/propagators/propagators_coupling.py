from numpy import array

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.pic.pusher import Pusher

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import LinOpWithTransp

import numpy as np


class StepEfieldWeights(Propagator):
    r'''Solve the following Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        = 
        \begin{bmatrix}
            0 & \frac{\Delta t}{2} \mathbb{K}^T \\
            \frac{\Delta t}{2} \mathbb{K} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        (\mathbb{K})_p & = \sqrt{f_0} \left( DF^{-1} \bv_p \right) \cdot \left( \mathbb{\Lambda}^1 \right)^T \,.

    make up the accumulation matrix :math:`\mathbb{K}^T \mathbb{K}` .

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass.

        params : dict
            Solver parameters for this splitting step.

        alpha : float
            = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields
    '''

    def __init__(self, domain, derham, e, particles, mass_ops, params_bs, params_solver, alpha):

        assert isinstance(e, BlockVector)

        # Read out relevant parameters for Accumulator object
        self.f0_spec = params_bs['type']
        self.moms_spec = params_bs['moms_spec']
        self.f0_params = params_bs['moms_params']
        # raise NotImplementedError('Parameters are not correct yet!')

        # Initialize Accumulator object
        self._accum = Accumulator(domain, derham, 'Hcurl', 'linear_vlasov_maxwell',
                                  self.f0_spec, array(
                                      self.moms_spec), array(self.f0_params),
                                  alpha,
                                  do_vector=True, symmetry='symm')

        self._e = e
        self._particles = particles
        self._info = params_solver['info']

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        self._domain = domain
        self._derham = derham

        self._accum.accumulate(self._particles.markers, self._particles.n_mks)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params_solver['pc'] == None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params_solver["pc"])
            self._pc = pc_class(mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1
        _BC = self._accum.matrix / 4

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC, pc=self._pc, solver_type=params_solver['type'],
                                         tol=params_solver['tol'], maxiter=params_solver['maxiter'],
                                         verbose=params_solver['verbose'])

        self._pusher = Pusher(derham, domain, 'push_weights_with_efield')

    @property
    def variables(self):
        return [self._e, self._particles]

    def __call__(self, dt):

        # current variables
        en = self.variables[0]

        self._accum.accumulate(self._particles.markers, self._particles.n_mks)

        # Update Schur solver
        self._schur_solver.BC = - self._accum.matrix / 4

        # allocate temporary FemFields _e during solution
        _e, info = self._schur_solver(en, - self._accum.vector / 2, dt)

        # Store old weights
        self._old_weights[~self._particles.holes] = self._particles.markers[~self._particles.holes, 6]

        # Update weights
        self._pusher(self._particles, dt,
                     (_e + en).blocks[0]._data, (_e +
                                                 en).blocks[1]._data, (_e + en).blocks[2]._data,
                     self.f0_spec, array(
                         self.moms_spec), array(self.f0_params),
                     int(self._particles.n_mks))

        # write new coeffs into Propagator.variables
        max_de, = self.in_place_update(_e)

        # Print out max differences for weights and efield
        if self._info:
            print('Status          for StepEfieldWeights:', info['success'])
            print('Iterations      for StepEfieldWeights:', info['niter'])
            print('Maxdiff    e1   for StepEfieldWeights:', max_de)
            max_diff = np.max(np.abs(self._old_weights[~self._particles.holes]
                                     - self._particles.markers[~self._particles.holes, 6]))
            print('Maxdiff weights for StepEfieldWeights:', max_diff)
            print()


class StepHybridDensity(Propagator):
    r'''Solve the following Crank-Nicolson step in hybrid model with unknowns f and A.

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        = 
        \begin{bmatrix}
            0 & \frac{\Delta t}{2} \mathbb{K}^T \\
            \frac{\Delta t}{2} \mathbb{K} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        (\mathbb{K})_p & = \sqrt{f_0} \left( DF^{-1} \bv_p \right) \cdot \left( \mathbb{\Lambda}^1 \right)^T \,.

    make up the accumulation matrix :math:`\mathbb{K}^T \mathbb{K}` .

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, domain, derham, particles):

        # Initialize Accumulator object
        self._accum = Accumulator(domain, derham, 'H1', 'hybrid_fA',
                                  do_vector=False, symmetry='None')

        self._particles = particles

        self._domain = domain
        self._derham = derham

        self._accum.accumulate(self._particles.markers, self._particles.n_mks)

    @property
    def variables(self):
        return [self._particles]

    def __call__(self, dt):

        self._accum.accumulate(self._particles.markers, self._particles.n_mks)


class StepPressurecoupling(Propagator):
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

    def __init__(self, u, u_space, coupling, particles, derham, domain, mass_ops, mhd_ops, coupling_solver):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._coupling_solver = coupling_solver
        self._info = coupling_solver['info']
        self._rank = derham.comm.Get_rank()

        if u_space == 'Hcurl':
            id_Mn = 'M1n'
            id_X = 'X1'
            id_fun = '_fun_M1n'
        elif u_space == 'Hdiv':
            id_Mn = 'M2n'
            id_X = 'X2'
            id_fun = '_fun_M2n'
        elif u_space == 'H1vec':
            id_Mn = 'Mvn'
            id_X = 'X0'
            id_fun = '_fun_Mvn'

        # Preconditioner
        _pc_fun = getattr(mass_ops, id_fun)
        if coupling_solver['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, coupling_solver['pc'])
            self._pc = pc_class(getattr(mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        args = []

        if coupling == 'perp':
            self._ACC = Accumulator(self._domain, self._derham, 'Hcurl',
                                    'pc_lin_mhd_6d', *args, do_vector=True, symmetry='pressure')
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu')

        elif coupling == 'full':
            self._ACC = Accumulator(self._domain, self._derham, 'Hcurl',
                                    'pc_lin_mhd_6d_full', *args, do_vector=True, symmetry='pressure')
            self._pusher = Pusher(
                self._derham, self._domain, 'push_pc_GXu_full')

        else:
            raise NotImplementedError(
                'Given coupling scheme is not implemented!')

        # Define operators
        self._A = getattr(mass_ops, id_Mn)
        self._X = getattr(mhd_ops, id_X)
        self._XT = self._X.transpose()

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._particles.mpi_sort_markers()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13],
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23],
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC = [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        GT_VEC = BlockVector(self._derham.Vh['v'], blocks=[self._GT.dot(VEC[0]),
                                                           self._GT.dot(
                                                               VEC[1]),
                                                           self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._XT,
                      self.GT_MAT_G(self._derham, MAT), self._X))

        BV = Multiply(-1/2, self._XT).dot(GT_VEC)

        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._coupling_solver['type'],
                                   tol=self._coupling_solver['tol'], maxiter=self._coupling_solver['maxiter'],
                                   verbose=self._coupling_solver['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt)

        # calculate GXu
        GXu_1 = self._G.dot(self._X.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._X.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._X.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync:
                GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync:
                GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync:
                GXu_3[i].update_ghost_regions()

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
            return self._derham.Vh['v'].dtype

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

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

            assert self._vector.space == self.codomain

            return self._vector
