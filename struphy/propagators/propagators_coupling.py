from numpy import array

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

import numpy as np

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.pic.pusher import Pusher

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import LinOpWithTransp

from struphy.psydac_api.utilities import apply_essential_bc_to_array


class StepEfieldWeights( Propagator ):
    r'''Solve the following Crank-Nicolson step

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        = 
        \begin{bmatrix}
            0 & - \mathbb{H} \\
            - \mathbb{A} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n
            \mathbf{W}^{n+1} + \mathbf{W}^n
        \end{bmatrix}

    based on the :ref:`Schur complement <schur_solver>` where

    .. math::
        \mathbb{A} & = - \frac{\Delta t}{2} \hat{\mathbf{F}}_0 \mathbb{V} \overline{DL}^T \left( \mathbb{P}^1 \right)^T \,,
        \mathbb{H} & = \frac{\Delta t}{2} \mathbb{P}^1 \overline{G} \left( \mathbb{V} \right)^T \hat{\mathbf{F}}_0 \,.

    make up the accumulation matrix :math:`\mathbb{H} \mathbb{A}` .

    Parameters
    ---------- 
        e : psydac.linalg.block.BlockVector
            FE coefficients of a 1-form.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.

        params : dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, domain, derham, e, particles, mass_ops, params_bs, params_solver):
        
        assert isinstance(e, BlockVector)

        # Read out relevant parameters for Accumulator object
        self.f0_spec   = params_bs['type']
        self.moms_spec = params_bs['moms_spec']
        self.f0_params = params_bs['moms_params']
        # raise NotImplementedError('Parameters are not correct yet!')

        # Initialize Accumulator object
        self._accum = Accumulator(domain, derham, 'Hcurl', 'linear_vlasov_maxwell',
                                  self.f0_spec, array(self.moms_spec), array(self.f0_params), do_vector=True)

        self._e = e
        self._particles = particles
        self._info = params_solver['info']

        self._domain = domain
        self._derham = derham

        self._accum.accumulate(particles.markers)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if params_solver['pc'] == None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params_solver["pc"])
            self._pc = pc_class(derham, 'V1', mass_ops._fun_M1)

        r'''
        .. math::

            \begin{bmatrix}
                \mathbb{M}_1 & \mathbb{H} \\
                \mathbb{A} & \mathbb{1}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{e}^{n+1} \\
                \mathbf{W}^{n+1}
            \end{bmatrix}
            =
            \begin{bmatrix}
                \mathbb{M}_1 & -\mathbb{H} \\
                -\mathbb{A} & \mathbb{1}
            \end{bmatrix}
            \begin{bmatrix}
                \mathbf{e}^{n} \\
                \mathbf{W}^{n}
            \end{bmatrix}
        '''
        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = mass_ops.M1
        _BC = self._accum.matrix

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

        self._accum.accumulate(self._particles.markers)

        # Update Schur solver
        self._schur_solver.BC = self._accum.matrix

        # allocate temporary FemFields _e during solution
        _e, info = self._schur_solver(en, self._accum.vector, dt)

        # write new coeffs into Propagator.variables
        de = self.in_place_update(_e)
        self._pusher(self._particles, dt,
                     en.blocks[0]._data, en.blocks[1]._data, en.blocks[2]._data,
                     self.f0_spec, array(self.moms_spec), array(self.f0_params))

        # TODO: Implement info for weights as well
        if self._info:
            print('Status     for StepEfieldWeights:', info['success'])
            print('Iterations for StepEfieldWeights:', info['niter'])
            print('Maxdiff e1 for StepEfieldWeights:', max(de))
            print()


class StepFullPressurecouplingHcurl( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_1)^{-1} V^\top (\bar {\mathcal X}^1)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^1 V (\mathbb M^n_1)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_1}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V1', mass_ops._fun_M1n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d_full', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M1n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu_full')

    @property
    def variables(self):
        return [self._u]
    
    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X1T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X1))

        BV = Multiply(-1/2, self._mhd_ops.X1T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt) 

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[2])
            
        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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


class StepFullPressurecouplingHdiv( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_2)^{-1} V^\top (\bar {\mathcal X}^2)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^2 V (\mathbb M^n_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_2}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        particles : struphy.pic.particles.Particles6D
        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V2', mass_ops._fun_M2n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d_full', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M2n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu_full')

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        # assemble G^T dot VEC
        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])


        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X2T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X2))

        BV = Multiply(-1/2, self._mhd_ops.X2T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt)

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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


class StepFullPressurecouplingH1vec( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_v)^{-1} V^\top (\bar {\mathcal X}^0)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^0 V (\mathbb M^n_v)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_v}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete vector field (0-form discretization in each component).

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V0vec', mass_ops._fun_Mvn)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d_full', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.Mvn

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu_full')

    @property
    def variables(self):
        return [self._u]
    
    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        # assemble G^T dot VEC
        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X0T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X0))

        BV = Multiply(-1/2, self._mhd_ops.X0T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt) 

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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


class StepPressurecouplingHcurl( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_1)^{-1} V^\top (\bar {\mathcal X}^1)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^1 V (\mathbb M^n_1)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_1}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 1-form.

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V1', mass_ops._fun_M1n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M1n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu')

    @property
    def variables(self):
        return [self._u]
    
    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X1T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X1))

        BV = Multiply(-1/2, self._mhd_ops.X1T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt) 

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X1.dot(un + _u)[2])
            
        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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


class StepPressurecouplingHdiv( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_2)^{-1} V^\top (\bar {\mathcal X}^2)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^2 V (\mathbb M^n_2)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_2}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete 2-form.

        particles : struphy.pic.particles.Particles6D
        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V2', mass_ops._fun_M2n)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.M2n

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu')

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        # current variables
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        # assemble G^T dot VEC
        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])


        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X2T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X2))

        BV = Multiply(-1/2, self._mhd_ops.X2T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt)

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X2.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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


class StepPressurecouplingH1vec( Propagator ):
    r'''Crank-Nicolson step for pressure coupling term in MHD equations and velocity update with the force term :math:`\nabla \mathbf U \cdot \mathbf v`.

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix} 
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n_v)^{-1} V^\top (\bar {\mathcal X}^0)^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X}^0 V (\mathbb M^n_v)^{-1} & 0 \end{bmatrix} 
        \begin{bmatrix} {\mathbb M^n_v}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} ,

    based on the :ref:`Schur complement <schur_solver>`.

    Parameters
    ---------- 
        u : psydac.linalg.block.BlockVector
            FE coefficients of a discrete vector field (0-form discretization in each component).

        particles : struphy.pic.particles.Particles6D

        domain : struphy.geometry.base.Domain
            Infos regarding mapping.
            
        mass_ops : struphy.psydac_api.mass.WeightedMass
            Weighted mass matrices from struphy.psydac_api.mass.
            
        mhd_ops : struphy.psydac_api.basis_projection_ops.MHDOperators
            Linear MHD operators from struphy.psydac_api.basis_projection_ops.

        params: dict
            Solver parameters for this splitting step.
    '''

    def __init__(self, u, particles, derham, domain, mass_ops, mhd_ops, params):

        assert isinstance(u, BlockVector)

        self._u = u
        self._particles = particles
        self._derham = derham
        self._G = derham.grad
        self._GT = derham.grad.transpose()
        self._domain = domain
        self._params = params
        self._info = params['info']
        self._mass_ops = mass_ops
        self._mhd_ops = mhd_ops
        self._rank = derham.comm.Get_rank()

        # Preconditioner
        if params['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, params['pc'])
            self._pc = pc_class(derham, 'V0vec', mass_ops._fun_Mvn)

        # call the accumulation class
        args = []
        self._ACC = Accumulator(self._domain, self._derham, 'Hcurl', 'pc_lin_mhd_6d', *args, do_vector = True, symmetry = 'pressure')

        # Define A of the Schur block matrix [[A, B], [C, I]] (B and C are needed to be defined every time the propagater is called since they include accumulated values)
        self._A = mass_ops.Mvn

        # call Pusher class
        self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu')

    @property
    def variables(self):
        return [self._u]
    
    def __call__(self, dt):
        un = self.variables[0]

        # reorganize particles
        self._derham.comm.Barrier()
        self._particles.mpi_sort_markers()
        self._derham.comm.Barrier()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles.markers, self._particles.n_mks)

        MAT = [[self._ACC.matrix11, self._ACC.matrix12, self._ACC.matrix13], 
               [self._ACC.matrix12, self._ACC.matrix22, self._ACC.matrix23], 
               [self._ACC.matrix13, self._ACC.matrix23, self._ACC.matrix33]]
        VEC =  [self._ACC.vector1, self._ACC.vector2, self._ACC.vector3]

        # assemble G^T dot VEC
        GT_VEC = BlockVector(self._derham.V0vec.vector_space, blocks=[self._GT.dot(VEC[0]),
                                                                      self._GT.dot(VEC[1]),
                                                                      self._GT.dot(VEC[2])])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, Compose(self._mhd_ops.X0T, self.GT_MAT_G(self._derham, MAT), self._mhd_ops.X0))

        BV = Multiply(-1/2, self._mhd_ops.X0T).dot(GT_VEC)
        
        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._params['type'], 
                                         tol=self._params['tol'], maxiter=self._params['maxiter'],
                                         verbose=self._params['verbose'])

        # allocate temporary FemFields _u during solution
        _u, info = schur_solver(un, BV, dt) 

        # calculate GXu
        GXu_1 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[0])
        GXu_2 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[1])
        GXu_3 = self._G.dot(self._mhd_ops.X0.dot(un + _u)[2])

        # push particles
        # check if ghost regions are synchronized
        for i in range(3):
            if not GXu_1[i].ghost_regions_in_sync: GXu_1[i].update_ghost_regions()
            if not GXu_2[i].ghost_regions_in_sync: GXu_2[i].update_ghost_regions()
            if not GXu_3[i].ghost_regions_in_sync: GXu_3[i].update_ghost_regions()

        self._pusher(self._particles, dt,
                     GXu_1[0]._data, GXu_1[1]._data, GXu_1[2]._data,
                     GXu_2[0]._data, GXu_2[1]._data, GXu_2[2]._data,
                     GXu_3[0]._data, GXu_3[1]._data, GXu_3[2]._data)

        # write new coeffs into Propagator.variables
        du = self.in_place_update(_u)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max(du))
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

            self._domain = derham.V0vec.vector_space
            self._codomain = derham.V0vec.vector_space
            self._MAT = MAT

            v1 = StencilVector(derham.V0vec.vector_space.spaces[0])
            v2 = StencilVector(derham.V0vec.vector_space.spaces[1])
            v3 = StencilVector(derham.V0vec.vector_space.spaces[2])
            list_blocks = [v1, v2, v3]
            self._vector = BlockVector(derham.V0vec.vector_space, blocks=list_blocks)

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.V0vec.vector_space.dtype

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
