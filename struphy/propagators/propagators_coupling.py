import numpy as np
from numpy import array

from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector

from struphy.propagators.base import Propagator
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.pic.particles_to_grid import Accumulator
from struphy.pic.pusher import Pusher
from struphy.polar.basic import PolarVector
from struphy.kinetic_background.analytical import Maxwellian6D

from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_api import preconditioner
from struphy.psydac_api.linear_operators import LinOpWithTransp
from struphy.psydac_api.mass import WeightedMassOperator


class StepEfieldWeights(Propagator):
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

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        particles : struphy.pic.particles.Particles6D
            Particles object.

        mass_ops : struphy.psydac_api.mass.WeightedMassOperators
            Weighted mass matrices from struphy.psydac_api.mass.

        params_backgr : dict
            Parameters for the background distribution function

        params_solver : dict
            Solver parameters for this splitting step.

        alpha : float
            = Omega_c / Omega_p ; Parameter determining the coupling strength between particles and fields
    '''

    def __init__(self, domain, derham, e, particles, mass_ops, params_bckgr, params_solver, alpha):

        assert isinstance(e, BlockVector)

        # Read out relevant parameters for Accumulator object
        self.bckgr_type = params_bckgr['type']
        # constant values
        self.moms_spec = [0, 0, 0, 0, 0, 0, 0]
        self.f0_params = [params_bckgr[self.bckgr_type]['n'],
                          params_bckgr[self.bckgr_type]['ux'],
                          params_bckgr[self.bckgr_type]['uy'],
                          params_bckgr[self.bckgr_type]['uz'],
                          params_bckgr[self.bckgr_type]['vthx'],
                          params_bckgr[self.bckgr_type]['vthy'],
                          params_bckgr[self.bckgr_type]['vthz'],
                          ]
        self.alpha = alpha

        # Initialize Accumulator object
        self._accum = Accumulator(derham, domain, 'Hcurl', 'linear_vlasov_maxwell',
                                  add_vector=True, symmetry='symm')

        # Create pointers to the variables
        self._e = e
        self._particles = particles
        self._domain = domain
        self._derham = derham
        self._info = params_solver['info']

        # Create buffers to store temporarily _e and its sum with old e
        self._e_temp = e.copy()
        self._e_sum = e.copy()

        # store old weights to compute difference
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

        self._domain = domain
        self._derham = derham

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
        _BC = - self._accum.operators[0].matrix / 4.

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(_A, _BC, pc=self._pc, solver_type=params_solver['type'],
                                         tol=params_solver['tol'], maxiter=params_solver['maxiter'],
                                         verbose=params_solver['verbose'])

        # Instantiate particle pusher
        self._pusher = Pusher(derham, domain, 'push_weights_with_efield')

    @property
    def variables(self):
        return [self._e, self._particles]

    def __call__(self, dt):
        """
        TODO
        """

        self._accum.accumulate(self._particles,
                               array(self.moms_spec),
                               array(self.f0_params), self.alpha)

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
                     array(self.moms_spec),
                     array(self.f0_params),
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
        if coupling == 'perp':
            self._ACC = Accumulator(self._derham, self._domain, 'Hcurl',
                                    'pc_lin_mhd_6d', add_vector=True, symmetry='pressure')
            self._pusher = Pusher(self._derham, self._domain, 'push_pc_GXu')

        elif coupling == 'full':
            self._ACC = Accumulator(self._derham, self._domain, 'Hcurl',
                                    'pc_lin_mhd_6d_full', add_vector=True, symmetry='pressure')
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
        self._ACC.accumulate(self._particles)

        MAT = [[self._ACC.operators[0].matrix, self._ACC.operators[1].matrix, self._ACC.operators[2].matrix],
               [self._ACC.operators[1].matrix, self._ACC.operators[3].matrix, self._ACC.operators[4].matrix],
               [self._ACC.operators[2].matrix, self._ACC.operators[4].matrix, self._ACC.operators[5].matrix]]
        VEC = [self._ACC.vectors[0], self._ACC.vectors[1], self._ACC.vectors[2]]

        GT_VEC = BlockVector(self._derham.Vh['v'],
                             blocks=[self._GT.dot(VEC[0]),
                                     self._GT.dot(VEC[1]),
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


class CurrentCoupling6DCurrent(Propagator):
    """
    TODO
    """

    def __init__(self, particles, derham, domain, mass_ops, solver_params, coupling_params, u, u_space, *b_vectors, f0=None):

        assert isinstance(u, (BlockVector, PolarVector))

        for b in b_vectors:
            assert isinstance(b, (BlockVector, PolarVector))

        assert u_space in {'Hcurl', 'Hdiv', 'H1vec'}

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(derham.spaces_dict[u_space])

        # needed variables
        self._particles = particles
        self._u = u
        self._b_vectors = b_vectors

        # load accumulator
        self._accumulator = Accumulator(
            derham, domain, u_space, 'cc_lin_mhd_6d_2', add_vector=True, symmetry='symm')

        nuh = coupling_params['nuh']
        kap = coupling_params['kappa']
        Ab = coupling_params['Ab']
        Ah = coupling_params['Ah']
        Zh = coupling_params['Zh']

        self._coupling_mat = nuh*kap**2*Zh**2/(Ab*Ah)
        self._coupling_vec = nuh*kap*Zh/Ab
        
        self._scale_push = kap*Zh/Ah
        
        # distribution function (control variate, without control variate f0=None)
        self._f0 = f0

        # evaluate and save nh0 (0-form) * uh0 (2-form if H1vec or vector if Hdiv) at quadrature points for control variate
        if f0 is not None:

            # f0 must be a 6d Maxwellian
            assert isinstance(f0, Maxwellian6D)

            quad_pts = [quad_grid.points.flatten()
                        for quad_grid in derham.Vh_fem['0'].quad_grids]

            uh0_cart = [f0.ux, f0.uy, f0.uz]

            if u_space == 'H1vec':
                self._nuh0_at_quad = domain.pull(
                    uh0_cart, *quad_pts, kind='2_form', squeeze_out=False, coordinates='logical')
            else:
                self._nuh0_at_quad = domain.pull(
                    uh0_cart, *quad_pts, kind='vector', squeeze_out=False, coordinates='logical')

            self._nuh0_at_quad[0] *= domain.pull(
                [f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[1] *= domain.pull(
                [f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')
            self._nuh0_at_quad[2] *= domain.pull(
                [f0.n], *quad_pts, kind='0_form', squeeze_out=False, coordinates='logical')

        # load particle pusher
        self._pusher = Pusher(derham, domain, 'push_bxu_' + u_space)

        # FEM spaces and basis extraction operators for u and b
        self._fem_space_u = derham.Vh_fem[derham.spaces_dict[u_space]]
        self._fem_space_b = derham.Vh_fem['2']

        self._Eu = derham.E[derham.spaces_dict[u_space]]
        self._Eb = derham.E['2']

        self._EuT = derham.E[derham.spaces_dict[u_space]].transpose()
        self._EbT = derham.E['2'].transpose()

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(mass_ops, 'M' + derham.spaces_dict[u_space] + 'n')

        # preconditioner
        if solver_params['pc'] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver_params['pc'])
            pc = pc_class(_A)

        _BC = Multiply(-1/4, self._accumulator.operators[0])

        self._schur_solver = SchurSolver(_A, _BC, pc=pc, solver_type=solver_params['type'],
                                         tol=solver_params['tol'], maxiter=solver_params['maxiter'],
                                         verbose=solver_params['verbose'])

        self._info = solver_params['info']
        self._rank = derham.comm.Get_rank()

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
        b_full = self._b_vectors[0].space.zeros()

        for b in self._b_vectors:
            b_full += b

        # extract coefficients to tensor product space
        b_full = self._EbT.dot(b_full)

        # update ghost regions because of non-local access in pusher kernel
        b_full.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        if self._f0 is not None:

            # evaluate total magnetic field at quadrature points
            b_quad = WeightedMassOperator.eval_quad(self._fem_space_b, b_full)

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


class CurrentCoupling5DCurrent1( Propagator ):
    r'''
    TODO
    '''

    def __init__(self, particles, derham, domain, mass_ops, epsilon, u, u_space, b, bc, coupling_solver, coupling_params, *mhd_equil):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)

        self._particles = particles
        self._derham = derham
        self._epsilon = epsilon
        self._u = u
        self._bc = bc
        self._rank = derham.comm.Get_rank()
        self._coupling_solver = coupling_solver
        self._info = coupling_solver['info']

        # define equilibrium fields
        self._b_eq = mhd_equil[0]
        self._norm_b1 = mhd_equil[1]
        self._curl_norm_b = derham.curl.dot(self._norm_b1)

        self._norm_b1.update_ghost_regions()
        self._curl_norm_b.update_ghost_regions()

        # define full magnetic field
        self._b_full = b + self._b_eq

        nuh = coupling_params['nuh']
        kap = coupling_params['kappa']
        Ab = coupling_params['Ab']
        Ah = coupling_params['Ah']
        Zh = coupling_params['Zh']

        if u_space == 'Hcurl':
            id_Mn = 'M1n'
            id_fun = '_fun_M1n'
        elif u_space == 'Hdiv':
            id_Mn = 'M2n'
            id_fun = '_fun_M2n'
        elif u_space == 'H1vec':
            id_Mn = 'Mvn'
            id_fun = '_fun_Mvn'

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(derham.spaces_dict[u_space])

        # TODO
        self._coupling_mat = nuh
        self._coupling_vec = nuh

        # Preconditioner
        _pc_fun = getattr(mass_ops, id_fun)
        if coupling_solver['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, coupling_solver['pc'])
            self._pc = pc_class(getattr(mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(self._derham, domain, u_space, 'cc_lin_mhd_5d_J1', add_vector=True, symmetry='symm')
        self._pusher = Pusher(derham, domain, 'push_gc_cc_J1_' + u_space)

        # Define operators
        self._A = getattr(mass_ops, id_Mn)

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        un = self.variables[0]

        # update ghost regions because of non-local access in pusher kernel
        self._b_full.update_ghost_regions()

        # reorganize particles
        self._particles.mpi_sort_markers()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles,
                             self._epsilon,
                             self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                             self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, self._ACC.operators[0])

        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._coupling_solver['type'],
                                   tol=self._coupling_solver['tol'], maxiter=self._coupling_solver['maxiter'],
                                   verbose=self._coupling_solver['verbose'])

        # solve linear system for updated u coefficients
        u_new, info = schur_solver(un, -self._ACC.vectors[0]/2, dt)

        # calculate average u
        u_avg = (un + u_new)/2
        u_avg.update_ghost_regions()

        self._pusher(self._particles, dt,
                     self._epsilon,
                     self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                     self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     u_avg[0]._data, u_avg[1]._data, u_avg[2]._data)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(u_new)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max_du)
            print()


class CurrentCoupling5DCurrent2( Propagator ):
    r'''
    TODO
    '''

    def __init__(self, particles, derham, domain, mass_ops, basis_ops, epsilon, u, u_space, b, bc, coupling_solver, *mhd_equil):

        assert isinstance(u, BlockVector)
        assert isinstance(b, BlockVector)

        self._particles = particles
        self._derham = derham
        self._epsilon = epsilon
        self._u = u
        self._bc = bc
        self._rank = derham.comm.Get_rank()
        self._coupling_solver = coupling_solver
        self._info = coupling_solver['info']

        # define equilibrium fields
        self._b_eq = mhd_equil[0]
        self._norm_b1 = mhd_equil[1]
        self._norm_b2 = mhd_equil[2]
        self._abs_b = mhd_equil[3]
        self._curl_norm_b = derham.curl.dot(self._norm_b1)

        self._abs_b.update_ghost_regions()
        self._norm_b1.update_ghost_regions()
        self._norm_b2.update_ghost_regions()
        self._curl_norm_b.update_ghost_regions()

        # define full magnetic field
        self._b_full = b + self._b_eq

        # define gradient of absolute value of parallel magnetic field
        PB = getattr(basis_ops, 'PB')
        self._PB = PB.dot(self._b_full)
        self._PB.update_ghost_regions()

        self._grad_PB = derham.grad.dot(self._PB)
        self._grad_PB.update_ghost_regions()

        if u_space == 'Hcurl':
            id_Mn = 'M1n'
            id_fun = '_fun_M1n'
        elif u_space == 'Hdiv':
            id_Mn = 'M2n'
            id_fun = '_fun_M2n'
        elif u_space == 'H1vec':
            id_Mn = 'Mvn'
            id_fun = '_fun_Mvn'

        if u_space == 'H1vec':
            self._space_key_int = 0
        else:
            self._space_key_int = int(derham.spaces_dict[u_space])

        # TODO
        self._coupling_mat = 1.
        self._coupling_vec = 1.

        # Preconditioner
        _pc_fun = getattr(mass_ops, id_fun)
        if coupling_solver['pc'] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, coupling_solver['pc'])
            self._pc = pc_class(getattr(mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(self._derham, domain, u_space, 'cc_lin_mhd_5d_J2', add_vector=True, symmetry='symm')
        self._pusher = Pusher(derham, domain, 'push_gc_cc_J2_' + u_space)

        # Define operators
        self._A = getattr(mass_ops, id_Mn)

    @property
    def variables(self):
        return [self._u]

    def __call__(self, dt):

        un = self.variables[0]

        # update ghost regions because of non-local access in pusher kernel
        self._b_full.update_ghost_regions()

        # reorganize particles
        self._particles.mpi_sort_markers()

        # acuumulate MAT and VEC
        self._ACC.accumulate(self._particles,
                             self._epsilon,
                             self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                             self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                             self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                             self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                             self._grad_PB[0]._data, self._grad_PB[1]._data, self._grad_PB[2]._data,
                             self._space_key_int, self._coupling_mat, self._coupling_vec)

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        BC = Multiply(-1/4, self._ACC.operators[0])

        # call SchurSolver class
        schur_solver = SchurSolver(self._A, BC, pc=self._pc, solver_type=self._coupling_solver['type'],
                                   tol=self._coupling_solver['tol'], maxiter=self._coupling_solver['maxiter'],
                                   verbose=self._coupling_solver['verbose'])

        # solve linear system for updated u coefficients
        u_new, info = schur_solver(un, -self._ACC.vector/2., dt)

        # calculate average u
        u_avg = (un + u_new)/2.
        u_avg.update_ghost_regions()

        self._pusher(self._particles, dt,
                     self._epsilon,
                     self._b_full[0]._data, self._b_full[1]._data, self._b_full[2]._data,
                     self._norm_b1[0]._data, self._norm_b1[1]._data, self._norm_b1[2]._data,
                     self._norm_b2[0]._data, self._norm_b2[1]._data, self._norm_b2[2]._data,
                     self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
                     u_avg[0]._data, u_avg[1]._data, u_avg[2]._data)
                     
        self._particles.save_magnetic_energy(self._derham, self._PB)

        # write new coeffs into Propagator.variables
        max_du, = self.in_place_update(u_new)

        if self._info and self._rank == 0:
            print('Status     for StepPressurecoupling:', info['success'])
            print('Iterations for StepPressurecoupling:', info['niter'])
            print('Maxdiff u1 for StepPressurecoupling:', max_du)
            print()
