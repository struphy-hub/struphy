import numpy as np


class MHD_ops:

    def __init__(self, DERHAM, nq_pr, EQ_MHD_L, F, assemble_all=False, mpi_comm=None):
        '''Assembles required MHD projection operators.

        See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x`.

        Parameters
        ----------
        DERHAM : Psydac object
            The Derham sequence object, obained from Psydac's discretize.

        nq_pr : list
            Number of quadrature points used in histopolation in each direction.

        EQ_MHD_L : Struphy object
            MHD equilibirum on the logical domain from struphy.mhd_equil.mhd_equil_logical.Equilibrium_mhd_logical

        F : Psydac mapping
            Obtained via .get_callable_mapping()

        assemble_all : bool
            Assemble all `MHD_operator`s in constructor. Only for testing. Please assemble individually by calling `assemble_XX()` for each operator.

        mpi_comm : MPI communicator

        Notes
        -----
        The `X1`, `X2` operators are handled differently, because it outputs 3 scalar spaces instead of a pure scalar or vector space.
        In order not to modify the `MHD_operator` class, we give a set of three functions, each accessing each row of the input matrix-valued function.
        '''

        self._mpi_comm = mpi_comm

        # Missing in Psydac: inverse metric tensor
        def _Ginv(x1, x2, x3): return np.matmul(
            F.jacobian_inv(x1, x2, x3), F.jacobian_inv(x1, x2, x3).T)

        # Psydac spline spaces
        _V0 = DERHAM.V0
        _V1 = DERHAM.V1
        _V2 = DERHAM.V2
        _V3 = DERHAM.V3
        self._V0 = _V0
        self._V1 = _V1
        self._V2 = _V2
        self._V3 = _V3

        # Psydac projectors
        _P0, _P1, _P2, _P3 = DERHAM.projectors(nquads=nq_pr)
        self._Pi0, self._Pi1, self._Pi2, self._Pi3 = _P0, _P1, _P2, _P3

        # print('\nV0.spaces[0]._interpolator:', self._V0.spaces[0]._interpolator)
        # print('\nV0.spaces[1]._interpolator:', self._V0.spaces[1]._interpolator)
        # print('\nV0.spaces[2]._interpolator:', self._V0.spaces[2]._interpolator)

        # print('\nV3.spaces[0]._histopolator:', self._V3.spaces[0]._histopolator)
        # print('\nV3.spaces[1]._histopolator:', self._V3.spaces[1]._histopolator)
        # print('\nV3.spaces[2]._histopolator:', self._V3.spaces[2]._histopolator)

        # Cross product matrices:
        _cross_mask = [
            [1, -1,  1],
            [1,  1, -1],
            [-1,  1,  1],
        ]
        _j2_cross = [
            [lambda x1, x2, x3: 0,   EQ_MHD_L.j2_eq_3,   EQ_MHD_L.j2_eq_2],
            [EQ_MHD_L.j2_eq_3, lambda x1, x2, x3: 0,   EQ_MHD_L.j2_eq_1],
            [EQ_MHD_L.j2_eq_2,   EQ_MHD_L.j2_eq_1, lambda x1, x2, x3: 0],
        ]
        _b2_cross = [
            [lambda x1, x2, x3: 0,   EQ_MHD_L.b2_eq_3,   EQ_MHD_L.b2_eq_2],
            [EQ_MHD_L.b2_eq_3, lambda x1, x2, x3: 0,   EQ_MHD_L.b2_eq_1],
            [EQ_MHD_L.b2_eq_2,   EQ_MHD_L.b2_eq_1, lambda x1, x2, x3: 0],
        ]

        def _eval_cross(x1, x2, x3, fun_list): return np.array(
            [[_cross_mask[m][n] * ele(x1, x2, x3) for n, ele in enumerate(row)] for m, row in enumerate(fun_list)])

        # Scalar functions
        def _fun_K1(x1, x2, x3): return EQ_MHD_L.p3_eq(
            x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))
        _fun_K10 = EQ_MHD_L.p0_eq

        def _fun_K2(x1, x2, x3): return EQ_MHD_L.p3_eq(
            x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))

        def _fun_Y20(x1, x2, x3): return np.sqrt(F.metric_det(x1, x2, x3))

        # 'Matrix' functions
        _fun_Q1 = []
        _fun_W1 = []
        _fun_U1 = []
        _fun_P1 = []
        _fun_S1 = []
        _fun_S10 = []
        _fun_T1 = []
        _fun_X1 = []

        _fun_Q2 = []
        _fun_T2 = []
        _fun_P2 = []
        _fun_S2 = []
        _fun_X2 = []
        _fun_Z20 = []
        _fun_S20 = []

        for m in range(3):
            _fun_Q1 += [[]]
            _fun_W1 += [[]]
            _fun_U1 += [[]]
            _fun_P1 += [[]]
            _fun_S1 += [[]]
            _fun_S10 += [[]]
            _fun_T1 += [[]]
            _fun_X1 += [[]]

            _fun_Q2 += [[]]
            _fun_T2 += [[]]
            _fun_P2 += [[]]
            _fun_S2 += [[]]
            _fun_X2 += [[]]
            _fun_Z20 += [[]]
            _fun_S20 += [[]]
            for n in range(3):
                # See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF_for_tests.projectors_dot_x`.
                _fun_Q1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: EQ_MHD_L.r3_eq(x1, x2, x3) * _Ginv(x1, x2, x3)[m, n]]
                _fun_W1[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD_L.r3_eq(
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_U1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: np.sqrt(F.metric_det(x1, x2, x3)) * _Ginv(x1, x2, x3)[m, n]]
                _fun_P1[-1] += [lambda x1, x2, x3, m=m, n=n: _cross_mask[m][n] *
                                _j2_cross[m][n](x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_S1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: EQ_MHD_L.p3_eq(x1, x2, x3) * _Ginv(x1, x2, x3)[m, n]]
                _fun_S10[-1] += [lambda x1, x2, x3, m=m,
                                 n=n: EQ_MHD_L.p0_eq(x1, x2, x3) if m == n else 0.]
                _fun_T1[-1] += [lambda x1, x2, x3, m=m, n=n: (_eval_cross(
                    x1, x2, x3, _b2_cross) @ _Ginv(x1, x2, x3))[m, n]]  # Matrix product!
                _fun_X1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: (F.jacobian_inv(x1, x2, x3).T)[m, n]]

                _fun_Q2[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD_L.r3_eq(
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_T2[-1] += [lambda x1, x2, x3, m=m, n=n: _cross_mask[m][n] *
                                _b2_cross[m][n](x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_P2[-1] += [lambda x1, x2, x3, m=m, n=n: (_Ginv(x1, x2, x3) @ _eval_cross(
                    x1, x2, x3, _j2_cross))[m, n]]  # Matrix product!
                _fun_S2[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD_L.p3_eq(
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_X2[-1] += [lambda x1, x2, x3, m=m,
                                n=n: F.jacobian(x1, x2, x3)[m, n] / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_Z20[-1] += [lambda x1, x2, x3, m=m,
                                 n=n: F.metric(x1, x2, x3)[m, n] / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_S20[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD_L.p0_eq(
                    x1, x2, x3) * F.metric(x1, x2, x3)[m, n] / np.sqrt(F.metric_det(x1, x2, x3))]

        # Scalar functions
        self._fun_K1 = _fun_K1
        self._fun_K10 = _fun_K10

        self._fun_K2 = _fun_K2
        self._fun_Y20 = _fun_Y20

        # 'Matrix' functions
        self._fun_Q1 = _fun_Q1
        self._fun_W1 = _fun_W1
        self._fun_U1 = _fun_U1
        self._fun_P1 = _fun_P1
        self._fun_S1 = _fun_S1
        self._fun_S10 = _fun_S10
        self._fun_T1 = _fun_T1
        self._fun_X1 = _fun_X1

        self._fun_Q2 = _fun_Q2
        self._fun_T2 = _fun_T2
        self._fun_P2 = _fun_P2
        self._fun_S2 = _fun_S2
        self._fun_X2 = _fun_X2
        self._fun_Z20 = _fun_Z20
        self._fun_S20 = _fun_S20

        # Assemble operators only when needed. Otherwise it takes a full minute to initialize the following classes.
        if assemble_all:

            # MHD operators with velocity (up) as 1-form:
            self._Q1 = MHD_operator(_V1, _V2, _P2, _fun_Q1, self._mpi_comm)
            self._W1 = MHD_operator(_V1, _V1, _P1, _fun_W1, self._mpi_comm)
            self._U1 = MHD_operator(_V1, _V2, _P2, _fun_U1, self._mpi_comm)
            self._P1 = MHD_operator(_V2, _V1, _P1, _fun_P1, self._mpi_comm)
            self._S1 = MHD_operator(_V1, _V2, _P2, _fun_S1, self._mpi_comm)
            self._S10 = MHD_operator(_V1, _V1, _P1, _fun_S10, self._mpi_comm)
            self._K1 = MHD_operator(_V3, _V3, _P3, [[_fun_K1]], self._mpi_comm)
            self._K10 = MHD_operator(_V0, _V0, _P0, [[_fun_K10]], self._mpi_comm)
            self._T1 = MHD_operator(_V1, _V1, _P1, _fun_T1, self._mpi_comm)
            self._X11 = MHD_operator(_V1, _V0, _P0, [_fun_X1[0]], self._mpi_comm)  # Row 1
            self._X12 = MHD_operator(_V1, _V0, _P0, [_fun_X1[1]], self._mpi_comm)  # Row 2
            self._X13 = MHD_operator(_V1, _V0, _P0, [_fun_X1[2]], self._mpi_comm)  # Row 3
            self._X1 = [self._X11, self._X12, self._X13]

            # MHD operators with velocity (up) as 2-form:
            self._Q2 = MHD_operator(_V2, _V2, _P2, _fun_Q2, self._mpi_comm)
            self._T2 = MHD_operator(_V2, _V1, _P1, _fun_T2, self._mpi_comm)
            self._P2 = MHD_operator(_V2, _V2, _P2, _fun_P2, self._mpi_comm)
            self._S2 = MHD_operator(_V2, _V2, _P2, _fun_S2, self._mpi_comm)
            self._K2 = MHD_operator(_V3, _V3, _P3, [[_fun_K2]], self._mpi_comm)
            self._X21 = MHD_operator(_V2, _V0, _P0, [_fun_X2[0]], self._mpi_comm)  # Row 1
            self._X22 = MHD_operator(_V2, _V0, _P0, [_fun_X2[1]], self._mpi_comm)  # Row 2
            self._X23 = MHD_operator(_V2, _V0, _P0, [_fun_X2[2]], self._mpi_comm)  # Row 3
            self._Z20 = MHD_operator(_V2, _V1, _P1, _fun_Z20, self._mpi_comm)
            self._Y20 = MHD_operator(_V0, _V3, _P3, [[_fun_Y20]], self._mpi_comm)
            self._S20 = MHD_operator(_V2, _V1, _P1, _fun_S20, self._mpi_comm)
            self._X2 = [self._X21, self._X22, self._X23]

    # Assemble operators only when needed. Otherwise it takes a full minute to initialize the following classes.

    def assemble_Q1(self):
        self._Q1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_Q1, self._mpi_comm)

    def assemble_W1(self):
        self._W1 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_W1, self._mpi_comm)

    def assemble_U1(self):
        self._U1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_U1, self._mpi_comm)

    def assemble_P1(self):
        self._P1 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_P1, self._mpi_comm)

    def assemble_S1(self):
        self._S1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_S1, self._mpi_comm)

    def assemble_S10(self):
        self._S10 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_S10, self._mpi_comm)

    def assemble_K1(self):
        self._K1 = MHD_operator(
            self._V3, self._V3, self._Pi3, [[self._fun_K1]], self._mpi_comm)

    def assemble_K10(self):
        self._K10 = MHD_operator(
            self._V0, self._V0, self._Pi0, [[self._fun_K10]], self._mpi_comm)

    def assemble_T1(self):
        self._T1 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_T1, self._mpi_comm)

    def assemble_X1(self):
        self._X11 = MHD_operator(self._V1, self._V0, self._Pi0, [
                                 self._fun_X1[0]], self._mpi_comm)  # Row 1
        self._X12 = MHD_operator(self._V1, self._V0, self._Pi0, [
                                 self._fun_X1[1]], self._mpi_comm)  # Row 2
        self._X13 = MHD_operator(self._V1, self._V0, self._Pi0, [
                                 self._fun_X1[2]], self._mpi_comm)  # Row 3
        self._X1 = [self._X11, self._X12, self._X13]

    def assemble_Q2(self):
        self._Q2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_Q2, self._mpi_comm)

    def assemble_T2(self):
        self._T2 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_T2, self._mpi_comm)

    def assemble_P2(self):
        self._P2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_P2, self._mpi_comm)

    def assemble_S2(self):
        self._S2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_S2, self._mpi_comm)

    def assemble_K2(self):
        self._K2 = MHD_operator(
            self._V3, self._V3, self._Pi3, [[self._fun_K2]], self._mpi_comm)

    def assemble_X2(self):
        self._X21 = MHD_operator(self._V2, self._V0, self._Pi0, [
                                 self._fun_X2[0]], self._mpi_comm)  # Row 1
        self._X22 = MHD_operator(self._V2, self._V0, self._Pi0, [
                                 self._fun_X2[1]], self._mpi_comm)  # Row 2
        self._X23 = MHD_operator(self._V2, self._V0, self._Pi0, [
                                 self._fun_X2[2]], self._mpi_comm)  # Row 3
        self._X2 = [self._X21, self._X22, self._X23]

    def assemble_Z20(self):
        self._Z20 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_Z20, self._mpi_comm)

    def assemble_Y20(self):
        self._Y20 = MHD_operator(
            self._V0, self._V3, self._Pi3, [[self._fun_Y20]], self._mpi_comm)

    def assemble_S20(self):
        self._S20 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_S20, self._mpi_comm)

    # The actual MHD operators.

    def Q1(self, x):
        return self._Q1.dot(x)

    def W1(self, x):
        return self._W1.dot(x)

    def U1(self, x):
        return self._U1.dot(x)

    def P1(self, x):
        return self._P1.dot(x)

    def S1(self, x):
        return self._S1.dot(x)

    def S10(self, x):
        return self._S10.dot(x)

    def K1(self, x):
        return self._K1.dot(x)

    def K10(self, x):
        return self._K10.dot(x)

    def T1(self, x):
        return self._T1.dot(x)

    def X1(self, x):
        return [self._X11.dot(x), self._X12.dot(x), self._X13.dot(x)]

    def Q2(self, x):
        return self._Q2.dot(x)

    def T2(self, x):
        return self._T2.dot(x)

    def P2(self, x):
        return self._P2.dot(x)

    def S2(self, x):
        return self._S2.dot(x)

    def K2(self, x):
        return self._K2.dot(x)

    def X2(self, x):
        return [self._X21.dot(x), self._X22.dot(x), self._X23.dot(x)]

    def Z20(self, x):
        return self._Z20.dot(x)

    def Y20(self, x):
        return self._Y20.dot(x)

    def S20(self, x):
        return self._S20.dot(x)


class MHD_operator:

    def __init__(self, V, W, pi_W, fun, mpi_comm):
        '''
        Class for MHD specific projection operators PI_ijk(fun Lambda_mno).

        Parameters
        ----------
            V : TensorFemSpace or ProductFemSpace
                Domain of the operator, henceforth called "input space".

            W : TensorFemSpace or ProductFemSpace
                Codomain of the operator, henceforth called "output space".

            pi_W : GlobalProjector
                Psydac de Rham projector into space W.

            fun : list
                List of functions of (eta1, eta2, eta3) that multiply the basis functions of the input space V.
                3x3 matrix-valued (nested list [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]]) if V is ProductFemSPace, 
                scalar-valued [f] list otherwise.

            mpi_comm : MPI communicator
        '''

        from psydac.fem.tensor import TensorFemSpace
        from psydac.fem.vector import ProductFemSpace
        from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
        from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

        from struphy.psydac_linear_operators.mhd_ops_kernels_pure_psydac import assemble_dofs_for_weighted_basisfuns as assemble
        from struphy.psydac_linear_operators.prepare_projection import evaluate_fun_weights

        assert isinstance(V, TensorFemSpace) or isinstance(V, ProductFemSpace)
        assert isinstance(W, TensorFemSpace) or isinstance(W, ProductFemSpace)

        self._V = V
        self._W = W
        self._pi_W = pi_W
        self._fun = fun
        self._mpi_comm = mpi_comm

        # Retrieve solver
        self._solver = pi_W.solver

        # Input space: Stencil vector spaces and 1d spaces
        if V.symbolic_space.name in {'H1', 'L2'}:
            _Vspaces = [V.vector_space]
            _V1ds = [V.spaces]
        else:
            _Vspaces = V.vector_space
            _V1ds = [comp.spaces for comp in V.spaces]

        # Output space: Stencil vector spaces and 1d spaces
        if W.symbolic_space.name in {'H1', 'L2'}:
            _Wspaces = [W.vector_space]
            _W1ds = [W.spaces]

        else:
            _Wspaces = W.vector_space
            _W1ds = [comp.spaces for comp in W.spaces]

        print(f'From {V.symbolic_space.name} to {W.symbolic_space.name}')

        # Block matrix for dofs
        _blocks = []
        # Ouptut vector space (codomain), row of block
        for Wspace, W1d, fun_line in zip(_Wspaces, _W1ds, fun):
            _blocks += [[]]
            # Input vector space (domain), column of block
            for Vspace, V1d, f in zip(_Vspaces, _V1ds, fun_line):

                # Initiate cell of block matrix
                _dofs = StencilMatrix(Vspace, Wspace)

                _starts_in = _dofs.domain.starts
                _ends_in = _dofs.domain.ends
                _pads_in = _dofs.domain.pads
                _starts_out = _dofs.codomain.starts
                _ends_out = _dofs.codomain.ends
                _pads_out = _dofs.codomain.pads

                _ptsG, _wtsG, _spans, _bases, _subs = self.prepare_projection_of_basis(V1d, W1d, _starts_out, _ends_out)

                _fun_w = evaluate_fun_weights(_ptsG, _wtsG, f)

                # Call the kernel
                assemble(_dofs._data,
                         _starts_in, _ends_in, _pads_in,
                         _starts_out, _ends_out, _pads_out,
                         _fun_w,
                         _spans[0], _spans[1], _spans[2],
                         _bases[0], _bases[1], _bases[2],
                         _subs[0], _subs[1], _subs[2],
                         V1d[0].nbasis, V1d[1].nbasis, V1d[2].nbasis,
                         W1d[0].degree, W1d[1].degree, W1d[2].degree)

                _blocks[-1] += [_dofs]

        _len = sum([len(li) for li in _blocks])

        if _len > 1:
            self._dofs_mat = BlockMatrix(
                V.vector_space, W.vector_space, _blocks)
        else:
            self._dofs_mat = _blocks[0][0]

    def dot(self, u):
        '''Applies the MHD operator to the FE coefficients u belonging to the domain.

        Parameters
        ----------
            u : StencilVector or BlockVector
                Input FE coefficients from V.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from W.vector_space.'''

        rhs = self._dofs_mat.dot(u)

        rhs.update_ghost_regions()

        assert rhs.space == self._solver.space

        return self._solver.solve(rhs)

    def prepare_projection_of_basis(self, V1d, W1d, starts_out, ends_out):
        '''Obtain knot span indices and basis functions evaluated at projection point sets of a given space.

        Parameters
        ----------
            V1d : 3-list
                Three SplineSpace objects from Psydac from the input space (to be projected).

            W1d : 3-list
                Three SplineSpace objects from Psydac from the output space (projected onto).

            starts_out : 3-list
                Global starting indices of process. 

            ends_out : 3-list
                Global ending indices of process.

        Returns
        -------
            ptsG : 3-tuple of 2d float arrays
                Quadrature points (or Greville points for interpolation) in each dimension in format (interval, quadrature point).

            wtsG : 3-tuple of 2d float arrays
                Quadrature weights (or ones for interpolation) in each dimension in format (interval, quadrature point).

            spans : 3-tuple of 2d int arrays
                Knot span indices in each direction in format (n, nq).

            bases : 3-tuple of 3d float arrays
                Values of p + 1 non-zero eta basis functions at quadrature points in format (n, nq, basis).'''

        import numpy as np
        import math

        import psydac.core.bsplines as bsp

        x_grid = []
        subs = []
        pts = []
        wts = []
        spans = []
        bases = []

        # Loop over direction, prepare point sets and evaluate basis functions
        k = 0
        for space_in, space_out, s, e in zip(V1d, W1d, starts_out, ends_out):

            _greville_loc = space_out.greville[s : e + 1]
            _histopol_loc = space_out.histopolation_grid[s : e + 2]

            # k += 1
            # print(f'\nrank: {self._mpi_comm.Get_rank()} | Direction {k}, space_out attributes:')
            # # # print('--------------------------------')
            # print(f'rank: {self._mpi_comm.Get_rank()} | breaks       : {space_out.breaks}')
            # # # print(f'rank: {self._mpi_comm.Get_rank()} | degree       : {space_out.degree}')
            # # # print(f'rank: {self._mpi_comm.Get_rank()} | kind         : {space_out.basis}')
            # # print(f'rank: {self._mpi_comm.Get_rank()} | greville        : {space_out.greville}')
            # # print(f'rank: {self._mpi_comm.Get_rank()} | greville[s:e+1] : {_greville_loc}')
            # # # print(f'rank: {self._mpi_comm.Get_rank()} | ext_greville : {space_out.ext_greville}')
            # print(f'rank: {self._mpi_comm.Get_rank()} | histopol_grid : {space_out.histopolation_grid}')
            # print(f'rank: {self._mpi_comm.Get_rank()} | _histopol_loc : {_histopol_loc}')
            # print(f'rank: {self._mpi_comm.Get_rank()} | dim W: {space_out.nbasis}')
            # # # print(f'rank: {self._mpi_comm.Get_rank()} | project' + V1d[0].basis + V1d[1].basis + V1d[2].basis + ' to ' + W1d[0].basis + W1d[1].basis + W1d[2].basis)

            self._mpi_comm.Barrier()

            if space_out.basis == 'B':
                x_grid += [_greville_loc]
                pts += [_greville_loc[:, None]]
                wts += [np.ones(pts[-1].shape, dtype=float)]
                # sub-interval index is always 0 for interpolation.
                subs += [np.zeros(pts[-1].shape[0], dtype=int)]

            elif space_out.basis == 'M':

                if space_out.degree % 2 == 0:
                    union_breaks = space_out.breaks
                else:
                    union_breaks = space_out.breaks[:-1]

                # Make union of Greville and break points
                tmp = set(np.round_(space_out.histopolation_grid, decimals=14)).union(
                    np.round_(union_breaks, decimals=14))

                tmp = list(tmp)
                tmp.sort()
                tmp_a = np.array(tmp)

                x_grid += [tmp_a[np.logical_and(tmp_a >= np.min(_histopol_loc) - 1e-14, tmp_a <= np.max(_histopol_loc) + 1e-14)]]

                # determine subinterval index (= 0 or 1):
                subs += [np.zeros(x_grid[-1][:-1].size, dtype=int)]
                for n, x_h in enumerate(x_grid[-1][:-1]):
                    add = 1
                    for x_g in _histopol_loc:
                        if math.isclose(x_h, x_g):
                            add = 0
                    subs[-1][n] += add

                # Gauss - Legendre quadrature points and weights
                _n_quad = space_in.degree + 1  # products of basis functions are integrated exactly
                _pts_loc, _wts_loc = np.polynomial.legendre.leggauss(_n_quad)

                _x, _w = bsp.quadrature_grid(x_grid[-1], _pts_loc, _wts_loc)

                pts += [_x % 1.]
                wts += [_w]

            #print(f'rank: {self._mpi_comm.Get_rank()} | Direction {k}, x_grid       : {x_grid[-1]}')

            # Knot span inidices and V-basis functions evaluated at W-point sets
            _s, _b = self.get_span_and_basis(pts[-1], space_in)

            spans += [_s]
            bases += [_b]

        ptsG = tuple(pts)
        wtsG = tuple(wts)
        spans = tuple(spans)
        bases = tuple(bases)
        subs = tuple(subs)

        return ptsG, wtsG, spans, bases, subs

    def get_span_and_basis(self, pts, space):
        '''Compute the knot span index and the values of p + 1 basis function at each point in pts.

        Parameters
        ----------
            pts : np.array
                2d array of points (interval, quadrature point).

            space : SplineSpace
                Psydac object, the 1d spline space to be projected.

        Returns
        -------
            span : np.array
                2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

            basis : np.array
                3d array of values of basis functions indexed by (n, nq, basis function). 
        '''

        import struphy.feec.bsplines as bsp_s
        import psydac.core.bsplines as bsp

        # Extract knot vectors, degree and kind of basis
        T = space.knots
        p = space.degree

        span = np.zeros(pts.shape, dtype=int)
        basis = np.zeros((*pts.shape, p + 1), dtype=float)

        for n in range(pts.shape[0]):
            for nq in range(pts.shape[1]):
                # avoid 1. --> 0. for clamped interpolation
                x = pts[n, nq] % (1. + 1e-14)
                span_tmp = bsp.find_span(T, p, x)
                basis[n, nq, :] = bsp.basis_funs_all_ders(
                    T, p, x, span_tmp, 0, normalization=space.basis)
                span[n, nq] = span_tmp  # % space.nbasis

        return span, basis
