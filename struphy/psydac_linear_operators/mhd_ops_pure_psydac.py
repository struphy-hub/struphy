import numpy as np

from psydac.fem.basic import FemSpace
from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix

from struphy.psydac_linear_operators.mhd_ops_kernels_pure_psydac import assemble_dofs_for_weighted_basisfuns as assemble
from struphy.psydac_linear_operators.prepare_projection import evaluate_fun_weights
from struphy.psydac_linear_operators.linear_operators import LinOpWithTransp
from struphy.psydac_linear_operators.H1vec_psydac import Projector_H1vec


class MHD_ops:
    '''Assembles some or all MHD operators needed for various discretizations of linear MHD equations.

        See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x`.

        Parameters
        ----------
        DERHAM : Psydac object
            The Derham sequence object, obained from Psydac's discretize.

        V0vec : Femspace
            ProductFemSpace(V0, V0, V0).

        nq_pr : list
            Number of quadrature points used in histopolation in each direction.

        EQ_MHD : Struphy object
            MHD equilibrium from struphy.fields_equil.mhd_equil (pullbacks must be enabled).

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

    def __init__(self, DERHAM, V0vec, nq_pr, EQ_MHD, F, assemble_all=False, mpi_comm=None):

        self._mpi_comm = mpi_comm

        # Missing in Psydac: inverse metric tensor
        def _Ginv(x1, x2, x3): return np.matmul(F.jacobian_inv(x1, x2, x3), F.jacobian_inv(x1, x2, x3).T)

        # Psydac spline spaces
        _V0 = DERHAM.V0
        _V1 = DERHAM.V1
        _V2 = DERHAM.V2
        _V3 = DERHAM.V3
        self._V0 = _V0
        self._V1 = _V1
        self._V2 = _V2
        self._V3 = _V3

        # Spaces for operators X1 and X2
        self._V0vec = V0vec
        self._Pi0vec = Projector_H1vec(self._V0vec)

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
            [lambda x1, x2, x3: 0,   EQ_MHD.j2_eq_3,   EQ_MHD.j2_eq_2],
            [EQ_MHD.j2_eq_3, lambda x1, x2, x3: 0,   EQ_MHD.j2_eq_1],
            [EQ_MHD.j2_eq_2,   EQ_MHD.j2_eq_1, lambda x1, x2, x3: 0],
        ]
        _b2_cross = [
            [lambda x1, x2, x3: 0,   EQ_MHD.b2_eq_3,   EQ_MHD.b2_eq_2],
            [EQ_MHD.b2_eq_3, lambda x1, x2, x3: 0,   EQ_MHD.b2_eq_1],
            [EQ_MHD.b2_eq_2,   EQ_MHD.b2_eq_1, lambda x1, x2, x3: 0],
        ]

        def _eval_cross(x1, x2, x3, fun_list): return np.array(
            [[_cross_mask[m][n] * ele(x1, x2, x3) for n, ele in enumerate(row)] for m, row in enumerate(fun_list)])

        # Scalar functions
        def _fun_K1(x1, x2, x3): return EQ_MHD.p3_eq(x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))

        _fun_K10 = EQ_MHD.p0_eq

        def _fun_K2(x1, x2, x3): return EQ_MHD.p3_eq(x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))

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
                                n=n: EQ_MHD.n3_eq(x1, x2, x3) * _Ginv(x1, x2, x3)[m, n]]
                _fun_W1[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD.n3_eq( 
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_U1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: np.sqrt(F.metric_det(x1, x2, x3)) * _Ginv(x1, x2, x3)[m, n]]
                _fun_P1[-1] += [lambda x1, x2, x3, m=m, n=n: _cross_mask[m][n] *
                                _j2_cross[m][n](x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_S1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: EQ_MHD.p3_eq(x1, x2, x3) * _Ginv(x1, x2, x3)[m, n]]
                _fun_S10[-1] += [lambda x1, x2, x3, m=m,
                                 n=n: EQ_MHD.p0_eq(x1, x2, x3) if m == n else 0.]
                _fun_T1[-1] += [lambda x1, x2, x3, m=m, n=n: (_eval_cross(
                    x1, x2, x3, _b2_cross) @ _Ginv(x1, x2, x3))[m, n]]  # Matrix product!
                _fun_X1[-1] += [lambda x1, x2, x3, m=m,
                                n=n: (F.jacobian_inv(x1, x2, x3).T)[m, n]]

                _fun_Q2[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD.n3_eq( 
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_T2[-1] += [lambda x1, x2, x3, m=m, n=n: _cross_mask[m][n] *
                                _b2_cross[m][n](x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_P2[-1] += [lambda x1, x2, x3, m=m, n=n: (_Ginv(x1, x2, x3) @ _eval_cross(
                    x1, x2, x3, _j2_cross))[m, n]]  # Matrix product!
                _fun_S2[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD.p3_eq(
                    x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3)) if m == n else 0.]
                _fun_X2[-1] += [lambda x1, x2, x3, m=m,
                                n=n: F.jacobian(x1, x2, x3)[m, n] / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_Z20[-1] += [lambda x1, x2, x3, m=m,
                                 n=n: F.metric(x1, x2, x3)[m, n] / np.sqrt(F.metric_det(x1, x2, x3))]
                _fun_S20[-1] += [lambda x1, x2, x3, m=m, n=n: EQ_MHD.p0_eq(
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
            self.assemble_Q1()
            self.assemble_W1()
            self.assemble_U1()
            self.assemble_P1()
            self.assemble_S1()
            self.assemble_S10()
            self.assemble_K1()
            self.assemble_K10()
            self.assemble_T1()
            self.assemble_X1()

            # MHD operators with velocity (up) as 2-form:
            self.assemble_Q2() 
            self.assemble_T2() 
            self.assemble_P2() 
            self.assemble_S2() 
            self.assemble_K2()
            self.assemble_Z20()
            self.assemble_Y20()
            self.assemble_S20()
            self.assemble_X2()

    def assemble_Q1(self):
        self.Q1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_Q1, self._mpi_comm)
        self.Q1T = self.Q1.transpose()

    def assemble_W1(self):
        self.W1 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_W1, self._mpi_comm)
        self.W1T = self.W1.transpose()

    def assemble_U1(self):
        self.U1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_U1, self._mpi_comm)
        self.U1T = self.U1.transpose()

    def assemble_P1(self):
        self.P1 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_P1, self._mpi_comm)
        self.P1T = self.P1.transpose()

    def assemble_S1(self):
        self.S1 = MHD_operator(self._V1, self._V2, self._Pi2, self._fun_S1, self._mpi_comm)
        self.S1T = self.S1.transpose()

    def assemble_S10(self):
        self.S10 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_S10, self._mpi_comm)
        self.S10T = self.S10.transpose()

    def assemble_K1(self):
        self.K1 = MHD_operator(self._V3, self._V3, self._Pi3, [[self._fun_K1]], self._mpi_comm)
        self.K1T = self.K1.transpose()

    def assemble_K10(self):
        self.K10 = MHD_operator(self._V0, self._V0, self._Pi0, [[self._fun_K10]], self._mpi_comm)
        self.K10T = self.K10.transpose()

    def assemble_T1(self):
        self.T1 = MHD_operator(self._V1, self._V1, self._Pi1, self._fun_T1, self._mpi_comm)
        self.T1T = self.T1.transpose()

    def assemble_X1(self):
        self.X1 = MHD_operator(self._V1, self._V0vec, self._Pi0vec, self._fun_X1, self._mpi_comm) 
        self.X1T = self.X1.transpose()

    def assemble_Q2(self):
        self.Q2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_Q2, self._mpi_comm)
        self.Q2T = self.Q2.transpose()

    def assemble_T2(self):
        self.T2 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_T2, self._mpi_comm)
        self.T2T = self.T2.transpose()

    def assemble_P2(self):
        self.P2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_P2, self._mpi_comm)
        self.P2T = self.P2.transpose()

    def assemble_S2(self):
        self.S2 = MHD_operator(self._V2, self._V2, self._Pi2, self._fun_S2, self._mpi_comm)
        self.S2T = self.S2.transpose()

    def assemble_K2(self):
        self.K2 = MHD_operator(self._V3, self._V3, self._Pi3, [[self._fun_K2]], self._mpi_comm)
        self.K2T = self.K2.transpose()

    def assemble_X2(self):
        self.X2 = MHD_operator(self._V2, self._V0vec, self._Pi0vec, self._fun_X2, self._mpi_comm) 
        self.X2T = self.X2.transpose()

    def assemble_Z20(self):
        self.Z20 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_Z20, self._mpi_comm)
        self.Z20T = self.Z20.transpose()

    def assemble_Y20(self):
        self.Y20 = MHD_operator(self._V0, self._V3, self._Pi3, [[self._fun_Y20]], self._mpi_comm)
        self.Y20T = self.Y20.transpose()

    def assemble_S20(self):
        self.S20 = MHD_operator(self._V2, self._V1, self._Pi1, self._fun_S20, self._mpi_comm)
        self.S20T = self.S20.transpose()


class MHD_operator( LinOpWithTransp ):
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

        transposed : boolean
            False: map V -> W or True: map W -> v.
    '''

    def __init__(self, V, W, pi_W, fun, mpi_comm, transposed=False):

        assert isinstance(V, FemSpace) 
        assert isinstance(W, FemSpace) 

        self._V = V
        self._W = W
        self._pi_W = pi_W
        self._fun = fun
        self._mpi_comm = mpi_comm
        self._transposed = transposed

        # Retrieve solver
        self._solver = pi_W.solver

        # Handle transpose
        if transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        # Input space: Stencil vector spaces and 1d spaces
        if hasattr(V.symbolic_space, 'name'):
            if V.symbolic_space.name in {'H1', 'L2'}:
                _Vspaces = [V.vector_space]
                _V1ds = [V.spaces]
            else:
                _Vspaces = V.vector_space
                _V1ds = [comp.spaces for comp in V.spaces]
            print(f'From {V.symbolic_space.name} ...')
        else:
            _Vspaces = V.vector_space
            _V1ds = [comp.spaces for comp in V.spaces]
            print(f'From H1vec ...')

        # Output space: Stencil vector spaces and 1d spaces
        if hasattr(W.symbolic_space, 'name'):
            if W.symbolic_space.name in {'H1', 'L2'}:
                _Wspaces = [W.vector_space]
                _W1ds = [W.spaces]

            else:
                _Wspaces = W.vector_space
                _W1ds = [comp.spaces for comp in W.spaces]
            print(f'... to {W.symbolic_space.name}.')
        else:
            _Wspaces = W.vector_space
            _W1ds = [comp.spaces for comp in W.spaces]
            print(f'... to H1vec.')

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

                _ptsG, _wtsG, _spans, _bases, _subs = self._prepare_projection_of_basis(
                    V1d, W1d, _starts_out, _ends_out)

                _fun_w = evaluate_fun_weights(_ptsG, _wtsG, f)

                # Call the kernel
                assemble(_dofs._data,
                         np.array(_starts_in), np.array(_ends_in), np.array(_pads_in),
                         np.array(_starts_out), np.array(_ends_out), np.array(_pads_out),
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

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._V.dtype

    def dot( self, v, out=None ):
        if self._transposed:
            tmp = self._transpose_dot(v)
        else:
            tmp = self._dot(v)
        return tmp

    def transpose(self):
        return MHD_operator(self._V, self._W, self._pi_W, self._fun, self._mpi_comm, transposed=True) # Everything is assembled again here, bad.

    def _dot(self, v):
        '''Applies the MHD operator to the FE coefficients v belonging to V.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Input FE coefficients from V.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from W.vector_space.'''

        assert v.space == self.domain.vector_space

        rhs = self._dofs_mat.dot(v)
        rhs.update_ghost_regions()

        assert rhs.space == self._solver.space

        tmp = self._solver.solve(rhs)
        tmp.update_ghost_regions()

        return tmp

    def _transpose_dot(self, v):
        '''Applies the transposed MHD operator to the FE coefficients v belonging to W.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Output FE coefficients from W.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from V.vector_space.'''

        assert v.space == self.domain.vector_space

        tmp = self._solver.solve(v, transposed=True)
        tmp.update_ghost_regions()

        assert tmp.space == self._dofs_mat.codomain

        tmp2 = self._dofs_mat.transpose().dot(tmp)
        tmp2.update_ghost_regions()

        return tmp2

    def _prepare_projection_of_basis(self, V1d, W1d, starts_out, ends_out):
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

            _greville_loc = space_out.greville[s: e + 1]
            _histopol_loc = space_out.histopolation_grid[s: e + 2]

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

                x_grid += [tmp_a[np.logical_and(tmp_a >= np.min(
                    _histopol_loc) - 1e-14, tmp_a <= np.max(_histopol_loc) + 1e-14)]]

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
            _s, _b = self._get_span_and_basis(pts[-1], space_in)

            spans += [_s]
            bases += [_b]

        ptsG = tuple(pts)
        wtsG = tuple(wts)
        spans = tuple(spans)
        bases = tuple(bases)
        subs = tuple(subs)

        return ptsG, wtsG, spans, bases, subs

    def _get_span_and_basis(self, pts, space):
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
