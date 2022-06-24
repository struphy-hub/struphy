import numpy as np

import psydac.core.bsplines as bsp

from psydac.linalg.basic import LinearOperator
from psydac.fem.basic import FemSpace
from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix
from psydac.feec.global_projectors import GlobalProjector

from struphy.psydac_api.mhd_ops_kernels_pure_psydac import assemble_dofs_for_weighted_basisfuns_1d as assemble_1d
from struphy.psydac_api.mhd_ops_kernels_pure_psydac import assemble_dofs_for_weighted_basisfuns_2d as assemble_2d
from struphy.psydac_api.mhd_ops_kernels_pure_psydac import assemble_dofs_for_weighted_basisfuns_3d as assemble_3d

from struphy.psydac_api.prepare_projection import evaluate_fun_weights_1d, evaluate_fun_weights_2d, evaluate_fun_weights_3d


class MHDOperators:
    """
    Assembles some or all MHD operators needed for various discretizations of linear MHD equations.

    See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x`.

    Parameters
    ----------
    derham : Derham
        The Derham sequence object, obained from Psydac's discretize.

    eq_mhd : Struphy object
        MHD equilibrium from struphy.fields_equil.mhd_equil (pullbacks must be enabled).

    F : Psydac mapping
        Obtained via .get_callable_mapping().

    assemble_all : bool
        Assemble all `MHDOperator`s in constructor. Only for testing. Please assemble individually by calling `assemble_XX()` for each operator.

    Notes
    -----
    The `X1`, `X2` operators are handled differently, because it outputs 3 scalar spaces instead of a pure scalar or vector space.
    In order not to modify the `MHDOperator` class, we give a set of three functions, each accessing each row of the input matrix-valued function.
    """

    def __init__(self, derham, eq_mhd, F, assemble_all=False):

        # Missing in Psydac: inverse metric tensor
        def _Ginv(e1, e2, e3): return np.matmul(F.jacobian_inv(e1, e2, e3), F.jacobian_inv(e1, e2, e3).T)

        # Psydac spline spaces
        self._V0 = derham.V0
        self._V1 = derham.V1
        self._V2 = derham.V2
        self._V3 = derham.V3
        self._V0vec = derham.V0vec
        
        # Psydac projectors
        self._P0 = derham.P0
        self._P1 = derham.P1
        self._P2 = derham.P2
        self._P3 = derham.P3
        self._P0vec = derham.P0vec

        # Cross product matrices:
        _cross_mask = [
            [1, -1,  1],
            [1,  1, -1],
            [-1,  1,  1],
        ]
        _j2_cross = [
            [lambda e1, e2, e3: 0,   eq_mhd.j2_eq_3,   eq_mhd.j2_eq_2],
            [eq_mhd.j2_eq_3, lambda e1, e2, e3: 0,   eq_mhd.j2_eq_1],
            [eq_mhd.j2_eq_2,   eq_mhd.j2_eq_1, lambda e1, e2, e3: 0],
        ]
        _b2_cross = [
            [lambda e1, e2, e3: 0,   eq_mhd.b2_eq_3,   eq_mhd.b2_eq_2],
            [eq_mhd.b2_eq_3, lambda e1, e2, e3: 0,   eq_mhd.b2_eq_1],
            [eq_mhd.b2_eq_2,   eq_mhd.b2_eq_1, lambda e1, e2, e3: 0],
        ]

        def _eval_cross(e1, e2, e3, fun_list): return np.array(
            [[_cross_mask[m][n] * ele(e1, e2, e3) for n, ele in enumerate(row)] for m, row in enumerate(fun_list)])

        
        # Scalar functions
        _fun_K1 = [[lambda e1, e2, e3 : eq_mhd.p3_eq(e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3))]]
        _fun_K10 = [[lambda e1, e2, e3 : eq_mhd.p0_eq(e1, e2, e3)]]
        
        _fun_K2 = [[lambda e1, e2, e3 : eq_mhd.p3_eq(e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3))]]
        _fun_Y20 = [[lambda e1, e2, e3 : np.sqrt(F.metric_det(e1, e2, e3))]]
        
        # 'Matrix' functions
        _fun_Q1 = []
        _fun_W1 = []
        _fun_U1 = []
        _fun_P1 = []
        _fun_S1 = []
        _fun_T1 = []
        _fun_X1 = []
        _fun_S10 = []
        
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
            _fun_T1 += [[]]
            _fun_X1 += [[]]
            _fun_S10 += [[]]

            _fun_Q2 += [[]]
            _fun_T2 += [[]]
            _fun_P2 += [[]]
            _fun_S2 += [[]]
            _fun_X2 += [[]]
            _fun_Z20 += [[]]
            _fun_S20 += [[]]
            for n in range(3):
                # See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF_for_tests.projectors_dot_x`.
                _fun_Q1[-1] += [lambda e1, e2, e3, m=m,
                                n=n: eq_mhd.n3_eq(e1, e2, e3) * _Ginv(e1, e2, e3)[m, n]]
                _fun_W1[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.n3_eq( 
                    e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3)) if m == n else 0.]
                _fun_U1[-1] += [lambda e1, e2, e3, m=m,
                                n=n: np.sqrt(F.metric_det(e1, e2, e3)) * _Ginv(e1, e2, e3)[m, n]]
                _fun_P1[-1] += [lambda e1, e2, e3, m=m, n=n: _cross_mask[m][n] *
                                _j2_cross[m][n](e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3))]
                _fun_S1[-1] += [lambda e1, e2, e3, m=m,
                                n=n: eq_mhd.p3_eq(e1, e2, e3) * _Ginv(e1, e2, e3)[m, n]]
                _fun_T1[-1] += [lambda e1, e2, e3, m=m, n=n: (_eval_cross(
                    e1, e2, e3, _b2_cross) @ _Ginv(e1, e2, e3))[m, n]]  # Matrix product!
                _fun_X1[-1] += [lambda e1, e2, e3, m=m,
                                n=n: (F.jacobian_inv(e1, e2, e3).T)[m, n]]
                _fun_S10[-1] += [lambda e1, e2, e3, m=m,
                                 n=n: eq_mhd.p0_eq(e1, e2, e3) if m == n else 0.]

                _fun_Q2[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.n3_eq( 
                    e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3)) if m == n else 0.]
                _fun_T2[-1] += [lambda e1, e2, e3, m=m, n=n: _cross_mask[m][n] *
                                _b2_cross[m][n](e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3))]
                _fun_P2[-1] += [lambda e1, e2, e3, m=m, n=n: (_Ginv(e1, e2, e3) @ _eval_cross(
                    e1, e2, e3, _j2_cross))[m, n]]  # Matrix product!
                _fun_S2[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.p3_eq(
                    e1, e2, e3) / np.sqrt(F.metric_det(e1, e2, e3)) if m == n else 0.]
                _fun_X2[-1] += [lambda e1, e2, e3, m=m,
                                n=n: F.jacobian(e1, e2, e3)[m, n] / np.sqrt(F.metric_det(e1, e2, e3))]
                _fun_Z20[-1] += [lambda e1, e2, e3, m=m,
                                 n=n: F.metric(e1, e2, e3)[m, n] / np.sqrt(F.metric_det(e1, e2, e3))]
                _fun_S20[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.p0_eq(
                    e1, e2, e3) * F.metric(e1, e2, e3)[m, n] / np.sqrt(F.metric_det(e1, e2, e3))]

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
        self._fun_T1 = _fun_T1
        self._fun_X1 = _fun_X1
        self._fun_S10 = _fun_S10

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
            self.assemble_K1()
            self.assemble_Q1()
            self.assemble_W1()
            self.assemble_U1()
            self.assemble_P1()
            self.assemble_S1()
            self.assemble_T1()
            self.assemble_X1()
            self.assemble_K10()
            self.assemble_S10()

            # MHD operators with velocity (up) as 2-form:
            self.assemble_Q2() 
            self.assemble_T2() 
            self.assemble_P2() 
            self.assemble_S2() 
            self.assemble_K2()
            self.assemble_X2()
            self.assemble_Y20()
            self.assemble_Z20()
            self.assemble_S20()
            
    # MHD operators with velocity (up) as 1-form:
    def assemble_K1(self, with_transposed=True):
        self.K1 = MHDOperator(self._P3, self._V3, self._fun_K1, with_transposed)
    
    def assemble_Q1(self, with_transposed=True):
        self.Q1 = MHDOperator(self._P2, self._V1, self._fun_Q1, with_transposed)

    def assemble_W1(self, with_transposed=True):
        self.W1 = MHDOperator(self._P1, self._V1, self._fun_W1, with_transposed)

    def assemble_U1(self, with_transposed=True):
        self.U1 = MHDOperator(self._P2, self._V1, self._fun_U1, with_transposed)

    def assemble_P1(self, with_transposed=True):
        self.P1 = MHDOperator(self._P1, self._V2, self._fun_P1, with_transposed)

    def assemble_S1(self, with_transposed=True):
        self.S1 = MHDOperator(self._P2, self._V1, self._fun_S1, with_transposed)

    def assemble_T1(self, with_transposed=True):
        self.T1 = MHDOperator(self._P1, self._V1, self._fun_T1, with_transposed)

    def assemble_X1(self, with_transposed=True):
        self.X1 = MHDOperator(self._P0vec, self._V1, self._fun_X1, with_transposed) 

    def assemble_K10(self, with_transposed=True):
        self.K10 = MHDOperator(self._P0, self._V0, self._fun_K10, with_transposed)

    def assemble_S10(self, with_transposed=True):
        self.S10 = MHDOperator(self._P1, self._V1, self._fun_S10, with_transposed)

    # MHD operators with velocity (up) as 2-form:
    def assemble_K2(self, with_transposed=True):
        self.K2 = MHDOperator(self._P3, self._V3, self._fun_K2, with_transposed)
    
    def assemble_Q2(self, with_transposed=True):
        self.Q2 = MHDOperator(self._P2, self._V2, self._fun_Q2, with_transposed)

    def assemble_T2(self, with_transposed=True):
        self.T2 = MHDOperator(self._P1, self._V2, self._fun_T2, with_transposed)

    def assemble_P2(self, with_transposed=True):
        self.P2 = MHDOperator(self._P2, self._V2, self._fun_P2, with_transposed)

    def assemble_S2(self, with_transposed=True):
        self.S2 = MHDOperator(self._P2, self._V2, self._fun_S2, with_transposed)

    def assemble_X2(self, with_transposed=True):
        self.X2 = MHDOperator(self._P0vec, self._V2, self._fun_X2, with_transposed) 
     
    def assemble_Y20(self, with_transposed=True):
        self.Y20 = MHDOperator(self._P3, self._V0, self._fun_Y20, with_transposed)
    
    def assemble_Z20(self, with_transposed=True):
        self.Z20 = MHDOperator(self._P1, self._V2, self._fun_Z20, with_transposed)

    def assemble_S20(self, with_transposed=True):
        self.S20 = MHDOperator(self._P1, self._V2, self._fun_S20, with_transposed)


class MHDOperator(LinearOperator):
    '''
    Class for MHD specific projection operators PI_ijk(fun Lambda_mno).

    Parameters
    ----------
        P : GlobalProjector
            Psydac de Rham projector into space W = P.space (codomain of operator), henceforth called "output space".
        
        V : TensorFemSpace or ProductFemSpace
            Domain of the operator, henceforth called "input space".

        fun : list
            List of functions of (eta1, eta2, eta3) that multiply the basis functions of the input space V.
            3x3 matrix-valued (nested list [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]]) if V is ProductFemSPace, 
            scalar-valued [f] list otherwise.

        with_transposed : boolean
            False: map V -> W or True: map W -> V.
    '''

    def __init__(self, P, V, fun, with_transposed=True):

        assert isinstance(P, GlobalProjector) 
        assert isinstance(V, FemSpace) 

        self._P = P
        self._V = V
        self._fun = fun
        self._with_transposed = with_transposed
        
        self._domain = V.vector_space
        self._codomain = P.space.vector_space

        # Retrieve solver
        self._solver = P.solver

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
        if hasattr(P.space.symbolic_space, 'name'):
            if P.space.symbolic_space.name in {'H1', 'L2'}:
                _Wspaces = [P.space.vector_space]
                _W1ds = [P.space.spaces]
            else:
                _Wspaces = P.space.vector_space
                _W1ds = [comp.spaces for comp in P.space.spaces]
            print(f'... to {P.space.symbolic_space.name}.')
        else:
            _Wspaces = P.space.vector_space
            _W1ds = [comp.spaces for comp in P.space.spaces]
            print(f'... to H1vec.')
            
        # Retrieve number of quadrature points
        _nqs = []
        
        for d in range(P.dim):
            if hasattr(P.space.symbolic_space, 'name'):
                if P.space.symbolic_space.name == 'Hcurl':
                    _nqs += [P.grid_x[d][d].shape[1]]
                
                elif P.space.symbolic_space.name == 'Hdiv':
                    
                    if P.dim == 2:
                        if   d == 0:
                            _nqs += [P.grid_x[1][0].shape[1]]
                        elif d == 1:
                            _nqs += [P.grid_x[0][1].shape[1]]
                    else:
                        if   d == 0:
                            _nqs += [P.grid_x[2][0].shape[1]]
                        elif d == 1:
                            _nqs += [P.grid_x[0][1].shape[1]]
                        elif d == 2:
                            _nqs += [P.grid_x[1][2].shape[1]]       
                    
                elif P.space.symbolic_space.name == 'L2':
                    _nqs += [P.grid_x[0][d].shape[1]]
                else:
                    _nqs += [1]
            else:
                _nqs += [1]

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

                _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                    V1d, W1d, _starts_out, _ends_out, _nqs)
                
                # Evaluate weight function times quadrature weights
                if   V.ldim == 1:
                    _fun_w = evaluate_fun_weights_1d(_ptsG, _wtsG, f)
                elif V.ldim == 2:
                    _fun_w = evaluate_fun_weights_2d(_ptsG, _wtsG, f)
                elif V.ldim == 3:
                    _fun_w = evaluate_fun_weights_3d(_ptsG, _wtsG, f)
                
                
                # Call the kernel if weight function is not zero
                if np.any(_fun_w):
                    if V.ldim == 1:
                        
                        assemble_1d(_dofs._data,
                                 np.array(_starts_in), np.array(_ends_in), np.array(_pads_in),
                                 np.array(_starts_out), np.array(_ends_out), np.array(_pads_out),
                                 _fun_w,
                                 _spans[0],
                                 _bases[0],
                                 _subs[0],
                                 V1d[0].nbasis,
                                 W1d[0].degree)

                    elif V.ldim == 2:

                        assemble_2d(_dofs._data,
                                 np.array(_starts_in), np.array(_ends_in), np.array(_pads_in),
                                 np.array(_starts_out), np.array(_ends_out), np.array(_pads_out),
                                 _fun_w,
                                 _spans[0], _spans[1],
                                 _bases[0], _bases[1],
                                 _subs[0], _subs[1],
                                 V1d[0].nbasis, V1d[1].nbasis,
                                 W1d[0].degree, W1d[1].degree)

                    elif V.ldim == 3:

                        # Call the kernel
                        assemble_3d(_dofs._data,
                                 np.array(_starts_in), np.array(_ends_in), np.array(_pads_in),
                                 np.array(_starts_out), np.array(_ends_out), np.array(_pads_out),
                                 _fun_w,
                                 _spans[0], _spans[1], _spans[2],
                                 _bases[0], _bases[1], _bases[2],
                                 _subs[0], _subs[1], _subs[2],
                                 V1d[0].nbasis, V1d[1].nbasis, V1d[2].nbasis,
                                 W1d[0].degree, W1d[1].degree, W1d[2].degree)

                    _blocks[-1] += [_dofs]
                    
                else:
                    _blocks[-1] += [None]

        _len = sum([len(li) for li in _blocks])

        if _len > 1:
            self._dofs_mat = BlockMatrix(
                V.vector_space, P.space.vector_space, _blocks)
        else:
            if _blocks[0][0] is not None:
                self._dofs_mat = _blocks[0][0]
            else:
                self._dofs_mat = _dofs
                
        if with_transposed:
            self._dofs_mat_T = self._dofs_mat.transpose()

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._V.dtype

    @property
    def with_transposed(self):
        return self._with_transposed
    
    @with_transposed.setter
    def with_transposed(self, value):
    
        if value:
            self._dofs_mat_T = self._dofs_mat.transpose()
            
        self._with_transposed = value
    

    def dot(self, v, out=None):
        '''Applies the MHD operator to the FE coefficients v belonging to V.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Input FE coefficients from V.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from W.vector_space.'''

        assert v.space == self.domain

        rhs = self._dofs_mat.dot(v)
        rhs.update_ghost_regions()

        assert rhs.space == self.codomain

        tmp = self._solver.solve(rhs)
        tmp.update_ghost_regions()
        
        assert tmp.space == self.codomain

        return tmp

    def transpose_dot(self, v):
        '''Applies the transposed MHD operator to the FE coefficients v belonging to W.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Output FE coefficients from W.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from V.vector_space.'''

        assert v.space == self.codomain

        tmp = self._solver.solve(v, transposed=True)
        tmp.update_ghost_regions()

        assert tmp.space == self.codomain

        tmp2 = self._dofs_mat_T.dot(tmp)
        tmp2.update_ghost_regions()
        
        assert tmp2.space == self.domain

        return tmp2

    
    
    
    
    
    
    
def prepare_projection_of_basis(V1d, W1d, starts_out, ends_out, n_quad=None):
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
            
        n_quad : 3_list
            Number of quadrature points per histpolation interval. If not given, is set to V1d.degree + 1.

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

    import psydac.core.bsplines as bsp

    x_grid, subs, pts, wts, spans, bases = [], [], [], [], [], []
    
    # Loop over direction, prepare point sets and evaluate basis functions
    direction = 0
    for space_in, space_out, s, e in zip(V1d, W1d, starts_out, ends_out):

        greville_loc = space_out.greville[s: e + 1]
        histopol_loc = space_out.histopolation_grid[s: e + 2]
        
        # make sure that greville points used for interpolation are in [0, 1]
        assert np.all(np.logical_and(greville_loc >= 0., greville_loc <= 1.))

        # k += 1
        # print(f'\nrank: {self._mpi_comm.Get_rank()} | Direction {k}, space_out attributes:')
        # # # print('--------------------------------')
        # print(f'rank: {self._mpi_comm.Get_rank()} | breaks       : {space_out.breaks}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | degree       : {space_out.degree}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | kind         : {space_out.basis}')
        # # print(f'rank: {self._mpi_comm.Get_rank()} | greville        : {space_out.greville}')
        # # print(f'rank: {self._mpi_comm.Get_rank()} | greville[s:e+1] : {greville_loc}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | ext_greville : {space_out.ext_greville}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | histopol_grid : {space_out.histopolation_grid}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | histopol_loc : {histopol_loc}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | dim W: {space_out.nbasis}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | project' + V1d[0].basis + V1d[1].basis + V1d[2].basis + ' to ' + W1d[0].basis + W1d[1].basis + W1d[2].basis)

        # interpolation
        if space_out.basis == 'B':
            x_grid += [greville_loc]
            pts += [greville_loc[:, None]]
            wts += [np.ones(pts[-1].shape, dtype=float)]
            
            # sub-interval index is always 0 for interpolation.
            subs += [np.zeros(pts[-1].shape[0], dtype=int)]

        # histopolation
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
                histopol_loc) - 1e-14, tmp_a <= np.max(histopol_loc) + 1e-14)]]

            # determine subinterval index (= 0 or 1):
            subs += [np.zeros(x_grid[-1][:-1].size, dtype=int)]
            for n, x_h in enumerate(x_grid[-1][:-1]):
                add = 1
                for x_g in histopol_loc:
                    if abs(x_h - x_g) < 1e-14:
                        add = 0
                subs[-1][n] += add

            # Gauss - Legendre quadrature points and weights
            if n_quad is None:
                # products of basis functions are integrated exactly
                nq = space_in.degree + 1  
            else:
                nq = n_quad[direction]
                
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(nq)

            x, w = bsp.quadrature_grid(x_grid[-1], pts_loc, wts_loc)

            pts += [x % 1.]
            wts += [w]

        #print(f'rank: {self._mpi_comm.Get_rank()} | Direction {k}, x_grid       : {x_grid[-1]}')

        # Knot span indices and V-basis functions evaluated at W-point sets
        s, b = get_span_and_basis(pts[-1], space_in)

        spans += [s]
        bases += [b]
        
        direction += 1

    return tuple(pts), tuple(wts), tuple(spans), tuple(bases), tuple(subs)

def get_span_and_basis(pts, space):
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
