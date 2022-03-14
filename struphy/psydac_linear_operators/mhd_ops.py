import numpy as np

class MHD_ops:

    def __init__(self, DERHAM, nq_pr, EQ_MHD_L, F, projectors_1d):
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
                
            projectors_1d : 3-tuple of 1d projectors.
                From Struphy Projectors_global_1d class. They contain all the necessary information for Derham commuting projectors.
        '''

        # Missing in Psydac: inverse metric tensor
        self._Ginv = lambda x1, x2, x3 : np.matmul(F.jacobian_inv(x1, x2, x3), F.jacobian_inv(x1, x2, x3).T)

        # Psydac spline spaces
        self._V0 = DERHAM.V0
        self._V1 = DERHAM.V1
        self._V2 = DERHAM.V2
        self._V3 = DERHAM.V3

        # Psydac projectors
        self._P0, self._P1, self._P2, self._P3  = DERHAM.projectors(nquads=nq_pr)

        # print('\nV0.spaces[0]._interpolator:', self._V0.spaces[0]._interpolator)
        # print('\nV0.spaces[1]._interpolator:', self._V0.spaces[1]._interpolator)
        # print('\nV0.spaces[2]._interpolator:', self._V0.spaces[2]._interpolator)

        # print('\nV3.spaces[0]._histopolator:', self._V3.spaces[0]._histopolator)
        # print('\nV3.spaces[1]._histopolator:', self._V3.spaces[1]._histopolator)
        # print('\nV3.spaces[2]._histopolator:', self._V3.spaces[2]._histopolator)

        # Other
        self._EQ     = EQ_MHD_L
        self._F       = F
        self._projectors_1d = projectors_1d

    # 'Scalar' operators
    def assemble_K1(self):
        _fun_K1  = lambda x1, x2, x3 : self._EQ.p3_eq(x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3))
        self._K1 = MHD_operator(self._V3, self._V3, self._P3, [[_fun_K1]],  self._projectors_1d)
        
    def assemble_K10(self):
        _fun_K10  = self._EQ.p0_eq
        self._K10 = MHD_operator(self._V0, self._V0, self._P0, [[_fun_K10]], self._projectors_1d)

    def assemble_K2(self):
        _fun_K2   = lambda x1, x2, x3 : self._EQ.p3_eq(x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3))
        self._K2  = MHD_operator(self._V3, self._V3, self._P3, [[_fun_K2]], self._projectors_1d)

    def assemble_Y20(self):
        _fun_Y20  = lambda x1, x2, x3 : np.sqrt(self._F.metric_det(x1, x2, x3))
        self._Y20 = MHD_operator(self._V0, self._V3, self._P3, [[_fun_Y20]], self._projectors_1d)

    # 'Matrix' operators
    def assemble_Q1(self):
        _fun_Q1  = []
        for m in range(3):
            _fun_Q1  += [[]]
            for n in range(3):
                _fun_Q1[ -1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.r3_eq(x1, x2, x3) * self._Ginv(x1, x2, x3)[m, n]]

        self._Q1  = MHD_operator(self._V1, self._V2, self._P2, _fun_Q1, self._projectors_1d)

    def assemble_W1(self):
        _fun_W1  = []
        for m in range(3):
            _fun_W1  += [[]]
            for n in range(3):
                _fun_W1[ -1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.r3_eq(x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3)) if m==n else 0.]
    
        self._W1  = MHD_operator(self._V1, self._V1, self._P1, _fun_W1, self._projectors_1d)

    def assemble_U1(self):
        _fun_U1  = []
        for m in range(3):
            _fun_U1  += [[]]
            for n in range(3):
                _fun_U1[ -1] += [lambda x1, x2, x3, m=m, n=n : np.sqrt(self._F.metric_det(x1, x2, x3)) * self._Ginv(x1, x2, x3)[m, n]]

        self._U1  = MHD_operator(self._V1, self._V2, self._P2, _fun_U1, self._projectors_1d)
                
    def assemble_P1(self):
        _fun_P1  = []
        for m in range(3):
            _fun_P1  += [[]]
            for n in range(3):
                _fun_P1[ -1] += [lambda x1, x2, x3, m=m, n=n : [self._EQ.j2_eq_1, self._EQ.j2_eq_2, self._EQ.j2_eq_3][m](x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3))]

        self._P1  = MHD_operator(self._V2, self._V1, self._P1, _fun_P1, self._projectors_1d)

    def assemble_S1(self):
        _fun_S1  = []
        for m in range(3):
            _fun_S1  += [[]]
            for n in range(3):
                _fun_S1[ -1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.p3_eq(x1, x2, x3) * self._Ginv(x1, x2, x3)[m, n]]

        self._S1  = MHD_operator(self._V1, self._V2, self._P2, _fun_S1, self._projectors_1d)

    def assemble_S10(self):
        _fun_S10 = []
        for m in range(3):
            _fun_S10  += [[]]
            for n in range(3):
                _fun_S10[-1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.p3_eq(x1, x2, x3) if m==n else 0.]

        self._S10 = MHD_operator(self._V1, self._V1, self._P1, _fun_S10, self._projectors_1d)

    def assemble_T1(self):
        _fun_T1  = []
        for m in range(3):
            _fun_T1  += [[]]
            for n in range(3):
                _fun_T1[ -1] += [lambda x1, x2, x3, m=m, n=n : [self._EQ.b2_eq_1, self._EQ.b2_eq_2, self._EQ.b2_eq_3][m](x1, x2, x3) * self._Ginv(x1, x2, x3)[m, n]]

        self._T1  = MHD_operator(self._V1, self._V1, self._P1, _fun_T1,  self._projectors_1d)

    def assemble_X1(self):
        _fun_X1  = []
        for m in range(3):
            _fun_X1  += [[]]
            for n in range(3):
                _fun_X1[ -1] += [lambda x1, x2, x3, m=m, n=n : (self._F.jacobian_inv(x1, x2, x3).T)[m, n]]

        self._X1_comp1  = MHD_operator(self._V1, self._V0, self._P0, [_fun_X1[0]], self._projectors_1d)
        self._X1_comp2  = MHD_operator(self._V1, self._V0, self._P0, [_fun_X1[1]], self._projectors_1d)
        self._X1_comp3  = MHD_operator(self._V1, self._V0, self._P0, [_fun_X1[2]], self._projectors_1d)

    def assemble_Q2(self):
        _fun_Q2  = []
        for m in range(3):
            _fun_Q2  += [[]]
            for n in range(3):
                _fun_Q2[ -1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.r3_eq(x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3)) if m==n else 0.]

        self._Q2  = MHD_operator(self._V2, self._V2, self._P2, _fun_Q2, self._projectors_1d)

    def assemble_T2(self):
        _fun_T2  = []
        for m in range(3):
            _fun_T2  += [[]]
            for n in range(3):
                _fun_T2[ -1] += [lambda x1, x2, x3, m=m, n=n : [self._EQ.b2_eq_1, self._EQ.b2_eq_2, self._EQ.b2_eq_3][m](x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3))]
               
        self._T2  = MHD_operator(self._V2, self._V1, self._P1, _fun_T2, self._projectors_1d)

    def assemble_P2(self):
        _fun_P2  = []
        for m in range(3):
            _fun_P2  += [[]]
            for n in range(3):
                _fun_P2[ -1] += [lambda x1, x2, x3, m=m, n=n : self._Ginv(x1, x2, x3)[m, n] * [self._EQ.j2_eq_1, self._EQ.j2_eq_2, self._EQ.j2_eq_3][n](x1, x2, x3)]
               
        self._P2  = MHD_operator(self._V2, self._V2, self._P2, _fun_P2, self._projectors_1d)

    def assemble_S2(self):
        _fun_S2  = []
        for m in range(3):
            _fun_S2  += [[]]
            for n in range(3):
                _fun_S2[ -1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.p3_eq(x1, x2, x3) / np.sqrt(self._F.metric_det(x1, x2, x3)) if m==n else 0.]

        self._S2  = MHD_operator(self._V2, self._V2, self._P2, _fun_S2, self._projectors_1d)

    def assemble_X2(self):
        _fun_X2  = []
        for m in range(3):
            _fun_X2  += [[]]
            for n in range(3):
                _fun_X2[ -1] += [lambda x1, x2, x3, m=m, n=n : self._F.jacobian(x1, x2, x3)[m, n] / np.sqrt(self._F.metric_det(x1, x2, x3))]

        self._X2_comp1  = MHD_operator(self._V2, self._V0, self._P0, [_fun_X2[0]], self._projectors_1d)
        self._X2_comp2  = MHD_operator(self._V2, self._V0, self._P0, [_fun_X2[1]], self._projectors_1d)
        self._X2_comp3  = MHD_operator(self._V2, self._V0, self._P0, [_fun_X2[2]], self._projectors_1d)

    def assemble_Z20(self):
        _fun_Z20 = []
        for m in range(3):
            _fun_Z20  += [[]]
            for n in range(3):
                _fun_Z20[-1] += [lambda x1, x2, x3, m=m, n=n : self._F.metric(x1, x2, x3)[m, n]]

        self._Z20 = MHD_operator(self._V2, self._V1, self._P1, _fun_Z20,  self._projectors_1d)

    def assemble_S20(self):
        _fun_S20 = []
        for m in range(3):
            _fun_S20  += [[]]
            for n in range(3):
                _fun_S20[-1] += [lambda x1, x2, x3, m=m, n=n : self._EQ.p3_eq(x1, x2, x3) * self._F.metric(x1, x2, x3)[m, n] / np.sqrt(self._F.metric_det(x1, x2, x3))]

        self._S20 = MHD_operator(self._V2, self._V1, self._P1, _fun_S20,  self._projectors_1d)


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
        return self._X1.dot(x)

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
        return self._X2.dot(x)

    def Z20(self, x):
        return self._Z20.dot(x)

    def Y20(self, x):
        return self._Y20.dot(x)

    def S20(self, x):
        return self._S20.dot(x)



class MHD_operator:

    def __init__(self, V, W, pi_W, fun, projectors):
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

            projectors : 3-tuple of 1d projectors.
                From Struphy Projectors_global_1d class. They contain all the necessary information for Derham commuting projectors.
        '''

        from psydac.fem.tensor import TensorFemSpace 
        from psydac.fem.vector import ProductFemSpace
        from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
        from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

        from struphy.psydac_linear_operators.mhd_ops_kernels import assemble_dofs_for_weighted_basisfuns as assemble
        from struphy.psydac_linear_operators.prepare_projection import evaluate_fun_weights
        
        assert isinstance(V, TensorFemSpace) or isinstance(V, ProductFemSpace)
        assert isinstance(W, TensorFemSpace) or isinstance(W, ProductFemSpace)

        self._V = V
        self._W = W
        self._pi_W = pi_W
        self._fun = fun
        self._projectors = projectors

        # Retrieve solver
        self._solver = pi_W.solver

        # Space identifiers
        _name_V = V.symbolic_space.name
        _name_W = W.symbolic_space.name

        # Stencil vector spaces
        if _name_V in {'H1', 'L2'}:
            _Vspaces = [V.vector_space]
            _Vcomps  = [_name_V]
        else:
            _Vspaces = V.vector_space
            _Vcomps  = [_name_V + str(n) for n in range(1,4)]

        if _name_W in {'H1', 'L2'}:
            _Wspaces = [W.vector_space]
            _Wcomps  = [_name_W]
        else:
            _Wspaces = W.vector_space
            _Wcomps  = [_name_W + str(n) for n in range(1,4)]

        # Block matrix for dofs
        _blocks = []
        # Ouptut vector space (codomain), row of block
        for Wspace, Wcomp, fun_line in zip(_Wspaces, _Wcomps, fun):
            _blocks += [[]]
            # Input vector space (domain), column of block
            for Vspace, Vcomp, f in zip(_Vspaces, _Vcomps, fun_line):
                _ptsG, _wtsG, _spans, _bases = self.prepare_projection_of_basis(Vcomp, Wcomp)

                # Initiate cell of block matrix
                _dofs = StencilMatrix(Vspace, Wspace)

                _starts_in = _dofs.domain.starts
                _ends_in   = _dofs.domain.ends
                _pads_in   = _dofs.domain.pads
                _starts_out = _dofs.codomain.starts
                _ends_out   = _dofs.codomain.ends
                _pads_out   = _dofs.codomain.pads

                _fun_w = evaluate_fun_weights(_ptsG, _wtsG, f)

                # Call the kernel
                assemble(_dofs._data, _starts_in, _ends_in, _pads_in, _starts_out, _ends_out, _pads_out, _fun_w, _spans[0], _spans[1], _spans[2], _bases[0], _bases[1], _bases[2])

                _blocks[-1] += [_dofs]

        _len = sum([len(li) for li in _blocks])

        if _len > 1:
            self._dofs_mat = BlockMatrix(V.vector_space, W.vector_space, _blocks)
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
        
        return self._solver.solve(rhs), rhs


    def prepare_projection_of_basis(self, Vcomp, Wcomp):
        '''Obtain projection point sets, weights, knot span indices 
        and basis functions evaluated at projection point sets of a given vector space.

        Parameters
        ----------
            Vcomp : str
                Identifier of the input space component ('H1', 'Hcurl1', 'Hcurl2', 'Hcurl3', 'Hdiv1', 'Hdiv2', 'Hdiv3', 'L2).

            Wcomp : str
                Identifier of the output space component ('H1', 'Hcurl1', 'Hcurl2', 'Hcurl3', 'Hdiv1', 'Hdiv2', 'Hdiv3', 'L2).
                
        Returns
        -------
            ptsG : 3-tuple of 2d float arrays
                Quadrature points (or Greville points for interpolation) in each dimension in format (element, quadrature point).
        
            wtsG : 3-tuple of 2d float arrays
                Quadrature weights (or ones for interpolation) in each dimension in format (element, quadrature point).
                
            spans : 3-tuple of 2d int arrays
                Knot span indices in each direction in format (element, quadrature point).
                
            bases : 3-tuple of 3d float arrays
                Values of p + 1 non-zero eta basis functions at quadrature points in format (element, quadrature point, basis function).'''

        import numpy as np

        ptsG   = [None, None, None]
        wtsG   = [None, None, None]
        spans = [None, None, None]
        bases = [None, None, None]

        #######################
        # Output vector space #
        #######################
        if Wcomp == 'H1':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].x_int[:, None]
            ptsG[1] = self._projectors[1].x_int[:, None]
            ptsG[2] = self._projectors[2].x_int[:, None]
            # Weigths or set them to one
            wtsG[0] = np.ones_like(ptsG[0])
            wtsG[1] = np.ones_like(ptsG[1])
            wtsG[2] = np.ones_like(ptsG[2])

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_x_int_N
                bases[0] = self._projectors[0].basis_x_int_N
            else:
                spans[0] = self._projectors[0].span_x_int_D
                bases[0] = self._projectors[0].basis_x_int_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_x_int_N
                bases[1] = self._projectors[1].basis_x_int_N
            else:
                spans[1] = self._projectors[1].span_x_int_D
                bases[1] = self._projectors[1].basis_x_int_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_x_int_N
                bases[2] = self._projectors[2].basis_x_int_N
            else:
                spans[2] = self._projectors[2].span_x_int_D
                bases[2] = self._projectors[2].basis_x_int_D

        elif Wcomp == 'Hcurl1':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].ptsG
            ptsG[1] = self._projectors[1].x_int[:, None]
            ptsG[2] = self._projectors[2].x_int[:, None]
            # Retrieve weights or set them to one
            wtsG[0] = self._projectors[0].wtsG
            wtsG[1] = np.ones_like(ptsG[1])
            wtsG[2] = np.ones_like(ptsG[2])

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_ptsG_N
                bases[0] = self._projectors[0].basis_ptsG_N
            else:
                spans[0] = self._projectors[0].span_ptsG_D
                bases[0] = self._projectors[0].basis_ptsG_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_x_int_N
                bases[1] = self._projectors[1].basis_x_int_N
            else:
                spans[1] = self._projectors[1].span_x_int_D
                bases[1] = self._projectors[1].basis_x_int_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_x_int_N
                bases[2] = self._projectors[2].basis_x_int_N
            else:
                spans[2] = self._projectors[2].span_x_int_D
                bases[2] = self._projectors[2].basis_x_int_D

        elif Wcomp == 'Hcurl2':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].x_int[:, None]
            ptsG[1] = self._projectors[1].ptsG
            ptsG[2] = self._projectors[2].x_int[:, None]
            # Retrieve weights or set them to one
            wtsG[0] = np.ones_like(ptsG[0])
            wtsG[1] = self._projectors[1].wtsG
            wtsG[2] = np.ones_like(ptsG[2])

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_x_int_N
                bases[0] = self._projectors[0].basis_x_int_N
            else:
                spans[0] = self._projectors[0].span_x_int_D
                bases[0] = self._projectors[0].basis_x_int_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_ptsG_N
                bases[1] = self._projectors[1].basis_ptsG_N
            else:
                spans[1] = self._projectors[1].span_ptsG_D
                bases[1] = self._projectors[1].basis_ptsG_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_x_int_N
                bases[2] = self._projectors[2].basis_x_int_N
            else:
                spans[2] = self._projectors[2].span_x_int_D
                bases[2] = self._projectors[2].basis_x_int_D

        elif Wcomp == 'Hcurl3':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].x_int[:, None]
            ptsG[1] = self._projectors[1].x_int[:, None]
            ptsG[2] = self._projectors[2].ptsG
            # Retrieve weights or set them to one
            wtsG[0] = np.ones_like(ptsG[0])
            wtsG[1] = np.ones_like(ptsG[1])
            wtsG[2] = self._projectors[2].wtsG

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_x_int_N
                bases[0] = self._projectors[0].basis_x_int_N
            else:
                spans[0] = self._projectors[0].span_x_int_D
                bases[0] = self._projectors[0].basis_x_int_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_x_int_N
                bases[1] = self._projectors[1].basis_x_int_N
            else:
                spans[1] = self._projectors[1].span_x_int_D
                bases[1] = self._projectors[1].basis_x_int_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_ptsG_N
                bases[2] = self._projectors[2].basis_ptsG_N
            else:
                spans[2] = self._projectors[2].span_ptsG_D
                bases[2] = self._projectors[2].basis_ptsG_D

        elif Wcomp == 'Hdiv1':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].x_int[:, None]
            ptsG[1] = self._projectors[1].ptsG
            ptsG[2] = self._projectors[2].ptsG
            # Retrieve weights or set them to one
            wtsG[0] = np.ones_like(ptsG[0])
            wtsG[1] = self._projectors[1].wtsG
            wtsG[2] = self._projectors[2].wtsG

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_x_int_N
                bases[0] = self._projectors[0].basis_x_int_N
            else:
                spans[0] = self._projectors[0].span_x_int_D
                bases[0] = self._projectors[0].basis_x_int_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_ptsG_N
                bases[1] = self._projectors[1].basis_ptsG_N
            else:
                spans[1] = self._projectors[1].span_ptsG_D
                bases[1] = self._projectors[1].basis_ptsG_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_ptsG_N
                bases[2] = self._projectors[2].basis_ptsG_N
            else:
                spans[2] = self._projectors[2].span_ptsG_D
                bases[2] = self._projectors[2].basis_ptsG_D

        elif Wcomp == 'Hdiv2':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].ptsG
            ptsG[1] = self._projectors[1].x_int[:, None]
            ptsG[2] = self._projectors[2].ptsG
            # Retrieve weights or set them to one
            wtsG[0] = self._projectors[0].wtsG
            wtsG[1] = np.ones_like(ptsG[1])
            wtsG[2] = self._projectors[2].wtsG

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_ptsG_N
                bases[0] = self._projectors[0].basis_ptsG_N
            else:
                spans[0] = self._projectors[0].span_ptsG_D
                bases[0] = self._projectors[0].basis_ptsG_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_x_int_N
                bases[1] = self._projectors[1].basis_x_int_N
            else:
                spans[1] = self._projectors[1].span_x_int_D
                bases[1] = self._projectors[1].basis_x_int_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_ptsG_N
                bases[2] = self._projectors[2].basis_ptsG_N
            else:
                spans[2] = self._projectors[2].span_ptsG_D
                bases[2] = self._projectors[2].basis_ptsG_D

        elif Wcomp == 'Hdiv3':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].ptsG
            ptsG[1] = self._projectors[1].ptsG
            ptsG[2] = self._projectors[2].x_int[:, None]
            # Retrieve weights or set them to one
            wtsG[0] = self._projectors[0].wtsG
            wtsG[1] = self._projectors[1].wtsG
            wtsG[2] = np.ones_like(ptsG[2])

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_ptsG_N
                bases[0] = self._projectors[0].basis_ptsG_N
            else:
                spans[0] = self._projectors[0].span_ptsG_D
                bases[0] = self._projectors[0].basis_ptsG_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_ptsG_N
                bases[1] = self._projectors[1].basis_ptsG_N
            else:
                spans[1] = self._projectors[1].span_ptsG_D
                bases[1] = self._projectors[1].basis_ptsG_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_x_int_N
                bases[2] = self._projectors[2].basis_x_int_N
            else:
                spans[2] = self._projectors[2].span_x_int_D
                bases[2] = self._projectors[2].basis_x_int_D

        elif Wcomp == 'L2':

            # Retrieve point sets (Quadrature between Greville, or Greville) 
            ptsG[0] = self._projectors[0].ptsG
            ptsG[1] = self._projectors[1].ptsG
            ptsG[2] = self._projectors[2].ptsG
            # Retrieve weights or set them to one
            wtsG[0] = self._projectors[0].wtsG
            wtsG[1] = self._projectors[1].wtsG
            wtsG[2] = self._projectors[2].wtsG

            # Retrieve knot span inidices and V-basis functions evaluated at W-point sets
            # First direction
            if Vcomp in {'H1', 'Hcurl2', 'Hcurl3', 'Hdiv1'}:  
                spans[0] = self._projectors[0].span_ptsG_N
                bases[0] = self._projectors[0].basis_ptsG_N
            else:
                spans[0] = self._projectors[0].span_ptsG_D
                bases[0] = self._projectors[0].basis_ptsG_D

            # Second direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl3', 'Hdiv2'}: 
                spans[1] = self._projectors[1].span_ptsG_N
                bases[1] = self._projectors[1].basis_ptsG_N
            else:
                spans[1] = self._projectors[1].span_ptsG_D
                bases[1] = self._projectors[1].basis_ptsG_D

            # Third direction
            if Vcomp in {'H1', 'Hcurl1', 'Hcurl2', 'Hdiv3'}:
                spans[2] = self._projectors[2].span_ptsG_N
                bases[2] = self._projectors[2].basis_ptsG_N
            else:
                spans[2] = self._projectors[2].span_ptsG_D
                bases[2] = self._projectors[2].basis_ptsG_D

        ptsG   = tuple(ptsG)
        wtsG   = tuple(wtsG)
        spans = tuple(spans)
        bases = tuple(bases)

        return ptsG, wtsG, spans, bases
  