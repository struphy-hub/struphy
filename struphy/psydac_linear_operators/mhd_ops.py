class MHD_ops:

    def __init__(self, DERHAM, nq_pr, EQ_MHD_L, F, projectors_1d):
        '''Assembles required MHD projection operators.
        
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

        import numpy as np

        # Missing in Psydac: inverse metric tensor
        _Ginv = lambda x1, x2, x3 : np.matmul(F.jacobian_inv(x1, x2, x3), F.jacobian_inv(x1, x2, x3).T)

        # Psydac spline spaces
        _V0 = DERHAM.V0
        _V1 = DERHAM.V1
        _V2 = DERHAM.V2
        _V3 = DERHAM.V3

        # Psydac projectors
        _P0, _P1, _P2, _P3  = DERHAM.projectors(nquads=nq_pr)

        # funs
        _fun_K1  = lambda x1, x2, x3 : EQ_MHD_L.p3_eq(x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))
        _fun_K10 = EQ_MHD_L.p0_eq

        # MHD operators
        self._K1  = MHD_operator(_V3, _V3, _P3, [[_fun_K1]],  projectors_1d)
        self._K10 = MHD_operator(_V0, _V0, _P0, [[_fun_K10]], projectors_1d)

    def K1(self, x):
        return self._K1.dot(x)

    def K10(self, x):
        return self._K10.dot(x)

           

class MHD_operator:

    def __init__(self, V, W, pi_W, fun, projectors):
        '''
        Class for MHD specific projection operators PI_ijk(fun*Lambda_mno).

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

        u.update_ghost_regions()
        self._dofs_mat.update_ghost_regions()

        rhs = self._dofs_mat.dot(u)

        rhs.update_ghost_regions()

        assert rhs.space == self._solver.space
        
        return self._solver.solve(rhs)


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
  