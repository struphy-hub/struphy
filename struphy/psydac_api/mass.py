import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api import mass_kernels
from struphy.psydac_api.linear_operators import LinOpWithTransp, CompositeLinearOperator, IdentityOperator, BoundaryOperator

from struphy.polar.basic import PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator


class WeightedMassOperators:
    """
    Class for assembling weighted mass matrices in 3d.
    
    Parameters
    ----------
    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : struphy.geometry.domains
        Mapping from logical unit cube to physical domain and corresponding metric coefficients.

    **weights
        General object providing access to callables that can serve as weight functions.
    """
    
    def __init__(self, derham, domain, **weights):
        
        self._derham = derham
        self._domain = domain
        
        # Wrapper functions for evaluating metric coefficients in right order (3x3 entries are last two axes!!)
        def DF(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

        def DFT(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)
            
        def DFinv(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

        def DFinvT(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)

        def G(e1, e2, e3):
            return domain.metric(e1, e2, e3, change_out_order=True, squeeze_out=False)
            
        def Ginv(e1, e2, e3):
            return domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False)
            
        def sqrt_g(e1, e2, e3):
            return abs(domain.jacobian_det(e1, e2, e3, squeeze_out=False))
        
        # Cross product matrices and evaluation of cross products
        cross_mask = [[ 1, -1,  1], 
                      [ 1,  1, -1], 
                      [-1,  1,  1]]
        
        def eval_cross(e1, e2, e3, fun_list): 
            
            cross = np.array([[cross_mask[m][n] * fun(e1, e2, e3) for n, fun in enumerate(row)] for m, row in enumerate(fun_list)])
            
            return np.transpose(cross, axes=(2, 3, 4, 0, 1))
        
        
        if 'eq_mhd' in weights:
            j2_cross = [[lambda e1, e2, e3 : 0*e1, weights['eq_mhd'].j2_3, weights['eq_mhd'].j2_2],
                        [weights['eq_mhd'].j2_3, lambda e1, e2, e3 : 0*e2, weights['eq_mhd'].j2_1],
                        [weights['eq_mhd'].j2_2, weights['eq_mhd'].j2_1, lambda e1, e2, e3 : 0*e3]]
       
        # scalar functions
        fun_M0 = [[lambda e1, e2, e3 :   sqrt_g(e1, e2, e3)]]
        fun_M3 = [[lambda e1, e2, e3 : 1/sqrt_g(e1, e2, e3)]]
        
        # matrix functions
        fun_M1 = []
        fun_M2 = []
        fun_Mv = []
        
        if 'eq_mhd' in weights:
            fun_M1n = []
            fun_M2n = []
            fun_Mvn = []
            fun_M1J = []
            fun_M2J = []
            fun_MvJ = []
        
        # assembly of weight functions
        for m in range(3):
            fun_M1 += [[]]
            fun_M2 += [[]]
            fun_Mv += [[]]
            
            if 'eq_mhd' in weights:
                fun_M1n += [[]]
                fun_M2n += [[]]
                fun_Mvn += [[]]
                fun_M1J += [[]]
                fun_M2J += [[]]
                fun_MvJ += [[]]
            
            for n in range(3):
                fun_M1[-1] += [lambda e1, e2, e3, m=m, n=n : Ginv(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)]
                fun_M2[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]/sqrt_g(e1, e2, e3)]
                fun_Mv[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)]
                
                if 'eq_mhd' in weights:
                    fun_M1n[-1] += [lambda e1, e2, e3, m=m, n=n : Ginv(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]
                    fun_M2n[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]/sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]
                    fun_Mvn[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]
                    fun_M1J[-1] += [lambda e1, e2, e3, m=m, n=n : (Ginv(e1, e2, e3) @ eval_cross(e1, e2, e3, j2_cross))[:, :, :, m, n]]
                    fun_M2J[-1] += [lambda e1, e2, e3, m=m, n=n : cross_mask[m][n]*j2_cross[m][n](e1, e2, e3)/sqrt_g(e1, e2, e3)]
                    fun_MvJ[-1] += [lambda e1, e2, e3, m=m, n=n : cross_mask[m][n]*j2_cross[m][n](e1, e2, e3)]
                    
                          
        self._fun_M0 = fun_M0
        self._fun_M3 = fun_M3
        
        self._fun_M1 = fun_M1
        self._fun_M2 = fun_M2
        self._fun_Mv = fun_Mv
        
        if 'eq_mhd' in weights:
            self._fun_M1n = fun_M1n
            self._fun_M2n = fun_M2n
            self._fun_Mvn = fun_Mvn
            self._fun_M1J = fun_M1J
            self._fun_M2J = fun_M2J
            self._fun_MvJ = fun_MvJ
     
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
    
    @property
    def derham(self):
        return self._derham
    
    @property
    def domain(self):
        return self._domain
        
    #######################################################################
    # Mass matrices related to L2-scalar products in all 3d derham spaces #
    #######################################################################
    @property
    def M0(self):
        """ Mass matrix M0_(ijk lmn) = integral( Lambda^0_(ijk) * Lambda^0_(lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_M0'):
            self._M0 = WeightedMassOperator(self.derham.Vh_fem['0'], self.derham.Vh_fem['0'], 
                                            V_extraction_op=self.derham.E['0'], W_extraction_op=self.derham.E['0'],
                                            V_boundary_op=self.derham.B['0'], W_boundary_op=self.derham.B['0'], 
                                            weights=self._fun_M0, transposed=False)
            self._M0.assemble()
            self._M0.matrix.exchange_assembly_data()
        
        return self._M0
    
    @property
    def M1(self):
        """ Mass matrix M1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * G_inv_ab * Lambda^1_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_M1'):
            self._M1 = WeightedMassOperator(self.derham.Vh_fem['1'], self.derham.Vh_fem['1'], 
                                            V_extraction_op=self.derham.E['1'], W_extraction_op=self.derham.E['1'],
                                            V_boundary_op=self.derham.B['1'], W_boundary_op=self.derham.B['1'], 
                                            weights=self._fun_M1, transposed=False)
            self._M1.assemble()
            self._M1.matrix.exchange_assembly_data()
        
        return self._M1
    
    @property
    def M2(self):
        """ Mass matrix M2_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M2'):
            self._M2 = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'], 
                                            V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                            V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'], 
                                            weights=self._fun_M2, transposed=False)
            self._M2.assemble()
            self._M2.matrix.exchange_assembly_data()
        
        return self._M2
    
    @property
    def M3(self):
        """ Mass matrix M3_(ijk lmn) = integral( Lambda^3_(ijk) * Lambda^3_(lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M3'):
            self._M3 = WeightedMassOperator(self.derham.Vh_fem['3'], self.derham.Vh_fem['3'], 
                                            V_extraction_op=self.derham.E['3'], W_extraction_op=self.derham.E['3'],
                                            V_boundary_op=self.derham.B['3'], W_boundary_op=self.derham.B['3'], 
                                            weights=self._fun_M3, transposed=False)
            self._M3.assemble()
            self._M3.matrix.exchange_assembly_data()
        
        return self._M3
    
    @property
    def Mv(self):
        """ Mass matrix Mv_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * G_ab * Lambda^v_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_Mv'):
            self._Mv = WeightedMassOperator(self.derham.Vh_fem['v'], self.derham.Vh_fem['v'], 
                                            V_extraction_op=self.derham.E['v'], W_extraction_op=self.derham.E['v'],
                                            V_boundary_op=self.derham.B['v'], W_boundary_op=self.derham.B['v'],
                                            weights=self._fun_Mv, transposed=False)
            self._Mv.assemble()
            self._Mv.matrix.exchange_assembly_data()
        
        return self._Mv
    
    ########################################################################
    # Mass matrices in several spaces weighted with MHD equilibrium fields #
    ########################################################################
    @property
    def M1n(self):
        """ Mass matrix Mn1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^1_(b,lmn) * sqrt(g) * n^0_eq * G_inv_ab ).
        """
        
        assert hasattr(self, '_fun_M1n'), 'MHD equilibrium has not been set!'
        
        if not hasattr(self, '_M1n'):
            self._M1n = WeightedMassOperator(self.derham.Vh_fem['1'], self.derham.Vh_fem['1'],
                                             V_extraction_op=self.derham.E['1'], W_extraction_op=self.derham.E['1'],
                                             V_boundary_op=self.derham.B['1'], W_boundary_op=self.derham.B['1'], 
                                             weights=self._fun_M1n, transposed=False)
            self._M1n.assemble()
            self._M1n.matrix.exchange_assembly_data()
        
        return self._M1n
    
    @property
    def M2n(self):
        """ Mass matrix M2n_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_M2n'):
            self._M2n = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'], 
                                             weights=self._fun_M2n, transposed=False)
            self._M2n.assemble()
            self._M2n.matrix.exchange_assembly_data()
        
        return self._M2n
    
    @property
    def Mvn(self):
        """ Mass matrix Mvn_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^v_(b,lmn) * sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_Mvn'):
            self._Mvn = WeightedMassOperator(self.derham.Vh_fem['v'], self.derham.Vh_fem['v'],
                                             V_extraction_op=self.derham.E['v'], W_extraction_op=self.derham.E['v'],
                                             V_boundary_op=self.derham.B['v'], W_boundary_op=self.derham.B['v'],
                                             weights=self._fun_Mvn, transposed=False)
            self._Mvn.assemble()
            self._Mvn.matrix.exchange_assembly_data()
        
        return self._Mvn
    
    @property
    def M1J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c * G_inv_ab ).
        """
        
        if not hasattr(self, '_M1J'):
            self._M1J = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['1'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['1'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['1'], 
                                             weights=self._fun_M1J, transposed=False)
            self._M1J.assemble()
            self._M1J.matrix.exchange_assembly_data()
        
        return self._M1J
    
    @property
    def M2J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * epsilon_(acb) * J^2_eq_c).
        """
        
        if not hasattr(self, '_M2J'):
            self._M2J = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'],
                                             weights=self._fun_M2J, transposed=False)
            self._M2J.assemble()
            self._M2J.matrix.exchange_assembly_data()
        
        return self._M2J
    
    @property
    def MvJ(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c ).
        """
        
        if not hasattr(self, '_MvJ'):
            self._MvJ = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['v'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['v'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['v'],
                                             weights=self._fun_MvJ, transposed=False)
            self._MvJ.assemble()
            self._MvJ.matrix.exchange_assembly_data()
        
        return self._MvJ

    
class WeightedMassOperator( LinOpWithTransp ):
    """
    Weighted mass matrix of the form B * E * M * E^T * B^T, with E and B being basis extraction and boundary operators, respectively.
    
    Parameters
    ----------
    V : TensorFemSpace | ProductFemSpace
        Tensor product spline space from psydac.fem.tensor (domain, input space).

    W : TensorFemSpace | ProductFemSpace
        Tensor product spline space from psydac.fem.tensor (codomain, output space).

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    W_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of W.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    W_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    weights : list | NoneType
        Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain. If None, identity weights are assumed.
        
    symmetry : str
        Symmetry of block matrices ('symm', 'asym' or 'diag').

    transposed : bool
        Whether to assemble the transposed operator.
    """
    
    def __init__(self, V, W, V_extraction_op=None, W_extraction_op=None, V_boundary_op=None, W_boundary_op=None, weights=None, symmetry=None, transposed=False):
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
        
        assert isinstance(V, (TensorFemSpace, ProductFemSpace))
        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        
        self._V = V
        self._W = W
        
        # set basis extraction operators
        if V_extraction_op is not None:
            assert isinstance(V_extraction_op, (PolarExtractionOperator, IdentityOperator))
            assert V_extraction_op.domain == V.vector_space
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.vector_space)
            
        if W_extraction_op is not None:
            assert isinstance(W_extraction_op, (PolarExtractionOperator, IdentityOperator))
            assert W_extraction_op.domain == W.vector_space
            self._W_extraction_op = W_extraction_op
        else:
            self._W_extraction_op = IdentityOperator(W.vector_space)
        
        # set boundary operators
        if V_boundary_op is not None:
            assert isinstance(V_boundary_op, (BoundaryOperator, IdentityOperator))
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(self._V_extraction_op.codomain)
            
        if W_boundary_op is not None:
            assert isinstance(W_boundary_op, (BoundaryOperator, IdentityOperator))
            self._W_boundary_op = W_boundary_op
        else:
            self._W_boundary_op = IdentityOperator(self._W_extraction_op.codomain)
        
        self._symmetry = symmetry
        self._transposed = transposed
        
        self._dtype = V.vector_space.dtype
        
        # set domain and codomain symbolic names
        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            if V.ldim == 3 or V.ldim == 2:
                V_name = 'H1vec'
            elif V.ldim == 1:
                if V.spaces[0].basis == 'B':
                    V_name = 'H1'
                else:
                    V_name = 'L2'

        if hasattr(W.symbolic_space, 'name'):
            W_name = W.symbolic_space.name
        else:
            if W.ldim == 3 or W.ldim == 2:
                W_name = 'H1vec'
            elif W.ldim == 1:
                if W.spaces[0].basis == 'B':
                    W_name = 'H1'
                else:
                    W_name = 'L2'
        
        if transposed:
            self._domain_femspace = W
            self._domain_symbolic_name = W_name
            
            self._codomain_femspace = V
            self._codomain_symbolic_name = V_name
        else:
            self._domain_femspace = V
            self._domain_symbolic_name = V_name
            
            self._codomain_femspace = W
            self._codomain_symbolic_name = W_name

        # ====== initialize Stencil-/BlockMatrix ====
        
        # collect TensorFemSpaces for each component in tuple
        if isinstance(V, TensorFemSpace):
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
            
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
    
        # initialize blocks according to given symmetry
        if symmetry is not None:
            
            assert symmetry in {'symm', 'asym', 'diag', 'upper_tri', 'lower_tri'}
            assert V_name in {'Hcurl', 'Hdiv', 'H1vec'}
            assert V_name == W_name, 'only square matrices (V=W) allowed!'
            
            if   symmetry == 'symm':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL) for Vs in V.spaces] for Ws in W.spaces]
            elif symmetry == 'asym':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL) if i != j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'diag':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL) if i == j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'upper_tri':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL) if i <= j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'lower_tri':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL) if i <= j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            
            self._mat = BlockMatrix(V.vector_space, W.vector_space, blocks=blocks)
            
            # set default identity weights if not given
            self._weights = []
            for a, row in enumerate(blocks):
                self._weights += [[]]
                for b, col in enumerate(row):
                    if col is None:
                        self._weights[-1] += [None]
                    else:
                        if weights is None:
                            if symmetry == 'asym' and (a, b) in [(1, 0), (2, 0), (2, 1)]:
                                self._weights[-1] += [lambda *etas : -np.ones(etas[0].shape, dtype=float)]
                            else:
                                self._weights[-1] += [lambda *etas : np.ones(etas[0].shape, dtype=float)]
                        else:
                            self._weights[-1] += [weights[a][b]]
            
                        
        # initialize all blocks or according to given weights
        else:
            
            blocks = []
            self._weights = []
        
            # loop over codomain spaces (rows)
            for a, wspace in enumerate(Wspaces):
                blocks += [[]]
                self._weights += [[]]

                # loop over domain spaces (columns)
                for b, vspace in enumerate(Vspaces):
                    
                    # set default identity weights if not given
                    if weights is None:
                        blocks[-1] += [StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)]
                        self._weights[-1] += [lambda *etas : np.ones(etas[0].shape, dtype=float)]
                        
                    else:
                        
                        if weights[a][b] is None:
                            blocks[-1] += [None]
                            self._weights[-1] += [None]
                            
                        else:
                        
                            # test weight function at quadrature points to identify zero blocks
                            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]

                            if callable(weights[a][b]):
                                PTS = np.meshgrid(*pts, indexing='ij')
                                mat_w = weights[a][b](*PTS).copy()
                            elif isinstance(weights[a][b], np.ndarray):
                                mat_w = weights[a][b]

                            assert mat_w.shape == tuple([pt.size for pt in pts])

                            if np.any(np.abs(mat_w) > 1e-14):
                                blocks[-1] += [StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)]
                                self._weights[-1] += [weights[a][b]]
                            else:
                                blocks[-1] += [None]
                                self._weights[-1] += [None]

            if len(blocks) == len(blocks[0]) == 1:
                self._mat = blocks[0][0]
            else:
                self._mat = BlockMatrix(V.vector_space, W.vector_space, blocks=blocks)
        
        # transpose of matrix and weights
        if transposed:
            self._mat = self._mat.transpose()
            
            n_rows = len(self._weights)
            n_cols = len(self._weights[0])
            
            tmp_weights = []
            
            for m in range(n_cols):
                tmp_weights += [[]]
                for n in range(n_rows):
                    if self._weights[n][m] is not None:
                        tmp_weights[-1] += [self._weights[n][m]]
                    else:
                        tmp_weights[-1] += [None]
                        
            self._weights = tmp_weights
                    
        # ===============================================
        
        # some shortcuts
        BW = self._W_boundary_op
        BV = self._V_boundary_op
        
        EW = self._W_extraction_op
        EV = self._V_extraction_op

        # build composite linear operators BW * EW * M * EV^T * BV^T, resp. IDV * EV * M^T * EW^T * IDW^T
        if transposed:
            self._M  = CompositeLinearOperator(IdentityOperator(EV.codomain), EV, self._mat, EW.transpose(), IdentityOperator(EW.codomain).transpose())
            self._M0 = CompositeLinearOperator(BV, EV, self._mat, EW.transpose(), BW.transpose())
        else:
            self._M  = CompositeLinearOperator(IdentityOperator(EW.codomain), EW, self._mat, EV.transpose(), IdentityOperator(EV.codomain).transpose())
            self._M0 = CompositeLinearOperator(BW, EW, self._mat, EV.transpose(), BV.transpose())
        
        # set domain and codomain
        self._domain = self._M.domain
        self._codomain = self._M.codomain
        
        # load assembly kernel
        self._assembly_kernel = getattr(mass_kernels, 'kernel_' + str(self._V.ldim) + 'd_mat')
    
    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain
    
    @property
    def domain_femspace(self):
        return self._domain_femspace

    @property
    def codomain_femspace(self):
        return self._codomain_femspace

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def M(self):
        return self._M
    
    @property
    def M0(self):
        return self._M0
    
    @property
    def matrix(self):
        return self._mat
    
    @property
    def weights(self):
        return self._weights
    
    def dot(self, v, out=None, apply_bc=True):
        """
        Applies the weighted mass operator to the FE coefficients v.

        Parameters
        ----------
        v : StencilVector | BlockVector | PolarVector
            Input FE coefficients the mass operator is applied to.

        apply_bc : bool
            Whether to apply boundary operators to input/output.

        Returns
        -------
        out : StencilVector | BlockVector | PolarVector
            Output FE coefficients.
        """

        assert v.space == self.domain
        
        # newly created output vector
        if out is None:
            if apply_bc:
                out = self._M0.dot(v)
            else:
                out = self._M.dot(v)
        
        # in-place dot-product (result is written to out)
        else:
            
            assert isinstance(out, (StencilVector, BlockVector, PolarVector))
            
            if apply_bc:
                tmp = self._M0.dot(v)
            else:
                tmp = self._M.dot(v)
            
            if isinstance(tmp, PolarVector):
                out.set_vector(tmp)
            elif isinstance(tmp, StencilVector):
                out[:] = tmp[:]
            elif isinstance(tmp, BlockVector):
                out[0][:] = tmp[0][:]
                out[1][:] = tmp[1][:]
                out[2][:] = tmp[2][:]
        
        assert out.space == self.codomain
        
        return out
    
    def transpose(self):
        """
        Returns the transposed operator.
        """
        
        # bring weights back in "right" (not transposed order)
        if self._transposed:
            n_rows = len(self._weights)
            n_cols = len(self._weights[0])

            weights = []

            for m in range(n_cols):
                weights += [[]]
                for n in range(n_rows):
                    if self._weights[n][m] is not None:
                        weights[-1] += [self._weights[n][m]]
                    else:
                        weights[-1] += [None]
        else:
            weights = self._weights
            
        
        M = WeightedMassOperator(self._V, self._W, 
                                 self._V_extraction_op, self._W_extraction_op, 
                                 self._V_boundary_op, self._W_boundary_op, 
                                 weights, self._symmetry, not self._transposed)
        
        M.assemble(verbose=False)
        M._mat.exchange_assembly_data()
        
        return M
    
    def assemble(self, weights=None, verbose=True):
        """
        Assembles a weighted mass matrix (StencilMatrix/BlockMatrix) corresponding to given domain/codomain spline spaces.
        
        General form (in 3d) is mat_(ijk,lmn) = integral[ Lambda_ijk * weight * Lambda_lmn ],
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.
        
        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.
        
        Parameters
        ----------
        weights : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain. If weights=None, the weight is taken from the given weights in the instanziation of the object, else it will ne overriden.
            
        verbose : bool
            Whether to do some printing.
        """
        
        # identify rank for printing
        if self._domain_symbolic_name in {'H1', 'L2'}:
            if self._V.vector_space.cart.comm is not None:
                rank = self._V.vector_space.cart.comm.Get_rank()
            else:
                rank = 0
        else:
            if self._V.vector_space[0].cart.comm is not None:
                rank = self._V.vector_space[0].cart.comm.Get_rank()
            else:
                rank = 0
        
        if rank == 0 and verbose:
            print(f'Assembling matrix of WeightedMassOperator with V={self._domain_symbolic_name}, W={self._codomain_symbolic_name}.')
        
        # collect domain/codomain TensorFemSpaces for each component in tuple
        if self._transposed:
            if isinstance(self._W, TensorFemSpace):
                domain_spaces = (self._W,)
            else:
                domain_spaces = self._W.spaces

            if isinstance(self._V, TensorFemSpace):
                codomain_spaces = (self._V,)
            else:
                codomain_spaces = self._V.spaces
        else:
            if isinstance(self._V, TensorFemSpace):
                domain_spaces = (self._V,)
            else:
                domain_spaces = self._V.spaces

            if isinstance(self._W, TensorFemSpace):
                codomain_spaces = (self._W,)
            else:
                codomain_spaces = self._W.spaces
        
        # override weights
        if weights is not None:
            self._weights = weights
    
        # loop over codomain spaces (rows)
        for a, codomain_space in enumerate(codomain_spaces):
            
            # knot span indices of elements of local domain
            codomain_spans = [quad_grid.spans for quad_grid in codomain_space.quad_grids]

            # global start spline index on process
            codomain_starts = [int(start) for start in codomain_space.vector_space.starts]

            # pads (ghost regions)
            codomain_pads = codomain_space.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten() for quad_grid in codomain_space.quad_grids]
            wts = [quad_grid.weights          for quad_grid in codomain_space.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            codomain_basis = [quad_grid.basis for quad_grid in codomain_space.quad_grids]

            # loop over domain spaces (columns)
            for b, domain_space in enumerate(domain_spaces):
                
                if self._weights[a][b] is not None:

                    if callable(self._weights[a][b]):
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = self._weights[a][b](*PTS).copy()
                    elif isinstance(self._weights[a][b], np.ndarray):
                        mat_w = self._weights[a][b]

                # None weight blocks are identified as zeros
                else:
                    mat_w = np.zeros([pt.size for pt in pts], dtype=float)

                assert mat_w.shape == tuple([pt.size for pt in pts])

                # evaluated basis functions at quadrature points of domain space
                domain_basis = [quad_grid.basis for quad_grid in domain_space.quad_grids]
                
                # assemble matrix (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)                
                if np.any(np.abs(mat_w) > 1e-14):
                    if isinstance(self._mat, StencilMatrix):
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *codomain_starts, *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat._data)
                    else:
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *codomain_starts, *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat[a, b]._data)
                        
        if rank == 0 and verbose:
            print('Done.')

    @staticmethod
    def assemble_vec(W, vec, weight=None):
        """
        Assembles (in 3d) vec_ijk = integral[ weight * Lambda_ijk ] into the Stencil-/BlockVector vec,
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.
        
        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.
        
        Parameters
        ----------
        W : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor.
            
        vec : StencilVector | BlockVector
            The vector to be filled.

        weight : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 1d list of shape corresponding to number of components.
        """

        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        assert isinstance(vec, (StencilVector, BlockVector))
        assert W.vector_space == vec.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # loag assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(W.ldim) + 'd_vec')

        # loop over components
        for a, wspace in enumerate(Wspaces):

            # knot span indices of elements of local domain
            spans = [quad_grid.spans for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            if weight is not None:
                if weight[a] is not None:

                    if callable(weight[a]):
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = weight[a](*PTS).copy()
                    elif isinstance(weight[a], np.ndarray):
                        mat_w = weight[a]

                else:
                    mat_w = np.zeros([pt.size for pt in pts], dtype=float)
            else:
                mat_w = np.ones([pt.size for pt in pts], dtype=float)

            assert mat_w.shape == tuple([pt.size for pt in pts])

            # assemble vector (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
            if np.any(np.abs(mat_w) > 1e-14):
                if isinstance(vec, StencilVector):
                    kernel(*spans, *wspace.degree, *starts, *pads, 
                           *wts, *basis, mat_w, vec._data)
                else:
                    kernel(*spans, *wspace.degree, *starts, *pads, 
                           *wts, *basis, mat_w, vec[a]._data)
                    
    @staticmethod
    def eval_quad(W, coeffs):
        """
        Evaluates a given FEM field defined by its coefficients at the L2 quadrature points.
        
        Parameters
        ----------
        W : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor.
            
        coeffs : StencilVector | BlockVector
            The coefficient vector corresponding to the FEM field. Ghost regions must be up-to-date!
        """

        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        assert isinstance(coeffs, (StencilVector, BlockVector))
        assert W.vector_space == coeffs.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # loag assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(W.ldim) + 'd_eval')
        
        blocks = []

        # loop over components
        for a, wspace in enumerate(Wspaces):

            # knot span indices of elements of local domain
            spans = [quad_grid.spans for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # perform evaluation by calling the appropriate kernel (1d, 2d or 3d)
            values = np.zeros([pt.size for pt in pts], dtype=float)
            
            if isinstance(coeffs, StencilVector):
                kernel(*spans, *wspace.degree, *starts, *pads, *basis, coeffs._data, values)
            else:
                kernel(*spans, *wspace.degree, *starts, *pads, *basis, coeffs[a]._data, values)
                
            blocks += [values]
                
        if len(blocks) == 1:
            return blocks[0]
        else:
            return blocks
    