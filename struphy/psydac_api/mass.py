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
            All things mapping.
            
        **weights
            A general object providing access to callables that serve as weight functions (will be called with weights['keyword'].fun).
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
                                            weight=self._fun_M0, transposed=False)
        
        return self._M0
    
    @property
    def M1(self):
        """ Mass matrix M1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * G_inv_ab * Lambda^1_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_M1'):
            self._M1 = WeightedMassOperator(self.derham.Vh_fem['1'], self.derham.Vh_fem['1'], 
                                            V_extraction_op=self.derham.E['1'], W_extraction_op=self.derham.E['1'],
                                            V_boundary_op=self.derham.B['1'], W_boundary_op=self.derham.B['1'], 
                                            weight=self._fun_M1, transposed=False)
        
        return self._M1
    
    @property
    def M2(self):
        """ Mass matrix M2_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M2'):
            self._M2 = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'], 
                                            V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                            V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'], 
                                            weight=self._fun_M2, transposed=False)
        
        return self._M2
    
    @property
    def M3(self):
        """ Mass matrix M3_(ijk lmn) = integral( Lambda^3_(ijk) * Lambda^3_(lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M3'):
            self._M3 = WeightedMassOperator(self.derham.Vh_fem['3'], self.derham.Vh_fem['3'], 
                                            V_extraction_op=self.derham.E['3'], W_extraction_op=self.derham.E['3'],
                                            V_boundary_op=self.derham.B['3'], W_boundary_op=self.derham.B['3'], 
                                            weight=self._fun_M3, transposed=False)
        
        return self._M3
    
    @property
    def Mv(self):
        """ Mass matrix Mv_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * G_ab * Lambda^v_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_Mv'):
            self._Mv = WeightedMassOperator(self.derham.Vh_fem['v'], self.derham.Vh_fem['v'], 
                                            V_extraction_op=self.derham.E['v'], W_extraction_op=self.derham.E['v'],
                                            V_boundary_op=self.derham.B['v'], W_boundary_op=self.derham.B['v'],
                                            weight=self._fun_Mv, transposed=False)
        
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
                                             weight=self._fun_M1n, transposed=False)
        
        return self._M1n
    
    @property
    def M2n(self):
        """ Mass matrix M2n_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_M2n'):
            self._M2n = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'], 
                                             weight=self._fun_M2n, transposed=False)
        
        return self._M2n
    
    @property
    def Mvn(self):
        """ Mass matrix Mvn_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^v_(b,lmn) * sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_Mvn'):
            self._Mvn = WeightedMassOperator(self.derham.Vh_fem['v'], self.derham.Vh_fem['v'],
                                             V_extraction_op=self.derham.E['v'], W_extraction_op=self.derham.E['v'],
                                             V_boundary_op=self.derham.B['v'], W_boundary_op=self.derham.B['v'],
                                             weight=self._fun_Mvn, transposed=False)
        
        return self._Mvn
    
    @property
    def M1J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c * G_inv_ab ).
        """
        
        if not hasattr(self, '_M1J'):
            self._M1J = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['1'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['1'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['1'], 
                                             weight=self._fun_M1J, transposed=False)
        
        return self._M1J
    
    @property
    def M2J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * epsilon_(acb) * J^2_eq_c).
        """
        
        if not hasattr(self, '_M2J'):
            self._M2J = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['2'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['2'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['2'],
                                             weight=self._fun_M2J, transposed=False)
        
        return self._M2J
    
    @property
    def MvJ(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c ).
        """
        
        if not hasattr(self, '_MvJ'):
            self._MvJ = WeightedMassOperator(self.derham.Vh_fem['2'], self.derham.Vh_fem['v'],
                                             V_extraction_op=self.derham.E['2'], W_extraction_op=self.derham.E['v'],
                                             V_boundary_op=self.derham.B['2'], W_boundary_op=self.derham.B['v'],
                                             weight=self._fun_MvJ, transposed=False)
        
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

    weight : list
        Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.

    transposed : bool
        Whether to assemble the transposed operator.
    """
    
    def __init__(self, V, W, V_extraction_op=None, W_extraction_op=None, V_boundary_op=None, W_boundary_op=None, weight=None, transposed=False):
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
        
        assert isinstance(V, FemSpace)
        assert isinstance(W, FemSpace)
        
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
        
        self._weight = weight
        self._transposed = transposed
        
        self._dtype = V.vector_space.dtype
        
        # set domain and codomain symbolic names
        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            V_name = 'H1vec'

        if hasattr(W.symbolic_space, 'name'):
            W_name = W.symbolic_space.name
        else:
            W_name = 'H1vec'
        
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
        
        if V_name in {'H1', 'L2'}:
            comm = V.vector_space.cart.comm
        else:
            comm = V.vector_space[0].cart.comm

        # ====== assemble tensor-product mass matrix ====
        if comm.Get_rank() == 0:
            print(f'Assembling WeightedMassOperator with V={V_name}, W={W_name}.')
        
        # collect TensorFemSpaces for each component in tuple
        if isinstance(V, TensorFemSpace):
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
            
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
        
        blocks = []
    
        # loop over codomain spaces (rows)
        for a, wspace in enumerate(Wspaces):
            blocks += [[]]
            
            # loop over domain spaces (columns)
            for b, vspace in enumerate(Vspaces):

                # test weight function at quadrature points to identify zero blocks
                pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]
                
                if weight is not None:
                    if weight[a][b] is not None:

                        if callable(weight[a][b]):
                            PTS = np.meshgrid(*pts, indexing='ij')
                            mat_w = weight[a][b](*PTS).copy()
                        elif isinstance(weight[a][b], np.ndarray):
                            mat_w = weight[a][b]

                    else:
                        mat_w = np.zeros([pt.size for pt in pts], dtype=float)
                else:
                    mat_w = np.ones([pt.size for pt in pts], dtype=float)

                assert mat_w.shape == tuple([pt.size for pt in pts])
        
                if np.any(np.abs(mat_w) > 1e-14):
                    blocks[-1] += [StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)]
                else:
                    blocks[-1] += [None]

        if len(blocks) == len(blocks[0]) == 1:
            self._mat = blocks[0][0]
        else:
            self._mat = BlockMatrix(V.vector_space, W.vector_space, blocks=blocks)
        
        # fill matrix
        WeightedMassOperator.assemble_mat(V, W, self._mat, weight)
        
        if transposed:
            self._mat = self._mat.transpose()
            
        if comm.Get_rank() == 0:
            print('Done.')
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
    def transposed(self):
        return self._transposed
    
    @property
    def M(self):
        return self._M
    
    @property
    def M0(self):
        return self._M0
    
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
        return WeightedMassOperator(self._V, self._W, 
                                    self._V_extraction_op, self._W_extraction_op, 
                                    self._V_boundary_op, self._W_boundary_op, 
                                    self._weight, not self.transposed)
    
    @staticmethod
    def assemble_mat(V, W, mat, weight=None):
        """
        Assembles a weighted mass matrix (StencilMatrix/BlockMatrix) corresponding to given domain/codomain spline spaces.
        
        General form (in 3d) is mat_(ijk,lmn) = integral[ Lambda_ijk * weight * Lambda_lmn ],
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.
        
        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.
        
        Parameters
        ----------
        V : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor (domain, input space).

        W : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor (codomain, output space).
            
        mat : StencilMatrix | BlockMatrix
            The matrix to be filled.

        weight : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain.
        """
        
        assert isinstance(V, (TensorFemSpace, ProductFemSpace))
        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        assert isinstance(mat, (StencilMatrix, BlockMatrix))
        assert V.vector_space == mat.domain
        assert W.vector_space == mat.codomain
        
        # collect TensorFemSpaces for each component in tuple
        if isinstance(V, TensorFemSpace):
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
            
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
        
        # load assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(V.ldim) + 'd_mat')
    
        # loop over codomain spaces (rows)
        for a, wspace in enumerate(Wspaces):
            
            # periodicity: True (1) or False (0)
            periodic = [int(periodic) for periodic in wspace.periodic]

            # global element indices on process over which integration is performed
            el_loc_indices = [quad_grid.indices for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts_out = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads_out = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            nqs = [quad_grid.num_quad_pts     for quad_grid in wspace.quad_grids]
            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis_o = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # loop over domain spaces (columns)
            for b, vspace in enumerate(Vspaces):

                if weight is not None:
                    if weight[a][b] is not None:

                        if callable(weight[a][b]):
                            PTS = np.meshgrid(*pts, indexing='ij')
                            mat_w = weight[a][b](*PTS).copy()
                        elif isinstance(weight[a][b], np.ndarray):
                            mat_w = weight[a][b]

                    else:
                        mat_w = np.zeros([pt.size for pt in pts], dtype=float)
                else:
                    mat_w = np.ones([pt.size for pt in pts], dtype=float)

                assert mat_w.shape == tuple([pt.size for pt in pts])

                # evaluated basis functions at quadrature points of output space
                basis_i = [quad_grid.basis for quad_grid in vspace.quad_grids]

                # assemble matrix (if weight is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                if np.any(np.abs(mat_w) > 1e-14):
                    if isinstance(mat, StencilMatrix):
                        kernel(*el_loc_indices, *wspace.degree, *vspace.degree, *periodic, *starts_out, *pads_out, 
                               *nqs, *wts, *basis_o, *basis_i, mat_w, mat._data)
                    else:
                        kernel(*el_loc_indices, *wspace.degree, *vspace.degree, *periodic, *starts_out, *pads_out, 
                               *nqs, *wts, *basis_o, *basis_i, mat_w, mat[a, b]._data)
        
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

            # periodicity: True (1) or False (0)
            periodic = [int(periodic) for periodic in wspace.periodic]

            # global element indices on process over which integration is performed
            el_loc_indices = [quad_grid.indices for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            nqs = [quad_grid.num_quad_pts     for quad_grid in wspace.quad_grids]
            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # evaluation of weight function at quadrature points (optional)
            if weight is not None:
                if weight[a] is not None:
                    
                    if callable(weight[a]):
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = weight[a](*PTS).copy()
                    elif isinstance(weight[a], np.ndarray):
                        mat_w = weight[a]
                        
                else:
                    mat_w = np.ones([pt.size for pt in pts], dtype=float)
            else:
                mat_w = np.ones([pt.size for pt in pts], dtype=float)

            assert mat_w.shape == tuple([pt.size for pt in pts])

            # assemble matrix (if weight is not zero) by calling the appropriate kernel (1d, 2d or 3d)
            if np.any(np.abs(mat_w) > 1e-14):
                if isinstance(vec, StencilVector):
                    kernel(*el_loc_indices, *wspace.degree, *periodic, *starts, *pads, 
                           *nqs, *wts, *basis, mat_w, vec._data)
                else:
                    kernel(*el_loc_indices, *wspace.degree, *periodic, *starts, *pads, 
                           *nqs, *wts, *basis, mat_w, vec[a]._data)
                    
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

            # periodicity: True (1) or False (0)
            periodic = [int(periodic) for periodic in wspace.periodic]

            # global element indices on process over which integration is performed
            el_loc_indices = [quad_grid.indices for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            nqs = [quad_grid.num_quad_pts     for quad_grid in wspace.quad_grids]
            pts = [quad_grid.points.flatten() for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # perform evaluation by calling the appropriate kernel (1d, 2d or 3d)
            values = np.zeros([pt.size for pt in pts], dtype=float)
            
            if isinstance(coeffs, StencilVector):
                kernel(*el_loc_indices, *wspace.degree, *periodic, *starts, *pads, 
                       *nqs, *basis, coeffs._data, values)
            else:
                kernel(*el_loc_indices, *wspace.degree, *periodic, *starts, *pads, 
                       *nqs, *basis, coeffs[a]._data, values)
                
            blocks += [values]
                
        if len(blocks) == 1:
            return blocks[0]
        else:
            return blocks
            