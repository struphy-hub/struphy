import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api import mass_kernels
from struphy.psydac_api.linear_operators import LinOpWithTransp, CompositeLinearOperator
from struphy.psydac_api.utilities import apply_essential_bc_to_array, apply_essential_bc_to_pol

from struphy.polar.basic import PolarVector


class WeightedMassOperators:
    """
    Class for assembling weighted mass matrices in 3d.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.
            
        weights : obj
            A general object that provides access to callables that serve as weight functions (e.g. instance of a subclass of struphy.fields_background.mhd_equil.base.EquilibriumMHD).
    """
    
    def __init__(self, derham, domain, **weights):
        
        self._derham = derham
        self._domain = domain
        
        # Wrapper functions for metric coefficients
        def DF(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, False, False, True, False)

        def DFT(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, False, False, True, True)
            
        def DFinv(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, False, False, True, False)

        def DFinvT(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, False, False, True, True)

        def G(e1, e2, e3):
            return domain.metric(e1, e2, e3, False, False, True)
            
        def Ginv(e1, e2, e3):
            return domain.metric_inv(e1, e2, e3, False, False, True)
            
        def sqrt_g(e1, e2, e3):
            return abs(domain.jacobian_det(e1, e2, e3, False, False))
        
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
                    fun_M1n[-1] += [lambda e1, e2, e3, m=m, n=n : Ginv(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
                    fun_M2n[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]/sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
                    fun_Mvn[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]*sqrt_g(e1, e2, e3)*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
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
            self._M0 = WeightedMassOperator(self.derham.V0, self.derham.V0, V_extraction_op=self.derham.E0, W_extraction_op=self.derham.E0, weight=self._fun_M0, transposed=False, bc=self.derham.bc)
        
        return self._M0
    
    @property
    def M1(self):
        """ Mass matrix M1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * G_inv_ab * Lambda^1_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_M1'):
            self._M1 = WeightedMassOperator(self.derham.V1, self.derham.V1, V_extraction_op=self.derham.E1, W_extraction_op=self.derham.E1, weight=self._fun_M1, transposed=False, bc=self.derham.bc)
        
        return self._M1
    
    @property
    def M2(self):
        """ Mass matrix M2_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M2'):
            self._M2 = WeightedMassOperator(self.derham.V2, self.derham.V2, V_extraction_op=self.derham.E2, W_extraction_op=self.derham.E2, weight=self._fun_M2, transposed=False, bc=self.derham.bc)
        
        return self._M2
    
    @property
    def M3(self):
        """ Mass matrix M3_(ijk lmn) = integral( Lambda^3_(ijk) * Lambda^3_(lmn) / sqrt(g) ). 
        """
        
        if not hasattr(self, '_M3'):
            self._M3 = WeightedMassOperator(self.derham.V3, self.derham.V3, V_extraction_op=self.derham.E3, W_extraction_op=self.derham.E3, weight=self._fun_M3, transposed=False, bc=self.derham.bc)
        
        return self._M3
    
    @property
    def Mv(self):
        """ Mass matrix Mv_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * G_ab * Lambda^v_(b,lmn) * sqrt(g) ). 
        """
        
        if not hasattr(self, '_Mv'):
            self._Mv = WeightedMassOperator(self.derham.V0vec, self.derham.V0vec, V_extraction_op=self.derham.Ev, W_extraction_op=self.derham.Ev, weight=self._fun_Mv, transposed=False, bc=self.derham.bc)
        
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
            self._M1n = WeightedMassOperator(self.derham.V1, self.derham.V1, V_extraction_op=self.derham.E1, W_extraction_op=self.derham.E1, weight=self._fun_M1n, transposed=False, bc=self.derham.bc)
        
        return self._M1n
    
    @property
    def M2n(self):
        """ Mass matrix M2n_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_M2n'):
            self._M2n = WeightedMassOperator(self.derham.V2, self.derham.V2, V_extraction_op=self.derham.E2, W_extraction_op=self.derham.E2, weight=self._fun_M2n, transposed=False, bc=self.derham.bc)
        
        return self._M2n
    
    @property
    def Mvn(self):
        """ Mass matrix Mvn_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^v_(b,lmn) * sqrt(g) * n^0_eq * G_ab ).
        """
        
        if not hasattr(self, '_Mvn'):
            self._Mvn = WeightedMassOperator(self.derham.V0vec, self.derham.V0vec, V_extraction_op=self.derham.Ev, W_extraction_op=self.derham.Ev, weight=self._fun_Mvn, transposed=False, bc=self.derham.bc)
        
        return self._Mvn
    
    @property
    def M1J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c * G_inv_ab ).
        """
        
        if not hasattr(self, '_M1J'):
            self._M1J = WeightedMassOperator(self.derham.V2, self.derham.V1, V_extraction_op=self.derham.E2, W_extraction_op=self.derham.E1, weight=self._fun_M1J, transposed=False, bc=self.derham.bc)
        
        return self._M1J
    
    @property
    def M2J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * epsilon_(acb) * J^2_eq_c).
        """
        
        if not hasattr(self, '_M2J'):
            self._M2J = WeightedMassOperator(self.derham.V2, self.derham.V2, V_extraction_op=self.derham.E2, W_extraction_op=self.derham.E2, weight=self._fun_M2J, transposed=False, bc=self.derham.bc)
        
        return self._M2J
    
    @property
    def MvJ(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c ).
        """
        
        if not hasattr(self, '_MvJ'):
            self._MvJ = WeightedMassOperator(self.derham.V2, self.derham.V0vec, V_extraction_op=self.derham.E2, W_extraction_op=self.derham.Ev, weight=self._fun_MvJ, transposed=False, bc=self.derham.bc)
        
        return self._MvJ

    
    
class WeightedMassOperator( LinOpWithTransp ):
    """
    Weighted mass matrix in the full tensor-product space (i.e. without polar extraction operators and/or boundary operators).
    
    Parameters
    ----------
        V : TensorFemSpace or ProductFemSpace
            Tensor product spline space from psydac.fem.tensor (domain, input space).
            
        W : TensorFemSpace or ProductFemSpace
            Tensor product spline space from psydac.fem.tensor (codomain, output space).
            
        V_extraction_op : PolarExtractionOperator | NoneType
            Extraction operator to polar sub-space of V.
            
        W_extraction_op : PolarExtractionOperator | NoneType
            Extraction operator to polar sub-space of W.
            
        weight : list | NoneType
            Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.
            
        transposed : bool
            Whether to assemble the transposed operator.
            
        bc : list | NoneType
            Boundary conditions in each direction in format [[e1(0), e1(1)], [e2(0), e2(1)], [e3(0), e3(1)]].
    """
    
    def __init__(self, V, W, V_extraction_op=None, W_extraction_op=None, weight=None, transposed=False, bc=None):
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
        
        assert isinstance(V, FemSpace)
        assert isinstance(W, FemSpace)
        
        self._V = V
        self._W = W
        
        if V_extraction_op is not None:
            assert V_extraction_op.domain == V.vector_space
            
        if W_extraction_op is not None:
            assert W_extraction_op.domain == W.vector_space
        
        self._V_extraction_op = V_extraction_op
        self._W_extraction_op = W_extraction_op
        
        self._weight = weight
        self._transposed = transposed
        self._bc = bc
        
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
            self._domain_symbolic_name = W_name
            self._codomain_symbolic_name = V_name
        else:
            self._domain_symbolic_name = V_name
            self._codomain_symbolic_name = W_name
        
        # ====== assemble tensor-product mass matrix ====
        self._mat = WeightedMassOperator.assemble_mat(V, W, weight)
        # ===============================================
        
        # build composite linear operator E * M * E^T with basis extraction operators
        if V_extraction_op is None:
            self._operator = CompositeLinearOperator(W_extraction_op, self._mat)
        else:
            self._operator = CompositeLinearOperator(W_extraction_op, self._mat, V_extraction_op.transpose())
        
        if transposed:
            self._operator = self.operator.transpose()
            
        # set domain and codomain
        self._domain = self.operator.domain
        self._codomain = self.operator.codomain
    
    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def transposed(self):
        return self._transposed
    
    @property
    def operator(self):
        return self._operator
    
    def dot(self, v, out=None, apply_bc=True):
        """
        Applies the basis projection operator to the FE coefficients v belonging to V.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Input FE coefficients from V.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from W.vector_space.
        """

        assert v.space == self.domain
        
        out = self.operator.dot(v)
        
        # apply boundary conditions to output vector
        if apply_bc and self._bc is not None:
            if isinstance(out, PolarVector):
                apply_essential_bc_to_array(self._codomain_symbolic_name, out.tp, self._bc)
                apply_essential_bc_to_pol(self._codomain_symbolic_name, out.pol, self._bc[2])
            else:
                apply_essential_bc_to_array(self._codomain_symbolic_name, out, self._bc)
        
        assert out.space == self.codomain
        
        return out
    
    def transpose(self):
        """
        Returns the transposed operator.
        """
        return WeightedMassOperator(self._V, self._W, self._V_extraction_op, self._W_extraction_op, self._weight, not self.transposed, self._bc)
    
    @staticmethod
    def assemble_mat(V, W, weight=None):
        """
        TODO
        """
        
        # identify space IDs
        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            V_name = 'H1vec'

        if hasattr(W.symbolic_space, 'name'):
            W_name = W.symbolic_space.name
        else:
            W_name = 'H1vec'
        
        # collect TensorFemSpaces in tuple
        if V_name in {'H1', 'L2'}:
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
            
        if W_name in {'H1', 'L2'}:
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
        
        blocks = []
    
        # loop over codomain spaces (rows)
        for a, wspace in enumerate(Wspaces):
            blocks += [[]]

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

                # evaluation of weight function at quadrature points (optional)
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = weight[a][b](*PTS).copy()
                    else:
                        mat_w = np.ones_like([pt.size for pt in pts], dtype=float)
                else:
                    mat_w = np.ones([pt.size for pt in pts], dtype=float)

                # evaluated basis functions at quadrature points of output space
                basis_i = [quad_grid.basis for quad_grid in vspace.quad_grids]

                # assemble matrix (if weight is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                if np.any(mat_w):
                    M = StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                    
                    kernel = getattr(mass_kernels, 'kernel_' + str(V.ldim) + 'd')
                    
                    kernel(*el_loc_indices, *wspace.degree, *vspace.degree, *periodic, *starts_out, *pads_out,
                           *nqs, *wts, *basis_o, *basis_i, mat_w, M._data)

                    blocks[-1] += [M]

                else:
                    blocks[-1] += [None]

        if len(blocks) == len(blocks[0]) == 1:
            return blocks[0][0] 
        else:
            return BlockMatrix(V.vector_space, W.vector_space, blocks)