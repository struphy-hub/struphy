import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockLinearOperator

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api import mass_kernels
from struphy.psydac_api.linear_operators import LinOpWithTransp, CompositeLinearOperator, IdentityOperator, BoundaryOperator

from struphy.polar.basic import PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator
import struphy.psydac_api.quadrature_evaluation_kernels as hybrid_weight


class HybridOperators:
    """
    Class for assembling matrices in 3d.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.
            
        density :
            A stencilmatrix type object providing the data of values of density at quadrature points
    
        a : BlockVector
            A BlockVector saving the finte element coefficients of vector potential

        beq : BlockVector
            A BlockVector saving the finte element coefficients of background magnetic field
    """
    
    def __init__(self, derham, domain, density, a, beq):
        
        self._derham = derham
        self._domain = domain
        self._C = self._derham.curl
        self._density = density

        self._curla = self._C.dot(a)
        for i in range(3):
            if not self._curla[i].ghost_regions_in_sync: self._curla[i].update_ghost_regions()
        
        self._beq = beq
        for i in range(3):
            if not self._beq[i].ghost_regions_in_sync: self._beq[i].update_ghost_regions()

        self._space = derham.Vh[derham.spaces_dict['H1']]
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
    
    @property
    def derham(self):
        return self._derham
    
    @property
    def domain(self):
        return self._domain
        
    #######################################################################
    # matrices related to L2-scalar products in all 3d derham spaces #
    #######################################################################
    @property
    def HybridM1(self):
        """ Mass matrix M1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * G_inv_ab * Lambda^1_(b,lmn) * sqrt(g) ). 
        """
        if not hasattr(self, '_HybridM1'):
            self._HybridM1 = HybridOperator(self.derham.Vh_fem['1'], self.derham.Vh_fem['1'], self.derham.Vh_fem['2'], self._density, self._curla, self._beq, self.derham, self.domain)
        
        return self._HybridM1

    
    
class HybridOperator:
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
    
    def __init__(self, V, W, U, density, curl_a, b_eq, derham, domain):
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
        
        assert isinstance(V, FemSpace)
        assert isinstance(W, FemSpace)
        
        self._V = V
        self._W = W
        self._U = U
        self._density = density
        self._b = b_eq + curl_a
        self._derham = derham
        self._domain = domain
        

        # ====== assemble tensor-product mass matrix ====
        self._mat = HybridOperator.assemble_mat(V, W, U, density, self._derham, self._b, self._domain)
        # ===============================================

    
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
                tmp = self._mat.dot(v)
            else:
                tmp = self._mat.dot(v)
            
            if isinstance(tmp, PolarVector):
                out.set_vector(tmp)
            elif isinstance(tmp, StencilVector):
                out[:] = tmp[:]
            elif isinstance(tmp, BlockVector):
                out[0][:] = tmp[0][:]
                out[1][:] = tmp[1][:]
                out[2][:] = tmp[2][:]
        
        return out
    
    @staticmethod
    def assemble_mat(V, W, U, density, derham, b_value, domain):
        """
        Assembles weighted mass matrix as StencilMatrix/BlockLinearOperator corresponding to given domain/codomain spline spaces.
        
        Parameters
        ----------
            V : TensorFemSpace or ProductFemSpace of 1 form
                Tensor product spline space from psydac.fem.tensor (domain, input space).
            
            W : TensorFemSpace or ProductFemSpace of 1 form 
                Tensor product spline space from psydac.fem.tensor (codomain, output space).
                
            U : TensorFemSpace or ProductFemSpace of 2 form

            density : values of density at all the quadrature points, a stencilmatrix type object
                
        Returns
        -------
            mat : StencilMatrix | BlockLinearOperator
                Weighted mass matrix in the full tensor product FEM space.
        """
        
        crossset = np.zeros((3,3), dtype=int)
        crossset[0,1] = 2 
        crossset[0,2] = 1
        crossset[1,0] = 2 
        crossset[1,2] = 0
        crossset[2,0] = 1
        crossset[2,1] = 0

        # collect TensorFemSpaces for each component in tuple
        if isinstance(V, TensorFemSpace):
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
            
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        if isinstance(U, TensorFemSpace):
            Uspaces = (U,)
        else:
            Uspaces = U.spaces
        
        blocks = []

        weight_blocks = []
        # loop over weight spaces
        index1 = [derham.indN[0], derham.indD[0], derham.indD[0]]
        index2 = [derham.indD[1], derham.indN[1], derham.indD[1]]
        index3 = [derham.indD[2], derham.indD[2], derham.indN[2]]

        for a, uspace in enumerate(Uspaces):
            weight_blocks += [[]]

            # periodicity: True (1) or False (0)
            periodic = [int(periodic) for periodic in uspace.periodic]

            # global element indices on process over which integration is performed
            el_loc_indices = [quad_grid.indices for quad_grid in uspace.quad_grids]

            # global start spline index on process
            starts_out = [int(start) for start in uspace.vector_space.starts]

            # pads (ghost regions)
            pads_out = uspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            nqs = [quad_grid.num_quad_pts     for quad_grid in uspace.quad_grids]
            pts = [quad_grid.points           for quad_grid in uspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in uspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis_o = [quad_grid.basis for quad_grid in uspace.quad_grids]

            weight_blocks[a] = np.zeros( (len(el_loc_indices[0]), len(el_loc_indices[1]), len(el_loc_indices[2]), nqs[0], nqs[1], nqs[2]), dtype=float)
        
            # get values of weights at quadrature points.         
            hybrid_weight.quadrature_kernel(index1[a], index2[a], index3[a], *el_loc_indices, *uspace.degree, *periodic, *nqs, *basis_o, weight_blocks[a], b_value[a]._data)
        
        hybrid_weight.kernelg(*pts, *el_loc_indices, *periodic, *nqs, *wts, *weight_blocks, density._data, *domain.args_map)    
        # assemble the blocks in mass matrices
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
            pts = [quad_grid.points           for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights          for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis_o = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # loop over domain spaces (columns)
            for b, vspace in enumerate(Vspaces):

                basis_i = [quad_grid.basis for quad_grid in vspace.quad_grids]

                c = crossset[a,b]
                # assemble matrix (if weight is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                if np.any(np.abs(weight_blocks[c]) > 1e-14):

                    if a != b:
                        M = StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                    
                        #kernel = getattr(mass_kernels, 'kernel_' + str(V.ldim) + 'd')
                    
                        #kernel(*el_loc_indices, *wspace.degree, *vspace.degree, *periodic, *starts_out, *pads_out,
                           #*nqs, *wts, *basis_o, *basis_i, weight_blocks[c], M._data)

                        blocks[-1] += [np.sign(a-b)*M]
                    else:
                        # when a = b, values are 0 in this block
                        M = StencilMatrix(vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                        blocks[-1] += [M]
                else:
                    blocks[-1] += [None]

        if len(blocks) == len(blocks[0]) == 1:
            return blocks[0][0] 
        else:
            return BlockLinearOperator(V.vector_space, W.vector_space, blocks)