from cmath import polar
import numpy as np
from scipy.sparse import csr_matrix

from psydac.linalg.basic import VectorSpace, Vector
from psydac.linalg.stencil import StencilVector, StencilVectorSpace
from psydac.linalg.block import BlockVector, BlockVectorSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from struphy.psydac_api.linear_operators import LinOpWithTransp

from mpi4py import MPI


class PolarDerhamSpace(VectorSpace):
    """
    Derham space with polar basis in eta1-eta2.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        space_id : str
            Space identifier for the field (H1, Hcurl, Hdiv, L2 or H1vec).
    """

    def __init__(self, derham, space_id):

        assert derham.spl_kind[0] == False, 'Spline basis in eta1 must be clamped'
        assert derham.spl_kind[1], 'Spline basis in eta2 must be periodic'

        assert derham.p[0] > 1 and derham.p[
            1] > 1, 'Spline degrees in (eta1, eta2) must be at least two'

        # other properties
        self._dtype = float
        self._derham = derham
        self._space_id = space_id

        # dimensions of 1d spaces
        self._n = [space.nbasis for space in derham.V0.spaces]
        self._d = [space.nbasis for space in derham.V3.spaces]

        self._parent_space = getattr(
            derham, derham.spaces_dict[space_id]).vector_space
        
        self._starts = self.parent_space.starts
        self._ends = self._parent_space.ends

        # polar properties
        if space_id == 'H1':
            self._n_polar = (3,)
            self._n_rings = (2,)
            self._dimension = (
                (self.n[0] - self.n_rings[0])*self.n[1] + self.n_polar[0])*self.n[2]
            self._n2 = (self.n[1],)
            self._n3 = (self.n[2],)
            self._type_of_basis_3 = (self.derham.spline_types['V0'][2],)
        elif space_id == 'Hcurl':
            self._n_polar = (0, 2, 3)
            self._n_rings = (1, 2, 2)
            dim1 = ((self.d[0] - self.n_rings[0]) *
                    self.n[1] + self.n_polar[0])*self.n[2]
            dim2 = ((self.n[0] - self.n_rings[1]) *
                    self.d[1] + self.n_polar[1])*self.n[2]
            dim3 = ((self.n[0] - self.n_rings[2]) *
                    self.n[1] + self.n_polar[2])*self.d[2]
            self._dimension = dim1 + dim2 + dim3
            self._n2 = (self.n[1], self.d[1], self.n[1])
            self._n3 = (self.n[2], self.n[2], self.d[2])
            self._type_of_basis_3 = (self.derham.spline_types['V1'][0][2],
                                     self.derham.spline_types['V1'][1][2],
                                     self.derham.spline_types['V1'][2][2])
        elif space_id == 'Hdiv':
            self._n_polar = (2, 0, 0)
            self._n_rings = (2, 1, 1)
            dim1 = ((self.n[0] - self.n_rings[0]) *
                    self.d[1] + self.n_polar[0])*self.d[2]
            dim2 = ((self.d[0] - self.n_rings[1]) *
                    self.n[1] + self.n_polar[1])*self.d[2]
            dim3 = ((self.d[0] - self.n_rings[2]) *
                    self.d[1] + self.n_polar[2])*self.n[2]
            self._dimension = dim1 + dim2 + dim3
            self._n2 = (self.d[1], self.n[1], self.d[1])
            self._n3 = (self.d[2], self.d[2], self.n[2])
            self._type_of_basis_3 = (self.derham.spline_types['V2'][0][2],
                                     self.derham.spline_types['V2'][1][2],
                                     self.derham.spline_types['V2'][2][2])
        elif space_id == 'L2':
            self._n_polar = (0,)
            self._n_rings = (1,)
            self._dimension = (
                (self.d[0] - self.n_rings[0])*self.d[1] + self.n_polar[0])*self.d[2]
            self._n2 = (self.d[1],)
            self._n3 = (self.d[2],)
            self._type_of_basis_3 = (self.derham.spline_types['V3'][2],)
        elif space_id == 'H1vec':
            self._n_polar = (3, 3, 3)
            self._n_rings = (2, 2, 2)
            self._dimension = (
                ((self.n[0] - self.n_rings[0])*self.n[1] + self.n_polar[0])*self.n[2]) * 3
            self._n2 = (self.n[1], self.n[1], self.n[1])
            self._n3 = (self.n[2], self.n[2], self.n[2])
            self._type_of_basis_3 = (self.derham.spline_types['V0vec'][0][2],
                                     self.derham.spline_types['V0vec'][1][2],
                                     self.derham.spline_types['V0vec'][2][2])
        else:
            raise ValueError('Space not supported.')

        self._n_comps = len(self.n_polar)

        if self.n_comps == 1:
            if self.starts[0] == 0:
                assert self.ends[0] > self.n_rings[0], 'MPI coeff decomposition in eta_1 too small for polar splines!'
        else:
            for n in range(3):
                if self.starts[n][0] == 0:
                    assert self.ends[n][0] > self.n_rings[n], 'MPI coeff decomposition in eta_1 too small for polar splines!'

    @property
    def dtype(self):
        """ TODO 
        """
        return self._dtype

    @property
    def derham(self):
        """ TODO 
        """
        return self._derham

    @property
    def space_id(self):
        """ TODO 
        """
        return self._space_id

    @property
    def n_comps(self):
        """ TODO 
        """
        return self._n_comps

    @property
    def n(self):
        """ TODO 
        """
        return self._n

    @property
    def d(self):
        """ TODO 
        """
        return self._d

    @property
    def parent_space(self):
        """ The parent space (StencilVectorSpace or BlockVectorSpace) of which the PolarDerhamSpace is a sub-space.
        """
        return self._parent_space

    @property
    def starts(self):
        """ TODO 
        """
        return self._starts

    @property
    def ends(self):
        """ TODO 
        """
        return self._ends

    @property
    def dimension(self):
        """ TODO 
        """
        return self._dimension

    @property
    def n_polar(self):
        """ Number of polar basis functions in each component (tuple for vector-valued).
        """
        return self._n_polar

    @property
    def n_rings(self):
        """ Number of rings to be set to zero in tensor-product basis (tuple for vector-valued).
        """
        return self._n_rings

    @property
    def n2(self):
        """ Tuple holding total (global) number of basis function in eta2, for each component.
        """
        return self._n2

    @property
    def n3(self):
        """ Tuple holding total (global) number of basis function in eta3, for each component.
        """
        return self._n3
    
    @property
    def type_of_basis_3(self):
        """ Tuple holding type of spline basis (B-splines or M-splines), for each component.
        """
        return self._type_of_basis_3

    def zeros(self):
        """ 
        Creates an element of the vector space filled with zeros.
        """
        
        # polar coeffs
        zeros_pol = [np.zeros((m, n)) for m, n in zip(
            self.n_polar, self.n3)] 
        
        # full tensor product vector
        zeros_tp = self.parent_space.zeros()
        
        return zeros_pol, zeros_tp


class PolarVector(Vector):
    """
    Element of a PolarDerhamSpace.

    An instance of a PolarVector consists of two parts:
        1. a list of np.arrays of the polar coeffs (not distributed) 
        2. a tensor product StencilVector/BlockVector of the parent space with inner rings set to zero (distributed).
    
    Parameters
    ----------
        V : PolarDerhamSpace
            Vector space which the polar vector to be created belongs to.
    """

    def __init__(self, V):

        assert isinstance(V, PolarDerhamSpace)
        self._space = V
        self._dtype = V.dtype

        # initialize as zero vector
        self._pol, self._tp = V.zeros()

    @property
    def space(self):
        """ TODO 
        """
        return self._space

    @property
    def dtype(self):
        """ TODO 
        """
        return self._dtype

    @property
    def pol(self):
        """ Polar coefficients as np.array.
        """
        return self._pol
    
    @pol.setter
    def pol(self, v):
        """ In-place setter for polar coefficients.
        """
        assert isinstance(v, list)
        assert len(v) == self.space.n_comps
        for n in range(self.space.n_comps):
            self._pol[n][:] = v[n]

    @property
    def tp(self):
        """ Tensor product Stencil-/BlockVector with inner rings set to zero.
        """
        return self._tp

    @tp.setter
    def tp(self, v):
        """ In-place setter for tensor product Stencil-/BlockVector with constraint that inner rings must be zero.
        """
        assert v.space == self.space.parent_space

        if isinstance(v, StencilVector):
            self._tp[:] = v[:]
        elif isinstance(v, BlockVector):
            for n, starts in enumerate(v.space.starts):
                self._tp[n][:] = v[n][:]
        else:
            raise ValueError('Attribute can only be set with instances of either StencilVector or BlockVector!')

        self.set_tp_coeffs_to_zero()

    def dot(self, v):
        """ 
        Scalar product with another instance of PolarVector.
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space
        
        # tensor-product part
        out = self.tp.dot(v.tp)
        
        # polar part
        out += sum([a1.flatten().dot(a2.flatten()) for a1, a2 in zip(self.pol, v.pol)])
        
        return out

    def set_tp_coeffs_to_zero(self):
        """
        Sets inner tensor-product rings that make up the polar splines to zero.
        """
        set_tp_rings_to_zero(self.tp, self.space.n_rings)

    def toarray(self):
        """
        Converts the polar vector to a 1d numpy array.
        """

        if isinstance(self.tp, StencilVector):
            s1, s2, s3 = self.space.starts
            e1, e2, e3 = self.space.ends

            out = self.tp.toarray()[self.space.n_rings[0]*
                                    self.space.n[1]*self.space.n3[0]:]
            out = np.concatenate((self.pol[0].flatten(), out))

        else:
            out1 = self.tp[0].toarray()[self.space.n_rings[0]*
                                        self.space.n[1]*self.space.n3[0]:]
            out2 = self.tp[1].toarray()[self.space.n_rings[1]*
                                        self.space.n[1]*self.space.n3[1]:]
            out3 = self.tp[2].toarray()[self.space.n_rings[2]*
                                        self.space.n[1]*self.space.n3[2]:]

            out = np.concatenate((self.pol[0].flatten(), out1,
                                  self.pol[1].flatten(), out2,
                                  self.pol[2].flatten(), out3))
            
        return out   
    
    def toarray_tp(self):
        """
        Converts the Stencil-/BlockVector to a 1d numpy array but NOT the polar part.
        """
        return self.pol, self.tp.toarray()

    def copy(self):
        """ TODO 
        """
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0]
            w._tp[:] = self.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n]
                w._tp[n][:] = self.tp[n][:]
        return w

    def __neg__(self):
        """ TODO 
        """
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = -self.pol[0]
            w._tp[:] = -self.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = -self.pol[n]
                w._tp[n][:] = -self.tp[n][:]
        return w

    def __mul__(self, a):
        """ TODO 
        """
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0]*a
            w._tp[:] = self.tp[:]*a
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n]*a
                w._tp[n][:] = self.tp[n][:]*a
        return w

    def __rmul__(self, a):
        """ TODO 
        """
        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = a*self.pol[0]
            w._tp[:] = a*self.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = a*self.pol[n]
                w._tp[n][:] = a*self.tp[n][:]
        return w

    def __add__(self, v):
        """ TODO 
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0] + v.pol[0]
            w._tp[:] = self.tp[:] + v.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n] + v.pol[n]
                w._tp[n][:] = self.tp[n][:] + v.tp[n][:]
        return w

    def __sub__(self, v):
        """ TODO 
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        if isinstance(w.tp, StencilVector):
            w._pol[0][:] = self.pol[0] - v.pol[0]
            w._tp[:] = self.tp[:] - v.tp[:]
        else:
            for n in range(3):
                w._pol[n][:] = self.pol[n] - v.pol[n]
                w._tp[n][:] = self.tp[n][:] - v.tp[n][:]
        return w

    def __imul__(self, a):
        """ TODO 
        """
        if isinstance(self.tp, StencilVector):
            self._pol[0] *= a
            self._tp *= a
        else:
            for n in range(3):
                self._pol[n] *= a
                self._tp[n] *= a
        return self

    def __iadd__(self, v):
        """ TODO 
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        if isinstance(self.tp, StencilVector):
            self._pol[0] += v.pol[0]
            self._tp += v.tp
        else:
            for n in range(3):
                self._pol[n] += v.pol[n]
                self._tp[n] += v.tp[n]
        return self

    def __isub__(self, v):
        """ TODO 
        """
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        if isinstance(self.tp, StencilVector):
            self._pol[0] -= v.pol[0]
            self._tp -= v.tp
        else:
            for n in range(3):
                self._pol[n] -= v.pol[n]
                self._tp[n] -= v.tp[n]
        return self


class PolarExtractionOperator(LinOpWithTransp):
    """
    Linear operator mapping from Stencil-/BlockVectorSpace (V) to PolarDerhamSpace (W).
    
    For fixed k, the dot product maps tensor-product (tp) basis functions/DOFs a_ijk with index i <= n_rings to
    
        a) n_polar polar basis functions/DOFs (b_0k, ... , b_(n_polar-1, k)), 
        b) "first tp ring" b_ijk with index i = n_rings,
        
    and leaves the outer zone unchanged (identity map). For notation, see Fig. below.
 
                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |     
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |    
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |

    Fig : Indexing of 3d spline tensor-product (tp) basis functions/DOFs. 
          A linear combination of n_rings rings is used to construct n_polar polar basis functions/DOFs 
          and "first tp ring" basis functions/DOFs.
    
    Parameters
    ----------
        V : StencilVectorSpace | BlockVectorSpace
            Domain of the operator (always corresponding to not transposed operator).

        W : PolarDerhamSpace
            Codomain of the operator (always corresponding to not transposed operator).

        pol_blocks : list
            Blocks that map inner most n_rings + 1 tp rings to polar basis functions/DOFs. 
                * shape[m][n] = (n_polar[m], (n_rings[n] + 1)*n2) if transposed=False.
                * shape[m][n] = ((n_rings[m] + 1)*n2, n_polar[n]) if transposed=True.

        tp_blocks : list
            Blocks that map inner most n_rings + 1 tp rings to "first tp ring". 
                * shape[m][n] = (n2, (n_rings[n] + 1)*n2) if transposed=False.
                * shape[m][n] = ((n_rings[m] + 1)*n2, n2) if transposed=True.
        
        transposed : bool
            Whether the transposed extraction operator shall be constructed.
    """

    def __init__(self, V, W, pol_blocks=None, tp_blocks=None, transposed=False):

        assert isinstance(V, (StencilVectorSpace, BlockVectorSpace))
        assert isinstance(W, PolarDerhamSpace)
        assert W.parent_space == V

        self._transposed = transposed
        
        if self.transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        self._dtype = V.dtype

        # demanded shapes of polar blocks and tp blocks
        self._pol_blocks_shapes = []
        self._tp_blocks_shapes = []
        
        # loop over codomain components
        for m in range(W.n_comps):
            
            self._pol_blocks_shapes += [[]]
            self._tp_blocks_shapes += [[]]
            
            # loop over domain components
            for n in range(W.n_comps):
                
                # dot product not possible if types of basis in eta_3 are not compatible
                if W.type_of_basis_3[m] != W.type_of_basis_3[n]:
                    self._pol_blocks_shapes[-1] += [None]
                    self._tp_blocks_shapes[-1] += [None]
                else:
                    if transposed:
                        self._pol_blocks_shapes[-1] += [((W.n_rings[m] + 1)*W.n2[m], W.n_polar[n])]
                        self._tp_blocks_shapes[-1] += [((W.n_rings[m] + 1)*W.n2[m], W.n2[n])]
                    else:
                        self._pol_blocks_shapes[-1] += [(W.n_polar[m], (W.n_rings[n] + 1)*W.n2[n])]
                        self._tp_blocks_shapes[-1] += [(W.n2[m], (W.n_rings[n] + 1)*W.n2[n])]


        # set polar blocks (map from first n_rings + 1 tensor-product rings to polar basis functions/DOFs)
        self.pol_blocks = pol_blocks

        # set tensor-product blocks (map from first n_rings + 1 tensor-product rings to "first tp ring")
        self.tp_blocks = tp_blocks

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
    def pol_blocks_shapes(self):
        return self._pol_blocks_shapes

    @property
    def pol_blocks(self):
        return self._pol_blocks

    @pol_blocks.setter
    def pol_blocks(self, blocks):
        """ TODO
        """
        
        assert isinstance(blocks, list) or blocks is None
        
        if blocks is None:
            self._pol_blocks = set_blocks(self.pol_blocks_shapes)
        else:
            check_blocks(blocks, self.pol_blocks_shapes)
            self._pol_blocks = blocks

    @property
    def tp_blocks_shapes(self):
        return self._tp_blocks_shapes
    
    @property
    def tp_blocks(self):
        return self._tp_blocks
    
    @tp_blocks.setter
    def tp_blocks(self, blocks):
        """ TODO
        """
        
        assert isinstance(blocks, list) or blocks is None
        
        if blocks is None:
            self._tp_blocks = set_blocks(self.tp_blocks_shapes)
        else:
            check_blocks(blocks, self.tp_blocks_shapes)
            self._tp_blocks = blocks
            
    def dot(self, v, out=None):
        """
        Dot product mapping from Stencil-/BlockVector to PolarVector (or vice versa in case of transposed)

        Parameters
        ----------
            v : StencilVector | BlockVector | (PolarVector, if transposed)
                Input vector.

            out : PolarVector | (StencilVector | BlockVector, if transposed) 
                Optional output vector the result will be written to. If not given, a vector is returned. 
        """
        
        assert v.space == self.domain

        # transposed operator (polar vector --> tensor-product vector)
        if self.transposed:

            # 1. identity operation
            if out is None:
                do_return = True
                out = v.tp.copy()
            else:
                do_return = False
                if isinstance(out, StencilVector):
                    out[:] = v.tp[:]
                elif isinstance(out, BlockVector):
                    for n in range(3):
                        out[n][:] = v.tp[n][:]
            
            # 2. map polar coeffs
            dot_with_parts_of_polar(self.pol_blocks, v, out)

            # 3. map "first tp ring"
            out2 = v.space.parent_space.zeros()
            dot_with_parts_of_polar(self.tp_blocks, v, out2)

            # 4. sum results
            out += out2

        # "standard" operator (tensor-product vector --> polar vector)
        else:
            
            if out is None:
                do_return = True
                out = PolarVector(self.codomain)
            else:
                do_return = False
                assert isinstance(out, PolarVector)
            
            # 1. identity operation
            out.tp = v
            
            # 2. map to polar coeffs 
            dot_with_all_rings(self.pol_blocks, v, out)

            # 3. map to "first tp ring"
            dot_with_all_rings(self.tp_blocks, v, out)

        if do_return:
            return out

    def transpose(self):
        """
        Returns the transposed operator.
        """

        if self.transposed:
            V = self.codomain
            W = self.domain
        else:
            V = self.domain
            W = self.codomain

        pol_blocks = transpose_block_mat(self.pol_blocks)
        tp_blocks = transpose_block_mat(self.tp_blocks)

        return PolarExtractionOperator(V, W, pol_blocks=pol_blocks, tp_blocks=tp_blocks, transposed=not self.transposed)


class PolarLinearOperator(LinOpWithTransp):
    """
    Linear operator mapping from PolarDerhamSpace (V) to PolarDerhamSpace (W).
    
                     /       /       /               /               /
             k = 1  /       /       /               /               /
           k = 0   /       /       /               /               /
                  ------------------------------- -----------------
          j = 0   |       |       |               |               |
          j = 1   |       |       |               |               |     
                  |       |       |               |               |
                  | i = 0 |  ...  |  i = n_rings  |  i > n_rings  |    
                  |       |       |               |               |   /
                  |       |       |               |               |  /
        j = n2-1  |       |       |               |               | /
                  -------------------------------------------------
                  |  polar rings  | first tp ring | outer tp zone |
                  |               |         total tp zone         |

    Fig : Indexing of 3d spline tensor-product (tp) basis functions/DOFs. 
          A linear combination of n_rings rings is used to construct n_polar polar basis functions/DOFs 
          and "first tp ring" basis functions/DOFs.

    Parameters
    ----------
        V : PolarDerhamSpace
            Domain of the operator (always corresponding to not transposed operator).

        W : PolarDerhamSpace
            Codomain of the operator (always corresponding to not transposed operator).

        pol_blocks : list
            Blocks that map inner most n_rings + 1 tp rings to polar basis functions/DOFs. 
                * shape[m][n] = (n_polar[m], (n_rings[n] + 1)*n2) if transposed=False.
                * shape[m][n] = ((n_rings[m] + 1)*n2, n_polar[n]) if transposed=True.

        tp_blocks : list
            Blocks that map inner most n_rings + 1 tp rings to "first tp ring". 
                * shape[m][n] = (n2, (n_rings[n] + 1)*n2) if transposed=False.
                * shape[m][n] = ((n_rings[m] + 1)*n2, n2) if transposed=True.
                
        tiny_blocks : list
            TODO
            
        tp_operator : LinearOperator
            TODO
        
        transposed : bool
            Whether to create the transposed extraction operator.
    """

    def __init__(self, V, W, pol_blocks=None, tp_blocks=None, tiny_blocks=None, tp_operator=None, transposed=False):

        assert isinstance(V, PolarDerhamSpace)
        assert isinstance(W, PolarDerhamSpace)

        self._transposed = transposed
        
        if self.transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        self._dtype = V.dtype
        
        _dom = self.domain
        _codom = self.codomain

        # number of components of domain/codomain elements
        n_comps_dom = _dom.n_comps
        n_comps_codom = _codom.n_comps

        # demanded shape of polar blocks, tensor-product blocks and tiny blocks
        self._pol_blocks_shapes = []
        self._tp_blocks_shapes = []
        self._tiny_blocks_shapes = []
        
        # loop over codomain components
        for m in range(n_comps_codom):
            
            self._pol_blocks_shapes += [[]]
            self._tp_blocks_shapes += [[]]
            self._tiny_blocks_shapes += [[]]
            
            # loop over domain components
            for n in range(n_comps_dom):
                
                # dot product not possible if types of basis in eta_3 are not compatible
                if _codom.type_of_basis_3[m] != _dom.type_of_basis_3[n]:
                    self._pol_blocks_shapes[-1] += [None]
                    self._tp_blocks_shapes[-1] += [None]
                    self._tiny_blocks_shapes[-1] += [None]
                else:
                    if self.transposed:
                        self._pol_blocks_shapes[-1] += [((_codom.n_rings[m] + 1)*_codom.n2[m], _dom.n_polar[n])]
                        self._tp_blocks_shapes[-1] += [((_codom.n_rings[m] + 1)*_codom.n2[m], _dom.n2[n])]
                    else:
                        self._pol_blocks_shapes[-1] += [(_codom.n_polar[m], (_dom.n_rings[n] + 1)*_dom.n2[n])]
                        self._tp_blocks_shapes[-1] += [(_codom.n2[m], (_dom.n_rings[n] + 1)*_dom.n2[n])]
                        
                    self._tiny_blocks_shapes[-1] += [(_codom.n_polar[m], _dom.n_polar[n])]

        # set polar blocks (map from first n_rings + 1 tensor-product rings to polar basis functions/DOFs)
        self.pol_blocks = pol_blocks

        # set tensor-product blocks (map from first n_rings + 1 tensor-product rings to "first tp ring")
        self.tp_blocks = tp_blocks

        # set tiny blocks (map from polar basis functions/DOFs to polar basis functions/DOFs)
        self.tiny_blocks = tiny_blocks

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
    def pol_blocks_shapes(self):
        return self._pol_blocks_shapes

    @property
    def pol_blocks(self):
        return self._pol_blocks

    @pol_blocks.setter
    def pol_blocks(self, blocks):
        """ TODO
        """
        
        assert isinstance(blocks, list) or blocks is None
        
        if blocks is None:
            self._pol_blocks = set_blocks(self.pol_blocks_shapes)
        else:
            check_blocks(blocks, self.pol_blocks_shapes)
            self._pol_blocks = blocks
            
    @property
    def tp_blocks_shapes(self):
        return self._tp_blocks_shapes
    
    @property
    def tp_blocks(self):
        return self._tp_blocks
    
    @tp_blocks.setter
    def tp_blocks(self, blocks):
        """ TODO
        """
        
        assert isinstance(blocks, list) or blocks is None
        
        if blocks is None:
            self._tp_blocks = set_blocks(self.tp_blocks_shapes)
        else:
            check_blocks(blocks, self.tp_blocks_shapes)
            self._tp_blocks = blocks

    @property
    def tiny_blocks_shapes(self):
        return self._tiny_blocks_shapes
    
    @property
    def tiny_blocks(self):
        return self._tiny_blocks
    
    @tiny_blocks.setter
    def tiny_blocks(self, blocks):
        """ TODO
        """
        
        assert isinstance(blocks, list) or blocks is None
        
        if blocks is None:
            self._tiny_blocks = set_blocks(self.tiny_blocks_shapes)
        else:
            check_blocks(blocks, self.tiny_blocks_shapes)
            self._tiny_blocks = blocks
            
    def dot(self, v, out=None):
        """
        Maps between two PolarDerhamSpaces. 
            a) maps "first tp ring" to polar coeffs (dense), 
            b) maps "first tp ring" to "first tp ring" (dense),
            c) maps polar coeffs to polar coeffs (dense),
            d) maps total tp zone to total tp zone (stencil). 

        Parameters
        ----------
            v : StencilVector | BlockVector | (PolarVector, if transposed)
                Input vector.

            out : PolarVector | (StencilVector | BlockVector, if transposed) 
                Optional output vector the result will be written to. 
        """
        
        assert v.space == self.domain

        # transposed operator
        if self.transposed:

            # 1. identity operation
            if out is None:
                do_return = True
                out = v.tp.copy()
            else:
                do_return = False
                if isinstance(out, StencilVector):
                    out[:] = v.tp[:]
                elif isinstance(out, BlockVector):
                    for n in range(3):
                        out[n][:] = v.tp[n][:]
            
            # 2. map polar coeffs
            dot_with_parts_of_polar(self.pol_blocks, v, out)

            # 3. map "first tp ring"
            out2 = v.space.parent_space.zeros()
            dot_with_parts_of_polar(self.tp_blocks, v, out2)

            # 4. sum results
            out += out2

        # "standard" operator
        else:
            
            if out is None:
                do_return = True
                out = PolarVector(self.codomain)
            else:
                do_return = False
                assert isinstance(out, PolarVector)
            
            # 1. identity operation
            out.tp = v
            
            # 2. map to polar coeffs 
            dot_with_all_rings(self.pol_blocks, v, out)

            # 3. map to "first tp ring"
            dot_with_all_rings(self.tp_blocks, v, out)

        if do_return:
            return out

    def transpose(self):
        """
        Returns the transposed operator.
        """

        if self.transposed:
            V = self.codomain
            W = self.domain
        else:
            V = self.domain
            W = self.codomain

        pol_blocks = transpose_block_mat(self.pol_blocks)
        tp_blocks = transpose_block_mat(self.tp_blocks)
        tiny_blocks = transpose_block_mat(self.tiny_blocks)

        return PolarLinearOperator(V, W, pol_blocks=pol_blocks, tp_blocks=tp_blocks, tiny_blocks=tiny_blocks, tp_operators=self.tp_operator.transpose(), transposed=not self.transposed)


def set_tp_rings_to_zero(v, n_rings):
    """
    Sets a certain number of rings of a Stencil-/BlockVector in eta_1 direction to zero.
    
    Parameters
    ----------
        v : StencilVector | BlockVector
            The vector whose inner rings shall be set to zero.
            
        n_rings : tuple
            The number of rings that shall be set to zero (has lenght 1 for StencilVector and 3 for BlockVector).
    """
    assert isinstance(n_rings, tuple)

    if isinstance(v, StencilVector):
        if v.starts[0] == 0:
            v[:n_rings[0], :, :] = 0.
    elif isinstance(v, BlockVector):
        for n, starts in enumerate(v.space.starts):
            if starts[0] == 0:
                v[n][:n_rings[n], :, :] = 0.
    else:
        raise ValueError('Input vector must be an instance of StencilVector of BlockVector!')

def dot_with_all_rings(blocks, v, out):
    """
    Maps "polar rings" and "first tp ring" to either polar coeffs (blocks.shape[0] = # polar coeffs)
    or "first tp ring" (blocks.shape[0] = n2), where n2 is the number of coeffs in eta_2 (see Fig).

                 /               /              /              /
                /   k = 1       /              /              /
               /  k = 0        /              /              / 
              -----------------------------------------------
    j = 0     |       |       |              |              |
    j = 1     |       |       |              |              |     
              |       |       |              |              |
              | i = 0 |  ...  | i = n_rings  |  i > n_rings |    
              |       |       |              |              |   /
              |       |       |              |              |  /
    j = n2-1  |       |       |              |              | /
              -----------------------------------------------
              |  polar rings  |first tp ring |outer tp zone |

    Fig : Indexing of 3d spline tensor-product (tp) basis functions. 
          "Polar rings" are used in the linear combination of polar spline basis functions.

    Parameters
    ----------
        blocks : list[list[ndarray]]
            Blocks 3x3 or 1x1 definig the dot product. Shape is (n_rows, (n_rings + 1)*n2), 
            where n_rows = # polar coeffs or n_rows = n2.
        
        v : StencilVector | BlockVector
            Input vector.

        out : PolarVector
            Output vector. 
    """

    assert isinstance(blocks, list)
    assert isinstance(v, (StencilVector, BlockVector))
    assert isinstance(out, PolarVector)

    polar_space = out.space

    # scalar space or vector-valued space
    is_scalar = isinstance(v, StencilVector)
    n_comps = polar_space.n_comps

    # n_rings (see Fig)
    n_rings = polar_space.n_rings

    # number of coeffs in eta_2 and eta_3 direction
    n2 = polar_space.n2
    n3 = polar_space.n3

    # extract size of blocks
    if n_comps == 3:
        n_rows = [blocks[m][n].shape[0] for m, n in zip((0, 1, 2), (0, 0, 2))]
    else:
        n_rows = [blocks[0][0].shape[0]]

    # determine if mapped to polar coeffs or "first tp ring"
    map_to_tp = True if n_rows[0] == n2[0] else False

    # convert needed scalar attributes to lists
    in_starts = [v.space.starts] if is_scalar else v.space.starts
    in_ends = [v.space.ends] if is_scalar else v.space.ends
    in_vec = [v] if is_scalar else v
    
    out_starts = [polar_space.starts] if is_scalar else polar_space.starts
    out_ends = [polar_space.ends] if is_scalar else polar_space.ends
    out_tp = [out.tp] if is_scalar else out.tp

    # loop over codomain components
    for m in range(n_comps):
                    
        res = np.zeros((n_rows[m], n3[m]), dtype=float)
        
        # loop over domain components
        for n in range(n_comps):
            if in_starts[n][0] == 0:
                s1, s2, s3 = in_starts[n]
                e1, e2, e3 = in_ends[n]

                if blocks[m][n] is not None:
                    tmp = np.zeros((n_rings[n] + 1, n2[n], n3[n]), dtype=float)
                    tmp[:, s2:e2 + 1, s3:e3 + 1] = in_vec[n][0:n_rings[n] + 1, s2:e2 + 1, s3:e3 + 1]
                    res += blocks[m][n].dot(tmp.reshape((n_rings[n] + 1)*n2[n], n3[n]))         
                    
        # sum up local dot products
        if polar_space.derham.comm is not None:
            polar_space.derham.comm.Allreduce(
                MPI.IN_PLACE, res, op=MPI.SUM)

        # write result to output polar vector (in-place)
        if map_to_tp:
            s1, s2, s3 = out_starts[m]
            e1, e2, e3 = out_ends[m]
            out_tp[m][n_rings[m], s2:e2 + 1, s3:e3 + 1] = res[s2:e2 + 1, s3:e3 + 1]
        else:
            out.pol[m][:, :] = res

def dot_with_parts_of_polar(blocks, v, out):
    """
    Maps either polar coeffs (blocks.shape[1] = # polar coeffs) or "first tp ring" (blocks.shape[1] = n2) 
    to "polar rings" and "first tp ring", where n2 is the number of coeffs in eta_2 (see Fig).

                 /               /              /              /
                /   k = 1       /              /              /
               /  k = 0        /              /              / 
              -----------------------------------------------
    j = 0     |       |       |              |              |
    j = 1     |       |       |              |              |     
              |       |       |              |              |
              | i = 0 |  ...  | i = n_rings  |  i > n_rings |    
              |       |       |              |              |   /
              |       |       |              |              |  /
    j = n2-1  |       |       |              |              | /
              -----------------------------------------------
              |  polar rings  |first tp ring |outer tp zone |

    Fig : Indexing of 3d spline tensor-product (tp) basis functions. 
          "Polar rings" are used in the linear combination of polar spline basis functions.

    Parameters
    ----------
        blocks : list[list[ndarray]]
            Blocks 3x3 or 1x1 definig the dot product. Shape is ((n_rings + 1)*n2, n_cols), 
            where n_cols = # polar coeffs or n_cols = n2.
        
        v : PolarVector
            Input vector. 
        
        out : StencilVector | BlockVector
            Output vector.
    """

    assert isinstance(blocks, list)
    assert isinstance(v, PolarVector)
    assert isinstance(out, (StencilVector, BlockVector))
    
    polar_space = v.space

    # scalar space or vector-valued space
    is_scalar = isinstance(out, StencilVector)
    n_comps = polar_space.n_comps

    # n_rings (see Fig)
    n_rings = polar_space.n_rings

    # number of coeffs in eta_2 and eta_3 direction
    n2 = polar_space.n2
    n3 = polar_space.n3

    # extract size of blocks
    if n_comps == 3:
        n_cols = [blocks[m][n].shape[1] for m, n in zip((0, 0, 2), (0, 1, 2))]
    else:
        n_cols = [blocks[0][0].shape[1]]

    # determine if mapped from polar coeffs or "first tp ring"
    map_from_tp = True if n_cols[0] == n2[0] else False

    # convert needed scalar attributes to lists
    out_starts = [out.space.starts] if is_scalar else out.space.starts
    out_ends = [out.space.ends] if is_scalar else out.space.ends
    out_vec = [out] if is_scalar else out
    
    in_starts = [polar_space.starts] if is_scalar else polar_space.starts
    in_ends = [polar_space.ends] if is_scalar else polar_space.ends
    in_tp = [v.tp] if is_scalar else v.tp

    # loop over codomain components
    for m in range(n_comps):
        
        res = np.zeros((n_rings[m] + 1, n2[m], n3[m]), dtype=float)
        
        # loop over domain components
        for n in range(n_comps):
            
            # use (n, m) block instead of (m, n) because of transposed
            if blocks[m][n] is not None:
                
                if map_from_tp:
                    if in_starts[n][0] == 0:
                        s1, s2, s3 = in_starts[n]
                        e1, e2, e3 = in_ends[n]
                        tmp_loc = np.zeros((n2[n], n3[n]), dtype=float)
                        tmp_loc[s2:e2 + 1, s3:e3 + 1] = in_tp[n][n_rings[n], s2:e2 + 1, s3:e3 + 1]
                        res += blocks[m][n].dot(tmp_loc).reshape(n_rings[m] + 1, n2[m], n3[m])
                else:
                    res += blocks[m][n].dot(v.pol[n]).reshape(n_rings[m] + 1, n2[m], n3[m])

        if map_from_tp:
            if polar_space.derham.comm is not None:
                polar_space.derham.comm.Allreduce(
                MPI.IN_PLACE, res, op=MPI.SUM)

        if out_starts[m][0] == 0:
            s1, s2, s3 = out_starts[m]
            e1, e2, e3 = out_ends[m]
            out_vec[m][0:n_rings[m] + 1, s2:e2 + 1, s3:e3 + 1] = res[:, s2:e2 + 1, s3:e3 + 1]

def dot_pol_pol(blocks, v, out):
    """
    Maps polar coeffs to polar coeffs.

    Parameters
    ----------
        blocks : list[list[ndarray]]
            blocks[m][n].shape = (out.pol[m].shape[0], v.pol[n].shape[0])

        v : PolarVector
            Input vector.

        out : PolarVector
            output vector.
    """

    assert isinstance(v, PolarVector)
    assert isinstance(out, PolarVector)

    for out_pol, row in zip(out.pol, blocks):
        out_pol[:, :] = 0.
        for v_pol, block in zip(v.pol, row):
            if block is not None:
                out_pol[:, :] += block.dot(v_pol)

def transpose_block_mat(blocks):
    """
    Transpose 2D nested list of numpy arrays.

    Parameters
    ----------
        blocks : list[list[ndarray]]
            Block matrix.

    Returns
    -------
        out : list[list[ndarray]]
            Transposed block matrix. 
    """

    n_rows = len(blocks)
    n_cols = len(blocks[0])

    out = []

    for m in range(n_cols):
        out += [[]]
        for n in range(n_rows):
            if blocks[n][m] is not None:
                out[-1] += [blocks[n][m].T]
            else:
                out[-1] += [None]

    return out

def set_blocks(shapes):
    """
    Creates zero blocks list of given shapes.
    
    Parameters
    ----------
        shapes : list
            2D list of tuples (.,.) holding the shape of individual blocks (shapes[m][n] = (.,.)).
            
    Returns
    -------
        blocks : list
            2D list of 2D matrices (scipy.sparse or nd.ndarray) filled with zeros.
    """
    
    assert isinstance(shapes, list)
    
    blocks = []
    
    for row in shapes:
        blocks += [[]]
        for shp in row:
            if shp is None:
                blocks[-1] += [None]
            else:
                blocks[-1] += [csr_matrix(shp, dtype=float)]
                    
    return blocks

def check_blocks(blocks, shapes):
    """
    Checks if given blocks have correct shape.
    
    Parameters
    ----------
        blocks : list
            2D list of 2D matrices (scipy.sparse or nd.ndarray).
    
        shapes : list
            2D list of tuples (.,.) holding the shape of individual blocks (shapes[m][n] = (.,.)).
    """
    
    assert isinstance(shapes, list)
    assert isinstance(blocks, list)
    
    for blk_row, shp_row in zip(blocks, shapes):
        for blk, shp in zip(blk_row, shp_row):
            if shp is None:
                assert blk is None
            else:
                assert blk.shape == shp