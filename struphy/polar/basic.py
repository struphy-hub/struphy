from psydac.linalg.basic import VectorSpace, Vector
from psydac.linalg.stencil import StencilVector, StencilVectorSpace
from psydac.linalg.block import BlockVector, BlockVectorSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace
from struphy.psydac_api.linear_operators import LinOpWithTransp
import numpy as np

from mpi4py import MPI


class PolarDerhamSpace(VectorSpace):
    '''Derham space with polar basis in eta1-eta2.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.

        space_id : str
            Space identifier for the field (H1, Hcurl, Hdiv, L2 or H1vec).
        '''

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
            self._n_pol = (3,)
            self._rings = (2,)
            self._dimension = (
                (self.n[0] - self.rings[0])*self.n[1] + self.n_pol[0])*self.n[2]
            self._n3_loc = (self.ends[2] + 1 - self.starts[2],)
            self._n3_glo = (self.n[2],)
        elif space_id == 'Hcurl':
            self._n_pol = (0, 2, 3)
            self._rings = (1, 2, 2)
            dim1 = ((self.d[0] - self.rings[0]) *
                    self.n[1] + self.n_pol[0])*self.n[2]
            dim2 = ((self.n[0] - self.rings[1]) *
                    self.d[1] + self.n_pol[1])*self.n[2]
            dim3 = ((self.n[0] - self.rings[2]) *
                    self.n[1] + self.n_pol[2])*self.d[2]
            self._dimension = dim1 + dim2 + dim3
            self._n3_loc = (self.ends[0][2] + 1 - self.starts[0][2],
                            self.ends[1][2] + 1 - self.starts[1][2],
                            self.ends[2][2] + 1 - self.starts[2][2])
            self._n3_glo = (self.n[2], self.n[2], self.d[2])
        elif space_id == 'Hdiv':
            self._n_pol = (2, 0, 0)
            self._rings = (2, 1, 1)
            dim1 = ((self.n[0] - self.rings[0]) *
                    self.d[1] + self.n_pol[0])*self.d[2]
            dim2 = ((self.d[0] - self.rings[1]) *
                    self.n[1] + self.n_pol[1])*self.d[2]
            dim3 = ((self.d[0] - self.rings[2]) *
                    self.d[1] + self.n_pol[2])*self.n[2]
            self._dimension = dim1 + dim2 + dim3
            self._n3_loc = (self.ends[0][2] + 1 - self.starts[0][2],
                            self.ends[1][2] + 1 - self.starts[1][2],
                            self.ends[2][2] + 1 - self.starts[2][2])
            self._n3_glo = (self.d[2], self.d[2], self.n[2])
        elif space_id == 'L2':
            self._n_pol = (0,)
            self._rings = (1,)
            self._dimension = (
                (self.d[0] - self.rings[0])*self.d[1] + self.n_pol[0])*self.d[2]
            self._n3_loc = (self.ends[2] + 1 - self.starts[2],)
            self._n3_glo = (self.d[2],)
        elif space_id == 'H1vec':
            self._n_pol = (3, 3, 3)
            self._rings = (2, 2, 2)
            self._dimension = (
                ((self.n[0] - self.rings[0])*self.n[1] + self.n_pol[0])*self.n[2]) * 3
            self._n3_loc = (self.ends[0][2] + 1 - self.starts[0][2],
                            self.ends[1][2] + 1 - self.starts[1][2],
                            self.ends[2][2] + 1 - self.starts[2][2])
            self._n3_glo = (self.n[2], self.n[2], self.n[2])
        else:
            raise ValueError('Space not supported.')

        self._n_comps = len(self.n_pol)

    @property
    def dtype(self):
        ''' TODO '''
        return self._dtype

    @property
    def derham(self):
        ''' TODO '''
        return self._derham

    @property
    def space_id(self):
        ''' TODO '''
        return self._space_id

    @property
    def n_comps(self):
        ''' TODO '''
        return self._n_comps

    @property
    def n(self):
        ''' TODO '''
        return self._n

    @property
    def d(self):
        ''' TODO '''
        return self._d

    @property
    def parent_space(self):
        ''' Tensor product Stencil-/BlockVectorSpace of the parent space. '''
        return self._parent_space

    @property
    def starts(self):
        ''' TODO '''
        return self._starts

    @property
    def ends(self):
        ''' TODO '''
        return self._ends

    @property
    def dimension(self):
        ''' TODO '''
        return self._dimension

    @property
    def n_pol(self):
        ''' Number of polar basis functions in each component (tuple for vector-valued).'''
        return self._n_pol

    @property
    def rings(self):
        ''' Number of rings to be set to zero in tensor-product basis (tuple for vector-valued).'''
        return self._rings

    @property
    def n3_loc(self):
        ''' Tuple holding number of basis function in eta3 on current process, for each component.'''
        return self._n3_loc

    @property
    def n3_glo(self):
        ''' Tuple holding total (global) number of basis function in eta3, for each component.'''
        return self._n3_glo

    def zeros(self):
        ''' TODO '''
        zeros_pol = [np.zeros((m, n)) for m, n in zip(
            self.n_pol, self.n3_glo)]  # polar coeffs
        # full tensor product vector (0th- and 1st ring must always be set to zero)
        zeros_tp = self.parent_space.zeros()
        return zeros_pol, zeros_tp


class PolarVector(Vector):
    '''Element of a PolarDerhamSpace.

    A PolarVector is a 2-list: the first entry is a list of np.arrays of the polar coeffs. 
    The second entry is a tensor product Stencil/BlockVector of the parent space.
    '''

    def __init__(self, V):

        assert isinstance(V, PolarDerhamSpace)
        self._space = V
        self._dtype = V.dtype

        # initialize as zero vector
        self._pol, self._tp = V.zeros()

    @property
    def space(self):
        return self._space

    @property
    def dtype(self):
        return self._dtype

    @property
    def pol(self):
        '''Polar coefficients as np.array.'''
        return self._pol

    @property
    def tp(self):
        '''Tensor product Stencil-/BlockVector with inner ring entries set to zero.'''
        return self._tp

    @tp.setter
    def tp(self, v):
        '''Tensor product Stencil-/BlockVector with inner ring entries set to zero.'''
        assert v.space == self.space.parent_space

        if isinstance(v, StencilVector):
            start_eta1 = v.starts[0]

            if start_eta1 == 0:
                assert np.all(v[:self.space.rings[0], :, :] == 0.)
        else:
            start_eta1 = [start[0] for start in v.space.starts]

            for n, s_e1 in enumerate(start_eta1):
                if s_e1 == 0:
                    assert np.all(v[n][:self.space.rings[n], :, :] == 0.)

        self._tp = v

    def dot(self, v):
        assert isinstance(v, PolarVector)
        assert v.space == self.space
        return sum([a1.flatten().dot(a2.flatten()) for a1, a2 in zip(self.pol, v.pol)]) + self.tp.dot(v.tp)

    def set_tp_coeffs_to_zero(self):
        set_tp_rings_to_zero(self.tp, self.space.rings)

    def toarray(self):
        return self.pol, self.tp.toarray()

    def toarray_concatenated(self):

        if isinstance(self.tp, StencilVector):
            s1, s2, s3 = self.space.starts
            e1, e2, e3 = self.space.ends

            out = self.tp.toarray()[self.space.rings[0]
                                    * self.space.n[1]*self.space.n3_glo[0]:]
            out = np.concatenate((self.pol[0].flatten(), out))

        else:
            out1 = self.tp[0].toarray()[self.space.rings[0] *
                                        self.space.n[1]*self.space.n3_glo[0]:]
            out2 = self.tp[1].toarray()[self.space.rings[1] *
                                        self.space.n[1]*self.space.n3_glo[1]:]
            out3 = self.tp[2].toarray()[self.space.rings[2] *
                                        self.space.n[1]*self.space.n3_glo[2]:]

            out = np.concatenate((self.pol[0].flatten(), out1,
                                  self.pol[1].flatten(), out2,
                                  self.pol[2].flatten(), out3))

        return out

    def copy(self):
        w = PolarVector(self.space)
        w.pol[:] = [arr.copy() for arr in self.pol]
        try:
            w.tp[:] = self.tp.copy()[:]
        except:
            w.tp[0][:] = self.tp[0].copy()[:]
            w.tp[1][:] = self.tp[1].copy()[:]
            w.tp[2][:] = self.tp[2].copy()[:]
        return w

    def __neg__(self):
        w = PolarVector(self.space)
        w.pol[:] = [-arr for arr in self.pol]
        try:
            w.tp[:] = -self.tp[:]
        except:
            w.tp[0][:] = -self.tp[0][:]
            w.tp[1][:] = -self.tp[1][:]
            w.tp[2][:] = -self.tp[2][:]
        return w

    def __mul__(self, a):
        w = PolarVector(self.space)
        w.pol[:] = [arr*a for arr in self.pol]
        try:
            w.tp[:] = (self.tp*a)[:]
        except:
            w.tp[0][:] = (self.tp[0]*a)[:]
            w.tp[1][:] = (self.tp[1]*a)[:]
            w.tp[2][:] = (self.tp[2]*a)[:]
        return w

    def __rmul__(self, a):
        w = PolarVector(self.space)
        w.pol[:] = [a*arr for arr in self.pol]
        try:
            w.tp[:] = (a*self.tp)[:]
        except:
            w.tp[0][:] = (a*self.tp[0])[:]
            w.tp[1][:] = (a*self.tp[1])[:]
            w.tp[2][:] = (a*self.tp[2])[:]
        return w

    def __add__(self, v):
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        w.pol[:] = [a1 + a2 for a1, a2 in zip(self.pol, v.pol)]
        try:
            w.tp[:] = (self.tp + v.tp)[:]
        except:
            w.tp[0][:] = (self.tp[0] + v.tp[0])[:]
            w.tp[1][:] = (self.tp[1] + v.tp[1])[:]
            w.tp[2][:] = (self.tp[2] + v.tp[2])[:]
        return w

    def __sub__(self, v):
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        w = PolarVector(self.space)
        w.pol[:] = [a1 - a2 for a1, a2 in zip(self.pol, v.pol)]
        try:
            w.tp[:] = (self.tp - v.tp)[:]
        except:
            w.tp[0][:] = (self.tp[0] - v.tp[0])[:]
            w.tp[1][:] = (self.tp[1] - v.tp[1])[:]
            w.tp[2][:] = (self.tp[2] - v.tp[2])[:]
        return w

    def __imul__(self, a):
        self.pol[:] = [arr*a for arr in self.pol]
        try:
            self.tp[:] = (self.tp*a)[:]
        except:
            self.tp[0][:] = (self.tp[0]*a)[:]
            self.tp[1][:] = (self.tp[1]*a)[:]
            self.tp[2][:] = (self.tp[2]*a)[:]
        return self

    def __iadd__(self, v):
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        self.pol[:] = [a1 + a2 for a1, a2 in zip(self.pol, v.pol)]
        try:
            self.tp[:] = (self.tp + v.tp)[:]
        except:
            self.tp[0][:] = (self.tp[0] + v.tp[0])[:]
            self.tp[1][:] = (self.tp[1] + v.tp[1])[:]
            self.tp[2][:] = (self.tp[2] + v.tp[2])[:]
        return self

    def __isub__(self, v):
        assert isinstance(v, PolarVector)
        assert v.space == self.space

        self.pol[:] = [a1 - a2 for a1, a2 in zip(self.pol, v.pol)]
        try:
            self.tp[:] = (self.tp - v.tp)[:]
        except:
            self.tp[0][:] = (self.tp[0] - v.tp[0])[:]
            self.tp[1][:] = (self.tp[1] - v.tp[1])[:]
            self.tp[2][:] = (self.tp[2] - v.tp[2])[:]
        return self


class PolarExtractionOperator(LinOpWithTransp):
    """
    Linear operator mapping from a Derham tensor product space to its polar subspace.

    Parameters
    ----------
        V : StencilVectorSpace | BlockVectorSpace
            Domain of the operator.

        W : PolarDerhamSpace
            Codomain of the operator.

        pol_blocks : list[list[ndarray]]
            Blocks that map two inner most rings of tensor-product basis function to polar basis functions 
            (shape = (n_pol, 2*n_eta2)).

        transposed : bool
            Whether to create the transposed extraction operator.

        tp_operator : LinOpWithTransp
            Operator that maps tensor-product basis functions to tensor-product basis functions. 
            Must be transposed if transposed=True.
    """

    def __init__(self, V, W, pol_blocks=None, transposed=False, tp_operator=None):

        assert isinstance(V, (StencilVectorSpace, BlockVectorSpace))
        assert isinstance(W, PolarDerhamSpace)

        if transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W

        self._transposed = transposed
        self._dtype = V.dtype

        self._n2 = W.n[1]

        # shape of polar blocks
        self._pol_blocks_shapes = []
        for n, n_pol in enumerate(W.n_pol):
            self._pol_blocks_shapes += [[]]
            for m in range(W.n_comps):
                self._pol_blocks_shapes[-1] += [(n_pol, 2*self.n2)]

        # polar blocks (map from tensor-product rings to polar coefficients)
        self.pol_blocks = pol_blocks

        # tensor-product blocks (map from tensor-product rings to tensor-product coefficients)
        self.tp_operator = tp_operator

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
    def n2(self):
        return self._n2

    @property
    def pol_blocks(self):
        return self._pol_blocks

    @pol_blocks.setter
    def pol_blocks(self, blocks):

        if self.transposed:
            n_comps = self.domain.n_comps
        else:
            n_comps = self.codomain.n_comps

        if blocks is None:
            blocks = [[None for n in range(n_comps)] for m in range(n_comps)]
        else:
            assert isinstance(blocks, list)
            for n, row in enumerate(blocks):
                for m, col in enumerate(row):
                    if (n == 0 or n == 1) and m == 2:
                        assert col is None
                    elif n == 2 and (m == 0 or m == 1):
                        assert col is None
                    else:
                        assert col.shape == self._pol_blocks_shapes[n][m]

        self._pol_blocks = blocks

    @property
    def tp_operator(self):
        return self._tp_operator
    
    @tp_operator.setter
    def tp_operator(self, operator):

        if self.transposed:
            V = self.domain.parent_space
            W = self.codomain
        else:
            V = self.domain
            W = self.codomain.parent_space

        if operator is not None:
            assert isinstance(operator, LinOpWithTransp)
            assert operator.domain == V
            assert operator.codomain == W

        self._tp_operator = operator  

    def dot(self, v, out=None):

        if self.transposed:

            assert isinstance(v, PolarVector)
            assert v.space == self.domain

            # tensor-product coefficients first
            if self.tp_operator is not None:
                out = self.tp_operator.dot(v.tp)
            else:
                out = v.tp.copy()

            # overwrite polar coefficients (inner rings)
            if isinstance(self.codomain, StencilVectorSpace):
                if self.pol_blocks[0][0] is not None:
                    s1, s2, s3 = self.codomain.starts
                    e1, e2, e3 = self.codomain.ends

                    if s1 == 0:
                        out[0:v.space.rings[0], s2:e2 + 1, s3:e3 + 1] = self.pol_blocks[0][0].T.dot(v.pol[0]).reshape(
                            2, self.n2, self.domain.n3_glo[0])[0:v.space.rings[0], s2:e2 + 1, s3:e3 + 1]

            elif isinstance(self.codomain, BlockVectorSpace):
                set_tp_rings_to_zero(out, v.space.rings)
                for m in range(3):
                    s1, s2, s3 = self.codomain.starts[m]
                    e1, e2, e3 = self.codomain.ends[m]

                    if s1 == 0:
                        for n in range(3):
                            if self.pol_blocks[n][m] is not None:
                                out[m][0:v.space.rings[m], s2:e2 + 1, s3:e3 + 1] += self.pol_blocks[n][m].T.dot(v.pol[n]).reshape(
                                    2, self.n2, self.domain.n3_glo[n])[0:v.space.rings[m], s2:e2 + 1, s3:e3 + 1]

            else:
                raise ValueError('Wrong codomain in transposed extraction operator.')

        else:

            assert isinstance(v, (StencilVector, BlockVector))
            assert v.space == self.domain

            out = PolarVector(self.codomain)

            if isinstance(v, StencilVector):
                if self.pol_blocks[0][0] is not None:
                    s1 = self.domain.starts[0]
                    out_loc = np.zeros(
                        (self.codomain.n_pol[0], self.codomain.n3_glo[0]), dtype=float)

                    if s1 == 0:
                        s2, s3 = self.domain.starts[1:]
                        e2, e3 = self.domain.ends[1:]
                        tmp_vec = np.zeros(
                            (2, self.n2, self.codomain.n3_glo[0]), dtype=float)
                        tmp_vec[:, s2:e2 + 1, s3:e3 +
                                1] = v[0:2, s2:e2 + 1, s3:e3 + 1]
                        out_loc[:, :] = self.pol_blocks[0][0].dot(
                            tmp_vec.reshape(2*self.n2, self.codomain.n3_glo[0]))

                    if self.codomain.derham.comm is not None:
                        self.codomain.derham.comm.Allreduce(
                            out_loc, out.pol[0], op=MPI.SUM)
                    else:
                        out.pol[0][:] = out_loc

            else:

                for m in range(3):
                    out_loc = np.zeros(
                        (self.codomain.n_pol[m], self.codomain.n3_glo[m]), dtype=float)

                    for n in range(3):
                        if self.pol_blocks[m][n] is not None:
                            s1 = self.domain.starts[n][0]

                            if s1 == 0:
                                s2, s3 = self.domain.starts[n][1:]
                                e2, e3 = self.domain.ends[n][1:]
                                tmp_vec = np.zeros(
                                    (2, self.n2, self.codomain.n3_glo[n]), dtype=float)
                                tmp_vec[:, s2:e2 + 1, s3:e3 +
                                        1] = v[n][0:2, s2:e2 + 1, s3:e3 + 1]
                                out_loc += self.pol_blocks[m][n].dot(
                                    tmp_vec.reshape(2*self.n2, self.codomain.n3_glo[n]))

                    if self.codomain.derham.comm is not None:
                        self.codomain.derham.comm.Allreduce(
                            out_loc, out.pol[m], op=MPI.SUM)
                    else:
                        out.pol[m][:] = out_loc

            if self.tp_operator is not None:
                temp = self.tp_operator.dot(v)
            else:
                temp = v.copy()

            set_tp_rings_to_zero(temp, self.codomain.rings)
            out.tp = temp    

        return out

    def transpose(self):
        if self.tp_operator is None:
            tp_op_transposed = None
        else:
            tp_op_transposed = self.tp_operator.transpose()

        return PolarExtractionOperator(self.domain, self.codomain, pol_blocks=self.pol_blocks, tp_operator=tp_op_transposed, transposed=True)


class PolarLinearOperator(LinOpWithTransp):
    """
    Linear operator mapping between two PolarDerhamSpaces.
    """

    def __init__(self, V, W):

        assert isinstance(V, PolarDerhamSpace)
        assert isinstance(W, PolarDerhamSpace)

        self._domain = V
        self._codomain = W
        self._dtype = V.dtype

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def dot(self, v, out=None):
        pass

    def transpose(self):
        pass



def set_tp_rings_to_zero(v, n_rings):
    assert isinstance(n_rings, tuple)

    if isinstance(v, StencilVector):
        start_eta1 = v.starts[0]

        if start_eta1 == 0:
            v[:n_rings[0], :, :] = 0.
    elif isinstance(v, BlockVector):
        start_eta1 = [start[0] for start in v.space.starts]

        for n, s_e1 in enumerate(start_eta1):
            if s_e1 == 0:
                v[n][:n_rings[n], :, :] = 0.