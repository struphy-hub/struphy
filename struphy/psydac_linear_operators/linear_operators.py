from psydac.linalg.basic import LinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector

class Product_matrix(LinearOperator):
    '''Product A*B of two linear operators.'''

    def __init__(self, A, B):
        '''
        Parameters
        ----------
            A: LinearOperator
                A.domain.dimension == B.codomain.dimension.
            
            B: LinearOperator
                A.domain.dimension == B.codomain.dimension.
        '''

        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert A.domain.dimension == B.codomain.dimension

        self._A = A
        self._B = B
        self._domain = B.domain
        self._codomain = A.codomain
        self._dtype = A.dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v ):
        assert isinstance(v, (StencilVector, BlockVector))
        assert v.space.dimension == self._B.domain.dimension
        assert self._B.dot(v).space.dimension == self._A.domain.dimension
        return self._A.dot(self._B.dot(v))


class Sum_matrix(LinearOperator):
    '''Sum A+B of two linear operators.'''

    def __init__(self, A, B):
        '''
        Parameters
        ----------
            A: LinearOperator
            
            B: LinearOperator
        '''

        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert A.domain.dimension == B.domain.dimension
        assert A.codomain.dimension == B.codomain.dimension

        self._A = A
        self._B = B
        self._domain = A.domain
        self._codomain = A.codomain
        self._dtype = A.dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v, out=None ):
        return self._A.dot(v) + self._B.dot(v)


class Difference_matrix(LinearOperator):
    '''Differece A-B of two inear operators.'''

    def __init__(self, A, B):
        '''
        Parameters
        ----------
            A: LinearOperator
            
            B: LinearOperator
        '''

        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert A.domain.dimension == B.domain.dimension
        assert A.codomain.dimension == B.codomain.dimension

        self._A = A
        self._B = B
        self._domain = A.domain
        self._codomain = A.codomain
        self._dtype = A.dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v, out=None ):
        return self._A.dot(v) - self._B.dot(v)