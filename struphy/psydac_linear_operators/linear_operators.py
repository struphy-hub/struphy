from abc   import abstractmethod

from psydac.linalg.basic import LinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import pcg


class LinOpWithTransp(LinearOperator):

    @abstractmethod
    def transpose(self):
        raise NotImplementedError()


class CompositeLinearOperator(LinOpWithTransp):
    '''L_n(L_{n-1}(...L_2(L_1(v))...) .'''

    def __init__(self, *operators):
        '''
        Parameters
        ----------
            operators: LinOpWithTransp | StencilMatrix | BlockMatrix
                Sequence: L_n, L_{n-1}, ..., L_2, L_1 .
        '''

        self._operators = list(operators)[::-1]

        assert len(self._operators) > 1

        for op2, op1 in zip(self._operators[1:], self._operators[:-1]):
            assert isinstance(op1, (LinOpWithTransp, StencilMatrix, BlockMatrix))
            assert isinstance(op2, (LinOpWithTransp, StencilMatrix, BlockMatrix))
            assert op2.domain == op1.codomain

        self._domain = operators[-1].domain
        self._codomain = operators[0].codomain
        self._dtype = operators[-1].dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector))
        assert v.space == self.domain

        tmp = v
        for op in self._operators:
            tmp = op.dot(tmp)

        return tmp

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return CompositeLinearOperator(*[op.transpose() for op in self._operators])


class ScalarTimesLinearOperator(LinOpWithTransp):
    '''a*L with a in R.'''

    def __init__(self, a, operator):
        '''
        Parameters
        ----------
            a : float

            operator: LinOpWithTransp | StencilMatrix | BlockMatrix
        '''

        assert isinstance(a, (int, float, complex))
        assert isinstance(operator, (LinOpWithTransp, StencilMatrix, BlockMatrix))

        self._a = a
        self._operator = operator

        self._domain = operator.domain
        self._codomain = operator.codomain
        self._dtype = operator.dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector))
        assert v.space == self.domain

        return self._operator.dot(self._a * v)

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return ScalarTimesLinearOperator(self._a, self._operator.transpose())


class SumLinearOperator(LinOpWithTransp):
    '''(L_n + L_{n-1} + ... + L_2 + L_1)(v) .'''

    def __init__(self, *operators):
        '''
        Parameters
        ----------
            operators: LinOpWithTransp | StencilMatrix | BlockMatrix
        '''

        self._operators = list(operators)

        assert len(self._operators) > 1

        for op2, op1 in zip(self._operators[::-1][1:], self._operators[::-1][:-1]):
            assert isinstance(op1, (LinOpWithTransp, StencilMatrix, BlockMatrix))
            assert isinstance(op2, (LinOpWithTransp, StencilMatrix, BlockMatrix))
            assert op2.domain == op1.domain
            assert op2.codomain == op1.codomain

        self._domain = operators[0].domain
        self._codomain = operators[0].codomain
        self._dtype = operators[0].dtype

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector))
        assert v.space == self.domain

        tmp = self._operators[0].codomain.zeros()
        for op in self._operators:
            tmp += op.dot(v)

        return tmp

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return SumLinearOperator(*[op.transpose() for op in self._operators])


class InverseLinearOperator(LinOpWithTransp):
    '''L^{-1}.'''

    def __init__(self, operator, pc=None, tol=1e-6, maxiter=1000, verbose=False):
        '''
        Parameters
        ----------
            operator: LinOpWithTransp | StencilMatrix | BlockMatrix
                Should be symmetric, positive semi-definite.

            pc: NoneType | str | psydac.linalg.basic.LinearSolver | Callable
                Preconditioner for "operator", it should approximate the inverse of "operator".
                Can either be:
                * None, i.e. not pre-conditioning (this calls the standard `cg` method)
                * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)
                * A LinearSolver object (in which case the out parameter is used)
                * A callable with two parameters ("operator", r), where r is the residual.

            tol : float
                Absolute tolerance for L2-norm of residual r = A*x - b.

            maxiter: int
                Maximum number of iterations.

            verbose : bool
                If True, L2-norm of residual r is printed at each iteration. 
        '''

        assert isinstance(operator, (LinOpWithTransp, StencilMatrix, BlockMatrix))
        assert operator.domain.dimension == operator.codomain.dimension

        self._operator = operator
        self._pc = pc
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose

        self._domain = operator.domain
        self._codomain = operator.codomain
        self._dtype = operator.dtype
        self._info = None

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return self._dtype

    @property
    def info( self ):
        return self._info

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector))
        assert v.space == self.codomain

        x, self._info = pcg(self._operator, v, self._pc, tol=self._tol,
                      maxiter=self._maxiter, verbose=self._verbose)

        assert isinstance(x, (StencilVector, BlockVector))

        return x

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return InverseLinearOperator(self._operator.transpose(), pc=self._pc, tol=self._tol, maxiter=self._maxiter, verbose=self._verbose)