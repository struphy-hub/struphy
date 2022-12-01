from abc import abstractmethod

from psydac.linalg.basic import LinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.linalg.iterative_solvers import pcg

from struphy.psydac_api.utilities import apply_essential_bc_to_array
from struphy.polar.basic import PolarVector
from struphy.linear_algebra.iterative_solvers import pbicgstab


class LinOpWithTransp(LinearOperator):

    @abstractmethod
    def transpose(self):
        raise NotImplementedError()


class CompositeLinearOperator(LinOpWithTransp):
    r"""
    Composition of n linear operators: :math:`A(\mathbf v)=L_n(L_{n-1}(...L_2(L_1(\mathbf v))...)`.
    A 'None' operator is treated as identity.
    
    Parameters
    ----------
        operators: LinOpWithTransp | StencilMatrix | BlockMatrix | None
            The sequence of n linear operators (None is treated as identity).
    """

    def __init__(self, *operators):

        self._operators = [op for op in list(operators)[::-1] if op is not None]

        if len(self._operators) > 1:

            for op2, op1 in zip(self._operators[1:], self._operators[:-1]):
                assert isinstance(op1, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))
                assert isinstance(op2, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))
                assert op2.domain == op1.codomain

        self._domain = self._operators[0].domain
        self._codomain = self._operators[-1].codomain
        self._dtype = self._operators[-1].dtype

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
    def otype(self):
        return [type(op) for op in self._operators]
    
    @property
    def operators(self):
        return self._operators

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.domain

        tmp = v
        for op in self._operators:
            tmp = op.dot(tmp)

        return tmp

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return CompositeLinearOperator(*[op.transpose() for op in self._operators])


class ScalarTimesLinearOperator(LinOpWithTransp):
    r"""
    Multiplication of a linear operator with a scalar: :math:`A(\mathbf v)=aL(\mathbf v)` with :math:`a \in \mathbb R`.
    
    Parameters
    ----------
        a : float
            The scalar that is multiplied with the linear operator. 

        operator: LinOpWithTransp | StencilMatrix | BlockMatrix
            The linear operator.
    """

    def __init__(self, a, operator):

        assert isinstance(a, (int, float, complex))
        assert isinstance(operator, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))

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

    @property
    def otype(self):
        return type(self._operator)

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.domain

        return self._operator.dot(self._a * v)

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return ScalarTimesLinearOperator(self._a, self._operator.transpose())


class SumLinearOperator(LinOpWithTransp):
    r"""
    Sum of n linear operators: :math:`A(\mathbf v)=(L_n + L_{n-1} + ... + L_2 + L_1)(\mathbf v)`.
    
    Parameters
    ----------
        operators: LinOpWithTransp | StencilMatrix | BlockMatrix
            The sequence of n linear operators.
    """

    def __init__(self, *operators):

        self._operators = list(operators)

        assert len(self._operators) > 1

        for op2, op1 in zip(self._operators[::-1][1:], self._operators[::-1][:-1]):
            assert isinstance(op1, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))
            assert isinstance(op2, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))
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

    @property
    def otype(self):
        return [type(op) for op in self._operators]

    def dot( self, v , out=None ):
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.domain

        tmp = self._operators[0].codomain.zeros()
        for op in self._operators:
            tmp += op.dot(v)

        return tmp

    def transpose(self):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        return SumLinearOperator(*[op.transpose() for op in self._operators])


class InverseLinearOperator(LinOpWithTransp):
    r"""
    Inverse linear operator: :math:`A(\mathbf v)=L^{-1}(\mathbf v)`.
    
    Parameters
    ----------
        operator : LinOpWithTransp | StencilMatrix | BlockMatrix
            Should be symmetric, positive semi-definite.

        pc : NoneType | str | psydac.linalg.basic.LinearSolver | Callable
             Preconditioner for "operator", it should approximate the inverse of "operator".
             Can either be:
             * None, i.e. not pre-conditioning (this calls the standard `cg` method)
             * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)
             * A LinearSolver object (in which case the out parameter is used)
             * A callable with two parameters ("operator", r), where r is the residual.

        tol : float
            Absolute tolerance for L2-norm of residual r = A*x - b.

        maxiter : int
            Maximum number of iterations.

        solver_name : str
            The name of the iterative solver to be used for matrix inversion. Can either be:
            * pcg (default)
            * pbicgstab
    """

    def __init__(self, operator, pc=None, tol=1e-6, maxiter=1000, solver_name='pcg'):

        assert isinstance(operator, (LinOpWithTransp, StencilMatrix, BlockMatrix, KroneckerStencilMatrix))
        assert operator.domain.dimension == operator.codomain.dimension
        assert solver_name in {'pcg', 'pbicgstab'}

        self._operator = operator
        self._pc = pc
        self._tol = tol
        self._maxiter = maxiter
        self._solver_name = solver_name
        
        self._solver = globals()[solver_name]
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
    def otype(self):
        return type(self._operator)

    @property
    def info( self ):
        return self._info

    def dot( self, v , out=None, x0=None, verbose=False):
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.codomain

        x, self._info = self._solver(self._operator, v, self._pc, x0=x0, tol=self._tol,
                      maxiter=self._maxiter, verbose=verbose)

        assert isinstance(x, (StencilVector, BlockVector, PolarVector))

        return x

    def transpose(self, new_pc=None):
        # NOTE: we re-allocate all temporary vectors here... Maybe re-use instead?
        if new_pc is None:
            return InverseLinearOperator(self._operator.transpose(), pc=self._pc, tol=self._tol, maxiter=self._maxiter, solver_name=self._solver_name)
        else:
            return InverseLinearOperator(self._operator.transpose(), pc=new_pc, tol=self._tol, maxiter=self._maxiter, solver_name=self._solver_name)
        
    
class BoundaryOperator(LinOpWithTransp):
    r"""
    Applies homogeneous Dirichlet boundary conditions to a vector.
    
    Parameters
    ----------
        vector_space : StencilVectorSpace | BlockVectorSpace | PolarDerhamSpace
            The vector space associated to the operator.
            
        space_id : str
            Symbolic space ID of vector_space (H1, Hcurl, Hdiv, L2 or H1vec).
            
        bc : list
            Boundary conditions in each direction in format [[bc_e1=0, bc_e1=1], [bc_e2=0, bc_e2=1], [bc_e3=0, bc_e3=1]].
    """
    
    def __init__(self, vector_space, space_id, bc):

        self._domain = vector_space
        self._codomain = vector_space
        self._dtype = vector_space.dtype
        
        self._space_id = space_id
        self._bc = bc
        
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
    def bc(self):
        return self._bc

    def dot(self, v, out=None):
        """
        TODO
        """
        
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.domain
        
        if out is None:
            out = v.copy()
            return_flag = True
        else:
            assert type(out) == type(v)
            return_flag = False
        
        # apply boundary conditions to output vector
        apply_essential_bc_to_array(self._space_id, out, self._bc)
        
        if return_flag:
            return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BoundaryOperator(self._domain, self._space_id, self._bc)
    
    
class IdentityOperator(LinOpWithTransp):
    r"""
    Identity operation applied to a vector in a certain vector space.
    
    Parameters
    ----------
        vector_space : StencilVectorSpace | BlockVectorSpace | PolarDerhamSpace
            The vector space associated to the operator.
    """
    
    def __init__(self, vector_space):

        self._domain = vector_space
        self._codomain = vector_space
        self._dtype = vector_space.dtype
        
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
        """
        TODO
        """
        
        assert isinstance(v, (StencilVector, BlockVector, PolarVector))
        assert v.space == self.domain
        
        if out is None:
            out = v.copy()
            return out
        else:
            assert type(out) == type(v)

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return IdentityOperator(self._domain)