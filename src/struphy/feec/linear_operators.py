from abc import abstractmethod

from mpi4py import MPI
import numpy as np
from scipy import sparse

from psydac.linalg.basic import Vector, VectorSpace, LinearOperator, LinearSolver
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.linalg.kron import KroneckerStencilMatrix

from struphy.feec.utilities import apply_essential_bc_to_array
from struphy.polar.basic import PolarDerhamSpace

import struphy.linear_algebra.iterative_solvers as it_solvers


class LinOpWithTransp(LinearOperator):
    """
    Base class for linear operators that MUST implement a transpose method which returns a new transposed operator.
    """

    @abstractmethod
    def transpose(self):
        pass

    # Function that returns the matrix corresponding to the linear operator. Returns a numpy array.
    # At the moment only works in one processor.
    def toarray_struphy(self, out=None):
        """
        Transforms the linear operator into a matrix.

        Parameters
        ----------
        out : Numpy.ndarray, optional
            If given, the output will be written in-place into this array.

        Returns
        -------
        out : Numpy.ndarray
            The matrix form of the linear operator.
        """
        # v will be the unit vector with which we compute Av = ith column of A.
        v = self.domain.zeros()

        # For the time being only works in 1 processor
        if isinstance(self.domain, BlockVectorSpace):
            comm = self.domain.spaces[0].cart.comm
        elif isinstance(self.domain, StencilVectorSpace):
            comm = self.domain.cart.comm
        assert comm.size == 1

        if out is None:
            # We declare the matrix form of our linear operator
            out = np.zeros(
                [self.codomain.dimension, self.domain.dimension], dtype=self.dtype)
        else:
            assert isinstance(out, np.ndarray)
            assert out.shape[0] == self.codomain.dimension
            assert out.shape[1] == self.domain.dimension
        # This auxiliary counter allows us to know which column of A we are computing in the following for loops.
        cont = 0

        # We define a temporal vector
        tmp2 = self.codomain.zeros()

        # V is either a BlockVector or a StencilVector depending on the domain of the linear operator.
        if isinstance(self.domain, BlockVectorSpace):
            # we collect all starts and ends in two big lists
            starts = [vi.starts for vi in v]
            ends = [vi.ends for vi in v]

            # We iterate over each entry of the block vector v, setting one entry to one at the time while all others remain zero.
            for vv, ss, ee in zip(v, starts, ends):
                for i in range(ss[0], ee[0]+1):
                    for j in range(ss[1], ee[1]+1):
                        for k in range(ss[2], ee[2]+1):
                            vv[i, j, k] = 1.0
                            # Compute dot product with the linear operator
                            self.dot(v, out=tmp2)
                            # We set the column number cont of our matrix to the dot product of the linear operator with the unit vector
                            # number cont
                            out[:, cont] = tmp2.copy().toarray()
                            vv[i, j, k] = 0.0
                            cont += 1
        elif isinstance(self.domain, StencilVectorSpace):
            # We get the start and endpoint for each sublist in v
            starts = v.starts
            ends = v.ends
            # We iterate over each entry of the stencil vector v, setting one entry to one at the time while all others remain zero.
            for i in range(starts[0], ends[0]+1):
                for j in range(starts[1], ends[1]+1):
                    for k in range(starts[2], ends[2]+1):
                        v[i, j, k] = 1.0
                        # Compute dot product with the linear operator.
                        self.dot(v, out=tmp2)
                        # We set the column number cont of our matrix to the dot product of the linear operator with the unit vector
                        # number cont
                        out[:, cont] = tmp2.copy().toarray()
                        v[i, j, k] = 0.0
                        cont += 1
        else:
            # I cannot conceive any situation where this error should be thrown, but I put it here just in case something unexpected happens.
            raise Exception(
                'Function toarray_struphy() only supports Stencil Vectors or Block Vectors.')

        return out
    
    # Function that returns the sparse matrix corresponding to the linear operator. Returns a scipy sparse matrix.
    # At the moment only works in one processor.
    def tosparse_struphy(self, format = "csr"):
        """
        Transforms the linear operator into a Scipy sparse matrix.

        Parameters
        ----------
        format : string, optional
            Specifies the format in which the sparse matrix is to be stored. Choose from "csr" (Compressed Sparse Row, default),
            "csc" (Compressed Sparse Column), "bsr" (Block Sparse Row ), "lil" (List of Lists), "dok" (Dictionary of Keys), 
            "coo" (COOrdinate format) and "dia" (DIAgonal).

        Returns
        -------
        out : scipy.sparse.csr.csr_matrix
            The sparse matrix form of the linear operator in the specified format.
        """
        
        # v will be the unit vector with which we compute Av = ith column of A.
        v = self.domain.zeros()

        # For the time being only works in 1 processor
        if isinstance(v, BlockVector):
            comm = self.domain.spaces[0].cart.comm
        elif isinstance(v, StencilVector):
            comm = self.domain.cart.comm
        assert comm.size == 1

        #We define a list to store the non-zero data, a list to sotre the row index of said data and a list to store the column index.
        data = []
        row = []
        col = []
        
        # This auxiliary counter allows us to know which column of A we are computing in the following for loops.
        cont = 0

        # We define a temporal vector
        tmp2 = self.codomain.zeros()
        # We define the 1D numpy array version of this vector
        aux = tmp2.toarray()
        #We define the number of rows for our matrix
        numrows = len(aux)

        # V is either a BlockVector or a StencilVector depending on the domain of the linear operator.
        if isinstance(v, BlockVector):
            # we collect all starts and ends in two big lists
            starts = [vi.starts for vi in v]
            ends = [vi.ends for vi in v]

            # We iterate over each entry of the block vector v, setting one entry to one at the time while all others remain zero.
            for vv, ss, ee in zip(v, starts, ends):
                for i in range(ss[0], ee[0]+1):
                    for j in range(ss[1], ee[1]+1):
                        for k in range(ss[2], ee[2]+1):
                            vv[i, j, k] = 1.0
                            # Compute dot product with the linear operator
                            self.dot(v, out=tmp2)
                            aux = tmp2.toarray()
                            # We now need to now which entries on tmp2 are non-zero and store then in our data list
                            for l in np.where(aux !=0)[0]:
                                data.append(aux[l])
                                col.append(cont)
                                row.append(l) 
                            vv[i, j, k] = 0.0
                            cont += 1
        elif isinstance(v, StencilVector):
            # We get the start and endpoint for each sublist in v
            starts = v.starts
            ends = v.ends
            # We iterate over each entry of the stencil vector v, setting one entry to one at the time while all others remain zero.
            for i in range(starts[0], ends[0]+1):
                for j in range(starts[1], ends[1]+1):
                    for k in range(starts[2], ends[2]+1):
                        v[i, j, k] = 1.0
                        # Compute dot product with the linear operator.
                        self.dot(v, out=tmp2)
                        aux = tmp2.toarray()
                        # We now need to now which entries on tmp2 are non-zero and store then in our data list
                        for l in np.where(aux !=0)[0]:
                            data.append(aux[l])
                            col.append(cont)
                            row.append(l) 
                        v[i, j, k] = 0.0
                        cont += 1
        else:
            # I cannot conceive any situation where this error should be thrown, but I put it here just in case something unexpected happens.
            raise Exception(
                'Function toarray_struphy() only supports Stencil Vectors or Block Vectors.')
        
        
        if format == "csr":
            return sparse.csr_matrix((data, (row, col)), shape=(numrows, cont))
        elif format == "csc":
            return sparse.csc_matrix((data, (row, col)), shape=(numrows, cont))
        elif format == "bsr":
            return sparse.bsr_matrix((data, (row, col)), shape=(numrows, cont))
        elif format == "lil":
            return sparse.csr_matrix((data, (row, col)), shape=(numrows, cont)).tolil()
        elif format == "dok":
            return sparse.csr_matrix((data, (row, col)), shape=(numrows, cont)).todok()
        elif format == "coo":
            return sparse.coo_matrix((data, (row, col)), shape=(numrows, cont))
        elif format == "dia":
            return sparse.csr_matrix((data, (row, col)), shape=(numrows, cont)).todia()
        else:
            raise Exception(
                'The selected sparse matrix format must be one of the following : csr, csc, bsr, lil, dok,  coo or dia.')
        
        
        


class CompositeLinearOperator(LinOpWithTransp):
    r"""
    Composition of n linear operators: :math:`A(\mathbf v)=L_n(L_{n-1}(...L_2(L_1(\mathbf v))...)`.
    A 'None' operator is treated as identity.

    Parameters
    ----------
    operators: LinOpWithTransp | StencilMatrix | BlockLinearOperator | None
        The sequence of n linear operators (None is treated as identity).
    """

    def __init__(self, *operators):

        self._operators = [op for op in list(
            operators)[::-1] if op is not None]

        if len(self._operators) > 1:

            for op2, op1 in zip(self._operators[1:], self._operators[:-1]):
                assert isinstance(op1, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                        BlockLinearOperator, KroneckerStencilMatrix))
                assert isinstance(op2, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                        BlockLinearOperator, KroneckerStencilMatrix))
                assert op2.domain == op1.codomain

        self._domain = self._operators[0].domain
        self._codomain = self._operators[-1].codomain
        self._dtype = self._operators[-1].dtype

        # temporary vectors for dot product
        tmp_vectors = []
        for op in self._operators[:-1]:
            tmp_vectors.append(op.codomain.zeros())

        self._tmp_vectors = tuple(tmp_vectors)

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def otype(self):
        return [type(op) for op in self._operators]

    @property
    def operators(self):
        return self._operators

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        # successive dot products with all but last operator
        x = v
        for i in range(len(self._tmp_vectors)):
            y = self._tmp_vectors[i]
            A = self._operators[i]
            A.dot(x, out=y)
            x = y

        # last operator
        A = self._operators[-1]
        if out is None:
            out = A.dot(x)
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            A.dot(x, out=out)

        return out

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return CompositeLinearOperator(*[op.transpose() for op in self._operators])


class ScalarTimesLinearOperator(LinOpWithTransp):
    r"""
    Multiplication of a linear operator with a scalar: :math:`A(\mathbf v)=aL(\mathbf v)` with :math:`a \in \mathbb R`.

    Parameters
    ----------
    a : float
        The scalar that is multiplied with the linear operator. 

    operator: LinOpWithTransp | StencilMatrix | BlockLinearOperator
        The linear operator.
    """

    def __init__(self, a, operator):

        assert isinstance(a, (int, float, complex))
        assert isinstance(operator, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                     BlockLinearOperator, KroneckerStencilMatrix))

        self._a = a
        self._operator = operator

        self._domain = operator.domain
        self._codomain = operator.codomain
        self._dtype = operator.dtype

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def otype(self):
        return type(self._operator)

    def dot(self, v, out=None):
        """
        Dot product of operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        if out is None:
            out = self._operator.dot(v)
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            self._operator.dot(v, out=out)

        out *= self._a

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return ScalarTimesLinearOperator(self._a, self._operator.transpose())


class SumLinearOperator(LinOpWithTransp):
    r"""
    Sum of n linear operators: :math:`A(\mathbf v)=(L_n + L_{n-1} + ... + L_2 + L_1)(\mathbf v)`.

    Parameters
    ----------
    operators: LinOpWithTransp | StencilMatrix | BlockLinearOperator
        The sequence of n linear operators.
    """

    def __init__(self, *operators):

        self._operators = list(operators)

        assert len(self._operators) > 1

        for op2, op1 in zip(self._operators[::-1][1:], self._operators[::-1][:-1]):
            assert isinstance(op1, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                    BlockLinearOperator, KroneckerStencilMatrix))
            assert isinstance(op2, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                    BlockLinearOperator, KroneckerStencilMatrix))
            assert op2.domain == op1.domain
            assert op2.codomain == op1.codomain

        self._domain = operators[0].domain
        self._codomain = operators[0].codomain
        self._dtype = operators[0].dtype

        # temporary vectors for summing
        self._tmp1 = self._codomain.zeros()
        self._tmp2 = self._codomain.zeros()

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def otype(self):
        return [type(op) for op in self._operators]

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        # reset array
        self._tmp1 *= 0.

        for op in self._operators:
            op.dot(v, out=self._tmp2)
            self._tmp1 += self._tmp2

        if out is None:
            out = self._tmp1.copy()
        else:
            assert isinstance(v, Vector)
            assert out.space == self._codomain
            self._tmp1.copy(out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return SumLinearOperator(*[op.transpose() for op in self._operators])


class InverseLinearOperator(LinOpWithTransp):
    r"""
    Inverse linear operator: :math:`A(\mathbf v)=L^{-1}(\mathbf v)`.

    Parameters
    ----------
    operator : LinOpWithTransp | StencilMatrix | BlockLinearOperator
        The linear operator to be inverted.

    pc : NoneType | psydac.linalg.basic.LinearOperator
         Preconditioner for "operator", it should approximate the inverse of "operator". Must have a dot method.

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter : int
        Maximum number of iterations.

    solver_name : str
        The name of the iterative solver to be used for matrix inversion. Currently available:
            * ConjugateGradient (default)
            * BiConjugateGradientStab
    """

    def __init__(self, operator, pc=None, tol=1e-6, maxiter=1000, solver_name='ConjugateGradient'):

        assert isinstance(operator, (LinearOperator, LinOpWithTransp, StencilMatrix,
                                     BlockLinearOperator, KroneckerStencilMatrix))

        # only square matrices possible
        assert operator.domain == operator.codomain

        if pc is not None:
            assert isinstance(pc, LinearOperator)

        self._domain = operator.domain
        self._codomain = operator.codomain
        self._dtype = operator.dtype

        self._operator = operator
        self._pc = pc
        self._tol = tol
        self._maxiter = maxiter

        # load linear solver (if pc is given, load pre-conditioned solver)
        self._solver_name = solver_name
        if pc is None:
            self._solver = getattr(it_solvers, solver_name)(operator.domain)
        else:
            self._solver = getattr(
                it_solvers, 'P' + self._solver_name)(operator.domain)
        self._info = None

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def otype(self):
        return type(self._operator)

    @property
    def info(self):
        return self._info

    def dot(self, v, out=None, x0=None, verbose=False):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        x0 : psydac.linalg.basic.Vector, optional
            Initial guess for the output vector.

        verbose : bool
            Whether to print information about the residual norm in each iteration step.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.codomain

        # solve linear system (in-place if out is not None)

        # solvers with preconditioner (must start with a 'P')
        if self._pc is not None:
            x, self._info = self._solver.solve(self._operator, v,
                                               self._pc, x0=x0, tol=self._tol,
                                               maxiter=self._maxiter,
                                               verbose=verbose, out=out)

        # solvers without preconditioner
        else:
            x, self._info = self._solver.solve(self._operator, v,
                                               x0=x0, tol=self._tol,
                                               maxiter=self._maxiter,
                                               verbose=verbose, out=out)

        return x

    def transpose(self, new_pc=None):
        """
        Returns the transposed operator.
        """
        if new_pc is None:
            return InverseLinearOperator(self._operator.T, pc=self._pc,
                                         tol=self._tol, maxiter=self._maxiter,
                                         solver_name=self._solver_name)
        else:
            return InverseLinearOperator(self._operator.T, pc=new_pc,
                                         tol=self._tol, maxiter=self._maxiter,
                                         solver_name=self._solver_name)


class BoundaryOperator(LinOpWithTransp):
    r"""
    Applies homogeneous Dirichlet boundary conditions to a vector.

    Parameters
    ----------
    vector_space : psydac.linalg.basic.VectorSpace
        The vector space associated to the operator.

    space_id : str
        Symbolic space ID of vector_space (H1, Hcurl, Hdiv, L2 or H1vec).

    dirichlet_bc : list[list[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).
    """

    def __init__(self, vector_space, space_id, dirichlet_bc):

        assert isinstance(vector_space, VectorSpace)
        assert isinstance(space_id, str)

        self._domain = vector_space
        self._codomain = vector_space
        self._dtype = vector_space.dtype

        self._space_id = space_id
        self._bc = dirichlet_bc

        assert isinstance(dirichlet_bc, list)
        assert len(dirichlet_bc) == 3

        # number of non-zero elements in poloidal/toroidal direction
        if isinstance(vector_space, PolarDerhamSpace):
            vec_space_ten = vector_space.parent_space
        else:
            vec_space_ten = vector_space

        if isinstance(vec_space_ten, StencilVectorSpace):
            n_pts = vec_space_ten.npts
        else:
            n_pts = [comp.npts for comp in vec_space_ten.spaces]

        dim_nz1_pol = 1
        dim_nz2_pol = 1
        dim_nz3_pol = 1

        dim_nz1_tor = 1
        dim_nz2_tor = 1
        dim_nz3_tor = 1

        if space_id == 'H1':

            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0] -
                                self.bc[0][1])*n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol*dim_nz1_tor,)

        elif space_id == 'Hcurl':

            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] -
                                vector_space.n_rings[0])*n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] - vector_space.n_rings[1] -
                                self.bc[0][1])*n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] - vector_space.n_rings[2] -
                                self.bc[0][1])*n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0]
                dim_nz1_pol *= n_pts[0][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz2_pol *= n_pts[1][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz2_pol *= n_pts[1][1]

                dim_nz3_pol *= n_pts[2][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz3_pol *= n_pts[2][1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[0][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz2_tor *= n_pts[1][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz3_tor *= n_pts[2][2]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol*dim_nz1_tor,
                            dim_nz2_pol*dim_nz2_tor,
                            dim_nz3_pol*dim_nz3_tor)

        elif space_id == 'Hdiv' or space_id == 'H1vec':

            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] - vector_space.n_rings[0] -
                                self.bc[0][1])*n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] -
                                vector_space.n_rings[1])*n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] -
                                vector_space.n_rings[2])*n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[0][1]

                dim_nz2_pol *= n_pts[1][0]
                dim_nz2_pol *= n_pts[1][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz3_pol *= n_pts[2][0]
                dim_nz3_pol *= n_pts[2][1]

            dim_nz1_tor *= n_pts[0][2]
            dim_nz2_tor *= n_pts[1][2]
            dim_nz3_tor *= n_pts[2][2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol*dim_nz1_tor,
                            dim_nz2_pol*dim_nz2_tor,
                            dim_nz3_pol*dim_nz3_tor)

        else:

            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0])*n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0]
                dim_nz1_pol *= n_pts[1]

            dim_nz1_tor *= n_pts[2]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol*dim_nz1_tor,)

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def bc(self):
        return self._bc

    @property
    def dim_nz_pol(self):
        return self._dim_nz_pol

    @property
    def dim_nz_tor(self):
        return self._dim_nz_tor

    @property
    def dim_nz(self):
        return self._dim_nz

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        if out is None:
            out = v.copy()
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            v.copy(out=out)

        # apply boundary conditions to output vector
        apply_essential_bc_to_array(self._space_id, out, self.bc)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BoundaryOperator(self._domain, self._space_id, self.bc)


class IdentityOperator(LinOpWithTransp):
    r"""
    Identity operation applied to a vector in a certain vector space.

    Parameters
    ----------
    vector_space : psydac.linalg.basic.VectorSpace
        The vector space associated to the operator.
    """

    def __init__(self, vector_space):

        assert isinstance(vector_space, VectorSpace)

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

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        if out is None:
            out = v.copy()
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            v.copy(out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return IdentityOperator(self._domain)
