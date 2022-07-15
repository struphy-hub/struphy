# coding: utf-8
"""
This module provides iterative solvers with and without preconditioners.

"""
import numpy as np

from math import sqrt
from psydac.linalg.basic     import LinearSolver, LinearOperator
from psydac.linalg.utilities import _sym_ortho


__all__ = ['bicgstab', 'pbicgstab']


# ...
def bicgstab(A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Stabilized biconjugate gradient (BiCGSTAB) algorithm for solving linear system Ax=b.
    Implementation from [1], page 208.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    b : psydac.linalg.basic.Vector
        Right-hand-side vector of linear system. Individual entries b[i] need
        not be accessed, but b has 'shape' attribute and provides 'copy()' and
        'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
        scalar multiplication and sum operations are available.

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

    Returns
    -------
    xj : psydac.linalg.basic.Vector
        Numerical solution of linear system.

    info : dict
        Dictionary containing convergence information:
          - 'niter'    = (int) number of iterations
          - 'success'  = (boolean) whether convergence criteria have been met
          - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    
    n = A.shape[0]

    assert A .shape == (n, n)
    assert b .shape == (n,)

    # First guess of solution
    if x0 is None:
        xj = 0.0 * b.copy()
    else:
        assert x0.shape == (n,)
        xj = x0.copy()

    # First values (j=0)
    rj = b - A.dot( xj )
    pj = rj.copy()
    
    rhoj = rj.dot( rj )
    
    # save initial residual vector
    r0 = rj.copy()
    
    # results of needed matrix-vector products
    vj = 0.0 * b.copy()
    tj = 0.0 * b.copy()

    res_sqr = 1*rhoj
    tol_sqr = tol**2

    if verbose:
        print( "BiCGSTAB solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for j in range(maxiter):

        if res_sqr < tol_sqr:
            break
            
        vj = A.dot(pj, out=vj)
        
        alphaj = rhoj / vj.dot( r0 ) 
        
        sj = rj - alphaj*vj
        
        tj = A.dot(sj, out=tj)
        
        omegaj = tj.dot( sj ) / tj.dot( tj )
        
        xj += alphaj*pj + omegaj*sj
        
        rj = sj - omegaj*tj
        
        rhoj1 = rj.dot( r0 )
        
        betaj = (alphaj* rhoj1)/ (omegaj*rhoj)
        
        pj = rj + betaj*(pj - omegaj*vj)
        
        rhoj = rhoj1
        
        res_sqr = rj.dot( rj )
        

        if verbose:
            print( template.format(m, sqrt(res_sqr)) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': j, 'success': res_sqr < tol_sqr, 'res_norm': sqrt( res_sqr ) }

    return xj, info
# ...



# ...
def pbicgstab(A, b, pc, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Preconditioned stabilized biconjugate gradient (BiCGSTAB) algorithm for solving linear system Ax=b.
    Implementation from [1], page 251.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    b : psydac.linalg.basic.Vector
        Right-hand-side vector of linear system. Individual entries b[i] need
        not be accessed, but b has 'shape' attribute and provides 'copy()' and
        'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
        scalar multiplication and sum operations are available.
        
    pc: NoneType | str | psydac.linalg.basic.LinearSolver | Callable
        Preconditioner for A, it should approximate the inverse of A.
        Can either be:
        * None, i.e. not pre-conditioning (this calls the standard `cg` method)
        * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)
        * A LinearSolver object (in which case the out parameter is used)
        * A callable with two parameters (A, r), where A is the LinearOperator from above, and r is the residual.

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

    Returns
    -------
    xj : psydac.linalg.basic.Vector
        Numerical solution of linear system.

    info : dict
        Dictionary containing convergence information:
          - 'niter'    = (int) number of iterations
          - 'success'  = (boolean) whether convergence criteria have been met
          - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    
    n = A.shape[0]

    assert A .shape == (n, n)
    assert b .shape == (n,)

    # First guess of solution
    if x0 is None:
        xj = 0.0 * b.copy()
    else:
        assert x0.shape == (n,)
        xj = x0.copy()
        
    # Preconditioner
    if pc is None:
        # for now, call the bicgstab method here
        return bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter, verbose=verbose)
    elif isinstance(pc, str):
        pcfun = globals()[pc]
        psolve = lambda r: pcfun(A, r)
    elif isinstance(pc, LinearSolver):
        s = b.space.zeros()
        psolve = lambda r: pc.solve(r, out=s)
    elif hasattr(pc, '__call__'):
        psolve = lambda r: pc(A, r)

    # First values (j=0)
    rj = b - A.dot( xj )
    pj = rj.copy()
    
    s = psolve(rj)
    ppj = s.copy()
    
    rhopj = s.dot( s )
    
    # save initial residual vector
    rp0 = s.copy()
    
    # results of needed matrix-vector products
    vj = 0.0 * b.copy()
    tj = 0.0 * b.copy()

    res_sqr = rj.dot( rj )
    tol_sqr = tol**2

    if verbose:
        print( "Preconditioned BiCGSTAB solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for j in range(maxiter):

        if res_sqr < tol_sqr:
            break
            
        vj = A.dot(ppj, out=vj)
        s = psolve(vj)
        vpj = 1*s
        
        alphapj = rhopj / vpj.dot( rp0 ) 
        
        sj = rj - alphapj*vj
        s = psolve(sj)
        spj = 1*s
        
        tj = A.dot(spj, out=tj)
        s = psolve(tj)
        tpj = 1*s
        
        omegapj = tpj.dot( spj ) / tpj.dot( tpj )
        
        xj += alphapj*ppj + omegapj*spj
        
        rj = sj - omegapj*tj
        rpj = spj - omegapj*tpj
        
        rhopj1 = rpj.dot( rp0 )
        
        betapj = (alphapj* rhopj1)/ (omegapj*rhopj)
        
        ppj = rpj + betapj*(ppj - omegapj*vpj)
        
        rhopj = rhopj1
        
        res_sqr = rj.dot( rj )
        

        if verbose:
            print( template.format(m, sqrt(res_sqr)) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': j, 'success': res_sqr < tol_sqr, 'res_norm': sqrt( res_sqr ) }

    return xj, info
# ...