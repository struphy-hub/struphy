#==============================================================================
@external_call
@types('double[:]','double[:]','int','double[:]','bool','bool')
def collocation_matrix( knots, el_b, degree, xgrid, periodic, normalize):
    """
    Compute the collocation matrix $C_ij = B_j(x_i)$, which contains the
    values of each B-spline basis function $B_j$ at all locations $x_i$.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.
        
    el_b : 1D array like
        Breakpoints.

    degree : int
        Polynomial degree of B-splines.

    xgrid : 1D array_like
        Evaluation points.

    periodic : bool
        True if domain is periodic, False otherwise.

    Returns
    -------
    mat : 2D numpy.ndarray
        Collocation matrix: values of all basis functions on each point in xgrid.

    """
    
    from numpy import empty
    from numpy import zeros
    from numpy import ones
    
    Nl = empty(degree,     dtype=float)
    Nr = empty(degree,     dtype=float)
    N  = zeros(degree + 1, dtype=float)
    
    span = 0
    
    ne = len(el_b) - 1 
    
    if normalize:
        x_norm = zeros((ne, degree + 1))
        
        for ie in range(ne):
            for il in range(degree + 1):
                x_norm[ie, il] = (degree + 1)/(knots[ie + il + degree + 1] - knots[ie + il])
            
    else:
        x_norm = ones((ne, degree + 1))
        
    
    # Number of basis functions (in periodic case remove degree repeated elements)
    nb = len(knots) - degree - 1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)


    # Collocation matrix as 2D Numpy array (dense storage)
    mat = zeros((nx, nb))
    
    # Fill in non-zero matrix values
    for i,x in enumerate( xgrid ):
        span  =  find_span( knots, degree, x )
        basis_funs(knots, degree, x, span, Nl, Nr, N) * x_norm[span - degree]
        
        for jl in range(degree + 1):
            
            j = (span - jl)%nb
            mat[i, j] = N[degree - jl]

    
    
    return mat