def evaluate_fun_weights_1d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 1.

    Parameters
    ----------
    pts : 1-tuple of 2d float arrays
        Quadrature points in each dimension in format (element, quadrature point).
    
    wts : 1-tuple of 2d float arrays
        Quadrature weights in each dimension in format (element, quadrature point).
    
    fun : callable
        The function which shall be evaluated at eta1.
    
    Returns
    -------
    A 2d array (1 cell grid dimension, 1 quadratue point dimension)
    which contains all the pre-evaluated values.
    """

    import numpy as np

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[0].shape[1]), dtype=float)
                      
    for i in range(pts[0].shape[0]): # element index
        for iq in range(pts[0].shape[1]): # quadrature point index
            values[i, iq] = fun(pts[0][i, iq]) *  wts[0][i, iq]
            
    return values


def evaluate_fun_weights_2d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 2.

    Parameters
    ----------
    pts : 2-tuple of 2d float arrays
        Quadrature points in each dimension in format (element, quadrature point).
    
    wts : 2-tuple of 2d float arrays
        Quadrature weights in each dimension in format (element, quadrature point).
    
    fun : callable
        The function which shall be evaluated at eta1, eta2.
    
    Returns
    -------
    A 4d array (2 cell grid dimensions, 2 quadratue point dimensions)
    which contains all the pre-evaluated values.
    """

    import numpy as np

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[1].shape[0], 
                       pts[0].shape[1], pts[1].shape[1]), dtype=float)
    
    for i in range(pts[0].shape[0]): # element index
        for j in range(pts[1].shape[0]):
            for iq in range(pts[0].shape[1]): # quadrature point index
                for jq in range(pts[1].shape[1]):
                        funval = fun(pts[0][i, iq], pts[1][j, jq])
                        weightval = wts[0][i, iq] * wts[1][j, jq]
                        values[i, j, iq, jq] = weightval * funval
                            
    return values


def evaluate_fun_weights_3d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 3.

    Parameters
    ----------
    pts : 3-tuple of 2d float arrays
        Quadrature points in each dimension in format (element, quadrature point).
    
    wts : 3-tuple of 2d float arrays
        Quadrature weights in each dimension in format (element, quadrature point).
    
    fun : callable
        The function which shall be evaluated at eta1, eta2, eta3.
    
    Returns
    -------
    A 6d array (3 cell grid dimensions, 3 quadratue point dimensions)
    which contains all the pre-evaluated values.
    """

    import numpy as np

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[1].shape[0], pts[2].shape[0], 
                       pts[0].shape[1], pts[1].shape[1], pts[2].shape[1]), dtype=float)
    
    for i in range(pts[0].shape[0]): # element index
        for j in range(pts[1].shape[0]):
            for k in range(pts[2].shape[0]):
                for iq in range(pts[0].shape[1]): # quadrature point index
                    for jq in range(pts[1].shape[1]):
                        for kq in range(pts[2].shape[1]):
                            funval = fun(pts[0][i, iq], pts[1][j, jq], pts[2][k, kq])
                            weightval = wts[0][i, iq] * wts[1][j, jq] * wts[2][k, kq]
                            values[i, j, k, iq, jq, kq] = weightval * funval
                            
    return values


def assemble_funccache_numpy(u, w, func):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version tries to use numpy where possible, and is usable in arbitrary dimensions.

    Parameters
    ----------
    u : three-tuple of two-dimensional numpy arrays
        The quadrature points in each dimension.
    
    w : three-tuple of two-dimensional numpy arrays
        The quadrature weights in each dimension for the respective points.
    
    func : callable, with three parameters
        The function which shall be evaluated.
    
    Returns
    -------
    A 6-dimensional array (3 cell grid dimensions, 3 quadratue point dimensions)
    which contains all the pre-evaluated values.
    """

    import numpy as np

    funcvec = np.vectorize(func)
    grid = np.meshgrid(*u, sparse=True, indexing='ij')
    funceval = funcvec(*grid)

    for wg in np.meshgrid(*w, sparse=True, indexing='ij'):
        funceval *= wg
    
    funceval.shape = tuple(uxx for ux in u for uxx in ux.shape)

    n = len(u)
    return funceval.transpose([2*i for i in range(n)] + [2*i+1 for i in range(n)])


