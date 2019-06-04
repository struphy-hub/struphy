import numpy as np
import psydac.core.interface as inter
import psydac.core.bsplines as bsp
import scipy.sparse as sparse



def integrate_1d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 1d.
    
    Parameters
    ----------
    points : 2d np.array
        quadrature points in format (local point, element)
        
    weights : 2d np.array
        quadrature weights in format (local point, element)
    
    fun : callable
        1d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    k = points.shape[0]
    n = points.shape[1]
    
    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[g, ie]*fun(points[g, ie])
        
    return f_int



def integrate_2d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 2d.
    
    Parameters
    ----------
    points : list of 2d np.arrays
        quadrature points in format (local point, element)
        
    weights : list of 2d np.arrays
        quadrature weights in format (local point, element)
    
    fun : callable
        2d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    pts_0, pts_1 = points
    wts_0, wts_1 = weights
    
    k0 = pts_0.shape[0]
    k1 = pts_1.shape[0]
    n0 = pts_0.shape[1]
    n1 = pts_1.shape[1]
    
    
    f_int = np.zeros((n0, n1))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for g_0 in range(k0):
                for g_1 in range(k1):
                    f_int[ie_0, ie_1] += wts_0[g_0, ie_0]*wts_1[g_1, ie_1]*fun(pts_0[g_0, ie_0], pts_1[g_1, ie_1])
                     
    return f_int


def integrate_3d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 3d.
    
    Parameters
    ----------
    points : list of 2d np.arrays
        quadrature points in format (local point, element)
        
    weights : list of 2d np.arrays
        quadrature weights in format (local point, element)
    
    fun : callable
        3d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    pts_0, pts_1, pts_2 = points
    wts_0, wts_1, wts_2 = weights
    
    
    k0 = pts_0.shape[0]
    k1 = pts_1.shape[0]
    k2 = pts_2.shape[0]
    n0 = pts_0.shape[1]
    n1 = pts_1.shape[1]
    n2 = pts_2.shape[1]
    
    f_int = np.zeros((n0, n1, n2))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for ie_2 in range(n2):
                for g_0 in range(k0):
                    for g_1 in range(k1):
                        for g_2 in range(k2):
                            f_int[ie_0, ie_1, ie_2] += wts_0[g_0, ie_0]*wts_1[g_1, ie_1]*wts_2[g_2, ie_2]*fun(pts_0[g_0, ie_0], pts_1[g_1, ie_1], pts_2[g_2, ie_2])
                     
    return f_int



def L2_prod_V0_1d(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V0
    using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1 
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p + 1, d, T, pts)
    
    f_int = np.zeros(Nbase)
    
    for ie in range(ne):
        for il in range(p + 1):
            i = ie + il
            
            value = 0.
            for g in range(p + 1):
                value += wts[g, ie]*fun(pts[g, ie])*basis[il, 0, g, ie]
                
            f_int[i] += value
            
    return f_int
    
    
    
    
def L2_prod_V1_1d(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1
    using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1 
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)
    
    t = T[1:-1]
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p - 1, Nbase - 1, p + 1, d, t, pts)
    
    f_int = np.zeros(Nbase - 1)
    
    for ie in range(ne):
        for il in range(p):
            i = ie + il
            
            value = 0.
            for g in range(p + 1):
                value += wts[g, ie]*fun(pts[g, ie])*basis[il, 0, g, ie]
                
            f_int[i] += value*p/(t[i + p] - t[i])
            
    return f_int



def L2_prod_V0(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V0
    using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
        
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, d, Tz, pts_z)
    
    f_int = np.zeros((Nbase_x, Nbase_y, Nbase_z))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px + 1):
                    for il_y in range(py + 1):
                        for il_z in range(pz + 1):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value
    
    return f_int
    
    
        
def L2_prod_V1_x(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1 (x-component)
    using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, d, Tz, pts_z)
    
    f_int = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px):
                    for il_y in range(py + 1):
                        for il_z in range(pz + 1):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])
    
    return f_int

    
        
def L2_prod_V1_y(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1 (y-component)
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    ty = Ty[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, d, Tz, pts_z)
    
    f_int = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px + 1):
                    for il_y in range(py):
                        for il_z in range(pz + 1):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*py/(ty[iy + py] - ty[iy])
    
    return f_int
    
    
    
def L2_prod_V1_z(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1 (z-component)
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, d, tz, pts_z)
    
    f_int = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px + 1):
                    for il_y in range(py + 1):
                        for il_z in range(pz):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    
    
        
def L2_prod_V2_x(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V2 (x-component)
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, d, tz, pts_z)
    
    f_int = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px + 1):
                    for il_y in range(py):
                        for il_z in range(pz):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*py/(ty[iy + py] - ty[iy])*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    

    
def L2_prod_V2_y(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V2 (y-component)
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, d, tz, pts_z)
    
    f_int = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px):
                    for il_y in range(py + 1):
                        for il_z in range(pz):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    
 
 
def L2_prod_V2_z(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V2 (z-component)
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, d, Tz, pts_z)
    
    f_int = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px):
                    for il_y in range(py):
                        for il_z in range(pz + 1):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])*py/(ty[iy + py] - ty[iy])
    
    return f_int


def L2_prod_V3(fun, p, Nbase, T):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V3
    using a quadrature rule of order p.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    ne_x = len(el_b_x) - 1
    ne_y = len(el_b_y) - 1
    ne_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, d, tz, pts_z)
    
    f_int = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z - 1))
    
    for ie_x in range(ne_x):
        for ie_y in range(ne_y):
            for ie_z in range(ne_z):
                for il_x in range(px):
                    for il_y in range(py):
                        for il_z in range(pz):
                            
                            ix = ie_x + il_x
                            iy = ie_y + il_y
                            iz = ie_z + il_z
                            
                            value = 0.
                            for g_x in range(px + 1):
                                for g_y in range(py + 1):
                                    for g_z in range(pz + 1):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += px/(tx[ix + px] - tx[ix])*py/(ty[iy + py] - ty[iy])*pz/(tz[iz + pz] - tz[iz])*value
                            
    return f_int



def histopolation_matrix_1d(p, Nbase, T, grev, bc):
    """
    Computest the 1d histopolation matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
    
    T : np.array 
        knot vector
    
    grev : np.array
        greville points
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    D : 2d np.array
        histopolation matrix
    """
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    Nel = len(el_b) - 1
    
    if bc == False:
        return inter.histopolation_matrix(p, Nbase, T, grev)
    
    else:
        if p%2 != 0:
            dx = el_b[-1]/Nel
            ne = Nbase - 1
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            pts, wts = inter.construct_quadrature_grid(ne, p - 1, pts_loc, wts_loc, grev)

            col_quad = inter.collocation_matrix(p - 1, ne, T[1:-1], (pts%el_b[-1]).flatten())/dx

            D = np.zeros((ne, ne))

            for i in range(ne):
                for j in range(ne):
                    for k in range(p - 1):
                        D[i, j] += wts[k, i]*col_quad[i + ne*k, j]

            lower = int(np.ceil(p/2) - 1)
            upper = -int(np.floor(p/2))

            D[:, :(p - 1)] += D[:, -(p - 1):]
            D = D[lower:upper, :D.shape[1] - (p - 1)]

            return D
        
        else:
            dx = el_b[-1]/Nel
            ne = Nbase - 1
            
            a = grev
            b = grev + dx/2
            c = np.vstack((a, b)).reshape((-1,), order = 'F')[:-1]
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1) 
            pts, wts = inter.construct_quadrature_grid(2*ne, p - 1, pts_loc, wts_loc, c)
            
            col_quad = inter.collocation_matrix(p - 1, ne, T[1:-1], (pts%el_b[-1]).flatten())/dx
            
            D = np.zeros((ne, ne))
            
            for il in range(2*ne):
                i = int(np.floor(il/2))
                for j in range(ne):
                    for k in range(p - 1):
                        D[i, j] += wts[k, il]*col_quad[il + 2*ne*k, j] 
                        
            lower = int(np.ceil(p/2) - 1)
            upper = -int(np.floor(p/2))

            D[:, :(p - 1)] += D[:, -(p - 1):]
            D = D[lower:upper, :D.shape[1] - (p - 1)]

            return D


        
def mass_matrix_V0_1d(p, Nbase, T, bc, full=False):
    """
    Computes the 1d mass matrix of the space V0.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    mass : 2d np.array
        mass matrix in V0
    """
    
    if bc == True: 
        bcon = 1
        Nbase_0 = Nbase - p
    else:
        bcon = 0
        Nbase_0 = Nbase
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)

    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p + 1, d, T, pts)

    mass = np.zeros((Nbase_0, Nbase_0))

    for ie in range(ne):
        for il in range(p + 1):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value = 0.

                for g in range(p + 1):
                    value += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]

                mass[i%Nbase_0, j%Nbase_0] += value
    
    
    if full == False:
        mass = mass[(1 - bcon):Nbase_0 - (1 - bcon), (1 - bcon):Nbase_0 - (1 - bcon)]
                    
    return mass



def mass_matrix_V1_1d(p, Nbase, T, bc):
    """
    Computes the 1d mass matrix of the space V1.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    mass : 2d np.array
        mass matrix in V1
    """
    
    if bc == True: 
        bcon = 1
        Nbase_0 = Nbase - p
    else:
        bcon = 0
        Nbase_0 = Nbase
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(ne, p, pts_loc, wts_loc, el_b)
    
    t = T[1:-1]
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p - 1, Nbase - 1, p, d, t, pts)

    mass = np.zeros((Nbase_0 - (1 - bcon), Nbase_0 - (1 - bcon)))

    for ie in range(ne):
        for il in range(p):
            for jl in range(p):
                i = ie + il
                j = ie + jl

                value = 0.

                for g in range(p):
                    value += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]

                mass[i%Nbase_0, j%Nbase_0] += p/(t[i + p] - t[i])*p/(t[j + p] - t[j])*value
                    
                    
    return mass



def mass_matrix_V0(p, Nbase, T, bc, full=[False, False, False]):
    """
    Computes the mass matrix of the space V0.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    M0 : sparse matrix
        mass matrix in V0
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    Mx = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc_x, full=full_x))
    My = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc_y, full=full_y))
    Mz = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc_z, full=full_z))
    
    M0 = sparse.kron(sparse.kron(Mx, My), Mz, format='csr')
    
    return M0



def mass_matrix_V1(p, Nbase, T, bc, full=[False, False, False]):
    """
    Computes the mass matrix of the space V1.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    M1 : sparse matrix
        mass matrix in V1
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc_x, full=full_x))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc_y, full=full_y))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc_z, full=full_z))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc_x))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc_y))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc_z))
    
    Maa = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_NN_z)
    Mbb = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_NN_z)
    Mcc = sparse.kron(sparse.kron(M_NN_x, M_NN_y), M_DD_z)
    
    M1 = sparse.block_diag((Maa, Mbb, Mcc), format='csr')
    
    return M1



def mass_matrix_V2(p, Nbase, T, bc, full=[False, False, False]):
    """
    Computes the mass matrix of the space V2.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    M2 : sparse matrix
        mass matrix in V2
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc_x, full_x))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc_y, full_y))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc_z, full_z))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc_x))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc_y))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc_z))
    
    Maa = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_DD_z)
    Mbb = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_DD_z)
    Mcc = sparse.kron(sparse.kron(M_DD_x, M_DD_y), M_NN_z)
    
    M2 = sparse.block_diag((Maa, Mbb, Mcc), format='csr')
    
    return M2



def mass_matrix_V3(p, Nbase, T, bc):
    """
    Computes the mass matrix of the space V3.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    M3 : sparse matrix
        mass matrix in V3
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    Mx = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc_x))
    My = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc_y))
    Mz = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc_z))
    
    M3 = sparse.kron(sparse.kron(Mx, My), Mz, format='csr')
    
    return M3



def normalization_V0_1d(p, Nbase, T):
    """
    Computes the normalization of all basis functions of the space V0.
    
    Parameters
    ---------- 
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    ----------
    norm: np.array
        the integral of each basis function over the entire computational domain
    
    """
    
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    Nel = len(el_b) - 1
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(Nel, p, pts_loc, wts_loc, el_b)

    basis = inter.eval_on_grid_splines_ders(p, Nbase, p, 0, T, pts) 
    
    norm = np.zeros(Nbase)
    
    for ie in range(Nel):
        for il in range(p + 1):
            i = ie + il
                            
            value = 0.
            for g in range(p):
               
                value += wts[g, ie]*basis[il, 0, g, ie]

            norm[i] += value
                            
    return norm



def normalization_V1_1d(p, Nbase, T):
    """
    Computes the normalization of all basis functions of the space V1.
    
    Parameters
    ---------- 
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    ----------
    norm: np.array
        the integral of each basis function over the entire computational domain
    
    """
    
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    Nel = len(el_b) - 1
    t = T[1:-1]
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
    pts, wts = inter.construct_quadrature_grid(Nel, p - 1, pts_loc, wts_loc, el_b)

    basis = inter.eval_on_grid_splines_ders(p - 1, Nbase - 1, p - 1, 0, t, pts) 
    
    norm = np.zeros(Nbase - 1)
    
    for ie in range(Nel):
        for il in range(p):
            i = ie + il
                            
            value = 0.
            for g in range(p - 1):
               
                value += wts[g, ie]*basis[il, 0, g, ie]

            norm[i] += value*p/(t[i + p] - t[i])
                            
    return norm



def normalization_V0(p, Nbase, T):
    """
    Computes the normalization of all basis functions of the space V0.
    
    Parameters
    ---------- 
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    ----------
    norm: np.array
        the integral of each basis function N_i*N_j*N_k over the entire computational domain
    
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)

    pts_x, wts_x = inter.construct_quadrature_grid(Nel_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(Nel_y, py, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(Nel_z, pz, pts_z_loc, wts_z_loc, el_b_z)

    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px, 0, Tx, pts_x) 
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py, 0, Ty, pts_y) 
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz, 0, Tz, pts_z)
    
    norm = np.zeros((Nbase_x, Nbase_y, Nbase_z))
    
    for ie_0 in range(Nel_x):
        for ie_1 in range(Nel_y):
            for ie_2 in range(Nel_z):
                for il_0 in range(px + 1):
                    for il_1 in range(py + 1):
                        for il_2 in range(pz + 1):
                            i0 = ie_0 + il_0
                            i1 = ie_1 + il_1
                            i2 = ie_2 + il_2

                            value = 0.
                            for g_0 in range(px):
                                for g_1 in range(py):
                                    for g_2 in range(pz):

                                        wvol = wts_x[g_0, ie_0]*wts_y[g_1, ie_1]*wts_z[g_2, ie_2]
                                        value += wvol*basis_x[il_0, 0, g_0, ie_0]*basis_y[il_1, 0, g_1, ie_1]*basis_z[il_2, 0, g_2, ie_2]


                            norm[i0, i1, i2] += value
                            
    return norm