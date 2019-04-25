import numpy as np
import psydac.core.interface as inter
import psydac.core.bsplines as bsp
import scipy.sparse as sparse


def integrate_1d(points, weights, fun):
    """Integrates the function fun over the quadrature grid defined by (points, weights) in 1d.
    
    points : 2d np.array
        quadrature points in format (element, local point)
        
    weights : 2d np.array
        quadrature weights in format (element, local weight)
        
    fun : callable
        1d function to be integrated
    """
    
    n = points.shape[0]
    k = points.shape[1]
    
    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[ie, g]*fun(points[ie, g])
        
    return f_int



def integrate_2d(points, weights, fun):
    """Integrates the function fun over the quadrature grid defined by (points, weights) in 2d.
    
    points : list of 2d np.arrays
        quadrature points in x and y diretion in format (element, local point)
        
    weights : list of 2d np.arrays
        quadrature weights in x and y direction in format (element, local weight)
        
    fun : callable
        2d function to be integrated
    """
    
    pts_0, pts_1 = points
    wts_0, wts_1 = weights
    
    n0 = pts_0.shape[0]
    n1 = pts_1.shape[0]
    k0 = pts_0.shape[1]
    k1 = pts_1.shape[1]
    
    f_int = np.zeros((n0, n1))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for g_0 in range(k0):
                for g_1 in range(k1):
                    f_int[ie_0, ie_1] += wts_0[ie_0, g_0]*wts_1[ie_1, g_1]*fun(pts_0[ie_0, g_0], pts_1[ie_1, g_1])
                     
    return f_int



def integrate_3d(points, weights, fun):
    """Integrates the function fun over the quadrature grid defined by (points, weights) in 3d.
    
    points : list of 2d np.arrays
        quadrature points in x, y and z diretion in format (element, local point)
        
    weights : list of 2d np.arrays
        quadrature weights in x, y and z direction in format (element, local weight)
        
    fun : callable
        3d function to be integrated
    """
    
    pts_0, pts_1, pts_2 = points
    wts_0, wts_1, wts_2 = weights
    
    n0 = pts_0.shape[0]
    n1 = pts_1.shape[0]
    n2 = pts_2.shape[0]
    k0 = pts_0.shape[1]
    k1 = pts_1.shape[1]
    k2 = pts_2.shape[1]
    
    f_int = np.zeros((n0, n1, n2))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for ie_2 in range(n2):
                for g_0 in range(k0):
                    for g_1 in range(k1):
                        for g_2 in range(k2):
                            f_int[ie_0, ie_1, ie_2] += wts_0[ie_0, g_0]*wts_1[ie_1, g_1]*wts_2[ie_2, g_2]*fun(pts_0[ie_0, g_0], pts_1[ie_1, g_1], pts_2[ie_2, g_2])
                     
    return f_int




def L2_prod_V0(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V0
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz, d, Tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value
    
    return f_int
    
    
    
    
def L2_prod_V1_x(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V1 (x-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz, d, Tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])
    
    return f_int

    
    
    
def L2_prod_V1_y(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V1 (y-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    ty = Ty[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz, d, Tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*py/(ty[iy + py] - ty[iy])
    
    return f_int
    
    
    
def L2_prod_V1_z(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V1 (z-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz, d, tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    
    
    
    
    
    
def L2_prod_V2_x(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V2 (x-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px, d, Tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz, d, tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*py/(ty[iy + py] - ty[iy])*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    

    
def L2_prod_V2_y(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V2 (y-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py, d, Ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz, d, tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])*pz/(tz[iz + pz] - tz[iz])
    
    return f_int
    
 
 
def L2_prod_V2_z(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V2 (z-component)
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz, d, Tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += value*px/(tx[ix + px] - tx[ix])*py/(ty[iy + py] - ty[iy])
    
    return f_int


def L2_prod_V3(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V3
    using a quadrature rule of order p = (px, py, pz).
    
    fun: callable
        function for scalar product
    
    p: list of ints
        spline degrees in x, y and z direction
    
    Nbase: list of ints
        number of spline functions in x, y and z direction
        
    T: list of 1d np.arrays
        knot vector
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

    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(ne_x, px, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(ne_y, px, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(ne_z, px, pts_z_loc, wts_z_loc, el_b_z)
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    d = 0
    basis_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px, d, tx, pts_x)
    basis_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py, d, ty, pts_y)
    basis_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz, d, tz, pts_z)
    
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
                            for g_x in range(px):
                                for g_y in range(py):
                                    for g_z in range(pz):
                                        
                                        wvol = wts_x[g_x, ie_x]*wts_y[g_y, ie_y]*wts_z[g_z, ie_z]
                                        basi = basis_x[il_x, 0, g_x, ie_x]*basis_y[il_y, 0, g_y, ie_y]*basis_z[il_z, 0, g_z, ie_z]
                                        
                                        value += wvol*fun(pts_x[g_x, ie_x], pts_y[g_y, ie_y], pts_z[g_z, ie_z])*basi
                                        
                            f_int[ix, iy, iz] += px/(tx[ix + px] - tx[ix])*py/(ty[iy + py] - ty[iy])*pz/(tz[iz + pz] - tz[iz])*value
                            
    return f_int
                            

    
    
    
    

def L2_prod_V0_1d(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V0
    using a quadrature rule of order p.
    
    fun: callable
        function for scalar product
    
    p: int
        spline degree
    
    Nbase: int
        number of spline functions
        
    T: np.array
        knot vector
    """
    
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1 
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(ne, p, pts_loc, wts_loc, el_b)
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase, p, d, T, pts)
    
    f_int = np.zeros(Nbase)
    
    for ie in range(ne):
        for il in range(p + 1):
            i = ie + il
            
            value = 0.
            for g in range(p):
                value += wts[g, ie]*fun(pts[g, ie])*basis[il, 0, g, ie]
                
            f_int[i] += value
            
    return f_int
    
    
    
    
def L2_prod_V1_1d(fun, p, Nbase, T):
    """Returns the L2 scalar product of the function fun with the B-splines of the space V1
    using a quadrature rule of order p.
    
    fun: callable
        function for scalar product
    
    p: int
        spline degree
    
    Nbase: int
        number of spline functions
        
    T: np.array
        knot vector
    """
    
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    ne = len(el_b) - 1 
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(ne, p, pts_loc, wts_loc, el_b)
    
    t = T[1:-1]
    
    d = 0
    basis = inter.eval_on_grid_splines_ders(p - 1, Nbase - 1, p, d, t, pts)
    
    f_int = np.zeros(Nbase - 1)
    
    for ie in range(ne):
        for il in range(p):
            i = ie + il
            
            value = 0.
            for g in range(p):
                value += wts[g, ie]*fun(pts[g, ie])*basis[il, 0, g, ie]
                
            f_int[i] += value*p/(t[i + p] - t[i])
            
    return f_int
    
    
    

def histopolation_matrix(T, p, greville, bc):
    
    if bc == False:
        Nbase = len(T) - p - 1
        return inter.histopolation_matrix(p, Nbase, T, greville)
    
    if bc == True:
        if p%2 != 0:
            dx = 1./(len(T) - 2*p - 1)
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            pts, wts = bsp.quadrature_grid(np.append(greville, 1.), pts_loc, wts_loc)

            col_quad = bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc)/dx

            ng = len(greville)
            
            D = np.zeros((ng, ng))

            for i in range(ng):
                for j in range(ng):
                    for k in range(len(pts[0])):
                        D[i, j] += wts[i, k]*col_quad[i*(p - 1) + k, j]
                        
            return D
        
        else:
            dx = 1./(len(T) - 2*p - 1)
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            
            dx = 1./(len(T) - 2*p - 1)
            a = greville - dx/2
            b = greville
            c = np.vstack((a, b)).reshape((-1,), order = 'F')
            
            pts, wts = bsp.quadrature_grid(np.append(c, 1.), pts_loc, wts_loc)
            
            col_quad = bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc)/dx
            
            ng = len(greville)

            D = np.zeros((ng, ng))

            for il in range(2*ng):
                i = int(np.floor((il - 1)/2))%ng

                for jl in range(p):
                    j = int(np.floor(il/2) + jl)%ng

                    for k in range(len(pts[0])):
                        D[i, j] += wts[il, k]*col_quad[il*(p - 1) + k, j] 
                        
            return D
        
        
        
        
def mass_matrix_V0_1d(p, Nbase, T, bc):
    
    if bc == True: 
        bcon = 1
    else:
        bcon = 0
    
    el_b = inter.construct_grid_from_knots(p, Nbase + bcon*p, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts, wts = inter.construct_quadrature_grid(ne, p + 1, pts_loc, wts_loc, el_b)

    d = 0
    basis = inter.eval_on_grid_splines_ders(p, Nbase + bcon*p, p + 1, d, T, pts)

    mass = np.zeros((Nbase, Nbase))

    for ie in range(ne):
        for il in range(p + 1):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value = 0.

                for g in range(p + 1):
                    value += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]

                mass[i%Nbase, j%Nbase] += value
                    
                    
    return mass[(1 - bcon):Nbase - (1 - bcon), (1 - bcon):Nbase - (1 - bcon)]



def mass_matrix_V1_1d(p, Nbase, T, bc):
    
    if bc == True: 
        bcon = 1
    else:
        bcon = 0
    
    el_b = inter.construct_grid_from_knots(p, Nbase + bcon*p, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(ne, p, pts_loc, wts_loc, el_b)
    
    t = T[1:-1]
    d = 0
    basis = inter.eval_on_grid_splines_ders(p - 1, Nbase + bcon*p - 1, p, d, t, pts)

    mass = np.zeros((Nbase - (1 - bcon), Nbase - (1 - bcon)))

    for ie in range(ne):
        for il in range(p):
            for jl in range(p):
                i = ie + il
                j = ie + jl

                value = 0.

                for g in range(p):
                    value += wts[g, ie]*basis[il, 0, g, ie]*basis[jl, 0, g, ie]

                mass[i%Nbase, j%Nbase] += p/(t[i + p] - t[i])*p/(t[j + p] - t[j])*value
                    
                    
    return mass



def mass_matrix_DN_1d(p, Nbase, T, bc):
    
    if bc == True: 
        bcon = 1
    else:
        bcon = 0
    
    el_b = inter.construct_grid_from_knots(p, Nbase + bcon*p, T)
    ne = len(el_b) - 1

    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(ne, p, pts_loc, wts_loc, el_b)
    
    t = T[1:-1]
    d = 0
    
    basis1 = inter.eval_on_grid_splines_ders(p - 1, Nbase + bcon*p - 1, p, d, t, pts)
    basis2 = inter.eval_on_grid_splines_ders(p, Nbase + bcon*p, p, d, T, pts)
    
    
    mass = np.zeros((Nbase - (1 - bcon), Nbase)) 
    
    for ie in range(ne):
        for il in range(p):
            for jl in range(p + 1):
                i = ie + il
                j = ie + jl

                value = 0.

                for g in range(p):
                    value += wts[g, ie]*basis1[il, 0, g, ie]*basis2[jl, 0, g, ie]

                mass[i%Nbase, j%Nbase] += p/(t[i + p] - t[i])*value
                    
                    
    return mass[:, (1 - bcon):Nbase - (1 - bcon)]


def mass_matrix_V0(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    Mx = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc))
    My = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc))
    Mz = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc))
    
    return sparse.kron(sparse.kron(Mx, My), Mz, format='csr')



def mass_matrix_V1(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc))
    
    M_DN_x = sparse.csr_matrix(mass_matrix_DN_1d(px, Nbase_x, Tx, bc))
    M_DN_y = sparse.csr_matrix(mass_matrix_DN_1d(py, Nbase_y, Ty, bc))
    M_DN_z = sparse.csr_matrix(mass_matrix_DN_1d(pz, Nbase_z, Tz, bc))
    
    Maa = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_NN_z)
    Mab = sparse.kron(sparse.kron(M_DN_x, M_DN_y.transpose()), M_NN_z)
    Mac = sparse.kron(sparse.kron(M_DN_x, M_NN_y), M_DN_z.transpose())
    Mba = sparse.kron(sparse.kron(M_DN_x.transpose(), M_DN_y), M_NN_z)
    Mbb = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_NN_z)
    Mbc = sparse.kron(sparse.kron(M_NN_x, M_DN_y), M_DN_z.transpose())
    Mca = sparse.kron(sparse.kron(M_DN_x.transpose(), M_NN_y), M_DN_z)
    Mcb = sparse.kron(sparse.kron(M_NN_x, M_DN_y.transpose()), M_DN_z)
    Mcc = sparse.kron(sparse.kron(M_NN_x, M_NN_y), M_DD_z)
    
    
    return sparse.bmat([[Maa, Mab, Mac], [Mba, Mbb, Mbc], [Mca, Mcb, Mcc]], format='csr')

def mass_matrix_V1_block_diag(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc))
    
    
    Maa = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_NN_z)
    Mbb = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_NN_z)
    Mcc = sparse.kron(sparse.kron(M_NN_x, M_NN_y), M_DD_z)
    
    
    return sparse.block_diag((Maa, Mbb, Mcc), format='csr')


def mass_matrix_V2(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc))
    
    M_DN_x = sparse.csr_matrix(mass_matrix_DN_1d(px, Nbase_x, Tx, bc))
    M_DN_y = sparse.csr_matrix(mass_matrix_DN_1d(py, Nbase_y, Ty, bc))
    M_DN_z = sparse.csr_matrix(mass_matrix_DN_1d(pz, Nbase_z, Tz, bc))
    
    Maa = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_DD_z)
    Mab = sparse.kron(sparse.kron(M_DN_x.transpose(), M_DN_y), M_DD_z)
    Mac = sparse.kron(sparse.kron(M_DN_x.transpose(), M_DD_y), M_DN_z)
    Mba = sparse.kron(sparse.kron(M_DN_x, M_DN_y.transpose()), M_DD_z)
    Mbb = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_DD_z)
    Mbc = sparse.kron(sparse.kron(M_DD_x, M_DN_y.transpose()), M_DN_z)
    Mca = sparse.kron(sparse.kron(M_DN_x, M_DD_y), M_DN_z.transpose())
    Mcb = sparse.kron(sparse.kron(M_DD_x, M_DN_y), M_DN_z.transpose())
    Mcc = sparse.kron(sparse.kron(M_DD_x, M_DD_y), M_NN_z)
    
    
    return sparse.bmat([[Maa, Mab, Mac], [Mba, Mbb, Mbc], [Mca, Mcb, Mcc]], format='csr')


def mass_matrix_V2_block_diag(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    M_NN_x = sparse.csr_matrix(mass_matrix_V0_1d(px, Nbase_x, Tx, bc))
    M_NN_y = sparse.csr_matrix(mass_matrix_V0_1d(py, Nbase_y, Ty, bc))
    M_NN_z = sparse.csr_matrix(mass_matrix_V0_1d(pz, Nbase_z, Tz, bc))
    
    M_DD_x = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc))
    M_DD_y = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc))
    M_DD_z = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc))
    
    
    Maa = sparse.kron(sparse.kron(M_NN_x, M_DD_y), M_DD_z)
    Mbb = sparse.kron(sparse.kron(M_DD_x, M_NN_y), M_DD_z)
    Mcc = sparse.kron(sparse.kron(M_DD_x, M_DD_y), M_NN_z)
    
    return sparse.block_diag((Maa, Mbb, Mcc), format='csr')

def mass_matrix_V3(px, py, pz, Nbase_x, Nbase_y, Nbase_z, Tx, Ty, Tz, bc):
    
    Mx = sparse.csr_matrix(mass_matrix_V1_1d(px, Nbase_x, Tx, bc))
    My = sparse.csr_matrix(mass_matrix_V1_1d(py, Nbase_y, Ty, bc))
    Mz = sparse.csr_matrix(mass_matrix_V1_1d(pz, Nbase_z, Tz, bc))
    
    return sparse.kron(sparse.kron(Mx, My), Mz, format='csr')
