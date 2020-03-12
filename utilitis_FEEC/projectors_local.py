import numpy                  as np

import utilitis_FEEC.kernels_projectors as kernels
import utilitis_FEEC.bsplines           as bsp





# ===================================================================
def integrate_1d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 1d.
    
    Parameters
    ----------
    points : 2d np.array
        quadrature points in format (element, local point)
        
    weights : 2d np.array
        quadrature weights in format (element, local point)
    
    fun : callable
        1d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    n1  = points.shape[0]
    nq1 = points.shape[1]
    
    f_int = np.empty(n1)
    mat_f = np.empty(nq1)
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        
        w1   = weights[ie1, :]
        Pts1 = points[ie1, :]
        
        mat_f[:] = fun(Pts1)
        
        kernels.kernel_int_1d(nq1, w1, mat_f, f_loc)
        
        f_int[ie1] = f_loc
        
    return f_int
# ===================================================================









class projectors_local_1d:
    
    def __init__(self, T, p, bc):
        
        self.T         = T
        self.p         = p
        self.bc        = bc
        self.el_b      = bsp.breakpoints(self.T, self.p)
        self.Nel       = len(self.el_b) - 1
        self.NbaseN    = self.Nel + self.p - self.bc*self.p
        self.NbaseD    = self.NbaseN - (1 - self.bc)
        self.delta     = 1/self.Nel
        self.quad_loc  = np.polynomial.legendre.leggauss(self.p + 1)
        
        
    # quasi interpolation
    def PI_0(self, fun):

        n        = self.NbaseN
        lambda_j = np.zeros(n, dtype=float)
        
        if self.bc == False:


            # 1 - point linear interpolation
            if self.p == 1:

                for j in range(n):

                    x0 = self.T[j + 1]
                    lambda_j[j] = fun(x0)

            # 3 - point quadratic interpolation
            if self.p == 2:

                for j in range(n):

                    if   j == 0:
                        lambda_j[j] = fun(self.T[2])

                    elif j == n - 1:
                        lambda_j[j] = fun(self.T[n])

                    else:

                        x0 =  self.T[j + 1]
                        x1 = (self.T[j + 1] + self.T[j + 2])/2
                        x2 =  self.T[j + 2]

                        lambda_j[j] = -fun(x0)/2 + 2*fun(x1) - fun(x2)/2

            # 5 - point cubic interpolation            
            elif self.p == 3:

                for j in range(n):

                    if   j == 0:
                        lambda_j[j] = fun(self.T[3])

                    elif j == 1:

                        xj =  2
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]

                        lambda_j[j] = 1/18*(-5*fun(x0) + 40*fun(x1) - 24*fun(x2) + 8*fun(x3) - fun(x4))

                    elif j == n - 2:

                        xj =  n - 3
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]

                        lambda_j[j] = 1/18*(-fun(x0) + 8*fun(x1) - 24*fun(x2) + 40*fun(x3) - 5*fun(x4))

                    elif j == n - 1:
                        lambda_j[j] = fun(self.T[n])

                    else:

                        x0 =  self.T[j + 1]
                        x1 = (self.T[j + 1] + self.T[j + 2])/2
                        x2 =  self.T[j + 2]
                        x3 = (self.T[j + 2] + self.T[j + 3])/2
                        x4 =  self.T[j + 3]

                        lambda_j[j] = 1/6*(fun(x0) - 8*fun(x1) + 20*fun(x2) - 8*fun(x3) + fun(x4))

            # 7 - point quartic interpolation            
            elif self.p == 4:

                for j in range(n):

                    if   j == 0:
                        lambda_j[j] = fun(self.T[4])

                    elif j == 1:

                        xj =  3
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]
                        x5 = (self.T[xj + 3] + self.T[xj + 4])/2
                        x6 =  self.T[xj + 4]

                        lambda_j[j] = 1/90*(-59*fun(x0)/4 + 236*fun(x1) - 250*fun(x2) + 180*fun(x3) - 305*fun(x4)/4 + 16*fun(x5) - fun(x6))

                    elif j == 2:

                        xj =  3
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]
                        x5 = (self.T[xj + 3] + self.T[xj + 4])/2
                        x6 =  self.T[xj + 4]

                        lambda_j[j] = 1/45*(23*fun(x0)/8 - 46*fun(x1) + 395*fun(x2)/2 - 170*fun(x3) + 605*fun(x4)/8 - 16*fun(x5) + fun(x6))

                    elif j == n - 3:

                        xj =  n - 4
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]
                        x5 = (self.T[xj + 3] + self.T[xj + 4])/2
                        x6 =  self.T[xj + 4]

                        lambda_j[j] = 1/45*(fun(x0) - 16*fun(x1) + 605*fun(x2)/8 - 170*fun(x3) + 395*fun(x4)/2 - 46*fun(x5) + 23*fun(x6)/8)

                    elif j == n - 2:

                        xj =  n - 4
                        x0 =  self.T[xj + 1]
                        x1 = (self.T[xj + 1] + self.T[xj + 2])/2
                        x2 =  self.T[xj + 2]
                        x3 = (self.T[xj + 2] + self.T[xj + 3])/2
                        x4 =  self.T[xj + 3]
                        x5 = (self.T[xj + 3] + self.T[xj + 4])/2
                        x6 =  self.T[xj + 4]

                        lambda_j[j] = 1/90*(-fun(x0) + 16*fun(x1) - 305*fun(x2)/4 + 180*fun(x3) - 255*fun(x4) + 236*fun(x5) - 59*fun(x6)/4)

                    elif j == n - 1:
                        lambda_j[j] = fun(self.T[n])

                    else:

                        x0 =  self.T[j + 1]
                        x1 = (self.T[j + 1] + self.T[j + 2])/2
                        x2 =  self.T[j + 2]
                        x3 = (self.T[j + 2] + self.T[j + 3])/2
                        x4 =  self.T[j + 3]
                        x5 = (self.T[j + 3] + self.T[j + 4])/2
                        x6 =  self.T[j + 4]

                        lambda_j[j] = 2/45*(-fun(x0) + 16*fun(x1) - 295*fun(x2)/4 + 140*fun(x3) - 295*fun(x4)/4 + 16*fun(x5) - fun(x6))
                        
        else:
            
            # 1 - point linear interpolation
            if self.p == 1:

                for j in range(n):

                    x0 = self.T[j + 1]
                    lambda_j[j] = fun(x0%1.)
                    
            # 3 - point quadratic interpolation
            if self.p == 2:

                for j in range(n):

                    x0 =  self.T[j + 1]
                    x1 = (self.T[j + 1] + self.T[j + 2])/2
                    x2 =  self.T[j + 2]

                    lambda_j[j] = -fun(x0%1.)/2 + 2*fun(x1%1.) - fun(x2%1.)/2
                    
            # 5 - point cubic interpolation            
            elif self.p == 3:

                for j in range(n):

                    x0 =  self.T[j + 1]
                    x1 = (self.T[j + 1] + self.T[j + 2])/2
                    x2 =  self.T[j + 2]
                    x3 = (self.T[j + 2] + self.T[j + 3])/2
                    x4 =  self.T[j + 3]

                    lambda_j[j] = 1/6*(fun(x0%1.) - 8*fun(x1%1.) + 20*fun(x2%1.) - 8*fun(x3%1.) + fun(x4%1.))
                    
            # 7 - point quartic interpolation            
            elif self.p == 4:

                for j in range(n):

                    x0 =  self.T[j + 1]
                    x1 = (self.T[j + 1] + self.T[j + 2])/2
                    x2 =  self.T[j + 2]
                    x3 = (self.T[j + 2] + self.T[j + 3])/2
                    x4 =  self.T[j + 3]
                    x5 = (self.T[j + 3] + self.T[j + 4])/2
                    x6 =  self.T[j + 4]

                    lambda_j[j] = 2/45*(-fun(x0%1.) + 16*fun(x1%1.) - 295*fun(x2%1.)/4 + 140*fun(x3%1.) - 295*fun(x4%1.)/4 + 16*fun(x5%1.) - fun(x6%1.))

        
        return lambda_j
    
    # quasi histopolation
    def PI_1(self, fun):
        
        n        = self.NbaseD
        lambda_j = np.zeros(n, dtype=float)
        
        if self.p == 1:
            
            for j in range(n):
            
                xm2 =  self.T[j + 1]
                x0  =  self.T[j + 2]

                x = np.array([xm2, x0])

                pts, wts = bsp.quadrature_grid(x, self.quad_loc[0], self.quad_loc[1])
                pts = pts%1.

                f_int = integrate_1d(pts, wts, fun)

                lambda_j[j] = f_int[0]
        
        if self.p == 2:
            
            for j in range(n):
            
                xm2 =  self.T[j + 1]
                xm1 = (self.T[j + 1] + self.T[j + 2])/2
                x0  =  self.T[j + 2]
                x1  = (self.T[j + 2] + self.T[j + 3])/2
                x2  =  self.T[j + 3]

                x = np.array([xm2, xm1, x0, x1, x2])

                pts, wts = bsp.quadrature_grid(x, self.quad_loc[0], self.quad_loc[1])
                pts = pts%1.

                f_int = integrate_1d(pts, wts, fun)

                lambda_j[j] = -f_int[0]/2 + 3*(f_int[1] + f_int[2])/2 - f_int[3]/2 
            
        return lambda_j