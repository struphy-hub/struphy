import numpy as np
import psydac.core.interface as inter
import psydac.core.bsplines as bsp


def integrate_1d(points, weights, fun):
    
    n = points.shape[0]
    k = points.shape[1]
    
    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[ie, g]*fun(points[ie, g])
        
    return f_int



def integrate_2d(points, weights, fun):
    
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