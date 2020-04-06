import numpy as np
import scipy.sparse as spa
import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.kernels_projectors_local_mhd as ker_loc
import hylife.utilitis_FEEC.kernels_projectors_local_ini as ker_loc_ini


class projectors_local_mhd:
    
    def __init__(self, T, p, bc):
        
        self.T         = T                                                                     # knot vectors
        self.p         = p                                                                     # spline degrees
        self.bc        = bc                                                                    # boundary conditions
        self.el_b      = [bsp.breakpoints(T, p) for T, p in zip(self.T, self.p)]               # element boundaries
        self.Nel       = [len(el_b) - 1 for el_b in self.el_b]                                 # number of elements
        self.NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, self.p, self.bc)]     # number of basis functions (N)
        self.NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, self.bc)]       # number of basis functions (D)
        self.delta     = [1/Nel for Nel in self.Nel]                                           # element length
        self.n_quad    = [p + 1 for p in self.p]                                               # degree of quadrature rule
        self.quad_loc  = [np.polynomial.legendre.leggauss(n_quad) for n_quad in self.n_quad]   # quadrature points and weights in (-1, 1)
        
        # set interpolation and histopolation coefficients
        self.coeff_i = [0, 0, 0]
        self.coeff_h = [0, 0, 0]
        
        for a in range(3):
            if self.bc[a] == True:
                self.coeff_i[a] = np.zeros((1, 2*(self.p[a] - 1) + 1), dtype=float, order='F')
                self.coeff_h[a] = np.zeros((1, 2*self.p[a]),           dtype=float, order='F')
                
            
                if   self.p[a] == 1:
                    self.coeff_i[a][0, :] = np.array([1.])
                    self.coeff_h[a][0, :] = np.array([1., 1.])

                elif self.p[a] == 2:
                    self.coeff_i[a][0, :] = 1/2 * np.array([-1., 4., -1.])
                    self.coeff_h[a][0, :] = 1/2 * np.array([-1., 3., 3., -1.])

                elif self.p[a] == 3:
                    self.coeff_i[a][0, :] = 1/6 * np.array([1., -8., 20., -8., 1.])
                    self.coeff_h[a][0, :] = 1/6 * np.array([1., -7., 12., 12., -7., 1.])

                elif self.p[a] == 4:
                    self.coeff_i[a][0, :] = 2/45 * np.array([-1., 16., -295/4, 140., -295/4, 16., -1.])
                    self.coeff_h[a][0, :] = 2/45 * np.array([-1., 15., -231/4, 265/4, 265/4, -231/4, 15.,-1.])

                else:
                    print('degree > 4 not implemented!')

            else:
                self.coeff_i[a] = np.zeros((2*(self.p[a] - 1) + 1, 2*(self.p[a] - 1) + 1), dtype=float, order='F')
                self.coeff_h[a] = np.zeros((2*(self.p[a] - 1) + 1, 2*self.p[a]),           dtype=float, order='F')

                if   self.p[a] == 1:
                    self.coeff_i[a][0, :] = np.array([1.])
                    self.coeff_h[a][0, :] = np.array([1., 1.])

                elif self.p[a] == 2:
                    self.coeff_i[a][0, :] = 1/2 * np.array([ 2., 0.,  0.])
                    self.coeff_i[a][1, :] = 1/2 * np.array([-1., 4., -1.])
                    self.coeff_i[a][2, :] = 1/2 * np.array([ 0., 0.,  2.])

                    self.coeff_h[a][0, :] = 1/2 * np.array([ 3., -1.,  0.,  0.])
                    self.coeff_h[a][1, :] = 1/2 * np.array([-1.,  3.,  3., -1.])
                    self.coeff_h[a][2, :] = 1/2 * np.array([ 0.,  0., -1.,  3.])

                elif self.p[a] == 3:
                    self.coeff_i[a][0, :] = 1/18 * np.array([18.,   0.,   0.,   0.,   0.])
                    self.coeff_i[a][1, :] = 1/18 * np.array([-5.,  40., -24.,   8.,  -1.])
                    self.coeff_i[a][2, :] = 1/18 * np.array([ 3., -24.,  60., -24.,   3.])
                    self.coeff_i[a][3, :] = 1/18 * np.array([-1.,   8., -24.,  40.,  -5.])
                    self.coeff_i[a][4, :] = 1/18 * np.array([ 0.,   0.,   0.,   0.,  18.])

                    self.coeff_h[a][0, :] = 1/18 * np.array([23., -17.,   7.,  -1.,   0.,  0.]) 
                    self.coeff_h[a][1, :] = 1/18 * np.array([-8.,  56., -28.,   4.,   0.,  0.])
                    self.coeff_h[a][2, :] = 1/18 * np.array([ 3., -21.,  36.,  36., -21.,  3.])
                    self.coeff_h[a][3, :] = 1/18 * np.array([ 0.,   0.,   4., -28.,  56., -8.])
                    self.coeff_h[a][4, :] = 1/18 * np.array([ 0.,   0.,  -1.,   7., -17., 23.]) 

                elif self.p[a] == 4:
                    self.coeff_i[a][0, :] = 1/360 * np.array([360.,    0.,     0.,     0.,     0.,    0.,   0.])
                    self.coeff_i[a][1, :] = 1/360 * np.array([-59.,  944., -1000.,   720.,  -305.,   64.,  -4.])
                    self.coeff_i[a][2, :] = 1/360 * np.array([ 23., -368.,  1580., -1360.,   605., -128.,   8.])
                    self.coeff_i[a][3, :] = 1/360 * np.array([-16.,  256., -1180.,  2240., -1180.,  256., -16.])
                    self.coeff_i[a][4, :] = 1/360 * np.array([  8., -128.,   605., -1360.,  1580., -368.,  23.])
                    self.coeff_i[a][5, :] = 1/360 * np.array([ -4.,   64.,  -305.,   720., -1000.,  944., -59.])
                    self.coeff_i[a][6, :] = 1/360 * np.array([  0.,    0.,     0.,     0.,     0.,    0., 360.])

                    self.coeff_h[a][0, :] = 1/360 * np.array([ 419., -525.,   475.,  -245.,    60.,    -4.,    0.,   0.])
                    self.coeff_h[a][1, :] = 1/360 * np.array([ -82., 1230., -1350.,   730.,  -180.,    12.,    0.,   0.])
                    self.coeff_h[a][2, :] = 1/360 * np.array([  39., -585.,  2175., -1425.,   360.,   -24.,    0.,   0.])
                    self.coeff_h[a][3, :] = 1/360 * np.array([ -16.,  240.,  -924.,  1060.,  1060.,  -924.,  240., -16.])
                    self.coeff_h[a][4, :] = 1/360 * np.array([   0.,    0.,   -24.,   360., -1425.,  2175., -585.,  39.])
                    self.coeff_h[a][5, :] = 1/360 * np.array([   0.,    0.,    12.,  -180.,   730., -1350., 1230., -82.])
                    self.coeff_h[a][6, :] = 1/360 * np.array([   0.,    0.,    -4.,    60.,  -245.,   475., -525., 419.])

                else:
                    print('degree > 4 not implemented!')
        
        
        # set interpolation points
        n_lambda_int        = [NbaseN for NbaseN in self.NbaseN]
        
        self.n_int          = [2*p - 1 for p in self.p]      # number of interpolation points
        self.n_int_locbf_N  = [2*p - 1 for p in self.p]      # number of non-vanishing N bf in interpolation interval
        self.n_int_locbf_D  = [2*p - 2 for p in self.p]      # number of non-vanishing D bf in interpolation interval
        
        self.x_int = [np.zeros((n_lambda_int, n_int), dtype=float, order='F') for n_lambda_int, n_int in zip(n_lambda_int, self.n_int)]
        
        self.int_global_N   = [np.zeros((n_lambda_int, n_int_locbf_N), dtype=int, order='F') for n_lambda_int, n_int_locbf_N in zip(n_lambda_int, self.n_int_locbf_N)]
        self.int_global_D   = [np.zeros((n_lambda_int, n_int_locbf_D), dtype=int, order='F') for n_lambda_int, n_int_locbf_D in zip(n_lambda_int, self.n_int_locbf_D)]
        
        self.int_loccof_N   = [np.zeros((n_lambda_int, n_int_locbf_N), dtype=int, order='F') for n_lambda_int, n_int_locbf_N in zip(n_lambda_int, self.n_int_locbf_N)]
        self.int_loccof_D   = [np.zeros((n_lambda_int, n_int_locbf_D), dtype=int, order='F') for n_lambda_int, n_int_locbf_D in zip(n_lambda_int, self.n_int_locbf_D)]
        
        self.x_int_indices  = [np.zeros((n_lambda_int, n_int), dtype=int, order='F') for n_lambda_int, n_int in zip(n_lambda_int, self.n_int)]
        self.coeffi_indices = [np.zeros( n_lambda_int, dtype=int) for n_lambda_int in n_lambda_int]
        
        
        self.n_int_nvcof_D  = [None, None, None]
        self.n_int_nvcof_N  = [None, None, None]
        
        self.int_add_D      = [None, None, None]
        self.int_add_N      = [None, None, None]
        
        self.int_shift_D    = [0, 0, 0]
        self.int_shift_N    = [0, 0, 0]
        
        
        for a in range(3):
            if self.bc[a] == False:
                
                # maximum number of non-vanishing coefficients
                self.n_int_nvcof_D[a] = 3*self.p[a] - 3
                self.n_int_nvcof_N[a] = 3*self.p[a] - 2
                
                # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
                self.int_add_D[a] = np.arange(self.n_int[a] - 2) + 1
                self.int_add_N[a] = np.arange(self.n_int[a] - 1) + 1
                
                counter_D = 0
                counter_N = 0
                
                # shift local coefficients --> global coefficients (D)
                self.int_shift_D[a] = np.arange(self.NbaseD[a]) - (self.p[a] - 2)
                self.int_shift_D[a][:2*self.p[a] - 2] = 0
                self.int_shift_D[a][-(2*self.p[a] - 2):] = self.int_shift_D[a][-(2*self.p[a] - 2)]

                # shift local coefficients --> global coefficients (N)
                self.int_shift_N[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 1)
                self.int_shift_N[a][:2*self.p[a] - 1]  = 0
                self.int_shift_N[a][-(2*self.p[a] - 1):] = self.int_shift_N[a][-(2*self.p[a] - 1)]
                
                counter_coeffi = np.copy(self.p[a])
                
                for i in range(n_lambda_int[a]):
                    
                    # left boundary region
                    if  i < self.p[a] - 1:
                        self.int_global_N[a][i]   = np.arange(self.n_int_locbf_N[a])
                        self.int_global_D[a][i]   = np.arange(self.n_int_locbf_D[a])
                        
                        self.x_int_indices[a][i]  = np.arange(self.n_int[a])
                        self.coeffi_indices[a][i] = i
                        for j in range(2*(self.p[a] - 1) + 1):
                            xi                  =  self.p[a] - 1
                            self.x_int[a][i, j] = (self.T[a][xi + 1 + int(j/2)] + self.T[a][xi + 1 + int((j + 1)/2)])/2

                    # right boundary region
                    elif i > n_lambda_int[a] - self.p[a]:
                        self.int_global_N[a][i] = np.arange(self.n_int_locbf_N[a]) + n_lambda_int[a] - self.p[a] - (self.p[a] - 1)
                        self.int_global_D[a][i] = np.arange(self.n_int_locbf_D[a]) + n_lambda_int[a] - self.p[a] - (self.p[a] - 1)
                        
                        self.x_int_indices[a][i] = np.arange(self.n_int[a]) + 2*(n_lambda_int[a] - self.p[a] - (self.p[a] - 1))
                        self.coeffi_indices[a][i] = counter_coeffi
                        counter_coeffi += 1
                        for j in range(2*(self.p[a] - 1) + 1):
                            xi               =  n_lambda_int[a] - self.p[a]
                            self.x_int[a][i, j] = (self.T[a][xi + 1 + int(j/2)] + self.T[a][xi + 1 + int((j + 1)/2)])/2

                    # interior
                    else:
                        self.int_global_N[a][i] = np.arange(self.n_int_locbf_N[a]) + i - (self.p[a] - 1)
                        self.int_global_D[a][i] = np.arange(self.n_int_locbf_D[a]) + i - (self.p[a] - 1)
                        
                        self.x_int_indices[a][i] = np.arange(self.n_int[a]) + 2*(i - (self.p[a] - 1))
                        self.coeffi_indices[a][i] = self.p[a] - 1
                        for j in range(2*(self.p[a] - 1) + 1):
                            self.x_int[a][i, j]  = (self.T[a][i + 1 + int(j/2)] + self.T[a][i + 1 + int((j + 1)/2)])/2
                            
                            
                    # local coefficient index
                    if i > 0:
                        for il in range(self.n_int_locbf_D[a]):
                            k_glob_new = self.int_global_D[a][i, il]
                            bol = (k_glob_new == self.int_global_D[a][i - 1])

                            if np.any(bol):
                                self.int_loccof_D[a][i, il] = self.int_loccof_D[a][i - 1, np.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_int[a] - self.p[a] - (self.p[a] - 2)) and (self.int_loccof_D[a][i, il] == 0):
                                self.int_loccof_D[a][i, il] = self.int_add_D[a][counter_D]
                                counter_D += 1

                        for il in range(self.n_int_locbf_N[a]):
                            k_glob_new = self.int_global_N[a][i, il]
                            bol = (k_glob_new == self.int_global_N[a][i - 1])

                            if np.any(bol):
                                self.int_loccof_N[a][i, il] = self.int_loccof_N[a][i - 1, np.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_int[a] - self.p[a] - (self.p[a] - 2)) and (self.int_loccof_N[a][i, il] == 0):
                                self.int_loccof_N[a][i, il] = self.int_add_N[a][counter_N]
                                counter_N += 1
                                
            else:
                for i in range(n_lambda_int[a]):

                    # maximum number of non-vanishing coefficients
                    self.n_int_nvcof_D[a] = 2*self.p[a] - 2
                    self.n_int_nvcof_N[a] = 2*self.p[a] - 1

                    # shift local coefficients --> global coefficients (D)
                    self.int_shift_D[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 2)

                    # shift local coefficients --> global coefficients (N)
                    self.int_shift_N[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 1)

                    self.int_global_N[a][i] = (np.arange(self.n_int_locbf_N[a]) + i - (self.p[a] - 1))%self.NbaseN[a]
                    self.int_global_D[a][i] = (np.arange(self.n_int_locbf_D[a]) + i - (self.p[a] - 1))%self.NbaseD[a] 
                    self.int_loccof_N[a][i] =  np.arange(self.n_int_locbf_N[a] - 1, -1, -1)
                    self.int_loccof_D[a][i] =  np.arange(self.n_int_locbf_D[a] - 1, -1, -1)
                    
                    self.x_int_indices[a][i] = (np.arange(self.n_int[a]) + 2*(i - (self.p[a] - 1)))%(2*self.Nel[a])
                    self.coeffi_indices[a][i] = 0

                    for j in range(2*(self.p[a] - 1) + 1):
                        self.x_int[a][i, j] = ((self.T[a][i + 1 + int(j/2)] + self.T[a][i + 1 + int((j + 1)/2)])/2)%1.
            
        
        
        # set histopolation points, quadrature points and weights
        n_lambda_his = [NbaseD for NbaseD in self.NbaseD]
        
        self.n_his         = [2*p     for p in self.p]     # number of histopolation intervals
        self.n_his_locbf_N = [2*p     for p in self.p]     # number of non-vanishing N bf in histopolation interval
        self.n_his_locbf_D = [2*p - 1 for p in self.p]     # number of non-vanishing D bf in histopolation interval
        
        self.x_his = [np.zeros((n_lambda_his, n_his + 1), dtype=float) for n_lambda_his, n_his in zip(n_lambda_his, self.n_his)]  
        
        self.his_global_N = [np.zeros((n_lambda_his, n_his_locbf_N), dtype=int, order='F') for n_lambda_his, n_his_locbf_N in zip(n_lambda_his, self.n_his_locbf_N)]
        self.his_global_D = [np.zeros((n_lambda_his, n_his_locbf_D), dtype=int, order='F') for n_lambda_his, n_his_locbf_D in zip(n_lambda_his, self.n_his_locbf_D)]
        
        self.his_loccof_N = [np.zeros((n_lambda_his, n_his_locbf_N), dtype=int, order='F') for n_lambda_his, n_his_locbf_N in zip(n_lambda_his, self.n_his_locbf_N)]
        self.his_loccof_D = [np.zeros((n_lambda_his, n_his_locbf_D), dtype=int, order='F') for n_lambda_his, n_his_locbf_D in zip(n_lambda_his, self.n_his_locbf_D)]
        
        
        self.x_his_indices  = [np.zeros((n_lambda_his, n_his), dtype=int, order='F') for n_lambda_his, n_his in zip(n_lambda_his, self.n_his)]
        self.coeffh_indices = [np.zeros( n_lambda_his, dtype=int) for n_lambda_his in n_lambda_his]
        
        self.pts = [0, 0, 0]
        self.wts = [0, 0, 0]
        
        self.n_his_nvcof_D = [None, None, None]
        self.n_his_nvcof_N = [None, None, None]
        
        self.his_add_D     = [None, None, None]
        self.his_add_N     = [None, None, None]
        
        self.his_shift_D   = [0, 0, 0]
        self.his_shift_N   = [0, 0, 0]
        
        for a in range(3):
            if self.bc[a] == False:
                
                # maximum number of non-vanishing coefficients
                self.n_his_nvcof_D[a] = 3*self.p[a] - 2
                self.n_his_nvcof_N[a] = 3*self.p[a] - 1

                # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
                self.his_add_D[a] = np.arange(self.n_his[a] - 2) + 1
                self.his_add_N[a] = np.arange(self.n_his[a] - 1) + 1

                counter_D = 0
                counter_N = 0
                
                # shift local coefficients --> global coefficients (D)
                self.his_shift_D[a] = np.arange(self.NbaseD[a]) - (self.p[a] - 1)
                self.his_shift_D[a][:2*self.p[a] - 1] = 0
                self.his_shift_D[a][-(2*self.p[a] - 1):] = self.his_shift_D[a][-(2*self.p[a] - 1)]

                # shift local coefficients --> global coefficients (N)
                self.his_shift_N[a] = np.arange(self.NbaseN[a]) -  self.p[a]
                self.his_shift_N[a][:2*self.p[a]]  = 0
                self.his_shift_N[a][-2*self.p[a]:] = self.his_shift_N[a][-2*self.p[a]]
                
                counter_coeffh = np.copy(self.p[a])
                
                for i in range(n_lambda_his[a]):
                    
                    # left boundary region
                    if  i < self.p[a] - 1:
                        self.his_global_N[a][i] = np.arange(self.n_his_locbf_N[a])
                        self.his_global_D[a][i] = np.arange(self.n_his_locbf_D[a])
                        
                        self.x_his_indices[a][i] = np.arange(self.n_his[a])
                        self.coeffh_indices[a][i] = i
                        for j in range(2*self.p[a] + 1):
                            xi                =  self.p[a] - 1
                            self.x_his[a][i, j]  = (self.T[a][xi + 1 + int(j/2)] + self.T[a][xi + 1 + int((j + 1)/2)])/2

                    # right boundary region
                    elif i > n_lambda_his[a] - self.p[a]:
                        self.his_global_N[a][i] = np.arange(self.n_his_locbf_N[a]) + n_lambda_his[a] - self.p[a] - (self.p[a] - 1)
                        self.his_global_D[a][i] = np.arange(self.n_his_locbf_D[a]) + n_lambda_his[a] - self.p[a] - (self.p[a] - 1)
                        
                        self.x_his_indices[a][i] = np.arange(self.n_his[a]) + 2*(n_lambda_his[a] - self.p[a] - (self.p[a] - 1))
                        self.coeffh_indices[a][i] = counter_coeffh
                        counter_coeffh += 1
                        for j in range(2*self.p[a] + 1):
                            xi                =  n_lambda_his[a] - self.p[a]
                            self.x_his[a][i, j]  = (self.T[a][xi + 1 + int(j/2)] + self.T[a][xi + 1 + int((j + 1)/2)])/2

                    # interior
                    else:
                        self.his_global_N[a][i] = np.arange(self.n_his_locbf_N[a]) + i - (self.p[a] - 1)
                        self.his_global_D[a][i] = np.arange(self.n_his_locbf_D[a]) + i - (self.p[a] - 1)
                        
                        self.x_his_indices[a][i] = np.arange(self.n_his[a]) + 2*(i - (self.p[a] - 1))
                        self.coeffh_indices[a][i] = self.p[a] - 1
                        for j in range(2*self.p[a] + 1):
                            self.x_his[a][i, j]  = (self.T[a][i + 1 + int(j/2)] + self.T[a][i + 1 + int((j + 1)/2)])/2
                    
                    
                    # local coefficient index
                    if i > 0:
                        for il in range(self.n_his_locbf_D[a]):
                            k_glob_new = self.his_global_D[a][i, il]
                            bol = (k_glob_new == self.his_global_D[a][i - 1])

                            if np.any(bol):
                                self.his_loccof_D[a][i, il] = self.his_loccof_D[a][i - 1, np.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_his[a] - self.p[a] - (self.p[a] - 2)) and (self.his_loccof_D[a][i, il] == 0):
                                self.his_loccof_D[a][i, il] = self.his_add_D[a][counter_D]
                                counter_D += 1

                        for il in range(self.n_his_locbf_N[a]):
                            k_glob_new = self.his_global_N[a][i, il]
                            bol = (k_glob_new == self.his_global_N[a][i - 1])

                            if np.any(bol):
                                self.his_loccof_N[a][i, il] = self.his_loccof_N[a][i - 1, np.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_his[a] - self.p[a] - (self.p[a] - 2)) and (self.his_loccof_N[a][i, il] == 0):
                                self.his_loccof_N[a][i, il] = self.his_add_N[a][counter_N]
                                counter_N += 1
                                
                # quadrature points and weights
                self.pts[a], self.wts[a] = bsp.quadrature_grid(np.unique(self.x_his[a].flatten()), self.quad_loc[a][0], self.quad_loc[a][1])
                self.pts[a] = np.asfortranarray(self.pts[a])
                self.wts[a] = np.asfortranarray(self.wts[a])
                                
                                
            else:
                for i in range(n_lambda_his[a]):

                    # maximum number of non-vanishing coefficients
                    self.n_his_nvcof_D[a] = 2*self.p[a] - 1
                    self.n_his_nvcof_N[a] = 2*self.p[a]

                    # shift local coefficients --> global coefficients (D)
                    self.his_shift_D[a] = np.arange(self.NbaseD[a]) - (self.p[a] - 1)

                    # shift local coefficients --> global coefficients (N)
                    self.his_shift_N[a] = np.arange(self.NbaseD[a]) -  self.p[a]

                    self.his_global_N[a][i] = (np.arange(self.n_his_locbf_N[a]) + i - (self.p[a] - 1))%self.NbaseN[a]
                    self.his_global_D[a][i] = (np.arange(self.n_his_locbf_D[a]) + i - (self.p[a] - 1))%self.NbaseD[a] 
                    self.his_loccof_N[a][i] =  np.arange(self.n_his_locbf_N[a] - 1, -1, -1)
                    self.his_loccof_D[a][i] =  np.arange(self.n_his_locbf_D[a] - 1, -1, -1)
                    
                    self.x_his_indices[a][i] = (np.arange(self.n_his[a]) + 2*(i - (self.p[a] - 1)))%(2*self.Nel[a])
                    self.coeffh_indices[a][i] = 0

                    for j in range(2*self.p[a] + 1):
                        self.x_his[a][i, j] = (self.T[a][i + 1 + int(j/2)] + self.T[a][i + 1 + int((j + 1)/2)])/2

                # quadrature points and weights
                self.pts[a], self.wts[a] = bsp.quadrature_grid(np.append(np.unique(self.x_his[a].flatten()%1.), 1.), self.quad_loc[a][0], self.quad_loc[a][1])
                self.pts[a] = np.asfortranarray(self.pts[a])
                self.wts[a] = np.asfortranarray(self.wts[a])
                    
                    
    # ======================================================================== 
    def projection_Q(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_2(rho_eq * Ginv * lambda^1) with the output (coefficients, basis_fun).
        
        non-vanishing coefficients
        (int, his, his) : (D, N, N)*Ginv_00*rho_eq, (N, D, N)*Ginv_01*rho_eq, (N, N, D)*Ginv_02*rho_eq
        (his, int, his) : (D, N, N)*Ginv_10*rho_eq, (N, D, N)*Ginv_11*rho_eq, (N, N, D)*Ginv_12*rho_eq
        (his, his, int) : (D, N, N)*Ginv_20*rho_eq, (N, D, N)*Ginv_21*rho_eq, (N, N, D)*Ginv_22*rho_eq
        '''
        
        # non-vanishing coefficients
        tau11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_N[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau12 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau13 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        tau21 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau23 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        tau31 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_his_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau32 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_his_nvcof_N[1], self.n_int_nvcof_D[2]), dtype=float, order='F')
        
        # unique interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [x_int1.size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, x_int2.size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, x_int3.size]
        
        # evaluation of basis functions at interpolation and quadrature points
        basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisN_his = [np.asfortranarray(bsp.collocation_matrix(T, p, pts.flatten(), bc).reshape(pts[:, 0].size, pts[0, :].size, NbaseN)) for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)] 
        
        basisD_int = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, x_int, bc, normalize=True)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 
        
        
        # assembly of 1 - component (PI_2_1 : int, his, his)
        mat_eq = np.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique1, x_int1, self.pts[1].flatten(), self.pts[2].flatten(), mat_eq, kind_fun=11, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_1([self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.n_quad[1], self.n_quad[2]], [self.n_int[0], self.n_his[1], self.n_his[2]], [self.n_int_locbf_D[0], self.n_his_locbf_N[1], self.n_his_locbf_N[2]], self.int_global_D[0], self.his_global_N[1], self.his_global_N[2], self.int_loccof_D[0], self.his_loccof_N[1], self.his_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffh_indices[2], basisD_int[0], basisN_his[1], basisN_his[2], self.x_int_indices[0], self.x_his_indices[1], self.x_his_indices[2], self.wts[1], self.wts[2], tau11, mat_eq.reshape(n_unique1[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique1, x_int1, self.pts[1].flatten(), self.pts[2].flatten(), mat_eq, kind_fun=12, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_1([self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.n_quad[1], self.n_quad[2]], [self.n_int[0], self.n_his[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_his_locbf_N[2]], self.int_global_N[0], self.his_global_D[1], self.his_global_N[2], self.int_loccof_N[0], self.his_loccof_D[1], self.his_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffh_indices[2], basisN_int[0], basisD_his[1], basisN_his[2], self.x_int_indices[0], self.x_his_indices[1], self.x_his_indices[2], self.wts[1], self.wts[2], tau12, mat_eq.reshape(n_unique1[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique1, x_int1, self.pts[1].flatten(), self.pts[2].flatten(), mat_eq, kind_fun=13, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_1([self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.n_quad[1], self.n_quad[2]], [self.n_int[0], self.n_his[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_his_locbf_D[2]], self.int_global_N[0], self.his_global_N[1], self.his_global_D[2], self.int_loccof_N[0], self.his_loccof_N[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_h[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffh_indices[2], basisN_int[0], basisN_his[1], basisD_his[2], self.x_int_indices[0], self.x_his_indices[1], self.x_his_indices[2], self.wts[1], self.wts[2], tau13, mat_eq.reshape(n_unique1[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        # assembly of 2 - component (PI_2_2 : his, int, his)
        mat_eq = np.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique2, self.pts[0].flatten(), x_int2, self.pts[2].flatten(), mat_eq, kind_fun=14, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_2([self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], [self.n_quad[0], self.n_quad[2]], [self.n_his[0], self.n_int[1], self.n_his[2]], [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]], self.his_global_D[0], self.int_global_N[1], self.his_global_N[2], self.his_loccof_D[0], self.int_loccof_N[1], self.his_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_h[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisD_his[0], basisN_int[1], basisN_his[2], self.x_his_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[0], self.wts[2], tau21, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique2[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique2, self.pts[0].flatten(), x_int2, self.pts[2].flatten(), mat_eq, kind_fun=15, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_2([self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], [self.n_quad[0], self.n_quad[2]], [self.n_his[0], self.n_int[1], self.n_his[2]], [self.n_his_locbf_N[0], self.n_int_locbf_D[1], self.n_his_locbf_N[2]], self.his_global_N[0], self.int_global_D[1], self.his_global_N[2], self.his_loccof_N[0], self.int_loccof_D[1], self.his_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_h[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_his[0], basisD_int[1], basisN_his[2], self.x_his_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[0], self.wts[2], tau22, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique2[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique2, self.pts[0].flatten(), x_int2, self.pts[2].flatten(), mat_eq, kind_fun=16, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_2([self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], [self.n_quad[0], self.n_quad[2]], [self.n_his[0], self.n_int[1], self.n_his[2]], [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]], self.his_global_N[0], self.int_global_N[1], self.his_global_D[2], self.his_loccof_N[0], self.int_loccof_N[1], self.his_loccof_D[2], self.coeff_h[0], self.coeff_i[1], self.coeff_h[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_his[0], basisN_int[1], basisD_his[2], self.x_his_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[0], self.wts[2], tau23, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique2[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        # assembly of 3 - component (PI_2_3 : his, his, int)
        mat_eq = np.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique3, self.pts[0].flatten(), self.pts[1].flatten(), x_int3, mat_eq, kind_fun=17, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_3([self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], [self.n_quad[0], self.n_quad[1]], [self.n_his[0], self.n_his[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]], self.his_global_D[0], self.his_global_N[1], self.int_global_N[2], self.his_loccof_D[0], self.his_loccof_N[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_h[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisD_his[0], basisN_his[1], basisN_int[2], self.x_his_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[0], self.wts[1], tau31, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique3[2]))
        
        ker_loc_ini.kernel_eva(n_unique3, self.pts[0].flatten(), self.pts[1].flatten(), x_int3, mat_eq, kind_fun=18, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_3([self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], [self.n_quad[0], self.n_quad[1]], [self.n_his[0], self.n_his[1], self.n_int[2]], [self.n_his_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]], self.his_global_N[0], self.his_global_D[1], self.int_global_N[2], self.his_loccof_N[0], self.his_loccof_D[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_h[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_his[0], basisD_his[1], basisN_int[2], self.x_his_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[0], self.wts[1], tau32, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique3[2]))
        
        ker_loc_ini.kernel_eva(n_unique3, self.pts[0].flatten(), self.pts[1].flatten(), x_int3, mat_eq, kind_fun=19, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi2_3([self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], [self.n_quad[0], self.n_quad[1]], [self.n_his[0], self.n_his[1], self.n_int[2]], [self.n_his_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_D[2]], self.his_global_N[0], self.his_global_N[1], self.int_global_D[2], self.his_loccof_N[0], self.his_loccof_N[1], self.int_loccof_D[2], self.coeff_h[0], self.coeff_h[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_his[0], basisN_his[1], basisD_int[2], self.x_his_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[0], self.wts[1], tau33, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique3[2]))
        
        
        
        # conversion to sparse matrices (1 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_N[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseD[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau11 = spa.csc_matrix((tau11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]))         
        tau11.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseD[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau12 = spa.csc_matrix((tau12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]))         
        tau12.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseD[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau13 = spa.csc_matrix((tau13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]))         
        tau13.eliminate_zeros()
        
        # conversion to sparse matrices (2 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau21 = spa.csc_matrix((tau21.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau21.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau22 = spa.csc_matrix((tau22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau22.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau23 = spa.csc_matrix((tau23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau23.eliminate_zeros()
        
        # conversion to sparse matrices (3 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_his_nvcof_N[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau31 = spa.csc_matrix((tau31.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau31.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau32 = spa.csc_matrix((tau32.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau32.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_his_nvcof_N[1], self.n_int_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau33 = spa.csc_matrix((tau33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau33.eliminate_zeros()
        
        
        tau = spa.bmat([[tau11.T, tau12.T, tau13.T], [tau21.T, tau22.T, tau23.T], [tau31.T, tau32.T, tau33.T]], format='csc')
        
        return tau
    
    
    
    # ======================================================================== 
    def projection_T(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_1(B_eq * Ginv * lambda^1) with the output (coefficients, basis_fun).
        
        non-vanishing coefficients
        (his, int, int) : (D, N, N)*(B1*Ginv_20 - B2*Ginv_10), (N, D, N)*(B1*Ginv_21 - B2*Ginv_11), (N, N, D)*(B1*Ginv_22 - B2*Ginv_12)
        (int, his, int) : (D, N, N)*(B2*Ginv_00 - B0*Ginv_20), (N, D, N)*(B2*Ginv_01 - B0*Ginv_21), (N, N, D)*(B2*Ginv_02 - B0*Ginv_22)
        (int, int, his) : (D, N, N)*(B0*Ginv_10 - B1*Ginv_00), (N, N, D)*(B0*Ginv_11 - B1*Ginv_01), (N, N, D)*(B0*Ginv_12 - B1*Ginv_02)
        '''
        
        # non-vanishing coefficients
        tau11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau12 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_int_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau13 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_D[2]), dtype=float, order='F')
        
        tau21 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        tau23 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_N[1], self.n_int_nvcof_D[2]), dtype=float, order='F')
        
        tau31 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau32 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_N[2]), dtype=float, order='F')
        tau33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        # unique interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [self.pts[0].flatten().size, x_int2.size, x_int3.size]
        n_unique2 = [x_int1.size, self.pts[1].flatten().size, x_int3.size]
        n_unique3 = [x_int1.size, x_int2.size, self.pts[2].flatten().size]
        
        # evaluation of basis functions at interpolation and quadrature points
        basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisN_his = [np.asfortranarray(bsp.collocation_matrix(T, p, pts.flatten(), bc).reshape(pts[:, 0].size, pts[0, :].size, NbaseN)) for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)] 
        
        basisD_int = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, x_int, bc, normalize=True)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 
        
        # assembly of 1 - component (PI_1_1 : his, int, int)
        mat_eq = np.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=21, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_int_locbf_N[2]], self.his_global_D[0], self.int_global_N[1], self.int_global_N[2], self.his_loccof_D[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisD_his[0], basisN_int[1], basisN_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau11, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=22, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_N[0], self.n_int_locbf_D[1], self.n_int_locbf_N[2]], self.his_global_N[0], self.int_global_D[1], self.int_global_N[2], self.his_loccof_N[0], self.int_loccof_D[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisN_his[0], basisD_int[1], basisN_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau12, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=23, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_int_locbf_D[2]], self.his_global_N[0], self.int_global_N[1], self.int_global_D[2], self.his_loccof_N[0], self.int_loccof_N[1], self.int_loccof_D[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisN_his[0], basisN_int[1], basisD_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau13, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        
        # assembly of 2 - component (PI_1_2 : int, his, int)
        mat_eq = np.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=24, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_D[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]], self.int_global_D[0], self.his_global_N[1], self.int_global_N[2], self.int_loccof_D[0], self.his_loccof_N[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisD_int[0], basisN_his[1], basisN_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau21, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=25, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]], self.int_global_N[0], self.his_global_D[1], self.int_global_N[2], self.int_loccof_N[0], self.his_loccof_D[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_int[0], basisD_his[1], basisN_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau22, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=26, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_D[2]], self.int_global_N[0], self.his_global_N[1], self.int_global_D[2], self.int_loccof_N[0], self.his_loccof_N[1], self.int_loccof_D[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_int[0], basisN_his[1], basisD_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau23, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        
        # assembly of 3 - component (PI_1_3 : int, int, his)
        mat_eq = np.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=27, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]], self.int_global_D[0], self.int_global_N[1], self.his_global_N[2], self.int_loccof_D[0], self.int_loccof_N[1], self.his_loccof_N[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisD_int[0], basisN_int[1], basisN_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau31, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=28, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_int_locbf_D[1], self.n_his_locbf_N[2]], self.int_global_N[0], self.int_global_D[1], self.his_global_N[2], self.int_loccof_N[0], self.int_loccof_D[1], self.his_loccof_N[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_int[0], basisD_int[1], basisN_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau32, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=29, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]], self.int_global_N[0], self.int_global_N[1], self.his_global_D[2], self.int_loccof_N[0], self.int_loccof_N[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_int[0], basisN_int[1], basisD_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau33, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        
        # conversion to sparse matrices (1 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau11 = spa.csc_matrix((tau11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau11.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_N[0], self.n_int_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau12 = spa.csc_matrix((tau12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau12.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau13 = spa.csc_matrix((tau13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau13.eliminate_zeros()
        
        
        # conversion to sparse matrices (2 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_N[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau21 = spa.csc_matrix((tau21.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau21.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau22 = spa.csc_matrix((tau22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau22.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_N[1], self.n_int_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau23 = spa.csc_matrix((tau23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau23.eliminate_zeros()
        
        
        # conversion to sparse matrices (3 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau31 = spa.csc_matrix((tau31.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau31.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau32 = spa.csc_matrix((tau32.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau32.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau33 = spa.csc_matrix((tau33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau33.eliminate_zeros()
        
        tau = spa.bmat([[tau11.T, tau12.T, tau13.T], [tau21.T, tau22.T, tau23.T], [tau31.T, tau32.T, tau33.T]], format='csc')
        
        return tau
        
        
    # ======================================================================== 
    def projection_W(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_1(rho_eq/g_sqrt * lambda^1) with the output (coefficients, basis_fun).
        
        non-vanishing coefficients
        (his, int, int) : (D, N, N)*rho_eq/g_sqrt, 0, 0
        (int, his, int) : 0, (N, D, N)*rho_eq/g_sqrt, 0
        (int, int, his) : 0, 0, (N, N, D)*rho_eq/g_sqrt
        '''
        
        
        # non-vanishing coefficients
        tau11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        # unique interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [self.pts[0].flatten().size, x_int2.size, x_int3.size]
        n_unique2 = [x_int1.size, self.pts[1].flatten().size, x_int3.size]
        n_unique3 = [x_int1.size, x_int2.size, self.pts[2].flatten().size]
        
        # evaluation of basis functions at interpolation and quadrature points
        basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 
        
        # assembly of 1 - component (PI_1_1 : his, int, int)
        mat_eq = np.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=31, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_int_locbf_N[2]], self.his_global_D[0], self.int_global_N[1], self.int_global_N[2], self.his_loccof_D[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisD_his[0], basisN_int[1], basisN_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau11, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        # assembly of 2 - component (PI_1_2 : int, his, int)
        mat_eq = np.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=31, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]], self.int_global_N[0], self.his_global_D[1], self.int_global_N[2], self.int_loccof_N[0], self.his_loccof_D[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_int[0], basisD_his[1], basisN_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau22, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        # assembly of 3 - component (PI_1_3 : int, int, his)
        mat_eq = np.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=31, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]], self.int_global_N[0], self.int_global_N[1], self.his_global_D[2], self.int_loccof_N[0], self.int_loccof_N[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_int[0], basisN_int[1], basisD_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau33, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        # conversion to sparse matrices (1 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau11 = spa.csc_matrix((tau11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau11.eliminate_zeros()
        
        # conversion to sparse matrices (2 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau22 = spa.csc_matrix((tau22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau22.eliminate_zeros()
        
        # conversion to sparse matrices (3 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau33 = spa.csc_matrix((tau33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau33.eliminate_zeros()
        
        tau = spa.bmat([[tau11.T, None, None], [None, tau22.T, None], [None, None, tau33.T]], format='csc')
        
        return tau
    
    
    # ======================================================================== 
    def projection_S(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_1(p_eq * lambda^1) with the output (coefficients, basis_fun).
        
        non-vanishing coefficients
        (his, int, int) : (D, N, N)*p_eq, 0, 0
        (int, his, int) : 0, (N, D, N)*p_eq, 0
        (int, int, his) : 0, 0, (N, N, D)*p_eq
        '''
        
        
        # non-vanishing coefficients
        tau11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        # unique interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [self.pts[0].flatten().size, x_int2.size, x_int3.size]
        n_unique2 = [x_int1.size, self.pts[1].flatten().size, x_int3.size]
        n_unique3 = [x_int1.size, x_int2.size, self.pts[2].flatten().size]
        
        # evaluation of basis functions at interpolation and quadrature points
        basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 
        
        # assembly of 1 - component (PI_1_1 : his, int, int)
        mat_eq = np.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=91, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_int_locbf_N[2]], self.his_global_D[0], self.int_global_N[1], self.int_global_N[2], self.his_loccof_D[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisD_his[0], basisN_int[1], basisN_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau11, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        # assembly of 2 - component (PI_1_2 : int, his, int)
        mat_eq = np.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=91, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]], self.int_global_N[0], self.his_global_D[1], self.int_global_N[2], self.int_loccof_N[0], self.his_loccof_D[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_int[0], basisD_his[1], basisN_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau22, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        # assembly of 3 - component (PI_1_3 : int, int, his)
        mat_eq = np.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=91, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]], self.int_global_N[0], self.int_global_N[1], self.his_global_D[2], self.int_loccof_N[0], self.int_loccof_N[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_int[0], basisN_int[1], basisD_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau33, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        # conversion to sparse matrices (1 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau11 = spa.csc_matrix((tau11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau11.eliminate_zeros()
        
        # conversion to sparse matrices (2 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau22 = spa.csc_matrix((tau22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau22.eliminate_zeros()
        
        # conversion to sparse matrices (3 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau33 = spa.csc_matrix((tau33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau33.eliminate_zeros()
        
        tau = spa.bmat([[tau11.T, None, None], [None, tau22.T, None], [None, None, tau33.T]], format='csc')
        
        return tau
    
    
    
    # ======================================================================== 
    def projection_P(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_1(curlb_eq * lambda^2) with the output (coefficients, basis_fun).
        
        non-vanishing coefficients
        (his, int, int) :  0, -(D, N, D)*curlb3_eq, (D, D, N)*curlb2_eq 
        (int, his, int) :  (N, D, D)*curlb3_eq, 0, -(D, D, N)*curlb1_eq
        (int, int, his) : -(N, D, D)*curlb2_eq, (D, N, D)*curlb1_eq,  0
        '''
        
        # non-vanishing coefficients
        tau12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_D[2]), dtype=float, order='F')
        tau13 = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau21 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_D[2]), dtype=float, order='F')
        tau23 = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        tau31 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        tau32 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]), dtype=float, order='F')
        
        # unique interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [self.pts[0].flatten().size, x_int2.size, x_int3.size]
        n_unique2 = [x_int1.size, self.pts[1].flatten().size, x_int3.size]
        n_unique3 = [x_int1.size, x_int2.size, self.pts[2].flatten().size]
        
        # evaluation of basis functions at interpolation and quadrature points
        basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisN_his = [np.asfortranarray(bsp.collocation_matrix(T, p, pts.flatten(), bc).reshape(pts[:, 0].size, pts[0, :].size, NbaseN)) for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)] 
        
        basisD_int = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, x_int, bc, normalize=True)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 
        
        # assembly of 1 - component (PI_1_1 : his, int, int)
        mat_eq = np.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=41, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_int_locbf_D[2]], self.his_global_D[0], self.int_global_N[1], self.int_global_D[2], self.his_loccof_D[0], self.int_loccof_N[1], self.int_loccof_D[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisD_his[0], basisN_int[1], basisD_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau12, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        ker_loc_ini.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_eq, kind_fun=42, kind_map=kind_map, params=params_map)
            
        ker_loc.kernel_pi1_1([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.n_quad[0], [self.n_his[0], self.n_int[1], self.n_int[2]], [self.n_his_locbf_D[0], self.n_int_locbf_D[1], self.n_int_locbf_N[2]], self.his_global_D[0], self.int_global_D[1], self.int_global_N[2], self.his_loccof_D[0], self.int_loccof_D[1], self.int_loccof_N[2], self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basisD_his[0], basisD_int[1], basisN_int[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], tau13, mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]))
        
        # assembly of 2 - component (PI_1_2 : int, his, int)
        mat_eq = np.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=43, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_D[2]], self.int_global_N[0], self.his_global_D[1], self.int_global_D[2], self.int_loccof_N[0], self.his_loccof_D[1], self.int_loccof_D[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisN_int[0], basisD_his[1], basisD_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau21, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        ker_loc_ini.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_eq, kind_fun=44, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_2([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.n_quad[1], [self.n_int[0], self.n_his[1], self.n_int[2]], [self.n_int_locbf_D[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]], self.int_global_D[0], self.his_global_D[1], self.int_global_N[2], self.int_loccof_D[0], self.his_loccof_D[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], basisD_int[0], basisD_his[1], basisN_int[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], tau23, mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]))
        
        # assembly of 3 - component (PI_1_3 : int, int, his)
        mat_eq = np.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=45, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_N[0], self.n_int_locbf_D[1], self.n_his_locbf_D[2]], self.int_global_N[0], self.int_global_D[1], self.his_global_D[2], self.int_loccof_N[0], self.int_loccof_D[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisN_int[0], basisD_int[1], basisD_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau31, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        ker_loc_ini.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_eq, kind_fun=46, kind_map=kind_map, params=params_map)
        
        ker_loc.kernel_pi1_3([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.n_quad[2], [self.n_int[0], self.n_int[1], self.n_his[2]], [self.n_int_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]], self.int_global_D[0], self.int_global_N[1], self.his_global_D[2], self.int_loccof_D[0], self.int_loccof_N[1], self.his_loccof_D[2], self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], basisD_int[0], basisN_int[1], basisD_his[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], tau32, mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size))
        
        # conversion to sparse matrices (1 - component)
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.n_his_nvcof_D[0], self.n_int_nvcof_N[1], self.n_int_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau12 = spa.csc_matrix((tau12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau12.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], self.n_his_nvcof_D[0], self.n_int_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None])%self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau13 = spa.csc_matrix((tau13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau13.eliminate_zeros()
        
        
        # conversion to sparse matrices (2 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_his_nvcof_D[1], self.n_int_nvcof_D[2]))
        row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau21 = spa.csc_matrix((tau21.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau21.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], self.n_int_nvcof_D[0], self.n_his_nvcof_D[1], self.n_int_nvcof_N[2]))
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None])%self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        tau23 = spa.csc_matrix((tau23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))         
        tau23.eliminate_zeros()
        
        
        # conversion to sparse matrices (3 - component)
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.n_int_nvcof_N[0], self.n_int_nvcof_D[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau31 = spa.csc_matrix((tau31.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau31.eliminate_zeros()
        
        
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.n_int_nvcof_D[0], self.n_int_nvcof_N[1], self.n_his_nvcof_D[2]))
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None])%self.NbaseD[2]
        
        col  = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        tau32 = spa.csc_matrix((tau32.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))         
        tau32.eliminate_zeros()
        
        tau = spa.bmat([[None, tau12.T, tau13.T], [tau21.T, None, tau23.T], [tau31.T, tau32.T, None]], format='csc')
        
        return tau
    
    
    # ========================================================================                
    def projection_K(self, kind_map, params_map):
        '''
        Computes the matrix of the expression Pi_0(p_eq * lambda^0) with the output (coefficients, basis_fun)
        
        non-vanishing coefficients
        (int, int, int) : (N, N, N)*p_eq
        '''
        
        # non-vanishing coefficients
        tau = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float, order='F')
        
        # evaluation of equilibrium pressure at interpolation points
        x_int1   = np.unique(self.x_int[0].flatten())
        x_int2   = np.unique(self.x_int[1].flatten())
        x_int3   = np.unique(self.x_int[2].flatten())
        
        n_unique = [x_int1.size, x_int2.size, x_int3.size]
        
        mat_eq   = np.zeros((n_unique[0], n_unique[1], n_unique[2]), dtype=float, order='F')
        
        ker_loc_ini.kernel_eva(n_unique, x_int1, x_int2, x_int3, mat_eq, kind_fun=91, kind_map=kind_map, params=params_map)
        
        # evaluation of basis functions at interpolation points
        basis = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, [x_int1, x_int2, x_int3], self.bc)]
        
        # assembly of tau
        ker_loc.kernel_pi0(self.NbaseN, self.n_int, self.n_int_locbf_N, self.int_global_N[0], self.int_global_N[1], self.int_global_N[2], self.int_loccof_N[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_i[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], basis[0], basis[1], basis[2], self.x_int_indices[0], self.x_int_indices[1], self.x_int_indices[2], tau, mat_eq)
        
        # conversion to sparse matrix
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]))
        
        # row indices
        row  = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        # column indices
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None])%self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None])%self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None])%self.NbaseN[2]
        
        col  = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        # create sparse matrix 
        tau = spa.csc_matrix((tau.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))         
        tau.eliminate_zeros()

        return tau.T