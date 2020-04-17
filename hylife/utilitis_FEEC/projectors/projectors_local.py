# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Classes for local projectors in 1d and 3d based on quasi-interpolation
"""

import numpy as np

import hylife.utilitis_FEEC.bsplines as bsp

import hylife.utilitis_FEEC.projectors.kernels_projectors_local     as ker_loc
import hylife.utilitis_FEEC.projectors.kernels_projectors_local_eva as ker_loc_eva



# ======================= 1d ====================================
class projectors_local_1d:
    """
    Local commuting projectors pi_0 and pi_1 in 1d.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d space of B-splines
        
    n_quad : int
        number of quadrature points per integration interval for histopolations
    """
    
    def __init__(self, spline_space, n_quad):
        
        self.T       = spline_space.T       # knot vector
        self.p       = spline_space.p       # spline degree
        self.bc      = spline_space.bc      # boundary conditions
        
        self.NbaseN  = spline_space.NbaseN  # number of basis functions (N)
        self.NbaseD  = spline_space.NbaseD  # number of basis functions (D)
        
        self.n_quad  = n_quad               # number of quadrature point per integration interval
        
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0]  
        self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1]
        
        
        # set interpolation and histopolation coefficients
        if self.bc == True:
            self.coeff_i = np.zeros((1, 2*self.p - 1), dtype=float)
            self.coeff_h = np.zeros((1, 2*self.p)    , dtype=float)
             
            if   self.p == 1:
                self.coeff_i[0, :] = np.array([1.])
                self.coeff_h[0, :] = np.array([1., 1.])
                
            elif self.p == 2:
                self.coeff_i[0, :] = 1/2 * np.array([-1., 4., -1.])
                self.coeff_h[0, :] = 1/2 * np.array([-1., 3., 3., -1.])
                
            elif self.p == 3:
                self.coeff_i[0, :] = 1/6 * np.array([1., -8., 20., -8., 1.])
                self.coeff_h[0, :] = 1/6 * np.array([1., -7., 12., 12., -7., 1.])
                
            elif self.p == 4:
                self.coeff_i[0, :] = 2/45 * np.array([-1., 16., -295/4, 140., -295/4, 16., -1.])
                self.coeff_h[0, :] = 2/45 * np.array([-1., 15., -231/4, 265/4, 265/4, -231/4, 15.,-1.])
                
            else:
                print('degree > 4 not implemented!')
                
        else:
            self.coeff_i = np.zeros((2*self.p - 1, 2*self.p - 1), dtype=float)
            self.coeff_h = np.zeros((2*self.p - 1, 2*self.p)    , dtype=float)
            
            if   self.p == 1:
                self.coeff_i[0, :] = np.array([1.])
                self.coeff_h[0, :] = np.array([1., 1.])
                
            elif self.p == 2:
                self.coeff_i[0, :] = 1/2 * np.array([ 2., 0.,  0.])
                self.coeff_i[1, :] = 1/2 * np.array([-1., 4., -1.])
                self.coeff_i[2, :] = 1/2 * np.array([ 0., 0.,  2.])
                
                self.coeff_h[0, :] = 1/2 * np.array([ 3., -1.,  0.,  0.])
                self.coeff_h[1, :] = 1/2 * np.array([-1.,  3.,  3., -1.])
                self.coeff_h[2, :] = 1/2 * np.array([ 0.,  0., -1.,  3.])
        
            elif self.p == 3:
                self.coeff_i[0, :] = 1/18 * np.array([18.,   0.,   0.,   0.,   0.])
                self.coeff_i[1, :] = 1/18 * np.array([-5.,  40., -24.,   8.,  -1.])
                self.coeff_i[2, :] = 1/18 * np.array([ 3., -24.,  60., -24.,   3.])
                self.coeff_i[3, :] = 1/18 * np.array([-1.,   8., -24.,  40.,  -5.])
                self.coeff_i[4, :] = 1/18 * np.array([ 0.,   0.,   0.,   0.,  18.])
                
                self.coeff_h[0, :] = 1/18 * np.array([23., -17.,   7.,  -1.,   0.,  0.]) 
                self.coeff_h[1, :] = 1/18 * np.array([-8.,  56., -28.,   4.,   0.,  0.])
                self.coeff_h[2, :] = 1/18 * np.array([ 3., -21.,  36.,  36., -21.,  3.])
                self.coeff_h[3, :] = 1/18 * np.array([ 0.,   0.,   4., -28.,  56., -8.])
                self.coeff_h[4, :] = 1/18 * np.array([ 0.,   0.,  -1.,   7., -17., 23.]) 
                
            elif self.p == 4:
                self.coeff_i[0, :] = 1/360 * np.array([360.,    0.,     0.,     0.,     0.,    0.,   0.])
                self.coeff_i[1, :] = 1/360 * np.array([-59.,  944., -1000.,   720.,  -305.,   64.,  -4.])
                self.coeff_i[2, :] = 1/360 * np.array([ 23., -368.,  1580., -1360.,   605., -128.,   8.])
                self.coeff_i[3, :] = 1/360 * np.array([-16.,  256., -1180.,  2240., -1180.,  256., -16.])
                self.coeff_i[4, :] = 1/360 * np.array([  8., -128.,   605., -1360.,  1580., -368.,  23.])
                self.coeff_i[5, :] = 1/360 * np.array([ -4.,   64.,  -305.,   720., -1000.,  944., -59.])
                self.coeff_i[6, :] = 1/360 * np.array([  0.,    0.,     0.,     0.,     0.,    0., 360.])
                
                self.coeff_h[0, :] = 1/360 * np.array([ 419., -525.,   475.,  -245.,    60.,    -4.,    0.,   0.])
                self.coeff_h[1, :] = 1/360 * np.array([ -82., 1230., -1350.,   730.,  -180.,    12.,    0.,   0.])
                self.coeff_h[2, :] = 1/360 * np.array([  39., -585.,  2175., -1425.,   360.,   -24.,    0.,   0.])
                self.coeff_h[3, :] = 1/360 * np.array([ -16.,  240.,  -924.,  1060.,  1060.,  -924.,  240., -16.])
                self.coeff_h[4, :] = 1/360 * np.array([   0.,    0.,   -24.,   360., -1425.,  2175., -585.,  39.])
                self.coeff_h[5, :] = 1/360 * np.array([   0.,    0.,    12.,  -180.,   730., -1350., 1230., -82.])
                self.coeff_h[6, :] = 1/360 * np.array([   0.,    0.,    -4.,    60.,  -245.,   475., -525., 419.])
                
                
            else:
                print('degree > 4 not implemented!')
                
                
        # set interpolation points
        n_lambda_int        = np.copy(self.NbaseN)
        
        self.n_int          = 2*self.p - 1       # number of local interpolation points
        self.n_int_locbf_N  = 2*self.p - 1       # number of non-vanishing N bf in interpolation interval
        self.n_int_locbf_D  = 2*self.p - 2       # number of non-vanishing D bf in interpolation interval
        
        self.x_int          = np.zeros((n_lambda_int, self.n_int)        , dtype=float) # interpolation points
        
        self.int_global_N   = np.zeros((n_lambda_int, self.n_int_locbf_N), dtype=int)   # global indices of non-vanishing N bf
        self.int_global_D   = np.zeros((n_lambda_int, self.n_int_locbf_D), dtype=int)   # global indices of non-vanishing D bf
        
        self.int_loccof_N   = np.zeros((n_lambda_int, self.n_int_locbf_N), dtype=int)   # index of non-vanishing coeff. (N)
        self.int_loccof_D   = np.zeros((n_lambda_int, self.n_int_locbf_D), dtype=int)   # index of non-vanishing coeff. (D)
        
        self.x_int_indices  = np.zeros((n_lambda_int, self.n_int)        , dtype=int)
        
        self.coeffi_indices = np.zeros( n_lambda_int                     , dtype=int)
        
        
        if self.bc == False:
            
            # maximum number of non-vanishing coefficients
            self.n_int_nvcof_D = 3*self.p - 3
            self.n_int_nvcof_N = 3*self.p - 2
            
            # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
            self.int_add_D = np.arange(self.n_int - 2) + 1
            self.int_add_N = np.arange(self.n_int - 1) + 1

            counter_D = 0
            counter_N = 0
            
            # shift local coefficients --> global coefficients (D)
            self.int_shift_D = np.arange(self.NbaseD) - (self.p - 2)
            self.int_shift_D[:2*self.p - 2] = 0
            self.int_shift_D[-(2*self.p - 2):] = self.int_shift_D[-(2*self.p - 2)]
            
            # shift local coefficients --> global coefficients (N)
            self.int_shift_N = np.arange(self.NbaseN) - (self.p - 1)
            self.int_shift_N[:2*self.p - 1]  = 0
            self.int_shift_N[-(2*self.p - 1):] = self.int_shift_N[-(2*self.p - 1)]
            
            counter_coeffi = np.copy(self.p)
            
            
            for i in range(n_lambda_int):


                # left boundary region
                if  i < self.p - 1:
                    self.int_global_N[i] = np.arange(self.n_int_locbf_N)
                    self.int_global_D[i] = np.arange(self.n_int_locbf_D)
                    
                    
                    self.x_int_indices[i] = np.arange(self.n_int)
                    
                    
                    self.coeffi_indices[i] = i
                    for j in range(2*(self.p - 1) + 1):
                        xi                =  self.p - 1
                        self.x_int[i, j]  = (self.T[xi + 1 + int(j/2)] + self.T[xi + 1 + int((j + 1)/2)])/2

                # right boundary region
                elif i > n_lambda_int - self.p:
                    self.int_global_N[i] = np.arange(self.n_int_locbf_N) + n_lambda_int - self.p - (self.p - 1)
                    self.int_global_D[i] = np.arange(self.n_int_locbf_D) + n_lambda_int - self.p - (self.p - 1)
                    
                    self.x_int_indices[i] = np.arange(self.n_int) + 2*(n_lambda_int - self.p - (self.p - 1))
                    self.coeffi_indices[i] = counter_coeffi
                    counter_coeffi += 1
                    for j in range(2*(self.p - 1) + 1):
                        xi                =  n_lambda_int - self.p
                        self.x_int[i, j]  = (self.T[xi + 1 + int(j/2)] + self.T[xi + 1 + int((j + 1)/2)])/2

                # interior
                else:
                    self.int_global_N[i] = np.arange(self.n_int_locbf_N) + i - (self.p - 1)
                    self.int_global_D[i] = np.arange(self.n_int_locbf_D) + i - (self.p - 1)
                    
                    
                    if self.p == 1:
                        self.x_int_indices[i] = i
                    else:
                        self.x_int_indices[i] = np.arange(self.n_int) + 2*(i - (self.p - 1))
                    
                    
                    self.coeffi_indices[i] = self.p - 1
                    for j in range(2*(self.p - 1) + 1):
                        self.x_int[i, j]  = (self.T[i + 1 + int(j/2)] + self.T[i + 1 + int((j + 1)/2)])/2

                
                # local coefficient index
                if i > 0:
                    for il in range(self.n_int_locbf_D):
                        k_glob_new = self.int_global_D[i, il]
                        bol = (k_glob_new == self.int_global_D[i - 1])

                        if np.any(bol):
                            self.int_loccof_D[i, il] = self.int_loccof_D[i - 1, np.where(bol)[0][0]] + 1
                            
                        if (k_glob_new >= n_lambda_int - self.p - (self.p - 2)) and (self.int_loccof_D[i, il] == 0):
                            self.int_loccof_D[i, il] = self.int_add_D[counter_D]
                            counter_D += 1
                            
                    for il in range(self.n_int_locbf_N):
                        k_glob_new = self.int_global_N[i, il]
                        bol = (k_glob_new == self.int_global_N[i - 1])

                        if np.any(bol):
                            self.int_loccof_N[i, il] = self.int_loccof_N[i - 1, np.where(bol)[0][0]] + 1

                        if (k_glob_new >= n_lambda_int - self.p - (self.p - 2)) and (self.int_loccof_N[i, il] == 0):
                            self.int_loccof_N[i, il] = self.int_add_N[counter_N]
                            counter_N += 1
                    
            
              
        else:
            for i in range(n_lambda_int):
                
                # maximum number of non-vanishing coefficients
                self.n_int_nvcof_D = 2*self.p - 2
                self.n_int_nvcof_N = 2*self.p - 1
                
                # shift local coefficients --> global coefficients (D)
                self.int_shift_D = np.arange(self.NbaseN) - (self.p - 2)
                
                # shift local coefficients --> global coefficients (N)
                self.int_shift_N = np.arange(self.NbaseN) - (self.p - 1)
                
                self.int_global_N[i] = (np.arange(self.n_int_locbf_N) + i - (self.p - 1))%self.NbaseN
                self.int_global_D[i] = (np.arange(self.n_int_locbf_D) + i - (self.p - 1))%self.NbaseD 
                self.int_loccof_N[i] =  np.arange(self.n_int_locbf_N - 1, -1, -1)
                self.int_loccof_D[i] =  np.arange(self.n_int_locbf_D - 1, -1, -1)
                
                if self.p == 1:
                    self.x_int_indices[i] = i
                else:
                    self.x_int_indices[i] = np.arange(self.n_int) + 2*(i - (self.p - 1))
                
                self.coeffi_indices[i] = 0
                
                for j in range(2*(self.p - 1) + 1):
                    self.x_int[i, j] = ((self.T[i + 1 + int(j/2)] + self.T[i + 1 + int((j + 1)/2)])/2)%1.
                    
                    
        # set histopolation points, quadrature points and weights
        n_lambda_his        = np.copy(self.NbaseD)
        
        self.n_his          = 2*self.p           # number of histopolation intervals
        self.n_his_locbf_N  = 2*self.p           # number of non-vanishing N bf in histopolation interval
        self.n_his_locbf_D  = 2*self.p - 1       # number of non-vanishing D bf in histopolation interval
        
        self.x_his          = np.zeros((n_lambda_his, self.n_his + 1)    , dtype=float)    # histopolation points
        
        self.his_global_N   = np.zeros((n_lambda_his, self.n_his_locbf_N), dtype=int)
        self.his_global_D   = np.zeros((n_lambda_his, self.n_his_locbf_D), dtype=int)
        
        self.his_loccof_N   = np.zeros((n_lambda_his, self.n_his_locbf_N), dtype=int)
        self.his_loccof_D   = np.zeros((n_lambda_his, self.n_his_locbf_D), dtype=int)
        
        self.x_his_indices  = np.zeros((n_lambda_his, self.n_his)        , dtype=int)
        
        self.coeffh_indices = np.zeros( n_lambda_his                     , dtype=int)
        
        
        if self.bc == False:
            
            # maximum number of non-vanishing coefficients
            self.n_his_nvcof_D = 3*self.p - 2
            self.n_his_nvcof_N = 3*self.p - 1
            
            # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
            self.his_add_D = np.arange(self.n_his - 2) + 1
            self.his_add_N = np.arange(self.n_his - 1) + 1

            counter_D = 0
            counter_N = 0
            
            # shift local coefficients --> global coefficients (D)
            self.his_shift_D = np.arange(self.NbaseD) - (self.p - 1)
            self.his_shift_D[:2*self.p - 1] = 0
            self.his_shift_D[-(2*self.p - 1):] = self.his_shift_D[-(2*self.p - 1)]
            
            # shift local coefficients --> global coefficients (N)
            self.his_shift_N = np.arange(self.NbaseN) -  self.p
            self.his_shift_N[:2*self.p]  = 0
            self.his_shift_N[-2*self.p:] = self.his_shift_N[-2*self.p]
            
            counter_coeffh = np.copy(self.p)
            
            for i in range(n_lambda_his):
           
                # left boundary region
                if  i < self.p - 1:
                    self.his_global_N[i] = np.arange(self.n_his_locbf_N)
                    self.his_global_D[i] = np.arange(self.n_his_locbf_D)
                    
                    self.x_his_indices[i] = np.arange(self.n_his)
                    self.coeffh_indices[i] = i
                    for j in range(2*self.p + 1):
                        xi                =  self.p - 1
                        self.x_his[i, j]  = (self.T[xi + 1 + int(j/2)] + self.T[xi + 1 + int((j + 1)/2)])/2

                # right boundary region
                elif i > n_lambda_his - self.p:
                    self.his_global_N[i] = np.arange(self.n_his_locbf_N) + n_lambda_his - self.p - (self.p - 1)
                    self.his_global_D[i] = np.arange(self.n_his_locbf_D) + n_lambda_his - self.p - (self.p - 1)
                    
                    self.x_his_indices[i] = np.arange(self.n_his) + 2*(n_lambda_his - self.p - (self.p - 1))
                    self.coeffh_indices[i] = counter_coeffh
                    counter_coeffh += 1
                    for j in range(2*self.p + 1):
                        xi                =  n_lambda_his - self.p
                        self.x_his[i, j]  = (self.T[xi + 1 + int(j/2)] + self.T[xi + 1 + int((j + 1)/2)])/2
            
                # interior
                else:
                    self.his_global_N[i] = np.arange(self.n_his_locbf_N) + i - (self.p - 1)
                    self.his_global_D[i] = np.arange(self.n_his_locbf_D) + i - (self.p - 1)
                    
                    self.x_his_indices[i] = np.arange(self.n_his) + 2*(i - (self.p - 1))
                    self.coeffh_indices[i] = self.p - 1
                    for j in range(2*self.p + 1):
                        self.x_his[i, j]  = (self.T[i + 1 + int(j/2)] + self.T[i + 1 + int((j + 1)/2)])/2

                
                # local coefficient index
                if i > 0:
                    for il in range(self.n_his_locbf_D):
                        k_glob_new = self.his_global_D[i, il]
                        bol = (k_glob_new == self.his_global_D[i - 1])

                        if np.any(bol):
                            self.his_loccof_D[i, il] = self.his_loccof_D[i - 1, np.where(bol)[0][0]] + 1
                            
                        if (k_glob_new >= n_lambda_his - self.p - (self.p - 2)) and (self.his_loccof_D[i, il] == 0):
                            self.his_loccof_D[i, il] = self.his_add_D[counter_D]
                            counter_D += 1
                            
                    for il in range(self.n_his_locbf_N):
                        k_glob_new = self.his_global_N[i, il]
                        bol = (k_glob_new == self.his_global_N[i - 1])

                        if np.any(bol):
                            self.his_loccof_N[i, il] = self.his_loccof_N[i - 1, np.where(bol)[0][0]] + 1

                        if (k_glob_new >= n_lambda_his - self.p - (self.p - 2)) and (self.his_loccof_N[i, il] == 0):
                            self.his_loccof_N[i, il] = self.his_add_N[counter_N]
                            counter_N += 1
                
            # quadrature points and weights
            self.pts, self.wts = bsp.quadrature_grid(np.unique(self.x_his.flatten()), self.pts_loc, self.wts_loc)
        
        else:
            for i in range(n_lambda_his):
                
                # maximum number of non-vanishing coefficients
                self.n_his_nvcof_D = 2*self.p - 1
                self.n_his_nvcof_N = 2*self.p
                
                # shift local coefficients --> global coefficients (D)
                self.his_shift_D = np.arange(self.NbaseD) - (self.p - 1)
                
                # shift local coefficients --> global coefficients (N)
                self.his_shift_N = np.arange(self.NbaseD) -  self.p
                
                self.his_global_N[i] = (np.arange(self.n_his_locbf_N) + i - (self.p - 1))%self.NbaseN
                self.his_global_D[i] = (np.arange(self.n_his_locbf_D) + i - (self.p - 1))%self.NbaseD 
                self.his_loccof_N[i] =  np.arange(self.n_his_locbf_N - 1, -1, -1)
                self.his_loccof_D[i] =  np.arange(self.n_his_locbf_D - 1, -1, -1)
                
                self.x_his_indices[i] = np.arange(self.n_his) + 2*(i - (self.p - 1))
                self.coeffh_indices[i] = 0
                for j in range(2*self.p + 1):
                    self.x_his[i, j] = (self.T[i + 1 + int(j/2)] + self.T[i + 1 + int((j + 1)/2)])/2
            
            # quadrature points and weights
            self.pts, self.wts = bsp.quadrature_grid(np.append(np.unique(self.x_his.flatten()%1.), 1.), self.pts_loc, self.wts_loc)
            
            
    # quasi interpolation
    def pi_0(self, fun):

        lambdas = np.zeros(self.NbaseN, dtype=float)
        
        # evaluate function at interpolation points
        mat_f = fun(np.unique(self.x_int.flatten()))
        
        for i in range(self.NbaseN):
            for j in range(self.n_int):
                lambdas[i] += self.coeff_i[self.coeffi_indices[i], j] * mat_f[self.x_int_indices[i, j]]

        return lambdas
    
    
    # quasi histopolation
    def pi_1(self, fun):
        
        lambdas = np.zeros(self.NbaseD, dtype=float)
        
        # evaluate function at quadrature points
        mat_f = fun(self.pts)
        
        for i in range(self.NbaseD):
            for j in range(self.n_his):
                
                f_int = 0.
                
                for q in range(self.n_quad):
                    f_int += self.wts[self.x_his_indices[i, j], q] * mat_f[self.x_his_indices[i, j], q]

                lambdas[i] += self.coeff_h[self.coeffh_indices[i], j] * f_int
                
        return lambdas
    
    
# ======================= 3d ====================================
class projectors_local_3d:
    """
    Local commuting projectors pi_0, pi_1, pi_2 and pi_3 in 3d.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 3d tensor product space of B-splines
        
    n_quad : list of ints
        number of quadrature points per integration interval for histopolations
    """
    
    
    def __init__(self, tensor_space, n_quad):
        
        self.T       = tensor_space.T       # knot vector
        self.p       = tensor_space.p       # spline degree
        self.bc      = tensor_space.bc      # boundary conditions
        
        self.Nel     = tensor_space.Nel     # number of elements
        self.NbaseN  = tensor_space.NbaseN  # number of basis functions (N)
        self.NbaseD  = tensor_space.NbaseD  # number of basis functions (D)
        
        self.n_quad  = n_quad               # number of quadrature point per integration interval
        
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [np.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [np.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]
        
        
        # set interpolation and histopolation coefficients
        self.coeff_i = [0, 0, 0]
        self.coeff_h = [0, 0, 0]

        for a in range(3):
            if self.bc[a] == True:
                self.coeff_i[a] = np.zeros((1, 2*self.p[a] - 1), dtype=float, order='F')
                self.coeff_h[a] = np.zeros((1, 2*self.p[a])    , dtype=float, order='F')


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
                self.coeff_i[a] = np.zeros((2*self.p[a] - 1, 2*self.p[a] - 1), dtype=float, order='F')
                self.coeff_h[a] = np.zeros((2*self.p[a] - 1, 2*self.p[a])    , dtype=float, order='F')

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
                        
                        if self.p[a] == 1:
                            self.x_int_indices[a][i] = i
                        else:
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
                    
                    
                    if self.p[a] == 1:
                        self.x_int_indices[a][i] = i
                    else:
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
                self.pts[a], self.wts[a] = bsp.quadrature_grid(np.unique(self.x_his[a].flatten()), self.pts_loc[a], self.wts_loc[a])
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
                self.pts[a], self.wts[a] = bsp.quadrature_grid(np.append(np.unique(self.x_his[a].flatten()%1.), 1.), self.pts_loc[a], self.wts_loc[a])
                self.pts[a] = np.asfortranarray(self.pts[a])
                self.wts[a] = np.asfortranarray(self.wts[a])
                
                
    
    # projector on space V0 (interpolation)
    def pi_0(self, fun, *args):
        """
        Local projector of the discrete space V0.
        
        Parameters
        ----------
        fun : callable
            the function (0-form) to be projected. If None, the function is called internally from hylife.interface.py, where args[0] selects the function, args[1] the mapping and args[2] is the parameters list for the mapping.
            
        Returns
        -------
        lambdas : array_like
            the coefficients in V0 corresponding to the projected function
        """
            
        # interpolation points
        x_int1   = np.unique(self.x_int[0].flatten())
        x_int2   = np.unique(self.x_int[1].flatten())
        x_int3   = np.unique(self.x_int[2].flatten())
        
        n_unique = [x_int1.size, x_int2.size, x_int3.size]
        
        # evaluation of function at interpolation points
        if fun == None:
            mat_f = np.zeros((n_unique[0], n_unique[1], n_unique[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique, x_int1, x_int2, x_int3, mat_f, kind_fun=args[0], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz = np.meshgrid(x_int1, x_int2, x_int3, indexing='ij')
            mat_f      = np.asfortranarray(fun(xx, yy, zz))  
            
        # coefficients
        lambdas  = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi0_3d(self.NbaseN, self.p, self.coeff_i[0], self.coeff_i[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], self.x_int_indices[0], self.x_int_indices[1], self.x_int_indices[2], mat_f, lambdas)
                                
        return lambdas
    
    
    # projector on space V1 ([histopolation, interpolation, interpolation], [interpolation, histopolation, interpolation], [interpolation, interpolation, histopolation])
    def pi_1(self, fun, *args):
        """
        Local projector of the discrete space V1.
        
        Parameters
        ----------
        fun : list of callables
            the functions (components of a 1-form) to be projected. If None, the function is called internally from hylife.interface.py, where args[0] selects the function, args[1] the mapping and args[2] is the parameters list for the mapping.
            
        Returns
        -------
        lambdas : list of array_like
            the coefficients in V1 corresponding to the projected function
        """
        
        # interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [self.pts[0].flatten().size, x_int2.size, x_int3.size]
        n_unique2 = [x_int1.size, self.pts[1].flatten().size, x_int3.size]
        n_unique3 = [x_int1.size, x_int2.size, self.pts[2].flatten().size]
        
        
        # ======== 1 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[0] == None:
            mat_f = np.zeros((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique1, self.pts[0].flatten(), x_int2, x_int3, mat_f, kind_fun=args[0][0], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(self.pts[0].flatten(), x_int2, x_int3, indexing='ij')
            mat_f       = np.asfortranarray(fun[0](xx, yy, zz))
        
        # coefficients
        lambdas1  = np.zeros((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi11_3d([self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], self.p, self.n_quad, self.coeff_h[0], self.coeff_i[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], self.x_his_indices[0], self.x_int_indices[1], self.x_int_indices[2], self.wts[0], mat_f.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]), lambdas1)
        
        
        # ======== 2 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[1] == None:
            mat_f = np.zeros((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique2, x_int1, self.pts[1].flatten(), x_int3, mat_f, kind_fun=args[0][1], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(x_int1, self.pts[1].flatten(), x_int3, indexing='ij')
            mat_f       = np.asfortranarray(fun[1](xx, yy, zz))
        
        # coefficients
        lambdas2  = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi12_3d([self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], self.p, self.n_quad, self.coeff_i[0], self.coeff_h[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], self.x_int_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[1], mat_f.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]), lambdas2)
        
        
        # ======== 3 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[2] == None:
            mat_f = np.zeros((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique3, x_int1, x_int2, self.pts[2].flatten(), mat_f, kind_fun=args[0][2], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(x_int1, x_int2, self.pts[2].flatten(), indexing='ij')
            mat_f       = np.asfortranarray(fun[2](xx, yy, zz))
        
        # coefficients
        lambdas3  = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi13_3d([self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], self.p, self.n_quad, self.coeff_i[0], self.coeff_i[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], self.x_int_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[2], mat_f.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size), lambdas3)
        
        return lambdas1, lambdas2, lambdas3
        
        
    # projector on space V2 ([interpolation, histopolation, histopolation], [histopolation, interpolation, histopolation], [histopolation, histopolation, interpolation])
    def pi_2(self, fun, *args):
        """
        Local projector of the discrete space V2.
        
        Parameters
        ----------
        fun : list of callables
            the functions (components of a 2-form) to be projected. If None, the function is called internally from hylife.interface.py, where args[0] selects the function, args[1] the mapping and args[2] is the parameters list for the mapping.
            
        Returns
        -------
        lambdas : list of array_like
            the coefficients in V2 corresponding to the projected function
        """
        
        
        
        # interpolation points
        x_int1    = np.unique(self.x_int[0].flatten())
        x_int2    = np.unique(self.x_int[1].flatten())
        x_int3    = np.unique(self.x_int[2].flatten())
        
        n_unique1 = [x_int1.size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, x_int2.size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, x_int3.size]
        
        # ======== 1 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[0] == None:
            mat_f = np.zeros((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique1, x_int1, self.pts[1].flatten(), self.pts[2].flatten(), mat_f, kind_fun=args[0][0], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(x_int1, self.pts[1].flatten(), self.pts[2].flatten(), indexing='ij')
            mat_f       = np.asfortranarray(fun[0](xx, yy, zz))
        
        # coefficients
        lambdas1  = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi21_3d([self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.p, self.n_quad, self.coeff_i[0], self.coeff_h[1], self.coeff_h[2], self.coeffi_indices[0], self.coeffh_indices[1], self.coeffh_indices[2], self.x_int_indices[0], self.x_his_indices[1], self.x_his_indices[2], self.wts[1], self.wts[2], mat_f.reshape(n_unique1[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, self.pts[2][:, 0].size, self.pts[2][0, :].size), lambdas1)
        
        
        # ======== 2 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[1] == None:
            mat_f = np.zeros((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique2, self.pts[0].flatten(), x_int2, self.pts[2].flatten(), mat_f, kind_fun=args[0][1], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(self.pts[0].flatten(), x_int2, self.pts[2].flatten(), indexing='ij')
            mat_f       = np.asfortranarray(fun[1](xx, yy, zz))
        
        # coefficients
        lambdas2  = np.zeros((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float, order='F')
        
        ker_loc.kernel_pi22_3d([self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.p, self.n_quad, self.coeff_h[0], self.coeff_i[1], self.coeff_h[2], self.coeffh_indices[0], self.coeffi_indices[1], self.coeffh_indices[2], self.x_his_indices[0], self.x_int_indices[1], self.x_his_indices[2], self.wts[0], self.wts[2], mat_f.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique2[1], self.pts[2][:, 0].size, self.pts[2][0, :].size), lambdas2)
        
        
        # ======== 3 - component ========
        
        # evaluation of function at interpolation/quadrature points
        if fun[2] == None:
            mat_f = np.zeros((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique3, self.pts[0].flatten(), self.pts[1].flatten(), x_int3, mat_f, kind_fun=args[0][2], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz  = np.meshgrid(self.pts[0].flatten(), self.pts[1].flatten(), x_int3, indexing='ij')
            mat_f       = np.asfortranarray(fun[2](xx, yy, zz))
        
        # coefficients
        lambdas3  = np.zeros((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float, order='F')
    
        ker_loc.kernel_pi23_3d([self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.p, self.n_quad, self.coeff_h[0], self.coeff_h[1], self.coeff_i[2], self.coeffh_indices[0], self.coeffh_indices[1], self.coeffi_indices[2], self.x_his_indices[0], self.x_his_indices[1], self.x_int_indices[2], self.wts[0], self.wts[1], mat_f.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique3[2]), lambdas3)
        
        return lambdas1, lambdas2, lambdas3
    
    
    # projector on space V0 (histopolation)
    def pi_3(self, fun, *args):
        """
        Local projector of the discrete space V3.
        
        Parameters
        ----------
        fun : callable
            the function (component of a 3-form) to be projected. If None, the function is called internally from hylife.interface.py, where args[0] selects the function, args[1] the mapping and args[2] is the parameters list for the mapping.
            
        Returns
        -------
        lambdas : array_like
            the coefficients in V3 corresponding to the projected function
        """
        
        n_unique = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        
        # evaluation of function at quadrature points
        if fun == None:
            mat_f = np.zeros((n_unique[0], n_unique[1], n_unique[2]), dtype=float, order='F')
            ker_loc_eva.kernel_eva(n_unique, self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten(), mat_f, kind_fun=args[0], kind_map=args[1], params=args[2])
        else:
            xx, yy, zz = np.meshgrid(self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten(), indexing='ij')
            mat_f      = np.asfortranarray(fun(xx, yy, zz))
            
        # coefficients
        lambdas  = np.zeros((self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]), dtype=float, order='F')
            
        ker_loc.kernel_pi3_3d(self.NbaseD, self.p, self.n_quad, self.coeff_h[0], self.coeff_h[1], self.coeff_h[2], self.coeffh_indices[0], self.coeffh_indices[1], self.coeffh_indices[2], self.x_his_indices[0], self.x_his_indices[1], self.x_his_indices[2], self.wts[0], self.wts[1], self.wts[2], mat_f.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, self.pts[1][:, 0].size, self.pts[1][0, :].size, self.pts[2][:, 0].size, self.pts[2][0, :].size), lambdas)
                                
        return lambdas