# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for local projections for linear ideal mhd in 3d based on quasi-interpolation
"""


import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp

import hylife.utilitis_FEEC.projectors.kernels_projectors_local_eva as ker_loc_eva
import hylife.utilitis_FEEC.projectors.kernels_projectors_local_mhd as ker_loc


class projectors_local_mhd:
    """
    Local commuting projections of various terms in linear ideal MHD.
    
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
        n_lambda_int        = [NbaseN for NbaseN in self.NbaseN] # number of coefficients in space V0 
        self.n_int          = [2*p - 1 for p in self.p]          # number of interpolation points (1, 3, 5, 7, ...)
        
        
        self.n_int_locbf_N = [0, 0, 0]
        self.n_int_locbf_D = [0, 0, 0]
        
        for a in range(3):
        
            if self.p[a] == 1:
                self.n_int_locbf_N[a]  = 2                # number of non-vanishing N bf in interpolation interval (2, 3, 5, 7)
                self.n_int_locbf_D[a]  = 1                # number of non-vanishing D bf in interpolation interval (1, 2, 4, 6)

            else:
                self.n_int_locbf_N[a]  = 2*self.p[a] - 1  # number of non-vanishing N bf in interpolation interval (2, 3, 5, 7)
                self.n_int_locbf_D[a]  = 2*self.p[a] - 2  # number of non-vanishing D bf in interpolation interval (1, 2, 4, 6)
        
        
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
                if self.p[a] == 1:
                    self.n_int_nvcof_D[a] = 2
                    self.n_int_nvcof_N[a] = 2
                    
                else:
                    self.n_int_nvcof_D[a] = 3*self.p[a] - 3
                    self.n_int_nvcof_N[a] = 3*self.p[a] - 2
                
                # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
                self.int_add_D[a] = np.arange(self.n_int[a] - 2) + 1
                self.int_add_N[a] = np.arange(self.n_int[a] - 1) + 1
                
                counter_D = 0
                counter_N = 0
                
                # shift local coefficients --> global coefficients (D)
                if self.p[a] == 1:
                    self.int_shift_D[a] = np.arange(self.NbaseD[a])
                else:
                    self.int_shift_D[a] = np.arange(self.NbaseD[a]) - (self.p[a] - 2)
                    self.int_shift_D[a][:2*self.p[a] - 2] = 0
                    self.int_shift_D[a][-(2*self.p[a] - 2):] = self.int_shift_D[a][-(2*self.p[a] - 2)]

                # shift local coefficients --> global coefficients (N)
                if self.p[a] == 1:
                    self.int_shift_N[a]     = np.arange(self.NbaseN[a])
                    self.int_shift_N[a][-1] = self.int_shift_N[a][-2]
                    
                else:
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
                        if self.p[a] == 1:
                            self.int_global_N[a][i] = np.arange(self.n_int_locbf_N[a]) + i
                            self.int_global_D[a][i] = np.arange(self.n_int_locbf_D[a]) + i

                            self.int_global_N[a][-1] = self.int_global_N[a][-2]
                            self.int_global_D[a][-1] = self.int_global_D[a][-2]
                            
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
                    if self.p[a] == 1:
                        self.int_loccof_N[a][i]  = np.array([0, 1])
                        self.int_loccof_D[a][-1] = np.array([1])
                
                    else:
                    
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
                
                # maximum number of non-vanishing coefficients
                if self.p[a] == 1:
                    self.n_int_nvcof_D[a] = 2*self.p[a] - 1
                    self.n_int_nvcof_N[a] = 2*self.p[a]

                else:
                    self.n_int_nvcof_D[a] = 2*self.p[a] - 2
                    self.n_int_nvcof_N[a] = 2*self.p[a] - 1

                # shift local coefficients --> global coefficients
                if self.p[a] == 1:
                    self.int_shift_D[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 1)
                    self.int_shift_N[a] = np.arange(self.NbaseN[a]) - (self.p[a])
                else:
                    self.int_shift_D[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 2)
                    self.int_shift_N[a] = np.arange(self.NbaseN[a]) - (self.p[a] - 1)
                
                
                for i in range(n_lambda_int[a]):

                    # global indices of non-vanishing basis functions and position of coefficients in final matrix
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
                        
                        
            # identify unique interpolation points to save memory
            self.x_int[a] = np.unique(self.x_int[a].flatten())
        
        
        # set histopolation points, quadrature points and weights
        n_lambda_his = [np.copy(NbaseD) for NbaseD in self.NbaseD] # number of coefficients in space V1
        
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
                
                # maximum number of non-vanishing coefficients
                self.n_his_nvcof_D[a] = 2*self.p[a] - 1
                self.n_his_nvcof_N[a] = 2*self.p[a]

                # shift local coefficients --> global coefficients (D)
                self.his_shift_D[a] = np.arange(self.NbaseD[a]) - (self.p[a] - 1)

                # shift local coefficients --> global coefficients (N)
                self.his_shift_N[a] = np.arange(self.NbaseD[a]) -  self.p[a]
                                
                for i in range(n_lambda_his[a]):

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
                
                
        # evaluate N basis functions at interpolation and quadrature points
        self.basisN_int = [np.asfortranarray(bsp.collocation_matrix(T, p, x_int, bc)) for T, p, x_int, bc in zip(self.T, self.p, self.x_int, self.bc)]
        
        self.basisN_his = [np.asfortranarray(bsp.collocation_matrix(T, p, pts.flatten(), bc).reshape(pts[:, 0].size, pts[0, :].size, NbaseN)) for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)] 
        
        # evaluate D basis functions at interpolation and quadrature points
        self.basisD_int = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, x_int, bc, normalize=True)) for T, p, x_int, bc in zip(self.T, self.p, self.x_int, self.bc)]
        
        self.basisD_his = [np.asfortranarray(bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(pts[:, 0].size, pts[0, :].size, NbaseD)) for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)] 