# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Module to handle mapped 3d domains.
"""


import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu

import hylife.geometry.mappings_3d       as mapping
import hylife.geometry.pullback_3d       as pb
import hylife.geometry.pushforward_3d    as pf
import hylife.linear_algebra.linalg_kron as linalg

import hylife.utilitis_FEEC.bsplines  as bsp



# ==================================================
def spline_interpolation_nd(p, grids_1d, values):
    '''
    nd spline interpolation with discrete input (nonuniform).

    The knot vector for the clamped spline interpolant is constructed from grids_1d.

    Parameters:
    -----------
        p : list of length n
            spline degree

        grids_1d : list of n 1d np.arrays
            interpolation points

        values: nd np.array
            function values at interpolation points. values.shape = (grid1.size, ..., gridn.size)

    Returns:
    --------
        coeffs : nd np.array
            spline coefficients

        T : list
            Knot vector of spline interpolant
    '''

    # dimension check
    for sh, x_grid in zip(values.shape, grids_1d):
        assert  sh == x_grid.size

    # list of break point arrays
    breaks = []

    for x_grid, p_i in zip(grids_1d, p):

        # dimension of the 1d spline spaces: dim = breaks.size - 1 + p = x_grid.size
        if p_i == 1:
            breaks.append(x_grid)
        elif p_i%2 == 0:
            breaks.append( x_grid[p_i//2 - 1:-p_i//2].copy() )
        else:
            breaks.append( x_grid[(p_i - 1)//2:-(p_i - 1)//2].copy() )

        # cells must be in interval [0, 1] 
        breaks[-1][0]  = 0.
        breaks[-1][-1] = 1.

    # interpolation with clamped splines (periodic=False)
    T     = [bsp.make_knots(breaks_i, p_i, periodic=False) for breaks_i, p_i in zip(breaks, p)]
    I_mat = [bsp.collocation_matrix(T_i, p_i, grids_1d_i, periodic=False) for T_i, p_i, grids_1d_i in zip(T, p, grids_1d)]

    I_LU  = [splu(spa.csc_matrix(I_mat_i)) for I_mat_i in I_mat] 

    # dimension check
    for I, x_grid in zip(I_mat, grids_1d):
        assert I.shape[0] == x_grid.size
        assert I.shape[0] == I.shape[1]

    # solve system
    if len(p) == 1:
        return I_LU[0].solve(values), T
    if len(p) == 2:
        return linalg.kron_lusolve_2d(I_LU, values), T
    elif len(p) == 3:
        return linalg.kron_lusolve_3d(I_LU, values), T
    else:
        raise AssertionError("Only dimensions < 4 are supported.")



# ==================================================
def interp_mapping(Nel, p, spl_kind, X, Y, Z=None):
    '''
    Interpolates the mapping (eta1, eta2, eta3) --> (X, Y, Z) on the given spline space.

    Parameters:
    -----------
        Nel, p, spl_kind: array-like
            defining the spline space

        X, Y: callable
            either X(eta1, eta2) in 2D or X(eta1, eta2, eta3) in 3D

        Z: callable Z(eta1, eta2, eta3)

    Returns:
    --------
        cx, cy (, cz): np.array
            spline coefficients
    '''
    
    # number of basis functions
    NbaseN = [Nel + p - kind*p for Nel, p, kind in zip(Nel, p, spl_kind)]
    
    # element boundaries
    el_b   = [np.linspace(0., 1., Nel + 1) for Nel in Nel]
    
    # spline knot vectors
    T      = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, p, spl_kind)]
    
    # greville points
    I_pts  = [bsp.greville(T, p, kind) for T, p, kind in zip(T, p, spl_kind)]
    
    # 1D interpolation matrices
    I_mat  = [spa.csc_matrix(bsp.collocation_matrix(T, p, I_pts, kind)) for T, p, I_pts, kind in zip(T, p, I_pts, spl_kind)]
    
    # 2D interpolation
    if len(Nel) == 2:
        I = spa.kron(I_mat[0], I_mat[1], format='csc')
        
        I_pts = np.meshgrid(I_pts[0], I_pts[1], indexing='ij')
        
        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1]).flatten()).reshape(NbaseN[0], NbaseN[1])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1]).flatten()).reshape(NbaseN[0], NbaseN[1])
        
        return cx, cy
    
    # 3D interpolation
    elif len(Nel) == 3:
        I = spa.kron(I_mat[0], spa.kron(I_mat[1], I_mat[2]), format='csc')
        
        I_pts = np.meshgrid(I_pts[0], I_pts[1], I_pts[2], indexing='ij')
        
        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        cz = spa.linalg.spsolve(I, Z(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        
        return cx, cy, cz
    
    else:
        print('wrong number of elements')
        
        return 0.



# ==================================================
class domain:
    '''Defines the mapped domain.

    Available mappings:
    -------------------
        - kind_map = 0  : 3d discrete spline mapping. All information is stored in control points cx, cy, cz. params_map = [].
        - kind_map = 1  : discrete cylinder. 2d discrete spline mapping in xy-plane and analytical in z. params_map = [].
        - kind_map = 2  : discrete torus. 2d discrete spline mapping in xy-plane and analytical in phi. params_map = [].

        - kind_map = 10 : cuboid. params_map = [lx, ly, lz].
        - kind_map = 11 : hollow cylinder. params_map = [a1, a2, lz].
        - kind_map = 12 : colella. params_map = [lx, ly, alpha, lz].
        - kind_map = 13 : othogonal. params_map = [ly, ly, alpha, lz].
        - kind_map = 14 : hollow torus. params_map = [a1, a2, r0].
        - kind_map = 15 : cuboid slice. A cuboid slice of the logical cube with begin and end points given for each axis. params_map = [b1, e1, b2, e2, b3, e3].

    Methods:
    --------
        evaluate(eta1, eta2, eta3, kind_fun)
        push(a, eta1, eta2, eta3, kind_fun)
        pull(a, eta1, eta2, eta3, kind_fun)

    Attributes:
    -----------
        kind_map: integer
            values <10 indicate a spline mapping

        params_map: array-like
            mapping parameters

        Nel, p, NbaseN, T: lists
            usual parameters of spline mapping

        cx, cy, cz: np.array
            spline coefficients

        keys_map: dictionary
            keys point to values for kind_fun in the 'evaluate' method.
            'x' : 1
            'y' : 2
            'z' : 3
            'det_df' : 4 
            'df_11' : 11
            'df_12' : 12
            'df_13' : 13 
            'df_21' : 14
            'df_22' : 15
            'df_23' : 16 
            'df_31' : 17
            'df_32' : 18
            'df_33' : 19 
            'df_inv_11' : 21
            'df_inv_12' : 22
            'df_inv_13' : 23
            'df_inv_21' : 24
            'df_inv_22' : 25
            'df_inv_23' : 26 
            'df_inv_31' : 27
            'df_inv_32' : 28
            'df_inv_33' : 29 
            'g_11' : 31
            'g_12' : 32
            'g_13' : 33
            'g_21' : 34
            'g_22' : 35
            'g_23' : 36
            'g_31' : 37
            'g_32' : 38
            'g_33' : 39
            'g_inv_11' : 41
            'g_inv_12' : 42
            'g_inv_13' : 43
            'g_inv_21' : 44
            'g_inv_22' : 45
            'g_inv_23' : 46
            'g_inv_31' : 47
            'g_inv_32' : 48
            'g_inv_33' : 49

        keys_pull: dictionary
            keys point to possible values for kind_fun in 'pull' method.
            '0_form' : 0
            '3_form' : 3
            '1_form_1' : 11
            '1_form_2' : 12
            '1_form_3' : 13
            '2_form_1' : 21
            '2_form_2' : 22
            '2_form_3' : 23
            'vector_1' : 31
            'vector_2' : 32
            'vector_3' : 33

        keys_push: dictionary
            keys point to possible values for kind_fun in 'push' method.
            '0_form' : 0
            '3_form' : 3
            '1_form_1' : 11
            '1_form_2' : 12
            '1_form_3' : 13
            '2_form_1' : 21
            '2_form_2' : 22
            '2_form_3' : 23
            'vector_1' : 31 
            'vector_2' : 32
            'vector_3' : 33
    '''
    
    def __init__(self, kind_map, params_map=None, Nel=None, p=None, spl_kind=None, cx=None, cy=None, cz=None):
        
        # ====== 3d discrete =======
        if kind_map == 'spline':
            self.kind_map   = 0
            self.params_map = []
            
        # ===== discrete cylinder ==
        elif kind_map == 'spline cylinder':
            self.kind_map   = 1
            self.params_map = []
                
        # ===== discrete torus =====
        elif kind_map == 'spline torus':
            self.kind_map   = 2
            self.params_map = []
        
        # ======== cuboid ==========
        elif kind_map == 'cuboid':
            self.kind_map = 10
            
            if params_map == None:
                self.params_map = [1., 1., 1.]
            else:
                self.params_map = params_map
        
        # ===== hollow cylinder ====    
        elif kind_map == 'hollow cylinder':
            self.kind_map = 11
            
            if params_map == None:
                self.params_map = [0.5, 1., 1.]
            else:
                self.params_map = params_map
            
        # ======= colella ==========
        elif kind_map == 'colella':
            self.kind_map = 12
            
            if params_map == None:
                self.params_map = [1., 1., 0.05, 1.]
            else:
                self.params_map = params_map
                
        # ======= colella ==========
        elif kind_map == 'orthogonal':
            self.kind_map = 13
            
            if params_map == None:
                self.params_map = [1., 1., 0.05, 1.]
            else:
                self.params_map = params_map
                
        # ====== hollow torus ======
        elif kind_map == 'hollow torus':
            self.kind_map = 14
            
            if params_map == None:
                self.params_map = [0.5, 1., 4.]
            else:
                self.params_map = params_map

        # ======== cuboid slice ==========
        elif kind_map == 'cuboid slice':
            self.kind_map = 15
            
            if params_map == None:
                self.params_map = [0., 1., 0., 1., 0., 1.]
            else:
                self.params_map = params_map

        else:
            raise ValueError('specified domain is not implemeted!')
            
        
        # create dummy variables for spline mappings
        if self.kind_map < 10:
            
            self.Nel    =  Nel
            self.p      =  p
            self.NbaseN = [Nel + p - kind*p for Nel, p, kind in zip(self.Nel, self.p, spl_kind)]
            
            el_b        = [np.linspace(0., 1., Nel + 1) for Nel in self.Nel]
            self.T      = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, self.p, spl_kind)]
            
            if self.kind_map == 0:
                
                self.cx =  cx
                self.cy =  cy
                self.cz =  cz
                
            else:
                
                self.Nel    = self.Nel    + [0]
                self.p      = self.p      + [0]
                self.NbaseN = self.NbaseN + [0]
                
                self.T      = self.T      + [np.zeros((1,), dtype=float)]
                
                self.cx     = cx.reshape(cx.shape[0], cx.shape[1], 1)
                self.cy     = cy.reshape(cy.shape[0], cy.shape[1], 1)
                self.cz     = np.zeros((1, 1, 1), dtype=float)
                
            
        # create dummy variables for analytical mappings
        else:
            
            self.Nel    = [0, 0, 0]
            self.p      = [0, 0, 0]
            self.NbaseN = [0, 0, 0]
            
            self.T      = [np.zeros((1,), dtype=float),
                           np.zeros((1,), dtype=float), 
                           np.zeros((1,), dtype=float)]

            self.cx     =  np.zeros((1, 1, 1), dtype=float)
            self.cy     =  np.zeros((1, 1, 1), dtype=float)
            self.cz     =  np.zeros((1, 1, 1), dtype=float)
            
        
        # keys for evaluating mapping related quantities
        self.keys_map  = {
            'x' : 1, 'y' : 2, 'z' : 3, 'det_df' : 4, 
            'df_11' : 11, 'df_12' : 12, 'df_13' : 13, 
            'df_21' : 14, 'df_22' : 15, 'df_23' : 16, 
            'df_31' : 17, 'df_32' : 18, 'df_33' : 19, 
            'df_inv_11' : 21, 'df_inv_12' : 22, 'df_inv_13' : 23,
            'df_inv_21' : 24, 'df_inv_22' : 25, 'df_inv_23' : 26, 
            'df_inv_31' : 27, 'df_inv_32' : 28, 'df_inv_33' : 29, 
            'g_11' : 31, 'g_12' : 32, 'g_13' : 33, 'g_21' : 34,
            'g_22' : 35, 'g_23' : 36, 'g_31' : 37, 'g_32' : 38, 'g_33' : 39, 
            'g_inv_11' : 41, 'g_inv_12' : 42, 'g_inv_13' : 43, 'g_inv_21' : 44,
            'g_inv_22' : 45, 'g_inv_23' : 46, 'g_inv_31' : 47, 'g_inv_32' : 48,
            'g_inv_33' : 49
            }
        
        # keys for performing pull-backs
        self.keys_pull = {
            '0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, 
            '1_form_3' : 13, '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23,
            'vector_1' : 31, 'vector_2' : 32, 'vector_3' : 33
            }
        
        # keys for performing push-forwards
        self.keys_push = {
            '0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, '1_form_3' : 13,
            '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23, 'vector_1' : 31, 
            'vector_2' : 32, 'vector_3' : 33
            }
       
    
    # ================================
    def evaluate(self, eta1, eta2, eta3, kind_fun, kind_eva='meshgrid'):
        '''Evaluate mapping/metric coefficients. 
        Depending on the dimension of eta1 either point-wise, tensor-product (meshgrid) or general.

        Parameters:
        -----------
            eta1, eta2, eta3:   array-like
                logical coordinates at which to evaluate
            kind_fun:   integer
                what metric coefficient to evaluate, see keys_map

        Returns:
        --------
            values: ndarray
                mapping/metric coefficients evaluated at (eta1, eta2, eta3) 
        '''
        
        # point-wise evaluation
        if isinstance(eta1, float):
            values = mapping.mappings_all(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
            
            return values

        # array evaluation
        elif isinstance(eta1, np.ndarray):
            
            # evaluation for point pairs of 1d arrays of same length
            if kind_eva == 'flat':
                assert eta1.ndim == eta2.ndim == eta3.ndim == 1
                assert eta1.size == eta2.size == eta3.size
                
                values = np.empty(eta1.size, dtype=float)
                
                mapping.kernel_evaluate_flat(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
                
                return values
            
            else:

                is_sparse_meshgrid = None

                # tensor-product evaluation
                if eta1.ndim == 1:
                    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij', sparse=True)
                    is_sparse_meshgrid = True

                # general evaluation
                else:
                    # Distinguish if input coordinates are from sparse or dense meshgrid.
                    # Sparse: eta1.shape = (n1,  1,  1)
                    # Dense : eta1.shape = (n1, n2, n3)
                    E1, E2, E3 = eta1, eta2, eta3

                    # `eta1` is a sparse meshgrid.
                    if max(eta1.shape) == eta1.size:
                        is_sparse_meshgrid = True
                    # `eta1` is a dense meshgrid. Process each point as default.
                    else:
                        is_sparse_meshgrid = False

                values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

                if is_sparse_meshgrid:
                    mapping.kernel_evaluate_sparse(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
                else:
                    mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

                return values


    # ================================
    def evaluate_12(self, eta1, eta2, eta3, kind_fun):
        '''Evaluate mapping/metric coefficients in the 12-plane at one point eta3. 
        Depending on the dimension of eta1 either point-wise, tensor-product (meshgrid) or matrix.

        Parameters:
        -----------
            eta1, eta2:   array-like
                logical coordinates in plane at eta3 
            eta3:   float
            kind_fun:   integer
                what metric coefficient to evaluate, see keys_map

        Returns:
        --------
            values: 2d array
                mapping/metric coefficients evaluated at (E1, E2, E3) and then squeezed, where
                - point-wise: E1 = eta1, E2 = eta2, E3 = eta3
                - tensor-product: E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
                - matrix: E1 = eta1[:, :, None], E2 = eta2[:, :, None], E3 = eta3*np.ones(E1.shape)
        '''
        
        # point-wise evaluation
        if isinstance(eta1, float):
            values = mapping.mappings_all(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
        # array evaluation
        elif isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
            
            # general evaluation
            else:
                E1 = eta1[:, :, None]
                E2 = eta2[:, :, None]
                E3 = eta3*np.ones(E1.shape)
                
            values = np.empty((E1.shape[0], E1.shape[1], 1), dtype=float)
                
            mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
                
        else:
            raise ValueError('given evaluation points are in wrong shape')
            
        return np.squeeze(values)


    # ================================
    def evaluate_13(self, eta1, eta2, eta3, kind_fun):
        '''Evaluate mapping/metric coefficients in the 13-plane at one point eta2. 
        Depending on the dimension of eta1 either point-wise, tensor-product (meshgrid) or matrix.

        Parameters:
        -----------
            eta1, eta3:   array-like
                logical coordinates in plane at eta3 
            eta2:   float
            kind_fun:   integer
                what metric coefficient to evaluate, see keys_map

        Returns:
        --------
            values: 2d array
                mapping/metric coefficients evaluated at (E1, E2, E3) and then squeezed, where
                - point-wise: E1 = eta1, E2 = eta2, E3 = eta3
                - tensor-product: E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
                - matrix: E1 = eta1[:, None, :], E2 = eta2*np.ones(E1.shape), E3 = eta3[:, None, :]
        '''

        # point-wise evaluation
        if isinstance(eta1, float):
            values = mapping.mappings_all(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
        # array evaluation
        elif isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
            
            # general evaluation
            else:
                E1 = eta1[:, None, :]
                E2 = eta2*np.ones(E1.shape)
                E3 = eta3[:, None, :]

            values = np.empty((E1.shape[0], 1, E1.shape[2]), dtype=float)
                
            mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
                
        else:
            raise ValueError('given evaluation points are in wrong shape')
            
        return np.squeeze(values)


    # ================================
    def evaluate_23(self, eta1, eta2, eta3, kind_fun):
        '''Evaluate mapping/metric coefficients in the 23-plane at one point eta1. 
        Depending on the dimension of eta1 either point-wise, tensor-product (meshgrid) or matrix.

        Parameters:
        -----------
            eta2, eta3:   array-like
                logical coordinates in plane at eta3 
            eta1:   float
            kind_fun:   integer
                what metric coefficient to evaluate, see keys_map

        Returns:
        --------
            values: 2d array
                mapping/metric coefficients evaluated at (E1, E2, E3) and then squeezed, where
                - point-wise: E1 = eta1, E2 = eta2, E3 = eta3
                - tensor-product: E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
                - matrix: E1 = eta1*np.ones(E2.shape), E2 = eta2[None, :, :], E3 = eta3[None, :, :]
        '''

        # point-wise evaluation
        if isinstance(eta2, float):
            values = mapping.mappings_all(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
        # array evaluation
        elif isinstance(eta2, np.ndarray):
            
            # tensor-product evaluation
            if eta2.ndim == 1:
                E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij')
            
            # general evaluation
            else:
                E2 = eta2[None, :, :]
                E3 = eta3[None, :, :]
                E1 = eta1*np.ones(E2.shape)

            values = np.empty((1, E1.shape[1], E1.shape[2]), dtype=float)
                
            mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
                
        else:
            raise ValueError('given evaluation points are in wrong shape')
            
        return np.squeeze(values)

       
    # ================================
    def pull(self, a, eta1, eta2, eta3, kind_fun):
        '''Pullback of p-forms. 
        Depending on the dimension of eta1 either point-wise, tensor-product or general.

        Parameters:
        -----------
            a:  callable or array-like
                the function a(x, y, z) to be pulled back (can be one component of a p-form)
            eta1, eta2, eta3:   array-like
                logical coordinates to which to pull back
            kind_fun:   integer
                which p-form pull back to apply, see keys_pull

        Returns:
        --------
            values: ndarray
                pullback of p-form (component) evaluated at (eta1, eta2, eta3)
        '''
        
        # point-wise evaluation
        if isinstance(eta1, float):
            
            if isinstance(a, list):
                
                if callable(a[0]):
                    
                    x = self.evaluate(eta1, eta2, eta3, 'x')
                    y = self.evaluate(eta1, eta2, eta3, 'y')
                    z = self.evaluate(eta1, eta2, eta3, 'z')
                    
                    a_in = [a[0](x, y, z), a[1](x, y, z), a[2](x, y, z)]
                else:
                    a_in =  a
                    
            else:
                
                if callable(a):
                    
                    x = self.evaluate(eta1, eta2, eta3, 'x')
                    y = self.evaluate(eta1, eta2, eta3, 'y')
                    z = self.evaluate(eta1, eta2, eta3, 'z')
                    
                    print(x, y, z)
                    
                    a_in = [a(x, y, z)]
                else:
                    a_in = [a]
                    
            values = pb.pull_all(a_in, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
            
        
        # array evaluation
        elif isinstance(eta1, np.ndarray):

            is_sparse_meshgrid = None

            # tensor-product evaluation
            if eta1.ndim == 1:
                E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij', sparse=True)
                is_sparse_meshgrid = True

            # general evaluation
            else:
                # Distinguish if input coordinates are from sparse or dense meshgrid.
                # Sparse: eta1.shape = (n1,  1,  1)
                # Dense : eta1.shape = (n1, n2, n3)
                E1, E2, E3 = eta1, eta2, eta3

                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    is_sparse_meshgrid = True
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    is_sparse_meshgrid = False

            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

            if isinstance(a, list):
                
                if callable(a[0]):
                    
                    X = self.evaluate(E1, E2, E3, 'x')
                    Y = self.evaluate(E1, E2, E3, 'y')
                    Z = self.evaluate(E1, E2, E3, 'z')
                    
                    a_in = np.array([a[0](X, Y, Z), a[1](X, Y, Z), a[2](X, Y, Z)])
                else:
                    a_in = np.array(a)
                    
            else:
                
                if callable(a):
                    
                    X = self.evaluate(E1, E2, E3, 'x')
                    Y = self.evaluate(E1, E2, E3, 'y')
                    Z = self.evaluate(E1, E2, E3, 'z')
                    
                    a_in = np.array([a(X, Y, Z)])
                else:
                    a_in = np.array([a])

            if is_sparse_meshgrid:
                pb.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            else:
                pb.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

        else:
            raise ValueError('given evaluation points are in wrong shape')
    
        return values
        
    
    # ================================
    def push(self, a, eta1, eta2, eta3, kind_fun):
        '''Push-forward of p-forms. 
        Depending on the dimension of eta1 either point-wise, tensor-product or general.
        
        Parameters:
        -----------
            a:  callable or array-like
                the function a(eta1, eta2, eta3) to be pushed forward (can be one component of a p-form)
            eta1, eta2, eta3:   array-like
                logical coordinates at which to push forward
            kind_fun:   integer
                which p-form push forward to apply, see keys_push

        Returns:
        --------
            values: ndarray
                push forward of p-form (component) evaluated at (eta1, eta2, eta3)
        '''
        
        # point-wise evaluation
        if isinstance(eta1, float):
            
            if isinstance(a, list):
                
                if callable(a[0]):
                    a_in = [a[0](eta1, eta2, eta3), a[1](eta1, eta2, eta3), a[2](eta1, eta2, eta3)]
                else:
                    a_in =  a
                    
            else:
                
                if callable(a):
                    a_in = [a(eta1, eta2, eta3)]
                else:
                    a_in = [a]
                    
            values = pf.push_all(a_in, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
            
        
        # array evaluation
        elif isinstance(eta1, np.ndarray):

            is_sparse_meshgrid = None

            # tensor-product evaluation
            if eta1.ndim == 1:
                E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing='ij', sparse=True)
                is_sparse_meshgrid = True

            # general evaluation
            else:
                # Distinguish if input coordinates are from sparse or dense meshgrid.
                # Sparse: eta1.shape = (n1,  1,  1)
                # Dense : eta1.shape = (n1, n2, n3)
                E1, E2, E3 = eta1, eta2, eta3

                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    is_sparse_meshgrid = True
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    is_sparse_meshgrid = False

            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

            if isinstance(a, list):
                
                if callable(a[0]):
                    a_in = np.array([a[0](E1, E2, E3), a[1](E1, E2, E3), a[2](E1, E2, E3)])
                else:
                    a_in = np.array(a)
                    
            else:
                
                if callable(a):
                    a_in = np.array([a(E1, E2, E3)])
                else:
                    a_in = np.array([a])

            if is_sparse_meshgrid:
                pf.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            else:
                pf.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

        else:
            raise ValueError('given evaluation points are in wrong shape')
    
        return values
