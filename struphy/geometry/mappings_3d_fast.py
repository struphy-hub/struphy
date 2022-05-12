# coding: utf-8
#
# Copnyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Efficient modules for point-wise evaluation of a 3d analytical (kind_map >= 10) or discrete (kind_map < 10) B-spline mapping.
Especially suited for PIC routines since it avoids computing the Jacobian matrix multiple times.
"""

from numpy import empty, cos, sin, pi

import struphy.feec.bsplines_kernels as bsp

from struphy.feec.basics.spline_evaluation_2d import evaluation_kernel_2d, eval_kernel_2d
from struphy.feec.basics.spline_evaluation_3d import evaluation_kernel_3d, eval_kernel_3d

import struphy.geometry.mappings_3d as mapping


# ==========================================================================
def df_all(kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', span_n1 : 'int', span_n2 : 'int', span_n3 : 'int', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', l1 : 'double[:]', l2 : 'double[:]', l3 : 'double[:]', r1 : 'double[:]', r2 : 'double[:]', r3 : 'double[:]', b1 : 'double[:,:]', b2 : 'double[:,:]', b3 : 'double[:,:]', d1 : 'double[:]', d2 : 'double[:]', d3 : 'double[:]', der1 : 'double[:]', der2 : 'double[:]', der3 : 'double[:]', eta1 : 'double', eta2 : 'double', eta3 : 'double', mat_out : 'double[:,:]', vec_out : 'double[:]', mat_or_vec : 'int'):
    """
    TODO: write documentation, implement faster eval_kernels (with list of global indices, not modulo-operation)
    """
    # 3d discrete mapping
    if kind_map == 0:
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(tn3, pn[2], eta3, span_n3, l3, r3, b3, d3, der3)
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)
            mat_out[0, 1] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)
            mat_out[0, 2] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)
            mat_out[1, 1] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)
            mat_out[1, 2] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = evaluation_kernel_3d(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
            mat_out[2, 1] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
            mat_out[2, 2] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)
            vec_out[1] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)
            vec_out[2] = evaluation_kernel_3d(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
            
           
    # discrete cylinder
    elif kind_map == 1:
        
        lz = 2*pi*cx[0, 0, 0]
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0])
            mat_out[0, 1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0])
            mat_out[0, 2] = 0.

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            mat_out[1, 1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            mat_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = 0.
            mat_out[2, 1] = 0.
            mat_out[2, 2] = lz
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0])
            vec_out[1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            vec_out[2] = lz * eta3
        
    # discrete torus
    elif kind_map == 2:
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * cos(2*pi*eta3)
            mat_out[0, 1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * cos(2*pi*eta3)
            mat_out[0, 2] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * sin(2*pi*eta3) * (-2*pi)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            mat_out[1, 1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            mat_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = evaluation_kernel_2d(pn[0], pn[1], der1, b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * sin(2*pi*eta3)
            mat_out[2, 1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], der2, span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * sin(2*pi*eta3)
            mat_out[2, 2] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * cos(2*pi*eta3) * 2*pi
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * cos(2*pi*eta3)
            vec_out[1] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cy[:, :, 0])
            vec_out[2] = evaluation_kernel_2d(pn[0], pn[1], b1[pn[0]], b2[pn[1]], span_n1, span_n2, nbase_n[0], nbase_n[1], cx[:, :, 0]) * sin(2*pi*eta3)
           
    
    # analytical mapping
    else:
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
        
            mat_out[0, 0] = mapping.df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[0, 1] = mapping.df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[0, 2] = mapping.df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

            mat_out[1, 0] = mapping.df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[1, 1] = mapping.df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[1, 2] = mapping.df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

            mat_out[2, 0] = mapping.df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[2, 1] = mapping.df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            mat_out[2, 2] = mapping.df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            vec_out[1] = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            vec_out[2] = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            
            
            
            
# ==========================================================================
def dl_all(kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', eta1 : 'double', eta2 : 'double', eta3 : 'double', mat_out : 'double[:,:]', vec_out : 'double[:]', mat_or_vec : 'int'):
    """
    function to write Jacobian matrix entries into mat_out

    Parameters:
    -----------
        kind_map : integer
            if kind_map is 0,1,2 then the mapping is given in terms of splines, otherwise an analytical expression is given

        params_map : array
            contains parameters for the analytical mapping
        
        tn1, tn2, tn3 : array
            contain the knot sequences in each direction
        
        pn : array of integers
            contains the degrees of the basis splines in each direction
        
        cx, cy, cz : array
            contains the spline coefficients for the mapping
        
        eta1, eta2, eta3 : double
            position, logical coordinates in [0,1]
        
        mat_out : array
            matrix, in which the resulting Jacobian matrix is written
        
        vec_out : array
            mapping vector is written in here
        
        mat_or_vec : int
            0: only Jacobian matrix, 1: only mapping vector, 2: both matrix and vector
    """
        
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # find indices for list of global indices
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    # non-vanishing B-splines at particle position
    b1 = empty( pn1 + 1, dtype=float)
    b2 = empty( pn2 + 1, dtype=float)
    b3 = empty( pn3 + 1, dtype=float)

    # evaluate non-vanishing basis functions and its derivatives
    bsp.b_splines_slim(tn1, pn1, eta1, span1, b1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, b2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, b3)
    
    # non-vanishing Derivatives of B-splines at particle position
    der1 = empty( pn1 + 1, dtype=float)
    der2 = empty( pn2 + 1, dtype=float)
    der3 = empty( pn3 + 1, dtype=float)

    bsp.b_spl_1st_der_slim(tn1, pn1, eta1, span1, der1)
    bsp.b_spl_1st_der_slim(tn2, pn2, eta2, span2, der2)
    bsp.b_spl_1st_der_slim(tn3, pn3, eta3, span3, der3)

    # 3d discrete mapping
    if kind_map == 0:

        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = eval_kernel_3d(pn1, pn2, pn3, der1, b2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cx)
            mat_out[0, 1] = eval_kernel_3d(pn1, pn2, pn3, b1, der2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cx)
            mat_out[0, 2] = eval_kernel_3d(pn1, pn2, pn3, b1, b2, der3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cx)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = eval_kernel_3d(pn1, pn2, pn3, der1, b2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cy)
            mat_out[1, 1] = eval_kernel_3d(pn1, pn2, pn3, b1, der2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cy)
            mat_out[1, 2] = eval_kernel_3d(pn1, pn2, pn3, b1, b2, der3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cy)

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = eval_kernel_3d(pn1, pn2, pn3, der1, b2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cz)
            mat_out[2, 1] = eval_kernel_3d(pn1, pn2, pn3, b1, der2, b3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cz)
            mat_out[2, 2] = eval_kernel_3d(pn1, pn2, pn3, b1, b2, der3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], cz)
        
           
    # discrete cylinder
    elif kind_map == 1:
        
        lz = 2*pi*cx[0, 0, 0]

        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = eval_kernel_2d(pn1, pn2, der1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0])
            mat_out[0, 1] = eval_kernel_2d(pn1, pn2, b1, der2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0])
            mat_out[0, 2] = 0.

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = eval_kernel_2d(pn1, pn2, der1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            mat_out[1, 1] = eval_kernel_2d(pn1, pn2, b1, der2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            mat_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = 0.
            mat_out[2, 1] = 0.
            mat_out[2, 2] = lz
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0])
            vec_out[1] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            vec_out[2] = lz * eta3
        
    # discrete torus
    elif kind_map == 2:
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:

            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = eval_kernel_2d(pn1, pn2, der1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * cos(2*pi*eta3)
            mat_out[0, 1] = eval_kernel_2d(pn1, pn2, b1, der2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * cos(2*pi*eta3)
            mat_out[0, 2] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * sin(2*pi*eta3) * (-2*pi)

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = eval_kernel_2d(pn1, pn2, der1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            mat_out[1, 1] = eval_kernel_2d(pn1, pn2, b1, der2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            mat_out[1, 2] = 0.

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = eval_kernel_2d(pn1, pn2, der1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * sin(2*pi*eta3)
            mat_out[2, 1] = eval_kernel_2d(pn1, pn2, b1, der2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * sin(2*pi*eta3)
            mat_out[2, 2] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * cos(2*pi*eta3) * 2*pi
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            vec_out[0] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * cos(2*pi*eta3)
            vec_out[1] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cy[:, :, 0])
            vec_out[2] = eval_kernel_2d(pn1, pn2, b1, b2, ind_n1[ie1, :], ind_n2[ie2, :], cx[:, :, 0]) * sin(2*pi*eta3)
           
    
    # analytical mapping
    else:
        
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:

            mapping.df_ana_mat(eta1, eta2, eta3, kind_map, params_map, mat_out)
        
        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            
            mapping.f_vec_ana(eta1, eta2, eta3, kind_map, params_map, vec_out)
            
 
        
# ===========================================================================
def df_inv_all(mat_in : 'double[:,:]', mat_out : 'double[:,:]'):
    """
    Inverts the Jacobain matrix (mat_in) and writes it to mat_out

    Parameters:
    -----------
        mat_in : array
            Jacobian matrix
        
        mat_out : array
            emtpy array where the inverse Jacobian matrix will be written
    """
    
    # inverse Jacobian determinant computed from Jacobian matrix (mat_in)
    over_det_df = 1.0 / (mat_in[0, 0]*(mat_in[1, 1]*mat_in[2, 2] - mat_in[2, 1]*mat_in[1, 2]) + mat_in[1, 0]*(mat_in[2, 1]*mat_in[0, 2] - mat_in[0, 1]*mat_in[2, 2]) + mat_in[2, 0]*(mat_in[0, 1]*mat_in[1, 2] - mat_in[1, 1]*mat_in[0, 2]))

    # inverse Jacobian matrix computed from Jacobian matrix (mat_in)
    mat_out[0, 0] = (mat_in[1, 1]*mat_in[2, 2] - mat_in[2, 1]*mat_in[1, 2]) * over_det_df
    mat_out[0, 1] = (mat_in[2, 1]*mat_in[0, 2] - mat_in[0, 1]*mat_in[2, 2]) * over_det_df
    mat_out[0, 2] = (mat_in[0, 1]*mat_in[1, 2] - mat_in[1, 1]*mat_in[0, 2]) * over_det_df

    mat_out[1, 0] = (mat_in[1, 2]*mat_in[2, 0] - mat_in[2, 2]*mat_in[1, 0]) * over_det_df
    mat_out[1, 1] = (mat_in[2, 2]*mat_in[0, 0] - mat_in[0, 2]*mat_in[2, 0]) * over_det_df
    mat_out[1, 2] = (mat_in[0, 2]*mat_in[1, 0] - mat_in[1, 2]*mat_in[0, 0]) * over_det_df

    mat_out[2, 0] = (mat_in[1, 0]*mat_in[2, 1] - mat_in[2, 0]*mat_in[1, 1]) * over_det_df
    mat_out[2, 1] = (mat_in[2, 0]*mat_in[0, 1] - mat_in[0, 0]*mat_in[2, 1]) * over_det_df
    mat_out[2, 2] = (mat_in[0, 0]*mat_in[1, 1] - mat_in[1, 0]*mat_in[0, 1]) * over_det_df
    
    
    
# ===========================================================================
def g_all(mat_in : 'double[:,:]', mat_out : 'double[:,:]'):
    """
    Compute the metric tensor (mat_out) from Jacobian matrix (mat_in)

    Parameters:
    -----------
        mat_in : array
            Jacobian matrix
        
        mat_out : array
            array where metric tensor will be written to
    """
    mat_out[0, 0] = mat_in[0, 0]*mat_in[0, 0] + mat_in[1, 0]*mat_in[1, 0] + mat_in[2, 0]*mat_in[2, 0]
    mat_out[0, 1] = mat_in[0, 0]*mat_in[0, 1] + mat_in[1, 0]*mat_in[1, 1] + mat_in[2, 0]*mat_in[2, 1]
    mat_out[0, 2] = mat_in[0, 0]*mat_in[0, 2] + mat_in[1, 2]*mat_in[1, 2] + mat_in[2, 0]*mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[0, 1]*mat_in[0, 1] + mat_in[1, 1]*mat_in[1, 1] + mat_in[2, 1]*mat_in[2, 1]
    mat_out[1, 2] = mat_in[0, 1]*mat_in[0, 2] + mat_in[1, 0]*mat_in[1, 2] + mat_in[2, 0]*mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[0, 2]*mat_in[0, 2] + mat_in[1, 2]*mat_in[1, 2] + mat_in[2, 2]*mat_in[2, 2]
    
    
    
# ===========================================================================
def g_inv_all(mat_in : 'double[:,:]', mat_out : 'double[:,:]'):
    """
    Compute the inverse metric tensor (mat_out) from inverse Jacobian matrix (mat_in)

    Parameters:
    -----------
        mat_in : array
            inverse Jacobian matrix
        
        mat_out : array
            array where inverse metric tensor will be written to
    """
    mat_out[0, 0] = mat_in[0, 0]*mat_in[0, 0] + mat_in[0, 1]*mat_in[0, 1] + mat_in[0, 2]*mat_in[0, 2]
    mat_out[0, 1] = mat_in[0, 0]*mat_in[1, 0] + mat_in[0, 1]*mat_in[1, 1] + mat_in[0, 2]*mat_in[1, 2]
    mat_out[0, 2] = mat_in[0, 0]*mat_in[2, 0] + mat_in[0, 1]*mat_in[2, 1] + mat_in[0, 2]*mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[1, 0]*mat_in[1, 0] + mat_in[1, 1]*mat_in[1, 1] + mat_in[1, 2]*mat_in[1, 2]
    mat_out[1, 2] = mat_in[1, 0]*mat_in[2, 0] + mat_in[1, 1]*mat_in[2, 1] + mat_in[1, 2]*mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[2, 0]*mat_in[2, 0] + mat_in[2, 1]*mat_in[2, 1] + mat_in[2, 2]*mat_in[2, 2]
