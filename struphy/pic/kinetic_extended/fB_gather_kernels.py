# import pyccel decorators
from pyccel.decorators import types

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

#import time

# ==============================================================================
@types('int[:]','int[:]','int[:]','int','int','int','int[:]','double[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','int[:]','double[:,:]','int','int','double[:,:,:,:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def gather_quadrature(index_shapex, index_shapey, index_shapez, index_diffx, index_diffy, index_diffz, p_shape, p_size, n_quad, pts1, pts2, pts3, Nel, particles_loc, Np_loc, Np, gather, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros, floor
    #==initialization of gather 
    gather[:,:,:,:,:,:] = 0.0
    #==========================
    cell_left    = empty(3, dtype=int)
    point_left   = zeros(3, dtype=float)
    point_right  = zeros(3, dtype=float)
    cell_number  = empty(3, dtype=int)


    temp1        = zeros(3, dtype=float)
    temp4        = zeros(3, dtype=float)

    compact      = zeros(3, dtype=float)
    compact[0]   = (p_shape[0]+1.0)*p_size[0]
    compact[1]   = (p_shape[1]+1.0)*p_size[1]
    compact[2]   = (p_shape[2]+1.0)*p_size[2]

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df        = empty((3, 3), dtype=float) 
    fx        = empty( 3    , dtype=float)
    # ==========================================================

    #$ omp parallel
    #$ omp do reduction ( + : gather) private (ip, eta1, eta2, eta3, weight_p, ie1, ie2, ie3, point_left, point_right, cell_left, cell_number, temp1, temp4, il1, il2, il3, q1, q2, q3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df)
    for ip in range(Np_loc):

        #time1 = time.time()
        eta1   = particles_loc[0,ip]
        eta2   = particles_loc[1,ip]
        eta3   = particles_loc[2,ip]
        
        weight_p = particles_loc[6,ip]/(p_size[0]*p_size[1]*p_size[2])/Np

        #======== for n ==================
        ie1 = int(eta1*Nel[0])
        ie2 = int(eta2*Nel[1])
        ie3 = int(eta3*Nel[2])

        #the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0]  = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1]  = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2]  = eta3 - 0.5*compact[2]
        point_right[2] = eta3 + 0.5*compact[2]

        cell_left[0] = int(floor(point_left[0]*Nel[0]))
        cell_left[1] = int(floor(point_left[1]*Nel[1]))
        cell_left[2] = int(floor(point_left[2]*Nel[2]))

        cell_number[0] = int(floor(point_right[0]*Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1]*Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2]*Nel[2])) - cell_left[2] + 1
        #time2 = time.time()
        #print('check_first_part', time2 - time1)
        #time1 = time.time()
        #======================================
        for il1 in range(cell_number[0]):
            for il2 in range(cell_number[1]):
                for il3 in range(cell_number[2]):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                #time1 = time.time()
                                temp1[0] = (cell_left[0] + il1)/Nel[0] + pts1[0,q1] # quadrature points in the cell x direction
                                temp4[0] = abs(temp1[0] - eta1) - compact[0]/2.0 # if > 0, result is 0

                                temp1[1] = (cell_left[1] + il2)/Nel[1] + pts2[0,q2] 
                                temp4[1] = abs(temp1[1] - eta2) - compact[1]/2.0 # if > 0, result is 0

                                temp1[2] = (cell_left[2] + il3)/Nel[2] + pts3[0,q3] 
                                temp4[2] = abs(temp1[2] - eta3) - compact[2]/2.0 # if > 0, result is 0
                                #time2 = time.time()
                                #print('check_second_part_insider_1', time2 - time1)
                                if temp4[0] < 0.0 and temp4[1] < 0.0 and temp4[2] < 0.0:
                                    #time1 = time.time()
                                    #time3 = time.time()
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], temp1[0] - eta1)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], temp1[1] - eta2)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], temp1[2] - eta3)
                                    #time4 = time.time()
                                    #print('check_bsp_time', time4 - time3)
                                    # ========= mapping evaluation =============
                                    #time5 = time.time()
                                    span1f = int(temp1[0]%1.0*nelf[0]) + pf1
                                    span2f = int(temp1[1]%1.0*nelf[1]) + pf2
                                    span3f = int(temp1[2]%1.0*nelf[2]) + pf3
                                    #time6 = time.time()
                                    #print('check_int_time', time6 - time5)
                                    # evaluate Jacobian matrix
                                    #time7 = time.time()
                                    mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, temp1[0]%1.0, temp1[1]%1.0, temp1[2]%1.0, df, fx, 0)
                                    # evaluate Jacobian determinant
                                    det_df = abs(linalg.det(df))
                                    #time8 = time.time()
                                    #print('check_mapping_time', time8 - time7)
                                    #time9 = time.time()
                                    # ==========================================
                                    gather[index_shapex[cell_left[0] + il1 + index_diffx], index_shapey[cell_left[1] + il2 + index_diffy], index_shapez[cell_left[2] + il3 + index_diffz], q1, q2, q3] += weight_p * value_x * value_y * value_z / det_df
                                    #time10 = time.time()
                                    #print('check_other_time', time10 - time9)
                                    #time2 = time.time()
                                    #print('check_second_part_insider_2', time2 - time1)
    #$ omp end do
    #$ omp end parallel
        #time2 = time.time()
        #print('check_second_part', time2 - time1)

    
    ierr = 0            



# ==============================================================================
@types('int[:]','int[:]','int[:]','int','int','int','int[:]','double[:]','int[:]','double[:,:]','int','int','double[:,:,:,:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def gather_grid(index_shapex, index_shapey, index_shapez, index_diffx, index_diffy, index_diffz, p_shape, p_size, Nel, particles_loc, Np_loc, Np, gather, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros, floor
    # put particles into grid points

    #==initialization of gather 
    gather[:,:,:] = 0.0
    #==========================
    cell_left    = empty(3, dtype=int)
    point_left   = zeros(3, dtype=float)
    point_right  = zeros(3, dtype=float)
    cell_number  = empty(3, dtype=int)


    temp1        = zeros(3, dtype=float)
    temp4        = zeros(3, dtype=float)


    compact      = zeros(3, dtype=float)
    compact[0]   = (p_shape[0]+1.0)*p_size[0]
    compact[1]   = (p_shape[1]+1.0)*p_size[1]
    compact[2]   = (p_shape[2]+1.0)*p_size[2]

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df        = empty((3, 3), dtype=float) 
    fx        = empty( 3    , dtype=float)
    # ==========================================================


    #$ omp parallel
    #$ omp do reduction ( + : gather) private (ip, eta1, eta2, eta3, weight_p, ie1, ie2, ie3, point_left, point_right, cell_left, cell_number, temp1, temp4, il1, il2, il3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df)
    for ip in range(Np_loc):

        eta1   = particles_loc[0,ip]
        eta2   = particles_loc[1,ip]
        eta3   = particles_loc[2,ip]
        
        weight_p = particles_loc[6,ip]/(p_size[0]*p_size[1]*p_size[2])/Np

        #======== for n ==================
        ie1 = int(eta1*Nel[0])
        ie2 = int(eta2*Nel[1])
        ie3 = int(eta3*Nel[2])

        #the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0]  = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1]  = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2]  = eta3 - 0.5*compact[2]
        point_right[2] = eta3 + 0.5*compact[2]

        cell_left[0] = int(floor(point_left[0]*Nel[0]))
        cell_left[1] = int(floor(point_left[1]*Nel[1]))
        cell_left[2] = int(floor(point_left[2]*Nel[2]))

        cell_number[0] = int(floor(point_right[0]*Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1]*Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2]*Nel[2])) - cell_left[2] + 1

        #======================================
        for il1 in range(cell_number[0]):
            for il2 in range(cell_number[1]):
                for il3 in range(cell_number[2]):

                    temp1[0] = (cell_left[0] + il1)/Nel[0] # quadrature points in the cell x direction
                    temp4[0] = abs(temp1[0] - eta1) - compact[0]/2.0 # if > 0, result is 0

                    temp1[1] = (cell_left[1] + il2)/Nel[1]
                    temp4[1] = abs(temp1[1] - eta2) - compact[1]/2.0 # if > 0, result is 0

                    temp1[2] = (cell_left[2] + il3)/Nel[2]
                    temp4[2] = abs(temp1[2] - eta3) - compact[2]/2.0 # if > 0, result is 0

                    if temp4[0] < 0.0 and temp4[1] < 0.0 and temp4[2] < 0.0:
                        value_x = bsp.piecewise(p_shape[0], p_size[0], temp1[0] - eta1)
                        value_y = bsp.piecewise(p_shape[1], p_size[1], temp1[1] - eta2)
                        value_z = bsp.piecewise(p_shape[2], p_size[2], temp1[2] - eta3)

                        # ========= mapping evaluation =============
                        span1f = int(temp1[0]%1.0*nelf[0]) + pf1
                        span2f = int(temp1[1]%1.0*nelf[1]) + pf2
                        span3f = int(temp1[2]%1.0*nelf[2]) + pf3
                        # evaluate Jacobian matrix
                        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, temp1[0]%1.0, temp1[1]%1.0, temp1[2]%1.0, df, fx, 0)
                        # evaluate Jacobian determinant
                        det_df = abs(linalg.det(df))

                        gather[index_shapex[cell_left[0] + il1 + index_diffx],index_shapey[cell_left[1] + il2 + index_diffy],index_shapez[cell_left[2] + il3 + index_diffz]] += weight_p * value_x * value_y * value_z / det_df

    #$ omp end do
    #$ omp end parallel

    ierr = 0            



# ==============================================================================
@types('double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double')
def quadraturelog(gather_quadrature, quadrature_log, Nel, n_quad, wts1, wts2, wts3, tol):
    from numpy import log
    #$ omp parallel
    #$ omp do private (il1, il2, il3, q1, q2, q3)
    for il1 in range(Nel[0]):
        for il2 in range(Nel[1]):
            for il3 in range(Nel[2]):
                for q1 in range(n_quad[0]):
                    for q2 in range(n_quad[1]):
                        for q3 in range(n_quad[2]):
                            if abs(gather_quadrature[il1, il2, il3, q1, q2, q3]) > tol:
                                quadrature_log[il1, il2, il3, q1, q2, q3] = log(gather_quadrature[il1, il2, il3, q1, q2, q3])*wts1[il1, q1]*wts2[il2, q2]*wts3[il3, q3]
                            else:
                                quadrature_log[il1, il2, il3, q1, q2, q3] = 0.0


    #$ omp end do
    #$ omp end parallel
    ierr = 0            


# ==============================================================================
@types('double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','int[:]','int[:]','double')
def quadratureinverse(gather_quadrature, quadrature_inverse, Nel, n_quad, tol):

    #$ omp parallel
    #$ omp do private (il1, il2, il3, q1, q2, q3)
    for il1 in range(Nel[0]):
        for il2 in range(Nel[1]):
            for il3 in range(Nel[2]):
                for q1 in range(n_quad[0]):
                    for q2 in range(n_quad[1]):
                        for q3 in range(n_quad[2]):
                            if abs(gather_quadrature[il1, il2, il3, q1, q2, q3]) > tol:
                                quadrature_inverse[il1, il2, il3, q1, q2, q3] = 1.0 / gather_quadrature[il1, il2, il3, q1, q2, q3]
                            else:
                                quadrature_inverse[il1, il2, il3, q1, q2, q3] = 0.0


    #$ omp end do
    #$ omp end parallel

    ierr = 0            


