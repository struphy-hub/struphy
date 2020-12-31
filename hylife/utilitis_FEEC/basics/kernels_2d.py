from pyccel.decorators import types

import hylife.geometry.mappings_analytical_2d as map_ana
import hylife.geometry.mappings_discrete_2d   as map_dis


# ==========================================================================================
@types('double','double','int','int','double[:]')        
def fun_ana(eta1, eta2, kind_fun, kind_map, params):
    
    # quantities for 0-form mass matrix
    if   kind_fun == 1:
        value = abs(map_ana.det_df(eta1, eta2, kind_map, params))
    
    # quantities for 1-form mass matrix (H curl)
    elif kind_fun == 11:
        value = map_ana.g_inv(eta1, eta2, kind_map, params, 11) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 12:
        value = map_ana.g_inv(eta1, eta2, kind_map, params, 21) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 13:
        value = map_ana.g_inv(eta1, eta2, kind_map, params, 22) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
        
    # quantities for 1-form mass matrix (H div)
    elif kind_fun == 21:
        value = map_ana.g(eta1, eta2, kind_map, params, 11) / abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 22:
        value = map_ana.g(eta1, eta2, kind_map, params, 21) / abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 23:
        value = map_ana.g(eta1, eta2, kind_map, params, 22) / abs(map_ana.det_df(eta1, eta2, kind_map, params))
        
    # quantities for 2-form mass matrix
    elif kind_fun == 2:
        value = 1. / abs(map_ana.det_df(eta1, eta2, kind_map, params))
        
    # quantities for vector mass matrix
    elif kind_fun == 31:
        value = map_ana.g(eta1, eta2, kind_map, params, 11) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 32:
        value = map_ana.g(eta1, eta2, kind_map, params, 21) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
    elif kind_fun == 33:
        value = map_ana.g(eta1, eta2, kind_map, params, 22) * abs(map_ana.det_df(eta1, eta2, kind_map, params))
    
    return value


# ==========================================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double','int')        
def fun_dis(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2, kind_fun):
    
    # quantities for 0-form mass matrix
    if   kind_fun == 1:
        value = abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    
    # quantities for 1-form mass matrix (H curl)
    elif kind_fun == 11:
        value = map_dis.ginv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 12:
        value = map_dis.ginv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 13:
        value = map_dis.ginv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
        
    # quantities for 1-form mass matrix (H div)
    elif kind_fun == 21:
        value = map_dis.g_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) / abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 22:
        value = map_dis.g_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) / abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 23:
        value = map_dis.g_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) / abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
        
    # quantities for 2-form mass matrix
    elif kind_fun == 2:
        value = 1. / abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
        
    # quantities for vector mass matrix
    elif kind_fun == 31:
        value = map_dis.g_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 32:
        value = map_dis.g_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    elif kind_fun == 33:
        value = map_dis.g_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2) * abs(map_dis.det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2))
    
    return value


# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:,:]','int','int','double[:]')        
def kernel_evaluation_ana(nel, nq, eta1, eta2, mat_f, kind_fun, kind_map, params):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, q1, q2)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
    
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):
                    mat_f[ie1, ie2, q1, q2] = fun_ana(eta1[ie1, q1], eta2[ie2, q2], kind_fun, kind_map, params)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:,:]','int')        
def kernel_evaluation_dis(tn1, tn2, pn, nbase_n, cx, cy, nel, nq, eta1, eta2, mat_f, kind_fun):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, q1, q2)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
    
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):
                    mat_f[ie1, ie2, q1, q2] = fun_dis(tn1, tn2, pn, nbase_n, cx, cy, eta1[ie1, q1], eta2[ie2, q2], kind_fun)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','double[:,:,:,:]','double[:,:,:,:]')
def kernel_mass(nel1, nel2, p1, p2, nq1, nq2, ni1, ni2, nj1, nj2, w1, w2, bi1, bi2, bj1, bj2, nbase1, nbase2, m, mat_map):
    
    m[:, :, :, :] = 0.
     
    #$ omp parallel
    #$ omp do reduction ( + : m) private (ie1, ie2, il1, il2, jl1, jl2, value, q1, q2, wvol, bi, bj)
    for ie1 in range(nel1):
        for ie2 in range(nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):
                    for jl1 in range(p1 + 1 - nj1):
                        for jl2 in range(p2 + 1 - nj2):

                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):

                                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[ie1, ie2, q1, q2]
                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            m[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2, p1 + jl1 - il1, p2 + jl2 - il2] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','double[:,:]','double[:,:]','double[:,:,:,:]')
def kernel_inner_1(nel1, nel2, p1, p2, nq1, nq2, ni1, ni2, w1, w2, bi1, bi2, nbase1, nbase2, mat, mat_f, mat_map):
    
    mat[:, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, il1, il2, value, q1, q2, wvol, bi)
    for ie1 in range(nel1):
        for ie2 in range(nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):

                    value = 0.

                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[ie1, ie2, q1, q2]
                            bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                            value += wvol * bi * mat_f[nq1*ie1 + q1, nq2*ie2 + q2]

                    mat[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','double[:,:]','double[:,:,:,:]')
def kernel_inner_2(nel1, nel2, p1, p2, nq1, nq2, ni1, ni2, w1, w2, bi1, bi2, nbase1, nbase2, mat, mat_f):
    
    mat[:, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, il1, il2, value, q1, q2, wvol, bi)
    for ie1 in range(nel1):
        for ie2 in range(nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):

                    value = 0.

                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            wvol = w1[ie1, q1] * w2[ie2, q2]
                            bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                            value += wvol * bi * mat_f[ie1, ie2, q1, q2]

                    mat[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
@types('int[:]','int[:]','int[:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]')
def kernel_l2error(nel, p, nq, w1, w2, ni, nj, bi1, bi2, bj1, bj2, nbi, nbj, error, mat_f1, mat_f2, mat_c1, mat_c2, mat_map):
    
    
    #$ omp parallel
    #$ omp do reduction ( + : error) private (ie1, ie2, q1, q2, wvol, bi, bj, il1, il2, jl1, jl2)
    
    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
                
            # loop over quadrature points in element
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):

                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[ie1, ie2, q1, q2]

                    # evaluate discrete fields at quadrature point
                    bi = 0.
                    bj = 0. 

                    for il1 in range(p[0] + 1 - ni[0]):
                        for il2 in range(p[1] + 1 - ni[1]):

                            bi += mat_c1[(ie1 + il1)%nbi[0], (ie2 + il2)%nbi[1]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                    for jl1 in range(p[0] + 1 - nj[0]):
                        for jl2 in range(p[1] + 1 - nj[1]):

                            bj += mat_c2[(ie1 + jl1)%nbj[0], (ie2 + jl2)%nbj[1]] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]


                    # compare this value to exact one and add contribution to error in element
                    error[ie1, ie2] += wvol * (bi - mat_f1[nq[0]*ie1 + q1, nq[1]*ie2 + q2]) * (bj - mat_f2[nq[0]*ie1 + q1, nq[1]*ie2 + q2])
                            
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0