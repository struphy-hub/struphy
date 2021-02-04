# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping


# ==========================================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')         
def fun(eta1, eta2, eta3, kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    # quantities for 0-form mass matrix (|det(DF)|)
    if   kind_fun == 1:
        value = abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    
    # quantities for 1-form mass matrix (G^(-1)|det(DF)|)
    elif kind_fun == 11:
        value = mapping.all_mappings(eta1, eta2, eta3, 41, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 12:
        value = mapping.all_mappings(eta1, eta2, eta3, 44, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 13:
        value = mapping.all_mappings(eta1, eta2, eta3, 45, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 14:
        value = mapping.all_mappings(eta1, eta2, eta3, 47, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 15:
        value = mapping.all_mappings(eta1, eta2, eta3, 48, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 16:
        value = mapping.all_mappings(eta1, eta2, eta3, 49, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
        
    # quantities for 2-form mass matrix (G/|det(DF)|)
    elif kind_fun == 21:
        value = mapping.all_mappings(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 22:
        value = mapping.all_mappings(eta1, eta2, eta3, 34, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 23:
        value = mapping.all_mappings(eta1, eta2, eta3, 35, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 24:
        value = mapping.all_mappings(eta1, eta2, eta3, 37, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 25:
        value = mapping.all_mappings(eta1, eta2, eta3, 38, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 26:
        value = mapping.all_mappings(eta1, eta2, eta3, 39, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
        
    # quantities for 3-form mass matrix (1/|det(DF)|)
    elif kind_fun == 2:
        value = 1. / abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
        
    # quantities for vector mass matrix (G|det(DF)|)
    elif kind_fun == 31:
        value = mapping.all_mappings(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 32:
        value = mapping.all_mappings(eta1, eta2, eta3, 34, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 33:
        value = mapping.all_mappings(eta1, eta2, eta3, 35, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 34:
        value = mapping.all_mappings(eta1, eta2, eta3, 37, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 35:
        value = mapping.all_mappings(eta1, eta2, eta3, 38, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
    elif kind_fun == 36:
        value = mapping.all_mappings(eta1, eta2, eta3, 39, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz) * abs(mapping.all_mappings(eta1, eta2, eta3, 4, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
        
    return value


# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_evaluate_quadrature(nel, nq, eta1, eta2, eta3, mat_f, kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
    
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            mat_f[ie1, ie2, ie3, q1, q2, q3] = fun(eta1[ie1, q1], eta2[ie2, q2], eta3[ie3, q3], kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','int','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_mass(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2, bj3, nbase1, nbase2, nbase3, m, mat_map):
    
    m[:, :, :, :, :, :] = 0.
     
    #$ omp parallel
    #$ omp do reduction ( + : m) private (ie1, ie2, ie3, il1, il2, il3, jl1, jl2, jl3, value, q1, q2, q3, wvol, bi, bj)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            for jl1 in range(p1 + 1 - nj1):
                                for jl2 in range(p2 + 1 - nj2):
                                    for jl3 in range(p3 + 1 - nj3):

                                        value = 0.

                                        for q1 in range(nq1):
                                            for q2 in range(nq2):
                                                for q3 in range(nq3):
                                                    
                                                    wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[ie1, ie2, ie3, q1, q2, q3]
                                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]
                                                    
                                                    value += wvol * bi * bj

                                        m[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2, (ie3 + il3)%nbase3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','int','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_inner_1(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, nbase1, nbase2, nbase3, mat, mat_f, mat_map):
    
    mat[:, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, ie3, il1, il2, il3, value, q1, q2, q3, wvol, bi)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            
                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[ie1, ie2, ie3, q1, q2, q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi * mat_f[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]

                            mat[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2, (ie3 + il3)%nbase3] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                            
                            

# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','int','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_inner_2(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, nbase1, nbase2, nbase3, mat, mat_f):
    
    mat[:, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, ie3, il1, il2, il3, value, q1, q2, q3, wvol, bi)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            
                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi * mat_f[ie1, ie2, ie3, q1, q2, q3]

                            mat[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2, (ie3 + il3)%nbase3] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0            
            
            
                       
# ==========================================================================================          
@types('int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_l2error(nel, p, nq, w1, w2, w3, ni, nj, bi1, bi2, bi3, bj1, bj2, bj3, nbi, nbj, error, mat_f1, mat_f2, mat_c1, mat_c2, mat_map):
    
    
    #$ omp parallel
    #$ omp do reduction ( + : error) private (ie1, ie2, ie3, q1, q2, q3, wvol, bi, bj, il1, il2, il3, jl1, jl2, jl3)
    
    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
                
                # loop over quadrature points in element
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[ie1, ie2, ie3, q1, q2, q3]

                            # evaluate discrete fields at quadrature point
                            bi = 0.
                            bj = 0. 

                            for il1 in range(p[0] + 1 - ni[0]):
                                for il2 in range(p[1] + 1 - ni[1]):
                                    for il3 in range(p[2] + 1 - ni[2]):

                                        bi += mat_c1[(ie1 + il1)%nbi[0], (ie2 + il2)%nbi[1], (ie3 + il3)%nbi[2]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                        
                            for jl1 in range(p[0] + 1 - nj[0]):
                                for jl2 in range(p[1] + 1 - nj[1]):
                                    for jl3 in range(p[2] + 1 - nj[2]):

                                        bj += mat_c2[(ie1 + jl1)%nbj[0], (ie2 + jl2)%nbj[1], (ie3 + jl3)%nbj[2]] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]

                            
                            # compare this value to exact one and add contribution to error in element
                            error[ie1, ie2, ie3] += wvol * (bi - mat_f1[nq[0]*ie1 + q1, nq[1]*ie2 + q2, nq[2]*ie3 + q3]) * (bj - mat_f2[nq[0]*ie1 + q1, nq[1]*ie2 + q2, nq[2]*ie3 + q3])
                            
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
       
# ==========================================================================================
@types('int[:]','int[:]','int[:]','int[:]','double[:,:,:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:,:,:]')        
def kernel_evaluate_2form(nel, p, ns, nq, b_coeff, nbase, bi1, bi2, bi3, b_eva):
    
    b_eva[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
                
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                
                            for il1 in range(p[0] + 1 - ns[0]):
                                for il2 in range(p[1] + 1 - ns[1]):
                                    for il3 in range(p[2] + 1 - ns[2]):
                                        
                                        b_eva[ie1, ie2, ie3, q1, q2, q3] += b_coeff[(ie1 + il1)%nbase[0], (ie2 + il2)%nbase[1], (ie3 + il3)%nbase[2]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0