# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


from pyccel.decorators import types


# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_mass(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2, bj3, ind_base1, ind_base2, ind_base3, mat, mat_fun):
    
    mat[:, :, :, :, :, :] = 0.
     
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, ie3, il1, il2, il3, jl1, jl2, jl3, value, q1, q2, q3, wvol, bi, bj)
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
                                                    
                                                    wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_fun[ie1, q1, ie2, q2, ie3, q3]
                                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]
                                                    
                                                    value += wvol * bi * bj

                                        mat[ind_base1[ie1, il1], ind_base2[ie2, il2], ind_base3[ie3, il3], p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_inner(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, ind_base1, ind_base2, ind_base3, mat, mat_fun):
    
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

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_fun[ie1, q1, ie2, q2, ie3, q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi

                            mat[ind_base1[ie1, il1], ind_base2[ie2, il2], ind_base3[ie3, il3]] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                 
                       
# ==========================================================================================          
@types('int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_l2error(nel, p, nq, w1, w2, w3, ni, nj, bi1, bi2, bi3, bj1, bj2, bj3, ind_basei1, ind_basei2, ind_basei3, ind_basej1, ind_basej2, ind_basej3, error, mat_f1, mat_f2, c1, c2, mat_map):
    
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, wvol, bi, bj, il1, il2, il3, jl1, jl2, jl3)
    
    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
                
                # loop over quadrature points in element
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[ie1, q1, ie2, q2, ie3, q3]

                            # evaluate discrete fields at quadrature point
                            bi = 0.
                            bj = 0. 

                            for il1 in range(p[0] + 1 - ni[0]):
                                for il2 in range(p[1] + 1 - ni[1]):
                                    for il3 in range(p[2] + 1 - ni[2]):

                                        bi += c1[ind_basei1[ie1, il1], ind_basei2[ie2, il2], ind_basei3[ie3, il3]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                        
                            for jl1 in range(p[0] + 1 - nj[0]):
                                for jl2 in range(p[1] + 1 - nj[1]):
                                    for jl3 in range(p[2] + 1 - nj[2]):

                                        bj += c2[ind_basej1[ie1, il1], ind_basej2[ie2, il2], ind_basej3[ie3, il3]] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]

                            
                            # compare this value to exact one and add contribution to error in element
                            error[ie1, ie2, ie3] += wvol * (bi - mat_f1[ie1, q1, ie2, q2, ie3, q3]) * (bj - mat_f2[ie1, q1, ie2, q2, ie3, q3])
                            
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
       
# ==========================================================================================
@types('int[:]','int[:]','int[:]','int[:]','double[:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:,:,:]')        
def kernel_evaluate_2form(nel, p, ns, nq, b_coeff, ind_base1, ind_base2, ind_base3, bi1, bi2, bi3, b_eva):
    
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
                                        
                                        b_eva[ie1, q1, ie2, q2, ie3, q3] += b_coeff[ind_base1[ie1, il1], ind_base2[ie2, il2], ind_base3[ie3, il3]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0