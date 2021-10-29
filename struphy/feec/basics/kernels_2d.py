# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


from pyccel.decorators import types


# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','double[:,:,:,:]','double[:,:,:,:]')
def kernel_mass(nel1, nel2, p1, p2, nq1, nq2, ni1, ni2, nj1, nj2, w1, w2, bi1, bi2, bj1, bj2, ind_base1, ind_base2, mat, mat_fun):
    
    mat[:, :, :, :] = 0.
     
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, il1, il2, jl1, jl2, value, q1, q2, wvol, bi, bj)
    for ie1 in range(nel1):
        for ie2 in range(nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):
                    for jl1 in range(p1 + 1 - nj1):
                        for jl2 in range(p2 + 1 - nj2):

                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):

                                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_fun[ie1, q1, ie2, q2]
                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            mat[ind_base1[ie1, il1], ind_base2[ie2, il2], p1 + jl1 - il1, p2 + jl2 - il2] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:,:,:]')
def kernel_inner(nel1, nel2, p1, p2, nq1, nq2, ni1, ni2, w1, w2, bi1, bi2, ind_base1, ind_base2, mat, mat_fun):
    
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

                            wvol = w1[ie1, q1] * w2[ie2, q2] * mat_fun[ie1, q1, ie2, q2]
                            bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                            value += wvol * bi

                    mat[ind_base1[ie1, il1], ind_base2[ie2, il2]] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    

    
# ==========================================================================================          
@types('int[:]','int[:]','int[:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]')
def kernel_l2error(nel, p, nq, w1, w2, ni, nj, bi1, bi2, bj1, bj2, ind_basei1, ind_basei2, ind_basej1, ind_basej2, error, mat_f1, mat_f2, mat_c1, mat_c2, mat_map):
    
    
    #$ omp parallel
    #$ omp do reduction ( + : error) private (ie1, ie2, q1, q2, wvol, bi, bj, il1, il2, jl1, jl2)
    
    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
                
            # loop over quadrature points in element
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):

                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[nq[0]*ie1 + q1, nq[1]*ie2 + q2]

                    # evaluate discrete fields at quadrature point
                    bi = 0.
                    bj = 0. 

                    for il1 in range(p[0] + 1 - ni[0]):
                        for il2 in range(p[1] + 1 - ni[1]):

                            bi += mat_c1[ind_basei1[ie1, il1], ind_basei2[ie2, il2]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                    for jl1 in range(p[0] + 1 - nj[0]):
                        for jl2 in range(p[1] + 1 - nj[1]):

                            bj += mat_c2[ind_basej1[ie1, jl1], ind_basej2[ie2, jl2]] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]


                    # compare this value to exact one and add contribution to error in element
                    error[ie1, ie2] += wvol * (bi - mat_f1[nq[0]*ie1 + q1, nq[1]*ie2 + q2]) * (bj - mat_f2[nq[0]*ie1 + q1, nq[1]*ie2 + q2])
                            
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================
@types('int[:]','int[:]','int[:]','int[:]','double[:,:]','int[:,:]','int[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]')        
def kernel_evaluate_2form(nel, p, ns, nq, b_coeff, ind_base1, ind_base2, bi1, bi2, b_eva):
    
    b_eva[:, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : b_eva) private (ie1, ie2, q1, q2, il1, il2)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
                
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):
          
                    for il1 in range(p[0] + 1 - ns[0]):
                        for il2 in range(p[1] + 1 - ns[1]):

                            b_eva[ie1, q1, ie2, q2] += b_coeff[ind_base1[ie1, il1], ind_base2[ie2, il2]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0