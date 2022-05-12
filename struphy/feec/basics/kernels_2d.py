# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


# ==========================================================================================          
def kernel_mass(nel1 : 'int', nel2 : 'int', p1 : 'int', p2 : 'int', nq1 : 'int', nq2 : 'int', ni1 : 'int', ni2 : 'int', nj1 : 'int', nj2 : 'int', w1 : 'double[:,:]', w2 : 'double[:,:]', bi1 : 'double[:,:,:,:]', bi2 : 'double[:,:,:,:]', bj1 : 'double[:,:,:,:]', bj2 : 'double[:,:,:,:]', ind_base1 : 'int[:,:]', ind_base2 : 'int[:,:]', mat : 'double[:,:,:,:]', mat_fun : 'double[:,:,:,:]'):
    
    mat[:, :, :, :] = 0.
     
    #$ omp parallel private(ie1, ie2, il1, il2, jl1, jl2, value, q1, q2, wvol, bi, bj)
    #$ omp for reduction ( + : mat)
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
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================          
def kernel_inner(nel1 : 'int', nel2 : 'int', n3 : 'int', p1 : 'int', p2 : 'int', nq1 : 'int', nq2 : 'int', ni1 : 'int', ni2 : 'int', w1 : 'double[:,:]', w2 : 'double[:,:]', bi1 : 'double[:,:,:,:]', bi2 : 'double[:,:,:,:]', ind_base1 : 'int[:,:]', ind_base2 : 'int[:,:]', mat : 'double[:,:,:]', mat_fun : 'double[:,:,:,:,:]'):
    
    mat[:, :, :] = 0.
    
    #$ omp parallel private(ie1, ie2, ie3, il1, il2, value, q1, q2, wvol, bi)
    #$ omp for reduction ( + : mat) 
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(n3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):

                        value = 0.

                        for q1 in range(nq1):
                            for q2 in range(nq2):

                                wvol = w1[ie1, q1] * w2[ie2, q2] * mat_fun[ie1, q1, ie2, q2, ie3]
                                bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                                value += wvol * bi

                        mat[ind_base1[ie1, il1], ind_base2[ie2, il2], ie3] += value
    #$ omp end parallel
    
    ierr = 0
    
    

    
# ==========================================================================================          
def kernel_l2error(nel : 'int[:]', p : 'int[:]', nq : 'int[:]', w1 : 'double[:,:]', w2 : 'double[:,:]', ni : 'int[:]', nj : 'int[:]', bi1 : 'double[:,:,:,:]', bi2 : 'double[:,:,:,:]', bj1 : 'double[:,:,:,:]', bj2 : 'double[:,:,:,:]', ind_basei1 : 'int[:,:]', ind_basei2 : 'int[:,:]', ind_basej1 : 'int[:,:]', ind_basej2 : 'int[:,:]', error : 'double[:,:]', mat_f1 : 'double[:,:,:,:]', mat_f2 : 'double[:,:,:,:]', c1 : 'double[:,:,:]', c2 : 'double[:,:,:]', mat_map : 'double[:,:,:,:]'):
    
    
    #$ omp parallel private(ie1, ie2, q1, q2, wvol, bi, bj, il1, il2, jl1, jl2)
    #$ omp for 
    
    # loop over all elements
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
                
            # loop over quadrature points in element
            for q1 in range(nq[0]):
                for q2 in range(nq[1]):

                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[ie1, q1, ie2, q2]

                    # evaluate discrete fields at quadrature point
                    bi = 0.
                    bj = 0. 

                    for il1 in range(p[0] + 1 - ni[0]):
                        for il2 in range(p[1] + 1 - ni[1]):

                            bi += c1[ind_basei1[ie1, il1], ind_basei2[ie2, il2], 0] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                    for jl1 in range(p[0] + 1 - nj[0]):
                        for jl2 in range(p[1] + 1 - nj[1]):

                            bj += c2[ind_basej1[ie1, jl1], ind_basej2[ie2, jl2], 0] * bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]


                    # compare this value to exact one and add contribution to error in element
                    error[ie1, ie2] += wvol * (bi - mat_f1[ie1, q1, ie2, q2]) * (bj - mat_f2[ie1, q1, ie2, q2])
                            
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================    
def kernel_evaluate_2form(nel : 'int[:]', n3 : 'int', p : 'int[:]', ns : 'int[:]', nq : 'int[:]', b_coeff : 'double[:,:,:]', ind_base1 : 'int[:,:]', ind_base2 : 'int[:,:]', bi1 : 'double[:,:,:,:]', bi2 : 'double[:,:,:,:]', b_eva : 'double[:,:,:,:,:]'):
    
    b_eva[:, :, :, :, :] = 0.
    
    #$ omp parallel private(ie1, ie2, ie3, q1, q2, il1, il2)
    #$ omp for 
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(n3):
                
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):

                        for il1 in range(p[0] + 1 - ns[0]):
                            for il2 in range(p[1] + 1 - ns[1]):
                        
                                b_eva[ie1, q1, ie2, q2, ie3] += b_coeff[ind_base1[ie1, il1], ind_base2[ie2, il2], ie3] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
    #$ omp end parallel
    
    ierr = 0