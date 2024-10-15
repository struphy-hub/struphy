from pyccel.decorators import types

import struphy.linear_algebra.linalg_kernels as linalg




# ==========================================================================================          
@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]', 'int','int','int','int','int','int','int','int','int','int','int','int','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def right_hand(idnx, idny, idnz, iddx, iddy, iddz, nel1, nel2, nel3, nq1, nq2, nq3, p1, p2, p3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, right_1, right_2, right_3, temp_vector_1, temp_vector_2, temp_vector_3):

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value = 0.
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(d3 + 1):

                                        value  += bn1[ie1, il1, 0, q1] * bd2[ie2, il2, 0, q2] * bd3[ie3, il3, 0, q3] * temp_vector_1[idnx[ie1, il1], iddy[ie2, il2], iddz[ie3, il3]]

                            right_1[ie1, ie2, ie3, q1, q2, q3] = value  

    #$ omp end do
    #$ omp end parallel


    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            
                            value = 0.
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):

                                        value += bd1[ie1, il1, 0, q1] * bn2[ie2, il2, 0, q2] * bd3[ie3, il3, 0, q3] * temp_vector_2[iddx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]
                                         

                            right_2[ie1, ie2, ie3, q1, q2, q3] = value  

    #$ omp end do
    #$ omp end parallel

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value = 0.
                            for il1 in range(d1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):

                                        value += bd1[ie1, il1, 0, q1] * bd2[ie2, il2, 0, q2] * bn3[ie3, il3, 0, q3] * temp_vector_3[iddx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]
                                         

                            right_3[ie1, ie2, ie3, q1, q2, q3] = value   

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0



# ==========================================================================================          
@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]', 'int','int','int','int','int','int','int','int','int','int','int','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def bvalue(idnx, idny, idnz, iddx, iddy, iddz, nel1, nel2, nel3, nq1, nq2, nq3, p1, p2, p3, d1, d2, d3, b1, b2, b3, bn1, bn2, bn3, bd1, bd2, bd3, b1value, b2value, b3value, temp_vector_1, temp_vector_2, temp_vector_3):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value = 0.
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += bd1[ie1, il1, 0, q1] * bn2[ie2, il2, 0, q2] * bn3[ie3, il3, 0, q3] * b1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]]

                            b1value[ie1, ie2, ie3, q1, q2, q3] = value  

    #$ omp end do
    #$ omp end parallel

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value = 0.
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += bn1[ie1, il1, 0, q1] * bd2[ie2, il2, 0, q2] * bn3[ie3, il3, 0, q3] * b2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]

                            b2value[ie1, ie2, ie3, q1, q2, q3] = value  

    #$ omp end do
    #$ omp end parallel

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value = 0.
                            for il1 in range(p1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += bn1[ie1, il1, 0, q1] * bn2[ie2, il2, 0, q2] * bd3[ie3, il3, 0, q3] * b3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]

                            b3value[ie1, ie2, ie3, q1, q2, q3] = value  

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0





# ==========================================================================================          
@types('int','int','int','int','int','int','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def weight(nel1, nel2, nel3, nq1, nq2, nq3, b1value, b2value, b3value, right_1, right_2, right_3, weight1, weight2, weight3):

    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            weight1[ie1,ie2,ie3,q1,q2,q3] = b2value[ie1,ie2,ie3,q1,q2,q3] * right_3[ie1,ie2,ie3,q1,q2,q3] - b3value[ie1,ie2,ie3,q1,q2,q3] * right_2[ie1,ie2,ie3,q1,q2,q3]
                            weight2[ie1,ie2,ie3,q1,q2,q3] = b3value[ie1,ie2,ie3,q1,q2,q3] * right_1[ie1,ie2,ie3,q1,q2,q3] - b1value[ie1,ie2,ie3,q1,q2,q3] * right_3[ie1,ie2,ie3,q1,q2,q3]
                            weight3[ie1,ie2,ie3,q1,q2,q3] = b1value[ie1,ie2,ie3,q1,q2,q3] * right_2[ie1,ie2,ie3,q1,q2,q3] - b2value[ie1,ie2,ie3,q1,q2,q3] * right_1[ie1,ie2,ie3,q1,q2,q3]
  
    #$ omp end do
    #$ omp end parallel
    ierr = 0




# ==========================================================================================          
@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]', 'int','int','int','int','int','int','int','int','int','int','int','int','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]')
def final(idnx, idny, idnz, iddx, iddy, iddz, nel1, nel2, nel3, nq1, nq2, nq3, p1, p2, p3, d1, d2, d3, weight1, weight2, weight3, temp_final_1, temp_final_2, temp_final_3, bn1, bn2, bn3, bd1, bd2, bd3):

    temp_final_1[:,:,:] = 0.0
    temp_final_2[:,:,:] = 0.0
    temp_final_3[:,:,:] = 0.0
    
    #$ omp parallel
    #$ omp do reduction ( + : temp_final_1) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(p1 + 1):
                    for il2 in range(d2 + 1):
                        for il3 in range(d3 + 1):

                            value = 0.
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += weight1[ie1,ie2,ie3,q1,q2,q3] * bn1[ie1, il1, 0, q1] * bd2[ie2, il2, 0, q2] * bd3[ie3, il3, 0, q3] 

                            temp_final_1[idnx[ie1, il1], iddy[ie2, il2], iddz[ie3, il3]] += value 

    #$ omp end do
    #$ omp end parallel

    #$ omp parallel
    #$ omp do reduction ( + : temp_final_2) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(d1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(d3 + 1):

                            value = 0.
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += weight2[ie1,ie2,ie3,q1,q2,q3] * bd1[ie1, il1, 0, q1] * bn2[ie2, il2, 0, q2] * bd3[ie3, il3, 0, q3] 
                            
                            temp_final_2[iddx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]] += value

    #$ omp end do
    #$ omp end parallel


    #$ omp parallel
    #$ omp do reduction ( + : temp_final_3) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(d1 + 1):
                    for il2 in range(d2 + 1):
                        for il3 in range(p3 + 1):

                            value = 0.
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += weight3[ie1,ie2,ie3,q1,q2,q3] * bd1[ie1, il1, 0, q1] * bd2[ie2, il2, 0, q2] * bn3[ie3, il3, 0, q3] 
                            
                            temp_final_3[iddx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]] += value
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0








# ==========================================================================================          
@types('double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:]','double[:]','double[:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def right_bwvalue(G_inv_11, G_inv_12, G_inv_13, G_inv_22, G_inv_23, G_inv_33, nel1, nel2, nel3, nq1, nq2, nq3, dft, pts1, pts2, pts3, wts1, wts2, wts3, generate_weight1, generate_weight2, generate_weight3, b1value, b2value, b3value, uvalue, weight1, weight2, weight3, right_1, right_2, right_3):
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, generate_weight1,  generate_weight3, temp)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            dft[0,0] = G_inv_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0,1] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[0,2] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[1,0] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[1,1] = G_inv_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1,2] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2,0] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[2,1] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2,2] = G_inv_33[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[0] = b1value[ie1,ie2,ie3,q1,q2,q3]
                            generate_weight1[1] = b2value[ie1,ie2,ie3,q1,q2,q3]
                            generate_weight1[2] = b3value[ie1,ie2,ie3,q1,q2,q3]
                            temp = uvalue[ie1,ie2,ie3,q1,q2,q3] * wts1[ie1, q1] * wts2[ie2, q2] * wts3[ie3, q3]
                            linalg.matrix_vector(dft, generate_weight1, generate_weight3)
                            b1value[ie1,ie2,ie3,q1,q2,q3] = generate_weight3[0] * temp 
                            b2value[ie1,ie2,ie3,q1,q2,q3] = generate_weight3[1] * temp 
                            b3value[ie1,ie2,ie3,q1,q2,q3] = generate_weight3[2] * temp # we here multilying uvalue and weights into b1, b2, b3
                            weight1[ie1,ie2,ie3,q1,q2,q3] = temp * (generate_weight3[1] * right_3[ie1,ie2,ie3,q1,q2,q3] - generate_weight3[2] * right_2[ie1,ie2,ie3,q1,q2,q3])
                            weight2[ie1,ie2,ie3,q1,q2,q3] = temp * (generate_weight3[2] * right_1[ie1,ie2,ie3,q1,q2,q3] - generate_weight3[0] * right_3[ie1,ie2,ie3,q1,q2,q3]) 
                            weight3[ie1,ie2,ie3,q1,q2,q3] = temp * (generate_weight3[0] * right_2[ie1,ie2,ie3,q1,q2,q3] - generate_weight3[1] * right_1[ie1,ie2,ie3,q1,q2,q3]) 

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0

