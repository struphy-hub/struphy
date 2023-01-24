import struphy.geometry.map_eval as kernel

from numpy import empty

# ================= 3d =================================                
def quadrature_kernel(index1 : 'float[:,:]', index2 : 'float[:,:]', index3 : 'float[:,:]', el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int,  periodic1 : int,  periodic2 : int,  periodic3 : int, nq1 : int, nq2 : int, nq3 : int, bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', data : 'float[:,:,:,:,:,:]', coeffs : 'float[:,:,:]'):
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)
    nel3 = len(el_loc_indices3)    

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]
                
            for iel3 in range(nel3):

                if periodic3 == 1:
                    ie3 = iel3
                else:    
                    ie3 = el_loc_indices3[iel3]

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            
                            value = 0.0 
                            for il1 in range(pi1 + 1):
                                i1 = ie1 + il1 #index1[ie1, il1]
                                for il2 in range(pi2 + 1):
                                    i2 = ie2 + il2 #index2[ie2, il2]
                                    for il3 in range(pi3 + 1): 
                                        i3 = ie3 + il3 #index3[ie3, il3]
                                        value +=  bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3] * coeffs[i1, i2, i3]         

                            data[ie1, ie2, ie3, q1, q2, q3] = value


# ================= 3d =================================                
def kernelg(pts1: 'float[:,:]', pts2: 'float[:,:]', pts3: 'float[:,:]', el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]',  periodic1 : int,  periodic2 : int,  periodic3 : int, nq1 : int, nq2 : int, nq3 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', data1 : 'float[:,:,:,:,:,:]', data2 : 'float[:,:,:,:,:,:]', data3 : 'float[:,:,:,:,:,:]', n_data : 'float[:,:,:,:,:,:]', kind_map: int, params_map: 'float[:]',
                      p: 'int[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                      ind1: 'int[:,:]', ind2: 'int[:,:]', ind3: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)
    nel3 = len(el_loc_indices3)

    df_out = empty((3,3), dtype=float)
    G      = empty((3,3), dtype=float)
    value_new  = empty(3, dtype=float)
    

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]
                
            for iel3 in range(nel3):

                if periodic3 == 1:
                    ie3 = iel3
                else:    
                    ie3 = el_loc_indices3[iel3]

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3]

                            value1 = data1[ie1, ie2, ie3, q1, q2, q3]
                            value2 = data2[ie1, ie2, ie3, q1, q2, q3]
                            value3 = data3[ie1, ie2, ie3, q1, q2, q3]

                            eta1 = pts1[ie1, q1]
                            eta2 = pts2[ie2, q2]
                            eta3 = pts3[ie3, q3]

                            kernel.df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_out)
                            sqrtg = kernel.det_df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)
                            G[0,0] = df_out[0,0]*df_out[0,0] + df_out[1,0]*df_out[1,0] + df_out[2,0]*df_out[2,0]
                            G[0,1] = df_out[0,0]*df_out[0,1] + df_out[1,0]*df_out[1,1] + df_out[2,0]*df_out[2,1]
                            G[0,2] = df_out[0,0]*df_out[0,2] + df_out[1,0]*df_out[1,2] + df_out[2,2]*df_out[2,2]


                            G[1,1] = df_out[0,1]*df_out[0,1] + df_out[1,1]*df_out[1,1] + df_out[2,1]*df_out[2,1]
                            G[1,2] = df_out[0,1]*df_out[0,2] + df_out[1,1]*df_out[1,2] + df_out[2,1]*df_out[2,2]

                            G[2,2] = df_out[0,2]*df_out[0,2] + df_out[1,2]*df_out[1,2] + df_out[2,2]*df_out[2,2]

                            G[1,0] = G[0,1]
                            G[2,0] = G[0,2]
                            G[2,1] = G[1,2]

                            if n_data[ie1, ie2, ie3, q1, q2, q3] < 0.001:
                                overn = 0.0
                            else:
                                overn = 1.0 / n_data[ie1, ie2, ie3, q1, q2, q3]

                            value_new[0] = (G[0,0]*value1 + G[0,1]*value2 + G[0,2]*value3)/sqrtg*wvol*overn
                            value_new[1] = (G[1,0]*value1 + G[1,1]*value2 + G[1,2]*value3)/sqrtg*wvol*overn
                            value_new[2] = (G[2,0]*value1 + G[2,1]*value2 + G[2,2]*value3)/sqrtg*wvol*overn

                            data1[ie1, ie2, ie3, q1, q2, q3] = value_new[0]
                            data2[ie1, ie2, ie3, q1, q2, q3] = value_new[1]
                            data3[ie1, ie2, ie3, q1, q2, q3] = value_new[2]

