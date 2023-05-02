import struphy.geometry.map_eval as kernel

from numpy import empty

# ================= 3d =================================                
def hybrid_curlA(starts1 : 'float[:,:]', starts2 : 'float[:,:]', starts3 : 'float[:,:]', spans1 : 'int[:]', spans2 : 'int[:]', spans3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int, nq1 : int, nq2 : int, nq3 : int, bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', data : 'float[:,:,:]', coeffs : 'float[:,:,:]'):
    
    nel1 = spans1.size
    nel2 = spans2.size
    nel3 = spans3.size 

    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, value, il1, il2, il3, i1, i2, i3)
    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            
                            value = 0.0 
                            for il1 in range(pi1 + 1):
                                i1 = spans1[iel1] - pi1 + il1 - starts1
                                for il2 in range(pi2 + 1):
                                    i2 = spans2[iel2] - pi2 + il2 - starts2
                                    for il3 in range(pi3 + 1): 
                                        i3 = spans3[iel3] - pi3 + il3 - starts3
                                        value +=  bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3] * coeffs[i1, i2, i3]         

                            data[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3] = value
    #$ omp end parallel

# ================= 3d =================================                
def hybrid_weight(pads1 : int, pads2 : int, pads3 : int, pts1: 'float[:,:]', pts2: 'float[:,:]', pts3: 'float[:,:]', 
                      spans1 : 'int[:]', spans2 : 'int[:]', spans3 : 'int[:]', 
                      nq1 : int, nq2 : int, nq3 : int, 
                      w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', 
                      data1 : 'float[:,:,:]', data2 : 'float[:,:,:]', data3 : 'float[:,:,:]', 
                      n_data : 'float[:,:,:,:,:,:]', kind_map: int, params_map: 'float[:]', p_map: 'int[:]',
                      t1: 'float[:]', t2: 'float[:]', t3: 'float[:]',
                      ind1: 'int[:,:]', ind2: 'int[:,:]', ind3: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    
    nel1 = spans1.size
    nel2 = spans2.size
    nel3 = spans3.size 

    df_out = empty((3,3), dtype=float)
    G      = empty((3,3), dtype=float)
    value_new  = empty(3, dtype=float)
    
    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, value1, value2, value3, eta1, eta2, eta3, df_out, G, overn, value_new)
    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            value1 = data1[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3]
                            value2 = data2[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3]
                            value3 = data3[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3]

                            eta1 = pts1[iel1, q1]
                            eta2 = pts2[iel2, q2]
                            eta3 = pts3[iel3, q3]

                            kernel.df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p_map, ind1, ind2, ind3, cx, cy, cz, df_out)
                            #sqrtg = kernel.det_df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p_map, ind1, ind2, ind3, cx, cy, cz)
                            
                            G[0,0] = df_out[0,0]*df_out[0,0] + df_out[1,0]*df_out[1,0] + df_out[2,0]*df_out[2,0]
                            G[0,1] = df_out[0,0]*df_out[0,1] + df_out[1,0]*df_out[1,1] + df_out[2,0]*df_out[2,1]
                            G[0,2] = df_out[0,0]*df_out[0,2] + df_out[1,0]*df_out[1,2] + df_out[2,2]*df_out[2,2]


                            G[1,1] = df_out[0,1]*df_out[0,1] + df_out[1,1]*df_out[1,1] + df_out[2,1]*df_out[2,1]
                            G[1,2] = df_out[0,1]*df_out[0,2] + df_out[1,1]*df_out[1,2] + df_out[2,1]*df_out[2,2]

                            G[2,2] = df_out[0,2]*df_out[0,2] + df_out[1,2]*df_out[1,2] + df_out[2,2]*df_out[2,2]

                            G[1,0] = G[0,1]
                            G[2,0] = G[0,2]
                            G[2,1] = G[1,2]

                            if n_data[pads1 + iel1, pads2 + iel2, pads3 + iel3, q1, q2, q3] < 0.001:
                                overn = 0.0
                            else:
                                overn = 1.0 / n_data[pads1 + iel1, pads2 + iel2, pads3 + iel3, q1, q2, q3]

                            value_new[0] = (G[0,0]*value1 + G[0,1]*value2 + G[0,2]*value3)*wvol*overn
                            value_new[1] = (G[1,0]*value1 + G[1,1]*value2 + G[1,2]*value3)*wvol*overn
                            value_new[2] = (G[2,0]*value1 + G[2,1]*value2 + G[2,2]*value3)*wvol*overn

                            data1[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3] = value_new[0]
                            data2[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3] = value_new[1]
                            data3[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3] = value_new[2]
    #$ omp end parallel
