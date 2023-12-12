from numpy import shape

import psydac.core.arrays as arrays

# ===========================================================================================================
#                                                   1d
# ===========================================================================================================

# =============================================================================
def rhs0_1d(row1 : 'int[:]', col1 : 'int[:]', bsp1 : 'float[:,:]', mat_eq : 'float[:]', rhs : 'float[:]'):
    
    n_rows_1 = len(row1)
    
    for i1 in range(n_rows_1):   
        rhs[i1] = bsp1[row1[i1], col1[i1]] * mat_eq[row1[i1]]
        
        
        
# =============================================================================
def rhs1_1d(row1 : 'int[:]', col1 : 'int[:]', subs1 : 'int[:]', subs_cum1 : 'int[:]', wts1 : 'float[:,:]', bsp1 : 'float[:,:,:]', mat_eq : 'float[:,:]', rhs : 'float[:]'):
    
    n_rows_1 = len(row1)
    
    nq1 = shape(wts1)[1]
    
    for i1 in range(n_rows_1): 
                
        value = 0.

        for j1 in range(subs1[row1[i1]]):
            for q1 in range(nq1):
                value += wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1]

        rhs[i1] = value
        
        
# =============================================================================                
def rhs0_f_1d(indices1 : 'int[:,:]', bsp11 : 'float[:,:]', bsp12 : 'float[:,:]', mat_eq : 'float[:]', f : 'float[:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    rhs[:] = 0.
    
    for i1 in range(len(indices1[0])):
        
        i = indices1[3, i1]
        
        rhs[i] += f[indices1[0, i1]] * bsp11[indices1[0, i1], indices1[1, i1]] * bsp12[indices1[0, i1], indices1[2, i1]] * mat_eq[indices1[0, i1]]
        
        row[i]  = indices1[1, i1]
        col[i]  = indices1[2, i1]

        
# =============================================================================                
def rhs1_f_1d(indices1 : 'int[:,:]', subs1 : 'int[:]', subs_cum1 : 'int[:]', wts1 : 'float[:,:]', bsp11 : 'float[:,:,:]', bsp12 : 'float[:,:,:]', mat_eq : 'float[:,:]', f : 'float[:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq1 = shape(wts1)[1]
    
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
                
        value = 0.

        for j1 in range(subs1[indices1[0, i1]]):
            for q1 in range(nq1):
                value += wts1[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1] * bsp11[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1, indices1[1, i1]] * bsp12[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1, indices1[2, i1]] * mat_eq[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1]
                
        i = indices1[3, i1]

        rhs[i] += f[indices1[0, i1]] * value
        
        row[i]  = indices1[1, i1]
        col[i]  = indices1[2, i1]



        
# ===========================================================================================================
#                                                   2d
# ===========================================================================================================        
        

# =============================================================================
def rhs0_2d(row1 : 'int[:]', row2 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:]', mat_eq : 'float[:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    
    n1i, n1j = shape(bsp1)
    n2i, n2j = shape(bsp2)
    
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
                
            i      = n_rows_2*i1 + i2

            bsp    = bsp1[row1[i1], col1[i1]] * bsp2[row2[i2], col2[i2]]

            rhs[i] = bsp * mat_eq[row1[i1], row2[i2]]

            row[i] = n2i*row1[i1] + row2[i2]
            col[i] = n2j*col1[i1] + col2[i2]
        

        
# =============================================================================
def rhs11_2d(row1 : 'int[:]', row2 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', subs1 : 'int[:]', subs_cum1 : 'int[:]', wts1 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    
    n1i = nbase_d[0]
    n2i = nbase_n[1]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[1]
    
    nq1 = shape(wts1)[1]
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
                
            value = 0.

            for j1 in range(subs1[row1[i1]]):
                for q1 in range(nq1):
                    value += wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * bsp1[row1[i1] + j1 +  subs_cum1[row1[i1]], q1, col1[i1]] * mat_eq[row1[i1] + j1 +  subs_cum1[row1[i1]], q1, row2[i2]]

            i        = n_rows_2*i1 + i2

            rhs[i]   = value * bsp2[row2[i2], col2[i2]]

            row[i]   = n2i*row1[i1] + row2[i2]
            col[i]   = n2j*col1[i1] + col2[i2]
        
        

# =============================================================================
def rhs12_2d(row1 : 'int[:]', row2 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', subs2 : 'int[:]', subs_cum2 : 'int[:]', wts2 : 'float[:,:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    
    n1i = nbase_n[0]
    n2i = nbase_d[1]
    
    n1j = shape(bsp1)[1]
    n2j = shape(bsp2)[2]
    
    nq2 = shape(wts2)[1]
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
                
            value = 0.

            for j2 in range(subs2[row2[i2]]):
                for q2 in range(nq2):
                    value += wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2] * bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]] * mat_eq[row1[i1], row2[i2] + j2 + subs_cum2[row2[i2]], q2]

            i        = n_rows_2*i1 + i2

            rhs[i]   = value * bsp1[row1[i1], col1[i1]]

            row[i]   = n2i*row1[i1] + row2[i2]
            col[i]   = n2j*col1[i1] + col2[i2]      


          
# =============================================================================
def rhs2_2d(row1 : 'int[:]', row2 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', subs1 : 'int[:]', subs2 : 'int[:]', subs_cum1 : 'int[:]', subs_cum2 : 'int[:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    
    n1i = nbase_d[0]
    n2i = nbase_d[1]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[2]
    
    nq1 = shape(wts1)[1]
    nq2 = shape(wts2)[1]
    
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
                
            value = 0.

            for j1 in range(subs1[row1[i1]]):
                for j2 in range(subs2[row2[i2]]):
                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            w_vol  = wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2]

                            basis  = bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]]

                            value += w_vol * basis * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1, row2[i2] + j2 + subs_cum2[row2[i2]], q2]


            i      = n_rows_2*i1 + i2

            rhs[i] = value

            row[i] = n2i*row1[i1] + row2[i2]
            col[i] = n2j*col1[i1] + col2[i2]
            
            
    
# =============================================================================                
def rhs0_f_2d(indices1 : 'int[:,:]', indices2 : 'int[:,:]', bsp11 : 'float[:,:]', bsp12 : 'float[:,:]', bsp21 : 'float[:,:]', bsp22 : 'float[:,:]', mat_eq : 'float[:,:]', f : 'complex[:,:]', rhs : 'complex[:]', row : 'int[:]', col : 'int[:]'):  
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    
    n1i = shape(bsp11)[1]
    n2i = shape(bsp21)[1]
    
    n1j = shape(bsp12)[1]
    n2j = shape(bsp22)[1]
    
    rhs[:] = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
                
            i = nv2*indices1[3, i1] + indices2[3, i2]

            rhs[i] += f[indices1[0, i1], indices2[0, i2]] * mat_eq[indices1[0, i1], indices2[0, i2]] * bsp11[indices1[0, i1], indices1[1, i1]] * bsp12[indices1[0, i1], indices1[2, i1]] * bsp21[indices2[0, i2], indices2[1, i2]] * bsp22[indices2[0, i2], indices2[2, i2]]

            row[i]  = n2i*indices1[1, i1] + indices2[1, i2]
            col[i]  = n2j*indices1[2, i1] + indices2[2, i2]
                
                
                
                
                
# =============================================================================                
def rhs11_f_2d(indices1 : 'int[:,:]', indices2 : 'int[:,:]', subs1 : 'int[:]', subs_cum1 : 'int[:]', wts1 : 'float[:,:]', bsp11 : 'float[:,:,:]', bsp12 : 'float[:,:,:]', bsp21 : 'float[:,:]', bsp22 : 'float[:,:]', mat_eq : 'float[:,:,:]', f : 'complex[:,:]', rhs : 'complex[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq1 = shape(wts1)[1]
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    
    n1i = shape(bsp11)[2]
    n2i = shape(bsp21)[1]
    
    n1j = shape(bsp12)[2]
    n2j = shape(bsp22)[1]
    
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
                
            value = 0.

            for j1 in range(subs1[indices1[0, i1]]):
                for q1 in range(nq1):
                    value += wts1[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1] * bsp11[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1, indices1[1, i1]] * bsp12[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1, indices1[2, i1]] * mat_eq[indices1[0, i1] + j1 + subs_cum1[indices1[0, i1]], q1, indices2[0, i2]]

            i = nv2*indices1[3, i1] + indices2[3, i2]

            rhs[i] += f[indices1[0, i1], indices2[0, i2]] * value * bsp21[indices2[0, i2], indices2[1, i2]] * bsp22[indices2[0, i2], indices2[2, i2]]

            row[i]  = n2i*indices1[1, i1] + indices2[1, i2]
            col[i]  = n2j*indices1[2, i1] + indices2[2, i2]          
                
                
                

# =============================================================================                
def rhs12_f_2d(indices1 : 'int[:,:]', indices2 : 'int[:,:]', subs2 : 'int[:]', subs_cum2 : 'int[:]', wts2 : 'float[:,:]', bsp11 : 'float[:,:]', bsp12 : 'float[:,:]', bsp21 : 'float[:,:,:]', bsp22 : 'float[:,:,:]', mat_eq : 'float[:,:,:]', f : 'complex[:,:]', rhs : 'complex[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq2 = shape(wts2)[1]
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    
    n1i = shape(bsp11)[1]
    n2i = shape(bsp21)[2]
    
    n1j = shape(bsp12)[1]
    n2j = shape(bsp22)[2]
    
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
                
            value = 0.

            for j2 in range(subs2[indices2[0, i2]]):
                for q2 in range(nq2):
                    value += wts2[indices2[0, i2] + j2 + subs_cum2[indices2[0, i2]], q2] * bsp21[indices2[0, i2] + j2 + subs_cum2[indices2[0, i2]], q2, indices2[1, i2]] * bsp22[indices2[0, i2] + j2 + subs_cum2[indices2[0, i2]], q2, indices2[2, i2]] * mat_eq[indices1[0, i1], indices2[0, i2] + j2 + subs_cum2[indices2[0, i2]], q2]

            i = nv2*indices1[3, i1] + indices2[3, i2]

            rhs[i] += f[indices1[0, i1], indices2[0, i2]] * value * bsp11[indices1[0, i1], indices1[1, i1]] * bsp12[indices1[0, i1], indices1[2, i1]]

            row[i]  = n2i*indices1[1, i1] + indices2[1, i2]
            col[i]  = n2j*indices1[2, i1] + indices2[2, i2]
            
            
            
            
            
               
               
               
# ===========================================================================================================
#                                                   3d
# ===========================================================================================================                
                
                
                
# =============================================================================
def rhs0(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:]', bsp3 : 'float[:,:]', mat_eq : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i, n1j = shape(bsp1)
    n2i, n2j = shape(bsp2)
    n3i, n3j = shape(bsp3)
    
    #$ omp parallel private(i1, i2, i3, i, bsp)
    #$ omp for
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                i      = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                bsp    = bsp1[row1[i1], col1[i1]] * bsp2[row2[i2], col2[i2]] * bsp3[row3[i3], col3[i3]]
                
                rhs[i] = bsp * mat_eq[row1[i1], row2[i2], row3[i3]]
                
                row[i] = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i] = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs11(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs1 : 'int[:]', subs_cum1 : 'int[:]', wts1 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:]', bsp3 : 'float[:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_d[0]
    n2i = nbase_n[1]
    n3i = nbase_n[2]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[1]
    n3j = shape(bsp3)[1]
    
    nq1 = shape(wts1)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j1, q1, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j1 in range(subs1[row1[i1]]):
                    for q1 in range(nq1):
                        value += wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1, row2[i2], row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp2[row2[i2], col2[i2]]* bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs12(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs2 : 'int[:]', subs_cum2 : 'int[:]', wts2 : 'float[:,:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:,:]', bsp3 : 'float[:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_n[0]
    n2i = nbase_d[1]
    n3i = nbase_n[2]
    
    n1j = shape(bsp1)[1]
    n2j = shape(bsp2)[2]
    n3j = shape(bsp3)[1]
    
    nq2 = shape(wts2)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j2, q2, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j2 in range(subs2[row2[i2]]):
                    for q2 in range(nq2):
                        value += wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2] * bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]] * mat_eq[row1[i1], row2[i2] + j2 + subs_cum2[row2[i2]], q2, row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]] * bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs13(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs3 : 'int[:]', subs_cum3 : 'int[:]', wts3 : 'float[:,:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:]', bsp3 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_n[0]
    n2i = nbase_n[1]
    n3i = nbase_d[2]
    
    n1j = shape(bsp1)[1]
    n2j = shape(bsp2)[1]
    n3j = shape(bsp3)[2]
    
    nq3 = shape(wts3)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j3, q3, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j3 in range(subs3[row3[i3]]):
                    for q3 in range(nq3):
                        value += wts3[row3[i3] + j3 + subs_cum3[row3[i3]], q3] * bsp3[row3[i3] + j3 + subs_cum3[row3[i3]], q3, col3[i3]] * mat_eq[row1[i1], row2[i2], row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]] * bsp2[row2[i2], col2[i2]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs21(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs2 : 'int[:]', subs3 : 'int[:]', subs_cum2 : 'int[:]', subs_cum3 : 'int[:]', wts2 : 'float[:,:]', wts3 : 'float[:,:]', bsp1 : 'float[:,:]', bsp2 : 'float[:,:,:]', bsp3 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_n[0]
    n2i = nbase_d[1]
    n3i = nbase_d[2]
    
    n1j = shape(bsp1)[1]
    n2j = shape(bsp2)[2]
    n3j = shape(bsp3)[2]
    
    nq2 = shape(wts2)[1]
    nq3 = shape(wts3)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j2, q2, j3, q3, w_vol, basis, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j2 in range(subs2[row2[i2]]):
                    for q2 in range(nq2):
                        for j3 in range(subs3[row3[i3]]):
                            for q3 in range(nq3):
                                
                                w_vol  = wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2] * wts3[row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                                
                                basis  = bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]] * bsp3[row3[i3] + j3 + subs_cum3[row3[i3]], q3, col3[i3]]
                                
                                value += w_vol * basis * mat_eq[row1[i1], row2[i2] + j2 + subs_cum2[row2[i2]], q2, row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs22(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs1 : 'int[:]', subs3 : 'int[:]', subs_cum1 : 'int[:]', subs_cum3 : 'int[:]', wts1 : 'float[:,:]', wts3 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:]', bsp3 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_d[0]
    n2i = nbase_n[1]
    n3i = nbase_d[2]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[1]
    n3j = shape(bsp3)[2]
    
    nq1 = shape(wts1)[1]
    nq3 = shape(wts3)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j1, q1, j3, q3, w_vol, basis, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j1 in range(subs1[row1[i1]]):
                    for q1 in range(nq1):
                        for j3 in range(subs3[row3[i3]]):
                            for q3 in range(nq3):
                                
                                w_vol  = wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * wts3[row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                                
                                basis  = bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * bsp3[row3[i3] + j3 + subs_cum3[row3[i3]], q3, col3[i3]]
                                
                                value += w_vol * basis * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1, row2[i2], row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp2[row2[i2], col2[i2]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs23(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs1 : 'int[:]', subs2 : 'int[:]', subs_cum1 : 'int[:]', subs_cum2 : 'int[:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:,:]', bsp3 : 'float[:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_d[0]
    n2i = nbase_d[1]
    n3i = nbase_n[2]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[2]
    n3j = shape(bsp3)[1]
    
    nq1 = shape(wts1)[1]
    nq2 = shape(wts2)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j1, q1, j2, q2, w_vol, basis, i)
    #$ omp for 
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j1 in range(subs1[row1[i1]]):
                    for q1 in range(nq1):
                        for j2 in range(subs2[row2[i2]]):
                            for q2 in range(nq2):
                                
                                w_vol  = wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2]
                                
                                basis  = bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]]
                                
                                value += w_vol * basis * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1, row2[i2] + j2 + subs_cum2[row2[i2]], q2, row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
    
    
# =============================================================================
def rhs3(row1 : 'int[:]', row2 : 'int[:]', row3 : 'int[:]', col1 : 'int[:]', col2 : 'int[:]', col3 : 'int[:]', subs1 : 'int[:]', subs2 : 'int[:]', subs3 : 'int[:]', subs_cum1 : 'int[:]', subs_cum2 : 'int[:]', subs_cum3 : 'int[:]', wts1 : 'float[:,:]', wts2 : 'float[:,:]', wts3 : 'float[:,:]', bsp1 : 'float[:,:,:]', bsp2 : 'float[:,:,:]', bsp3 : 'float[:,:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', mat_eq : 'float[:,:,:,:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i = nbase_d[0]
    n2i = nbase_d[1]
    n3i = nbase_d[2]
    
    n1j = shape(bsp1)[2]
    n2j = shape(bsp2)[2]
    n3j = shape(bsp3)[2]
    
    nq1 = shape(wts1)[1]
    nq2 = shape(wts2)[1]
    nq3 = shape(wts3)[1]
    
    #$ omp parallel private(i1, i2, i3, value, j1, q1, j2, q2, j3, q3, w_vol, basis, i)
    #$ omp for 
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                for j1 in range(subs1[row1[i1]]):
                    for q1 in range(nq1):
                        for j2 in range(subs2[row2[i2]]):
                            for q2 in range(nq2):
                                for j3 in range(subs3[row3[i3]]):
                                    for q3 in range(nq3):
                                        
                                        w_vol  = wts1[row1[i1] + j1 + subs_cum1[row1[i1]], q1] * wts2[row2[i2] + j2 + subs_cum2[row2[i2]], q2] * wts3[row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                                        
                                        basis  = bsp1[row1[i1] + j1 + subs_cum1[row1[i1]], q1, col1[i1]] * bsp2[row2[i2] + j2 + subs_cum2[row2[i2]], q2, col2[i2]] * bsp3[row3[i3] + j3 + subs_cum3[row3[i3]], q3, col3[i3]]
                                        
                                        value += w_vol * basis * mat_eq[row1[i1] + j1 + subs_cum1[row1[i1]], q1, row2[i2] + j2 + subs_cum2[row2[i2]], q2, row3[i3] + j3 + subs_cum3[row3[i3]], q3]
                
                
                i      = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i] = value
                
                row[i] = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i] = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
    #$ omp end parallel
    
    ierr = 0
                    
                    
# =============================================================================                
def rhs11_f(indices1 : 'int[:,:]', indices2 : 'int[:,:]', indices3 : 'int[:,:]', n_row_sub1 : 'int[:]', sub1_cum : 'int[:]', wts1 : 'float[:,:]', bsp11 : 'float[:,:,:]', bsp12 : 'float[:,:,:]', bsp21 : 'float[:,:]', bsp22 : 'float[:,:]', bsp31 : 'float[:,:]', bsp32 : 'float[:,:]', mat_eq : 'float[:,:,:,:]', f : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq1 = shape(wts1)[1]
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    nv3 = arrays.max_vec_int(indices3[3]) + 1
    
    n1i = shape(bsp11)[2]
    n2i = shape(bsp21)[1]
    n3i = shape(bsp31)[1]
    
    n1j = shape(bsp12)[2]
    n2j = shape(bsp22)[1]
    n3j = shape(bsp32)[1]
    
    counter1 = 0
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
            for i3 in range(len(indices3[0])):
                
                value = 0.
                
                counter1 = sub1_cum[indices1[0, i1]]
                
                for j1 in range(n_row_sub1[indices1[0, i1]]):
                    for q1 in range(nq1):
                        value += wts1[indices1[0, i1] + j1 + counter1, q1] * bsp11[indices1[0, i1] + j1 + counter1, q1, indices1[1, i1]] * bsp12[indices1[0, i1] + j1 + counter1, q1, indices1[2, i1]] * mat_eq[indices1[0, i1] + j1 + counter1, q1, indices2[0, i2], indices3[0, i3]]
                        
                i = nv2*nv3*indices1[3, i1] + nv3*indices2[3, i2] + indices3[3, i3]
                
                rhs[i] += f[indices1[0, i1], indices2[0, i2], indices3[0, i3]] * value * bsp21[indices2[0, i2], indices2[1, i2]] * bsp22[indices2[0, i2], indices2[2, i2]] * bsp31[indices3[0, i3], indices3[1, i3]] * bsp32[indices3[0, i3], indices3[2, i3]]
                
                row[i]  = n2i*n3i*indices1[1, i1] + n3i*indices2[1, i2] + indices3[1, i3]
                col[i]  = n2j*n3j*indices1[2, i1] + n3j*indices2[2, i2] + indices3[2, i3]
    
    
# =============================================================================                
def rhs12_f(indices1 : 'int[:,:]', indices2 : 'int[:,:]', indices3 : 'int[:,:]', n_row_sub2 : 'int[:]', sub2_cum : 'int[:]', wts2 : 'float[:,:]', bsp11 : 'float[:,:]', bsp12 : 'float[:,:]', bsp21 : 'float[:,:,:]', bsp22 : 'float[:,:,:]', bsp31 : 'float[:,:]', bsp32 : 'float[:,:]', mat_eq : 'float[:,:,:,:]', f : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq2 = shape(wts2)[1]
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    nv3 = arrays.max_vec_int(indices3[3]) + 1
    
    n1i = shape(bsp11)[1]
    n2i = shape(bsp21)[2]
    n3i = shape(bsp31)[1]
    
    n1j = shape(bsp12)[1]
    n2j = shape(bsp22)[2]
    n3j = shape(bsp32)[1]
    
    counter2 = 0
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
            for i3 in range(len(indices3[0])):
                
                value = 0.
                
                counter2 = sub2_cum[indices2[0, i2]]
                
                for j2 in range(n_row_sub2[indices2[0, i2]]):
                    for q2 in range(nq2):
                        value += wts2[indices2[0, i2] + j2 + counter2, q2] * bsp21[indices2[0, i2] + j2 + counter2, q2, indices2[1, i2]] * bsp22[indices2[0, i2] + j2 + counter2, q2, indices2[2, i2]] * mat_eq[indices1[0, i1], indices2[0, i2] + j2 + counter2, q2, indices3[0, i3]]
                        
                i = nv2*nv3*indices1[3, i1] + nv3*indices2[3, i2] + indices3[3, i3]
                
                rhs[i] += f[indices1[0, i1], indices2[0, i2], indices3[0, i3]] * value * bsp11[indices1[0, i1], indices1[1, i1]] * bsp12[indices1[0, i1], indices1[2, i1]] * bsp31[indices3[0, i3], indices3[1, i3]] * bsp32[indices3[0, i3], indices3[2, i3]]
                
                row[i]  = n2i*n3i*indices1[1, i1] + n3i*indices2[1, i2] + indices3[1, i3]
                col[i]  = n2j*n3j*indices1[2, i1] + n3j*indices2[2, i2] + indices3[2, i3]
                
                
# =============================================================================                
def rhs13_f(indices1 : 'int[:,:]', indices2 : 'int[:,:]', indices3 : 'int[:,:]', n_row_sub3 : 'int[:]', sub3_cum : 'int[:]', wts3 : 'float[:,:]', bsp11 : 'float[:,:]', bsp12 : 'float[:,:]', bsp21 : 'float[:,:]', bsp22 : 'float[:,:]', bsp31 : 'float[:,:,:]', bsp32 : 'float[:,:,:]', mat_eq : 'float[:,:,:,:]', f : 'float[:,:,:]', rhs : 'float[:]', row : 'int[:]', col : 'int[:]'):  
        
    nq3 = shape(wts3)[1]
    
    nv1 = arrays.max_vec_int(indices1[3]) + 1
    nv2 = arrays.max_vec_int(indices2[3]) + 1
    nv3 = arrays.max_vec_int(indices3[3]) + 1
    
    n1i = shape(bsp11)[1]
    n2i = shape(bsp21)[1]
    n3i = shape(bsp31)[2]
    
    n1j = shape(bsp12)[1]
    n2j = shape(bsp22)[1]
    n3j = shape(bsp32)[2]
    
    counter3 = 0
    rhs[:]   = 0.
    
    for i1 in range(len(indices1[0])):
        for i2 in range(len(indices2[0])):
            for i3 in range(len(indices3[0])):
                
                value = 0.
                
                counter3 = sub3_cum[indices3[0, i3]]
                
                for j3 in range(n_row_sub3[indices3[0, i3]]):
                    for q3 in range(nq3):
                        value += wts3[indices3[0, i3] + j3 + counter3, q3] * bsp31[indices3[0, i3] + j3 + counter3, q3, indices3[1, i3]] * bsp32[indices3[0, i3] + j3 + counter3, q3, indices3[2, i3]] * mat_eq[indices1[0, i1], indices2[0, i2], indices3[0, i3] + j3 + counter3, q3]
                        
                i = nv2*nv3*indices1[3, i1] + nv3*indices2[3, i2] + indices3[3, i3]
                
                rhs[i] += f[indices1[0, i1], indices2[0, i2], indices3[0, i3]] * value * bsp11[indices1[0, i1], indices1[1, i1]] * bsp12[indices1[0, i1], indices1[2, i1]] * bsp21[indices2[0, i2], indices2[1, i2]] * bsp22[indices2[0, i2], indices2[2, i2]]
                
                row[i]  = n2i*n3i*indices1[1, i1] + n3i*indices2[1, i2] + indices3[1, i3]
                col[i]  = n2j*n3j*indices1[2, i1] + n3j*indices2[2, i2] + indices3[2, i3]                