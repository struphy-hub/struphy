from pyccel.decorators import types

from numpy import shape


# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','double[:]','int[:]','int[:]')
def rhs0(row1, row2, row3, col1, col2, col3, bsp1, bsp2, bsp3, mat_eq, rhs, row, col):
    
    n_rows_1 = len(row1)
    n_rows_2 = len(row2)
    n_rows_3 = len(row3)
    
    n1i, n1j = shape(bsp1)
    n2i, n2j = shape(bsp2)
    n3i, n3j = shape(bsp3)
    
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                i      = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                bsp    = bsp1[row1[i1], col1[i1]] * bsp2[row2[i2], col2[i2]] * bsp3[row3[i3], col3[i3]]
                
                rhs[i] = bsp * mat_eq[row1[i1], row2[i2], row3[i3]]
                
                row[i] = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i] = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
                
                
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:,:]','double[:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:]','int[:]','int[:]')
def rhs11(row1, row2, row3, col1, col2, col3, n_row_sub1, sub1_cum, wts1, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter1 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter1 = sub1_cum[row1[i1]]
                
                for j1 in range(n_row_sub1[row1[i1]]):
                    for q1 in range(nq1):
                        value += wts1[row1[i1] + j1 + counter1, q1] * bsp1[row1[i1] + j1 + counter1, q1, col1[i1]] * mat_eq[row1[i1] + j1 + counter1, q1, row2[i2], row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp2[row2[i2], col2[i2]]* bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]   
                
                
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:]','double[:]','int[:]','int[:]')
def rhs12(row1, row2, row3, col1, col2, col3, n_row_sub2, sub2_cum, wts2, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter2 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter2 = sub2_cum[row2[i2]]
                
                for j2 in range(n_row_sub2[row2[i2]]):
                    for q2 in range(nq2):
                        value += wts2[row2[i2] + j2 + counter2, q2] * bsp2[row2[i2] + j2 + counter2, q2, col2[i2]] * mat_eq[row1[i1], row2[i2] + j2 + counter2, q2, row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]] * bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]   
    
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','int[:]','int[:]','double[:,:,:,:]','double[:]','int[:]','int[:]')
def rhs13(row1, row2, row3, col1, col2, col3, n_row_sub3, sub3_cum, wts3, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter3 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter3 = sub3_cum[row3[i3]]
                
                for j3 in range(n_row_sub3[row3[i3]]):
                    for q3 in range(nq3):
                        value += wts3[row3[i3] + j3 + counter3, q3] * bsp3[row3[i3] + j3 + counter3, q3, col3[i3]] * mat_eq[row1[i1], row2[i2], row3[i3] + j3 + counter3, q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]] * bsp2[row2[i2], col2[i2]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]   

                

                              
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','int[:]','int[:]','double[:,:,:,:,:]','double[:]','int[:]','int[:]')
def rhs21(row1, row2, row3, col1, col2, col3, n_row_sub2, n_row_sub3, sub2_cum, sub3_cum, wts2, wts3, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter2 = 0
    counter3 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter2 = sub2_cum[row2[i2]]
                counter3 = sub3_cum[row3[i3]]
                
                for j2 in range(n_row_sub2[row2[i2]]):
                    for q2 in range(nq2):
                        for j3 in range(n_row_sub3[row3[i3]]):
                            for q3 in range(nq3):
                                
                                w_vol  = wts2[row2[i2] + j2 + counter2, q2] * wts3[row3[i3] + j3 + counter3, q3]
                                
                                basis  = bsp2[row2[i2] + j2 + counter2, q2, col2[i2]] * bsp3[row3[i3] + j3 + counter3, q3, col3[i3]]
                                
                                value += w_vol * basis * mat_eq[row1[i1], row2[i2] + j2 + counter2, q2, row3[i3] + j3 + counter3, q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp1[row1[i1], col1[i1]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]                
                
                
                
                
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:]','double[:,:,:]','int[:]','int[:]','double[:,:,:,:,:]','double[:]','int[:]','int[:]')
def rhs22(row1, row2, row3, col1, col2, col3, n_row_sub1, n_row_sub3, sub1_cum, sub3_cum, wts1, wts3, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter1 = 0
    counter3 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter1 = sub1_cum[row1[i1]]
                counter3 = sub3_cum[row3[i3]]
                
                for j1 in range(n_row_sub1[row1[i1]]):
                    for q1 in range(nq1):
                        for j3 in range(n_row_sub3[row3[i3]]):
                            for q3 in range(nq3):
                                
                                w_vol  = wts1[row1[i1] + j1 + counter1, q1] * wts3[row3[i3] + j3 + counter3, q3]
                                
                                basis  = bsp1[row1[i1] + j1 + counter1, q1, col1[i1]] * bsp3[row3[i3] + j3 + counter3, q3, col3[i3]]
                                
                                value += w_vol * basis * mat_eq[row1[i1] + j1 + counter1, q1, row2[i2], row3[i3] + j3 + counter3, q3]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp2[row2[i2], col2[i2]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]          
                
                
                
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','int[:]','int[:]','double[:,:,:,:,:]','double[:]','int[:]','int[:]')
def rhs23(row1, row2, row3, col1, col2, col3, n_row_sub1, n_row_sub2, sub1_cum, sub2_cum, wts1, wts2, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter1 = 0
    counter2 = 0
    
    for i1 in range(n_rows_1): 
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter1 = sub1_cum[row1[i1]]
                counter2 = sub2_cum[row2[i2]]
                
                for j1 in range(n_row_sub1[row1[i1]]):
                    for q1 in range(nq1):
                        for j2 in range(n_row_sub2[row2[i2]]):
                            for q2 in range(nq2):
                                
                                w_vol  = wts1[row1[i1] + j1 + counter1, q1] * wts2[row2[i2] + j2 + counter2, q2]
                                
                                basis  = bsp1[row1[i1] + j1 + counter1, q1, col1[i1]] * bsp2[row2[i2] + j2 + counter2, q2, col2[i2]]
                                
                                value += w_vol * basis * mat_eq[row1[i1] + j1 + counter1, q1, row2[i2] + j2 + counter2, q2, row3[i3]]
                        
                i        = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i]   = value * bsp3[row3[i3], col3[i3]]
                
                row[i]   = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i]   = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]
                
                
                             
# =============================================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int[:]','int[:]','double[:,:,:,:,:,:]','double[:]','int[:]','int[:]')
def rhs3(row1, row2, row3, col1, col2, col3, n_row_sub1, n_row_sub2, n_row_sub3, sub1_cum, sub2_cum, sub3_cum, wts1, wts2, wts3, bsp1, bsp2, bsp3, nbase_n, nbase_d, mat_eq, rhs, row, col):
    
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
    
    counter1 = 0
    counter2 = 0
    counter3 = 0
    
    for i1 in range(n_rows_1):
        for i2 in range(n_rows_2):
            for i3 in range(n_rows_3):
                
                value = 0.
                
                counter1 = sub1_cum[row1[i1]]
                counter2 = sub2_cum[row2[i2]]
                counter3 = sub3_cum[row3[i3]]
                
                for j1 in range(n_row_sub1[row1[i1]]):
                    for q1 in range(nq1):
                        for j2 in range(n_row_sub2[row2[i2]]):
                            for q2 in range(nq2):
                                for j3 in range(n_row_sub3[row3[i3]]):
                                    for q3 in range(nq3):
                                        
                                        w_vol  = wts1[row1[i1] + j1 + counter1, q1] * wts2[row2[i2] + j2 + counter2, q2] * wts3[row3[i3] + j3 + counter3, q3]
                                        
                                        basis  = bsp1[row1[i1] + j1 + counter1, q1, col1[i1]] * bsp2[row2[i2] + j2 + counter2, q2, col2[i2]] * bsp3[row3[i3] + j3 + counter3, q3, col3[i3]]
                                        
                                        value += w_vol * basis * mat_eq[row1[i1] + j1 + counter1, q1, row2[i2] + j2 + counter2, q2, row3[i3] + j3 + counter3, q3]
                
                
                i      = n_rows_2*n_rows_3*i1 + n_rows_3*i2 + i3
                
                rhs[i] = value
                
                row[i] = n2i*n3i*row1[i1] + n3i*row2[i2] + row3[i3]
                col[i] = n2j*n3j*col1[i1] + n3j*col2[i2] + col3[i3]                    
                    
                    
# =============================================================================                
@types('int[:,:]','int[:,:]','int[:,:]','int[:]','int[:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:]','double[:]','int[:]','int[:]')
def rhs11_f(indices1, indices2, indices3, n_row_sub1, sub1_cum, wts1, bsp11, bsp12, bsp21, bsp22, bsp31, bsp32, mat_eq, f, rhs, row, col):  
        
    nq1 = shape(wts1)[1]
    
    nv1 = max(indices1[3]) + 1
    nv2 = max(indices2[3]) + 1
    nv3 = max(indices3[3]) + 1
    
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
@types('int[:,:]','int[:,:]','int[:,:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:]','double[:]','int[:]','int[:]')
def rhs12_f(indices1, indices2, indices3, n_row_sub2, sub2_cum, wts2, bsp11, bsp12, bsp21, bsp22, bsp31, bsp32, mat_eq, f, rhs, row, col):  
        
    nq2 = shape(wts2)[1]
    
    nv1 = max(indices1[3]) + 1
    nv2 = max(indices2[3]) + 1
    nv3 = max(indices3[3]) + 1
    
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
@types('int[:,:]','int[:,:]','int[:,:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:]','double[:,:,:]','double[:]','int[:]','int[:]')
def rhs13_f(indices1, indices2, indices3, n_row_sub3, sub3_cum, wts3, bsp11, bsp12, bsp21, bsp22, bsp31, bsp32, mat_eq, f, rhs, row, col):  
        
    nq3 = shape(wts3)[1]
    
    nv1 = max(indices1[3]) + 1
    nv2 = max(indices2[3]) + 1
    nv3 = max(indices3[3]) + 1
    
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
                
                
                
# ============================================================================               
@types('int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:,:]','double[:]')
def rhs1_d(row, col, n_row_sub, sub_cum, wts, bsp, rhs):
    
    n_rows = len(row)
    
    nq = shape(wts)[1]
    
    counter = 0
    
    for i in range(n_rows):
        
        value = 0.
        
        counter = sub_cum[row[i]]
        
        for j in range(n_row_sub[row[i]]):
            for q in range(nq):
                value += wts[row[i] + j + counter, q] * bsp[row[i] + j + counter, q, col[i]]

        rhs[i] = value