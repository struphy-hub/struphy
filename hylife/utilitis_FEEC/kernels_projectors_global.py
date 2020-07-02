from pyccel.decorators import types
import numpy  as np

#========= kernel for integration in 1d ========================================================================================= 
@types('int','double[:]','double[:]')
def kernel_int_1d(nq1, w1, mat_f):
    
    f_loc = 0.
    
    for q1 in range(nq1):
        f_loc += w1[q1] * mat_f[q1] 

    return f_loc
#================================================================================================================================




#========= kernel for integration in 2d ========================================================================================= 
@types('int','int','double[:]','double[:]','double[:,:]')
def kernel_int_2d(nq1, nq2, w1, w2, mat_f):
    
    f_loc = 0.
    
    for q1 in range(nq1):
        for q2 in range(nq2):
            f_loc += w1[q1] * w2[q2] * mat_f[q1, q2] 

    return f_loc
#================================================================================================================================
            


#========= kernel for integration in 3d ========================================================================================= 
@types('int','int','int','double[:]','double[:]','double[:]','double[:,:,:]')
def kernel_int_3d(nq1, nq2, nq3, w1, w2, w3, mat_f):
    
    f_loc = 0.
    
    for q1 in range(nq1):
        for q2 in range(nq2):
            for q3 in range(nq3):
                f_loc += w1[q1] * w2[q2] * w3[q3] * mat_f[q1, q2, q3]

    return f_loc
#================================================================================================================================
            


#========= kernel for integration in 1d along xi1 direction, reducing to a 3d array  ============================================
@types('double[:,:]','double[:,:,:,:]','double[:,:,:]')
def kernel_int_1d_ext_xi1(w1,mat_f,f_int):
    ne1,nq1,n2,n3 = np.shape(mat_f)
    for ie1 in range(ne1):
        for i2 in range(n2):
            for i3 in range(n3):
                f_int[ie1,i2,i3] = 0.
                for q1 in range(nq1):
                    f_int[ie1,i2,i3] += w1[ie1,q1]*mat_f[ie1,q1,i2,i3]
#================================================================================================================================

#========= kernel for integration in 1d along xi2 direction, reducing to a 3d array  ============================================
@types('double[:,:]','double[:,:,:,:]','double[:,:,:]')
def kernel_int_1d_ext_xi2(w2,mat_f,f_int):
    n1,ne2,nq2,n3 = np.shape(mat_f)
    for i1 in range(n1):
        for ie2 in range(ne2):
            for i3 in range(n3):
                f_int[i1,ie2,i3] = 0.
                for q2 in range(nq2):
                    f_int[i1,ie2,i3] += w2[ie2,q2]*mat_f[i1,ie2,q2,i3]
#================================================================================================================================


#========= kernel for integration in 1d along xi3 direction, reducing to a 3d array  ============================================
@types('double[:,:]','double[:,:,:,:]','double[:,:,:]')
def kernel_int_1d_ext_xi3(w3,mat_f,f_int):
    n1,n2,ne3,nq3 = np.shape(mat_f)
    for i1 in range(n1):
        for i2 in range(n2):
            for ie3 in range(ne3):
                f_int[i1,i2,ie3] = 0.
                for q3 in range(nq3):
                    f_int[i1,i2,ie3] += w3[ie3,q3]*mat_f[i1,i2,ie3,q3]
#================================================================================================================================

#========= kernel for integration in 2d in xi2-xi3 plane , reducing to a 3d array  ==============================================
@types('double[:,:]','double[:,:]','double[:,:,:,:,:]','double[:,:,:]')
def kernel_int_2d_ext_xi2_xi3(w2,w3, mat_f,f_int):
    n1,ne2,nq2,ne3,nq3 = np.shape(mat_f)
    for i1 in range(n1):
        for ie2 in range(ne2):
            for ie3 in range(ne3):
                f_int[i1,ie2,ie3] = 0.
                for q2 in range(nq2):
                    for q3 in range(nq3):
                        f_int[i1,ie2,ie3] += w2[ie2,q2]* w3[ie3,q3]*mat_f[i1,ie2,q2,ie3,q3]
#================================================================================================================================

#========= kernel for integration in 2d in xi1-xi3 plane , reducing to a 3d array  ==============================================
@types('double[:,:]','double[:,:]','double[:,:,:,:,:]','double[:,:,:]')
def kernel_int_2d_ext_xi1_xi3(w1,w3, mat_f,f_int):
    ne1,nq1,n2,ne3,nq3 = np.shape(mat_f)
    for ie1 in range(ne1):
        for i2 in range(n2):
            for ie3 in range(ne3):
                f_int[ie1,i2,ie3] = 0.
                for q1 in range(nq1):
                    for q3 in range(nq3):
                        f_int[ie1,i2,ie3] += w1[ie1,q1]* w3[ie3,q3]*mat_f[ie1,q1,i2,ie3,q3]
#================================================================================================================================

#========= kernel for integration in 2d in xi1-xi2 plane , reducing to a 3d array  ==============================================
@types('double[:,:]','double[:,:]','double[:,:,:,:,:]','double[:,:,:]')
def kernel_int_2d_ext_xi1_xi2(w1,w2, mat_f,f_int):
    ne1,nq1,ne2,nq2,n3 = np.shape(mat_f)
    for ie1 in range(ne1):
        for ie2 in range(ne2):
            for i3 in range(n3):
                f_int[ie1,ie2,i3] = 0.
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        f_int[ie1,ie2,i3] += w1[ie1,q1]* w2[ie2,q2]*mat_f[ie1,q1,ie2,q2,i3]
#================================================================================================================================

#========= kernel for integration in 3d, reducing to a 3d array  ================================================================
@types('double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:]')
def kernel_int_3d_ext(w1,w2,w3, mat_f,f_int):
    ne1,nq1,ne2,nq2,ne3,nq3=np.shape(mat_f)
    for ie1 in range(ne1):
        for ie2 in range(ne2):
            for ie3 in range(ne3):
                f_int[ie1,ie2,ie3] = 0.
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            f_int[ie1,ie2,ie3] += w1[ie1,q1]*w2[ie2,q2]* w3[ie3,q3]*mat_f[ie1,q1,ie2,q2,ie3,q3]
#================================================================================================================================
