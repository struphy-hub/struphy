from pyccel.decorators import types


#========= kernel for integration in 1d ========================================================================================= 
@types('int','double[:]','double[:]','double[:]')
def kernel_int_1d(nq1, w1, mat_f, f_loc):
    
    f_loc[:] = 0.
    
    for q1 in range(nq1):
        f_loc[:] += w1[q1] * mat_f[q1] 
#================================================================================================================================




#========= kernel for integration in 2d ========================================================================================= 
@types('int','int','double[:]','double[:]','double[:,:](order=F)','double[:]')
def kernel_int_2d(nq1, nq2, w1, w2, mat_f, f_loc):
    
    f_loc[:] = 0.
    
    for q1 in range(nq1):
        for q2 in range(nq2):
            f_loc[:] += w1[q1] * w2[q2] * mat_f[q1, q2] 
#================================================================================================================================
            


#========= kernel for integration in 3d ========================================================================================= 
@types('int','int','int','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:]')
def kernel_int_3d(nq1, nq2, nq3, w1, w2, w3, mat_f, f_loc):
    
    f_loc[:] = 0.
    
    for q1 in range(nq1):
        for q2 in range(nq2):
            for q3 in range(nq3):
                f_loc[:] += w1[q1] * w2[q2] * w3[q3] * mat_f[q1, q2, q3]
#================================================================================================================================