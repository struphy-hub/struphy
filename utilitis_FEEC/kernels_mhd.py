from pyccel.decorators import types
from pyccel.decorators import external_call



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_A_1(ni, nj, nk, nq2, nq3, w2, w3, mat_f1, rhs_1):
    
    rhs_1[:, :, :] = 0.
    
    for i in range(ni):
        
        for j in range(nj):
            for k in range(nk):
                
                f_loc = 0.
                
                for q2 in range(nq2):
                    for q3 in range(nq3):
                        f_loc += w2[j, q2] * w3[k, q3] * mat_f1[i, j*nq2 + q2, k*nq3 + q3]
                        
                rhs_1[i, j, k] = f_loc
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_A_2(ni, nj, nk, nq1, nq3, w1, w3, mat_f2, rhs_2):
    
    rhs_2[:, :, :] = 0.
    
    for j in range(nj):
        
        for i in range(ni):
            for k in range(nk):
                
                f_loc = 0.
                
                for q1 in range(nq1):
                    for q3 in range(nq3):
                        f_loc += w1[i, q1] * w3[k, q3] * mat_f2[i*nq1 + q1, j, k*nq3 + q3]
                        
                rhs_2[i, j, k] = f_loc
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_A_3(ni, nj, nk, nq1, nq2, w1, w2, mat_f3, rhs_3):
    
    rhs_3[:, :, :] = 0.
    
    for k in range(nk):
        
        for i in range(ni):
            for j in range(nj):
                
                f_loc = 0.
                
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        f_loc += w1[i, q1] * w2[j, q2] * mat_f3[i*nq1 + q1, j*nq2 + q2, k]
                        
                rhs_3[i, j, k] = f_loc
#==============================================================================================================================