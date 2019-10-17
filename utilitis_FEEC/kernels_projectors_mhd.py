from pyccel.decorators import types
from pyccel.decorators import external_call





#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI0(n1, n2, n3, pl1, pl2, pl3, b1, b2, b3, mat, rhs):
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            rhs[ie1, ie2, ie3, il1, il2, il3] = b1[ie1, il1, 0, 0] * b2[ie2, il2, 0, 0] * b3[ie3, il3, 0, 0] * mat[ie1, ie2, ie3]
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI11(n1, n2, n3, pl1, pl2, pl3, ies_1, il_add_1, nq1, w1, b1, b2, b3, mat, rhs_1):

    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q1 in range(nq1):
                                rhs_1[ies_1[ie1], ie2, ie3, il1 + il_add_1[ie1], il2, il3] += w1[ie1, q1] * b1[ie1, il1, 0, q1] * b2[ie2, il2, 0, 0] * b3[ie3, il3, 0, 0] * mat[ie1*nq1 + q1, ie2, ie3]
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI12(n1, n2, n3, pl1, pl2, pl3, ies_2, il_add_2, nq2, w2, b1, b2, b3, mat, rhs_2):
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q2 in range(nq2):
                                rhs_2[ie1, ies_2[ie2], ie3, il1, il2 + il_add_2[ie2], il3] += w2[ie2, q2] * b1[ie1, il1, 0, 0] * b2[ie2, il2, 0, q2] * b3[ie3, il3, 0, 0] * mat[ie1, ie2*nq2 + q2, ie3]
#============================================================================================================================== 



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI13(n1, n2, n3, pl1, pl2, pl3, ies_3, il_add_3, nq3, w3, b1, b2, b3, mat, rhs_3):
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q3 in range(nq3):
                                rhs_3[ie1, ie2, ies_3[ie3], il1, il2, il3 + il_add_3[ie3]] += w3[ie3, q3] * b1[ie1, il1, 0, 0] * b2[ie2, il2, 0, 0] * b3[ie3, il3, 0, q3] * mat[ie1, ie2, ie3*nq3 + q3]
#============================================================================================================================== 



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int[:]','int[:]','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI21(n1, n2, n3, pl1, pl2, pl3, ies_2, ies_3, il_add_2, il_add_3, nq2, nq3, w2, w3, b1, b2, b3, mat, rhs_1):
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q2 in range(nq2):
                                for q3 in range(nq3):
                                    
                                    rhs_1[ie1, ies_2[ie2], ies_3[ie3], il1, il2 + il_add_2[ie2], il3 + il_add_3[ie3]] += w2[ie2, q2] * w3[ie3, q3] * b1[ie1, il1, 0, 0] * b2[ie2, il2, 0, q2] * b3[ie3, il3, 0, q3] * mat[ie1, ie2*nq2 + q2, ie3*nq3 + q3]
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int[:]','int[:]','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI22(n1, n2, n3, pl1, pl2, pl3, ies_1, ies_3, il_add_1, il_add_3, nq1, nq3, w1, w3, b1, b2, b3, mat, rhs_2):
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q1 in range(nq1):
                                for q3 in range(nq3):
                                    
                                    rhs_2[ies_1[ie1], ie2, ies_3[ie3], il1 + il_add_1[ie1], il2, il3 + il_add_3[ie3]] += w1[ie1, q1] * w3[ie3, q3] * b1[ie1, il1, 0, q1] * b2[ie2, il2, 0, 0] * b3[ie3, il3, 0, q3] * mat[ie1*nq1 + q1, ie2, ie3*nq3 + q3]
#==============================================================================================================================



#==============================================================================================================================
@external_call
@types('int','int','int','int','int','int','int[:]','int[:]','int[:]','int[:]','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_PI23(n1, n2, n3, pl1, pl2, pl3, ies_1, ies_2, il_add_1, il_add_2, nq1, nq2, w1, w2, b1, b2, b3, mat, rhs_3):

    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(pl1):
                    for il2 in range(pl2):
                        for il3 in range(pl3):
                            
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    
                                    rhs_3[ies_1[ie1], ies_2[ie2], ie3, il1 + il_add_1[ie1], il2 + il_add_2[ie2], il3] += w1[ie1, q1] * w2[ie2, q2] * b1[ie1, il1, 0, q1] * b2[ie2, il2, 0, q2] * b3[ie3, il3, 0, 0] * mat[ie1*nq1 + q1, ie2*nq2 + q2, ie3]
#==============================================================================================================================