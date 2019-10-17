from pyccel.decorators import types
from pyccel.decorators import external_call




#================================================================================                                        
@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_mass(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2, bj3, Nbase1, Nbase2, Nbase3, M, mat_map):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):
            for ie3 in range(Nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            for jl1 in range(p1 + 1 - nj1):
                                for jl2 in range(p2 + 1 - nj2):
                                    for jl3 in range(p3 + 1 - nj3):

                                        value = 0.

                                        for q1 in range(nq1):
                                            for q2 in range(nq2):
                                                for q3 in range(nq3):
                                                    
                                                    wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]
                                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
                                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2] * bj3[ie3, jl3, 0, q3]
                                                    
                                                    value += wvol * bi * bj

                                        M[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2, (ie3 + il3)%Nbase3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] += value
#================================================================================






#================================================================================
@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_inner(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, L, mat_f, mat_map):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):
            for ie3 in range(Nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            
                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_map[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi * mat_f[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]

                            L[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2, (ie3 + il3)%Nbase3] += value
#================================================================================






#================================================================================
@external_call
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_L2error_V0(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, error, mat_f, mat_c, mat_g):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):
            for ie3 in range(Nel3):
                
                # Cycle over quadrature points
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_g[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]

                            # Evaluate basis at quadrature point
                            bi = 0.

                            for il1 in range(p1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(p3 + 1):

                                        bi += mat_c[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2, (ie3 + il3)%Nbase3] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                            error[ie1, ie2, ie3] += wvol * (bi - mat_f[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3])**2
#================================================================================