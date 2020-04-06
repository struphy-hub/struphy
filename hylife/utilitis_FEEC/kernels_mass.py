from pyccel.decorators import types

#from ..geometry import  mappings_analytical as mapping
import hylife.geometry.mappings_analytical as mapping


# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun_3d(xi1, xi2, xi3, kind_fun, kind_map, params):
    
    value = 0.
    
    # quantities for mass matrix V0
    if   kind_fun == 1:
        value = mapping.det_df(xi1, xi2, xi3, kind_map, params)
    
    # quantities for mass matrix V1
    elif kind_fun == 11:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 12:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 13:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 14:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 15:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 16:
        value = mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33) * mapping.det_df(xi1, xi2, xi3, kind_map, params)
        
    # quantities for mass matrix V2
    elif kind_fun == 21:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 11) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 22:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 21) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 23:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 22) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 24:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 31) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 25:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 32) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 26:
        value = mapping.g(xi1, xi2, xi3, kind_map, params, 33) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
        
    # quantities for mass matrix V3
    elif kind_fun == 2:
        value = 1. / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    
    return value


# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','int','int','double[:]')        
def kernel_eva_3d(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, params):
    
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] = fun_3d(xi1[i1], xi2[i2], xi3[i3], kind_fun, kind_map, params)




                
#================================================================================                                        
@types('int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','double[:,:,:,:](order=F)','double[:,:](order=F)')
def kernel_mass_2d(Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, nj1, nj2, w1, w2, bi1, bi2, bj1, bj2, Nbase1, Nbase2, M, mat_map):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):
                    for jl1 in range(p1 + 1 - nj1):
                        for jl2 in range(p2 + 1 - nj2):

                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):

                                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[nq1*ie1 + q1, nq2*ie2 + q2]
                                    bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]
                                    bj   = bj1[ie1, jl1, 0, q1] * bj2[ie2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            M[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2, p1 + jl1 - il1, p2 + jl2 - il2] += value
#================================================================================





#================================================================================                                        
@types('int','int','int','int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:,:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_mass_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, nj1, nj2, nj3, w1, w2, w3, bi1, bi2, bi3, bj1, bj2, bj3, Nbase1, Nbase2, Nbase3, M, mat_map):
    
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
@types('int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def kernel_inner_2d(Nel1, Nel2, p1, p2, nq1, nq2, ni1, ni2, w1, w2, bi1, bi2, Nbase1, Nbase2, L, mat_f, mat_map):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):

            for il1 in range(p1 + 1 - ni1):
                for il2 in range(p2 + 1 - ni2):

                    value = 0.

                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            wvol = w1[ie1, q1] * w2[ie2, q2] * mat_map[nq1*ie1 + q1, nq2*ie2 + q2]
                            bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                            value += wvol * bi * mat_f[nq1*ie1 + q1, nq2*ie2 + q2]

                    L[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2] += value
#================================================================================




#================================================================================
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_inner_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, L, mat_f, mat_map):
    
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
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def kernel_l2error_v0_2d(Nel1, Nel2, p1, p2, nq1, nq2, w1, w2, bi1, bi2, Nbase1, Nbase2, error, mat_f, mat_c, mat_g):
    
    for ie1 in range(Nel1):
        for ie2 in range(Nel2):
                
            # Cycle over quadrature points
            for q1 in range(nq1):
                for q2 in range(nq2):
            
                    wvol = w1[ie1, q1] * w2[ie2, q2] * mat_g[nq1*ie1 + q1, nq2*ie2 + q2]

                    # Evaluate basis at quadrature point
                    bi = 0.

                    for il1 in range(p1 + 1):
                        for il2 in range(p2 + 1):
                    
                            bi += mat_c[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2]

                    error[ie1, ie2] += wvol * (bi - mat_f[nq1*ie1 + q1, nq2*ie2 + q2])**2
#================================================================================




#================================================================================
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_l2error_v0_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, error, mat_f, mat_c, mat_g):
    
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
