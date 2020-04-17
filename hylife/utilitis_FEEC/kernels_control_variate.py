# import pyccel decorators
from pyccel.decorators import types

# import of subroutines
import hylife.geometry.mappings_analytical as mapping
import hylife.interface as inter



# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(xi1, xi2, xi3, kind_fun, kind_map, params_map):
    
    value = 0.
    
    if   kind_fun == 1:
        
        x     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
        y     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
        z     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)

        jhx   = inter.jhx_eq(x, y, z)
        jhy   = inter.jhy_eq(x, y, z)
        jhz   = inter.jhz_eq(x, y, z)
        
        df_11 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 11)
        df_21 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 21)
        df_31 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 31)
        
        value = df_11 * jhx + df_21 * jhy + df_31 * jhz
        
    elif kind_fun == 2:
        
        x     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
        y     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
        z     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)

        jhx   = inter.jhx_eq(x, y, z)
        jhy   = inter.jhy_eq(x, y, z)
        jhz   = inter.jhz_eq(x, y, z)
        
        df_12 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 12)
        df_22 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 22)
        df_32 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 32)
        
        value = df_12 * jhx + df_22 * jhy + df_32 * jhz
        
    elif kind_fun == 3:
        
        x     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
        y     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
        z     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)

        jhx   = inter.jhx_eq(x, y, z)
        jhy   = inter.jhy_eq(x, y, z)
        jhz   = inter.jhz_eq(x, y, z)
        
        df_13 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 13)
        df_23 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 23)
        df_33 = mapping.df(xi1, xi2, xi3, kind_map, params_map, 33)
        
        value = df_13 * jhx + df_23 * jhy + df_33 * jhz
    
    elif kind_fun == 4:
        
        x     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
        y     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
        z     = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)
        
        value = inter.nh_eq_phys(x, y, z)
    
    elif kind_fun == 11:
        value = inter.b1_eq(xi1, xi2, xi3, kind_map, params_map)
    elif kind_fun == 12:
        value = inter.b2_eq(xi1, xi2, xi3, kind_map, params_map)
    elif kind_fun == 13:
        value = inter.b3_eq(xi1, xi2, xi3, kind_map, params_map)
    
    return value

     
# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','int','int','double[:]')        
def kernel_eva(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, params_map):
    
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] += fun(xi1[i1], xi2[i2], xi3[i3], kind_fun, kind_map, params_map)
                
                
                
# ==========================================================================================
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','double[:,:,:,:](order=F)','int','int','int','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_inner_3d(Nel1, Nel2, Nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, L, mat_f):
    
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

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi * mat_f[nq1*ie1 + q1, nq2*ie2 + q2, nq3*ie3 + q3]

                            L[(ie1 + il1)%Nbase1, (ie2 + il2)%Nbase2, (ie3 + il3)%Nbase3] += value
                            
                            
# ==========================================================================================
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