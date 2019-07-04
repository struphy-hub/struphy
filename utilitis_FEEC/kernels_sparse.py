from pyccel.decorators import types
from pyccel.decorators import external_call, pure



@external_call
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:]','double[:]','double[:]')
def kernel0(ne1, ne2, ne3, n1, n2, n3, p1, p2, p3, b1, b2, b3, w1, w2, w3, mat_g, mat_m, i_loc, j_loc):
    
    mat_m[:] = 0.
    i_loc[:] = 0.
    j_loc[:] = 0.
    
    counter = 0
    
    for il1 in range(p1 + 1):
        for il2 in range(p2 + 1):
            for il3 in range(p3 + 1):
                
                for jl1 in range(p1 + 1):
                    for jl2 in range(p2 + 1):
                        for jl3 in range(p3 + 1):
                            
                            i1 = (n1 + il1)%ne1
                            i2 = (n2 + il2)%ne2
                            i3 = (n3 + il3)%ne3
                            
                            j1 = (n1 + jl1)%ne1
                            j2 = (n2 + jl2)%ne2
                            j3 = (n3 + jl3)%ne3
                            
                            i_loc[counter] = ne2*ne3*i1 + ne3*i2 + i3
                            j_loc[counter] = ne2*ne3*j1 + ne3*j2 + j3
                            
                            value = 0.
                       
                            for g1 in range(p1 + 1):
                                for g2 in range(p2 + 1):
                                    for g3 in range(p3 + 1):

                                        wvol = w1[g1] * w2[g2] * w3[g3] * mat_g[g1, g2, g3]
                                        bi = b1[il1, g1] * b2[il2, g2] * b3[il3, g3]
                                        bj = b1[jl1, g1] * b2[jl2, g2] * b3[jl3, g3]
                                        value += wvol * bi * bj
                                        
                            mat_m[counter] = value
                            
                            counter += 1