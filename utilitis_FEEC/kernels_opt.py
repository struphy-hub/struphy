from pyccel.decorators import types
from pyccel.decorators import external_call



@external_call
@types('int','int','double[:,:](order=F)','double[:]','double[:,:](order=F)')
def kernel_V0_1d(p1, nq1, bs1, w1, mat_m):
    
    # Reset element matrix
    mat_m[:, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1 + 1):
                
        # Cycle over non-zero test functions in element
        for jl1 in range(p1 + 1):
          
            # Reset integrals over element
            v_m = 0.

            # Cycle over quadrature points
            for q1 in range(nq1):

                bi = bs1[il1, q1] 
                bj = bs1[jl1, q1]

                v_m += w1[q1] * bi * bj

            # Update element matrix
            mat_m[il1, p1 + jl1 - il1] = v_m
            
            
            
@external_call
@types('int','int','double[:,:](order=F)','double[:]','double[:,:](order=F)')
def kernel_V0_1d_test(p1, nq1, bs1, w1, mat_m):
    
    # Reset element matrix
    mat_m[:, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1):
                
        # Cycle over non-zero test functions in element
        for jl1 in range(p1):
          
            # Reset integrals over element
            v_m = 0.

            # Cycle over quadrature points
            for q1 in range(nq1):

                bi = bs1[il1, q1] 
                bj = bs1[jl1, q1]

                v_m += w1[q1] * bi * bj

            # Update element matrix
            mat_m[il1, p1 + jl1 - il1] = v_m
            
            
            
            
@external_call
@types('int','int','int','double[:,:](order=F)','double[:]','double[:]','double[:,:](order=F)')
def kernel_V1_1d(k1, p1, nq1, bs1, t1, w1, mat_m):
    
    # Reset element matrix
    mat_m[:, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1 + 1):
                
        # Cycle over non-zero test functions in element
        for jl1 in range(p1 + 1):
            
            # Global index
            i1 = k1 + il1
            j1 = k1 + jl1
            
          
            # Reset integrals over element
            v_m = 0.

            # Cycle over quadrature points
            for q1 in range(nq1):

                bi = bs1[il1, q1] 
                bj = bs1[jl1, q1]

                v_m += w1[q1] * bi * bj
            
            
            # Scaling factors
            pi1 = p1 + 1
            pj1 = p1 + 1
            
            
            di1 = pi1/(t1[i1 + pi1] - t1[i1])
            dj1 = pj1/(t1[j1 + pj1] - t1[j1])

            # Update element matrix
            mat_m[il1, p1 + jl1 - il1] = v_m*di1*dj1








@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_V0(p1, p2, p3, nq1, nq2, nq3, bs1, bs2, bs3, w1, w2, w3, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :, :, :, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1 + 1):
        for il2 in range(p2 + 1):
            for il3 in range(p3 + 1):
                
                # Cycle over non-zero test functions in element
                for jl1 in range(p1 + 1):
                    for jl2 in range(p2 + 1):
                        for jl3 in range(p3 + 1):
                            
                            # Reset integrals over element
                            v_m = 0.
                       
                            # Cycle over quadrature points
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[q1] * w2[q2] * w3[q3] * mat_g[q1, q2, q3]
                                        
                                        bi = bs1[il1, q1] * bs2[il2, q2] * bs3[il3, q3]
                                        bj = bs1[jl1, q1] * bs2[jl2, q2] * bs3[jl3, q3]
                                        
                                        v_m += wvol * bi * bj
                                        
                            # Update element matrices
                            mat_m[il1, il2, il3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] = v_m
                            
 
 
@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_L0(p1, p2, p3, nq1, nq2, nq3, bs1, bs2, bs3, w1, w2, w3, mat_f, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :] = 0.
    
    # Cycle over non-zero test functions in element
    for il1 in range(p1 + 1):
        for il2 in range(p2 + 1):
            for il3 in range(p3 + 1):
                            
                # Reset integrals over element
                v_m = 0.

                # Cycle over quadrature points
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            wvol = w1[q1] * w2[q2] * w3[q3] * mat_g[q1, q2, q3]

                            bi = bs1[il1, q1] * bs2[il2, q2] * bs3[il3, q3]
                            
                            v_m += wvol * bi * mat_f[q1, q2, q3]

                # Update element matrices
                mat_m[il1, il2, il3] = v_m