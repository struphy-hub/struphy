from pyccel.decorators import types
from pyccel.decorators import external_call


#============== kernel for local element mass matrix in the space V0 in 1d ====================================================
@external_call
@types('int','int','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:,:](order=F)')
def kernel_V0_1d(p1, nq1, bsi_1, bsj_1, w1, mat_g, mat_m):
    
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

                wvol = w1[q1] * mat_g[q1]
                v_m += wvol * bsi_1[il1, q1] * bsj_1[jl1, q1]

            # Update element matrices
            mat_m[il1, p1 + jl1 - il1] = v_m
#==============================================================================================================================
            
            
            
#============== kernel for local element mass matrix in the space V1 in 1d ====================================================
@external_call
@types('int','int','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:,:](order=F)')
def kernel_V1_1d(p1, nq1, bsi_1, bsj_1, w1, mat_g, mat_m):
    
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

                wvol = w1[q1] / mat_g[q1]
                v_m += wvol * bsi_1[il1, q1] * bsj_1[jl1, q1]

            # Update element matrices
            mat_m[il1, p1 + jl1 - il1] = v_m
#==============================================================================================================================



#============== kernel for local element mass matrix in the space V0 in 3d ====================================================
@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_V0(p1, p2, p3, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_g, mat_m):
    
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
                                        
                                        bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                                        bj = bsj_1[jl1, q1] * bsj_2[jl2, q2] * bsj_3[jl3, q3]
                                        
                                        v_m += wvol * bi * bj
                                        
                            # Update element matrices
                            mat_m[il1, il2, il3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] = v_m
#==============================================================================================================================
 
    
    
#============== kernel for local element mass matrix in the space V1 in 3d ====================================================
@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_V1(p1, p2, p3, n1_i, n2_i, n3_i, n1_j, n2_j, n3_j, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_Ginv, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :, :, :, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1 + 1 - n1_i):
        for il2 in range(p2 + 1 - n2_i):
            for il3 in range(p3 + 1 - n3_i):
                
                # Cycle over non-zero test functions in element
                for jl1 in range(p1 + 1 - n1_j):
                    for jl2 in range(p2 + 1 - n2_j):
                        for jl3 in range(p3 + 1 - n3_j):
                            
                            # Reset integrals over element
                            v_m = 0.
                       
                            # Cycle over quadrature points
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[q1] * w2[q2] * w3[q3] * mat_g[q1, q2, q3] * mat_Ginv[q1, q2, q3]
                                        
                                        bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                                        bj = bsj_1[jl1, q1] * bsj_2[jl2, q2] * bsj_3[jl3, q3]
                                        
                                        v_m += wvol * bi * bj
                                        
                            # Update element matrices
                            mat_m[il1, il2, il3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] = v_m
#==============================================================================================================================



#============== kernel for local element mass matrix in the space V2 in 3d ====================================================
@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_V2(p1, p2, p3, n1_i, n2_i, n3_i, n1_j, n2_j, n3_j, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_GG, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :, :, :, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1 + 1 - n1_i):
        for il2 in range(p2 + 1 - n2_i):
            for il3 in range(p3 + 1 - n3_i):
                
                # Cycle over non-zero test functions in element
                for jl1 in range(p1 + 1 - n1_j):
                    for jl2 in range(p2 + 1 - n2_j):
                        for jl3 in range(p3 + 1 - n3_j):
                            
                            # Reset integrals over element
                            v_m = 0.
                       
                            # Cycle over quadrature points
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[q1] * w2[q2] * w3[q3] * mat_GG[q1, q2, q3] / mat_g[q1, q2, q3]
                                        
                                        bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                                        bj = bsj_1[jl1, q1] * bsj_2[jl2, q2] * bsj_3[jl3, q3]
                                        
                                        v_m += wvol * bi * bj
                                        
                            # Update element matrices
                            mat_m[il1, il2, il3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] = v_m
#==============================================================================================================================



#============== kernel for local element mass matrix in the space V3 in 3d ====================================================
@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_V3(p1, p2, p3, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :, :, :, :] = 0.
    
    # Cycle over non-zero trial functions in element
    for il1 in range(p1):
        for il2 in range(p2):
            for il3 in range(p3):
                
                # Cycle over non-zero test functions in element
                for jl1 in range(p1):
                    for jl2 in range(p2):
                        for jl3 in range(p3):
                            
                            # Reset integrals over element
                            v_m = 0.
                       
                            # Cycle over quadrature points
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[q1] * w2[q2] * w3[q3] / mat_g[q1, q2, q3]
                                        
                                        bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                                        bj = bsj_1[jl1, q1] * bsj_2[jl2, q2] * bsj_3[jl3, q3]
                                        
                                        v_m += wvol * bi * bj
                                        
                            # Update element matrices
                            mat_m[il1, il2, il3, p1 + jl1 - il1, p2 + jl2 - il2, p3 + jl3 - il3] = v_m
#==============================================================================================================================
 
 
 
#============== kernel for local element vector in the space V0 in 3d ========================================================= 
@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_L0(p1, p2, p3, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_g, mat_m):
    
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

                            bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                            
                            v_m += wvol * bi * mat_f[q1, q2, q3]

                # Update element matrices
                mat_m[il1, il2, il3] = v_m
#==============================================================================================================================



#============== kernel for local element vector in the space V1 in 3d ========================================================= 
@external_call
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_L1(p1, p2, p3, n1_i, n2_i, n3_i, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_Ginv, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :] = 0.
    
    # Cycle over non-zero test functions in element
    for il1 in range(p1 + 1 - n1_i):
        for il2 in range(p2 + 1 - n2_i):
            for il3 in range(p3 + 1 - n3_i):
                            
                # Reset integrals over element
                v_m = 0.

                # Cycle over quadrature points
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            wvol = w1[q1] * w2[q2] * w3[q3] * mat_g[q1, q2, q3] * mat_Ginv[q1, q2, q3]

                            bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                            
                            v_m += wvol * bi * mat_f[q1, q2, q3]

                # Update element matrices
                mat_m[il1, il2, il3] = v_m
#==============================================================================================================================



#============== kernel for local element vector in the space V2 in 3d ========================================================= 
@external_call
@types('int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_L2(p1, p2, p3, n1_i, n2_i, n3_i, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_GG, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :] = 0.
    
    # Cycle over non-zero test functions in element
    for il1 in range(p1 + 1 - n1_i):
        for il2 in range(p2 + 1 - n2_i):
            for il3 in range(p3 + 1 - n3_i):
                            
                # Reset integrals over element
                v_m = 0.

                # Cycle over quadrature points
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            wvol = w1[q1] * w2[q2] * w3[q3]  * mat_GG[q1, q2, q3] / mat_g[q1, q2, q3]

                            bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                            
                            v_m += wvol * bi * mat_f[q1, q2, q3]

                # Update element matrices
                mat_m[il1, il2, il3] = v_m
#==============================================================================================================================



#============== kernel for local element vector in the space V3 in 3d ========================================================= 
@external_call
@types('int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_L3(p1, p2, p3, nq1, nq2, nq3, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_g, mat_m):
    
    # Reset element matrix
    mat_m[:, :, :] = 0.
    
    # Cycle over non-zero test functions in element
    for il1 in range(p1):
        for il2 in range(p2):
            for il3 in range(p3):
                            
                # Reset integrals over element
                v_m = 0.

                # Cycle over quadrature points
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):

                            wvol = w1[q1] * w2[q2] * w3[q3] / mat_g[q1, q2, q3]

                            bi = bsi_1[il1, q1] * bsi_2[il2, q2] * bsi_3[il3, q3]
                            
                            v_m += wvol * bi * mat_f[q1, q2, q3]

                # Update element matrices
                mat_m[il1, il2, il3] = v_m
#==============================================================================================================================



#============== kernel for element contribution to L2-error in the space V0 in 3d ============================================= 
@external_call
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double[:]','double[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:]')
def kernel_L2error_V0(n1, n2, n3, nbase1, nbase2, nbase3, p1, p2, p3, nq1, nq2, nq3, bs_1, bs_2, bs_3, w1, w2, w3, mat_f, mat_g, mat_c, error):
    
    # Cycle over quadrature points
    for q1 in range(nq1):
        for q2 in range(nq2):
            for q3 in range(nq3):
                
                wvol = w1[q1] * w2[q2] * w3[q3] * mat_g[q1, q2, q3]
    
                # Evaluate basis at quadrature point
                bi = 0.
                
                for il1 in range(p1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(p3 + 1):
                            
                            # Get global indices for FEM coefficients
                            i1 = (n1 + il1)%nbase1
                            i2 = (n2 + il2)%nbase2
                            i3 = (n3 + il3)%nbase3

                            bi += mat_c[i1, i2, i3] * bs_1[il1, q1] * bs_2[il2, q2] * bs_3[il3, q3]
                            
                error[:] += wvol * (bi - mat_f[q1, q2, q3])**2
#==============================================================================================================================