from pyccel.decorators import types
from pyccel.decorators import pure
from pyccel.decorators import external_call


#==========================================================================================================
@pure
@types('double[:]','int','double[:]','int','double[:,:](order=F)')
def mapping_matrices(q, kind, params, output, A):
    
    A[:, :] = 0.
    
    # kind = 1 : slab geometry (params = [Lx, Ly, Lz], output = [DF, DF_inv, G, Ginv])
    if kind == 1:
    
        Lx = params[0]
        Ly = params[1]
        Lz = params[2]
        
        if output == 1:

            A[0, 0] = Lx
            A[1, 1] = Ly
            A[2, 2] = Lz
            
        elif output == 2:
            
            A[0, 0] = 1/Lx
            A[1, 1] = 1/Ly
            A[2, 2] = 1/Lz
            
        elif output == 3:
            
            A[0, 0] = Lx**2
            A[1, 1] = Ly**2
            A[2, 2] = Lz**2
            
        elif output == 4:
            
            A[0, 0] = 1/Lx**2
            A[1, 1] = 1/Ly**2
            A[2, 2] = 1/Lz**2
            
    # kind = 2 : hollow cylinder (params = [R1, R2, Lz], output = [DF, DF_inv, G, Ginv])
#==========================================================================================================


#==========================================================================================================
@pure
@types('double[:,:](order=F)','double[:]','double[:]')
def matrix_vector(A, b, c):
    
    c[:] = 0.
    
    for i in range(3):
        for j in range(3):
            c[i] += A[i, j]*b[j]      
#==========================================================================================================


#==========================================================================================================
@pure
@types('double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def matrix_matrix(A, B, C):
    
    C[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k]*B[k, j]      
#==========================================================================================================


#==========================================================================================================
@pure
@types('double[:,:](order=F)','double[:,:](order=F)')
def transpose(A, B):
    
    B[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            B[i, j] = A[j, i]
#==========================================================================================================


#==========================================================================================================
@pure
@types('double[:,:](order=F)')
def det(A):
    
    plus  = A[0, 0]*A[1, 1]*A[2, 2] + A[0, 1]*A[1, 2]*A[2, 0] + A[0, 2]*A[1, 0]*A[2, 1]
    minus = A[2, 0]*A[1, 1]*A[0, 2] + A[2, 1]*A[1, 2]*A[0, 0] + A[2, 2]*A[1, 0]*A[0, 1]
    
    return plus - minus
#==========================================================================================================



#==========================================================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    
    left [:]  = 0.
    right[:]  = 0.

    values[0] = 1.
    
    for j in range(degree):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        
        values[j + 1] = saved
#==========================================================================================================



#==========================================================================================================
@external_call
@types('double[:,:](order=F)','int[:]','int[:,:](order=F)','int[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double','double[:]','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def matrix_step3(particles, p0, spans0, Nbase, b1, b2, b3, T1, T2, T3, tt1, tt2, tt3, mapping, dt, Beq, mat11, mat12, mat13, mat22, mat23, mat33):
    
    from numpy import empty
    from numpy import zeros
    
    p0_1      = p0[0]
    p0_2      = p0[1]
    p0_3      = p0[2]
    
    p1_1      = p0_1 - 1
    p1_2      = p0_2 - 1
    p1_3      = p0_3 - 1
    
    delta1    = 1/Nbase[0]
    delta2    = 1/Nbase[1]
    delta3    = 1/Nbase[2]
    
    Nl1 = empty(p0_1,     dtype=float)
    Nr1 = empty(p0_1,     dtype=float)
    N1  = zeros(p0_1 + 1, dtype=float)
    
    Nl2 = empty(p0_2,     dtype=float)
    Nr2 = empty(p0_2,     dtype=float)
    N2  = zeros(p0_2 + 1, dtype=float)
    
    Nl3 = empty(p0_3,     dtype=float)
    Nr3 = empty(p0_3,     dtype=float)
    N3  = zeros(p0_3 + 1, dtype=float)
    
    Dl1 = empty(p1_1,     dtype=float)
    Dr1 = empty(p1_1,     dtype=float)
    D1  = zeros(p1_1 + 1, dtype=float)
    
    Dl2 = empty(p1_2,     dtype=float)
    Dr2 = empty(p1_2,     dtype=float)
    D2  = zeros(p1_2 + 1, dtype=float)
    
    Dl3 = empty(p1_3,     dtype=float)
    Dr3 = empty(p1_3,     dtype=float)
    D3  = zeros(p1_3 + 1, dtype=float)
    
    B         = zeros( 3    , dtype=float)
    
    temp_mat1 = zeros((3, 3), dytpe=float, order='F')
    temp_mat2 = zeros((3, 3), dytpe=float, order='F')
    
    B_prod    = zeros((3, 3), dtype=float, order='F')
    B_prod_T  = zeros((3, 3), dtype=float, order='F')
    
    Ginv      = zeros((3, 3), dypte=float, order='F') 
    
    q         = zeros( 3    , dtype=float)
    
    np = len(particles[:, 0])
    
    mat11[:, :, :, :, :, :] = 0.
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat22[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    mat33[:, :, :, :, :, :] = 0.
    
    for ip in range(np):
        # ... field evaluation (wave + background)
        B[0]    = Beq[0]
        B[1]    = Beq[1]
        B[2]    = Beq[2]
        
        pos1    = particles[ip, 0]
        pos2    = particles[ip, 1]
        pos3    = particles[ip, 2]
        
        span0_1 = spans0[ip, 0]
        span0_2 = spans0[ip, 1]
        span0_3 = spans0[ip, 2]
        
        span1_1 = span0_1 - 1
        span1_2 = span0_2 - 1
        span1_3 = span0_3 - 1
        
        basis_funs(T1,  p0_1, pos1, span0_1, Nl1, Nr1, N1)
        basis_funs(T2,  p0_2, pos2, span0_2, Nl2, Nr2, N2)
        basis_funs(T3,  p0_3, pos3, span0_3, Nl3, Nr3, N3)
        
        basis_funs(tt1, p1_1, pos1, span1_1, Dl1, Dr1, D1)
        basis_funs(tt2, p1_2, pos2, span1_2, Dl2, Dr2, D2)
        basis_funs(tt3, p1_3, pos3, span1_3, Dl3, Dr3, D3)
        
        D1[:] = D1/delta1
        D2[:] = D2/delta2
        D3[:] = D3/delta3
        
        
        # evaluation of 1 - component
        for il1 in range(p0_1 + 1):
            for il2 in range(p1_2 + 1):
                for il3 in range(p1_3 + 1):
                    
                    i1 = (span0_1 - il1)%Nbase[0]
                    i2 = (span1_2 - il2)%Nbase[1]
                    i3 = (span1_3 - il3)%Nbase[2]

                    B[0] += b1[i1, i2, i3] * N1[p0_1 - il1] * D2[p1_2 - il2] * D3[p1_3 - il3]
        
        
        # evaluation of 2 - component
        for il1 in range(p1_1 + 1):
            for il2 in range(p0_2 + 1):
                for il3 in range(p1_3 + 1):
                    
                    i1 = (span1_1 - il1)%Nbase[0]
                    i2 = (span0_2 - il2)%Nbase[1]
                    i3 = (span1_3 - il3)%Nbase[2]

                    B[1] += b2[i1, i2, i3] * D1[p1_1 - il1] * N2[p0_2 - il2] * D3[p1_3 - il3]
                                
        
        # evaluation of 3 - component
        for il1 in range(p1_1 + 1):
            for il2 in range(p1_2 + 1):
                for il3 in range(p0_3 + 1):
                    
                    i1 = (span1_1 - il1)%Nbase[0]
                    i2 = (span1_2 - il2)%Nbase[1]
                    i3 = (span0_3 - il3)%Nbase[2]

                    B[2] += b3[i1, i2, i3] * D1[p1_1 - il1] * D2[p1_2 - il2] * N3[p0_3 - il3]
                    
                    
        B_prod[0, 1] = -B[2]
        B_prod[0, 2] =  B[1]

        B_prod[1, 0] =  B[2]
        B_prod[1, 2] = -B[0]

        B_prod[2, 0] = -B[1]
        B_prod[2, 1] =  B[0]
        
        q[:] = particles[ip, 0:3]
        w    = particles[ip,   6]
        
        mapping_matrices(q, 1, mapping, 4, Ginv)
        matrix_matrix(Ginv, B_prod, temp_mat1)
        matrix_matrix(temp_mat1, Ginv, temp_mat2)
        transpose(B_prod, B_prod_T)
        matrix_matrix(temp_mat2, B_prod_T, temp_mat1)
        matrix_matrix(temp_mat1, Ginv, temp_mat2)
        
        temp11 = w*temp_mat2[0, 0]
        temp12 = w*temp_mat2[0, 1]
        temp13 = w*temp_mat2[0, 2]
        temp22 = w*temp_mat2[1, 1]
        temp23 = w*temp_mat2[1, 2]
        temp33 = w*temp_mat2[2, 2]
        
        
        
        # add contribution to 11 component (DNN DNN)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    for jl1 in range(p1_1 + 1):
                        j1 = (span1_1 - jl1)%Nbase[0]
                        for jl2 in range(p0_2 + 1):
                            j2 = (span0_2 - jl2)%Nbase[1]
                            for jl3 in range(p0_3 + 1):
                                j3 = (span0_3 - jl3)%Nbase[2]

                                mat11[i1, i2, i3, j1, j2, j3] += temp11 * D1[p1_1 - il1] * N2[p0_2 - il2] * N3[p0_3 - il3] * D1[p1_1 - jl1] * N2[p0_2 - jl2] * N3[p0_3 - jl3]
        
        
        # add contribution to 12 component (DNN NDN)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    for jl1 in range(p0_1 + 1):
                        j1 = (span0_1 - jl1)%Nbase[0]
                        for jl2 in range(p1_2 + 1):
                            j2 = (span1_2 - jl2)%Nbase[1]
                            for jl3 in range(p0_3 + 1):
                                j3 = (span0_3 - jl3)%Nbase[2]

                                mat12[i1, i2, i3, j1, j2, j3] += temp12 * D1[p1_1 - il1] * N2[p0_2 - il2] * N3[p0_3 - il3] * N1[p0_1 - jl1] * D2[p1_2 - jl2] * N3[p0_3 - jl3]
                                
                                                
        # add contribution to 13 component (DNN NND)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    for jl1 in range(p0_1 + 1):
                        j1 = (span0_1 - jl1)%Nbase[0]
                        for jl2 in range(p0_2 + 1):
                            j2 = (span0_2 - jl2)%Nbase[1]
                            for jl3 in range(p1_3 + 1):
                                j3 = (span1_3 - jl3)%Nbase[2]

                                mat13[i1, i2, i3, j1, j2, j3] += temp13 * D1[p1_1 - il1] * N2[p0_2 - il2] * N3[p0_3 - il3] * N1[p0_1 - jl1] * N2[p0_2 - jl2] * D3[p1_3 - jl3]
                                
                                
        # add contribution to 22 component (NDN NDN)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    for jl1 in range(p0_1 + 1):
                        j1 = (span0_1 - jl1)%Nbase[0]
                        for jl2 in range(p1_2 + 1):
                            j2 = (span1_2 - jl2)%Nbase[1]
                            for jl3 in range(p0_3 + 1):
                                j3 = (span0_3 - jl3)%Nbase[2]

                                mat22[i1, i2, i3, j1, j2, j3] += temp22 * N1[p0_1 - il1] * D2[p1_2 - il2] * N3[p0_3 - il3] * N1[p0_1 - jl1] * D2[p1_2 - jl2] * N3[p1_3 - jl3]
                                
                                
        # add contribution to 23 component (NDN NND)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    for jl1 in range(p0_1 + 1):
                        j1 = (span0_1 - jl1)%Nbase[0]
                        for jl2 in range(p0_2 + 1):
                            j2 = (span0_2 - jl2)%Nbase[1]
                            for jl3 in range(p1_3 + 1):
                                j3 = (span1_3 - jl3)%Nbase[2]

                                mat23[i1, i2, i3, j1, j2, j3] += temp23 * N1[p0_1 - il1] * D2[p1_2 - il2] * N3[p0_3 - il3] * N1[p0_1 - jl1] * N2[p0_2 - jl2] * D3[p1_3 - jl3]
                                
    
    ierr = 0
#==========================================================================================================