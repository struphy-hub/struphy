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
@external_call
@types('double[:,:](order=F)','int[:]','int[:,:](order=F)','int[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double','double[:]','double[:]')
def pusher_step3(particles, p0, spans0, Nbase, b1, b2, b3, u1, u2, u3, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, mapping, dt, Beq, Ueq):
    
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
    
    B         = empty( 3    , dtype=float)
    U         = empty( 3    , dtype=float)
    
    temp_mat1 = empty((3, 3), dytpe=float, order='F')
    temp_mat2 = empty((3, 3), dytpe=float, order='F')
    
    B_prod    = zeros((3, 3), dtype=float, order='F')
    
    Ginv      = empty((3, 3), dypte=float, order='F')
    
    DFinv     = empty((3, 3), dypte=float, order='F')
    DFinv_T   = empty((3, 3), dypte=float, order='F')
    
    temp_vec  = empty( 3    , dtype=float)
    
    v         = empty( 3    , dtype=float)
    q         = empty( 3    , dtype=float)
    
    np        = len(particles[:, 0])
    
    for ip in range(np):
        
        B[0]    = Beq[0]
        B[1]    = Beq[1]
        B[2]    = Beq[2]
        
        U[0]    = Ueq[0]
        U[1]    = Ueq[1]
        U[2]    = Ueq[2]
        
        pos1    = particles[ip, 0]
        pos2    = particles[ip, 1]
        pos3    = particles[ip, 2]
        
        span0_1 = spans0[ip, 0]
        span0_2 = spans0[ip, 1]
        span0_3 = spans0[ip, 2]
        
        span1_1 = span0_1 - 1
        span1_2 = span0_2 - 1
        span1_3 = span0_3 - 1
        
        # evaluation of B1 - component (NDD)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p1_3 + 1):
                    i3 = (span1_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p0_1 + 1):
                        for jl2 in range(p1_2 + 1):
                            for jl3 in range(p1_3 + 1):

                                N1 = pp0_1[p0_1 - il1, jl1]*((pos1 - (span0_1 - p0_1)*delta1))**jl1
                                D2 = pp1_2[p1_2 - il2, jl2]*((pos2 - (span1_2 - p1_2)*delta2))**jl2
                                D3 = pp1_3[p1_3 - il3, jl3]*((pos3 - (span1_3 - p1_3)*delta3))**jl3

                                B[0] += b1[i1, i2, i3] * N1 * D2 * D3
        
        
        # evaluation of B2 - component (DND)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p1_3 + 1):
                    i3 = (span1_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p1_1 + 1):
                        for jl2 in range(p0_2 + 1):
                            for jl3 in range(p1_3 + 1):

                                D1 = pp1_1[p1_1 - il1, jl1]*((pos1 - (span1_1 - p1_1)*delta1))**jl1
                                N2 = pp0_2[p0_2 - il2, jl2]*((pos2 - (span0_2 - p0_2)*delta2))**jl2
                                D3 = pp1_3[p1_3 - il3, jl3]*((pos3 - (span1_3 - p1_3)*delta3))**jl3

                                B[1] += b2[i1, i2, i3] * D1 * N2 * D3
                                
        
        # evaluation of B3 - component (DDN)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p1_1 + 1):
                        for jl2 in range(p1_2 + 1):
                            for jl3 in range(p0_3 + 1):

                                D1 = pp1_1[p1_1 - il1, jl1]*((pos1 - (span1_1 - p1_1)*delta1))**jl1
                                D2 = pp1_2[p1_2 - il2, jl2]*((pos2 - (span1_2 - p1_2)*delta2))**jl2
                                N3 = pp0_3[p0_3 - il3, jl3]*((pos3 - (span0_3 - p0_3)*delta3))**jl3

                                B[2] += b3[i1, i2, i3] * D1 * D2 * N3
        
        
        # evaluation of U1 - component (DNN)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p1_1 + 1):
                        for jl2 in range(p0_2 + 1):
                            for jl3 in range(p0_3 + 1):

                                D1 = pp1_1[p1_1 - il1, jl1]*((pos1 - (span1_1 - p1_1)*delta1))**jl1
                                N2 = pp0_2[p0_2 - il2, jl2]*((pos2 - (span0_2 - p0_2)*delta2))**jl2
                                N3 = pp0_3[p0_3 - il3, jl3]*((pos3 - (span0_3 - p0_3)*delta3))**jl3

                                U[0] += u1[i1, i2, i3] * D1 * N2 * N3
                                
                                
        # evaluation of U2 - component (NDN)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p0_1 + 1):
                        for jl2 in range(p1_2 + 1):
                            for jl3 in range(p0_3 + 1):

                                N1 = pp0_1[p0_1 - il1, jl1]*((pos1 - (span0_1 - p0_1)*delta1))**jl1
                                D2 = pp1_2[p1_2 - il2, jl2]*((pos2 - (span1_2 - p1_2)*delta2))**jl2
                                N3 = pp0_3[p0_3 - il3, jl3]*((pos3 - (span0_3 - p0_3)*delta3))**jl3

                                U[1] += u2[i1, i2, i3] * N1 * D2 * N3
                                
                                
        # evaluation of U3 - component (NND)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p1_3 + 1):
                    i3 = (span1_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p0_1 + 1):
                        for jl2 in range(p0_2 + 1):
                            for jl3 in range(p1_3 + 1):

                                N1 = pp0_1[p0_1 - il1, jl1]*((pos1 - (span0_1 - p0_1)*delta1))**jl1
                                N2 = pp0_2[p0_2 - il2, jl2]*((pos2 - (span0_2 - p0_2)*delta2))**jl2
                                D3 = pp1_3[p1_3 - il3, jl3]*((pos3 - (span1_3 - p1_3)*delta3))**jl3

                                U[2] += u3[i1, i2, i3] * N1 * N2 * D3
        
        
        B_prod[0, 1] = -B[2]
        B_prod[0, 2] =  B[1]

        B_prod[1, 0] =  B[2]
        B_prod[1, 2] = -B[0]

        B_prod[2, 0] = -B[1]
        B_prod[2, 1] =  B[0]
        
        v[:] = particles[ip, 3:6]
        q[:] = particles[ip, 0:3]
        
        mapping_matrices(q, 1, mapping, 2, DFinv)
        transpose(DFinv, DFinv_T)
        mapping_matrices(q, 1, mapping, 4, Ginv)
        matrix_matrix(DFinv_T, B_prod, temp_mat1)
        matrix_matrix(temp_mat1, Ginv, temp_mat2)
        matrix_vector(temp_mat2, U, temp_vec)
        
        particles[ip, 3] += dt/2*temp_vec[0]
        particles[ip, 4] += dt/2*temp_vec[1]
        particles[ip, 5] += dt/2*temp_vec[2]
        
    ierr = 0
#==========================================================================================================



#==========================================================================================================
@external_call
@types('double[:,:](order=F)','double[:]','double')
def pusher_step4(particles, mapping, dt):
    
    from numpy import empty
    
    DFinv = empty((3, 3), dtype=float, order='F')
    v     = empty( 3    , dtype=float)
    q     = empty( 3    , dtype=float)
    temp  = empty( 3    , dtype=float)
    
    np    = len(particles[:, 0])
    
    for ip in range(np):
        
        v[:] = particles[ip, 3:6]
        q[:] = particles[ip, 0:3]
        
        mapping_matrices(q, 1, mapping, 2, DFinv)
        matrix_vector(DFinv, v, temp)
        
        particles[ip, 0] = (q[0] + dt*temp[0])%mapping[0]
        particles[ip, 1] = (q[1] + dt*temp[1])%mapping[1]
        particles[ip, 2] = (q[2] + dt*temp[2])%mapping[2]
        
    ierr = 0
#==========================================================================================================



#==========================================================================================================
@external_call
@types('double[:,:](order=F)','int[:]','int[:,:](order=F)','int[:]','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:]','double','double[:]')
def pusher_step5(particles, p0, spans0, Nbase, b1, b2, b3, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, mapping, dt, Beq):
    
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
    
    B         = empty( 3    , dtype=float)
    
    temp_mat1 = empty((3, 3), dytpe=float, order='F')
    temp_mat2 = empty((3, 3), dytpe=float, order='F')
    
    rhs       = empty( 3    , dtype=float)
    
    B_prod    = zeros((3, 3), dtype=float, order='F')
    
    DFinv     = empty((3, 3), dypte=float, order='F') 
    DFinv_T   = empty((3, 3), dypte=float, order='F')
    
    I         = zeros((3, 3), dtype=float, order='F')
    I[0, 0]   = 1.
    I[1, 1]   = 1.
    I[2, 2]   = 1.
    
    lhs       = empty((3, 3), dtype=float, order='F')
    
    lhs1      = empty((3, 3), dtype=float, order='F')
    lhs2      = empty((3, 3), dtype=float, order='F')
    lhs3      = empty((3, 3), dtype=float, order='F')
    
    v         = empty( 3    , dtype=float)
    q         = empty( 3    , dtype=float)
    
    np        = len(particles[:, 0])
    
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
        
        # evaluation of B1 - component (NDD)
        for il1 in range(p0_1 + 1):
            i1 = (span0_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p1_3 + 1):
                    i3 = (span1_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p0_1 + 1):
                        for jl2 in range(p1_2 + 1):
                            for jl3 in range(p1_3 + 1):

                                N1 = pp0_1[p0_1 - il1, jl1]*((pos1 - (span0_1 - p0_1)*delta1))**jl1
                                D2 = pp1_2[p1_2 - il2, jl2]*((pos2 - (span1_2 - p1_2)*delta2))**jl2
                                D3 = pp1_3[p1_3 - il3, jl3]*((pos3 - (span1_3 - p1_3)*delta3))**jl3

                                B[0] += b1[i1, i2, i3] * N1 * D2 * D3
        
        
        # evaluation of B2 - component (DND)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p0_2 + 1):
                i2 = (span0_2 - il2)%Nbase[1]
                for il3 in range(p1_3 + 1):
                    i3 = (span1_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p1_1 + 1):
                        for jl2 in range(p0_2 + 1):
                            for jl3 in range(p1_3 + 1):

                                D1 = pp1_1[p1_1 - il1, jl1]*((pos1 - (span1_1 - p1_1)*delta1))**jl1
                                N2 = pp0_2[p0_2 - il2, jl2]*((pos2 - (span0_2 - p0_2)*delta2))**jl2
                                D3 = pp1_3[p1_3 - il3, jl3]*((pos3 - (span1_3 - p1_3)*delta3))**jl3

                                B[1] += b2[i1, i2, i3] * D1 * N2 * D3
                                
        
        # evaluation of B3 - component (DDN)
        for il1 in range(p1_1 + 1):
            i1 = (span1_1 - il1)%Nbase[0]
            for il2 in range(p1_2 + 1):
                i2 = (span1_2 - il2)%Nbase[1]
                for il3 in range(p0_3 + 1):
                    i3 = (span0_3 - il3)%Nbase[2]
                    
                    for jl1 in range(p1_1 + 1):
                        for jl2 in range(p1_2 + 1):
                            for jl3 in range(p0_3 + 1):

                                D1 = pp1_1[p1_1 - il1, jl1]*((pos1 - (span1_1 - p1_1)*delta1))**jl1
                                D2 = pp1_2[p1_2 - il2, jl2]*((pos2 - (span1_2 - p1_2)*delta2))**jl2
                                N3 = pp0_3[p0_3 - il3, jl3]*((pos3 - (span0_3 - p0_3)*delta3))**jl3

                                B[2] += b3[i1, i2, i3] * D1 * D2 * N3
        
        
        B_prod[0, 1] = -B[2]
        B_prod[0, 2] =  B[1]

        B_prod[1, 0] =  B[2]
        B_prod[1, 2] = -B[0]

        B_prod[2, 0] = -B[1]
        B_prod[2, 1] =  B[0]
        
        
        v[:] = particles[ip, 3:6]
        q[:] = particles[ip, 0:3]
        
        mapping_matrices(q, 1, mapping, 2, DFinv)
        matrix_matrix(B_prod, DFinv, temp_mat1)
        transpose(DFinv, DFinv_T)
        matrix_matrix(DFinv_T, temp_mat1, temp_mat2)
        matrix_vector(I - dt/2*temp_mat2, v, rhs)
        
        lhs[:, :] = I + dt/2*temp_mat2
        
        det_lhs = det(lhs)
        
        lhs1[:, 0] = rhs
        lhs1[:, 1] = lhs[:, 1]
        lhs1[:, 2] = lhs[:, 2]
        
        lhs2[:, 0] = lhs[:, 0]
        lhs2[:, 1] = rhs
        lhs2[:, 2] = lhs[:, 2]
        
        lhs3[:, 0] = lhs[:, 0]
        lhs3[:, 1] = lhs[:, 1]
        lhs3[:, 2] = rhs
        
        det_lhs1 = det(lhs1)
        det_lhs2 = det(lhs2)
        det_lhs3 = det(lhs3)
        
        particles[ip, 3] = det_lhs1/det_lhs
        particles[ip, 4] = det_lhs2/det_lhs
        particles[ip, 5] = det_lhs3/det_lhs
        # ...
        
        
    ierr = 0
#==========================================================================================================