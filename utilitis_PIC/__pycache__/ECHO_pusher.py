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
@types('double[:,:](order=F)','double[:]','double','double[:,:](order=F)','double[:,:](order=F)')
def pusher_step3(particles, mapping, dt, B_part, U_part):
    
    from numpy import empty
    from numpy import zeros
    
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
        
        B[0] = B_part[ip, 0]
        B[1] = B_part[ip, 1]
        B[2] = B_part[ip, 2]
        
        U[0] = U_part[ip, 0]
        U[1] = U_part[ip, 1]
        U[2] = U_part[ip, 2]
        
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
@types('double[:,:](order=F)','double[:]','double','double[:,:](order=F)')
def pusher_step5(particles, mapping, dt, B_part):
    
    from numpy import empty
    from numpy import zeros
    
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
        
        B[0]    = B_part[ip, 0]
        B[1]    = B_part[ip, 1]
        B[2]    = B_part[ip, 2]
        
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