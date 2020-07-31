# import pyccel decorators
from pyccel.decorators import types

# import modules for mapping related quantities
import hylife.geometry.mappings_analytical as mapping

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg


# ==========================================================================================================
@types('double[:,:]','double','double[:,:]','double[:,:]','int','double[:]')
def pusher_step3(particles, dt, b_part, u_part, kind_map, params_map):
    
    from numpy import empty, zeros
    
    b          = empty( 3    , dtype=float)
    u          = empty( 3    , dtype=float)
    
    b_prod     = zeros((3, 3), dtype=float)
    
    eta        = empty( 3    , dtype=float)
    v          = empty( 3    , dtype=float)
    
    dfinv      = empty((3, 3), dtype=float)
    dfinv_t    = empty((3, 3), dtype=float)
    ginv       = empty((3, 3), dtype=float)
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    
    temp_vec   = empty( 3    , dtype=float)
    
    np         = len(particles[:, 0])
    
    components = empty((3, 3), dtype=int)
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    #$ omp parallel
    #$ omp do private (ip, b, u, eta, v, i, j, dfinv, dfinv_t, ginv, temp_mat1, temp_mat2, temp_vec) firstprivate(b_prod)
    for ip in range(np):
        
        b[0] = b_part[ip, 0]
        b[1] = b_part[ip, 1]
        b[2] = b_part[ip, 2]
        
        u[0] = u_part[ip, 0]
        u[1] = u_part[ip, 1]
        u[2] = u_part[ip, 2]
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        
        eta[:] = particles[ip, 0:3]
        v[:]   = particles[ip, 3:6]
        
        # evaluate inverse Jacobian matrix
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0], eta[1], eta[2], kind_map, params_map, components[i, j])  
        
        # transpose of inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        
        # evaluate inverse metric tensor
        for i in range(3):
            for j in range(3):
                ginv[i, j] = mapping.g_inv(eta[0], eta[1], eta[2], kind_map, params_map, components[i, j]) 
                
        # perform matrix-matrix and matrix-vector products
        linalg.matrix_matrix(dfinv_t, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        linalg.matrix_vector(temp_mat2, u, temp_vec)
        
        # update particle velocities
        particles[ip, 3] += dt*temp_vec[0]
        particles[ip, 4] += dt*temp_vec[1]
        particles[ip, 5] += dt*temp_vec[2]
        
    #$ omp end do
    #$ omp end parallel
    
    
# ==========================================================================================================
@types('double[:,:]','double','int','double[:]')
def pusher_step4(particles, dt, kind_map, params_map):
    
    from numpy import empty
    
    eta        = empty( 3    , dtype=float)
    v          = empty( 3    , dtype=float)
    
    dfinv      = empty((3, 3), dtype=float)
    
    np         = len(particles[:, 0])
    
    components = empty((3, 3), dtype=int)
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    k1 = empty( 3, dtype=float)  
    k2 = empty( 3, dtype=float)  
    k3 = empty( 3, dtype=float)  
    k4 = empty( 3, dtype=float)  
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, dfinv, k1, k2, k3, k4)
    for ip in range(np):
        
        eta[:] = particles[ip, 0:3]
        v[:]   = particles[ip, 3:6]
        
        # step 1 in Runge-Kutta method
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0], eta[1], eta[2], kind_map, params_map, components[i, j])
                
        linalg.matrix_vector(dfinv, v, k1)
        
        # step 2 in Runge-Kutta method
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0] + dt*k1[0]/2, eta[1] + dt*k1[1]/2, eta[2] + dt*k1[2]/2, kind_map, params_map, components[i, j])
                
        linalg.matrix_vector(dfinv, v, k2)
        
        # step 3 in Runge-Kutta method
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0] + dt*k2[0]/2, eta[1] + dt*k2[1]/2, eta[2] + dt*k2[2]/2, kind_map, params_map, components[i, j])
                
        linalg.matrix_vector(dfinv, v, k3)
        
        # step 4 in Runge-Kutta method
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0] + dt*k3[0], eta[1] + dt*k3[1], eta[2] + dt*k3[2], kind_map, params_map, components[i, j])
                
        linalg.matrix_vector(dfinv, v, k4)
        
        # update logical coordinates
        particles[ip, 0] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.
        particles[ip, 1] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.
        particles[ip, 2] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.
    
    #$ omp end do
    #$ omp end parallel 
    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:,:]','int','double[:]')
def pusher_step5(particles, dt, b_part, kind_map, params_map):
    
    from numpy import empty, zeros
    
    b          = empty( 3    , dtype=float)
    
    b_prod     = zeros((3, 3), dtype=float)
    
    v          = empty( 3    , dtype=float)
    eta        = empty( 3    , dtype=float)
    
    dfinv      = empty((3, 3), dtype=float) 
    dfinv_t    = empty((3, 3), dtype=float)
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    
    rhs        = empty( 3    , dtype=float)
    
    identity   = zeros((3, 3), dtype=float)
    identity[0, 0] = 1.
    identity[1, 1] = 1.
    identity[2, 2] = 1.
    
    lhs        = empty((3, 3), dtype=float)
    
    lhs1       = empty((3, 3), dtype=float)
    lhs2       = empty((3, 3), dtype=float)
    lhs3       = empty((3, 3), dtype=float)
    
    np         = len(particles[:, 0])
    
    components = empty((3, 3), dtype=int)
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    #$ omp parallel
    #$ omp do private (ip, b, eta, v, dfinv, dfinv_t, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
    for ip in range(np):
        
        b[0] = b_part[ip, 0]
        b[1] = b_part[ip, 1]
        b[2] = b_part[ip, 2]
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]
        
        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]
        
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        
        eta[:] = particles[ip, 0:3]
        v[:]   = particles[ip, 3:6]
        
        # evaluate inverse Jacobian matrix
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta[0], eta[1], eta[2], kind_map, params_map, components[i, j])
        
        # transpose of inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        
        # perform matrix-matrix and matrix-vector multiplications
        linalg.matrix_matrix(b_prod, dfinv, temp_mat1)
        linalg.matrix_matrix(dfinv_t, temp_mat1, temp_mat2)
        
        # explicit part of update rule
        linalg.matrix_vector(identity - dt/2*temp_mat2, v, rhs)
        
        # implicit part of update rule
        lhs = identity + dt/2*temp_mat2
        
        # solve 3 x 3 system with Cramer's rule
        det_lhs = linalg.det(lhs)
        
        lhs1[:, 0] = rhs
        lhs1[:, 1] = lhs[:, 1]
        lhs1[:, 2] = lhs[:, 2]
        
        lhs2[:, 0] = lhs[:, 0]
        lhs2[:, 1] = rhs
        lhs2[:, 2] = lhs[:, 2]
        
        lhs3[:, 0] = lhs[:, 0]
        lhs3[:, 1] = lhs[:, 1]
        lhs3[:, 2] = rhs
        
        det_lhs1 = linalg.det(lhs1)
        det_lhs2 = linalg.det(lhs2)
        det_lhs3 = linalg.det(lhs3)
        
        particles[ip, 3] = det_lhs1/det_lhs
        particles[ip, 4] = det_lhs2/det_lhs
        particles[ip, 5] = det_lhs3/det_lhs
    
    #$ omp end do
    #$ omp end parallel     