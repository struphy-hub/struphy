# import pyccel decorators
from pyccel.decorators import types

# import modules for mapping related quantities
import hylife.geometry.mappings_analytical as mapping

# import input files for simulation setup
import input_run.equilibrium_MHD as eq_mhd

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]')
def pusher_step3(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, u1, u2, u3, kind_map, params_map):
    
    from numpy import empty, zeros
    
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    l1  = empty( pn1              , dtype=float)
    l2  = empty( pn2              , dtype=float)
    l3  = empty( pn3              , dtype=float)
    
    r1  = empty( pn1              , dtype=float)
    r2  = empty( pn2              , dtype=float)
    r3  = empty( pn3              , dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    # magnetic field and velocity field at particle positions
    b          = empty( 3    , dtype=float)
    u          = empty( 3    , dtype=float)
    
    b_prod     = zeros((3, 3), dtype=float)
    
    # mapping related quantities
    dfinv      = empty((3, 3), dtype=float)
    dfinv_t    = empty((3, 3), dtype=float)
    ginv       = empty((3, 3), dtype=float)
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    
    temp_vec   = empty( 3    , dtype=float)
    
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
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, u, b, i, j, dfinv, dfinv_t, ginv, temp_mat1, temp_mat2, temp_vec) firstprivate(b_prod)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        u[0] = eva.evaluation_kernel(pd1, pn2, pn3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pn3], span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
        u[1] = eva.evaluation_kernel(pn1, pd2, pn3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pn3], span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
        u[2] = eva.evaluation_kernel(pn1, pn2, pd3, b1[pn1], b2[pn2], b3[pd3, :pn3]*d3[:], span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b1_eq(eta1, eta2, eta3, kind_map, params_map)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq(eta1, eta2, eta3, kind_map, params_map)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b3_eq(eta1, eta2, eta3, kind_map, params_map)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================
        
        
        # ======== particle pushing ================
        
        # evaluate inverse Jacobian matrix
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, components[i, j])  
        
        # transpose of inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        
        # evaluate inverse metric tensor
        for i in range(3):
            for j in range(3):
                ginv[i, j] = mapping.g_inv(eta1, eta2, eta3, kind_map, params_map, components[i, j]) 
                
        # perform matrix-matrix and matrix-vector products
        linalg.matrix_matrix(dfinv_t, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        linalg.matrix_vector(temp_mat2, u, temp_vec)
        
        # update particle velocities
        particles[3, ip] += dt*temp_vec[0]
        particles[4, ip] += dt*temp_vec[1]
        particles[5, ip] += dt*temp_vec[2]
        
        # ==========================================
        
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================================
@types('double[:,:]','double','int','int','double[:]')
def pusher_step4(particles, dt, np, kind_map, params_map):
    
    from numpy import empty
    
    # particle position and velocity
    eta        = empty( 3    , dtype=float)
    v          = empty( 3    , dtype=float)
    
    # mapping related quantities
    dfinv      = empty((3, 3), dtype=float)
    
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
    
    # intermediate stps in 4th order Runge-Kutta
    k1 = empty( 3, dtype=float)  
    k2 = empty( 3, dtype=float)  
    k3 = empty( 3, dtype=float)  
    k4 = empty( 3, dtype=float)  
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, dfinv, k1, k2, k3, k4)
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
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
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]')
def pusher_step5(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, kind_map, params_map):
    
    from numpy import empty, zeros
    
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    l1  = empty( pn1              , dtype=float)
    l2  = empty( pn2              , dtype=float)
    l3  = empty( pn3              , dtype=float)
    
    r1  = empty( pn1              , dtype=float)
    r2  = empty( pn2              , dtype=float)
    r3  = empty( pn3              , dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    # magnetic field at particle positions
    b          = empty( 3    , dtype=float)
    b_prod     = zeros((3, 3), dtype=float)
    
    # particle velocity
    v          = empty( 3    , dtype=float)
    
    # mapping related quantities
    dfinv      = empty((3, 3), dtype=float) 
    dfinv_t    = empty((3, 3), dtype=float)
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    
    rhs        = empty( 3    , dtype=float)
    
    identity   = zeros((3, 3), dtype=float)
    
    lhs        = empty((3, 3), dtype=float)
    
    lhs1       = empty((3, 3), dtype=float)
    lhs2       = empty((3, 3), dtype=float)
    lhs3       = empty((3, 3), dtype=float)
    
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
    
    identity[0, 0]   = 1.
    identity[1, 1]   = 1.
    identity[2, 2]   = 1.
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, v, i, j, dfinv, dfinv_t, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b1_eq(eta1, eta2, eta3, kind_map, params_map)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq(eta1, eta2, eta3, kind_map, params_map)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b3_eq(eta1, eta2, eta3, kind_map, params_map)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================
        
        
        # ======== particle pushing ================
        v[:] = particles[3:6, ip]
        
        # evaluate inverse Jacobian matrix
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, components[i, j])
        
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
        det_lhs    = linalg.det(lhs)
        
        lhs1[:, 0] = rhs
        lhs1[:, 1] = lhs[:, 1]
        lhs1[:, 2] = lhs[:, 2]
        
        lhs2[:, 0] = lhs[:, 0]
        lhs2[:, 1] = rhs
        lhs2[:, 2] = lhs[:, 2]
        
        lhs3[:, 0] = lhs[:, 0]
        lhs3[:, 1] = lhs[:, 1]
        lhs3[:, 2] = rhs
        
        det_lhs1   = linalg.det(lhs1)
        det_lhs2   = linalg.det(lhs2)
        det_lhs3   = linalg.det(lhs3)
        
        # update particle velocities
        particles[3, ip] = det_lhs1/det_lhs
        particles[4, ip] = det_lhs2/det_lhs
        particles[5, ip] = det_lhs3/det_lhs
        # ==========================================
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0