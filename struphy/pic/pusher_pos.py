# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for B-spline evaluation
from struphy.feec.bsplines_kernels import basis_funs_all
from struphy.feec.basics.spline_evaluation_3d import evaluation_kernel_3d

# import modules for mapping evaluation
import struphy.geometry.mappings_3d as mapping



# ==========================================================================================================
def pusher_rk4(particles : 'float[:,:]', dt : float, kind_map : int, params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', ind1f : 'int[:,:]', ind2f : 'int[:,:]', ind3f : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, shape
    from numpy import sqrt, arctan2, pi, cos, sin
    
    np = shape(particles)[1]
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)
    
    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    fx    = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v   = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)  
    k2 = empty(3, dtype=float)  
    k3 = empty(3, dtype=float)  
    k4 = empty(3, dtype=float) 
    # ========================================================
    
    
    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, k1, k2, k3, k4)
    #$ omp for 
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # ----------- step 1 in Runge-Kutta method -----------------------
        pos1 = eta[0]
        pos2 = eta[1]
        pos3 = eta[2]
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, fx, df, 1)
        
        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1)
        # ------------------------------------------------------------------
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1 = (eta[0] + dt*k1[0]/2)%1.
        pos2 = (eta[1] + dt*k1[1]/2)%1.
        pos3 = (eta[2] + dt*k1[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, fx, df, 1)
        
        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1 = (eta[0] + dt*k2[0]/2)%1.
        pos2 = (eta[1] + dt*k2[1]/2)%1.
        pos3 = (eta[2] + dt*k2[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, fx, df, 1)
        
        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1 = (eta[0] + dt*k3[0])%1.
        pos2 = (eta[1] + dt*k3[1])%1.
        pos3 = (eta[2] + dt*k3[2])%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, fx, df, 1)
        
        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------
        
        
        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0
        # ------------------------------------------------------------------
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ========================================================================================================    
def reflect(df : 'float[:,:]', df_inv : 'float[:,:]', v : 'float[:]'):
    
    from numpy import empty, sqrt
    
    vg = empty(3, dtype=float)
    
    basis     = empty((3, 3), dtype=float)
    basis_inv = empty((3, 3), dtype=float)
    
    # calculate normalized basis vectors
    norm1 = sqrt(df_inv[0, 0]**2 + df_inv[0, 1]**2 + df_inv[0, 2]**2)
    
    norm2 = sqrt(df[0, 1]**2 + df[1, 1]**2 + df[2, 1]**2)
    norm3 = sqrt(df[0, 2]**2 + df[1, 2]**2 + df[2, 2]**2)
    
    basis[:, 0] = df_inv[0, :]/norm1
    
    basis[:, 1] = df[:, 1]/norm2
    basis[:, 2] = df[:, 2]/norm3
    
    linalg.matrix_inv(basis, basis_inv)
    
    linalg.matrix_vector(basis_inv, v, vg)
    
    vg[0] = -vg[0]
    
    linalg.matrix_vector(basis, vg, v)
    
    

# ==========================================================================================================
def pusher_rk4_pseudo(particles : 'float[:,:]', dt : float, kind_map : int, params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', ind1f : 'int[:,:]', ind2f : 'int[:,:]', ind3f : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', a : float, r0 : float):
    
    from numpy import empty, zeros, shape
    
    np = shape(particles)[1]
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)
    
    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    
    df_old    = empty((3, 3), dtype=float)
    dfinv_old = empty((3, 3), dtype=float)
    
    fx = empty(3, dtype=float)
    
    # needed mapping quantities for pseudo-cartesian coordinates
    df_pseudo = empty((3, 3), dtype=float)
    
    df_pseudo_old = empty((3, 3), dtype=float)
    fx_pseudo = empty(3, dtype=float)
    
    params_pseudo = empty(3, dtype=float)
    
    params_pseudo[0] = 0.
    params_pseudo[1] = a
    params_pseudo[2] = r0
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta    = empty(3, dtype=float)
    v      = empty(3, dtype=float)
    v_temp = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)  
    k2 = empty(3, dtype=float)  
    k3 = empty(3, dtype=float)  
    k4 = empty(3, dtype=float) 
    # ========================================================
    
    
    #$ omp parallel private (ip, eta, v, fx_pseudo, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df_old, fx, dfinv_old, df_pseudo_old, df, dfinv, df_pseudo, v_temp, k1, k2, k3, k4)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        # old logical coordinates and velocities
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # compute old pseudo-cartesian coordinates
        mapping.f(eta[0], eta[1], eta[2], 14, params_pseudo, tf1, tf2, tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, fx_pseudo)
       
        # evaluate old Jacobian matrix of mapping F
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], fx, df_old, 1)

        # evaluate old inverse Jacobian matrix of mapping F
        linalg.matrix_inv(df_old, dfinv_old)
        
        # evaluate old Jacobian matrix of mapping F_pseudo
        mapping.df(eta[0], eta[1], eta[2], 14, params_pseudo, tf1, tf2, tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, df_pseudo_old)
        
        while True:
            
            # ----------- step 1 in Runge-Kutta method -----------------------
            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv_old, v, v_temp)
            linalg.matrix_vector(df_pseudo_old, v_temp, k1)
            # ------------------------------------------------------------------
            
        
            # ----------------- step 2 in Runge-Kutta method -------------------
            mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 14, params_pseudo, eta)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], fx, df, 1)
            
            # evaluate inverse Jacobian matrix of mapping F
            linalg.matrix_inv(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            mapping.df(eta[0], eta[1], eta[2], 14, params_pseudo, tf1, tf2, tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, df_pseudo)
            
            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k2)
            # ------------------------------------------------------------------


            # ------------------ step 3 in Runge-Kutta method ------------------
            mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 14, params_pseudo, eta)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                   
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], fx, df, 1)

            # evaluate inverse Jacobian matrix of mapping F
            linalg.matrix_inv(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            mapping.df(eta[0], eta[1], eta[2], 14, params_pseudo, tf1, tf2, tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, df_pseudo)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k3)
            # ------------------------------------------------------------------


            # ------------------ step 4 in Runge-Kutta method ------------------
            mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 14, params_pseudo, eta)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], fx, df, 1)

            # evaluate inverse Jacobian matrix of mapping F
            linalg.matrix_inv(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            mapping.df(eta[0], eta[1], eta[2], 14, params_pseudo, tf1, tf2, tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, df_pseudo)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k4)
            # ------------------------------------------------------------------


            #  ---------------- update pseudo-cartesian coordinates ------------
            fx_pseudo[0] = fx_pseudo[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
            fx_pseudo[1] = fx_pseudo[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
            fx_pseudo[2] = fx_pseudo[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
            # ------------------------------------------------------------------

            # compute logical coordinates
            mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 14, params_pseudo, eta)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break

            particles[0, ip] = eta[0]
            particles[1, ip] = eta[1]
            particles[2, ip] = eta[2]
            
            # set particle velocity (will only change if particle was reflected)
            particles[3, ip] = v[0]
            particles[4, ip] = v[1]
            particles[5, ip] = v[2]
            
            break
    #$ omp end parallel
    
    ierr = 0
    
    

    
# ==========================================================================================================
def pusher_exact(particles : 'float[:,:]', dt : float, kind_map : int, params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', ind1f : 'int[:,:]', ind2f : 'int[:,:]', ind3f : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', tol : float):
    
    from numpy import shape, empty
    
    np = shape(particles)[1]
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)
    
    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    
    x_old = empty(3, dtype=float)
    x_new = empty(3, dtype=float)
    
    temp  = empty(3, dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)
    # ========================================================
    
    
    #$ omp parallel private(ip, e, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, x_old, x_new, dfinv, temp)
    #$ omp for 
    for ip in range(np):
        
        e[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]
        
        span1f = int(e[0]*nelf[0]) + pf1
        span2f = int(e[1]*nelf[1]) + pf2
        span3f = int(e[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix and current Cartesian coordinates
        mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, e[0], e[1], e[2], x_old, df, 2)
        
        # update cartesian coordinates exactly
        x_new[0] = x_old[0] + dt*v[0]
        x_new[1] = x_old[1] + dt*v[1]
        x_new[2] = x_old[2] + dt*v[2]
        
        # calculate new logical coordinates by solving inverse mapping with Newton-method
        
        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)
        
        while True:

            x_old[:] = x_old - x_new
            linalg.matrix_vector(dfinv, x_old, temp)
            
            e[0] =  e[0] - temp[0]
            e[1] = (e[1] - temp[1])%1.0
            e[2] = (e[2] - temp[2])%1.0
            
            span1f = int(e[0]*nelf[0]) + pf1
            span2f = int(e[1]*nelf[1]) + pf2
            span3f = int(e[2]*nelf[2]) + pf3
            
            # evaluate Jacobian matrix and mapping
            mapping.f_df_pic(kind_map, params_map, tf1, tf2, tf3, pf, span1f, span2f, span3f, ind1f, ind2f, ind3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, e[0], e[1], e[2], x_old, df, 2)
            
            if abs(x_old[0] - x_new[0]) < tol and abs(x_old[1] - x_new[1]) < tol and abs(x_old[2] - x_new[2]) < tol:
                particles[0:3, ip] = e
                break
            
            # evaluate inverse Jacobian matrix
            linalg.matrix_inv(df, dfinv)
    #$ omp end parallel
    
    ierr = 0



## ==========================================================================================================
#def pusher_rk4_pc_full(particles : 'float[:,:]', dt : float, t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : int, u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : int, kind_map : int, params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
#    
#    from numpy import empty
#
#    #============== for velocity evaluation ============
#    # spline degrees
#    pn1 = p[0]
#    pn2 = p[1]
#    pn3 = p[2]
#    
#    pd1 = pn1 - 1
#    pd2 = pn2 - 1
#    pd3 = pn3 - 1
#    
#    # p + 1 non-vanishing basis functions up tp degree p
#    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
#    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
#    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
#    
#    # left and right values for spline evaluation
#    l1 = empty( pn1, dtype=float)
#    l2 = empty( pn2, dtype=float)
#    l3 = empty( pn3, dtype=float)
#    
#    r1 = empty( pn1, dtype=float)
#    r2 = empty( pn2, dtype=float)
#    r3 = empty( pn3, dtype=float)
#    
#    # scaling arrays for M-splines
#    d1 = empty( pn1, dtype=float)
#    d2 = empty( pn2, dtype=float)
#    d3 = empty( pn3, dtype=float)
#    
#    # p + 1 non-vanishing derivatives
#    der1 = empty(pn1 + 1, dtype=float)
#    der2 = empty(pn2 + 1, dtype=float)
#    der3 = empty(pn3 + 1, dtype=float)
#    
#    # non-vanishing N-splines at particle position
#    bn1 = empty( pn1 + 1, dtype=float)
#    bn2 = empty( pn2 + 1, dtype=float)
#    bn3 = empty( pn3 + 1, dtype=float)
#    
#    # non-vanishing D-splines at particle position
#    bd1 = empty( pd1 + 1, dtype=float)
#    bd2 = empty( pd2 + 1, dtype=float)
#    bd3 = empty( pd3 + 1, dtype=float)
#
#    # # velocity field at particle position
#    u = empty(3, dtype=float)
#    # ==========================================================
#
#
#    # ================ for mapping evaluation ==================
#    # spline degrees
#    pf1 = pf[0]
#    pf2 = pf[1]
#    pf3 = pf[2]
#    
#    # pf + 1 non-vanishing basis functions up tp degree pf
#    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
#    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
#    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
#    
#    # left and right values for spline evaluation
#    l1f = empty( pf1, dtype=float)
#    l2f = empty( pf2, dtype=float)
#    l3f = empty( pf3, dtype=float)
#    
#    r1f = empty( pf1, dtype=float)
#    r2f = empty( pf2, dtype=float)
#    r3f = empty( pf3, dtype=float)
#    
#    # scaling arrays for M-splines
#    d1f = empty( pf1, dtype=float)
#    d2f = empty( pf2, dtype=float)
#    d3f = empty( pf3, dtype=float)
#    
#    # pf + 1 derivatives
#    der1f = empty( pf1 + 1, dtype=float)
#    der2f = empty( pf2 + 1, dtype=float)
#    der3f = empty( pf3 + 1, dtype=float)
#    
#    # needed mapping quantities
#    df      = empty((3, 3), dtype=float)
#    dfinv   = empty((3, 3), dtype=float)
#    dfinv_t = empty((3, 3), dtype=float)
#    Ginv    = empty((3, 3), dtype=float)
#    fx      = empty( 3    , dtype=float)
#    # ========================================================
#    
#    
#    # ======= particle position and velocity =================
#    eta = empty(3, dtype=float)
#    v   = empty(3, dtype=float)
#    # ========================================================
#    
#    
#    # ===== intermediate stps in 4th order Runge-Kutta =======
#    k1   = empty(3, dtype=float)  
#    k2   = empty(3, dtype=float)  
#    k3   = empty(3, dtype=float)  
#    k4   = empty(3, dtype=float)
#    k1_u = empty(3, dtype=float)  
#    k2_u = empty(3, dtype=float)  
#    k3_u = empty(3, dtype=float)  
#    k4_u = empty(3, dtype=float) 
#    k1_v = empty(3, dtype=float)  
#    k2_v = empty(3, dtype=float)  
#    k3_v = empty(3, dtype=float)  
#    k4_v = empty(3, dtype=float)  
#    # ========================================================
#    
#    
#    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, Ginv, det_df, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_u, k2_u, k3_u, k4_u, k1_v, k2_v, k3_v, k4_v)
#    #$ omp for 
#    for ip in range(np):
#        
#        eta[:] = particles[0:3, ip]
#        v[:]   = particles[3:6, ip]
#
#        # ----------- step 1 in Runge-Kutta method -----------------------
#        # ========= mapping evaluation =============
#        pos1 = eta[0]
#        pos2 = eta[1]
#        pos3 = eta[2]
#
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv) ###########
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k1_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k1_u[:] = u/det_df
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k1_v)
#
#        k1[:] = k1_v[:] + k1_u[:]
#
#        # ------------------------------------------------------------------
#        
#        
#        # ----------------- step 2 in Runge-Kutta method -------------------
#        pos1   = (eta[0] + dt*k1[0]/2)%1.
#        pos2   = (eta[1] + dt*k1[1]/2)%1.
#        pos3   = (eta[2] + dt*k1[2]/2)%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k2_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k2_u[:] = u/det_df
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k2_v)
#
#        k2[:] = k2_v[:] + k2_u[:]
#        # ------------------------------------------------------------------
#        
#        
#        # ------------------ step 3 in Runge-Kutta method ------------------
#        pos1   = (eta[0] + dt*k2[0]/2)%1.
#        pos2   = (eta[1] + dt*k2[1]/2)%1.
#        pos3   = (eta[2] + dt*k2[2]/2)%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k3_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k3_u[:] = u/det_df
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k3_v)
#
#        k3[:] = k3_v[:] + k3_u[:]
#        # ------------------------------------------------------------------
#        
#        
#        # ------------------ step 4 in Runge-Kutta method ------------------
#        pos1   = (eta[0] + dt*k3[0])%1.
#        pos2   = (eta[1] + dt*k3[1])%1.
#        pos3   = (eta[2] + dt*k3[2])%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k4_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k4_u[:] = u/det_df
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k4_v)
#
#        k4[:] = k4_v[:] + k4_u[:]
#        # ------------------------------------------------------------------
#
#        #  ---------------- update logical coordinates ---------------------
#        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
#        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
#        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0
#
#        # ------------------------------------------------------------------
#    #$ omp end parallel
#    
#    ierr = 0
#
#
## ==========================================================================================================
#def pusher_rk4_pc_perp(particles : 'float[:,:]', dt : float, t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : int, u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : int, kind_map : int, params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
#    
#    from numpy import empty
#
#    #============== for velocity evaluation ============
#    # spline degrees
#    pn1 = p[0]
#    pn2 = p[1]
#    pn3 = p[2]
#    
#    pd1 = pn1 - 1
#    pd2 = pn2 - 1
#    pd3 = pn3 - 1
#    
#    # p + 1 non-vanishing basis functions up tp degree p
#    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
#    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
#    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
#    
#    # left and right values for spline evaluation
#    l1 = empty( pn1, dtype=float)
#    l2 = empty( pn2, dtype=float)
#    l3 = empty( pn3, dtype=float)
#    
#    r1 = empty( pn1, dtype=float)
#    r2 = empty( pn2, dtype=float)
#    r3 = empty( pn3, dtype=float)
#    
#    # scaling arrays for M-splines
#    d1 = empty( pn1, dtype=float)
#    d2 = empty( pn2, dtype=float)
#    d3 = empty( pn3, dtype=float)
#    
#    # p + 1 non-vanishing derivatives
#    der1 = empty(pn1 + 1, dtype=float)
#    der2 = empty(pn2 + 1, dtype=float)
#    der3 = empty(pn3 + 1, dtype=float)
#    
#    # non-vanishing N-splines at particle position
#    bn1 = empty( pn1 + 1, dtype=float)
#    bn2 = empty( pn2 + 1, dtype=float)
#    bn3 = empty( pn3 + 1, dtype=float)
#    
#    # non-vanishing D-splines at particle position
#    bd1 = empty( pd1 + 1, dtype=float)
#    bd2 = empty( pd2 + 1, dtype=float)
#    bd3 = empty( pd3 + 1, dtype=float)
#
#    # # velocity field at particle position
#    u = empty(3, dtype=float)
#    # ==========================================================
#
#
#    # ================ for mapping evaluation ==================
#    # spline degrees
#    pf1 = pf[0]
#    pf2 = pf[1]
#    pf3 = pf[2]
#    
#    # pf + 1 non-vanishing basis functions up tp degree pf
#    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
#    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
#    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
#    
#    # left and right values for spline evaluation
#    l1f = empty( pf1, dtype=float)
#    l2f = empty( pf2, dtype=float)
#    l3f = empty( pf3, dtype=float)
#    
#    r1f = empty( pf1, dtype=float)
#    r2f = empty( pf2, dtype=float)
#    r3f = empty( pf3, dtype=float)
#    
#    # scaling arrays for M-splines
#    d1f = empty( pf1, dtype=float)
#    d2f = empty( pf2, dtype=float)
#    d3f = empty( pf3, dtype=float)
#    
#    # pf + 1 derivatives
#    der1f = empty( pf1 + 1, dtype=float)
#    der2f = empty( pf2 + 1, dtype=float)
#    der3f = empty( pf3 + 1, dtype=float)
#    
#    # needed mapping quantities
#    df      = empty((3, 3), dtype=float)
#    dfinv   = empty((3, 3), dtype=float)
#    dfinv_t = empty((3, 3), dtype=float)
#    Ginv    = empty((3, 3), dtype=float)
#    fx      = empty( 3    , dtype=float)
#    # ========================================================
#    
#    
#    # ======= particle position and velocity =================
#    eta = empty(3, dtype=float)
#    v   = empty(3, dtype=float)
#    # ========================================================
#    
#    
#    # ===== intermediate stps in 4th order Runge-Kutta =======
#    k1   = empty(3, dtype=float)  
#    k2   = empty(3, dtype=float)  
#    k3   = empty(3, dtype=float)  
#    k4   = empty(3, dtype=float)
#    k1_u = empty(3, dtype=float)  
#    k2_u = empty(3, dtype=float)  
#    k3_u = empty(3, dtype=float)  
#    k4_u = empty(3, dtype=float) 
#    k1_v = empty(3, dtype=float)  
#    k2_v = empty(3, dtype=float)  
#    k3_v = empty(3, dtype=float)  
#    k4_v = empty(3, dtype=float)  
#    # ========================================================
#    
#    
#    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, Ginv, det_df, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_u, k2_u, k3_u, k4_u, k1_v, k2_v, k3_v, k4_v)
#    #$ omp for 
#    for ip in range(np):
#        
#        eta[:] = particles[0:3, ip]
#        v[:]   = particles[3:6, ip]
#
#        # ----------- step 1 in Runge-Kutta method -----------------------
#        # ========= mapping evaluation =============
#        pos1 = eta[0]
#        pos2 = eta[1]
#        pos3 = eta[2]
#
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv) ###########
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k1_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k1_u[:] = u/det_df
#        
#        k1_u[0] = 0.
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k1_v)
#
#        k1[:] = k1_v[:] + k1_u[:]
#
#        # ------------------------------------------------------------------
#        
#        
#        # ----------------- step 2 in Runge-Kutta method -------------------
#        pos1   = (eta[0] + dt*k1[0]/2)%1.
#        pos2   = (eta[1] + dt*k1[1]/2)%1.
#        pos3   = (eta[2] + dt*k1[2]/2)%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k2_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k2_u[:] = u/det_df
#        
#        k2_u[0] = 0.
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k2_v)
#
#        k2[:] = k2_v[:] + k2_u[:]
#        # ------------------------------------------------------------------
#        
#        
#        # ------------------ step 3 in Runge-Kutta method ------------------
#        pos1   = (eta[0] + dt*k2[0]/2)%1.
#        pos2   = (eta[1] + dt*k2[1]/2)%1.
#        pos3   = (eta[2] + dt*k2[2]/2)%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k3_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k3_u[:] = u/det_df
#            
#        k3_u[0] = 0.
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k3_v)
#
#        k3[:] = k3_v[:] + k3_u[:]
#        # ------------------------------------------------------------------
#        
#        
#        # ------------------ step 4 in Runge-Kutta method ------------------
#        pos1   = (eta[0] + dt*k3[0])%1.
#        pos2   = (eta[1] + dt*k3[1])%1.
#        pos3   = (eta[2] + dt*k3[2])%1.
#        
#        # ========= mapping evaluation =============
#        span1f = int(eta[0]*nelf[0]) + pf1
#        span2f = int(eta[1]*nelf[1]) + pf2
#        span3f = int(eta[2]*nelf[2]) + pf3
#        
#        # evaluate Jacobian matrix
#        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
#        
#        # evaluate inverse Jacobian matrix
#        mapping_fast.df_inv_all(df, dfinv)
#
#        # evaluate Jacobian determinant
#        det_df = abs(linalg.det(df))
#
#        # evaluate transposed inverse Jacobian matrix
#        linalg.transpose(dfinv, dfinv_t)
#
#        # evaluate Ginv matrix
#        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
#        # ============================================
#
#        # ========== field evaluation ==============
#        span1 = int(pos1*nel[0]) + pn1
#        span2 = int(pos2*nel[1]) + pn2
#        span3 = int(pos3*nel[2]) + pn3
#        
#        # evaluation of basis functions and derivatives
#        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
#        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
#        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
#        
#        # N-splines and D-splines at particle positions
#        bn1[:] = b1[pn1, :]
#        bn2[:] = b2[pn2, :]
#        bn3[:] = b3[pn3, :]
#        
#        bd1[:] = b1[pd1, :pn1] * d1[:]
#        bd2[:] = b2[pd2, :pn2] * d2[:]
#        bd3[:] = b3[pd3, :pn3] * d3[:]
#
#        # velocity field
#        if basis_u == 1:
#            u[0] = evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
#            u[1] = evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
#            u[2] = evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
#            
#            linalg.matrix_vector(Ginv, u, k4_u)
#            
#        elif basis_u ==2:
#            u[0] = evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
#            u[1] = evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
#            u[2] = evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
#
#            k4_u[:] = u/det_df
#            
#        k4_u[0] = 0.
#        
#        # pull-back of velocity
#        linalg.matrix_vector(dfinv, v, k4_v)
#
#        k4[:] = k4_v[:] + k4_u[:]
#        # ------------------------------------------------------------------
#
#        #  ---------------- update logical coordinates ---------------------
#        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
#        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
#        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0
#
#        # ------------------------------------------------------------------
#    #$ omp end parallel
#    
#    ierr = 0
#
#