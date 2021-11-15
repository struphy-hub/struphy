# import pyccel decorators
from pyccel.decorators import types

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for mapping evaluation
import struphy.geometry.mappings_3d      as mapping
import struphy.geometry.mappings_3d_fast as mapping_fast

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eva


# ==========================================================================================================
@types('double[:,:]','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step4(particles, dt, np, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, sqrt, arctan2, pi, cos, sin
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, k1, k2, k3, k4)
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # ----------- step 1 in Runge-Kutta method -----------------------
        pos1   = eta[0]
        pos2   = eta[1]
        pos3   = eta[2]
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1)
        # ------------------------------------------------------------------
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1   = (eta[0] + dt*k1[0]/2)%1.
        pos2   = (eta[1] + dt*k1[1]/2)%1.
        pos3   = (eta[2] + dt*k1[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k2[0]/2)%1.
        pos2   = (eta[1] + dt*k2[1]/2)%1.
        pos3   = (eta[2] + dt*k2[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k3[0])%1.
        pos2   = (eta[1] + dt*k3[1])%1.
        pos3   = (eta[2] + dt*k3[2])%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------
        
        
        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0
        # ------------------------------------------------------------------
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ========================================================================================================    
@types('double[:,:]','double[:,:]','double[:]')
def reflect(df, df_inv, v):
    
    from numpy import empty, sqrt
    
    vg        = empty( 3    , dtype=float)
    
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
@types('double[:,:]','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double')
def pusher_step4_pcart(particles, dt, np, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, a, r0):
    
    from numpy import empty, zeros
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df        = empty((3, 3), dtype=float)
    dfinv     = empty((3, 3), dtype=float)
    
    df_old    = empty((3, 3), dtype=float)
    dfinv_old = empty((3, 3), dtype=float)
    
    fx        = empty( 3    , dtype=float)
    
    # needed mapping quantities for pseudo-cartesian coordinates
    df_pseudo     = empty((3, 3), dtype=float)
    
    df_pseudo_old = empty((3, 3), dtype=float)
    fx_pseudo     = empty( 3    , dtype=float)
    
    params_pseudo = empty( 3    , dtype=float)
    
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, fx_pseudo, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df_old, fx, dfinv_old, df_pseudo_old, df, dfinv, df_pseudo, v_temp, k1, k2, k3, k4)
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        # old logical coordinates and velocities
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # compute old pseudo-cartesian coordinates
        fx_pseudo[0] = mapping.f(eta[0], eta[1], eta[2], 1, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        fx_pseudo[1] = mapping.f(eta[0], eta[1], eta[2], 2, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        fx_pseudo[2] = mapping.f(eta[0], eta[1], eta[2], 3, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
       
        # evaluate old Jacobian matrix of mapping F
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df_old, fx, 0)

        # evaluate old inverse Jacobian matrix of mapping F
        mapping_fast.df_inv_all(df_old, dfinv_old)
        
        # evaluate old Jacobian matrix of mapping F_pseudo
        df_pseudo_old[0, 0] = mapping.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[0, 1] = mapping.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[0, 2] = mapping.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        df_pseudo_old[1, 0] = mapping.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[1, 1] = mapping.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[1, 2] = mapping.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        df_pseudo_old[2, 0] = mapping.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[2, 1] = mapping.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[2, 2] = mapping.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        while True:
            
            # ----------- step 1 in Runge-Kutta method -----------------------
            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv_old, v, v_temp)
            linalg.matrix_vector(df_pseudo_old, v_temp, k1)
            # ------------------------------------------------------------------
            
        
            # ----------------- step 2 in Runge-Kutta method -------------------
            eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 1, 14, params_pseudo)
            eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 2, 14, params_pseudo)
            eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)
            
            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mapping.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mapping.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mapping.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mapping.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mapping.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mapping.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mapping.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mapping.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k2)
            # ------------------------------------------------------------------


            # ------------------ step 3 in Runge-Kutta method ------------------
            eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 1, 14, params_pseudo)
            eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 2, 14, params_pseudo)
            eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                   
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mapping.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mapping.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mapping.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mapping.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mapping.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mapping.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mapping.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mapping.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k3)
            # ------------------------------------------------------------------


            # ------------------ step 4 in Runge-Kutta method ------------------
            eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 1, 14, params_pseudo)
            eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 2, 14, params_pseudo)
            eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mapping.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mapping.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mapping.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mapping.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mapping.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mapping.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mapping.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mapping.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

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
            eta[0] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 1, 14, params_pseudo)
            eta[1] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 2, 14, params_pseudo)
            eta[2] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 3, 14, params_pseudo)
            
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
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    

    
# ==========================================================================================================
@types('double[:,:]','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double')
def pusher_step4_cart(particles, dt, np, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, tol):
    
    from numpy import empty, sqrt, arctan2, pi, cos, sin
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    temp  = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v   = empty(3, dtype=float)
    
    fx  = empty(3, dtype=float)
    x   = empty(3, dtype=float)
    # ========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, temp, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, x)
    for ip in range(np):
        
        eta[:]  = particles[0:3, ip]
        v[:]    = particles[3:6, ip]
        temp[:] = 0.
        
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix and mapping
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 2)
        
        # update cartesian coordinates
        fx[0] = (1.0 + 0.1*eta[0]*cos(2*pi*eta[1]))*cos(2*pi*eta[2])
        fx[1] =        0.1*eta[0]*sin(2*pi*eta[1])
        fx[2] = (1.0 + 0.1*eta[0]*cos(2*pi*eta[1]))*sin(2*pi*eta[2])
        
        x[:] = fx + dt*v
        
        particles[0, ip] = sqrt((sqrt(x[0]**2 + x[2]**2) - 1.0)**2 + x[1]**2)/0.1
        particles[1, ip] = (arctan2(x[1], sqrt(x[0]**2 + x[2]**2) - 1.0)/(2*pi))%1.0
        particles[2, ip] = (arctan2(x[2], x[0])/(2*pi))%1.0
        
        # calculate new logical coordinates by solving inverse mapping with Newton-method
        # evaluate inverse Jacobian matrix
        #mapping_fast.df_inv_all(df, dfinv)
        
        #while True:
#
        #    fx[:] = fx - x
        #    linalg.matrix_vector(dfinv, fx, temp)
        #    
        #    eta[0] =  eta[0] - temp[0]
        #    eta[1] = (eta[1] - temp[1])%1.0
        #    eta[2] = (eta[2] - temp[2])%1.0
        #    
        #    span1f = int(eta[0]*nelf[0]) + pf1
        #    span2f = int(eta[1]*nelf[1]) + pf2
        #    span3f = int(eta[2]*nelf[2]) + pf3
        #    
        #    # evaluate Jacobian matrix and mapping
        #    mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 2)
        #    
        #    if abs(fx[0] - x[0]) < tol and abs(fx[1] - x[1]) < tol and abs(fx[2] - x[2]) < tol:
        #        particles[0:3, ip] = eta
        #        break
        #    
        #    # evaluate inverse Jacobian matrix
        #    mapping_fast.df_inv_all(df, dfinv)
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0