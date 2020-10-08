# import pyccel decorators
from pyccel.decorators import types

# import input files for simulation setup
import input_run.equilibrium_MHD as eq_mhd

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step3(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, u1, u2, u3, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, zeros
    
    # ====================== for field evaluation =============
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
    
    # left and right values for spline evaluation
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
    
    # magnetic field, velocity field and electric field at particle position
    u   = empty( 3    , dtype=float)
    b   = empty( 3    , dtype=float)
    e   = empty( 3    , dtype=float)
    # ==========================================================
    
    
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
    l1f   = empty( pf1              , dtype=float)
    l2f   = empty( pf2              , dtype=float)
    l3f   = empty( pf3              , dtype=float)
    
    r1f   = empty( pf1              , dtype=float)
    r2f   = empty( pf2              , dtype=float)
    r3f   = empty( pf3              , dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1              , dtype=float)
    d2f   = empty( pf2              , dtype=float)
    d3f   = empty( pf3              , dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1          , dtype=float)
    der2f = empty( pf2 + 1          , dtype=float)
    der3f = empty( pf3 + 1          , dtype=float)
    
    
    # needed mapping quantities
    df        = empty((3, 3), dtype=float)
    dfinv_t   = empty((3, 3), dtype=float)
    
    temp_vec  = empty( 3    , dtype=float)
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, u, b, e, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, over_det_df, dfinv_t, temp_vec)
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
        
        #u[0] = eva.evaluation_kernel(pd1, pn2, pn3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pn3], span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
        #u[1] = eva.evaluation_kernel(pn1, pd2, pn3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pn3], span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
        #u[2] = eva.evaluation_kernel(pn1, pn2, pd3, b1[pn1], b2[pn2], b3[pd3, :pn3]*d3[:], span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
        
        u[0] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u1)
        u[1] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u2)
        u[2] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u3)
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b2_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b2_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        e[0] = b[1]*u[2] - b[2]*u[1]
        e[1] = b[2]*u[0] - b[0]*u[2]
        e[2] = b[0]*u[1] - b[1]*u[0]
        # ==========================================
        
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, eta1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, eta2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, eta3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # transposed, inverse Jacobian matrix
        dfinv_t[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv_t[1, 0] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv_t[2, 0] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv_t[0, 1] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv_t[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv_t[2, 1] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv_t[0, 2] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv_t[1, 2] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv_t[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
        # ==========================================
        
        
        # ======== particle pushing ================
        
        # perform push-forward of 1-form electric field to physical space
        linalg.matrix_vector(dfinv_t, e, temp_vec)
        
        # update particle velocities
        particles[3, ip] += dt*temp_vec[0]
        particles[4, ip] += dt*temp_vec[1]
        particles[5, ip] += dt*temp_vec[2]
        
        # ==========================================
        
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================================
@types('double[:,:]','double','int','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step4(particles, dt, np, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty
    
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
    l1f   = empty( pf1              , dtype=float)
    l2f   = empty( pf2              , dtype=float)
    l3f   = empty( pf3              , dtype=float)
    
    r1f   = empty( pf1              , dtype=float)
    r2f   = empty( pf2              , dtype=float)
    r3f   = empty( pf3              , dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1              , dtype=float)
    d2f   = empty( pf2              , dtype=float)
    d3f   = empty( pf3              , dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1          , dtype=float)
    der2f = empty( pf2 + 1          , dtype=float)
    der3f = empty( pf3 + 1          , dtype=float)
    
    # needed mapping quantities
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    # ========================================================
    
    
    # ================ particle velocity =====================
    v     = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty( 3, dtype=float)  
    k2 = empty( 3, dtype=float)  
    k3 = empty( 3, dtype=float)  
    k4 = empty( 3, dtype=float) 
    # ========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, over_det_df, dfinv, k1, k2, k3, k4, pos1, pos2, pos3)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        v[:] = particles[3:6, ip]
        
        # ----------- step 1 in Runge-Kutta method -----------------------
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, eta1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, eta2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, eta3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
                
        linalg.matrix_vector(dfinv, v, k1)
        # ------------------------------------------------------------------
        
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1 = (eta1 + dt*k1[0]/2)%1.
        pos2 = (eta2 + dt*k1[1]/2)%1.
        pos3 = (eta3 + dt*k1[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, pos1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, pos2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, pos3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
                
        linalg.matrix_vector(dfinv, v, k2)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1 = (eta1 + dt*k2[0]/2)%1.
        pos2 = (eta2 + dt*k2[1]/2)%1.
        pos3 = (eta3 + dt*k2[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, pos1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, pos2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, pos3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
                
        linalg.matrix_vector(dfinv, v, k3)
        # ------------------------------------------------------------------
        
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1 = (eta1 + dt*k3[0])%1.
        pos2 = (eta2 + dt*k3[1])%1.
        pos3 = (eta3 + dt*k3[2])%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, pos1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, pos2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, pos3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
                
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------
        
        
        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta1 + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.
        particles[1, ip] = (eta2 + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.
        particles[2, ip] = (eta3 + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.
        # ------------------------------------------------------------------
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step5(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, zeros
    
    # ====================== for field evaluation =============
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
    
    # left and right values for spline evaluation
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
    b      = empty( 3    , dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    # ==========================================================
    
    
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
    l1f   = empty( pf1              , dtype=float)
    l2f   = empty( pf2              , dtype=float)
    l3f   = empty( pf3              , dtype=float)
    
    r1f   = empty( pf1              , dtype=float)
    r2f   = empty( pf2              , dtype=float)
    r3f   = empty( pf3              , dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1              , dtype=float)
    d2f   = empty( pf2              , dtype=float)
    d3f   = empty( pf3              , dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1          , dtype=float)
    der2f = empty( pf2 + 1          , dtype=float)
    der3f = empty( pf3 + 1          , dtype=float)
    
    # needed mapping quantities
    df         = empty((3, 3), dtype=float) 
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
    
    identity[0, 0] = 1.
    identity[1, 1] = 1.
    identity[2, 2] = 1.
    
    # particle velocity
    v          = empty( 3    , dtype=float)
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, over_det_df, dfinv, dfinv_t, v, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
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
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b2_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b2_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, eta1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, eta2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, eta3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
        
        # transpose of inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        
        
        # ======== particle pushing ================
        v[:] = particles[3:6, ip]
        
        # perform matrix-matrix and matrix-vector multiplications
        linalg.matrix_matrix(b_prod, dfinv, temp_mat1)
        linalg.matrix_matrix(dfinv_t, temp_mat1, temp_mat2)
        
        # explicit part of update rule
        linalg.matrix_vector(identity - dt/2*temp_mat2, v, rhs)
        
        # implicit part of update rule
        lhs[:] = identity + dt/2*temp_mat2
        
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