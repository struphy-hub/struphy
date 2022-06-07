# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

import struphy.feec.basics.spline_evaluation_2d as eva2
import struphy.feec.basics.spline_evaluation_3d as eva3

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast



# ==============================================================================
def set_particles_symmetric_6d(numbers : 'float[:,:]', particles : 'float[:,:]', np : 'int'):
    
    from numpy import zeros
    
    e = zeros(3, dtype=float)
    v = zeros(3, dtype=float)
    
    for i_part in range(np):
        ip = i_part%64
        
        if ip == 0:
            e[:] = numbers[int(i_part/64), 0:3]
            v[:] = numbers[int(i_part/64), 3:6]
            
        elif ip%32 == 0:
            v[2] = 1 - v[2]
            
        elif ip%16 == 0:
            v[1] = 1 - v[1]
            
        elif ip%8 == 0:
            v[0] = 1 - v[0]
            
        elif ip%4 == 0:
            e[2] = 1 - e[2] 
             
        elif ip%2 == 0:
            e[1] = 1 - e[1]
            
        else:
            e[0] = 1 - e[0]
        
        particles[0:3, i_part] = e
        particles[3:6, i_part] = v  
        
        
# ==============================================================================
def set_particles_symmetric_5d(numbers : 'float[:,:]', particles : 'float[:,:]', np : 'int'):
    
    from numpy import zeros
    
    e = zeros(2, dtype=float)
    v = zeros(3, dtype=float)
    
    for i_part in range(np):
        ip = i_part%32
        
        if ip == 0:
            e[:] = numbers[int(i_part/32), 0:2]
            v[:] = numbers[int(i_part/32), 2:5]
            
        elif ip%16 == 0:
            v[2] = 1 - v[2]
            
        elif ip%8 == 0:
            v[1] = 1 - v[1]
            
        elif ip%4 == 0:
            v[0] = 1 - v[0]
            
        elif ip%2 == 0:
            e[1] = 1 - e[1] 
            
        else:
            e[0] = 1 - e[0]
        
        particles[1:3, i_part] = e
        particles[3:6, i_part] = v               

        
# ==============================================================================
def convert(particles : 'float[:,:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b_eq_1 : 'float[:,:,:]', b_eq_2 : 'float[:,:,:]', b_eq_3 : 'float[:,:,:]', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, sqrt, cos, sin
    
    # ============== for magnetic field evaluation ============
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
    l1  = empty( pn1, dtype=float)
    l2  = empty( pn2, dtype=float)
    l3  = empty( pn3, dtype=float)
    
    r1  = empty( pn1, dtype=float)
    r2  = empty( pn2, dtype=float)
    r3  = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1, dtype=float)
    d2  = empty( pn2, dtype=float)
    d3  = empty( pn3, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # magnetic field at particle position (2-form, cartesian, normalized cartesian)
    b      = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0     = empty(3, dtype=float)
    
    # particle velocity (cartesian, perpendicular, v x b0, b0 x vperp)
    v        = empty(3, dtype=float)
    vperp    = empty(3, dtype=float)
    vxb0     = empty(3, dtype=float)
    b0xvperp = empty(3, dtype=float)
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
    fx    = empty( 3    , dtype=float)
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    # ==========================================================
    
    # local basis vectors perpendicular to magnetic field
    e1 = empty(3, dtype=float)
    e2 = empty(3, dtype=float)

    for ip in range(np):

        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        
        if nel[2] > 0:
            
            span3 = int(eta3*nel[2]) + pn3
            
            bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
            
            bn3[:] = b3[pn3, :]
            bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # magnetic field (2-form)
        if nel[2] > 0:
        
            b[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b_eq_1)
            b[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b_eq_2)
            b[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b_eq_3)
            
        else:
            
            b[0] = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
            b[1] = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
            b[2] = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 2)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        
        # extract basis vector perpendicular to b
        e1[0] = dfinv[0, 0]
        e1[1] = dfinv[0, 1]
        e1[2] = dfinv[0, 2]
        
        e1_norm = sqrt(e1[0]**2 + e1[1]**2 + e1[2]**2)
        
        e1[:] = e1/e1_norm
        # ==========================================
        
        # push-forward of magnetic field
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df
        
        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        
        # normalized magnetic field direction
        b0[:] = b_cart/b_norm
        
        # calculate e2 = b0 x e1
        linalg.cross(b0, e1, e2)
        
        # calculate Cartesian velocity components
        particles[3:6, ip] = particles[3, ip]*cos(particles[4, ip])*b0 + particles[3, ip]*sin(particles[4, ip])*(cos(particles[5, ip])*e1 + sin(particles[5, ip])*e2)