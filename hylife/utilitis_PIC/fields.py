# import pyccel decorators
from pyccel.decorators import types

# import input files for simulation setup
import input_run.equilibrium_MHD as eq_mhd

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==========================================================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','int','double[:]')
def evaluate_1form(particles_pos, t1, t2, t3, p, nel, nbase_n, nbase_d, np, u1, u2, u3, u_part, kind_map, params_map):
    
    from numpy import empty
    
    p1  = p[0]
    p2  = p[1]
    p3  = p[2]
    
    pd1 = p1 - 1
    pd2 = p2 - 1
    pd3 = p3 - 1
   
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((p1 + 1, p1 + 1), dtype=float)
    b2  = empty((p2 + 1, p2 + 1), dtype=float)
    b3  = empty((p3 + 1, p3 + 1), dtype=float)
    
    l1  = empty( p1             , dtype=float)
    l2  = empty( p2             , dtype=float)
    l3  = empty( p3             , dtype=float)
    
    r1  = empty( p1             , dtype=float)
    r2  = empty( p2             , dtype=float)
    r3  = empty( p3             , dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( p1             , dtype=float)
    d2  = empty( p2             , dtype=float)
    d3  = empty( p3             , dtype=float)
    
    #$ omp parallel
    #$ omp do private(ip, pos1, pos2, pos3, span1, span2, span3, b1, b2, b3, l1, l2, l3, r1, r2, r3, d1, d2, d3)
    for ip in range(np):
        
        pos1 = particles_pos[0, ip]
        pos2 = particles_pos[1, ip]
        pos3 = particles_pos[2, ip]
        
        # evaluation of equilibrium field
        u_part[0, ip] = eq_mhd.u1_eq(pos1, pos2, pos3, kind_map, params_map)
        u_part[1, ip] = eq_mhd.u2_eq(pos1, pos2, pos3, kind_map, params_map)
        u_part[2, ip] = eq_mhd.u3_eq(pos1, pos2, pos3, kind_map, params_map)
        
        # evaluation of perturbed field
        span1 = int(pos1*nel[0]) + p1
        span2 = int(pos2*nel[1]) + p2
        span3 = int(pos3*nel[2]) + p3
        
        bsp.basis_funs_all(t1, p1, pos1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, p2, pos2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, p3, pos3, span3, l3, r3, b3, d3)

        # 1 - component (DNN)
        u_part[0, ip] += eva.evaluation_kernel(pd1, p2, p3, b1[pd1, :p1]*d1[:], b2[p2], b3[p3], span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
        
        # 2 - component (NDN)
        u_part[1, ip] += eva.evaluation_kernel(p1, pd2, p3, b1[p1], b2[pd2, :p2]*d2[:], b3[p3], span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
        
        # 3 - component (NND)
        u_part[2, ip] += eva.evaluation_kernel(p1, p2, pd3, b1[p1], b2[p2], b3[pd3, :p3]*d3[:], span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','int','double[:]')
def evaluate_2form(particles_pos, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, b_part, kind_map, params_map):
    
    from numpy import empty
    
    p1  = p[0]
    p2  = p[1]
    p3  = p[2]
    
    pd1 = p1 - 1
    pd2 = p2 - 1
    pd3 = p3 - 1
   
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((p1 + 1, p1 + 1), dtype=float)
    b2  = empty((p2 + 1, p2 + 1), dtype=float)
    b3  = empty((p3 + 1, p3 + 1), dtype=float)
    
    l1  = empty( p1             , dtype=float)
    l2  = empty( p2             , dtype=float)
    l3  = empty( p3             , dtype=float)
    
    r1  = empty( p1             , dtype=float)
    r2  = empty( p2             , dtype=float)
    r3  = empty( p3             , dtype=float)
    
    # scaling arrays for M-splines
    d1 = empty( p1             , dtype=float)
    d2 = empty( p2             , dtype=float)
    d3 = empty( p3             , dtype=float)
    
    #$ omp parallel
    #$ omp do private(ip, pos1, pos2, pos3, span1, span2, span3, b1, b2, b3, l1, l2, l3, r1, r2, r3, d1, d2, d3)
    for ip in range(np):
        
        pos1 = particles_pos[0, ip]
        pos2 = particles_pos[1, ip]
        pos3 = particles_pos[2, ip]
        
        # evaluation of equilibrium field
        b_part[0, ip] = eq_mhd.b1_eq(pos1, pos2, pos3, kind_map, params_map)
        b_part[1, ip] = eq_mhd.b2_eq(pos1, pos2, pos3, kind_map, params_map)
        b_part[2, ip] = eq_mhd.b3_eq(pos1, pos2, pos3, kind_map, params_map)
        
        # evaluation of perturbed field
        span1 = int(pos1*nel[0]) + p1
        span2 = int(pos2*nel[1]) + p2
        span3 = int(pos3*nel[2]) + p3
        
        bsp.basis_funs_all(t1, p1, pos1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, p2, pos2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, p3, pos3, span3, l3, r3, b3, d3)

        # 1 - component (NDD)
        b_part[0, ip] += eva.evaluation_kernel(p1, pd2, pd3, b1[p1], b2[pd2, :p2]*d2[:], b3[pd3, :p3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        
        # 2 - component (DND)
        b_part[1, ip] += eva.evaluation_kernel(pd1, p2, pd3, b1[pd1, :p1]*d1[:], b2[p2], b3[pd3, :p3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        
        # 3 - component (DDN)
        b_part[2, ip] += eva.evaluation_kernel(pd1, pd2, p3, b1[pd1, :p1]*d1[:], b2[pd2, :p2]*d2[:], b3[p3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0