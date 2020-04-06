# import pyccel decorators
from pyccel.decorators import types

# absolute import of interface for simulation setup
import hylife.interface as inter


# ==============================================================================
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



        
# ==========================================================================================================
@types('double[:,:](order=F)','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:,:](order=F)','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','double[:]')
def evaluate_1form(particles_pos, t0_1, t0_2, t0_3, t1_1, t1_2, t1_3, p0, nel, nbase, np, u1, u2, u3, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, u_part, kind_map, params_map):
    
    from numpy import zeros
    from numpy import empty
    
    p0_1   = p0[0]
    p0_2   = p0[1]
    p0_3   = p0[2]
    
    p1_1   = p0_1 - 1
    p1_2   = p0_2 - 1
    p1_3   = p0_3 - 1
    
    delta1 = 1/float(nel[0])
    delta2 = 1/float(nel[1])
    delta3 = 1/float(nel[2])
    
    bl1    = p1_1*delta1
    bl2    = p1_2*delta2
    bl3    = p1_3*delta3
    
    br1    = 1 - p1_1*delta1
    br2    = 1 - p1_2*delta2
    br3    = 1 - p1_3*delta3
    
    nl1    = empty(p0_1,     dtype=float)
    nr1    = empty(p0_1,     dtype=float)
    nn1    = zeros(p0_1 + 1, dtype=float)
    
    nl2    = empty(p0_2,     dtype=float)
    nr2    = empty(p0_2,     dtype=float)
    nn2    = zeros(p0_2 + 1, dtype=float)
    
    nl3    = empty(p0_3,     dtype=float)
    nr3    = empty(p0_3,     dtype=float)
    nn3    = zeros(p0_3 + 1, dtype=float)
    
    dl1    = empty(p1_1    , dtype=float)
    dr1    = empty(p1_1    , dtype=float)
    dd1    = zeros(p1_1 + 1, dtype=float)
    
    dl2    = empty(p1_2    , dtype=float)
    dr2    = empty(p1_2    , dtype=float)
    dd2    = zeros(p1_2 + 1, dtype=float)
    
    dl3    = empty(p1_3    , dtype=float)
    dr3    = empty(p1_3    , dtype=float)
    dd3    = zeros(p1_3 + 1, dtype=float)
    
    
    #$ omp parallel
    #$ omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, posloc_1, posloc_2, posloc_3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, jl3, jl2, jl1, il3, il2, il1, pow1, pow2, pow3, i3, i2, i1, n3, n2, n1, d3, d2, d1)
    for ip in range(np):
        
        # evaluation of equilibrium field
        u_part[ip, 0] = inter.u1_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        u_part[ip, 1] = inter.u2_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        u_part[ip, 2] = inter.u3_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        
        span0_1  = int(particles_pos[ip, 0]*nel[0]) + p0_1
        span0_2  = int(particles_pos[ip, 1]*nel[1]) + p0_2
        span0_3  = int(particles_pos[ip, 2]*nel[2]) + p0_3
        
        span1_1  = span0_1 - 1
        span1_2  = span0_2 - 1
        span1_3  = span0_3 - 1
        
        posloc_1 = particles_pos[ip, 0] - (span0_1 - p0_1)*delta1
        posloc_2 = particles_pos[ip, 1] - (span0_2 - p0_2)*delta2
        posloc_3 = particles_pos[ip, 2] - (span0_3 - p0_3)*delta3
        
        
        # boundary region with recursive evaluation
        if (particles_pos[ip, 0] < bl1) or (particles_pos[ip, 0] > br1) or (particles_pos[ip, 1] < bl2) or (particles_pos[ip, 1] > br2) or (particles_pos[ip, 2] < bl3) or (particles_pos[ip, 2] > br3):
            
            basis_funs(t0_1, p0_1, particles_pos[ip, 0], span0_1, nl1, nr1, nn1)
            basis_funs(t0_2, p0_2, particles_pos[ip, 1], span0_2, nl2, nr2, nn2)
            basis_funs(t0_3, p0_3, particles_pos[ip, 2], span0_3, nl3, nr3, nn3)
            
            basis_funs(t1_1, p1_1, particles_pos[ip, 0], span1_1, dl1, dr1, dd1)
            basis_funs(t1_2, p1_2, particles_pos[ip, 1], span1_2, dl2, dr2, dd2)
            basis_funs(t1_3, p1_3, particles_pos[ip, 2], span1_3, dl3, dr3, dd3)
            
            # evaluation of 1 - component (DNN)
            for il1 in range(p1_1 + 1):
                i1 = (span1_1 - il1)%nbase[0, 0]
                d1 = dd1[p1_1 - il1]*p0_1/(t1_1[i1 + p0_1] - t1_1[i1])
                for il2 in range(p0_2 + 1):
                    i2 = (span0_2 - il2)%nbase[0, 1]
                    n2 = nn2[p0_2 - il2]
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%nbase[0, 2]
                        n3 = nn3[p0_3 - il3]
                        
                        u_part[ip, 0] += u1[i1, i2, i3] * d1 * n2 * n3
                        
            # evaluation of 2 - component (NDN)
            for il1 in range(p0_1 + 1):
                i1 = (span0_1 - il1)%nbase[1, 0]
                n1 = nn1[p0_1 - il1]
                for il2 in range(p1_2 + 1):
                    i2 = (span1_2 - il2)%nbase[1, 1]
                    d2 = dd2[p1_2 - il2]*p0_2/(t1_2[i2 + p0_2] - t1_2[i2])
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%nbase[1, 2]
                        n3 = nn3[p0_3 - il3]
                        
                        u_part[ip, 1] += u2[i1, i2, i3] * n1 * d2 * n3
                        
            # evaluation of 3 - component (NND)
            for il1 in range(p0_1 + 1):
                i1 = (span0_1 - il1)%nbase[2, 0]
                n1 = nn1[p0_1 - il1]
                for il2 in range(p0_2 + 1):
                    i2 = (span0_2 - il2)%nbase[2, 1]
                    n2 = nn2[p0_2 - il2]
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%nbase[2, 2]
                        d3 = dd3[p1_3 - il3]*p0_3/(t1_3[i3 + p0_3] - t1_3[i3])
                        
                        u_part[ip, 2] += u3[i1, i2, i3] * n1 * n2 * d3
        
        
        # interior with pp-form evaluation
        else:
            # evaluation of 1 - component (DNN)
            for jl3 in range(p0_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p0_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p1_1 + 1):
                        pow1 = posloc_1**jl1

                        for il3 in range(p0_3 + 1):
                            i3 = (span0_3 - il3)%nbase[0, 2]
                            n3 = pp0_3[p0_3 - il3, jl3] * pow3
                            for il2 in range(p0_2 + 1):
                                i2 = (span0_2 - il2)%nbase[0, 1]
                                n2 = pp0_2[p0_2 - il2, jl2] * pow2
                                for il1 in range(p1_1 + 1):
                                    i1 = (span1_1 - il1)%nbase[0, 0]
                                    d1 = pp1_1[p1_1 - il1, jl1] * pow1

                                    u_part[ip, 0] += u1[i1, i2, i3] * d1 * n2 * n3


            # evaluation of 2 - component (NDN)
            for jl3 in range(p0_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p1_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p0_1 + 1):
                        pow1 = posloc_1**jl1
                        
                        for il3 in range(p0_3 + 1):
                            i3 = (span0_3 - il3)%nbase[0, 2]
                            n3 = pp0_3[p0_3 - il3, jl3] * pow3
                            for il2 in range(p1_2 + 1):
                                i2 = (span1_2 - il2)%nbase[2, 1]
                                d2 = pp1_2[p1_2 - il2, jl2] * pow2
                                for il1 in range(p0_1 + 1):
                                    i1 = (span0_1 - il1)%nbase[0, 0]
                                    n1 = pp0_1[p0_1 - il1, jl1] * pow1 

                                    u_part[ip, 1] += u2[i1, i2, i3] * n1 * d2 * n3


            # evaluation of 3 - component (NND)
            for jl3 in range(p1_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p0_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p0_1 + 1):
                        pow1 = posloc_1**jl1

                        for il3 in range(p1_3 + 1):
                            i3 = (span1_3 - il3)%nbase[0, 2]
                            d3 = pp1_3[p1_3 - il3, jl3] * pow3
                            for il2 in range(p0_2 + 1):
                                i2 = (span0_2 - il2)%nbase[0, 1]
                                n2 = pp0_2[p0_2 - il2, jl2] * pow2
                                for il1 in range(p0_1 + 1):
                                    i1 = (span0_1 - il1)%nbase[0, 0]
                                    n1 = pp0_1[p0_1 - il1, jl1] * pow1 

                                    u_part[ip, 2] += u3[i1, i2, i3] * n1 * n2 * d3
                        
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0        
        
        
        
               
# ==========================================================================================================
@types('double[:,:](order=F)','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:,:](order=F)','int','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)','int','double[:]')
def evaluate_2form(particles_pos, t0_1, t0_2, t0_3, t1_1, t1_2, t1_3, p0, nel, nbase, np, b1, b2, b3, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, b_part, kind_map, params_map):
    
    from numpy import zeros
    from numpy import empty
    
    p0_1   = p0[0]
    p0_2   = p0[1]
    p0_3   = p0[2]
    
    p1_1   = p0_1 - 1
    p1_2   = p0_2 - 1
    p1_3   = p0_3 - 1
    
    delta1 = 1/float(nel[0])
    delta2 = 1/float(nel[1])
    delta3 = 1/float(nel[2])
    
    bl1    = p1_1*delta1
    bl2    = p1_2*delta2
    bl3    = p1_3*delta3
    
    br1    = 1 - p1_1*delta1
    br2    = 1 - p1_2*delta2
    br3    = 1 - p1_3*delta3
    
    nl1    = empty(p0_1,     dtype=float)
    nr1    = empty(p0_1,     dtype=float)
    nn1    = zeros(p0_1 + 1, dtype=float)
    
    nl2    = empty(p0_2,     dtype=float)
    nr2    = empty(p0_2,     dtype=float)
    nn2    = zeros(p0_2 + 1, dtype=float)
    
    nl3    = empty(p0_3,     dtype=float)
    nr3    = empty(p0_3,     dtype=float)
    nn3    = zeros(p0_3 + 1, dtype=float)
    
    dl1    = empty(p1_1    , dtype=float)
    dr1    = empty(p1_1    , dtype=float)
    dd1    = zeros(p1_1 + 1, dtype=float)
    
    dl2    = empty(p1_2    , dtype=float)
    dr2    = empty(p1_2    , dtype=float)
    dd2    = zeros(p1_2 + 1, dtype=float)
    
    dl3    = empty(p1_3    , dtype=float)
    dr3    = empty(p1_3    , dtype=float)
    dd3    = zeros(p1_3 + 1, dtype=float)
    
    
    #$ omp parallel
    #$ omp do private(ip, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, posloc_1, posloc_2, posloc_3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, jl3, jl2, jl1, il3, il2, il1, pow1, pow2, pow3, i3, i2, i1, n3, n2, n1, d3, d2, d1)
    for ip in range(np):
        
        # evaluation of equilibrium field
        b_part[ip, 0] = inter.b1_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        b_part[ip, 1] = inter.b2_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        b_part[ip, 2] = inter.b3_eq(particles_pos[ip, 0], particles_pos[ip, 1], particles_pos[ip, 2], kind_map, params_map)
        
        span0_1  = int(particles_pos[ip, 0]*nel[0]) + p0_1
        span0_2  = int(particles_pos[ip, 1]*nel[1]) + p0_2
        span0_3  = int(particles_pos[ip, 2]*nel[2]) + p0_3
        
        span1_1  = span0_1 - 1
        span1_2  = span0_2 - 1
        span1_3  = span0_3 - 1
        
        posloc_1 = particles_pos[ip, 0] - (span0_1 - p0_1)*delta1
        posloc_2 = particles_pos[ip, 1] - (span0_2 - p0_2)*delta2
        posloc_3 = particles_pos[ip, 2] - (span0_3 - p0_3)*delta3
        
        
        # boundary region with recursive evaluation
        if (particles_pos[ip, 0] < bl1) or (particles_pos[ip, 0] > br1) or (particles_pos[ip, 1] < bl2) or (particles_pos[ip, 1] > br2) or (particles_pos[ip, 2] < bl3) or (particles_pos[ip, 2] > br3):
            
            basis_funs(t0_1, p0_1, particles_pos[ip, 0], span0_1, nl1, nr1, nn1)
            basis_funs(t0_2, p0_2, particles_pos[ip, 1], span0_2, nl2, nr2, nn2)
            basis_funs(t0_3, p0_3, particles_pos[ip, 2], span0_3, nl3, nr2, nn3)
            
            basis_funs(t1_1, p1_1, particles_pos[ip, 0], span1_1, dl1, dr1, dd1)
            basis_funs(t1_2, p1_2, particles_pos[ip, 1], span1_2, dl2, dr2, dd2)
            basis_funs(t1_3, p1_3, particles_pos[ip, 2], span1_3, dl3, dr2, dd3)
            
            # evaluation of 1 - component (NDD)
            for il1 in range(p0_1 + 1):
                i1 = (span0_1 - il1)%nbase[0, 0]
                n1 = nn1[p0_1 - il1]
                for il2 in range(p1_2 + 1):
                    i2 = (span1_2 - il2)%nbase[0, 1]
                    d2 = dd2[p1_2 - il2]*p0_2/(t1_2[i2 + p0_2] - t1_2[i2])
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%nbase[0, 2]
                        d3 = dd3[p1_3 - il3]*p0_3/(t1_3[i3 + p0_3] - t1_3[i3])
                        
                        b_part[ip, 0] += b1[i1, i2, i3] * n1 * d2 * d3
                        
            # evaluation of 2 - component (DND)
            for il1 in range(p1_1 + 1):
                i1 = (span1_1 - il1)%nbase[1, 0]
                d1 = dd1[p1_1 - il1]*p0_1/(t1_1[i1 + p0_1] - t1_1[i1])
                for il2 in range(p0_2 + 1):
                    i2 = (span0_2 - il2)%nbase[1, 1]
                    n2 = nn2[p0_2 - il2]
                    for il3 in range(p1_3 + 1):
                        i3 = (span1_3 - il3)%nbase[1, 2]
                        d3 = dd3[p1_3 - il3]*p0_3/(t1_3[i3 + p0_3] - t1_3[i3])
                        
                        b_part[ip, 1] += b2[i1, i2, i3] * d1 * n2 * d3
                        
            # evaluation of 3 - component (DDN)
            for il1 in range(p1_1 + 1):
                i1 = (span1_1 - il1)%nbase[2, 0]
                d1 = dd1[p1_1 - il1]*p0_1/(t1_1[i1 + p0_1] - t1_1[i1])
                for il2 in range(p1_2 + 1):
                    i2 = (span1_2 - il2)%nbase[2, 1]
                    d2 = dd2[p1_2 - il2]*p0_2/(t1_2[i2 + p0_2] - t1_2[i2])
                    for il3 in range(p0_3 + 1):
                        i3 = (span0_3 - il3)%nbase[2, 2]
                        n3 = nn3[p0_3 - il3]
                        
                        b_part[ip, 2] += b3[i1, i2, i3] * d1 * d2 * n3
        
        
        # interior with pp-form evaluation
        else:
            # evaluation of 1 - component (NDD)
            for jl3 in range(p1_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p1_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p0_1 + 1):
                        pow1 = posloc_1**jl1

                        for il3 in range(p1_3 + 1):
                            i3 = (span1_3 - il3)%nbase[0, 2]
                            d3 = pp1_3[p1_3 - il3, jl3] * pow3
                            for il2 in range(p1_2 + 1):
                                i2 = (span1_2 - il2)%nbase[0, 1]
                                d2 = pp1_2[p1_2 - il2, jl2] * pow2 
                                for il1 in range(p0_1 + 1):
                                    i1 = (span0_1 - il1)%nbase[0, 0]
                                    n1 = pp0_1[p0_1 - il1, jl1] * pow1 

                                    b_part[ip, 0] += b1[i1, i2, i3] * n1 * d2 * d3


            # evaluation of 2 - component (DND)
            for jl3 in range(p1_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p0_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p1_1 + 1):
                        pow1 = posloc_1**jl1

                        for il3 in range(p1_3 + 1):
                            i3 = (span1_3 - il3)%nbase[1, 2]
                            d3 = pp1_3[p1_3 - il3, jl3] * pow3
                            for il2 in range(p0_2 + 1):
                                i2 = (span0_2 - il2)%nbase[1, 1]
                                n2 = pp0_2[p0_2 - il2, jl2] * pow2
                                for il1 in range(p1_1 + 1):
                                    i1 = (span1_1 - il1)%nbase[1, 0]
                                    d1 = pp1_1[p1_1 - il1, jl1] * pow1

                                    b_part[ip, 1] += b2[i1, i2, i3] * d1 * n2 * d3


            # evaluation of 3 - component (DDN)
            for jl3 in range(p0_3 + 1):
                pow3 = posloc_3**jl3
                for jl2 in range(p1_2 + 1):
                    pow2 = posloc_2**jl2
                    for jl1 in range(p1_1 + 1):
                        pow1 = posloc_1**jl1

                        for il3 in range(p0_3 + 1):
                            i3 = (span0_3 - il3)%nbase[2, 2]
                            n3 = pp0_3[p0_3 - il3, jl3] * pow3
                            for il2 in range(p1_2 + 1):
                                i2 = (span1_2 - il2)%nbase[2, 1]
                                d2 = pp1_2[p1_2 - il2, jl2] * pow2
                                for il1 in range(p1_1 + 1):
                                    i1 = (span1_1 - il1)%nbase[2, 0]
                                    d1 = pp1_1[p1_1 - il1, jl1] * pow1

                                    b_part[ip, 2] += b3[i1, i2, i3] * d1 * d2 * n3
                        
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
