# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

# import pyccel-ised background solutions
import struphy.kinetic_equil.analytical.background_sol as bs


# ==========================================================================================================
def push_weights(particles : 'double[:,:]', np : 'int', p : 'int[:]', t1 : 'double[:]', t2 : 'double[:]', t3 : 'double[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', indd1 : 'int[:,:]', indd2 : 'int[:,:]', indd3 : 'int[:,:]', nbase_n : 'int[:]', nbase_d : 'int[:]', e_field : 'double[:]', dt : 'double', v_shift : 'double[:]', v_th : 'double[:]', n0 :  'double'):
    """
    updates the single weights in the e_W substep of the linearized Vlasov Maxwell system

    Parameters :
    ------------
        particles : array
            shape (6,np) contains positions [:3,], velocities [3:6,], and weights [6,]
        
        np : integer
            total number of particles
        
        p : int array
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3

        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces
        
        e_field : array
            shape(3*Nel[0]*Nel[1]*Nel[2],) , contains the values for e^{n+1} + e^n
        
        dt : double
            value for time-stepping Delta t
        
        v_shift : array
            contains the 3 values of the shift velocity in the background solution (maxwellian) for all 3 directions
        
        v_th : array
            contains the 3 values for the thermal velocity of the background solution (maxwellian) for all 3 directions
        
        n0 : double
            homogeneous density of the background solution (maxwellian)
    """
    from numpy import empty, sqrt

    # total number of basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    v = empty( 3, dtype=float )

    #$ omp parallel private(ip, eta1, eta2, eta3, v1, v2, v3, v, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, ie1, ie2, ie3, f0, temp1, temp2, temp3, i1, i2, i3, il1, il2, il3, bi1, bi2, update)
    #$ omp for 
    for ip in range(np):
        
        # position
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # velocity
        v1      = particles[3, ip]
        v2      = particles[4, ip]
        v3      = particles[5, ip]
        v       = [v1, v2, v3]

        # spans (i.e. index for non-vanisle of manishing basis functions)
        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
        bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)
        
        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3
        
        f0 = bs.maxwellian_point(v, v_shift, v_th, n0)

        temp1 = 0.
        temp2 = 0.
        temp3 = 0.

        # first component (DNN)
        for il1 in range(pd1 + 1):
            i1  = indd1[ie1,il1]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = indn2[ie2,il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = indn3[ie3,il3]
                    temp1 += bi2 * bn3[il3] * e_field[ nbase_n[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

        # second component (NDN)
        for il1 in range(pn1 + 1):
            i1  = indn1[ie1,il1]
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                i2  = indd2[ie2,il2]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = indn3[ie3,il3]
                    temp2 += bi2 * bn3[il3] * e_field[ nbase_d[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]
        
        # third component (NND)
        for il1 in range(pn1 + 1):
            i1  = indn1[ie1,il1]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = indn2[ie2,il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    i3  = indd3[ie3,il3]
                    temp3 += bi2 * bd3[il3] * e_field[ nbase_n[1]*nbase_d[2]*i1 + nbase_d[2]*i2 + i3 ]

        update = ( temp1*v1 + temp2*v2 + temp3*v3 ) * sqrt(f0) * dt/2.
        particles[6,ip] += update
    #$ omp end parallel
    
    ierr = 0
