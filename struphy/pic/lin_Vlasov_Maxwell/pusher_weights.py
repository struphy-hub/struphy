# import pyccel decorators
from pyccel.decorators import types

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

# import pyccel-ised background solutions
import struphy.kinetic_equil.analytical.background_sol as bs


# ==========================================================================================================
@types(          'double[:,:]','int[:]','double[:]','double[:]','double[:]','int','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:]','int[:]','double[:]', 'double[:]', 'double', 'double[:]','double[:]','double')
def push_weights(particles,    p,       t1,         t2,         t3,         np,   indN1,     indN2,     indN3,     indD1,     indD2,     indD3,     nbase_n, nbase_d, e_field_old, e_field_new, dt      , v_shift,    v_th,       n0      ):
    """
    updates the single weights in the e_W substep of the linearized Vlasov Maxwell system

    Parameters :
    ------------
        particles : array
            shape (6,np) contains positions [:3,], velocities [3:6,], and weights [6,]
        
        p : int array
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3
        
        np : integer
            total number of particles

        indN1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indN2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indN3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indD1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indD2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indD3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces

        e_field_old : array
            contains the values for e^n
        
        e_field_new : array
            contains the values for e^{n+1}
        
        dt : double
            value for time-stepping Delta t
        
        v_shift : array
            contains the 3 values of the shift velocity in the background solution (maxwellian) for all 3 directions
        
        v_th : array
            contains the 3 values for the thermal velocity of the background solution (maxwellian) for all 3 directions
        
        n0 : double
            homogeneous density of the background solution (maxwellian)
    """
    from numpy import empty, array

    # total number of basis functions : B-splines (pn) and D-splines(pd)
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


    # add e_field_new and e_field_old
    e_field = e_field_new + e_field_old

    v = empty( 3, dtype=float )

    #$ omp parallel
    #$ omp do private (ip, particles, eta1, eta2, eta3, v1, v2, v3, v, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, efield, f0, temp1, temp2, temp3, dt, i1, i2, i3, il1, il2, il3)
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

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
        bsp.basis_funs_slim(t1, pn1, eta1, span1, bn1, bd1)
        bsp.basis_funs_slim(t2, pn2, eta2, span2, bn2, bd2)
        bsp.basis_funs_slim(t3, pn3, eta3, span3, bn3, bd3)
        
        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3
        
        f0 = bs.maxwellian_point(v, v_shift, v_th, n0)

        temp1 = 0.
        temp2 = 0.
        temp3 = 0.

        # first component (DNN)
        for il1 in range(pd1 + 1):
            i1  = indD1[ie1,il1]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = indN2[ie2,il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = indN3[ie3,il3]
                    temp1 += bi2 * bn3[il3] * e_field[ nbase_n[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

        # second component (NDN)
        for il1 in range(pn1 + 1):
            i1  = indN1[ie1,il1]
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                i2  = indD2[ie2,il2]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = indN3[ie3,il3]
                    temp2 += bi2 * bn3[il3] * e_field[ nbase_d[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]
        
        # third component (NND)
        for il1 in range(pn1 + 1):
            i1  = indN1[ie1,il1]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = indN2[ie2,il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    i3  = indD3[ie3,il3]
                    temp3 += bi2 * bd3[il3] * e_field[ nbase_n[1]*nbase_d[2]*i1 + nbase_d[2]*i2 + i3 ]


        particles[6,ip] += ( temp1*v1 + temp2*v2 + temp3*v3 ) * f0 * dt/2

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0

