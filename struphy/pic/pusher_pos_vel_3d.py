# import pyccel decorators
from pyccel.decorators import types

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

# ==========================================================================================================
@types(                'double[:]','double','int[:]','double[:]','double[:]','double[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:]','double[:]','int'  ,'int')
def aux_fun_x_v_stat_e(particle,   dt,      p,       t1,         t2,         t3,         indN1,     indN2,     indN3,     indD1,     indD2,     indD3,     loc1,       loc2,       loc3,       weight1,    weight2,    weight3,    nbase_n, nbase_d, e0_coeffs,  eps,        maxiter, ip):
    """
    Auxiliary function for the pusher_x_v_static_efield, introduced to enable time-step splitting if scheme does not converge for the standard dt

    Parameters:
    -----------
        particle : array
            shape(7), contains the values for the positions [0:3], velocities [3:6], and weights [6]
        
        dt : double
            time stepping

        p : int array
            contains the degrees of the basis splines in each direction
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3
        
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
        
        loc1 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 1

        loc2 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 2

        loc3 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 3
        
        weight1 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 1
        
        weight2 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 2
        
        weight3 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 3
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces
        
        e0_coeffs : array
            shape (3*p[0]*p[1]*p[2],) contains the values of the coefficient of the electric field
        
        eps: array
            determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

        maxiter : integer
            sets the maximum number of iterations for the iterative scheme
    """

    from numpy import empty, abs

    # total number of basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # number of quadrature points in direction 1
    n_quad1 = int(pd1*pn2*pn3/2) + 1
    # number of quadrature points in direction 2
    n_quad2 = int(pn1*pd2*pn3/2) + 1
    # number of quadrature points in direction 3
    n_quad3 = int(pn1*pn2*pd3/2) + 1

    eps_pos = eps[0]
    eps_vel = eps[1]

    # position
    eta1 = particle[0]
    eta2 = particle[1]
    eta3 = particle[2]

    # velocities
    v1 = particle[3]
    v2 = particle[4]
    v3 = particle[5]

    # set initial value for x_k^{n+1}
    eta1_curr = eta1
    eta2_curr = eta2
    eta3_curr = eta3

    # set initial value for v_k^{n+1}
    v1_curr = v1
    v2_curr = v2
    v3_curr = v3

    # set some initial value for eta_next
    eta1_next = 0.5
    eta2_next = 0.5
    eta3_next = 0.5

    # set some initial value for v_next
    v1_next = v1_curr
    v2_next = v2_curr
    v3_next = v3_curr

    runs = 0

    while abs(eta1_next - eta1_curr) > eps_pos or abs(eta2_next - eta2_curr) > eps_pos or abs(eta3_next - eta3_curr) > eps_pos or abs(v1_next - v1_curr) > eps_vel or abs(v2_next - v2_curr) > eps_vel or abs(v3_next - v3_curr) > eps_vel:

        # update the positions and velocities
        eta1_curr  = eta1_next
        eta2_curr  = eta2_next
        eta3_curr  = eta3_next

        v1_curr    = v1_next 
        v2_curr    = v2_next 
        v3_curr    = v3_next


        # ======================================================================================
        # update the positions and place them back into the computational domain
        eta1_next = ( eta1 + dt * (v1_curr + v1) / 2. )%1
        eta2_next = ( eta2 + dt * (v2_curr + v2) / 2. )%1
        eta3_next = ( eta3 + dt * (v3_curr + v3) / 2. )%1


        # ======================================================================================
        # update velocity in direction 1
        temp = 0.
        for k in range(n_quad1):

            # find position for quadrature formula; x1 + (1/2 * tau_i + 1/2) * ( x1_curr - x1 )
            quad_pos1 = ( eta1 + (loc1[k]+1)/2 * (eta1_curr - eta1) )%1
            quad_pos2 = ( eta2 + (loc1[k]+1)/2 * (eta2_curr - eta2) )%1
            quad_pos3 = ( eta3 + (loc1[k]+1)/2 * (eta3_curr - eta3) )%1

            # spans (i.e. index for non-vanishing basis functions)
            span1 = bsp.find_span(t1, pn1, quad_pos1)
            span2 = bsp.find_span(t2, pn2, quad_pos2)
            span3 = bsp.find_span(t3, pn3, quad_pos3)

            # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
            bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
            bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
            bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

            # find global index where non-zero basis functions begin
            ie1 = span1 - p[0]
            ie2 = span2 - p[1]
            ie3 = span3 - p[2]

            # (DNN)
            for il1 in range(pd1 + 1):
                i1 = indD1[ie1,il1]
                bi1 = bd1[il1]
                for il2 in range(pn2 +1):
                    i2 = indN2[ie2,il2]
                    bi2 = bi1 * bn2[il2]
                    for il3 in range(pn3 + 1):
                        i3 = indN3[ie3,il3]
                        bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_n[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

                        temp += bi3 * weight1[k]
        
        v1_next = v1 + dt * temp

        # ======================================================================================
        # update velocity in direction 2
        temp = 0.
        for k in range(n_quad2):

            # find position for quadrature formula; x1 + (1/2 * tau_i + 1/2) * ( x1_curr - x1 )
            quad_pos1 = ( eta1 + (loc1[k]+1)/2 * (eta1_curr - eta1) )%1
            quad_pos2 = ( eta2 + (loc1[k]+1)/2 * (eta2_curr - eta2) )%1
            quad_pos3 = ( eta3 + (loc1[k]+1)/2 * (eta3_curr - eta3) )%1

            # spans (i.e. index for non-vanishing basis functions)
            span1 = bsp.find_span(t1, pn1, quad_pos1)
            span2 = bsp.find_span(t2, pn2, quad_pos2)
            span3 = bsp.find_span(t3, pn3, quad_pos3)

            # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
            bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
            bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
            bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

            # find global index where non-zero basis functions begin
            ie1 = span1 - p[0]
            ie2 = span2 - p[1]
            ie3 = span3 - p[2]

            # (NDN)
            for il1 in range(pn1 + 1):
                i1 = indN1[ie1,il1]
                bi1 = bn1[il1]
                for il2 in range(pd2 +1):
                    i2 = indD2[ie2,il2]
                    bi2 = bi1 * bd2[il2]
                    for il3 in range(pn3 + 1):
                        i3 = indN3[ie3,il3]
                        bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_d[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]
                        
                        temp += bi3 * weight2[k]
        
        v2_next = v2 + dt * temp

        # ======================================================================================
        # update velocity in direction 3
        temp = 0.
        for k in range(n_quad3):

            # find position for quadrature formula; x1 + (1/2 * tau_i + 1/2) * ( x1_curr - x1 )
            quad_pos1 = ( eta1 + (loc1[k]+1)/2 * (eta1_curr - eta1) )%1
            quad_pos2 = ( eta2 + (loc1[k]+1)/2 * (eta2_curr - eta2) )%1
            quad_pos3 = ( eta3 + (loc1[k]+1)/2 * (eta3_curr - eta3) )%1

            # spans (i.e. index for non-vanishing basis functions)
            span1 = bsp.find_span(t1, pn1, quad_pos1)
            span2 = bsp.find_span(t2, pn2, quad_pos2)
            span3 = bsp.find_span(t3, pn3, quad_pos3)

            # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
            bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
            bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
            bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

            # find global index where non-zero basis functions begin
            ie1 = span1 - p[0]
            ie2 = span2 - p[1]
            ie3 = span3 - p[2]
        
            # (NND)
            for il1 in range(pn1 + 1):
                i1 = indN1[ie1,il1]
                bi1 = bn1[il1]
                for il2 in range(pn2 +1):
                    i2 = indN2[ie2,il2]
                    bi2 = bi1 * bn2[il2]
                    for il3 in range(pd3 + 1):
                        i3 = indD3[ie3,il3]
                        bi3 = bi2 * bd3[il3] * e0_coeffs[ nbase_n[1]*nbase_d[2]*i1 + nbase_d[2]*i2 + i3 ]
                        
                        temp += bi3 * weight3[k]
        
        v3_next = v3 + dt * temp

        runs += 1

        if runs == maxiter:
            break

    if runs < maxiter:
        # print('For convergence this took runs:', runs)
        # print()
        runs = 0

    # write the results in the particle array and impose periodic boundary conditions on the particles by taking modulo 1
    particle[0] = eta1_next%1
    particle[1] = eta2_next%1
    particle[2] = eta3_next%1
    particle[3] = v1_next
    particle[4] = v2_next
    particle[5] = v3_next

    return runs


# ==========================================================================================================
@types(                      'double[:,:]','double','int[:]','double[:]','double[:]','double[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','int','int[:]','int[:]','double[:]','double[:]','int')
def pusher_x_v_static_efield(particles,    dt,      p,       t1,         t2,         t3,         indN1,     indN2,     indN3,     indD1,     indD2,     indD3,     loc1,       loc2,       loc3,       weight1,    weight2,    weight3,    np,   nbase_n, nbase_d, e0_coeffs,  eps,        maxiter):
    """
    particle pusher for ODE dx/dt = v ; dv/dt = q/m * e_o(x)

    Parameters : 
    ------------
        particles : array
            shape(7, np), contains the values for the positions [:3,], velocities [3:6,], and weights [6,]
        
        dt : double
            time stepping

        p : int array
            contains the degrees of the basis splines in each direction
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3
        
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
        
        loc1 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 1

        loc2 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 2

        loc3 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 3
        
        weight1 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 1
        
        weight2 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 2
        
        weight3 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 3
        
        np : int
            number of particles
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces
        
        e0_coeffs : array
            shape (3*p[0]*p[1]*p[2],) contains the values of the coefficient of the electric field
        
        eps: array
            determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

        maxiter : integer
            sets the maximum number of iterations for the iterative scheme
    """

    from numpy import zeros

    particle = zeros( 7, dtype=float )

    #$ omp parallel
    #$ omp do private (ip, run, temp, k, m, particle, dt2)
    for ip in range(np):

        particle[:] = particles[:,ip]

        run = 1
        k   = 1
        
        while run != 0:
            k += 1
            if k == 5:
                print('Splitting the time steps into 4 has not been enough, aborting the iteration.')
                print()
                break

            run = 0
            
            dt2 = dt/k

            for m in range(k):
                temp = aux_fun_x_v_stat_e(particle, dt2, p, t1, t2, t3, indN1, indN2, indN3, indD1, indD2, indD3, loc1, loc2, loc3, weight1, weight2, weight3, nbase_n, nbase_d, e0_coeffs, eps, maxiter, ip)
                run += temp

        # write the results in the particles array
        particles[:,ip] = particle[:]

    #$ omp end do
    #$ omp end parallel

    ierr = 0
