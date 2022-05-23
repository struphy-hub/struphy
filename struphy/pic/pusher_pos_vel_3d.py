# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast

# ==========================================================================================================
def quicksort(a: 'float[:]', lo: 'int', hi: 'int'):
    """
    Implementation of the quicksort sorting algorithm

    Parameters:
    -----------
        a : array
            list that is to be sorted
        
        lo : integer
            lower index from which the sort to start
        
        hi : integer
            upper index until which the sort is to be done
    """
    i = lo
    j = hi
    while i < hi:
        pivot = a[(lo + hi) // 2]
        while i <= j :
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp  = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if lo < j:
            quicksort(a, lo, j)
        lo = i
        j = hi


# ==========================================================================================================
def find_taus(eta: 'float', eta_next: 'float', Nel: 'int', breaks: 'float[:]', uniform: 'int', tau_list: 'float[:]'):
    """
    Find the values of tau for which the particle crosses the cell boundaries while going from eta to eta_next

    Parameters:
    -----------
        eta : float
            old position
 
        eta_next : float
            new position

        Nel : integer
            contains the number of elements in this direction
        
        breaks : array
            break points in this direction

        uniform : integer
            0 if the grid is non-uniform, 1 if the grid is uniform
    """

    from numpy import floor

    if uniform == 1:
        index      = int( floor( eta * Nel ) )
        index_next = int( floor( eta_next * Nel ) )
        length     = int( abs( index_next - index ) )
        
        # break = eta / dx = eta * Nel

        for i in range(length):
            if index_next > index:
                tau_list[i] = (1.0 / Nel * (index + i + 1) - eta) / (eta_next - eta)
            elif index > index_next:
                tau_list[i] = (eta - 1.0 / Nel * (index - i)) / (eta - eta_next)
    
    elif uniform == 0:
        # TODO
        print('Not implemented yet')
    
    else:
        print('ValueError, uniform must be 1 or 0 !')


# ==========================================================================================================
def aux_fun_x_v_stat_e(particle: 'float[:]', kind_map: 'int', params_map: 'float[:]', dt: 'float', p: 'int[:]', Nel: 'int[:]', breaks1: 'float[:]', breaks2: 'float[:]', breaks3: 'float[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', indN1: 'int[:,:]', indN2: 'int[:,:]', indN3: 'int[:,:]', indD1: 'int[:,:]', indD2: 'int[:,:]', indD3: 'int[:,:]', loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]', weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]', nbase_n: 'int[:]', nbase_d: 'int[:]', e0_coeffs: 'float[:]', eps: 'float[:]', maxiter: 'int', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> 'int':
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
        
        Nel : int array
            contains the number of elements in each direction
        
        breaks1 : array
            contains the break points in direction 1

        breaks2 : array
            contains the break points in direction 2

        breaks3 : array
            contains the break points in direction 3
        
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

    from numpy import empty, abs, floor

    df      = empty( (3,3), dtype=float )
    df_inv  = empty( (3,3), dtype=float )
    fx      = empty( 3    , dtype=float )

    # total number of basis functions : B-splines (pn   ) and D-splines (pd)
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
    n_quad1 = int( floor( pd1*pn2*pn3 / 2 + 1 ) )
    # number of quadrature points in direction 2
    n_quad2 = int( floor( pn1*pd2*pn3 / 2 + 1 ) )
    # number of quadrature points in direction 3
    n_quad3 = int( floor( pn1*pn2*pd3 / 2 + 1 ) )

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

    # Use Euler method as a predictor for positions
    mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx, cy, cz, indN1, indN2, indN3, eta1, eta2, eta3, df, fx, 0)
    mapping_fast.df_inv_all(df, df_inv)

    v1_curv = df_inv[0,0]*(v1_curr + v1) + df_inv[0,1]*(v2_curr + v2) + df_inv[0,2]*(v3_curr + v3)
    v2_curv = df_inv[1,0]*(v1_curr + v1) + df_inv[1,1]*(v2_curr + v2) + df_inv[1,2]*(v3_curr + v3)
    v3_curv = df_inv[2,0]*(v1_curr + v1) + df_inv[2,1]*(v2_curr + v2) + df_inv[2,2]*(v3_curr + v3)

    eta1_next = ( eta1 + dt * v1_curv / 2. )%1
    eta2_next = ( eta2 + dt * v2_curv / 2. )%1
    eta3_next = ( eta3 + dt * v3_curv / 2. )%1

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

        # find Jacobian matrix
        mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx, cy, cz, indN1, indN2, indN3, (eta1_curr + eta1)/2, (eta2_curr + eta2)/2, (eta3_curr + eta3)/2, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, df_inv)

        # ======================================================================================
        # update the positions and place them back into the computational domain
        v1_curv = df_inv[0,0]*(v1_curr + v1) + df_inv[0,1]*(v2_curr + v2) + df_inv[0,2]*(v3_curr + v3)
        v2_curv = df_inv[1,0]*(v1_curr + v1) + df_inv[1,1]*(v2_curr + v2) + df_inv[1,2]*(v3_curr + v3)
        v3_curv = df_inv[2,0]*(v1_curr + v1) + df_inv[2,1]*(v2_curr + v2) + df_inv[2,2]*(v3_curr + v3)

        # x_{n+1} = x_n + dt/2 * DF^{-1}(x_{n+1}/2 + x_n/2) * (v_{n+1} + v_n)
        eta1_next = ( eta1 + dt * v1_curv / 2. )%1
        eta2_next = ( eta2 + dt * v2_curv / 2. )%1
        eta3_next = ( eta3 + dt * v3_curv / 2. )%1



        # ======================================================================================
        # Compute tau-values in [0,1] for crossings of cell-boundaries

        index1      = int( floor( eta1_curr * Nel[0] ) )
        index1_next = int( floor( eta1_next * Nel[0] ) )
        length1     = int( abs( index1_next - index1 ) )

        index2      = int( floor( eta2_curr * Nel[1] ) )
        index2_next = int( floor( eta2_next * Nel[1] ) )
        length2     = int( abs( index2_next - index2 ) )

        index3      = int( floor( eta3_curr * Nel[2] ) )
        index3_next = int( floor( eta3_next * Nel[2] ) )
        length3     = int( abs( index3_next - index3 ) )

        length = length1 + length2 + length3

        taus = empty( length + 2, dtype=float )

        taus[0]          = 0.0
        taus[length + 1] = 1.0

        find_taus(eta1_curr, eta1_next, Nel[0], breaks1, 1, taus[1:length1+1])
        find_taus(eta2_curr, eta2_next, Nel[1], breaks2, 1, taus[length1+1:length1+length2+1])
        find_taus(eta3_curr, eta3_next, Nel[2], breaks3, 1, taus[length1+length2+1:length+1])

        quicksort(taus, 1, length)


        # ======================================================================================
        # update velocity in direction 1

        temp1 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta1 + taus[k] * ( eta1_curr - eta1 )
            b      = eta1 + taus[k+1] * ( eta1_curr - eta1 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad1):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

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
                    # i1 = indN1[ie1-1,il1+1]
                    bi1 = bd1[il1]
                    for il2 in range(pn2 +1):
                        i2 = indN2[ie2,il2]
                        bi2 = bi1 * bn2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = indN3[ie3,il3]
                            bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_n[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

                            temp1 += bi3 * weight1[n]

        # ======================================================================================
        # update velocity in direction 2

        temp2 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta2 + taus[k] * ( eta2_curr - eta2 )
            b      = eta2 + taus[k+1] * ( eta2_curr - eta2 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad2):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

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
                        # i2 = indN2[ie2-1,il2+1]
                        bi2 = bi1 * bd2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = indN3[ie3,il3]
                            bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_d[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

                            temp2 += bi3 * weight2[n]
            
        # ======================================================================================
        # update velocity in direction 3

        temp3 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta3 + taus[k] * ( eta3_curr - eta3 )
            b      = eta3 + taus[k+1] * ( eta3_curr - eta3 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad3):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

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
                            # i3 = indN3[ie3-1,il3+1]
                            bi3 = bi2 * bd3[il3] * e0_coeffs[ nbase_n[1]*nbase_d[2]*i1 + nbase_d[2]*i2 + i3 ]

                            temp3 += bi3 * weight3[n]

        # v_{n+1} = v_n + dt * DF^{-T}(x_n) * int_0^1 d tau ( E(x_n + tau*(x_{n+1} - x_n) ) )
        v1_next = v1 + dt * ( df_inv[0,0]*temp1 + df_inv[1,0]*temp2 + df_inv[2,0]*temp3 )
        v2_next = v2 + dt * ( df_inv[0,1]*temp1 + df_inv[1,1]*temp2 + df_inv[2,1]*temp3 )
        v3_next = v3 + dt * ( df_inv[0,2]*temp1 + df_inv[1,2]*temp2 + df_inv[2,2]*temp3 )

        runs += 1
        del(taus)

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
def pusher_x_v_static_efield(particles: 'float[:,:]', kind_map: 'int', params_map: 'float[:]', dt: 'float', p: 'int[:]', Nel: 'int[:]', breaks1: 'float[:]', breaks2: 'float[:]', breaks3: 'float[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', indN1: 'int[:,:]', indN2: 'int[:,:]', indN3: 'int[:,:]', indD1: 'int[:,:]', indD2: 'int[:,:]', indD3: 'int[:,:]', loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]', weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]', np: 'int', nbase_n: 'int[:]', nbase_d: 'int[:]', e0_coeffs: 'float[:]', eps: 'float[:]', maxiter: 'int', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
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
        
        Nel : int array
            contains the number of elements in each direction

        breaks1 : array
            contains the break points in direction 1

        breaks2 : array
            contains the break points in direction 2

        breaks3 : array
            contains the break points in direction 3
        
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

    #$ omp parallel private(ip, run, temp, k, m, particle, dt2)
    #$ omp for
    for ip in range(np):

        particle[:] = particles[:,ip]

        run = 1
        k   = 0
        
        while run != 0:
            k += 1
            if k == 5:
                print('Splitting the time steps into 4 has not been enough, aborting the iteration.')
                print()
                break

            run = 0
            
            dt2 = dt/k

            for m in range(k):
                temp = aux_fun_x_v_stat_e(particle, kind_map, params_map, dt2, p, Nel, breaks1, breaks2, breaks3, t1, t2, t3, indN1, indN2, indN3, indD1, indD2, indD3, loc1, loc2, loc3, weight1, weight2, weight3, nbase_n, nbase_d, e0_coeffs, eps, maxiter, cx, cy, cz)
                run = run + temp

        # write the results in the particles array
        particles[:,ip] = particle[:]
    
    #$ omp end parallel

    ierr = 0
