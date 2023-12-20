from pyccel.decorators import pure, stack_array

from numpy import empty, sqrt, floor, zeros

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels


@stack_array('dfm', 'dfinv', 'dfinv_T', 'basis_normal', 'basis_normal_inv', 'norm_df', 'norm_dfinv_T', 'eta', 'eta_old', 'eta_boundary', 'v', 'v_logical', 'v_normal', 't')
def reflect(markers: 'float[:,:]',
            kind_map: 'int', params_map: 'float[:]',
            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
            outside_inds: 'int[:]', axis: 'int'):
    '''
    Reflect the particles which are pushed outside of the logical cube.

    Reflected particles' position:
    e.g. axis == 0

                                       |
                        o              |              o
          (1 - eta1%1, eta2, eta3)     |      (eta1, eta2, eta3)
                                       |

    Reflected particles' velocity:
    e.g. axis == 0

    normalized basis vectors normal to the plane which is spanned by axis 1 and 2
                   [DF^(-T)[0,0]/norm  DF[0,1]/norm  DF[0,2]/norm]
    basis_normal = [DF^(-T)[1,0]/norm  DF[1,1]/norm  DF[1,2]/norm]
                   [DF^(-T)[2,0]/norm  DF[2,1]/norm  DF[2,2]/norm]

    v_nomral     = basis_normal  x  v

    Reverse the v_normal, v_normal[0] = -v_normal[0]

    For the application, see `struphy.pic.particles.Particles6D.mpi_sort_markers` and `struphy.pic.particles.apply_kinetic_bc`.

    Parameters
    ----------
        markers : array[float]
            Local markers array

        domain.args_map : tuple of all needed mapping parameters
            kind_map, params_map, ..., cx, cy, cz

        outside_inds : array[int]
            inds indicate the particles which are pushed outside of the local cube

        axis : int
            0, 1 or 2
    '''

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinv_T = zeros((3, 3), dtype=float)
    basis_normal = zeros((3, 3), dtype=float)
    basis_normal_inv = zeros((3, 3), dtype=float)
    norm_df = empty(3, dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    eta_old = empty(3, dtype=float)
    eta_boundary = empty(3, dtype=float)
    v = empty(3, dtype=float)
    v_logical = empty(3, dtype=float)
    v_normal = empty(3, dtype=float)

    for ip in outside_inds:

        eta[:] = markers[ip, 0:3]
        eta_old[:] = markers[ip, 9:12]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta_old[0], eta_old[1], eta_old[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull back of the velocity
        linalg_kernels.matrix_vector(dfinv, v, v_logical)

        if eta[axis] > 1.:
            t = (1. - eta_old[axis])/v_logical[axis]
            eta_boundary[:] = eta_old + t*v_logical

            # assert allclose(eta_boundary[axis], 1.)

        else:
            t = (0. - eta_old[axis])/v[axis]
            eta_boundary[:] = eta_old + t*v_logical

            # assert allclose(eta_boundary[axis], 0.)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta_boundary[0], eta_boundary[1], eta_boundary[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_T)

        # assemble normalized basis which is normal to the reflection plane
        norm_df[0] = sqrt(dfm[0, 0]**2 + dfm[1, 0]**2 + dfm[2, 0]**2)
        norm_df[1] = sqrt(dfm[0, 1]**2 + dfm[1, 1]**2 + dfm[2, 1]**2)
        norm_df[2] = sqrt(dfm[0, 2]**2 + dfm[1, 2]**2 + dfm[2, 2]**2)

        norm_dfinv_T = sqrt(dfinv_T[0, axis]**2 +
                            dfinv_T[1, axis]**2 + dfinv_T[2, axis]**2)

        basis_normal[:] = dfm/norm_df
        basis_normal[:, axis] = dfinv_T[:, axis]/norm_dfinv_T

        linalg_kernels.matrix_inv(basis_normal, basis_normal_inv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(basis_normal_inv, v, v_normal)

        # reverse the velocity
        v_normal[axis] = -v_normal[axis]

        # push-forward of velocity
        linalg_kernels.matrix_vector(basis_normal, v_normal, v)

        # update the particle positions
        markers[ip, axis] = 1. - (markers[ip, axis]) % 1.

        # update the particle velocities
        markers[ip, 3:6] = v[:]


@pure
def quicksort(a: 'float[:]', lo: 'int', hi: 'int'):
    """
    Implementation of the quicksort sorting algorithm. Ref?

    Parameters
    ----------
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
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if lo < j:
            quicksort(a, lo, j)
        lo = i
        j = hi


def find_taus(eta: 'float', eta_next: 'float', Nel: 'int', breaks: 'float[:]', uniform: 'int', tau_list: 'float[:]'):
    """
    Find the values of tau for which the particle crosses the cell boundaries while going from eta to eta_next

    Parameters
    ----------
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

    if uniform == 1:
        index = int(floor(eta * Nel))
        index_next = int(floor(eta_next * Nel))
        length = int(abs(index_next - index))

        # break = eta / dx = eta * Nel

        for i in range(length):
            if index_next > index:
                tau_list[i] = (1.0 / Nel * (index + i + 1) -
                               eta) / (eta_next - eta)
            elif index > index_next:
                tau_list[i] = (eta - 1.0 / Nel * (index - i)) / \
                    (eta - eta_next)

    elif uniform == 0:
        # TODO
        print('Not implemented yet')

    else:
        print('ValueError, uniform must be 1 or 0 !')


@stack_array('Nel')
def aux_fun_x_v_stat_e(particle: 'float[:]',
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                       kind_map: 'int', params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       n_quad1: 'int', n_quad2: 'int', n_quad3: 'int',
                       dfm: 'float[:,:]', df_inv: 'float[:,:]',
                       bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]',
                       bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]',
                       taus: 'float[:]',
                       dt: 'float',
                       loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]',
                       weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]',
                       e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                       kappa: 'float',
                       eps: 'float[:]', maxiter: 'int') -> 'int':
    """
    Auxiliary function for the pusher_x_v_static_efield, introduced to enable time-step splitting if scheme does not converge for the standard dt

    Parameters
    ----------
    particle : array
        shape(7), contains the values for the positions [0:3], velocities [3:6], and weights [8]

    dt2 : double
        time stepping of substep

    loc1, loc2, loc3 : array
        contain the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in each direction

    weight1, weight2, weight3 : array
        contain the values of the weights for the Legendre-Gauss quadrature in each direction

    e1_1, e1_2, e1_3: array[float]
        3d array of FE coeffs of the background E-field as 1-form.

    eps: array
        determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

    maxiter : integer
        sets the maximum number of iterations for the iterative scheme
    """

    # Find number of elements in each direction
    Nel = empty(3, dtype=int)
    Nel[0] = len(tn1)
    Nel[1] = len(tn2)
    Nel[2] = len(tn3)

    # total number of basis functions : B-splines (pn) and D-splines (pd)
    pn1 = int(pn[0])
    pn2 = int(pn[1])
    pn3 = int(pn[2])

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

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
    evaluation_kernels.df(eta1, eta2, eta3, kind_map, params_map, t1_map, t2_map,
                          t3_map, p_map, ind1_map, ind2_map, ind3_map, cx, cy, cz, dfm)
    linalg_kernels.matrix_inv(dfm, df_inv)

    v1_curv = kappa * (df_inv[0, 0] * (v1_curr + v1) + df_inv[0, 1] *
                       (v2_curr + v2) + df_inv[0, 2] * (v3_curr + v3))
    v2_curv = kappa * (df_inv[1, 0] * (v1_curr + v1) + df_inv[1, 1] *
                       (v2_curr + v2) + df_inv[1, 2] * (v3_curr + v3))
    v3_curv = kappa * (df_inv[2, 0] * (v1_curr + v1) + df_inv[2, 1] *
                       (v2_curr + v2) + df_inv[2, 2] * (v3_curr + v3))

    eta1_next = (eta1 + dt * v1_curv / 2.) % 1
    eta2_next = (eta2 + dt * v2_curv / 2.) % 1
    eta3_next = (eta3 + dt * v3_curv / 2.) % 1

    # set some initial value for v_next
    v1_next = v1_curr
    v2_next = v2_curr
    v3_next = v3_curr

    runs = 0

    while abs(eta1_next - eta1_curr) > eps_pos or abs(eta2_next - eta2_curr) > eps_pos or abs(eta3_next - eta3_curr) > eps_pos or abs(v1_next - v1_curr) > eps_vel or abs(v2_next - v2_curr) > eps_vel or abs(v3_next - v3_curr) > eps_vel:
        taus[:] = 0.

        # update the positions and velocities
        eta1_curr = eta1_next
        eta2_curr = eta2_next
        eta3_curr = eta3_next

        v1_curr = v1_next
        v2_curr = v2_next
        v3_curr = v3_next

        # find Jacobian matrix
        evaluation_kernels.df((eta1_curr + eta1)/2, (eta2_curr + eta2)/2, (eta3_curr + eta3)/2,
                              kind_map, params_map,
                              t1_map, t2_map, t3_map,
                              p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz, dfm)

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # ======================================================================================
        # update the positions and place them back into the computational domain
        v1_curv = kappa * (df_inv[0, 0] * (v1_curr + v1) + df_inv[0, 1] *
                           (v2_curr + v2) + df_inv[0, 2] * (v3_curr + v3))
        v2_curv = kappa * (df_inv[1, 0] * (v1_curr + v1) + df_inv[1, 1] *
                           (v2_curr + v2) + df_inv[1, 2] * (v3_curr + v3))
        v3_curv = kappa * (df_inv[2, 0] * (v1_curr + v1) + df_inv[2, 1] *
                           (v2_curr + v2) + df_inv[2, 2] * (v3_curr + v3))

        # x_{n+1} = x_n + dt/2 * DF^{-1}(x_{n+1}/2 + x_n/2) * (v_{n+1} + v_n)
        eta1_next = (eta1 + dt * v1_curv / 2.) % 1
        eta2_next = (eta2 + dt * v2_curv / 2.) % 1
        eta3_next = (eta3 + dt * v3_curv / 2.) % 1

        # ======================================================================================
        # Compute tau-values in [0,1] for crossings of cell-boundaries

        index1 = int(floor(eta1_curr * Nel[0]))
        index1_next = int(floor(eta1_next * Nel[0]))
        length1 = int(abs(index1_next - index1))

        index2 = int(floor(eta2_curr * Nel[1]))
        index2_next = int(floor(eta2_next * Nel[1]))
        length2 = int(abs(index2_next - index2))

        index3 = int(floor(eta3_curr * Nel[2]))
        index3_next = int(floor(eta3_next * Nel[2]))
        length3 = int(abs(index3_next - index3))

        length = length1 + length2 + length3

        taus[0] = 0.0
        taus[length + 1] = 1.0

        tmp1 = taus[1:length1 + 1]
        find_taus(eta1_curr, eta1_next, Nel[0], tn1, 1, tmp1)
        taus[1:length1 + 1] = tmp1
        
        tmp2 = taus[length1 + 1:length1 + length2 + 1]
        find_taus(eta2_curr, eta2_next, Nel[1], tn2, 1, tmp2)
        taus[length1 + 1:length1 + length2 + 1] = tmp2
        
        tmp3 = taus[length1 + length2 + 1:length + 1]
        find_taus(eta3_curr, eta3_next, Nel[2], tn3, 1, tmp3)
        taus[length1 + length2 + 1:length + 1] = tmp3

        del tmp1, tmp2, tmp3

        if length != 0:
            tmp4 = taus[0:length + 1]
            quicksort(tmp4, 1, length)
            taus[0:length + 1] = tmp4
            del tmp4
        
        # ======================================================================================
        # update velocity in direction 1

        temp1 = 0.

        # loop over the cells
        for k in range(length + 1):

            a = eta1 + taus[k] * (eta1_curr - eta1)
            b = eta1 + taus[k + 1] * (eta1_curr - eta1)
            factor = (b - a) / 2
            adding = (a + b) / 2

            for n in range(n_quad1):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc1[n] + adding
                quad_pos3 = factor * loc1[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsplines_kernels.find_span(tn1, pn1, quad_pos1)
                span2 = bsplines_kernels.find_span(tn2, pn2, quad_pos2)
                span3 = bsplines_kernels.find_span(tn3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsplines_kernels.b_d_splines_slim(
                    tn1, pn1, quad_pos1, span1, bn1, bd1)
                bsplines_kernels.b_d_splines_slim(
                    tn2, pn2, quad_pos2, span2, bn2, bd2)
                bsplines_kernels.b_d_splines_slim(
                    tn3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - pn[0]
                ie2 = span2 - pn[1]
                ie3 = span3 - pn[2]

                # (DNN)
                for il1 in range(pd1 + 1):
                    i1 = ie1 + il1
                    bi1 = bd1[il1]
                    for il2 in range(pn2 + 1):
                        i2 = ie2 + il2
                        bi2 = bi1 * bn2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = ie3 + il3
                            bi3 = bi2 * bn3[il3] * e1_1[i1 - starts1[0] + pn1,
                                                        i2 - starts1[1] + pn2, i3 - starts1[2] + pn3]

                            temp1 += bi3 * weight1[n]

        # ======================================================================================
        # update velocity in direction 2

        temp2 = 0.

        # loop over the cells
        for k in range(length + 1):

            a = eta2 + taus[k] * (eta2_curr - eta2)
            b = eta2 + taus[k + 1] * (eta2_curr - eta2)
            factor = (b - a) / 2
            adding = (a + b) / 2

            for n in range(n_quad2):

                quad_pos1 = factor * loc2[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc2[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsplines_kernels.find_span(tn1, pn1, quad_pos1)
                span2 = bsplines_kernels.find_span(tn2, pn2, quad_pos2)
                span3 = bsplines_kernels.find_span(tn3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsplines_kernels.b_d_splines_slim(
                    tn1, pn1, quad_pos1, span1, bn1, bd1)
                bsplines_kernels.b_d_splines_slim(
                    tn2, pn2, quad_pos2, span2, bn2, bd2)
                bsplines_kernels.b_d_splines_slim(
                    tn3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - pn[0]
                ie2 = span2 - pn[1]
                ie3 = span3 - pn[2]

                # (NDN)
                for il1 in range(pn1 + 1):
                    i1 = ie1 + il1
                    bi1 = bn1[il1]
                    for il2 in range(pd2 + 1):
                        i2 = ie2 + il2
                        bi2 = bi1 * bd2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = ie3 + il3
                            bi3 = bi2 * bn3[il3] * e1_2[i1 - starts2[0] + pn1,
                                                        i2 - starts2[1] + pn2, i3 - starts2[2] + pn3]

                            temp2 += bi3 * weight2[n]

        # ======================================================================================
        # update velocity in direction 3

        temp3 = 0.

        # loop over the cells
        for k in range(length + 1):

            a = eta3 + taus[k] * (eta3_curr - eta3)
            b = eta3 + taus[k + 1] * (eta3_curr - eta3)
            factor = (b - a) / 2
            adding = (a + b) / 2

            for n in range(n_quad3):

                quad_pos1 = factor * loc3[n] + adding
                quad_pos2 = factor * loc3[n] + adding
                quad_pos3 = factor * loc3[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsplines_kernels.find_span(tn1, pn1, quad_pos1)
                span2 = bsplines_kernels.find_span(tn2, pn2, quad_pos2)
                span3 = bsplines_kernels.find_span(tn3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsplines_kernels.b_d_splines_slim(
                    tn1, pn1, quad_pos1, span1, bn1, bd1)
                bsplines_kernels.b_d_splines_slim(
                    tn2, pn2, quad_pos2, span2, bn2, bd2)
                bsplines_kernels.b_d_splines_slim(
                    tn3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - pn[0]
                ie2 = span2 - pn[1]
                ie3 = span3 - pn[2]

                # (NND)
                for il1 in range(pn1 + 1):
                    i1 = ie1 + il1
                    bi1 = bn1[il1]
                    for il2 in range(pn2 + 1):
                        i2 = ie2 + il2
                        bi2 = bi1 * bn2[il2]
                        for il3 in range(pd3 + 1):
                            i3 = ie3 + il3
                            bi3 = bi2 * bd3[il3] * e1_3[i1 - starts3[0] + pn1,
                                                        i2 - starts3[1] + pn2, i3 - starts3[2] + pn3]

                            temp3 += bi3 * weight3[n]

        # v_{n+1} = v_n + dt * DF^{-T}(x_n) * int_0^1 d tau ( E(x_n + tau*(x_{n+1} - x_n) ) )
        v1_next = v1 + dt * kappa * (df_inv[0, 0] * temp1 +
                                     df_inv[1, 0] * temp2 +
                                     df_inv[2, 0] * temp3)
        v2_next = v2 + dt * kappa * (df_inv[0, 1] * temp1 +
                                     df_inv[1, 1] * temp2 +
                                     df_inv[2, 1] * temp3)
        v3_next = v3 + dt * kappa * (df_inv[0, 2] * temp1 +
                                     df_inv[1, 2] * temp2 +
                                     df_inv[2, 2] * temp3)

        runs += 1

        if runs == maxiter:
            break

    if runs < maxiter:
        # print('For convergence this took runs:', runs)
        # print()
        runs = 0

    # write the results in the particle array and impose periodic boundary conditions on the particles by taking modulo 1
    particle[0] = eta1_next % 1
    particle[1] = eta2_next % 1
    particle[2] = eta3_next % 1
    particle[3] = v1_next
    particle[4] = v2_next
    particle[5] = v3_next

    return runs
