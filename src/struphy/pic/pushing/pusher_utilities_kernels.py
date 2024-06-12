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
