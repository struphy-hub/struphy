from numpy import empty, sqrt, floor

import struphy.linear_algebra.core as linalg


def reflect(df : 'float[:,:]', df_inv : 'float[:,:]', v : 'float[:]'):
    '''TODO'''

    vg        = empty( 3    , dtype=float)

    basis     = empty((3, 3), dtype=float)
    basis_inv = empty((3, 3), dtype=float)


    # calculate normalized basis vectors
    norm1 = sqrt(df_inv[0, 0]**2 + df_inv[0, 1]**2 + df_inv[0, 2]**2)

    norm2 = sqrt(df[0, 1]**2 + df[1, 1]**2 + df[2, 1]**2)
    norm3 = sqrt(df[0, 2]**2 + df[1, 2]**2 + df[2, 2]**2)

    basis[:, 0] = df_inv[0, :]/norm1

    basis[:, 1] = df[:, 1]/norm2
    basis[:, 2] = df[:, 2]/norm3

    linalg.matrix_inv(basis, basis_inv)

    linalg.matrix_vector(basis_inv, v, vg)

    vg[0] = -vg[0]

    linalg.matrix_vector(basis, vg, v)


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