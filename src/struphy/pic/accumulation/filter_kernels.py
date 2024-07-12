'Filtering kernels'


from pyccel.decorators import stack_array

from numpy import zeros, empty, shape, ones

@stack_array('vec_copy1','vec_copy2', 'vec_copy3', 'mask1d', 'mask', 'top', 'i_bottom', 'i_top', 'fi', 'ir')
def apply_three_points_filter(vec1: 'float[:,:,:]',
                              vec2: 'float[:,:,:]',
                              vec3: 'float[:,:,:]',
                              Nel: 'int[:]', spl_kind: 'bool[:]',
                              pn: 'int[:]', starts: 'int[:]', ends: 'int[:]',
                              alpha: 'float'):
    """
    Applying three point filter to the vector.
    """

    # allocate memory
    vec_copy1 = empty((shape(vec1)), dtype=float)
    vec_copy2 = empty((shape(vec2)), dtype=float)
    vec_copy3 = empty((shape(vec3)), dtype=float)

    mask1d = zeros(3, dtype=float)
    mask = ones((3,3,3), dtype=float)

    tmp = zeros(3, dtype=float)
    top = empty(3, dtype=int)
    i_bottom = zeros(3, dtype=int)
    i_top = zeros(3, dtype=int)
    fi = empty(3, dtype=int)
    ir = empty(3, dtype=int)

    # copy vectors
    vec_copy1[:,:,:] = vec1[:,:,:]
    vec_copy2[:,:,:] = vec2[:,:,:]
    vec_copy3[:,:,:] = vec3[:,:,:]

    # filtering mask
    mask1d[0] = 1/2 * (1 - alpha)
    mask1d[1] = alpha
    mask1d[2] = 1/2 * (1 - alpha)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                mask[i,j,k] *= mask1d[i]*mask1d[j]*mask1d[k]

    # consider left and right boundary
    for i in range(3):
        if spl_kind[i]: 
            top[i] = Nel[i] - 1
        else:
            top[i] = Nel[i] + pn[i] - 1

    for i in range(3):
        if starts[i] == 0:
            if spl_kind[i]:
                i_bottom[i] = -1
            else:
                i_bottom[i] = +1

        if ends[i] == top[i]:
            if spl_kind[i]:
                i_top[i] = +1
            else:
                i_top[i] = -1

    # index range
    for i in range(3):
        ir[i] = ends[i] + 1 + starts[i]

    # filtering
    for i in range(ir[0]):
        for j in range(ir[1]):
            for k in range(ir[2]):

                tmp[:] = 0.

                for il in range(3):
                    for jl in range(3):
                        for kl in range(3):

                            fi[0] = pn[0] + i + il -1
                            fi[1] = pn[1] + j + jl -1
                            fi[2] = pn[2] + k + kl -1

                            # if i == 0 and il == 0: fi[0] += i_bottom[0]
                            # if j == 0 and jl == 0: fi[1] += i_bottom[1]
                            # if k == 0 and kl == 0: fi[2] += i_bottom[2]

                            # if i == ir[0]-1 and il == 2: fi[0] += i_top[0]
                            # if j == ir[1]-1 and jl == 2: fi[1] += i_top[1]
                            # if k == ir[2]-1 and kl == 2: fi[2] += i_top[2]

                            tmp[0] += mask[il,jl,kl] * vec_copy1[fi[0], fi[1], fi[2]]
                            tmp[1] += mask[il,jl,kl] * vec_copy2[fi[0], fi[1], fi[2]]
                            tmp[2] += mask[il,jl,kl] * vec_copy3[fi[0], fi[1], fi[2]]
                
                vec1[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[0]
                vec2[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[1]
                vec3[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[2]