'Filtering kernels'


from pyccel.decorators import stack_array

from numpy import zeros, empty, shape, ones

@stack_array('vec_copy1','vec_copy2', 'vec_copy3', 'mask1d', 'mask')
def apply_three_points_filter(vec1: 'float[:,:,:]',
                              vec2: 'float[:,:,:]',
                              vec3: 'float[:,:,:]',
                              pn: 'int[:]', starts: 'int[:]', ends: 'int[:]',
                              alpha: 'float',
                              repeat: 'int'):
    """
    Applying three point filter to the vector.
    """

    # allocate memory
    vec_copy1 = empty((shape(vec1)), dtype=float)
    vec_copy2 = empty((shape(vec2)), dtype=float)
    vec_copy3 = empty((shape(vec3)), dtype=float)

    mask1d = zeros(3, dtype=float)
    mask = ones((3,3), dtype=float)

    tmp = zeros(3, dtype=float)

    # copy vectors
    vec_copy1[:] = vec1[:]
    vec_copy2[:] = vec2[:]
    vec_copy3[:] = vec3[:]

    # filtering mask
    mask1d[0] = 1/2 * (1 - alpha)
    mask1d[1] = alpha
    mask1d[2] = 1/2 * (1 - alpha)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                mask[i,j,k] *= mask1d[i]*mask1d[j]*mask1d[k]

    for c in range(repeat):

        for i in range(ends[0] + 1 - starts[0]):
            for j in range(ends[1] + 1 - starts[1]):
                for k in range(ends[2] + 1 - starts[2]):

                    for il in range(3):
                        for jl in range(3):
                            for kl in range(3):

                                tmp[0] = mask[il,jl,kl] * vec_copy1[pn[0] + i + il -1, pn[1] + j + jl -1, pn[2] + k + kl -1]
                                tmp[1] = mask[il,jl,kl] * vec_copy2[pn[0] + i + il -1, pn[1] + j + jl -1, pn[2] + k + kl -1]
                                tmp[2] = mask[il,jl,kl] * vec_copy3[pn[0] + i + il -1, pn[1] + j + jl -1, pn[2] + k + kl -1]
                    
                    vec1[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[0]
                    vec2[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[1]
                    vec3[pn[0]+i, pn[1]+j, pn[2]+k] = tmp[2]