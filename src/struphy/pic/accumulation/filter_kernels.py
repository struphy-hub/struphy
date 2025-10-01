"Filtering kernels"

from numpy import empty, ones, shape, zeros
from pyccel.decorators import stack_array


@stack_array("vec_copy", "mask1d", "mask", "top", "i_bottom", "i_top", "fi", "ir")
def apply_three_point_filter(
    vec: "float[:,:,:]",
    Nel: "int[:]",
    spl_kind: "bool[:]",
    pn: "int[:]",
    starts: "int[:]",
    ends: "int[:]",
    alpha: "float",
):
    r"""
    Applying three point filter to the spline coefficients of the accumulated vector (``._data`` of the StencilVector):

    .. math::

        v^{\textnormal{filtered}}_{i,j,k} = \sum^2_{l_1=0} \sum^2_{l_2=0} \sum^2_{l_3=0} S(l_1) S(l_2) S(l_3)\, v_{i-1+l_1,\, j-1+l_2,\, k-1+l_3} \,,

    where the 1d mask :math:`S` is defined as

    .. math::

        S(i) = \left\{
        \begin{aligned}
        &(1 - \alpha)/2 \quad && \text{if} \quad i=0
        \\
        &\alpha && \text{if} \quad i=1
        \\
        &(1 - \alpha)/2 \quad && \text{if} \quad i=2
        \end{aligned}
        \right. \,.
    """

    # allocate memory
    vec_copy = empty((shape(vec)), dtype=float)

    mask1d = zeros(3, dtype=float)
    mask = ones((3, 3, 3), dtype=float)

    top = empty(3, dtype=int)
    i_bottom = zeros(3, dtype=int)
    i_top = zeros(3, dtype=int)
    fi = empty(3, dtype=int)
    ir = empty(3, dtype=int)

    # copy vectors
    vec_copy[:, :, :] = vec[:, :, :]

    # filtering mask
    mask1d[0] = 1 / 2 * (1 - alpha)
    mask1d[1] = alpha
    mask1d[2] = 1 / 2 * (1 - alpha)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                mask[i, j, k] *= mask1d[i] * mask1d[j] * mask1d[k]

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
        ir[i] = ends[i] + 1 - starts[i]

    # filtering
    for i in range(ir[0]):
        for j in range(ir[1]):
            for k in range(ir[2]):
                tmp = 0.0

                for il in range(3):
                    for jl in range(3):
                        for kl in range(3):
                            fi[0] = pn[0] + i + il - 1
                            fi[1] = pn[1] + j + jl - 1
                            fi[2] = pn[2] + k + kl - 1

                            if i == 0 and il == 0:
                                fi[0] += i_bottom[0]
                            if j == 0 and jl == 0:
                                fi[1] += i_bottom[1]
                            if k == 0 and kl == 0:
                                fi[2] += i_bottom[2]

                            if i == ir[0] - 1 and il == 2:
                                fi[0] += i_top[0]
                            if j == ir[1] - 1 and jl == 2:
                                fi[1] += i_top[1]
                            if k == ir[2] - 1 and kl == 2:
                                fi[2] += i_top[2]

                            tmp += mask[il, jl, kl] * vec_copy[fi[0], fi[1], fi[2]]

                vec[pn[0] + i, pn[1] + j, pn[2] + k] = tmp
