"""Kernel banded_matvec_3d: product of a banded 3d matrix, stored per owned row, with a Stencil vector."""

from numpy import shape


def banded_matvec_3d(
    p1: int,
    p2: int,
    p3: int,
    g1: int,
    g2: int,
    g3: int,
    mat: "float[:,:,:,:,:,:]",
    x: "float[:,:,:]",
    out: "float[:,:,:]",
):
    r"""Matrix-vector product :math:`y_{\mathbf i} = \sum_{\mathbf k} A_{\mathbf i, \mathbf k}\, x_{\mathbf i + \mathbf k - \mathbf p}`
    for a matrix with the band :math:`|j_d - i_d| \leq p_d` in each direction, stored per owned row as
    ``mat[i1, i2, i3, k1, k2, k3]`` (column :math:`j_d = i_d + k_d - p_d`).

    Parameters
    ----------
    p1, p2, p3 : int
        Half band widths in direction 1, 2 and 3.
    g1, g2, g3 : int
        Widths of the ghost regions of ``x`` and ``out`` (pads * shifts of the Stencil space), at least the half band
        widths.
    mat : "float[:,:,:,:,:,:]"
        Matrix entries of the owned rows, indexed by [row 1, row 2, row 3, band 1, band 2, band 3].
    x : "float[:,:,:]"
        _data array of the input StencilVector (ghost regions up to date).
    out : "float[:,:,:]"
        _data array of the output StencilVector; only the owned entries are written.
    """
    n1 = shape(mat)[0]
    n2 = shape(mat)[1]
    n3 = shape(mat)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0
                for k1 in range(2 * p1 + 1):
                    for k2 in range(2 * p2 + 1):
                        for k3 in range(2 * p3 + 1):
                            value += (
                                mat[i1, i2, i3, k1, k2, k3] * x[g1 + i1 + k1 - p1, g2 + i2 + k2 - p2, g3 + i3 + k3 - p3]
                            )
                out[g1 + i1, g2 + i2, g3 + i3] = value
