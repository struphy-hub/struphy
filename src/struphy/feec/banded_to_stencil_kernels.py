from numpy import mod, shape
from pyccel.decorators import pure


@pure
def band_to_stencil_1d(arr: "float[:, :]", out: "float[:, :]"):
    """Converts the 2d banded matrix arr, of shape (n, m) and with 2*p + 1 < m bands centered around the diagonal,
    into the array out, of shape (n, 2*p + 1), which corresponds to the StencilMatrix format.
    """
    s = shape(arr)
    p = shape(out)[1] // 2

    for i in range(s[0]):
        for j in range(2 * p + 1):
            jj = mod(i - p + j, s[1])
            out[i, j] = arr[i, jj]


@pure
def band_to_stencil_2d(arr: "float[:, :, :, :]", out: "float[:, :, :, :]"):
    """Converts a 4d banded matrix to StencilMatrix format (see band_to_stencil_1d)."""
    s = shape(arr)
    p1 = shape(out)[2] // 2
    p2 = shape(out)[3] // 2

    for i1 in range(s[0]):
        for j1 in range(2 * p1 + 1):
            jj1 = mod(i1 - p1 + j1, s[2])

            for i2 in range(s[1]):
                for j2 in range(2 * p2 + 1):
                    jj2 = mod(i2 - p2 + j2, s[3])
                    out[i1, i2, j1, j2] = arr[i1, i2, jj1, jj2]


@pure
def band_to_stencil_3d(arr: "float[:, :, :, :, :, :]", out: "float[:, :, :, :, :, :]"):
    """Converts a 6d banded matrix to StencilMatrix format (see band_to_stencil_1d)."""
    s = shape(arr)
    p1 = shape(out)[3] // 2
    p2 = shape(out)[4] // 2
    p3 = shape(out)[5] // 2

    for i1 in range(s[0]):
        for j1 in range(2 * p1 + 1):
            jj1 = mod(i1 - p1 + j1, s[3])

            for i2 in range(s[1]):
                for j2 in range(2 * p2 + 1):
                    jj2 = mod(i2 - p2 + j2, s[4])

                    for i3 in range(s[2]):
                        for j3 in range(2 * p3 + 1):
                            jj3 = mod(i3 - p3 + j3, s[5])
                            out[i1, i2, i3, j1, j2, j3] = arr[i1, i2, i3, jj1, jj2, jj3]
