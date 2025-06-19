from numpy import shape, zeros
from pyccel.decorators import stack_array

# import modules for B-spline evaluation
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_2d as evaluation_kernels_2d
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d

# import module for mapping evaluation
import struphy.geometry.evaluation_kernels as evaluation_kernels

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.linalg_kernels as linalg_kernels


@stack_array("e", "v")
def set_particles_symmetric_3d_3v(numbers: "float[:,:]", markers: "float[:,:]"):
    e = zeros(3, dtype=float)
    v = zeros(3, dtype=float)

    np = 64 * shape(numbers)[0]

    for i_part in range(np):
        ip = i_part % 64

        if ip == 0:
            e[:] = numbers[int(i_part / 64), 0:3]
            v[:] = numbers[int(i_part / 64), 3:6]

        elif ip % 32 == 0:
            v[2] = 1 - v[2]

        elif ip % 16 == 0:
            v[1] = 1 - v[1]

        elif ip % 8 == 0:
            v[0] = 1 - v[0]

        elif ip % 4 == 0:
            e[2] = 1 - e[2]

        elif ip % 2 == 0:
            e[1] = 1 - e[1]

        else:
            e[0] = 1 - e[0]

        markers[i_part, 0:3] = e
        markers[i_part, 3:6] = v


@stack_array("e", "v")
def set_particles_symmetric_2d_3v(numbers: "float[:,:]", markers: "float[:,:]"):
    e = zeros(2, dtype=float)
    v = zeros(3, dtype=float)

    np = 32 * shape(numbers)[0]

    for i_part in range(np):
        ip = i_part % 32

        if ip == 0:
            e[:] = numbers[int(i_part / 32), 0:2]
            v[:] = numbers[int(i_part / 32), 2:5]

        elif ip % 16 == 0:
            v[2] = 1 - v[2]

        elif ip % 8 == 0:
            v[1] = 1 - v[1]

        elif ip % 4 == 0:
            v[0] = 1 - v[0]

        elif ip % 2 == 0:
            e[1] = 1 - e[1]

        else:
            e[0] = 1 - e[0]

        markers[i_part, 1:3] = e
        markers[i_part, 3:6] = v


def tile_int_kernel(
    fun: "float[:,:,:]",
    x_wts: "float[:]",
    y_wts: "float[:]",
    z_wts: "float[:]",
    out: "float[:,:,:]",
):
    """Compute integrals over all tiles in a single sorting box.

    Parameters
    ----------
    fun: np.ndarray
        The integrand evaluated at the quadrature points (meshgrid).

    x_wts, y_wts, z_wts: np.ndarray
        Quadrature weights for tile integral.

    out: np.ndarray
        The result holding all tile integrals in one sorting box."""

    _shp = shape(out)
    nq_x = shape(x_wts)[0]
    nq_y = shape(y_wts)[0]
    nq_z = shape(z_wts)[0]
    for i in range(_shp[0]):
        for j in range(_shp[1]):
            for k in range(_shp[2]):
                out[i, j, k] = 0.0
                for iq in range(nq_x):
                    for jq in range(nq_y):
                        for kq in range(nq_z):
                            out[i, j, k] += (
                                fun[i * nq_x + iq, j * nq_y + jq, k * nq_z + kq] * x_wts[iq] * y_wts[jq] * z_wts[kq]
                            )
