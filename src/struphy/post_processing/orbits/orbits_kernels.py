from numpy import abs, empty, log, pi, shape, sign, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.linear_algebra.linalg_kernels as linalg_kernels


@stack_array('x', 'v', 'B', 'unit_B', 'vperp', 'temp', 'Larmor_r')
def calculate_guiding_center_from_6d(
    markers: 'float[:,:]',
    B_cart: 'float[:,:]',
):
    r"""
    Calculate guiding center positions :math:`\overline{\mathbf X}` from 6d Cartesian phase-space coordinates :math:`(\mathbf x, \mathbf v)`:

    .. math::

        \overline{\mathbf X} = \mathbf x - \boldsymbol{\rho}_\textrm{L}(\mathbf x, \mathbf v) \,.

    where :math:`\boldsymbol{\rho}_\textrm{L}` is a normalized Larmor-radius vector defined as 

    .. math::

        \boldsymbol{\rho}_\textrm{L}(\mathbf x, \mathbf v) = \frac{1}{B_0(\mathbf x)} \mathbf{b}_0(\mathbf x) \times \mathbf v\,,

    where :math:`\mathbf{b}_0` is the unit magnetic field vector.
    """

    x = empty(3, dtype=float)
    v = empty(3, dtype=float)
    B = empty(3, dtype=float)
    unit_B = empty(3, dtype=float)
    vperp = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    Larmor_r = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # skip holes
        if (markers[ip, 0] == 0. and markers[ip, 1] == 0. and markers[ip, 2] == 0.):
            continue

        x = markers[ip, 0:3]
        v = markers[ip, 3:6]
        B = B_cart[ip, 0:3]

        # calculate magnitude of the magnetic field unit magnetic field
        absB = sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        unit_B = B/absB

        # calculate parallel velocity
        markers[ip, 3] = linalg_kernels.scalar_dot(v, unit_B)

        # calculate perpendicular velocity and magnetic moment
        linalg_kernels.cross(v, unit_B, temp)
        linalg_kernels.cross(unit_B, temp, vperp)

        vperp_square = sqrt(vperp[0]**2 + vperp[1]**2 + vperp[2]**2)

        markers[ip, 4] = vperp_square
        markers[ip, 5] = vperp_square**2/(2*absB)

        # calculate unit Larmor radius vector
        linalg_kernels.cross(unit_B, vperp, Larmor_r)
        Larmor_r /= absB

        # calculate guiding center positions
        markers[ip, 0:3] = x - Larmor_r
