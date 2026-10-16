"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("d1", "d2")
def gk_mag_density_gyro(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    vec: "float[:,:,:]",
    scale: float,
    epsilon: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    absB0: "float[:,:,:]",
):
    r"""Kernel for :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` into V0 with the
    disc-averaged filling (the transpose of the gyrocentre energy :math:`\mu \langle\langle B_{1,\parallel} \rangle\rangle`)

    .. math::

        B_p = s\, \mu_p w_p \langle\langle \Lambda^0 \rangle\rangle_p \,.

    Parameters
    ----------
        scale : float
            Scaling factor (:math:`\alpha^2`).

        epsilon, n_gyro, disc_s, disc_w, unit_b1_1, unit_b1_2, unit_b1_3, absB0 :
            Gyro-averaging arguments, see :mod:`struphy.pic.gyroaverage_kernels`.
    """

    markers = args_markers.markers
    mu_idx = args_markers.mu_idx

    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)

    for ip in range(shape(markers)[0]):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        weight = markers[ip, 5]
        mu = markers[ip, mu_idx]

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)
        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta1, eta2, eta3, args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        filling = mu * weight * scale

        gyroaverage_kernels.deposit_0form_gyro(
            eta1, eta2, eta3, d1, d2, rho, n_gyro, disc_s, disc_w, args_derham, vec, filling
        )
