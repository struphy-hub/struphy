"Diagnostics kernel for gyro-center (5D) particles."

from numpy import empty, mod, shape
from pyccel.decorators import stack_array

import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.gyroaverage_kernels as gyroaverage_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("eta", "d1", "d2")
def gk_energy_gyro(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    output_idx: int,
    mu_idx: int,
    epsilon: float,
    n_gyro: int,
    disc_s: "float[:]",
    disc_w: "float[:]",
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    absB0: "float[:,:,:]",
    B_tilde_par: "float[:,:,:]",
):
    r"""Evaluate the magnetic-moment energy :math:`\mu_p (|B_0| + \langle\langle \tilde B_\parallel \rangle\rangle)(\boldsymbol \eta_p)`
    of each marker and store it at ``markers[:, output_idx]``."""

    eta = empty(3, dtype=float)
    d1 = empty(3, dtype=float)
    d2 = empty(3, dtype=float)

    for ip in range(shape(markers)[0]):
        if markers[ip, 0] == -1.0:
            continue

        eta[:] = mod(markers[ip, 0:3], 1.0)
        mu = markers[ip, mu_idx]

        span1, span2, span3 = get_spans(eta[0], eta[1], eta[2], args_derham)
        b0 = eval_0form_spline_mpi(span1, span2, span3, args_derham, absB0)

        rho = gyroaverage_kernels.gyro_radius(epsilon, mu, b0)
        if n_gyro > 0 and rho > 0.0:
            gyroaverage_kernels.gyro_frame(
                eta[0], eta[1], eta[2], args_domain, args_derham, unit_b1_1, unit_b1_2, unit_b1_3, d1, d2
            )

        b_par = b0 + gyroaverage_kernels.eval_0form_gyro(
            eta[0], eta[1], eta[2], d1, d2, rho, n_gyro, disc_s, disc_w, args_derham, B_tilde_par
        )

        markers[ip, output_idx] = mu * b_par
