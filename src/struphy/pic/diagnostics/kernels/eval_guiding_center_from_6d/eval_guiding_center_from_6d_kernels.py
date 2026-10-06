"""Evaluate guiding center phase space of each particles:."""

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_0form_spline_mpi, eval_2form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("v", "dfm", "b2", "norm_b_cart", "temp", "v_perp", "Larmor_r")
def eval_guiding_center_from_6d(
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    first_diagnostics_idx: int,
    epsilon: float,
    b21: "float[:,:,:]",
    b22: "float[:,:,:]",
    b23: "float[:,:,:]",
    absB: "float[:,:,:]",
):
    r"""
    Evaluate guiding center phase space of each particles:
    markers[ip, first_diagnostics_idx: first_diagnostics_idx+3] : logical guiding center positions
    markers[ip, first_diagnostics_idx + 4] :  magnetic moment
    markers[ip, first_diagnostics_idx + 6] :  parallel velocity
    """

    v = empty(3, dtype=float)
    dfm = empty((3, 3), dtype=float)
    b2 = empty(3, dtype=float)
    norm_b_cart = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    v_perp = empty(3, dtype=float)
    Larmor_r = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        x = markers[ip, first_diagnostics_idx]
        y = markers[ip, first_diagnostics_idx + 1]
        z = markers[ip, first_diagnostics_idx + 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # magnetic field; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b21,
            b22,
            b23,
            b2,
        )

        # magnitude of the magnetic field; 0form
        abs_B = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            absB,
        )

        # calculate normalized magnetic filed; cartesian
        b2 /= abs_B
        linalg_kernels.matrix_vector(dfm, b2, norm_b_cart)
        norm_b_cart /= det_df

        # calculate parallel velocity
        v_parallel = linalg_kernels.scalar_dot(norm_b_cart, v)

        # extract perpendicular velocity
        linalg_kernels.cross(v, norm_b_cart, temp)
        linalg_kernels.cross(norm_b_cart, temp, v_perp)

        v_perp_square = v_perp[0] ** 2 + v_perp[1] ** 2 + v_perp[2] ** 2

        # parallel velocity
        markers[ip, first_diagnostics_idx + 6] = v_parallel

        # magnetic moment
        markers[ip, first_diagnostics_idx + 4] = 1 / 2 * v_perp_square / abs_B

        # calculate Larmor radius vector
        linalg_kernels.cross(norm_b_cart, v_perp, Larmor_r)
        Larmor_r /= abs_B
        Larmor_r *= epsilon

        # calculate cartesian guiding center positions
        markers[ip, first_diagnostics_idx] = x - Larmor_r[0]
        markers[ip, first_diagnostics_idx + 1] = y - Larmor_r[1]
        markers[ip, first_diagnostics_idx + 2] = z - Larmor_r[2]
