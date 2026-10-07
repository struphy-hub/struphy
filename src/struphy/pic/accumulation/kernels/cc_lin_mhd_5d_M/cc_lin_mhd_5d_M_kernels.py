"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "norm_b1", "filling_v")
def cc_lin_mhd_5d_M(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat11: "float[:,:,:,:,:,:]",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat22: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    mat33: "float[:,:,:,:,:,:]",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",  # model specific argument
    norm_b12: "float[:,:,:]",  # model specific argument
    norm_b13: "float[:,:,:]",  # model specific argument
    scale_vec: "float",  # model specific argument
):
    r"""Accumulation kernel for the propagator :class:`~struphy.propagators.propagators_fields.ShearAlfvenCurrentCoupling5D` and :class:`~struphy.propagators.propagators_fields.MagnetosonicCurrentCoupling5D`.

    Accumulates 2-form vector with the filling functions:

    .. math::

        B^\mu_p = \omega_p \mu_p\left(\sqrt{g}^{-1} \hat{\mathbf{b}}¹_0\right)_\mu \,.

    Parameters
    ----------

        norm_b11, norm_b12, norm_b13 : array[float]
            FE coefficients c_ijk of the normalized magnetic field as a 1-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    mu_idx = args_markers.mu_idx

    # allocate for a field evaluation
    norm_b1 = empty(3, dtype=float)

    # allocate for metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)

    # get number of markers
    n_markers_loc = shape(markers)[0]

    # -- removed omp: #$ omp parallel private(ip, boundary_cut, eta1, eta2, eta3, mu, weight, norm_b1, dfm, det_df, span1, span2, span3, norm_b11, norm_b12, norm_b13, filling_v)
    # -- removed omp: #$ omp for reduction ( + : vec1, vec2, vec3)

    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # marker weight and velocity
        weight = markers[ip, 5]
        mu = markers[ip, mu_idx]

        # b-field evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)

        det_df = linalg_kernels.det(dfm)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3, args_derham, norm_b11, norm_b12, norm_b13, norm_b1)

        filling_v[:] = weight * mu / det_df * scale_vec * norm_b1

        particle_to_mat_kernels.vec_fill_v2(
            args_derham,
            span1,
            span2,
            span3,
            vec1,
            vec2,
            vec3,
            filling_v[0],
            filling_v[1],
            filling_v[2],
        )
