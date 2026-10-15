"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_2form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "b2", "b_star", "c", "filling_m", "filling_v")
def gk_epar_coupling(
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
    epsilon: float,
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
):
    r"""Accumulation kernel for :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroEparCoupling`.

    Accumulates into V1 (symmetric block matrix and vector) with the filling functions

    .. math::

        A_p^{\mu, \nu} = w_p\, c_{p,\mu}\, c_{p,\nu} \,, \qquad B_p^\mu = w_p\, v_{\parallel,p}\, c_{p,\mu} \,,
        \qquad \mathbf c_p = \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p)\,,

    where :math:`\hat{\mathbf B}^{*2} = \hat{\mathbf B}^2 + \varepsilon v_\parallel \hat\nabla \times \hat{\mathbf b}^1_0`
    and :math:`\hat B^*_\parallel = \hat B_\parallel + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}`.
    The scalings :math:`\alpha^2/\varepsilon` are applied in the propagator.

    Parameters
    ----------
        epsilon : float
            Normalized cyclotron period of the species.

        b2_1, b2_2, b2_3 : array[float]
            FE coefficients c_ijk of the full magnetic field (equilibrium + perturbation) as a 2-form.

        curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3 : array[float]
            FE coefficients c_ijk of the curl of the equilibrium unit magnetic field as a 2-form.

        B_dot_b_coeffs : array[float]
            FE coefficients c_ijk of the parallel magnetic field (equilibrium + perturbation) as a 0-form.

        curl_unit_b_dot_b0 : array[float]
            FE coefficients c_ijk of (curl b_0) . b_0 as a 0-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    c = empty(3, dtype=float)

    # allocate for filling
    filling_m = empty((3, 3), dtype=float)
    filling_v = empty(3, dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    # fmt: off
    #$ omp parallel private(b2, b_star, b_star_para, B_dot_b, c, det_df, dfm, eta1, eta2, eta3, filling_m, filling_v, ip, span1, span2, span3, thread_derham, v, weight)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    # fmt: on
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, parallel velocity and weight
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v = markers[ip, 3]
        weight = markers[ip, 5]

        # evaluate Jacobian determinant
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # B*; 2-form
        eval_2form_spline_mpi(span1, span2, span3, thread_derham, b2_1, b2_2, b2_3, b2)
        eval_2form_spline_mpi(
            span1, span2, span3, thread_derham, curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3, b_star
        )
        b_star *= epsilon * v
        b_star += b2

        # B*_parallel (times det DF)
        B_dot_b = eval_0form_spline_mpi(span1, span2, span3, thread_derham, B_dot_b_coeffs)
        b_star_para = eval_0form_spline_mpi(span1, span2, span3, thread_derham, curl_unit_b_dot_b0)
        b_star_para *= epsilon * v
        b_star_para += B_dot_b
        b_star_para *= det_df

        # c_p
        c[:] = b_star / b_star_para

        # filling functions
        linalg_kernels.outer(c, c, filling_m)
        filling_m *= weight
        filling_v[:] = weight * v * c

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_symm(
            thread_derham,
            span1,
            span2,
            span3,
            mat11,
            mat12,
            mat13,
            mat22,
            mat23,
            mat33,
            filling_m[0, 0],
            filling_m[0, 1],
            filling_m[0, 2],
            filling_m[1, 1],
            filling_m[1, 2],
            filling_m[2, 2],
            vec1,
            vec2,
            vec3,
            filling_v[0],
            filling_v[1],
            filling_v[2],
        )

    # fmt: off
    #$ omp end parallel
    # fmt: on
