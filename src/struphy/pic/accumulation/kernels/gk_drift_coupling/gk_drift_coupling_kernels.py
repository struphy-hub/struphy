"Accumulation kernel for gyro-center (5D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.accumulation.particle_to_mat_kernels as particle_to_mat_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "unit_b1", "grad_b", "grad_b_x_b", "filling_v")
def gk_drift_coupling(
    args_markers: "MarkerArguments",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    mat12: "float[:,:,:,:,:,:]",
    mat13: "float[:,:,:,:,:,:]",
    mat23: "float[:,:,:,:,:,:]",
    vec1: "float[:,:,:]",
    vec2: "float[:,:,:]",
    vec3: "float[:,:,:]",
    epsilon: float,
    scale: float,
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
):
    r"""Accumulation kernel for :class:`~struphy.propagators.gyrokinetic_maxwell_zlr.GyroDriftCoupling`.

    Accumulates into V1 (anti-symmetric block matrix and vector) with the filling functions

    .. math::

        A_p^{\mu, \nu} = s\, w_p \frac{\epsilon_{\mu\nu\lambda} \hat b^1_{0,\lambda}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p) \,,
        \qquad
        B_p^\mu = s\, w_p\, \dot{\boldsymbol \eta}_{\nabla B, p} = - s\, w_p\, \varepsilon \mu_p \frac{(\hat \nabla \hat B_\parallel \times \hat{\mathbf b}^1_0)_\mu}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p) \,,

    such that :math:`A_p \hat{\mathbf e}^1 = s\, w_p\, (\hat{\mathbf e}^1 \times \hat{\mathbf b}^1_0) / (\sqrt g \hat B^*_\parallel)`
    is the logical :math:`E \times B` velocity (times the weight and the scaling :math:`s`),
    and :math:`\hat B^*_\parallel = \hat B_\parallel + \varepsilon v_\parallel \widehat{[(\nabla \times \mathbf b_0) \cdot \mathbf b_0]}`.

    Parameters
    ----------
        epsilon : float
            Normalized cyclotron period of the species.

        scale : float
            Scaling of the current of the species (:math:`\alpha^2/\varepsilon`).

        unit_b1_1, unit_b1_2, unit_b1_3 : array[float]
            FE coefficients c_ijk of the equilibrium unit magnetic field as a 1-form.

        grad_b_full_1, grad_b_full_2, grad_b_full_3 : array[float]
            FE coefficients c_ijk of the gradient of the parallel magnetic field (equilibrium + perturbation) as a 1-form.

        B_dot_b_coeffs : array[float]
            FE coefficients c_ijk of the parallel magnetic field (equilibrium + perturbation) as a 0-form.

        curl_unit_b_dot_b0 : array[float]
            FE coefficients c_ijk of (curl b_0) . b_0 as a 0-form.

    Note
    ----
        The above parameter list contains only the model specific input arguments.
    """

    markers = args_markers.markers
    mu_idx = args_markers.mu_idx

    # allocate for metric coefficients
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    unit_b1 = empty(3, dtype=float)
    grad_b = empty(3, dtype=float)
    grad_b_x_b = empty(3, dtype=float)

    # allocate for filling
    filling_v = empty(3, dtype=float)

    # get local number of markers
    n_markers_loc = shape(markers)[0]

    # fmt: off
    #$ omp parallel private(b_star_para, B_dot_b, det_df, dfm, eta1, eta2, eta3, fac, filling_m12, filling_m13, filling_m23, filling_v, grad_b, grad_b_x_b, ip, mu, span1, span2, span3, thread_derham, unit_b1, v, weight)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for reduction ( + : mat12, mat13, mat23, vec1, vec2, vec3)
    # fmt: on
    for ip in range(n_markers_loc):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        # marker positions, parallel velocity, magnetic moment and weight
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        v = markers[ip, 3]
        mu = markers[ip, mu_idx]
        weight = markers[ip, 5]

        # evaluate Jacobian determinant
        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # unit b and grad B_parallel; 1-forms
        eval_1form_spline_mpi(span1, span2, span3, thread_derham, unit_b1_1, unit_b1_2, unit_b1_3, unit_b1)
        eval_1form_spline_mpi(span1, span2, span3, thread_derham, grad_b_full_1, grad_b_full_2, grad_b_full_3, grad_b)

        # B*_parallel (times det DF)
        B_dot_b = eval_0form_spline_mpi(span1, span2, span3, thread_derham, B_dot_b_coeffs)
        b_star_para = eval_0form_spline_mpi(span1, span2, span3, thread_derham, curl_unit_b_dot_b0)
        b_star_para *= epsilon * v
        b_star_para += B_dot_b
        b_star_para *= det_df

        # matrix of a -> a x b (anti-symmetric): M_12 = b_3, M_13 = -b_2, M_23 = b_1
        fac = scale * weight / b_star_para

        filling_m12 = fac * unit_b1[2]
        filling_m13 = -fac * unit_b1[1]
        filling_m23 = fac * unit_b1[0]

        # grad-B drift velocity: -epsilon mu (grad B x b) / (sqrt g B*_parallel)
        linalg_kernels.cross(grad_b, unit_b1, grad_b_x_b)
        filling_v[:] = -fac * epsilon * mu * grad_b_x_b

        # call the appropriate matvec filler
        particle_to_mat_kernels.m_v_fill_v1_asym(
            thread_derham,
            span1,
            span2,
            span3,
            mat12,
            mat13,
            mat23,
            filling_m12,
            filling_m13,
            filling_m23,
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
