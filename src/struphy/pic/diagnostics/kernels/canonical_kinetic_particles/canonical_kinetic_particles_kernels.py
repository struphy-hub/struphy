"""Calculate kinetic energy of each particle and sum up the result."""

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@stack_array("dfm", "dfinv", "dfinv_t", "v", "a_form", "dfta_form")
def canonical_kinetic_particles(
    res: "float[:]",
    markers: "float[:,:]",
    args_derham: "DerhamArguments",
    args_domain: "DomainArguments",
    a1_1: "float[:,:,:]",
    a1_2: "float[:,:,:]",
    a1_3: "float[:,:,:]",
):
    r"""
    Calculate kinetic energy of each particle and sum up the result.

    .. math::

        \frac{1}{2} \sum_p w_p |{\mathbf p} -  \hat{\mathbf A}^1({\boldsymbol \eta}_p)|^2.
    """

    res[:] = 0.0
    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form components)
    a_form = empty(3, dtype=float)
    dfta_form = empty(3, dtype=float)
    # particle position and velocity
    v = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]
    # fmt: off
    #$ omp parallel private (ip, v, w, dfm, dfinv, dfinv_t, span1, span2, span3, a_form, dfta_form)
    #$ omp for reduction( + : res)
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]
        w = markers[ip, 6]
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # vector potential: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            a1_1,
            a1_2,
            a1_3,
            a_form,
        )

        dfta_form[0] = dfinv_t[0, 0] * a_form[0] + dfinv_t[0, 1] * a_form[1] + dfinv_t[0, 2] * a_form[2]
        dfta_form[1] = dfinv_t[1, 0] * a_form[0] + dfinv_t[1, 1] * a_form[1] + dfinv_t[1, 2] * a_form[2]
        dfta_form[2] = dfinv_t[2, 0] * a_form[0] + dfinv_t[2, 1] * a_form[1] + dfinv_t[2, 2] * a_form[2]

        res[0] += 0.5 * w * ((v[0] - dfta_form[0]) ** 2.0 + (v[1] - dfta_form[1]) ** 2.0 + (v[2] - dfta_form[2]) ** 2.0)
