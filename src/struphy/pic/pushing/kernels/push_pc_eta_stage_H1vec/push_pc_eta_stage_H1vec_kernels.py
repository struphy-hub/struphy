"Pusher kernel for full orbit (6D) particles."

from numpy import empty, shape
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.pic.pushing.pusher_utilities_kernels as pusher_utilities_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_vectorfield_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "dfinv_t", "ginv", "v", "u", "k", "k_v")
def push_pc_eta_stage_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    u_1: "float[:,:,:]",
    u_2: "float[:,:,:]",
    u_3: "float[:,:,:]",
    use_perp_model: "bool",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Fourth order Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
    u_1, u_2, u_3 : array[float]
        3d array of FE coeffs of U-field, either as 1-form or as 2-form.

    u_basis : int
        U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker velocity
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_pusher_idx = args_markers.first_pusher_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # metric coeffs
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)
        linalg_kernels.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # U-field
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u_1,
            u_2,
            u_3,
            u,
        )

        if use_perp_model:
            u[2] = 0.0

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update markers for the next stage
        markers[ip, 0:3] = (
            markers[ip, first_pusher_idx : first_pusher_idx + 3]
            + dt * k * a[stage]
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

        # apply kinetic boundary conditions
        pusher_utilities_kernels.apply_kinetic_bc_marker(ip, args_markers, args_domain, False)
