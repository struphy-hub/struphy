"Pusher kernel for full orbit (6D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array(
    "dfm",
    "dfinv",
    "dfinv_t",
    "e",
    "e_cart",
    "GXu",
    "v",
)
def push_pc_GXu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    GXu_11: "float[:,:,:]",
    GXu_12: "float[:,:,:]",
    GXu_13: "float[:,:,:]",
    GXu_21: "float[:,:,:]",
    GXu_22: "float[:,:,:]",
    GXu_23: "float[:,:,:]",
    GXu_31: "float[:,:,:]",
    GXu_32: "float[:,:,:]",
    GXu_33: "float[:,:,:]",
):
    r"""Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u`
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.feec.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
    grad_Xu_ij : array[float]
        3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)
    GXu = empty((3, 3), dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
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
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # Evaluate grad(X(u, v)) at the particle positions
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_11,
            GXu_12,
            GXu_13,
            GXu[0, :],
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            GXu_21,
            GXu_22,
            GXu_23,
            GXu[1, :],
        )

        e[0] = GXu[0, 0] * v[0] + GXu[1, 0] * v[1]
        e[1] = GXu[0, 1] * v[0] + GXu[1, 1] * v[1]
        e[2] = GXu[0, 2] * v[0] + GXu[1, 2] * v[1]

        linalg_kernels.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt * e_cart / 2.0
