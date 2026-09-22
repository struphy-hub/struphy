"""Ray-tracing pusher kernels with a per-marker time step.

Scaled copies of ``pusher_kernels.push_v_with_efield`` and ``pusher_kernels.push_eta_stage``: the step of marker ``ip``
is ``dt * dt_scale[ip]`` instead of ``dt``. For a steady-state ray trace, where each ray is followed on its own and only
the path integral of the current matters, every ray can then use the step that its local speed, cell size and
acceleration require (see :mod:`struphy.pic.ray_tracing`). With ``dt_scale == 1`` the results are identical to the
original kernels. The originals are left untouched; time-dependent PIC needs synchronous steps and must not use these.

``dt_scale`` is an array over the rows of the marker array (holes are skipped as in the originals).
``acceleration_magnitude`` evaluates ``const * |E|`` at every marker, so that the step can be chosen from the field the ray
is about to enter instead of the one it has just left.
"""

from numpy import empty, shape, sqrt, zeros
from pyccel.decorators import stack_array

# do not remove; needed to identify dependencies
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "dfinvt", "e_form", "e_cart")
def push_v_with_efield_scaled(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    const: "float",
    dt_scale: "float[:]",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1} - \mathbf v^n}{\Delta t} = c \, \bar{DF}^{-\top}  (\mathbb L^1)^\top \mathbf e

    where :math:`\mathbf e \in \mathbb R^{N_1}` are given FE coefficients of the 1-form spline field
    and :math:`c \in \mathbb R` is some constant.

    Parameters
    ----------
        e1_1, e1_2, e1_3 : ndarray[float]
            3d array of FE coeffs of E-field as 1-form.

        const : float
            A constant (usuallly related to the charge-to-mass ratio).
    """

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvt = zeros((3, 3), dtype=float)

    # allocate for field evaluations (1-form and Cartesian components)
    e_form = zeros(3, dtype=float)
    e_cart = zeros(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

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
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # electric field: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_form,
        )

        # electric field: Cartesian components
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * dt_scale[ip] * const * e_cart


@stack_array("dfm", "dfinv", "v", "k")
def push_eta_stage_scaled(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
    dt_scale: "float[:]",
):
    r"""Single stage of a s-stage Runge-Kutta solve of

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        # check if marker is a hole or a boundary particle
        if markers[ip, first_init_idx] == -1.0 or markers[ip, -1] == -2.0:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            e1,
            e2,
            e3,
            args_domain,
            dfm,
        )

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv(dfm, dfinv)

        # pull-back of velocity
        linalg_kernels.matrix_vector(dfinv, v, k)

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * dt_scale[ip] * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * dt_scale[ip] * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )


@stack_array("dfm", "dfinv", "dfinvt", "e_form", "e_cart")
def acceleration_magnitude(
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    const: "float",
    out: "float[:]",
):
    r"""Writes :math:`c\,|\mathbf E(\mathbf x_p)|` of every valid marker to ``out[ip]`` (0 for holes).

    :math:`\mathbf E` is the Cartesian field of the 1-form coefficients ``e1_*``, as in ``push_v_with_efield``.
    """

    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvt = zeros((3, 3), dtype=float)
    e_form = zeros(3, dtype=float)
    e_cart = zeros(3, dtype=float)

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        if not valid_mks[ip]:
            out[ip] = 0.0
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        evaluation_kernels.df(eta1, eta2, eta3, args_domain, dfm)
        linalg_kernels.matrix_inv(dfm, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)
        eval_1form_spline_mpi(span1, span2, span3, args_derham, e1_1, e1_2, e1_3, e_form)
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)

        out[ip] = const * sqrt(e_cart[0] ** 2 + e_cart[1] ** 2 + e_cart[2] ** 2)
