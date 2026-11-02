"Pusher kernel for full orbit (6D) particles with a magnetic moment (classical Pauli particles)."

from numpy import zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "dfinv", "dfinvt", "e_form", "e_cart", "gradb_form", "gradb_cart")
def push_v_with_efield_mu(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    gradb1_1: "float[:,:,:]",
    gradb1_2: "float[:,:,:]",
    gradb1_3: "float[:,:,:]",
    const_factor: "float",
):
    r"""Updates particle velocities as

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = \bar{DF}^{-\top} \left( c\, (\mathbb L^1)^\top \mathbf e
        - \mu_p\, (\mathbb L^1)^\top \mathbf g \right)\,,

    where :math:`\mathbf e \in \mathbb R^{N_1}` are the FE coefficients of the 1-form spline field,
    :math:`\mathbf g \in \mathbb R^{N_1}` those of the gradient of the magnetic field strength :math:`\nabla |\mathbf B_0|`
    (1-form), :math:`c \in \mathbb R` is a constant (``const_factor``) and :math:`\mu_p` is the magnetic moment
    of the marker, stored in the column ``args_markers.mu_idx``.

    Parameters
    ----------
        e1_1, e1_2, e1_3 : ndarray[float]
            3d array of FE coeffs of E-field as 1-form.

        gradb1_1, gradb1_2, gradb1_3 : ndarray[float]
            3d array of FE coeffs of grad |B0| as 1-form.

        const_factor : float
            A constant multiplying the E-field (usually related to the charge-to-mass ratio).
    """

    # allocate metric coeffs
    dfm = zeros((3, 3), dtype=float)
    dfinv = zeros((3, 3), dtype=float)
    dfinvt = zeros((3, 3), dtype=float)

    # allocate for field evaluations (1-form and Cartesian components)
    e_form = zeros(3, dtype=float)
    e_cart = zeros(3, dtype=float)
    gradb_form = zeros(3, dtype=float)
    gradb_cart = zeros(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks
    mu_idx = args_markers.mu_idx

    # fmt: off
    #$ omp parallel private(dfinv, dfinvt, dfm, e_cart, e_form, gradb_cart, gradb_form, eta1, eta2, eta3, ip, span1, span2, span3, thread_derham)
    # fmt: on
    # Each thread owns the mutable spline-evaluation buffers.
    thread_derham = DerhamArguments(
        args_derham.pn, args_derham.tn1, args_derham.tn2, args_derham.tn3, args_derham.starts
    )
    # fmt: off
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]

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
        span1, span2, span3 = get_spans(eta1, eta2, eta3, thread_derham)

        # electric field and grad |B0|: 1-form components
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            e1_1,
            e1_2,
            e1_3,
            e_form,
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            thread_derham,
            gradb1_1,
            gradb1_2,
            gradb1_3,
            gradb_form,
        )

        # Cartesian components
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)
        linalg_kernels.matrix_vector(dfinvt, gradb_form, gradb_cart)

        # update velocities
        markers[3:6, ip] += dt * (const_factor * e_cart - markers[mu_idx, ip] * gradb_cart)

    # fmt: off
    #$ omp end parallel
    # fmt: on
