"Pusher kernel for full orbit (6D) particles."

from numpy import empty
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import eval_1form_spline_mpi, get_spans
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "df_inv", "v", "df_inv_v", "e_vec")
def push_weights_with_efield_lin_va(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    e1_1: "float[:,:,:]",
    e1_2: "float[:,:,:]",
    e1_3: "float[:,:,:]",
    f0_values: "float[:]",
    kappa: "float",
    vth: "float",
):
    r"""
    updates the single weights in the e_W substep of the linear Vlasov Ampère system with delta-f;
    c.f. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`.

    Parameters
    ----------
    e1_1, e1_2, e1_3 : array[float]
        3d array of FE coeffs of E-field as 1-form.

    f0_values : array[float]
        Value of f0 for each particle.

    kappa : float
        = 2 * pi * Omega_c / omega ; Parameter determining the coupling strength between particles and fields
    """

    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    v = empty(3, dtype=float)
    df_inv_v = empty(3, dtype=float)

    e_vec = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    # total number of markers (weights are w_p = delta f_p / (N * s_0))
    n_markers_tot = args_markers.Np

    # fmt: off
    #$ omp parallel private (ip, eta1, eta2, eta3, dfm, df_inv, v, df_inv_v, span1, span2, span3, e_vec, update)
    #$ omp for
    # fmt: on
    for ip in range(n_markers):
        if markers[ip, 0] == -1.0 or markers[ip, -1] == -2.0:
            continue

        # position
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # get velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # Compute Jacobian matrix
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # invert Jacobian matrix
        linalg_kernels.matrix_inv(dfm, df_inv)

        # compute DF^{-1} v
        linalg_kernels.matrix_vector(df_inv, v, df_inv_v)

        # E-field (1-form)
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            e1_1,
            e1_2,
            e1_3,
            e_vec,
        )

        # w_{n+1} = w_n + kappa * dt / (2 * N * s_0 * v_th^2) * f_0 * ( DF^{-1} v_p ) \cdot ( e_{n+1} + e_n )
        update = (
            (df_inv_v[0] * e_vec[0] + df_inv_v[1] * e_vec[1] + df_inv_v[2] * e_vec[2])
            * f0_values[ip]
            * kappa
            * dt
            / (2 * n_markers_tot * markers[ip, 7] * vth**2)
        )
        markers[ip, 6] += update
