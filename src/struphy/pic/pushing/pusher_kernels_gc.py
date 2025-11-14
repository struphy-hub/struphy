"Pusher kernels for gyro-center (5D) dynamics."

from numpy import empty, mod, shape, sqrt, zeros
from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.geometry.evaluation_kernels as evaluation_kernels

# do not remove; needed to identify dependencies
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels
import struphy.linear_algebra.linalg_kernels as linalg_kernels
from struphy.bsplines.evaluation_kernels_3d import (
    eval_0form_spline_mpi,
    eval_1form_spline_mpi,
    eval_2form_spline_mpi,
    eval_3form_spline_mpi,
    eval_vectorfield_spline_mpi,
    get_spans,
)
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@stack_array("dfm", "unit_b1", "e_star", "e_field", "Exb", "k")
def push_gc_bxEstar_explicit_multistage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of an s-stage explicit Runge-Kutta scheme for solving

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = \frac{\hat{\mathbf E}^{*1} \times \hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\boldsymbol \eta_p(t)) \,,

    where

    .. math::

        \hat{\mathbf E}^{*1} = - \hat \nabla \hat \phi -\varepsilon \mu_p \hat \nabla  \hat B\,,\qquad \hat B^*_\parallel = \hat B + \varepsilon v_{\parallel,p} \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}\,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    unit_b1 = empty(3, dtype=float)
    e_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    Exb = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            unit_b1_1,
            unit_b1_2,
            unit_b1_3,
            unit_b1,
        )

        # compute E*
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            e_star,
        )

        e_star *= -epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )
            e_star += e_field

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v
        b_star_parallel += B_dot_b
        b_star_parallel *= det_df

        # calculate k
        linalg_kernels.cross(e_star, unit_b1, Exb)

        k[:] = Exb / b_star_parallel

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )


@stack_array("eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "unit_b1", "e_field", "Exb", "k")
def push_gc_bxEstar_discrete_gradient_1st_order(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Picard iteration (index :math:`k`) for

    .. math::

        \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} =
        \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n} )  \,,

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2}
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})
        = \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)
        + (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \,
        \frac{H(\mathbf Z_p^{n+1, k}) - H(\mathbf Z_p^{n}) - (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \cdot
        \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)}{||\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}||}\,,

    is the Gonzalez discrete gradient.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)

    unit_b1 = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    Exb = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n (from init_kernel)
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n (from init_kernel)
        b_star_parallel = markers[ip, first_free_idx + 1]
        unit_b1[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at (n+1, k) (from eval_kernel)
        H_k = markers[ip, first_free_idx + 5]

        # mid-point spline evaluation
        span1, span2, span3 = get_spans(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_derham,
        )

        # compute grad_H at n + 1/2
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute grad_I
        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H)
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff)

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        # calculate k
        linalg_kernels.cross(unit_b1, grad_I, Exb)

        k[:] = Exb / b_star_parallel

        # accumulation for last stage
        markers[ip, 0:3] = eta_n + dt * k

        # residual
        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2 + (markers[ip, 1] - eta_k[1]) ** 2 + (markers[ip, 2] - eta_k[2]) ** 2,
        )


@stack_array("dfm", "eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "unit_b1", "e_field", "Exb", "k")
def push_gc_bxEstar_discrete_gradient_2nd_order(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    unit_b1_1: "float[:,:,:]",
    unit_b1_2: "float[:,:,:]",
    unit_b1_3: "float[:,:,:]",
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Picard iteration (index :math:`k`) for

    .. math::

        \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} =
        \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} \left( \frac{\mathbf Z_p^{n+1, k}
        + \mathbf Z_p^{n}}{2} \right) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n} )  \,,

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2}
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})
        = \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)
        + (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \,
        \frac{H(\mathbf Z_p^{n+1, k}) - H(\mathbf Z_p^{n}) - (\boldsymbol \eta_p^{n+1, k} - \boldsymbol \eta_p^{n}) \cdot
        \frac{\partial H}{\partial \boldsymbol \eta} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)}{||\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}||}\,,

    is the Gonzalez discrete gradient.

    Notes
    -----
    This kernel performs evaluations at mid-points.
    Other evaluations are performed in ``init_kernels`` and ``eval_kernels``,
    respectively.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)

    unit_b1 = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    Exb = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # Hamiltonian at n (from init_kernel)
        H_n = markers[ip, first_free_idx]

        # Hamiltonian at (n+1, k) (from eval_kernel)
        H_k = markers[ip, first_free_idx + 1]

        # mid-point evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # mid-point spline evaluation
        span1, span2, span3 = get_spans(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_derham,
        )

        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            unit_b1_1,
            unit_b1_2,
            unit_b1_3,
            unit_b1,
        )

        # compute grad_H at n + 1/2
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute grad_I
        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H)
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff)

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v
        b_star_parallel += B_dot_b
        b_star_parallel *= det_df

        # calculate k
        linalg_kernels.cross(unit_b1, grad_I, Exb)

        k[:] = Exb / b_star_parallel

        # accumulation for last stage
        markers[ip, 0:3] = eta_n + dt * k

        # residual
        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2 + (markers[ip, 1] - eta_k[1]) ** 2 + (markers[ip, 2] - eta_k[2]) ** 2,
        )


@stack_array(
    "eta_k",
    "eta_n",
    "eta_k_shifted",
    "eta_diff",
    "grad_H_12",
    "grad_H",
    "unit_b1",
    "e_field",
    "grad_I",
    "Ddg",
    "bcross_mat",
    "func",
    "Dfunc",
    "Dfunc_inv",
    "k",
)
def push_gc_bxEstar_discrete_gradient_1st_order_newton(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    phi_coeffs: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Newton iteration for

    .. math::

        \frac{\boldsymbol \eta_p^{n+1} - \boldsymbol \eta_p^{n}}{\Delta t} =
        \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n} )  \,,

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2}
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n})
        = \begin{pmatrix}
        \frac{H(\eta_{p,1}^{n+1}) - H}{\eta_{p,1}^{n+1} - \eta_{p,1}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}) - H(\eta_{p,1}^{n+1})}{\eta_{p,2}^{n+1} - \eta_{p,2}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1})}{\eta_{p,3}^{n+1} - \eta_{p,3}^n}
        \end{pmatrix}\,,

    is the Itoh-Abe discrete gradient. The Newton algorithm searches the roots of

    .. math::

        \mathbf F(\boldsymbol \eta_p^{n+1}) = \boldsymbol \eta_p^{n+1} - \boldsymbol \eta_p^{n}
        - \Delta t \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times \frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n}) = 0\,,

    via (iteration index :math:`k`)

    .. math::

        \boldsymbol \eta_p^{n+1, k+1} = \boldsymbol \eta_p^{n+1, k}
        - D\mathbf F^{-1}(\boldsymbol \eta_p^{n+1, k}) \mathbf F( \boldsymbol \eta_p^{n+1, k})\,,

    where the Jacobian is given by

    .. math::

        D\mathbf F(\boldsymbol \eta_p^{n+1, k}) = \mathbb I_{3\times 3}
        - \Delta t \frac{\hat{\mathbf b}^1_0}{\sqrt g\,\hat B_\parallel^{*}} (\mathbf Z_p^{n}) \times
        D\frac{\partial \overline H}{\partial \boldsymbol \eta}
        (\boldsymbol \eta_p^{n+1}, \boldsymbol \eta_p^{n})\,.

    Notes
    -----
    This kernel performs evaluations at :math:`\boldsymbol \eta_p^{n+1, k}`.
    Other evaluations are performed in ``init_kernels`` and ``eval_kernels``,
    respectively.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_k_shifted = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H_12 = empty(2, dtype=float)
    grad_H = empty(3, dtype=float)

    unit_b1 = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    grad_I = zeros(3, dtype=float)
    Ddg = zeros((3, 3), dtype=float)
    bcross_mat = zeros((3, 3), dtype=float)
    func = zeros(3, dtype=float)
    Dfunc = zeros((3, 3), dtype=float)
    Dfunc_inv = zeros((3, 3), dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]
        eta_k_shifted[:] = eta_k + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_diff[:] = eta_k_shifted - eta_n

        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # Hamiltonian at n
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n
        b_star_parallel = markers[ip, first_free_idx + 1]
        unit_b1[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k)
        H_k1 = markers[ip, first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        H_k12 = markers[ip, first_free_idx + 6]

        # 1st comp of gradient of Hamiltonian at eta_1^(n+1, k)
        grad_H_1 = markers[ip, first_free_idx + 7]

        # 1st and 2nd comps of gradient of Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        grad_H_12[:] = markers[ip, first_free_idx + 8 : first_free_idx + 10]

        # evaluate H at (n+1, k)
        span1, span2, span3 = get_spans(eta_k[0], eta_k[1], eta_k[2], args_derham)

        if evaluate_e_field:
            phi = eval_0form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                phi_coeffs,
            )
        else:
            phi = 0.0

        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        H_k = epsilon * v**2 / 2.0 + epsilon * mu * B_dot_b + phi

        # compute grad_H at (n+1, k)
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute the Itoh discrete gradient
        if eta_diff[0] == 0.0:
            grad_I[0] = grad_H[0]
        else:
            grad_I[0] = (H_k1 - H_n) / (eta_diff[0])

        if eta_diff[1] == 0.0:
            grad_I[1] = grad_H[1]
        else:
            grad_I[1] = (H_k12 - H_k1) / (eta_diff[1])

        if eta_diff[2] == 0.0:
            grad_I[2] = grad_H[2]
        else:
            grad_I[2] = (H_k - H_k12) / (eta_diff[2])

        # compute matrix for cross product
        bcross_mat[0, 1] = -unit_b1[2]
        bcross_mat[0, 2] = unit_b1[1]
        bcross_mat[1, 0] = unit_b1[2]
        bcross_mat[1, 2] = -unit_b1[0]
        bcross_mat[2, 0] = -unit_b1[1]
        bcross_mat[2, 1] = unit_b1[0]
        bcross_mat /= b_star_parallel

        # compute F
        linalg_kernels.matrix_vector(bcross_mat, grad_I, func)
        func *= -dt
        func += eta_diff

        # compute the Jacobian of the discrete gradient
        if eta_diff[0] == 0.0:
            Ddg[0, 0] = 0.0
        else:
            Ddg[0, 0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / eta_diff[0] ** 2

        if eta_diff[1] == 0.0:
            Ddg[1, 1] = 0.0
            Ddg[1, 0] = 0.0
        else:
            Ddg[1, 1] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / eta_diff[1] ** 2
            Ddg[1, 0] = (grad_H_12[0] - grad_H_1) / eta_diff[1]

        if eta_diff[2] == 0.0:
            Ddg[2, 2] = 0.0
            Ddg[2, 0] = 0.0
            Ddg[2, 1] = 0.0
        else:
            Ddg[2, 2] = (grad_H[2] * eta_diff[2] - (H_k - H_k12)) / eta_diff[2] ** 2
            Ddg[2, 0] = (grad_H[0] - grad_H_12[0]) / eta_diff[2]
            Ddg[2, 1] = (grad_H[1] - grad_H_12[1]) / eta_diff[2]

        # compute Jacobian matrix DF
        linalg_kernels.matrix_matrix(bcross_mat, Ddg, Dfunc)
        Dfunc *= -dt
        Dfunc[0, 0] += 1.0
        Dfunc[1, 1] += 1.0
        Dfunc[2, 2] += 1.0

        # comute inverse and update
        linalg_kernels.matrix_inv(Dfunc, Dfunc_inv)
        linalg_kernels.matrix_vector(Dfunc_inv, func, k)

        markers[ip, 0:3] -= k

        # residual
        markers[ip, residual_idx] = sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)


@stack_array("dfm", "e_star", "b2", "b_star", "k")
def push_gc_Bstar_explicit_multistage(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of an s-stage explicit Runge-Kutta scheme for solving

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\hat{\mathbf B}^{*2}}{\sqrt g \,\hat B^{*}_\parallel}(\boldsymbol \eta_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= \frac{1}{\varepsilon} \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^{*}_\parallel} \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p(t)) \,,
            \end{aligned}
        \right.

    where

    .. math::

        \hat{\mathbf E}^{*1} = - \hat \nabla \hat \phi - \varepsilon \mu_p \hat \nabla \hat B\,,\qquad \hat{\mathbf B}^{*2} = \hat{\mathbf B}^2 + \varepsilon v_\parallel \hat \nabla \times \hat{\mathbf b}^1_0\,,\qquad  \hat B^*_\parallel = \hat B + \varepsilon v_{\parallel,p} \widehat{\left[(\nabla \times \mathbf b_0) \cdot \mathbf b_0\right]}\,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    e_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        # if stage == 0.:
        #     # save initial parallel velocity
        #     markers[ip, 14] = markers[ip, 3]

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, mu_idx]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # compute E*
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            e_star,
        )

        e_star *= -epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )
            e_star += e_field

        # compute B*
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b2,
        )

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b2_1,
            curl_unit_b2_2,
            curl_unit_b2_3,
            b_star,
        )

        b_star *= epsilon * v
        b_star += b2

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v
        b_star_parallel += B_dot_b
        b_star_parallel *= det_df

        # calculate k for eta
        k[:] = b_star / b_star_parallel * v

        # calculate k_v for v
        k_v = linalg_kernels.scalar_dot(b_star, e_star)
        k_v /= b_star_parallel * epsilon

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] += dt * b[stage] * k
        markers[ip, first_free_idx + 3] += dt * b[stage] * k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            + dt * a[stage] * k
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )
        markers[ip, 3] = markers[ip, first_init_idx + 3] + dt * a[stage] * k_v + last * markers[ip, first_free_idx + 3]


@stack_array("eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "e_field", "b_star", "k")
def push_gc_Bstar_discrete_gradient_1st_order(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Picard iteration (index :math:`k`) for

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} &= 
                \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g \,\hat B^{*}_\parallel} (\mathbf Z_p^{n}) 
                \frac{\partial \overline H}{\partial v_{\parallel}} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
                \\
                \frac{v_{\parallel,p}^{n+1,k+1} - v_{\parallel,p}^{n}}{\Delta t} &= 
                - \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^{*}_\parallel} (\mathbf Z_p^{n}) \cdot
                \frac{\partial \overline H}{\partial \boldsymbol \eta} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
            \end{aligned}
        \right.

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2} 
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) +  \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \mathbf Z}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})
        = \frac{\partial H}{\partial \mathbf Z} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)
        + (\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}) \,
        \frac{H(\mathbf Z_p^{n+1, k}) - H(\mathbf Z_p^{n}) - (\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}) \cdot
        \frac{\partial H}{\partial \mathbf Z} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)}{||\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}||}\,,

    is the Gonzalez discrete gradient.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)

    e_field = zeros(3, dtype=float)
    b_star = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v_mid = (v_k + v_n) / 2.0
        v_diff = v_k - v_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n (from init_kernel)
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n (from init_kernel)
        b_star_parallel = epsilon * markers[ip, first_free_idx + 1]
        b_star[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at (n+1, k) (from eval_kernel)
        H_k = markers[ip, first_free_idx + 5]

        # mid-point spline evaluation
        span1, span2, span3 = get_spans(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_derham,
        )

        # compute grad_H at n + 1/2
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute grad_I
        grad_H_v = epsilon * v_mid

        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H) + v_diff * grad_H_v
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff) + v_diff * v_diff

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
            grad_I_v = grad_H_v
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared
            grad_I_v = grad_H_v + v_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        # calculate k for eta
        k[:] = b_star / b_star_parallel * grad_I_v

        # calculate k_v for v
        k_v = linalg_kernels.scalar_dot(b_star, grad_I)
        k_v /= -b_star_parallel

        # compute values at (n+1, k+1)
        markers[ip, 0:3] = eta_n + dt * k
        markers[ip, 3] = v_n + dt * k_v

        # residual
        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2
            + (markers[ip, 1] - eta_k[1]) ** 2
            + (markers[ip, 2] - eta_k[2]) ** 2
            + ((markers[ip, 3] - v_k) / v_k) ** 2,
        )


@stack_array("dfm", "eta_k", "eta_n", "eta_mid", "eta_diff", "grad_H", "grad_I", "e_field", "b2", "b_star", "k")
def push_gc_Bstar_discrete_gradient_2nd_order(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    b2_1: "float[:,:,:]",
    b2_2: "float[:,:,:]",
    b2_3: "float[:,:,:]",
    curl_unit_b2_1: "float[:,:,:]",
    curl_unit_b2_2: "float[:,:,:]",
    curl_unit_b2_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    curl_unit_b_dot_b0: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Picard iteration (index :math:`k`) for

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} &= 
                \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g \,\hat B^{*}_\parallel} 
                \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right) 
                \frac{\partial \overline H}{\partial v_{\parallel}} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
                \\
                \frac{v_{\parallel,p}^{n+1,k+1} - v_{\parallel,p}^{n}}{\Delta t} &= 
                - \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^{*}_\parallel} 
                \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right) \cdot
                \frac{\partial \overline H}{\partial \boldsymbol \eta} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
            \end{aligned}
        \right.

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) =\varepsilon\frac{v_{\parallel}^2}{2} 
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \mathbf Z}
        (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})
        = \frac{\partial H}{\partial \mathbf Z} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)
        + (\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}) \,
        \frac{H(\mathbf Z_p^{n+1, k}) - H(\mathbf Z_p^{n}) - (\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}) \cdot
        \frac{\partial H}{\partial \mathbf Z} \left( \frac{\mathbf Z_p^{n+1, k} + \mathbf Z_p^{n}}{2} \right)}{||\mathbf Z_p^{n+1, k} - \mathbf Z_p^{n}||}\,,

    is the Gonzalez discrete gradient.

    Notes
    -----
    This kernel performs evaluations at mid-points. 
    Other evaluations are performed in ``init_kernels`` and ``eval_kernels``,
    respectively.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_mid = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)

    e_field = zeros(3, dtype=float)
    b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3] + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]

        eta_mid[:] = (eta_k + eta_n) / 2.0
        eta_mid[:] = mod(eta_mid, 1.0)
        eta_diff[:] = eta_k - eta_n

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v_mid = (v_k + v_n) / 2.0
        v_diff = v_k - v_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n (from init_kernel)
        H_n = markers[ip, first_free_idx]

        # Hamiltonian at (n+1, k) (from eval_kernel)
        H_k = markers[ip, first_free_idx + 1]

        # mid-point evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_domain,
            dfm,
        )

        det_df = linalg_kernels.det(dfm)

        # mid-point spline evaluation
        span1, span2, span3 = get_spans(
            eta_mid[0],
            eta_mid[1],
            eta_mid[2],
            args_derham,
        )

        # compute grad_H at n + 1/2
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute grad_I
        grad_H_v = epsilon * v_mid

        dZ_dot_grad_H = linalg_kernels.scalar_dot(eta_diff, grad_H) + v_diff * grad_H_v
        dZ_squared = linalg_kernels.scalar_dot(eta_diff, eta_diff) + v_diff * v_diff

        if dZ_squared == 0.0:
            grad_I[:] = grad_H
            grad_I_v = grad_H_v
        else:
            grad_I[:] = grad_H + eta_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared
            grad_I_v = grad_H_v + v_diff * (H_k - H_n - dZ_dot_grad_H) / dZ_squared

        # compute B*
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b2_1,
            b2_2,
            b2_3,
            b2,
        )

        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b2_1,
            curl_unit_b2_2,
            curl_unit_b2_3,
            b_star,
        )

        b_star *= epsilon * v_mid
        b_star += b2

        # compute B*_parallel
        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        b_star_parallel = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_unit_b_dot_b0,
        )

        b_star_parallel *= epsilon * v_mid
        b_star_parallel += B_dot_b
        b_star_parallel *= epsilon * det_df

        # calculate k for eta
        k[:] = b_star / b_star_parallel * grad_I_v

        # calculate k_v for v
        k_v = linalg_kernels.scalar_dot(b_star, grad_I)
        k_v /= -b_star_parallel

        # compute values at (n+1, k+1)
        markers[ip, 0:3] = eta_n + dt * k
        markers[ip, 3] = v_n + dt * k_v

        # residual
        markers[ip, residual_idx] = sqrt(
            (markers[ip, 0] - eta_k[0]) ** 2
            + (markers[ip, 1] - eta_k[1]) ** 2
            + (markers[ip, 2] - eta_k[2]) ** 2
            + ((markers[ip, 3] - v_k) / v_k) ** 2,
        )


@stack_array(
    "eta_k",
    "eta_n",
    "eta_k_shifted",
    "eta_diff",
    "grad_H_12",
    "grad_H",
    "b_star",
    "e_field",
    "grad_I",
    "J_vec",
    "Ddg",
    "DdgT",
    "func",
    "B",
    "C",
    "A_inv",
    "k",
)
def push_gc_Bstar_discrete_gradient_1st_order_newton(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    grad_b_full_1: "float[:,:,:]",
    grad_b_full_2: "float[:,:,:]",
    grad_b_full_3: "float[:,:,:]",
    B_dot_b_coeffs: "float[:,:,:]",
    e_field_1: "float[:,:,:]",
    e_field_2: "float[:,:,:]",
    e_field_3: "float[:,:,:]",
    phi_coeffs: "float[:,:,:]",
    evaluate_e_field: bool,
):
    r"""For each marker :math:`p` in markers array, make one step of Newton iteration for

    .. math::

        \left\{ 
            \begin{aligned} 
                \frac{\boldsymbol \eta_p^{n+1, k+1} - \boldsymbol \eta_p^{n}}{\Delta t} &= 
                \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g \,\hat B^{*}_\parallel} (\mathbf Z_p^{n}) 
                \frac{\partial \overline H}{\partial v_{\parallel}} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
                \\
                \frac{v_{\parallel,p}^{n+1,k+1} - v_{\parallel,p}^{n}}{\Delta t} &= 
                - \frac 1 \varepsilon\frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^{*}_\parallel} (\mathbf Z_p^{n}) \cdot
                \frac{\partial \overline H}{\partial \boldsymbol \eta} (\mathbf Z_p^{n+1, k}, \mathbf Z_p^{n})\,,
            \end{aligned}
        \right.

    where the Hamiltonian reads

    .. math::

        H(\mathbf Z) = H(\boldsymbol \eta, v_{\parallel}) = \varepsilon\frac{v_{\parallel}^2}{2} 
        + \varepsilon\mu_p |\hat{\mathbf B}| (\boldsymbol \eta) + \hat \phi(\boldsymbol \eta)\,,

    and where

    .. math::

        \frac{\partial \overline H}{\partial \mathbf Z}
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n})
        = \begin{pmatrix}
        \frac{H(\eta_{p,1}^{n+1}) - H}{\eta_{p,1}^{n+1} - \eta_{p,1}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}) - H(\eta_{p,1}^{n+1})}{\eta_{p,2}^{n+1} - \eta_{p,2}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1})}{\eta_{p,3}^{n+1} - \eta_{p,3}^n}
        \\[1mm]
        \frac{H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1}, v_{\parallel, p}^{n+1}) - H(\eta_{p,1}^{n+1}, \eta_{p,2}^{n+1}, \eta_{p,3}^{n+1})}{v_{\parallel, p}^{n+1} - v_{\parallel,p}^n}
        \end{pmatrix}\,,

    is the Itoh-Abe discrete gradient. The Newton algorithm searches the roots of

    .. math::

        \mathbf F(\mathbf Z_p^{n+1}) = \mathbf Z_p^{n+1} - \mathbf Z_p^{n} 
        - \Delta t \mathbb J (\mathbf Z_p^{n}) \frac{\partial \overline H}{\partial \mathbf Z} 
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n}) = 0\,,

    via (iteration index :math:`k`)

    .. math::

        \mathbf Z^{n+1, k+1} = \mathbf Z_p^{n+1, k} 
        - D\mathbf F^{-1}(\mathbf Z_p^{n+1, k}) \mathbf F( \mathbf Z_p^{n+1, k})\,,

    where the Jacobian is given by

    .. math::

        D\mathbf F(\boldsymbol \eta_p^{n+1, k}) = \mathbb I_{3\times 3} 
        - \Delta t \mathbb J (\mathbf Z_p^{n}) 
        D\frac{\partial \overline H}{\partial \mathbf Z} 
        (\mathbf Z_p^{n+1}, \mathbf Z_p^{n})\,.

    Notes
    -----
    This kernel performs evaluations at :math:`\mathbf Z_p^{n+1, k}`. 
    Other evaluations are performed in ``init_kernels`` and ``eval_kernels``,
    respectively.
    """

    # allocate stack arrays
    eta_k = empty(3, dtype=float)
    eta_n = empty(3, dtype=float)
    eta_k_shifted = empty(3, dtype=float)
    eta_diff = empty(3, dtype=float)

    grad_H_12 = empty(2, dtype=float)
    grad_H = empty(3, dtype=float)

    b_star = empty(3, dtype=float)
    e_field = zeros(3, dtype=float)
    grad_I = zeros(3, dtype=float)
    J_vec = empty(3, dtype=float)
    Ddg = zeros((3, 3), dtype=float)
    DdgT = zeros((3, 3), dtype=float)
    func = zeros(3, dtype=float)
    B = empty(3, dtype=float)
    C = empty(3, dtype=float)
    A_inv = zeros((3, 3), dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_shift_idx = args_markers.first_shift_idx
    residual_idx = args_markers.residual_idx
    first_free_idx = args_markers.first_free_idx

    for ip in range(n_markers):
        # check if marker is converged or a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta_k[:] = markers[ip, 0:3]
        eta_n[:] = markers[ip, first_init_idx : first_init_idx + 3]
        eta_k_shifted[:] = eta_k + markers[ip, first_shift_idx : first_shift_idx + 3]
        eta_diff[:] = eta_k_shifted - eta_n

        v_k = markers[ip, 3]
        v_n = markers[ip, first_init_idx + 3]
        v_diff = v_k - v_n

        mu = markers[ip, mu_idx]

        # Hamiltonian at n
        H_n = markers[ip, first_free_idx]

        # Poisson matrix at n
        b_star_parallel = epsilon * markers[ip, first_free_idx + 1]
        b_star[:] = markers[ip, first_free_idx + 2 : first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k)
        H_k1 = markers[ip, first_free_idx + 5]

        # Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        H_k12 = markers[ip, first_free_idx + 6]

        # 1st comp of gradient of Hamiltonian at eta_1^(n+1, k)
        grad_H_1 = markers[ip, first_free_idx + 7]

        # 1st and 2nd comps of gradient of Hamiltonian at eta_1^(n+1, k), eta_2^(n+1, k)
        grad_H_12[:] = markers[ip, first_free_idx + 8 : first_free_idx + 10]

        # evaluate H at (n+1, k)
        span1, span2, span3 = get_spans(eta_k[0], eta_k[1], eta_k[2], args_derham)

        if evaluate_e_field:
            phi = eval_0form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                phi_coeffs,
            )
        else:
            phi = 0.0

        B_dot_b = eval_0form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            B_dot_b_coeffs,
        )

        H_k = epsilon * v_k**2 / 2.0 + epsilon * mu * B_dot_b + phi
        H_k123 = epsilon * v_n**2 / 2.0 + epsilon * mu * B_dot_b + phi

        # compute grad_H at (n+1, k)
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            grad_b_full_1,
            grad_b_full_2,
            grad_b_full_3,
            grad_H,
        )

        grad_H *= epsilon * mu

        if evaluate_e_field:
            eval_1form_spline_mpi(
                span1,
                span2,
                span3,
                args_derham,
                e_field_1,
                e_field_2,
                e_field_3,
                e_field,
            )

            e_field *= -1.0
            grad_H += e_field

        # compute the Itoh discrete gradient
        grad_H_v = epsilon * v_k

        if eta_diff[0] == 0.0:
            grad_I[0] = grad_H[0]
        else:
            grad_I[0] = (H_k1 - H_n) / (eta_diff[0])

        if eta_diff[1] == 0.0:
            grad_I[1] = grad_H[1]
        else:
            grad_I[1] = (H_k12 - H_k1) / (eta_diff[1])

        if eta_diff[2] == 0.0:
            grad_I[2] = grad_H[2]
        else:
            grad_I[2] = (H_k123 - H_k12) / (eta_diff[2])

        if v_diff == 0.0:
            grad_I_v = grad_H_v
        else:
            grad_I_v = (H_k - H_k123) / (v_diff)

        # compute F; the Poisson matrix is [[0, Jvec], [-Jvec^T, 0]]
        J_vec[:] = b_star / b_star_parallel
        func[:] = J_vec * grad_I_v
        func *= -dt
        func += eta_diff

        func_v = linalg_kernels.scalar_dot(J_vec, grad_I)
        func_v *= -1.0
        func_v *= -dt
        func_v += v_diff

        # compute the Jacobian of the discrete gradient; it has the form [[Ddg, 0], [0, Ddg_v]]
        if eta_diff[0] == 0.0:
            Ddg[0, 0] = 0.0
        else:
            Ddg[0, 0] = (grad_H_1 * eta_diff[0] - (H_k1 - H_n)) / eta_diff[0] ** 2

        if eta_diff[1] == 0.0:
            Ddg[1, 1] = 0.0
            Ddg[1, 0] = 0.0
        else:
            Ddg[1, 1] = (grad_H_12[1] * eta_diff[1] - (H_k12 - H_k1)) / eta_diff[1] ** 2
            Ddg[1, 0] = (grad_H_12[0] - grad_H_1) / eta_diff[1]

        if eta_diff[2] == 0.0:
            Ddg[2, 2] = 0.0
            Ddg[2, 0] = 0.0
            Ddg[2, 1] = 0.0
        else:
            Ddg[2, 2] = (grad_H[2] * eta_diff[2] - (H_k123 - H_k12)) / eta_diff[2] ** 2
            Ddg[2, 0] = (grad_H[0] - grad_H_12[0]) / eta_diff[2]
            Ddg[2, 1] = (grad_H[1] - grad_H_12[1]) / eta_diff[2]

        if v_diff == 0.0:
            Ddg_v = 0.0
        else:
            Ddg_v = (grad_H_v * v_diff - (H_k - H_k123)) / v_diff**2

        # the matrix DF is [[I_3x3, -dt*J_vec*Ddg_v], [dt*(Ddg^T*J_vec)^T, 1]]
        # we compute its inverse with the Schur complement of [[I_3x3, B], [C, 1]]
        # block matrix B
        B[:] = J_vec
        B *= Ddg_v
        B *= -dt
        # block matrix C
        linalg_kernels.transpose(Ddg, DdgT)
        linalg_kernels.matrix_vector(DdgT, J_vec, C)
        C *= dt
        # Schur complement M/A
        schur_comp = 1.0 - linalg_kernels.scalar_dot(C, B)
        # inverse blocks
        linalg_kernels.outer(B, C, A_inv)
        A_inv /= schur_comp
        A_inv[0, 0] += 1.0
        A_inv[1, 1] += 1.0
        A_inv[2, 2] += 1.0

        B /= -schur_comp
        C /= -schur_comp

        # update
        linalg_kernels.matrix_vector(A_inv, func, k)
        k += B * func_v
        k_v = linalg_kernels.scalar_dot(C, func)
        k_v += func_v / schur_comp

        markers[ip, 0:3] -= k
        markers[ip, 3] -= k_v

        # residual
        markers[ip, residual_idx] = sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2 + (k_v / v_k) ** 2)


@stack_array("dfm", "e", "u", "b", "b_star", "norm_b1", "curl_norm_b")
def push_gc_cc_J1_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
):
    r"""Velocity update step for the `CurrentCoupling5DCurlb <https://struphy-hub.github.io/struphy/sections/subsections/propagators-coupling.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update :

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^v (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        v = markers[ip, 3]

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

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            b,
        )

        # u; H1vec
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # b_star; in H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp / abs_b_star_para * v * dt


@stack_array("dfm", "df_t", "g", "g_inv", "e", "u", "u0", "b", "b_star", "norm_b1", "curl_norm_b")
def push_gc_cc_J1_Hcurl(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
):
    r"""Velocity update step for the `CurrentCoupling5DCurlb <https://struphy-hub.github.io/struphy/sections/subsections/propagators-coupling.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^1 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    u0 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(
            eta1,
            eta2,
            eta3,
            args_domain,
            dfm,
        )

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            b,
        )

        # u; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # b_star; in H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        linalg_kernels.matrix_vector(g_inv, u, u0)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u0, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp / abs_b_star_para * v * dt


@stack_array("dfm", "e", "u", "b", "b_star", "norm_b1", "curl_norm_b")
def push_gc_cc_J1_Hdiv(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
    boundary_cut: float,
):
    r"""Velocity update step for the `CurrentCoupling5DCurlb <https://struphy-hub.github.io/struphy/sections/subsections/propagators-coupling.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^2 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers

    # -- removed omp: #$ omp parallel private(ip, boundary_cut, eta1, eta2, eta3, v, det_df, dfm, span1, span2, span3, b, u, e, curl_norm_b, norm_b1, b_star, temp, abs_b_star_para)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        if eta1 < boundary_cut or eta1 > 1.0 - boundary_cut:
            continue

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

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            b,
        )

        # u; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        u = u / det_df

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp / abs_b_star_para * v * dt

    # -- removed omp: #$ omp end parallel


@stack_array(
    "dfm",
    "df_t",
    "df_inv_t",
    "g_inv",
    "e",
    "u",
    "bb",
    "b_star",
    "norm_b1",
    "norm_b2",
    "curl_norm_b",
    "tmp1",
    "tmp2",
    "b_prod",
    "norm_b2_prod",
)
def push_gc_cc_J2_stage_H1vec(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    norm_b21: "float[:,:,:]",
    norm_b22: "float[:,:,:]",
    norm_b23: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
):
    r"""Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradB <https://struphy-hub.github.io/struphy/sections/subsections/propagators-coupling.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradB>`_

    Marker update:

    .. math::

        \mathbf X^{n+1} = \mathbf X^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) G^{-1}(\mathbf X_p) \hat{\mathbf b}_0^2(\mathbf X_p) \times G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times \Lambda^v (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    bb = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            bb,
        )

        # u; H1vec
        eval_vectorfield_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b21,
            norm_b22,
            norm_b23,
            norm_b2,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -bb[2]
        b_prod[0, 2] = +bb[1]
        b_prod[1, 0] = +bb[2]
        b_prod[1, 2] = -bb[0]
        b_prod[2, 0] = -bb[1]
        b_prod[2, 1] = +bb[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (bb + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
        linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

        linalg_kernels.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] -= dt * b[stage] * e

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            - dt * a[stage] * e
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )


@stack_array(
    "dfm",
    "df_inv",
    "df_inv_t",
    "g_inv",
    "e",
    "u",
    "bb",
    "b_star",
    "norm_b1",
    "norm_b2",
    "curl_norm_b",
    "tmp1",
    "tmp2",
    "b_prod",
    "norm_b2_prod",
)
def push_gc_cc_J2_stage_Hdiv(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
    args_derham: "DerhamArguments",
    epsilon: float,
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    norm_b11: "float[:,:,:]",
    norm_b12: "float[:,:,:]",
    norm_b13: "float[:,:,:]",
    norm_b21: "float[:,:,:]",
    norm_b22: "float[:,:,:]",
    norm_b23: "float[:,:,:]",
    curl_norm_b1: "float[:,:,:]",
    curl_norm_b2: "float[:,:,:]",
    curl_norm_b3: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
    a: "float[:]",
    b: "float[:]",
    c: "float[:]",
    boundary_cut: float,
):
    r"""Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradB <https://struphy-hub.github.io/struphy/sections/subsections/propagators-coupling.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradB>`_

    Marker update:

    .. math::

        \mathbf X^{n+1} = \mathbf X^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} G^{-1}(\mathbf X_p) \hat{\mathbf b}_0^2(\mathbf X_p) \times G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times \Lambda^2 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers for fields
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = zeros((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    bb = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get marker arguments
    markers = args_markers.markers
    n_markers = args_markers.n_markers
    mu_idx = args_markers.mu_idx
    first_init_idx = args_markers.first_init_idx
    first_free_idx = args_markers.first_free_idx

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.0
    else:
        last = 0.0

    # -- removed omp: #$ omp parallel firstprivate(b_prod, norm_b2_prod) private(ip, boundary_cut, eta1, eta2, eta3, v, det_df, dfm, df_inv, df_inv_t, g_inv, span1, span2, span3, bb, u, e, curl_norm_b, norm_b1, norm_b2, b_star, tmp1, tmp2, abs_b_star_para)
    # -- removed omp: #$ omp for
    for ip in range(n_markers):
        # check if marker is a hole
        if markers[ip, first_init_idx] == -1.0:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        if eta1 < boundary_cut or eta2 > 1.0 - boundary_cut:
            continue

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
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            b1,
            b2,
            b3,
            bb,
        )

        # u; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            u1,
            u2,
            u3,
            u,
        )

        # norm_b1; 1form
        eval_1form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b11,
            norm_b12,
            norm_b13,
            norm_b1,
        )

        # norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            norm_b21,
            norm_b22,
            norm_b23,
            norm_b2,
        )

        # curl_norm_b; 2form
        eval_2form_spline_mpi(
            span1,
            span2,
            span3,
            args_derham,
            curl_norm_b1,
            curl_norm_b2,
            curl_norm_b3,
            curl_norm_b,
        )

        # operator bx() as matrix
        b_prod[0, 1] = -bb[2]
        b_prod[0, 2] = +bb[1]
        b_prod[1, 0] = +bb[2]
        b_prod[1, 2] = -bb[0]
        b_prod[2, 0] = -bb[1]
        b_prod[2, 1] = +bb[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (bb + curl_norm_b * v * epsilon) / det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
        linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

        linalg_kernels.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para
        e /= det_df

        # accumulation for last stage
        markers[ip, first_free_idx : first_free_idx + 3] -= dt * b[stage] * e

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = (
            markers[ip, first_init_idx : first_init_idx + 3]
            - dt * a[stage] * e
            + last * markers[ip, first_free_idx : first_free_idx + 3]
        )

    # -- removed omp: #$ omp end parallel
