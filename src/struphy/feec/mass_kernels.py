"""
Integral kernels for mass matrices and L2-projections.

This module is intentionally restricted to constructs which Pyccel can
translate directly to Fortran without requiring gFTL container modules.
"""

import numpy as np
from numpy import shape

# ======================================================================
# 1D
# ======================================================================


def kernel_1d_mat(
    spans1: "int[:]",
    pi1: int,
    pj1: int,
    starts1: int,
    pads1: int,
    w1: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    mat_fun: "float[:]",
    data: "float[:,:]",
):
    """Assemble a 1D mass matrix."""

    ne1 = spans1.size
    nq1 = w1.shape[1]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            i_global1 = spans1[iel1] - pi1 + il1
            i_local1 = i_global1 - starts1

            for jl1 in range(pj1 + 1):
                value = 0.0

                for q1 in range(nq1):
                    value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * bj1[iel1, jl1, 0, q1] * mat_fun[iel1 * nq1 + q1]

                data[pads1 + i_local1, pads1 + jl1 - il1] += value


def kernel_1d_vec(
    spans1: "int[:]",
    pi1: int,
    starts1: int,
    pads1: int,
    w1: "float[:,:]",
    bi1: "float[:,:,:,:]",
    mat_fun: "float[:]",
    data: "float[:]",
):
    """Apply a 1D mass operator."""

    ne1 = spans1.size
    nq1 = w1.shape[1]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            i_global1 = spans1[iel1] - pi1 + il1
            i_local1 = i_global1 - starts1

            value = 0.0

            for q1 in range(nq1):
                value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * mat_fun[iel1 * nq1 + q1]

            data[pads1 + i_local1] += value


def kernel_1d_eval(
    spans1: "int[:]",
    pi1: int,
    starts1: int,
    pads1: int,
    bi1: "float[:,:,:,:]",
    coeffs_data: "float[:]",
    values: "float[:]",
):
    """Evaluate a 1D spline function at quadrature points."""

    values[:] = 0.0

    ne1 = spans1.size
    nq1 = bi1.shape[3]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            i_global1 = spans1[iel1] - pi1 + il1
            i_local1 = i_global1 - starts1

            coeff = coeffs_data[pads1 + i_local1]

            for q1 in range(nq1):
                values[iel1 * nq1 + q1] += coeff * bi1[iel1, il1, 0, q1]


# ======================================================================
# 2D
# ======================================================================


def kernel_2d_mat(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    pj1: int,
    pj2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    mat_fun: "float[:,:]",
    data: "float[:,:,:,:]",
):
    """Assemble a 2D mass matrix."""

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    for jl1 in range(pj1 + 1):
                        for jl2 in range(pj2 + 1):
                            value = 0.0

                            for q1 in range(nq1):
                                bi_1 = bi1[iel1, il1, 0, q1]
                                bj_1 = bj1[iel1, jl1, 0, q1]
                                w_1 = w1[iel1, q1]

                                for q2 in range(nq2):
                                    wvol = w_1 * w2[iel2, q2] * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]

                                    value += wvol * bi_1 * bi2[iel2, il2, 0, q2] * bj_1 * bj2[iel2, jl2, 0, q2]

                            data[pads1 + i_local1, pads2 + i_local2, pads1 + jl1 - il1, pads2 + jl2 - il2] += value


def kernel_2d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    mat_fun: "float[:,:]",
    data: "float[:,:]",
):
    """Apply a 2D mass operator."""

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    value = 0.0

                    for q1 in range(nq1):
                        bi_1 = bi1[iel1, il1, 0, q1]
                        w_1 = w1[iel1, q1]

                        for q2 in range(nq2):
                            value += (
                                w_1
                                * w2[iel2, q2]
                                * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                * bi_1
                                * bi2[iel2, il2, 0, q2]
                            )

                    data[pads1 + i_local1, pads2 + i_local2] += value


def kernel_2d_eval(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    coeffs_data: "float[:,:]",
    values: "float[:,:]",
):
    """Evaluate a 2D spline function at quadrature points."""

    values[:, :] = 0.0

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = bi1.shape[3]
    nq2 = bi2.shape[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    coeff = coeffs_data[pads1 + i_local1, pads2 + i_local2]

                    for q1 in range(nq1):
                        bi_1 = bi1[iel1, il1, 0, q1]

                        for q2 in range(nq2):
                            values[iel1 * nq1 + q1, iel2 * nq2 + q2] += coeff * bi_1 * bi2[iel2, il2, 0, q2]


# ======================================================================
# 3D
# ======================================================================


def kernel_3d_mat(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    pj1: int,
    pj2: int,
    pj3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    bj3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:,:,:,:]",
):
    """Assemble a 3D mass matrix."""

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]
    nq3 = w3.shape[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        for il3 in range(pi3 + 1):
                            i_global3 = spans3[iel3] - pi3 + il3
                            i_local3 = i_global3 - starts3

                            for jl1 in range(pj1 + 1):
                                for jl2 in range(pj2 + 1):
                                    for jl3 in range(pj3 + 1):
                                        value = 0.0

                                        for q1 in range(nq1):
                                            bi_1 = bi1[iel1, il1, 0, q1]
                                            bj_1 = bj1[iel1, jl1, 0, q1]
                                            w_1 = w1[iel1, q1]

                                            for q2 in range(nq2):
                                                bi_12 = bi_1 * bi2[iel2, il2, 0, q2]
                                                bj_12 = bj_1 * bj2[iel2, jl2, 0, q2]
                                                w_12 = w_1 * w2[iel2, q2]

                                                for q3 in range(nq3):
                                                    value += (
                                                        w_12
                                                        * w3[iel3, q3]
                                                        * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                                                        * bi_12
                                                        * bi3[iel3, il3, 0, q3]
                                                        * bj_12
                                                        * bj3[iel3, jl3, 0, q3]
                                                    )

                                        data[
                                            pads1 + i_local1,
                                            pads2 + i_local2,
                                            pads3 + i_local3,
                                            pads1 + jl1 - il1,
                                            pads2 + jl2 - il2,
                                            pads3 + jl3 - il3,
                                        ] += value


def kernel_3d_h1vec_weak_divergence(
    spansi1: "int[:]",
    spansi2: "int[:]",
    spansi3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    pj1: int,
    pj2: int,
    pj3: int,
    startsi1: int,
    startsi2: int,
    startsi3: int,
    padsi1: int,
    padsi2: int,
    padsi3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    bj3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    dlogj1: "float[:,:,:]",
    dlogj2: "float[:,:,:]",
    dlogj3: "float[:,:,:]",
    component_trial: int,
    data: "float[:,:,:,:,:,:]",
):
    r"""
    Assemble one component block of the weak H1vec divergence
    multiplication operator

    .. math::

        (B_w)_{k,(a,j)}
        =
        \int_{\widehat\Omega}
        \Lambda^3_k\,w\,
        \left(
            \partial_{\eta_a}N_j
            +
            N_j\partial_{\eta_a}\log|\det DF|
        \right)
        \,\mathrm d\boldsymbol\eta.

    The operator maps H1vec coefficients to the dual of the L2/3-form
    coefficient space.

    Parameters
    ----------
    bi1, bi2, bi3
        L2 test basis arrays.

    bj1, bj2, bj3
        H1 trial basis arrays. Derivative index zero contains basis values
        and derivative index one contains first derivatives.

    mat_fun
        Scalar weight ``w`` evaluated at quadrature points.

    component_trial
        H1vec component, equal to 0, 1 or 2.
    """

    ne1 = spansi1.size
    ne2 = spansi2.size
    ne3 = spansi3.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            # Global L2 test-basis indices.
                            i_global1 = spansi1[iel1] - pi1 + il1
                            i_global2 = spansi2[iel2] - pi2 + il2
                            i_global3 = spansi3[iel3] - pi3 + il3

                            # Local L2 test-basis indices.
                            i_local1 = i_global1 - startsi1
                            i_local2 = i_global2 - startsi2
                            i_local3 = i_global3 - startsi3

                            for jl1 in range(pj1 + 1):
                                for jl2 in range(pj2 + 1):
                                    for jl3 in range(pj3 + 1):
                                        value = 0.0

                                        for q1 in range(nq1):
                                            iq1 = iel1 * nq1 + q1

                                            for q2 in range(nq2):
                                                iq2 = iel2 * nq2 + q2

                                                for q3 in range(nq3):
                                                    iq3 = iel3 * nq3 + q3

                                                    test_basis = (
                                                        bi1[iel1, il1, 0, q1]
                                                        * bi2[iel2, il2, 0, q2]
                                                        * bi3[iel3, il3, 0, q3]
                                                    )

                                                    trial_basis = (
                                                        bj1[iel1, jl1, 0, q1]
                                                        * bj2[iel2, jl2, 0, q2]
                                                        * bj3[iel3, jl3, 0, q3]
                                                    )

                                                    trial_derivative = 0.0
                                                    dlogj = 0.0

                                                    if component_trial == 0:
                                                        trial_derivative = (
                                                            bj1[iel1, jl1, 1, q1]
                                                            * bj2[iel2, jl2, 0, q2]
                                                            * bj3[iel3, jl3, 0, q3]
                                                        )
                                                        dlogj = dlogj1[
                                                            iq1,
                                                            iq2,
                                                            iq3,
                                                        ]

                                                    elif component_trial == 1:
                                                        trial_derivative = (
                                                            bj1[iel1, jl1, 0, q1]
                                                            * bj2[iel2, jl2, 1, q2]
                                                            * bj3[iel3, jl3, 0, q3]
                                                        )
                                                        dlogj = dlogj2[
                                                            iq1,
                                                            iq2,
                                                            iq3,
                                                        ]

                                                    else:
                                                        trial_derivative = (
                                                            bj1[iel1, jl1, 0, q1]
                                                            * bj2[iel2, jl2, 0, q2]
                                                            * bj3[iel3, jl3, 1, q3]
                                                        )
                                                        dlogj = dlogj3[
                                                            iq1,
                                                            iq2,
                                                            iq3,
                                                        ]

                                                    physical_divergence = trial_derivative + trial_basis * dlogj

                                                    value += (
                                                        w1[iel1, q1]
                                                        * w2[iel2, q2]
                                                        * w3[iel3, q3]
                                                        * mat_fun[
                                                            iq1,
                                                            iq2,
                                                            iq3,
                                                        ]
                                                        * test_basis
                                                        * physical_divergence
                                                    )

                                        data[
                                            padsi1 + i_local1,
                                            padsi2 + i_local2,
                                            padsi3 + i_local3,
                                            padsi1 + jl1 - il1,
                                            padsi2 + jl2 - il2,
                                            padsi3 + jl3 - il3,
                                        ] += value


def kernel_3d_h1vec_divdiv(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    p1: int,
    p2: int,
    p3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    b1: "float[:,:,:,:]",
    b2: "float[:,:,:,:]",
    b3: "float[:,:,:,:]",
    weighted_rho: "float[:,:,:]",
    dlogj1: "float[:,:,:]",
    dlogj2: "float[:,:,:]",
    dlogj3: "float[:,:,:]",
    component_test: int,
    component_trial: int,
    data: "float[:,:,:,:,:,:]",
):
    r"""
    Assemble block (component_test, component_trial) of

        int rho div(u_h) div(v_h) dx

    for H1vec fields with push-forward u = DF * u_hat.

    ``rho`` is the L2/3-form proxy rho_hat = rho_physical * abs(det(DF)).
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(b1)[1]
    nq2 = shape(b2)[1]
    nq3 = shape(b3)[1]

    test_order1 = 1 if component_test == 0 else 0
    test_order2 = 1 if component_test == 1 else 0
    test_order3 = 1 if component_test == 2 else 0

    trial_order1 = 1 if component_trial == 0 else 0
    trial_order2 = 1 if component_trial == 1 else 0
    trial_order3 = 1 if component_trial == 2 else 0

    if component_test == 0:
        dlogj_test = dlogj1
    elif component_test == 1:
        dlogj_test = dlogj2
    else:
        dlogj_test = dlogj3

    if component_trial == 0:
        dlogj_trial = dlogj1
    elif component_trial == 1:
        dlogj_trial = dlogj2
    else:
        dlogj_trial = dlogj3

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(p1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(p3 + 1):
                            i_global1 = spans1[iel1] - p1 + il1
                            i_global2 = spans2[iel2] - p2 + il2
                            i_global3 = spans3[iel3] - p3 + il3

                            i_local1 = i_global1 - starts1
                            i_local2 = i_global2 - starts2
                            i_local3 = i_global3 - starts3

                            for jl1 in range(p1 + 1):
                                for jl2 in range(p2 + 1):
                                    for jl3 in range(p3 + 1):
                                        value = 0.0

                                        for q1 in range(nq1):
                                            iq1 = iel1 * nq1 + q1

                                            for q2 in range(nq2):
                                                iq2 = iel2 * nq2 + q2

                                                for q3 in range(nq3):
                                                    iq3 = iel3 * nq3 + q3

                                                    ni = (
                                                        b1[iel1, il1, 0, q1]
                                                        * b2[iel2, il2, 0, q2]
                                                        * b3[iel3, il3, 0, q3]
                                                    )
                                                    nj = (
                                                        b1[iel1, jl1, 0, q1]
                                                        * b2[iel2, jl2, 0, q2]
                                                        * b3[iel3, jl3, 0, q3]
                                                    )

                                                    dni = 0.0
                                                    dnj = 0.0
                                                    ci = 0.0
                                                    cj = 0.0

                                                    ni = (
                                                        b1[iel1, il1, 0, q1]
                                                        * b2[iel2, il2, 0, q2]
                                                        * b3[iel3, il3, 0, q3]
                                                    )

                                                    nj = (
                                                        b1[iel1, jl1, 0, q1]
                                                        * b2[iel2, jl2, 0, q2]
                                                        * b3[iel3, jl3, 0, q3]
                                                    )

                                                    dni = (
                                                        b1[iel1, il1, test_order1, q1]
                                                        * b2[iel2, il2, test_order2, q2]
                                                        * b3[iel3, il3, test_order3, q3]
                                                    )

                                                    dnj = (
                                                        b1[iel1, jl1, trial_order1, q1]
                                                        * b2[iel2, jl2, trial_order2, q2]
                                                        * b3[iel3, jl3, trial_order3, q3]
                                                    )

                                                    div_test = dni + dlogj_test[iq1, iq2, iq3] * ni

                                                    div_trial = dnj + dlogj_trial[iq1, iq2, iq3] * nj

                                                    div_test = dni + ci * ni
                                                    div_trial = dnj + cj * nj

                                                    value += weighted_rho[iq1, iq2, iq3] * div_test * div_trial

                                        data[
                                            pads1 + i_local1,
                                            pads2 + i_local2,
                                            pads3 + i_local3,
                                            pads1 + jl1 - il1,
                                            pads2 + jl2 - il2,
                                            pads3 + jl3 - il3,
                                        ] += value


def kernel_3d_h1vec_divergence_eval(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    p1: int,
    p2: int,
    p3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    b1: "float[:,:,:,:]",
    b2: "float[:,:,:,:]",
    b3: "float[:,:,:,:]",
    dlogj1: "float[:,:,:]",
    dlogj2: "float[:,:,:]",
    dlogj3: "float[:,:,:]",
    component: int,
    coeffs: "float[:,:,:]",
    values: "float[:,:,:]",
):
    r"""
    Add one H1vec component to

        Q u = div_x(DF * u_hat)

    evaluated at quadrature points.

    ``values`` is not cleared by this kernel, because the kernel is called
    once for every vector component.
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(b1)[3]
    nq2 = shape(b2)[3]
    nq3 = shape(b3)[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(p1 + 1):
                    i_global1 = spans1[iel1] - p1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(p2 + 1):
                        i_global2 = spans2[iel2] - p2 + il2
                        i_local2 = i_global2 - starts2

                        for il3 in range(p3 + 1):
                            i_global3 = spans3[iel3] - p3 + il3
                            i_local3 = i_global3 - starts3

                            coefficient = coeffs[
                                pads1 + i_local1,
                                pads2 + i_local2,
                                pads3 + i_local3,
                            ]

                            for q1 in range(nq1):
                                iq1 = iel1 * nq1 + q1

                                for q2 in range(nq2):
                                    iq2 = iel2 * nq2 + q2

                                    for q3 in range(nq3):
                                        iq3 = iel3 * nq3 + q3

                                        basis = b1[iel1, il1, 0, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 0, q3]

                                        derivative = 0.0
                                        dlogj = 0.0

                                        if component == 0:
                                            derivative = (
                                                b1[iel1, il1, 1, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 0, q3]
                                            )
                                            dlogj = dlogj1[iq1, iq2, iq3]

                                        elif component == 1:
                                            derivative = (
                                                b1[iel1, il1, 0, q1] * b2[iel2, il2, 1, q2] * b3[iel3, il3, 0, q3]
                                            )
                                            dlogj = dlogj2[iq1, iq2, iq3]

                                        else:
                                            derivative = (
                                                b1[iel1, il1, 0, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 1, q3]
                                            )
                                            dlogj = dlogj3[iq1, iq2, iq3]

                                        values[iq1, iq2, iq3] += coefficient * (derivative + dlogj * basis)


def kernel_3d_h1vec_divergence_transpose(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    p1: int,
    p2: int,
    p3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    b1: "float[:,:,:,:]",
    b2: "float[:,:,:,:]",
    b3: "float[:,:,:,:]",
    dlogj1: "float[:,:,:]",
    dlogj2: "float[:,:,:]",
    dlogj3: "float[:,:,:]",
    component: int,
    values: "float[:,:,:]",
    coeffs: "float[:,:,:]",
):
    r"""
    Apply one component of the Euclidean transpose Q.T.

    ``values`` already contains any desired square-root quadrature and
    density weight. No quadrature weights are applied in this kernel.
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(b1)[3]
    nq2 = shape(b2)[3]
    nq3 = shape(b3)[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(p1 + 1):
                    i_global1 = spans1[iel1] - p1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(p2 + 1):
                        i_global2 = spans2[iel2] - p2 + il2
                        i_local2 = i_global2 - starts2

                        for il3 in range(p3 + 1):
                            i_global3 = spans3[iel3] - p3 + il3
                            i_local3 = i_global3 - starts3

                            result = 0.0

                            for q1 in range(nq1):
                                iq1 = iel1 * nq1 + q1

                                for q2 in range(nq2):
                                    iq2 = iel2 * nq2 + q2

                                    for q3 in range(nq3):
                                        iq3 = iel3 * nq3 + q3

                                        basis = b1[iel1, il1, 0, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 0, q3]

                                        derivative = 0.0
                                        dlogj = 0.0

                                        if component == 0:
                                            derivative = (
                                                b1[iel1, il1, 1, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 0, q3]
                                            )
                                            dlogj = dlogj1[iq1, iq2, iq3]

                                        elif component == 1:
                                            derivative = (
                                                b1[iel1, il1, 0, q1] * b2[iel2, il2, 1, q2] * b3[iel3, il3, 0, q3]
                                            )
                                            dlogj = dlogj2[iq1, iq2, iq3]

                                        else:
                                            derivative = (
                                                b1[iel1, il1, 0, q1] * b2[iel2, il2, 0, q2] * b3[iel3, il3, 1, q3]
                                            )
                                            dlogj = dlogj3[iq1, iq2, iq3]

                                        result += values[iq1, iq2, iq3] * (derivative + dlogj * basis)

                            coeffs[
                                pads1 + i_local1,
                                pads2 + i_local2,
                                pads3 + i_local3,
                            ] += result


def kernel_3d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:]",
):
    """Apply a 3D mass operator."""

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]
    nq3 = w3.shape[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        for il3 in range(pi3 + 1):
                            i_global3 = spans3[iel3] - pi3 + il3
                            i_local3 = i_global3 - starts3

                            value = 0.0

                            for q1 in range(nq1):
                                bi_1 = bi1[iel1, il1, 0, q1]
                                w_1 = w1[iel1, q1]

                                for q2 in range(nq2):
                                    bi_12 = bi_1 * bi2[iel2, il2, 0, q2]
                                    w_12 = w_1 * w2[iel2, q2]

                                    for q3 in range(nq3):
                                        value += (
                                            w_12
                                            * w3[iel3, q3]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                                            * bi_12
                                            * bi3[iel3, il3, 0, q3]
                                        )

                            data[pads1 + i_local1, pads2 + i_local2, pads3 + i_local3] += value


def kernel_3d_eval(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    coeffs_data: "float[:,:,:]",
    values: "float[:,:,:]",
):
    """Evaluate a 3D spline function at quadrature points."""

    values[:, :, :] = 0.0

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = bi1.shape[3]
    nq2 = bi2.shape[3]
    nq3 = bi3.shape[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        for il3 in range(pi3 + 1):
                            i_global3 = spans3[iel3] - pi3 + il3
                            i_local3 = i_global3 - starts3

                            coeff = coeffs_data[pads1 + i_local1, pads2 + i_local2, pads3 + i_local3]

                            for q1 in range(nq1):
                                bi_1 = bi1[iel1, il1, 0, q1]

                                for q2 in range(nq2):
                                    bi_12 = bi_1 * bi2[iel2, il2, 0, q2]

                                    for q3 in range(nq3):
                                        values[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3] += (
                                            coeff * bi_12 * bi3[iel3, il3, 0, q3]
                                        )


def kernel_3d_matrixfree(
    spansi1: "int[:]",
    spansi2: "int[:]",
    spansi3: "int[:]",
    spansj1: "int[:]",
    spansj2: "int[:]",
    spansj3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    pj1: int,
    pj2: int,
    pj3: int,
    startsi1: int,
    startsi2: int,
    startsi3: int,
    startsj1: int,
    startsj2: int,
    startsj3: int,
    padsi1: int,
    padsi2: int,
    padsi3: int,
    padsj1: int,
    padsj2: int,
    padsj3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    bj3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data_out: "float[:,:,:]",
    data_in: "float[:,:,:]",
):
    """Apply a 3D mass matrix without assembling it."""

    ne1 = spansi1.size
    ne2 = spansi2.size
    ne3 = spansi3.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]
    nq3 = w3.shape[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            bj = 0.0

                            for jl1 in range(pj1 + 1):
                                j_global1 = spansj1[iel1] - pj1 + jl1
                                j_local1 = j_global1 - startsj1 + padsj1

                                bj_1 = bj1[iel1, jl1, 0, q1]

                                for jl2 in range(pj2 + 1):
                                    j_global2 = spansj2[iel2] - pj2 + jl2
                                    j_local2 = j_global2 - startsj2 + padsj2

                                    bj_12 = bj_1 * bj2[iel2, jl2, 0, q2]

                                    for jl3 in range(pj3 + 1):
                                        j_global3 = spansj3[iel3] - pj3 + jl3
                                        j_local3 = j_global3 - startsj3 + padsj3

                                        bj += bj_12 * bj3[iel3, jl3, 0, q3] * data_in[j_local1, j_local2, j_local3]

                            wvol = (
                                w1[iel1, q1]
                                * w2[iel2, q2]
                                * w3[iel3, q3]
                                * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                            )

                            for il1 in range(pi1 + 1):
                                i_global1 = spansi1[iel1] - pi1 + il1
                                i_local1 = i_global1 - startsi1 + padsi1

                                bi_1 = bi1[iel1, il1, 0, q1]

                                for il2 in range(pi2 + 1):
                                    i_global2 = spansi2[iel2] - pi2 + il2
                                    i_local2 = i_global2 - startsi2 + padsi2

                                    bi_12 = bi_1 * bi2[iel2, il2, 0, q2]

                                    for il3 in range(pi3 + 1):
                                        i_global3 = spansi3[iel3] - pi3 + il3
                                        i_local3 = i_global3 - startsi3 + padsi3

                                        data_out[i_local1, i_local2, i_local3] += (
                                            wvol * bi_12 * bi3[iel3, il3, 0, q3] * bj
                                        )


def kernel_3d_diag(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:]",
):
    """Compute the diagonal of a 3D mass matrix."""

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]
    nq3 = w3.shape[1]

    nb1 = data.shape[0]
    nb2 = data.shape[1]
    nb3 = data.shape[2]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_local1 = i_global1 - starts1

                    if i_local1 >= nb1:
                        i_local1 -= nb1

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        if i_local2 >= nb2:
                            i_local2 -= nb2

                        for il3 in range(pi3 + 1):
                            i_global3 = spans3[iel3] - pi3 + il3
                            i_local3 = i_global3 - starts3

                            if i_local3 >= nb3:
                                i_local3 -= nb3

                            value = 0.0

                            for q1 in range(nq1):
                                bi_1 = bi1[iel1, il1, 0, q1]

                                for q2 in range(nq2):
                                    bi_12 = bi_1 * bi2[iel2, il2, 0, q2]

                                    for q3 in range(nq3):
                                        value += (
                                            w1[iel1, q1]
                                            * w2[iel2, q2]
                                            * w3[iel3, q3]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                                            * bi_12
                                            * bi3[iel3, il3, 0, q3]
                                            * bi_12
                                            * bi3[iel3, il3, 0, q3]
                                            / (bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3])
                                        )

                            data[i_local1, i_local2, i_local3] += value


# ======================================================================
# 3D surface kernels
# ======================================================================


def surface_kernel_3d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    pi0: int,
    pi1: int,
    pi2: int,
    starts0: int,
    starts1: int,
    starts2: int,
    pads0: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    boundary_index: int,
    mat_fun: "float[:,:]",
    data: "float[:,:,:]",
):
    """Integrate a scalar function over a fixed 3D boundary surface."""

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]

    i_local0 = boundary_index - starts0

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                i_global1 = spans1[iel1] - pi1 + il1
                i_local1 = i_global1 - starts1

                for il2 in range(pi2 + 1):
                    i_global2 = spans2[iel2] - pi2 + il2
                    i_local2 = i_global2 - starts2

                    value = 0.0

                    for q1 in range(nq1):
                        bi_1 = bi1[iel1, il1, 0, q1]

                        for q2 in range(nq2):
                            value += (
                                w1[iel1, q1]
                                * w2[iel2, q2]
                                * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                * bi_1
                                * bi2[iel2, il2, 0, q2]
                            )

                    data[pads0 + i_local0, pads1 + i_local1, pads2 + i_local2] += value


def surface_kernel_3d_mat(
    spans1: "int[:]",
    spans2: "int[:]",
    pi0: int,
    pi1: int,
    pi2: int,
    pj0: int,
    pj1: int,
    pj2: int,
    starts0: int,
    starts1: int,
    starts2: int,
    pads0: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    boundary_index: int,
    normal_dir: int,
    mat_fun: "float[:,:]",
    data: "float[:,:,:,:,:,:]",
):
    """
    Assemble a surface mass matrix.

    No Python lists, tuples, comprehensions, dictionaries, sets, or other
    container objects are used here. The normal direction is handled
    explicitly to avoid generating gFTL container dependencies.
    """

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = w1.shape[1]
    nq2 = w2.shape[1]

    if normal_dir == 0:
        i_local_n = boundary_index - starts0

        for iel1 in range(ne1):
            for iel2 in range(ne2):
                for il1 in range(pi1 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_local1 = i_global1 - starts1

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        for jl1 in range(pj1 + 1):
                            for jl2 in range(pj2 + 1):
                                value = 0.0

                                for q1 in range(nq1):
                                    bi_1 = bi1[iel1, il1, 0, q1]
                                    bj_1 = bj1[iel1, jl1, 0, q1]

                                    for q2 in range(nq2):
                                        value += (
                                            w1[iel1, q1]
                                            * w2[iel2, q2]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                            * bi_1
                                            * bi2[iel2, il2, 0, q2]
                                            * bj_1
                                            * bj2[iel2, jl2, 0, q2]
                                        )

                                data[
                                    pads0 + i_local_n,
                                    pads1 + i_local1,
                                    pads2 + i_local2,
                                    pads0,
                                    pads1 + jl1 - il1,
                                    pads2 + jl2 - il2,
                                ] += value

    elif normal_dir == 1:
        i_local_n = boundary_index - starts1

        for iel1 in range(ne1):
            for iel2 in range(ne2):
                for il1 in range(pi0 + 1):
                    i_global1 = spans1[iel1] - pi0 + il1
                    i_local1 = i_global1 - starts0

                    for il2 in range(pi2 + 1):
                        i_global2 = spans2[iel2] - pi2 + il2
                        i_local2 = i_global2 - starts2

                        for jl1 in range(pj0 + 1):
                            for jl2 in range(pj2 + 1):
                                value = 0.0

                                for q1 in range(nq1):
                                    bi_1 = bi1[iel1, il1, 0, q1]
                                    bj_1 = bj1[iel1, jl1, 0, q1]

                                    for q2 in range(nq2):
                                        value += (
                                            w1[iel1, q1]
                                            * w2[iel2, q2]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                            * bi_1
                                            * bi2[iel2, il2, 0, q2]
                                            * bj_1
                                            * bj2[iel2, jl2, 0, q2]
                                        )

                                data[
                                    pads0 + i_local1,
                                    pads1 + i_local_n,
                                    pads2 + i_local2,
                                    pads0 + jl1 - il1,
                                    pads1,
                                    pads2 + jl2 - il2,
                                ] += value

    else:
        i_local_n = boundary_index - starts2

        for iel1 in range(ne1):
            for iel2 in range(ne2):
                for il1 in range(pi0 + 1):
                    i_global1 = spans1[iel1] - pi0 + il1
                    i_local1 = i_global1 - starts0

                    for il2 in range(pi1 + 1):
                        i_global2 = spans2[iel2] - pi1 + il2
                        i_local2 = i_global2 - starts1

                        for jl1 in range(pj0 + 1):
                            for jl2 in range(pj1 + 1):
                                value = 0.0

                                for q1 in range(nq1):
                                    bi_1 = bi1[iel1, il1, 0, q1]
                                    bj_1 = bj1[iel1, jl1, 0, q1]

                                    for q2 in range(nq2):
                                        value += (
                                            w1[iel1, q1]
                                            * w2[iel2, q2]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                            * bi_1
                                            * bi2[iel2, il2, 0, q2]
                                            * bj_1
                                            * bj2[iel2, jl2, 0, q2]
                                        )

                                data[
                                    pads0 + i_local1,
                                    pads1 + i_local2,
                                    pads2 + i_local_n,
                                    pads0 + jl1 - il1,
                                    pads1 + jl2 - il2,
                                    pads2,
                                ] += value
