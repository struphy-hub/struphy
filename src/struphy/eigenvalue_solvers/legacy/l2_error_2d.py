# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to compute L2-errors of discrete p-forms with analytical forms in 2D.
"""

import numpy as np
import scipy.sparse as spa

import struphy.eigenvalue_solvers.kernels_2d as ker


# ======= error in V0 ====================
def l2_error_V0(tensor_space_FEM, domain, f0, c0, method="standard"):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    f0 : callable or np.ndarray
        the 0-form with which the error shall be computed

    c0 : array_like
        the FEM coefficients of the discrete 0-form

    method : string
        method used to compute the error (standard : integrals are computed with matrix-vector products, else: kernel is used)

    Returns
    -------
    error : int
        the L2-error of the discrete form
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indN = (
        tensor_space_FEM.indN
    )  # global indices of local non-vanishing basis functions in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points in format (element, local quad_point)
    wts = tensor_space_FEM.wts  # global quadrature weights in format (element, local weight)

    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points

    # extract coefficients to tensor-product space
    if c0.ndim == 1:
        c0 = tensor_space_FEM.extract_0(c0)

    assert c0.ndim == 3

    # evaluation of |det(DF)| at quadrature points
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))

    # evaluation of exact 0-form at quadrature points
    if callable(f0):
        quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing="ij")
        f0 = f0(quad_mesh[0], quad_mesh[1], 0.0)

    if method == "standard":
        # evaluation of discrete 0-form at quadrature points
        f0_h = tensor_space_FEM.evaluate_NN(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c0, "V0")[:, :, 0]

        # compute error
        error = 0.0

        integrand = (f0_h - f0) ** 2 * det_df
        error += integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

    else:
        # compute error in each element
        error = np.zeros(Nel[:2], dtype=float)

        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [0, 0],
            [0, 0],
            basisN[0],
            basisN[1],
            basisN[0],
            basisN[1],
            indN[0],
            indN[1],
            indN[0],
            indN[1],
            error,
            f0.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f0.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c0,
            c0,
            det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        error = error.sum()

    return np.sqrt(error)


# ======= error in V1 ====================
def l2_error_V1(tensor_space_FEM, domain, f1, c1, method="standard"):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    f1 : list of callables or np.ndarrays
        the three 1-form components with which the error shall be computed

    c1 : list of array_like
        the FEM coefficients of the discrete components

    method : string
        method used to compute the error (standard : integrals are computed with matrix-vector products, else: kernel is used)

    Returns
    -------
    error : int
        the L2-error of the discrete form
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indN = (
        tensor_space_FEM.indN
    )  # global indices of non-vanishing basis functions (N) in format (element, global index)
    indD = (
        tensor_space_FEM.indD
    )  # global indices of non-vanishing basis functions (D) in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points
    wts = tensor_space_FEM.wts  # global quadrature weights

    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)

    # extract cicients to tensor-product space
    c1_1, c1_2, c1_3 = tensor_space_FEM.extract_1(c1)

    # evaluation of G^(-1)*|det(DF)| at quadrature points
    metric_coeffs = domain.metric_inv(pts[0].flatten(), pts[1].flatten(), 0.0)
    metric_coeffs *= abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))

    # evaluation of exact 1-form components at quadrature points
    if callable(f1[0]):
        quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing="ij")
        f1_1 = f1[0](quad_mesh[0], quad_mesh[1], 0.0)
        f1_2 = f1[1](quad_mesh[0], quad_mesh[1], 0.0)
        f1_3 = f1[2](quad_mesh[0], quad_mesh[1], 0.0)

    if method == "standard":
        # evaluation of discrete 1-form components at quadrature points
        f1_h_1 = tensor_space_FEM.evaluate_DN(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c1_1, "V1")[:, :, 0]
        f1_h_2 = tensor_space_FEM.evaluate_ND(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c1_2, "V1")[:, :, 0]
        f1_h_3 = tensor_space_FEM.evaluate_NN(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c1_3, "V1")[:, :, 0]

        # compute error
        error = 0.0

        # 1 * d_f1 * G^11 * |det(DF)| * d_f1
        integrand = (f1_h_1 - f1_1) * metric_coeffs[0, 0] * (f1_h_1 - f1_1)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 2 * d_f1 * G^12 * |det(DF)| * d_f2
        integrand = (f1_h_1 - f1_1) * metric_coeffs[0, 1] * (f1_h_2 - f1_2)
        error += 2 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 1 * d_f2 * G^22 * |det(DF)| * d_f2
        integrand = (f1_h_2 - f1_2) * metric_coeffs[1, 1] * (f1_h_2 - f1_2)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 1 * d_f3 * G^33 * |det(DF)| * d_f3
        integrand = (f1_h_3 - f1_3) * metric_coeffs[2, 2] * (f1_h_3 - f1_3)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

    else:
        # compute error in each element
        error = np.zeros(Nel[:2], dtype=float)

        # 1 * d_f1 * G^11 * |det(DF)| * d_f1
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [1, 0],
            [1, 0],
            basisD[0],
            basisN[1],
            basisD[0],
            basisN[1],
            indD[0],
            indN[1],
            indD[0],
            indN[1],
            error,
            f1_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f1_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c1_1,
            c1_1,
            1 * metric_coeffs[0, 0].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 2 * d_f1 * G^12 * |det(DF)| * d_f2
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [1, 0],
            [0, 1],
            basisD[0],
            basisN[1],
            basisN[0],
            basisD[1],
            indD[0],
            indN[1],
            indN[0],
            indD[1],
            error,
            f1_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f1_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c1_1,
            c1_2,
            2 * metric_coeffs[0, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 1 * d_f2 * G^22 * |det(DF)| * d_f2
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [0, 1],
            [0, 1],
            basisN[0],
            basisD[1],
            basisN[0],
            basisD[1],
            indN[0],
            indD[1],
            indN[0],
            indD[1],
            error,
            f1_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f1_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c1_2,
            c1_2,
            1 * metric_coeffs[1, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 1 * d_f3 * G^33 * |det(DF)| * d_f3
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [0, 0],
            [0, 0],
            basisN[0],
            basisN[1],
            basisN[0],
            basisN[1],
            indN[0],
            indN[1],
            indN[0],
            indN[1],
            error,
            f1_3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f1_3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c1_3,
            c1_3,
            1 * metric_coeffs[2, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        error = error.sum()

    return np.sqrt(error)


# ======= error in V2 ====================
def l2_error_V2(tensor_space_FEM, domain, f2, c2, method="standard"):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    f2 : list of callables or np.ndarrays
        the three 2-form components with which the error shall be computed

    c2 : list of array_like
        the FEM coefficients of the discrete components

    method : string
        method used to compute the error (standard : integrals are computed with matrix-vector products, else: kernel is used)

    Returns
    -------
    error : int
        the L2-error of the discrete form
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indN = (
        tensor_space_FEM.indN
    )  # global indices of non-vanishing basis functions (N) in format (element, global index)
    indD = (
        tensor_space_FEM.indD
    )  # global indices of non-vanishing basis functions (D) in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points
    wts = tensor_space_FEM.wts  # global quadrature weights

    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)

    # extract coefficients to tensor-product space
    c2_1, c2_2, c2_3 = tensor_space_FEM.extract_2(c2)

    # evaluation of G/|det(DF)| at quadrature points
    metric_coeffs = domain.metric(pts[0].flatten(), pts[1].flatten(), 0.0)
    metric_coeffs /= abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))

    # evaluation of exact 2-form components at quadrature points
    if callable(f2[0]):
        quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing="ij")
        f2_1 = f2[0](quad_mesh[0], quad_mesh[1], 0.0)
        f2_2 = f2[1](quad_mesh[0], quad_mesh[1], 0.0)
        f2_3 = f2[2](quad_mesh[0], quad_mesh[1], 0.0)

    if method == "standard":
        # evaluation of discrete 2-form components at quadrature points
        f2_h_1 = tensor_space_FEM.evaluate_ND(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c2_1, "V2")[:, :, 0]
        f2_h_2 = tensor_space_FEM.evaluate_DN(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c2_2, "V2")[:, :, 0]
        f2_h_3 = tensor_space_FEM.evaluate_DD(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c2_3, "V2")[:, :, 0]

        # compute error
        error = 0.0

        # 1 * d_f1 * G_11 / |det(DF)| * d_f1
        integrand = (f2_h_1 - f2_1) * metric_coeffs[0, 0] * (f2_h_1 - f2_1)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 2 * d_f1 * G_12 / |det(DF)| * d_f2
        integrand = (f2_h_1 - f2_1) * metric_coeffs[0, 1] * (f2_h_2 - f2_2)
        error += 2 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 1 * d_f2 * G_22 / |det(DF)| * d_f2
        integrand = (f2_h_2 - f2_2) * metric_coeffs[1, 1] * (f2_h_2 - f2_2)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

        # 1 * d_f3 * G_33 / |det(DF)| * d_f3
        integrand = (f2_h_3 - f2_3) * metric_coeffs[2, 2] * (f2_h_3 - f2_3)
        error += 1 * integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

    else:
        # compute error in each element
        error = np.zeros(Nel[:2], dtype=float)

        # 1 * d_f1 * G_11 / |det(DF)| * d_f1
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [0, 1],
            [0, 1],
            basisN[0],
            basisD[1],
            basisN[0],
            basisD[1],
            indN[0],
            indD[1],
            indN[0],
            indD[1],
            error,
            f2_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f2_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c2_1,
            c2_1,
            1 * metric_coeffs[0, 0].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 2 * d_f1 * G_12 / |det(DF)| * d_f2
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [0, 1],
            [1, 0],
            basisN[0],
            basisD[1],
            basisD[0],
            basisN[1],
            indN[0],
            indD[1],
            indD[0],
            indN[1],
            error,
            f2_1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f2_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c2_1,
            c2_2,
            2 * metric_coeffs[0, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 1 * d_f2 * G_22 / |det(DF)| * d_f2
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [1, 0],
            [1, 0],
            basisD[0],
            basisN[1],
            basisD[0],
            basisN[1],
            indD[0],
            indN[1],
            indD[0],
            indN[1],
            error,
            f2_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f2_2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c2_2,
            c2_2,
            1 * metric_coeffs[1, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        # 1 * d_f3 * G_33 / |det(DF)| * d_f3
        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [1, 1],
            [1, 1],
            basisD[0],
            basisD[1],
            basisD[0],
            basisD[1],
            indD[0],
            indD[1],
            indD[0],
            indD[1],
            error,
            f2_3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f2_3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c2_3,
            c2_3,
            1 * metric_coeffs[2, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        error = error.sum()

    return np.sqrt(error)


# ======= error in V3 ====================
def l2_error_V3(tensor_space_FEM, domain, f3, c3, method="standard"):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    f3 : callable or np.ndarray
        the 3-form component with which the error shall be computed

    c3 : array_like
        the FEM coefficients of the discrete function

    method : string
        method used to compute the error (standard : integrals are computed with matrix-vector products, else: kernel is used)

    Returns
    -------
    error : int
        the L2-error of the discrete form
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indD = (
        tensor_space_FEM.indD
    )  # global indices of local non-vanishing basis functions in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points in format (element, local quad_point)
    wts = tensor_space_FEM.wts  # global quadrature weights in format (element, local weight)

    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points

    # extract c3icients to tensor-product space
    if c3.ndim == 1:
        c3 = tensor_space_FEM.extract_3(c3)

    assert c3.ndim == 3

    # evaluation of |det(DF)| at quadrature points
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))

    # evaluation of exact 3-form at quadrature points
    if callable(f3):
        quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing="ij")
        f3 = f3(quad_mesh[0], quad_mesh[1], 0.0)

    if method == "standard":
        # evaluation of discrete 3-form at quadrature points
        f3_h = tensor_space_FEM.evaluate_DD(pts[0].flatten(), pts[1].flatten(), np.array([0.0]), c3, "V3")[:, :, 0]

        # compute error
        error = 0.0

        integrand = (f3_h - f3) ** 2 / det_df
        error += integrand.dot(wts[1].flatten()).dot(wts[0].flatten())

    else:
        # compute error in each element
        error = np.zeros(Nel[:2], dtype=float)

        ker.kernel_l2error(
            Nel,
            p,
            n_quad,
            wts[0],
            wts[1],
            [1, 1],
            [1, 1],
            basisD[0],
            basisD[1],
            basisD[0],
            basisD[1],
            indD[0],
            indD[1],
            indD[0],
            indD[1],
            error,
            f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
            c3,
            c3,
            1 / det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1]),
        )

        error = error.sum()

    return np.sqrt(error)
