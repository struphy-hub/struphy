# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute L2-errors of discrete p-forms with analytical forms in 3D.
"""

import scipy.sparse as spa

import struphy.eigenvalue_solvers.kernels_3d as ker
from struphy.utils.arrays import xp


# ======= error in V0 ====================
def l2_error_V0(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    fun : callable or xp.ndarray
        the 0-form with which the error shall be computed

    coeff : array_like
        the FEM coefficients of the discrete 0-form
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
    if coeff.ndim == 1:
        coeff = tensor_space_FEM.extract_0(coeff)

    assert coeff.ndim == 3

    # evaluation of |det(DF)| at quadrature points
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()))

    # evaluation of given 0-form at quadrature points
    mat_f = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)

    if callable(fun):
        quad_mesh = xp.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing="ij")
        mat_f[:, :, :] = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    else:
        mat_f[:, :, :] = fun

    # compute error
    error = xp.zeros(Nel, dtype=float)

    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 0, 0],
        [0, 0, 0],
        basisN[0],
        basisN[1],
        basisN[2],
        basisN[0],
        basisN[1],
        basisN[2],
        indN[0],
        indN[1],
        indN[2],
        indN[0],
        indN[1],
        indN[2],
        error,
        mat_f.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff,
        coeff,
        det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    return xp.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    fun : list of callables or xp.ndarrays
        the three 1-form components with which the error shall be computed

    coeff : list of array_like
        the FEM coefficients of the discrete components
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
    coeff1, coeff2, coeff3 = tensor_space_FEM.extract_1(coeff)

    # evaluation of G^(-1)*|det(DF)| at quadrature points
    metric_coeffs = domain.metric_inv(pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
    metric_coeffs *= abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()))

    # evaluation of given 1-form components at quadrature points
    mat_f1 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)
    mat_f2 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)
    mat_f3 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)

    if callable(fun[0]):
        quad_mesh = xp.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing="ij")
        mat_f1[:, :, :] = fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2])
        mat_f2[:, :, :] = fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2])
        mat_f3[:, :, :] = fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    else:
        mat_f1[:, :, :] = fun[0]
        mat_f2[:, :, :] = fun[1]
        mat_f3[:, :, :] = fun[2]

    # compute error
    error = xp.zeros(Nel, dtype=float)

    # 1 * f1 * G^11 * |det(DF)| * f1
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 0, 0],
        [1, 0, 0],
        basisD[0],
        basisN[1],
        basisN[2],
        basisD[0],
        basisN[1],
        basisN[2],
        [indD[0], indN[1], indN[2]],
        [indD[0], indN[1], indN[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff1,
        1 * metric_coeffs[0, 0].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f1 * G^12 * |det(DF)| * f2
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 0, 0],
        [0, 1, 0],
        basisD[0],
        basisN[1],
        basisN[2],
        basisN[0],
        basisD[1],
        basisN[2],
        [indD[0], indN[1], indN[2]],
        [indN[0], indD[1], indN[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff2,
        2 * metric_coeffs[0, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f1 * G^13 * |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 0, 0],
        [0, 0, 1],
        basisD[0],
        basisN[1],
        basisN[2],
        basisN[0],
        basisN[1],
        basisD[2],
        [indD[0], indN[1], indN[2]],
        [indN[0], indN[1], indD[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff3,
        2 * metric_coeffs[0, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 1 * f2 * G^22 * |det(DF)| * f2
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 1, 0],
        [0, 1, 0],
        basisN[0],
        basisD[1],
        basisN[2],
        basisN[0],
        basisD[1],
        basisN[2],
        [indN[0], indD[1], indN[2]],
        [indN[0], indD[1], indN[2]],
        error,
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff2,
        coeff2,
        1 * metric_coeffs[1, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f2 * G^23 * |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 1, 0],
        [0, 0, 1],
        basisN[0],
        basisD[1],
        basisN[2],
        basisN[0],
        basisN[1],
        basisD[2],
        [indN[0], indD[1], indN[2]],
        [indN[0], indN[1], indD[2]],
        error,
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff2,
        coeff3,
        2 * metric_coeffs[1, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 1 * f3 * G^33 * |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 0, 1],
        [0, 0, 1],
        basisN[0],
        basisN[1],
        basisD[2],
        basisN[0],
        basisN[1],
        basisD[2],
        [indN[0], indN[1], indD[2]],
        [indN[0], indN[1], indD[2]],
        error,
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff3,
        coeff3,
        1 * metric_coeffs[2, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    return xp.sqrt(error.sum())


# ======= error in V2 ====================
def l2_error_V2(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    fun : list of callables or xp.ndarrays
        the three 2-form components with which the error shall be computed

    coeff : list of array_like
        the FEM coefficients of the discrete components
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
    coeff1, coeff2, coeff3 = tensor_space_FEM.extract_2(coeff)

    # evaluation of G/|det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    metric_coeffs = domain.metric(pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
    metric_coeffs /= abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()))

    # evaluation of given 2-form components at quadrature points
    mat_f1 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)
    mat_f2 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)
    mat_f3 = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)

    if callable(fun[0]):
        quad_mesh = xp.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing="ij")
        mat_f1[:, :, :] = fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2])
        mat_f2[:, :, :] = fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2])
        mat_f3[:, :, :] = fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    else:
        mat_f1[:, :, :] = fun[0]
        mat_f2[:, :, :] = fun[1]
        mat_f3[:, :, :] = fun[2]

    # compute error
    error = xp.zeros(Nel, dtype=float)

    # 1 * f1 * G_11 / |det(DF)| * f1
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 1, 1],
        [0, 1, 1],
        basisN[0],
        basisD[1],
        basisD[2],
        basisN[0],
        basisD[1],
        basisD[2],
        [indN[0], indD[1], indD[2]],
        [indN[0], indD[1], indD[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff1,
        1 * metric_coeffs[0, 0].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f1 * G_12 / |det(DF)| * f2
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 1, 1],
        [1, 0, 1],
        basisN[0],
        basisD[1],
        basisD[2],
        basisD[0],
        basisN[1],
        basisD[2],
        [indN[0], indD[1], indD[2]],
        [indD[0], indN[1], indD[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff2,
        2 * metric_coeffs[0, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f1 * G_13 / |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [0, 1, 1],
        [1, 1, 0],
        basisN[0],
        basisD[1],
        basisD[2],
        basisD[0],
        basisD[1],
        basisN[2],
        [indN[0], indD[1], indD[2]],
        [indD[0], indD[1], indN[2]],
        error,
        mat_f1.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff1,
        coeff3,
        2 * metric_coeffs[0, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 1 * f2 * G_22 / |det(DF)| * f2
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 0, 1],
        [1, 0, 1],
        basisD[0],
        basisN[1],
        basisD[2],
        basisD[0],
        basisN[1],
        basisD[2],
        [indD[0], indN[1], indD[2]],
        [indD[0], indN[1], indD[2]],
        error,
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff2,
        coeff2,
        1 * metric_coeffs[1, 1].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 2 * f2 * G_23 / |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 0, 1],
        [1, 1, 0],
        basisD[0],
        basisN[1],
        basisD[2],
        basisD[0],
        basisD[1],
        basisN[2],
        [indD[0], indN[1], indD[2]],
        [indD[0], indD[1], indN[2]],
        error,
        mat_f2.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff2,
        coeff3,
        2 * metric_coeffs[1, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    # 1 * f3 * G_33 / |det(DF)| * f3
    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 1, 0],
        [1, 1, 0],
        basisD[0],
        basisD[1],
        basisN[2],
        basisD[0],
        basisD[1],
        basisN[2],
        [indD[0], indD[1], indN[2]],
        [indD[0], indD[1], indN[2]],
        error,
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f3.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff3,
        coeff3,
        1 * metric_coeffs[2, 2].reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    return xp.sqrt(error.sum())


# ======= error in V3 ====================
def l2_error_V3(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    fun : callable or xp.ndarray
        the 3-form component with which the error shall be computed

    coeff : array_like
        the FEM coefficients of the discrete function
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indD = (
        tensor_space_FEM.indD
    )  # global indices of non-vanishing basis functions (D) in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points in format (element, local quad_point)
    wts = tensor_space_FEM.wts  # global quadrature weights in format (element, local weight)

    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points

    # extract coefficients to tensor-product space
    if coeff.ndim == 1:
        coeff = tensor_space_FEM.extract_3(coeff)

    assert coeff.ndim == 3

    # evaluation of |det(DF)| at quadrature points
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()))

    # evaluation of given 3-form component at quadrature points
    mat_f = xp.empty((pts[0].size, pts[1].size, pts[2].size), dtype=float)

    if callable(fun):
        quad_mesh = xp.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing="ij")
        mat_f[:, :, :] = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    else:
        mat_f[:, :, :] = fun

    # compute error
    error = xp.zeros(Nel, dtype=float)

    ker.kernel_l2error(
        Nel,
        p,
        n_quad,
        wts[0],
        wts[1],
        wts[2],
        [1, 1, 1],
        [1, 1, 1],
        basisD[0],
        basisD[1],
        basisD[2],
        basisD[0],
        basisD[1],
        basisD[2],
        indD[0],
        indD[1],
        indD[2],
        indD[0],
        indD[1],
        indD[2],
        error,
        mat_f.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        mat_f.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
        coeff,
        coeff,
        1 / det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2]),
    )

    return xp.sqrt(error.sum())
