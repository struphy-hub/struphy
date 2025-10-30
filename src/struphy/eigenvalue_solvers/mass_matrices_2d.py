# coding: utf-8
#
# Copyright 2020 Florian Holderied

import cunumpy as xp
import scipy.sparse as spa

import struphy.eigenvalue_solvers.kernels_2d as ker


# ================ mass matrix in V0 ===========================
def get_M0(tensor_space_FEM, domain, apply_boundary_ops=False, weight=None):
    """
    Assembles the 2D mass matrix [[NN NN]] * |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from struphy.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    apply_boundary_ops : boolean
        whether to include boundary operators (True) or not (False)

    weight : callable
        optional additional weight function
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

    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # evaluation of weight function at quadrature points
    if weight is None:
        mat_w = xp.ones(det_df.shape, dtype=float)
    else:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), 0.0)
        mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # assembly of global mass matrix
    Ni = tensor_space_FEM.Nbase_0form
    Nj = tensor_space_FEM.Nbase_0form

    M = xp.zeros((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1), dtype=float)

    ker.kernel_mass(
        xp.array(Nel),
        xp.array(p),
        xp.array(n_quad),
        xp.array([0, 0]),
        xp.array([0, 0]),
        wts[0],
        wts[1],
        basisN[0],
        basisN[1],
        basisN[0],
        basisN[1],
        indN[0],
        indN[1],
        M,
        mat_w * det_df,
    )

    # conversion to sparse matrix
    indices = xp.indices((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1))

    shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

    row = (Ni[1] * indices[0] + indices[1]).flatten()

    col1 = (indices[2] + shift[0][:, None, None, None]) % Nj[0]
    col2 = (indices[3] + shift[1][None, :, None, None]) % Nj[1]

    col = Nj[1] * col1 + col2

    M = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1], Nj[0] * Nj[1]))
    M.eliminate_zeros()

    # apply spline extraction operator and return
    if apply_boundary_ops:
        M = tensor_space_FEM.E0_pol_0.dot(M.dot(tensor_space_FEM.E0_pol_0.T)).tocsr()
    else:
        M = tensor_space_FEM.E0_pol.dot(M.dot(tensor_space_FEM.E0_pol.T)).tocsr()

    return M


# ================ mass matrix in V1 ===========================
def get_M1(tensor_space_FEM, domain, apply_boundary_ops=False, weight=None):
    """
    Assembles the 2D mass matrix [[DN DN, DN ND, DN NN], [ND DN, ND ND, ND NN], [NN DN, NN ND, NN NN]] * G^(-1) * |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from struphy.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    apply_boundary_ops : boolean
        whether to include boundary operators (True) or not (False)

    weight : callable
        optional additional weight functions
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

    # indices and basis functions of components of a 1-form
    ind = [[indD[0], indN[1]], [indN[0], indD[1]], [indN[0], indN[1]]]
    basis = [[basisD[0], basisN[1]], [basisN[0], basisD[1]], [basisN[0], basisN[1]]]
    ns = [[1, 0], [0, 1], [0, 0]]

    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # evaluation of G^(-1) at eta3 = 0 and quadrature points in format (3, 3, Nel1*nq1, Nel2*nq2)
    g_inv = domain.metric_inv(pts[0].flatten(), pts[1].flatten(), 0.0)

    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # assembly of blocks
    for a in range(3):
        for b in range(3):
            Ni = tensor_space_FEM.Nbase_1form[a]
            Nj = tensor_space_FEM.Nbase_1form[b]

            M[a][b] = xp.zeros((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1), dtype=float)

            # evaluate inverse metric tensor at quadrature points
            if weight is None:
                mat_w = g_inv[a, b]
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), 0.0)

            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

            # assemble block if weight is not zero
            if xp.any(mat_w):
                ker.kernel_mass(
                    xp.array(Nel),
                    xp.array(p),
                    xp.array(n_quad),
                    xp.array(ns[a]),
                    xp.array(ns[b]),
                    wts[0],
                    wts[1],
                    basis[a][0],
                    basis[a][1],
                    basis[b][0],
                    basis[b][1],
                    ind[a][0],
                    ind[a][1],
                    M[a][b],
                    mat_w * det_df,
                )

            # convert to sparse matrix
            indices = xp.indices((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1))

            shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

            row = (Ni[1] * indices[0] + indices[1]).flatten()

            col1 = (indices[2] + shift[0][:, None, None, None]) % Nj[0]
            col2 = (indices[3] + shift[1][None, :, None, None]) % Nj[1]

            col = Nj[1] * col1 + col2

            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1], Nj[0] * Nj[1]))
            M[a][b].eliminate_zeros()

    # apply extraction operators
    M11 = spa.bmat([[M[0][0], M[0][1]], [M[1][0], M[1][1]]])
    M22 = M[2][2]

    if apply_boundary_ops:
        M11 = tensor_space_FEM.E1_pol_0.dot(M11.dot(tensor_space_FEM.E1_pol_0.T)).tocsr()
        M22 = tensor_space_FEM.E0_pol_0.dot(M22.dot(tensor_space_FEM.E0_pol_0.T)).tocsr()
    else:
        M11 = tensor_space_FEM.E1_pol.dot(M11.dot(tensor_space_FEM.E1_pol.T)).tocsr()
        M22 = tensor_space_FEM.E0_pol.dot(M22.dot(tensor_space_FEM.E0_pol.T)).tocsr()

    return M11, M22


# ================ mass matrix in V2 ===========================
def get_M2(tensor_space_FEM, domain, apply_boundary_ops=False, weight=None):
    """
    Assembles the 2D mass matrix [[ND ND, ND DN, ND DD], [DN ND, DN DN, DN DD], [DD ND, DD DN, DD DD]] * G / |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from struphy.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    apply_boundary_ops : boolean
        whether to include boundary operators (True) or not (False)

    weight : callable
        optional additional weight functions
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

    # indices and basis functions of components of a 2-form
    ind = [[indN[0], indD[1]], [indD[0], indN[1]], [indD[0], indD[1]]]
    basis = [[basisN[0], basisD[1]], [basisD[0], basisN[1]], [basisD[0], basisD[1]]]
    ns = [[0, 1], [1, 0], [1, 1]]

    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # evaluation of G at eta3 = 0 and quadrature points in format (3, 3, Nel1*nq1, Nel2*nq2)
    g = domain.metric(pts[0].flatten(), pts[1].flatten(), 0.0)

    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # assembly of blocks
    for a in range(3):
        for b in range(3):
            Ni = tensor_space_FEM.Nbase_2form[a]
            Nj = tensor_space_FEM.Nbase_2form[b]

            M[a][b] = xp.zeros((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1), dtype=float)

            # evaluate metric tensor at quadrature points
            if weight is None:
                mat_w = g[a, b]
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), 0.0)

            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

            # assemble block if weight is not zero
            if xp.any(mat_w):
                ker.kernel_mass(
                    xp.array(Nel),
                    xp.array(p),
                    xp.array(n_quad),
                    xp.array(ns[a]),
                    xp.array(ns[b]),
                    wts[0],
                    wts[1],
                    basis[a][0],
                    basis[a][1],
                    basis[b][0],
                    basis[b][1],
                    ind[a][0],
                    ind[a][1],
                    M[a][b],
                    mat_w / det_df,
                )

            # convert to sparse matrix
            indices = xp.indices((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1))

            shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

            row = (Ni[1] * indices[0] + indices[1]).flatten()

            col1 = (indices[2] + shift[0][:, None, None, None]) % Nj[0]
            col2 = (indices[3] + shift[1][None, :, None, None]) % Nj[1]

            col = Nj[1] * col1 + col2

            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1], Nj[0] * Nj[1]))
            M[a][b].eliminate_zeros()

    # apply extraction operators
    M11 = spa.bmat([[M[0][0], M[0][1]], [M[1][0], M[1][1]]])
    M22 = M[2][2]

    if apply_boundary_ops:
        M11 = tensor_space_FEM.E2_pol_0.dot(M11.dot(tensor_space_FEM.E2_pol_0.T)).tocsr()
        M22 = tensor_space_FEM.E3_pol_0.dot(M22.dot(tensor_space_FEM.E3_pol_0.T)).tocsr()
    else:
        M11 = tensor_space_FEM.E2_pol.dot(M11.dot(tensor_space_FEM.E2_pol.T)).tocsr()
        M22 = tensor_space_FEM.E3_pol.dot(M22.dot(tensor_space_FEM.E3_pol.T)).tocsr()

    return M11, M22


# ================ mass matrix in V3 ===========================
def get_M3(tensor_space_FEM, domain, apply_boundary_ops=False, weight=None):
    """
    Assembles the 3D mass matrix [[DD DD]] / |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from struphy.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    apply_boundary_ops : boolean
        whether to include boundary operators (True) or not (False)

    weight : callable
        optional additional weight function
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

    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # evaluation of weight function at quadrature points
    if weight is None:
        mat_w = xp.ones(det_df.shape, dtype=float)
    else:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), 0.0)
        mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # assembly of global mass matrix
    Ni = tensor_space_FEM.Nbase_3form
    Nj = tensor_space_FEM.Nbase_3form

    M = xp.zeros((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1), dtype=float)

    ker.kernel_mass(
        xp.array(Nel),
        xp.array(p),
        xp.array(n_quad),
        xp.array([1, 1]),
        xp.array([1, 1]),
        wts[0],
        wts[1],
        basisD[0],
        basisD[1],
        basisD[0],
        basisD[1],
        indD[0],
        indD[1],
        M,
        mat_w / det_df,
    )

    # conversion to sparse matrix
    indices = xp.indices((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1))

    shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

    row = (Ni[1] * indices[0] + indices[1]).flatten()

    col1 = (indices[2] + shift[0][:, None, None, None]) % Nj[0]
    col2 = (indices[3] + shift[1][None, :, None, None]) % Nj[1]

    col = Nj[1] * col1 + col2

    M = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1], Nj[0] * Nj[1]))
    M.eliminate_zeros()

    # apply spline extraction operator and return
    if apply_boundary_ops:
        M = tensor_space_FEM.E3_pol_0.dot(M.dot(tensor_space_FEM.E3_pol_0.T)).tocsr()
    else:
        M = tensor_space_FEM.E3_pol.dot(M.dot(tensor_space_FEM.E3_pol.T)).tocsr()

    return M


# ============= mass matrix of vector field =========================
def get_Mv(tensor_space_FEM, domain, apply_boundary_ops=False, weight=None):
    """
    Assembles the 2D mass matrix [[NN NN, NN NN, NN NN], [NN NN, NN NN, NN NN], [NN NN, NN NN, NN NN]] * G * |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from struphy.geometry.domain.

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    domain : domain
        domain object defining the geometry

    apply_boundary_ops : boolean
        whether to include boundary operators (True) or not (False)

    weight : callable
        optional additional weight functions
    """

    p = tensor_space_FEM.p  # spline degrees
    Nel = tensor_space_FEM.Nel  # number of elements
    indN = (
        tensor_space_FEM.indN
    )  # global indices of non-vanishing basis functions (N) in format (element, global index)

    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts = tensor_space_FEM.pts  # global quadrature points
    wts = tensor_space_FEM.wts  # global quadrature weights

    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)

    # indices and basis functions of components of a 0-form
    ind = [[indN[0], indN[1]], [indN[0], indN[1]], [indN[0], indN[1]]]
    basis = [[basisN[0], basisN[1]], [basisN[0], basisN[1]], [basisN[0], basisN[1]]]
    ns = [[0, 0], [0, 0], [0, 0]]

    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.jacobian_det(pts[0].flatten(), pts[1].flatten(), 0.0))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

    # evaluation of G at eta3 = 0 and quadrature points in format (3, 3, Nel1*nq1, Nel2*nq2)
    g = domain.metric(pts[0].flatten(), pts[1].flatten(), 0.0)

    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # assembly of blocks
    for a in range(3):
        for b in range(3):
            Ni = tensor_space_FEM.Nbase_0form
            Nj = tensor_space_FEM.Nbase_0form

            M[a][b] = xp.zeros((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1), dtype=float)

            # evaluate metric tensor at quadrature points
            if weight is None:
                mat_w = g[a, b]
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), 0.0)

            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])

            # assemble block if weight is not zero
            if xp.any(mat_w):
                ker.kernel_mass(
                    xp.array(Nel),
                    xp.array(p),
                    xp.array(n_quad),
                    xp.array(ns[a]),
                    xp.array(ns[b]),
                    wts[0],
                    wts[1],
                    basis[a][0],
                    basis[a][1],
                    basis[b][0],
                    basis[b][1],
                    ind[a][0],
                    ind[a][1],
                    M[a][b],
                    mat_w * det_df,
                )

            # convert to sparse matrix
            indices = xp.indices((Ni[0], Ni[1], 2 * p[0] + 1, 2 * p[1] + 1))

            shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

            row = (Ni[1] * indices[0] + indices[1]).flatten()

            col1 = (indices[2] + shift[0][:, None, None, None]) % Nj[0]
            col2 = (indices[3] + shift[1][None, :, None, None]) % Nj[1]

            col = Nj[1] * col1 + col2

            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1], Nj[0] * Nj[1]))
            M[a][b].eliminate_zeros()

    # apply extraction operators
    M11 = spa.bmat([[M[0][0], M[0][1]], [M[1][0], M[1][1]]])
    M22 = M[2][2]

    if apply_boundary_ops:
        M11 = tensor_space_FEM.Ev_pol_0.dot(M11.dot(tensor_space_FEM.Ev_pol_0.T)).tocsr()
        M22 = tensor_space_FEM.E0_pol.dot(M22.dot(tensor_space_FEM.E0_pol.T)).tocsr()
    else:
        M11 = tensor_space_FEM.Ev_pol.dot(M11.dot(tensor_space_FEM.Ev_pol.T)).tocsr()
        M22 = tensor_space_FEM.E0_pol.dot(M22.dot(tensor_space_FEM.E0_pol.T)).tocsr()

    return M11, M22
