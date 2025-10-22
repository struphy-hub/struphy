# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Modules to obtain preconditioners for mass matrices in 3D.
"""

import scipy.sparse as spa

import struphy.eigenvalue_solvers.spline_space as spl
import struphy.linear_algebra.linalg_kron as linkron
from struphy.utils.arrays import xp


# ================ inverse mass matrix in V0 ===========================
def get_M0_PRE(tensor_space_FEM, domain):
    """
    TODO
    """

    # 1d spaces for pre-conditioning with fft:
    Nel_pre = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN]
    spl_kind_pre = [True, True, True]
    spaces_pre = [
        spl.Spline_space_1d(Nel, p, spl_kind, nq_el)
        for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)
    ]

    # tensor product mass matrices for pre-conditioning
    # spaces_pre[0].set_extraction_operators()
    # spaces_pre[1].set_extraction_operators()
    # spaces_pre[2].set_extraction_operators()

    spaces_pre[0].assemble_M0(lambda eta: (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    c_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    return spa.linalg.LinearOperator(
        shape=tensor_space_FEM.M0.shape,
        matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten(),
    )


# ================ inverse mass matrix in V1 ===========================
def get_M1_PRE(tensor_space_FEM, domain):
    """
    TODO
    """

    # 1d spaces for pre-conditioning with fft:
    Nel_pre = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN]
    spl_kind_pre = [True, True, True]
    spaces_pre = [
        spl.Spline_space_1d(Nel, p, spl_kind, nq_el)
        for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)
    ]

    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    # spaces_pre[0].set_extraction_operators()
    # spaces_pre[1].set_extraction_operators()
    # spaces_pre[2].set_extraction_operators()

    spaces_pre[0].assemble_M0(lambda eta: (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    spaces_pre[0].assemble_M1(lambda eta: 1 / (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta: 1 / (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta: 1 / (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    c11_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]

    def solve(x):
        x1, x2, x3 = xp.split(x, 3)

        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])

        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()

        return xp.concatenate((r1, r2, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1.shape, matvec=solve)


# ================ inverse mass matrix in V2 ===========================
def get_M2_PRE(tensor_space_FEM, domain):
    """
    TODO
    """

    # 1d spaces for pre-conditioning with fft:
    Nel_pre = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN]
    spl_kind_pre = [True, True, True]
    spaces_pre = [
        spl.Spline_space_1d(Nel, p, spl_kind, nq_el)
        for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)
    ]

    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    # spaces_pre[0].set_extraction_operators()
    # spaces_pre[1].set_extraction_operators()
    # spaces_pre[2].set_extraction_operators()

    spaces_pre[0].assemble_M0(lambda eta: (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    spaces_pre[0].assemble_M1(lambda eta: 1 / (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta: 1 / (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta: 1 / (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    def solve(x):
        x1, x2, x3 = xp.split(x, 3)

        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])

        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()

        return xp.concatenate((r1, r2, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2.shape, matvec=solve)


# ================ inverse mass matrix in V3 ===========================
def get_M3_PRE(tensor_space_FEM, domain):
    """
    TODO
    """

    # 1d spaces for pre-conditioning with fft:
    Nel_pre = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN]
    spl_kind_pre = [True, True, True]
    spaces_pre = [
        spl.Spline_space_1d(Nel, p, spl_kind, nq_el)
        for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)
    ]

    # tensor product mass matrices for pre-conditioning
    # spaces_pre[0].set_extraction_operators()
    # spaces_pre[1].set_extraction_operators()
    # spaces_pre[2].set_extraction_operators()

    spaces_pre[0].assemble_M1(lambda eta: 1 / (domain.params[1] - domain.params[0]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta: 1 / (domain.params[3] - domain.params[2]) * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta: 1 / (domain.params[5] - domain.params[4]) * xp.ones(eta.shape, dtype=float))

    c_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]

    return spa.linalg.LinearOperator(
        shape=tensor_space_FEM.M3.shape,
        matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten(),
    )


# ================ inverse mass matrix in V0^3 ===========================
def get_Mv_PRE(tensor_space_FEM, domain):
    """
    TODO
    """

    # 1d spaces for pre-conditioning with fft:
    Nel_pre = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN]
    spl_kind_pre = [True, True, True]
    spaces_pre = [
        spl.Spline_space_1d(Nel, p, spl_kind, nq_el)
        for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)
    ]

    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    # spaces_pre[0].set_extraction_operators()
    # spaces_pre[1].set_extraction_operators()
    # spaces_pre[2].set_extraction_operators()

    spaces_pre[0].assemble_M0(lambda eta: domain.params[0] ** 3 * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: domain.params[1] * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: domain.params[2] * xp.ones(eta.shape, dtype=float))

    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    spaces_pre[0].assemble_M0(lambda eta: domain.params[0] * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: domain.params[1] ** 3 * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: domain.params[2] * xp.ones(eta.shape, dtype=float))

    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    spaces_pre[0].assemble_M0(lambda eta: domain.params[0] * xp.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta: domain.params[1] * xp.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta: domain.params[2] ** 3 * xp.ones(eta.shape, dtype=float))

    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    def solve(x):
        x1, x2, x3 = xp.split(x, 3)

        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])

        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()

        return xp.concatenate((r1, r2, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv.shape, matvec=solve)


# ==== inverse mass matrix in V0 (with boundary conditions) with decomposition poloidal x toroidal ====
def get_M0_PRE_3(tensor_space_FEM, mats_pol=None):
    """
    TODO
    """

    if mats_pol == None:
        mat = tensor_space_FEM.B0_pol.dot(tensor_space_FEM.M0_pol_mat.dot(tensor_space_FEM.B0_pol.T))
    else:
        mat = mats_pol

    # LU decomposition of poloidal mass matrix
    M0_pol_0_LU = spa.linalg.splu(mat.tocsc())

    # vector defining the circulant mass matrix in toroidal direction
    tor_vec0 = tensor_space_FEM.M0_tor.toarray()[:, 0]

    def solve(x):
        x = x.reshape(tensor_space_FEM.E0_pol_0.shape[0], tensor_space_FEM.NbaseN[2])

        r = linkron.kron_fftsolve_2d(M0_pol_0_LU, tor_vec0, x).flatten()

        return r

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M0_0.shape, matvec=solve)


# ==== inverse mass matrix in V1 (with boundary conditions) with decomposition poloidal x toroidal ====
def get_M1_PRE_3(tensor_space_FEM, mats_pol=None):
    """
    TODO
    """

    if mats_pol == None:
        mat = [
            tensor_space_FEM.B1_pol.dot(tensor_space_FEM.M1_pol_mat[0].dot(tensor_space_FEM.B1_pol.T)),
            tensor_space_FEM.B0_pol.dot(tensor_space_FEM.M1_pol_mat[1].dot(tensor_space_FEM.B0_pol.T)),
        ]
    else:
        mat = mats_pol

    # LU decomposition of poloidal mass matrix
    M1_pol_0_11_LU = spa.linalg.splu(mat[0].tocsc())
    M1_pol_0_22_LU = spa.linalg.splu(mat[1].tocsc())

    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = tensor_space_FEM.M0_tor.toarray()[:, 0]
    tor_vec1 = tensor_space_FEM.M1_tor.toarray()[:, 0]

    def solve(x):
        x1 = x[: tensor_space_FEM.E1_pol_0.shape[0] * tensor_space_FEM.NbaseN[2]].reshape(
            tensor_space_FEM.E1_pol_0.shape[0], tensor_space_FEM.NbaseN[2]
        )
        x2 = x[tensor_space_FEM.E1_pol_0.shape[0] * tensor_space_FEM.NbaseN[2] :].reshape(
            tensor_space_FEM.E0_pol_0.shape[0], tensor_space_FEM.NbaseD[2]
        )

        r1 = linkron.kron_fftsolve_2d(M1_pol_0_11_LU, tor_vec0, x1).flatten()
        r2 = linkron.kron_fftsolve_2d(M1_pol_0_22_LU, tor_vec1, x2).flatten()

        return xp.concatenate((r1, r2))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1_0.shape, matvec=solve)


# ==== inverse mass matrix in V2 (with boundary conditions) with decomposition poloidal x toroidal ====
def get_M2_PRE_3(tensor_space_FEM, mats_pol=None):
    """
    TODO
    """

    if mats_pol == None:
        mat = [
            tensor_space_FEM.B2_pol.dot(tensor_space_FEM.M2_pol_mat[0].dot(tensor_space_FEM.B2_pol.T)),
            tensor_space_FEM.B3_pol.dot(tensor_space_FEM.M2_pol_mat[1].dot(tensor_space_FEM.B3_pol.T)),
        ]
    else:
        mat = mats_pol

    # LU decomposition of poloidal mass matrix
    M2_pol_0_11_LU = spa.linalg.splu(mat[0].tocsc())
    M2_pol_0_22_LU = spa.linalg.splu(mat[1].tocsc())

    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = tensor_space_FEM.M0_tor.toarray()[:, 0]
    tor_vec1 = tensor_space_FEM.M1_tor.toarray()[:, 0]

    def solve(x):
        x1 = x[: tensor_space_FEM.E2_pol_0.shape[0] * tensor_space_FEM.NbaseD[2]].reshape(
            tensor_space_FEM.E2_pol_0.shape[0], tensor_space_FEM.NbaseD[2]
        )
        x2 = x[tensor_space_FEM.E2_pol_0.shape[0] * tensor_space_FEM.NbaseD[2] :].reshape(
            tensor_space_FEM.E3_pol_0.shape[0], tensor_space_FEM.NbaseN[2]
        )

        r1 = linkron.kron_fftsolve_2d(M2_pol_0_11_LU, tor_vec1, x1).flatten()
        r2 = linkron.kron_fftsolve_2d(M2_pol_0_22_LU, tor_vec0, x2).flatten()

        return xp.concatenate((r1, r2))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2_0.shape, matvec=solve)


# ==== inverse mass matrix in V3 (with boundary conditions) with decomposition poloidal x toroidal ====
def get_M3_PRE_3(tensor_space_FEM, mats_pol=None):
    """
    TODO
    """

    if mats_pol == None:
        mat = tensor_space_FEM.B3_pol.dot(tensor_space_FEM.M3_pol_mat.dot(tensor_space_FEM.B3_pol.T))
    else:
        mat = mats_pol

    # LU decomposition of poloidal mass matrix
    M3_pol_0_LU = spa.linalg.splu(mat.tocsc())

    # vector defining the circulant mass matrix in toroidal direction
    tor_vec1 = tensor_space_FEM.M1_tor.toarray()[:, 0]

    def solve(x):
        x = x.reshape(tensor_space_FEM.E3_pol_0.shape[0], tensor_space_FEM.NbaseD[2])

        r = linkron.kron_fftsolve_2d(M3_pol_0_LU, tor_vec1, x).flatten()

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M3_0.shape, matvec=solve)


# ==== inverse mass matrix in V0^3 (with boundary conditions) with decomposition poloidal x toroidal ====
def get_Mv_PRE_3(tensor_space_FEM, mats_pol=None):
    """
    TODO
    """

    if mats_pol == None:
        mat = [
            tensor_space_FEM.Bv_pol.dot(tensor_space_FEM.Mv_pol_mat[0].dot(tensor_space_FEM.Bv_pol.T)),
            tensor_space_FEM.Mv_pol_mat[1],
        ]
    else:
        mat = mats_pol

    # LU decomposition of poloidal mass matrix
    Mv_pol_0_11_LU = spa.linalg.splu(mat[0].tocsc())
    Mv_pol_0_22_LU = spa.linalg.splu(mat[1].tocsc())

    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = tensor_space_FEM.M0_tor.toarray()[:, 0]

    def solve(x):
        x1 = x[: tensor_space_FEM.Ev_pol_0.shape[0] * tensor_space_FEM.NbaseN[2]].reshape(
            tensor_space_FEM.Ev_pol_0.shape[0], tensor_space_FEM.NbaseN[2]
        )
        x2 = x[tensor_space_FEM.Ev_pol_0.shape[0] * tensor_space_FEM.NbaseN[2] :].reshape(
            tensor_space_FEM.E0_pol.shape[0], tensor_space_FEM.NbaseN[2]
        )

        r1 = linkron.kron_fftsolve_2d(Mv_pol_0_11_LU, tor_vec0, x1).flatten()
        r2 = linkron.kron_fftsolve_2d(Mv_pol_0_22_LU, tor_vec0, x2).flatten()

        return xp.concatenate((r1, r2))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv_0.shape, matvec=solve)
