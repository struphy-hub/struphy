"""Matrix-vector product and solvers for matrices with a 3D Kronecker product structure.

M_ijkmno = A_im * B_jn * C_ko

where matrices A, B, C stem from 1D problems.

COMMENT: the reshape of a matrix can be viewed as ravel+reshape.
    Let r = (r_ijk) be a 3D matrix of size M*N*O.
    ravel(r) = [r_111, r112, ... , r_MNO] (row major always --> last index runs fastest)
    reshape(ravel(r), (M, N*O)) = [[r_111, r112, ... , r_1NO],
                                    [r_211, r212, ... , r_2NO],
                                    ...,
                                    [r_M11, rM12, ... , r_MNO]]
"""

import cunumpy as xp
from scipy.linalg import solve_circulant
from scipy.sparse.linalg import splu


def kron_matvec_2d(kmat, vec2d):
    """
    2D Kronecker matrix-vector product.

    res_ij = (A_ik * B_jl) * vec2d_kl

    implemented as two matrix-matrix multiplications with intermediate transpose:

    step1(v1,k0) = ( kmat[0](k0,v0) * vec2d(v0,v1) )^T
    step2(k0,k1) = ( kmat[1](k1,v1) * step1(v1,k0) )^T
    res = step2(k0,k1)

    Parameters
    ----------
    kmat : 2 sparse matrices for each direction of size (k0,v0) and (k1,v1)

    vec2d : 2d array of size (v0,v1)

    Returns
    -------
    res : 2d array of size (k0,k1)
    """

    res = kmat[1].dot(kmat[0].dot(vec2d).T).T

    return res


def kron_matvec_3d(kmat, vec3d):
    """3D Kronecker matrix-vector product.

    res_ijk = (A_im * B_jn * C_ko) * vec3d_mno

    implemented as three matrix-matrix multiplications with intermediate reshape and transpose.
    step1(v1*v2,k0) <= ( kmat[0](k0,v0) * reshaped_vec3d(v0,v1*v2) )^T
    step2(v2*k0,k1) <= ( kmat[1](k1,v1) * reshaped_step1(v1,v2*k0) )^T
    step3(k0*k1*k2) <= ( kmat[2](k2,v2) * reshaped_step2(v2,k0*k1) )^T
    res <= reshaped_step3(k0,k1,k2)

    no overhead of numpy reshape command, as they do NOT copy the data.

    Parameters
    ----------
    kmat : 3 sparse matrices for each direction,
              of size (k0,v0),(k1,v1),(k2,v2)

    vec3d : 3d array of size (v0,v1,v2)


    Returns
    -------
    res : 3d array of size (k0,k1,k2)
    """

    v0, v1, v2 = vec3d.shape

    k0 = kmat[0].shape[0]
    k1 = kmat[1].shape[0]
    k2 = kmat[2].shape[0]

    res = (
        (
            kmat[2].dot(
                ((kmat[1].dot(((kmat[0].dot(vec3d.reshape(v0, v1 * v2))).T).reshape(v1, v2 * k0))).T).reshape(
                    v2, k0 * k1
                )
            )
        ).T
    ).reshape(k0, k1, k2)

    return res


def kron_matvec_3d_1(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k0 = kmat.shape[0]

    res = kmat.dot(vec3d.reshape(v0, v1 * v2)).reshape(k0, v1, v2)

    return res


def kron_matvec_3d_2(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k1 = kmat.shape[0]

    res = ((kmat.dot(((vec3d.reshape(v0, v1 * v2)).T).reshape(v1, v2 * v0)).reshape(k1 * v2, v0)).T).reshape(v0, k1, v2)

    return res


def kron_matvec_3d_3(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k2 = kmat.shape[0]

    res = (kmat.dot((vec3d.reshape(v0 * v1, v2)).T).T).reshape(v0, v1, k2)

    return res


def kron_matvec_3d_23(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k1 = kmat[0].shape[0]
    k2 = kmat[1].shape[0]

    res = (
        kmat[1].dot((kmat[0].dot(((vec3d.reshape(v0, v1 * v2)).T).reshape(v1, v2 * v0)).T).reshape(v2, v0 * k1)).T
    ).reshape(v0, k1, k2)

    return res


def kron_matvec_3d_13(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k0 = kmat[0].shape[0]
    k2 = kmat[1].shape[0]

    res = (kmat[1].dot((kmat[0].dot(vec3d.reshape(v0, v1 * v2)).reshape(k0 * v1, v2)).T).T).reshape(k0, v1, k2)

    return res


def kron_matvec_3d_12(kmat, vec3d):
    """ """

    v0, v1, v2 = vec3d.shape

    k0 = kmat[0].shape[0]
    k1 = kmat[1].shape[0]

    res = (
        (kmat[1].dot((kmat[0].dot(vec3d.reshape(v0, v1 * v2)).T).reshape(v1, v2 * k0)).reshape(k1 * v2, k0)).T
    ).reshape(k0, k1, v2)

    return res


def kron_matmat_fft_3d(a_vec, b_vec):
    """
    matrix-matrix product between 3d kronecker circulant matrices.

    res_ijk,qrs = (A_im * B_jn * C_ko) * (D_mq * E_nr * F_os)

    the 1d matrix-matrix product is computed as
    A*D = ifft(fft(a)*fft(d))

    This relies on the fact that the eigenvalues of a circulant matrix are given by

    lambda = fft(c) = ifft(c)*N for c real

    Parameters
    ----------
    a_vec : 3 vectors for each direction defining the circulant matrices of the first matrix

    b_vec : 3 vectors for each direction defining the circulant matrices of the second matrix

    Returns
    -------
    c_vec : 3 vectors for each direction defining the resulting circulant matrices
    """

    c_vec = [0, 0, 0]

    c_vec[0] = xp.fft.ifft(xp.fft.fft(a_vec[0]) * xp.fft.fft(b_vec[0]))
    c_vec[1] = xp.fft.ifft(xp.fft.fft(a_vec[1]) * xp.fft.fft(b_vec[1]))
    c_vec[2] = xp.fft.ifft(xp.fft.fft(a_vec[2]) * xp.fft.fft(b_vec[2]))

    return c_vec


# def kron_bmatbmat_fft_3d(a_list, b_list):
#    """
#    matrix-matrix product between two 3x3 block matrices where each block is a 3d kronecker circulant matrix.
#
#    Parameters
#    ----------
#    a_list : nested 3 x 3 x 3 list vectors for each direction defining the circulant matrices of the first matrix
#
#    b_list : nested 3 x 3 x 3 list vectors for each direction defining the circulant matrices of the second matrix
#
#    Returns
#    -------
#    c_list : nested 3 x 3 x 3 list
#    """
#
#    c_vec = []
#
#    for i in range(3):
#        for j in range(3):
#
#            I = a_list[i][j]
#            J = b_list[i][j]
#
#            A = spa.csr_matrix((I[0].size*I[1].size*I[2].size, J[0].size*J[1].size*J[2].size), dtype=float)
#
#            for k in range(3):
#
#                loc_vec = kron_matmat_fft_3d(a_vec[i][k], b_vec[k][j])
#
#                a = spa.linalg.circulant(loc_vec[0])
#                b = spa.linalg.circulant(loc_vec[1])
#                c = spa.linalg.circulant(loc_vec[2])
#
#                A += spa.kron(a, spa.kron(b, c), format='csr')
#
#
#            c_vec.append(A[:, 0])
#
#
#
#    return c_vec


def kron_lusolve_3d(kmatlu, rhs):
    """3D Kronecker LU solver.

    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)

    no overhead of numpy reshape command, as they do NOT copy the data.

    Parameters
    ----------
    kmatlu : 3 already LU decompositions of sparse matrices for each direction,
              of size (r0,r0),(r1,r1),(r2,r2)

    rhs : 3d array of size (r0,r1,r2), right-hand size


    Returns
    -------
    res : 3d array of size (r0,r1,r2), solution
    """

    r0, r1, r2 = rhs.shape

    res = (
        (
            kmatlu[2].solve(
                ((kmatlu[1].solve(((kmatlu[0].solve(rhs.reshape(r0, r1 * r2))).T).reshape(r1, r2 * r0))).T).reshape(
                    r2, r0 * r1
                )
            )
        ).T
    ).reshape(r0, r1, r2)

    return res


def kron_solve_3d(kmat, rhs):
    """3D Kronecker solver.

    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)

    no overhead of numpy reshape command, as they do NOT copy the data.

    Parameters
    ----------
    kmat : 3 sparse matrices for each direction,
              of size (r0,r0),(r1,r1),(r2,r2)

    rhs : 3d array of size (r0,r1,r2), right-hand size


    Returns
    -------
    res : 3d array of size (r0,r1,r2), solution
    """

    r0, r1, r2 = rhs.shape

    res = (
        (
            splu(kmat[2]).solve(
                (
                    (splu(kmat[1]).solve(((splu(kmat[0]).solve(rhs.reshape(r0, r1 * r2))).T).reshape(r1, r2 * r0))).T
                ).reshape(r2, r0 * r1)
            )
        ).T
    ).reshape(r0, r1, r2)

    return res


def kron_fftsolve_3d(cvec, rhs):
    """3D Kronecker fft solver for circulant matrices.

    solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

    implemented as three matrix-matrix solve with intermediate reshape and transpose.
    step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
    step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
    step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
    res <= reshaped_step3(r0,r1,r2)

    no overhead of numpy reshape command, as they do NOT copy the data.

    Parameters
        ----------
        cvec   : 3 vectors of size (r0),(r1),(r2) defining 3 circulant matrices for each direction,

        rhs   : 3d array of size (r0,r1,r2), right-hand size


        Returns
        -------
        res : 3d array of size (r0,r1,r2), solution

    """
    r0, r1, r2 = rhs.shape
    res = (
        (
            solve_circulant(
                cvec[2],
                (
                    (
                        solve_circulant(
                            cvec[1], ((solve_circulant(cvec[0], rhs.reshape(r0, r1 * r2))).T).reshape(r1, r2 * r0)
                        )
                    ).T
                ).reshape(r2, r0 * r1),
            )
        ).T
    ).reshape(r0, r1, r2)
    return res


# ---------------------- 2d ---------------------------------
def kron_lusolve_2d(kmatlu, rhs):
    """2D Kronecker LU solver.

    solve for x: (A_im * B_jn) * x_mn =  rhs_ij

    Parameters
    ----------
    kmatlu : 2 already LU decompositions of sparse matrices for each direction,
              of size (r0,r0),(r1,r1)

    rhs : 2d array of size (r0,r1), right-hand size

    Returns
    -------
    res : 2d array of size (r0,r1), solution
    """

    res = (kmatlu[1].solve((kmatlu[0].solve(rhs)).T)).T

    return res


def kron_fftsolve_2d(A_LU, b_vec, rhs):
    """
    ALERT: docstring wrong.

    Solve for 3d vector, matrix would be a 3d kronecker circulant matrix,
        but system is only solved in each direction.

        solve for x: (A_im * B_jn * C_ko) * x_mno =  rhs_ijk

        implemented as three matrix-matrix solve with intermediate reshape and transpose.
        step1(r1*r2,r0) <= ( A(r0,r0)^-1 *   reshaped_rhs(r0,r1*r2) )^T
        step2(r2*r0,r1) <= ( B(r1,r1)^-1 * reshaped_step1(r1,r2*r0) )^T
        step3(r0*r1*r2) <= ( C(r2,r2)^-1 * reshaped_step2(r2,r0*r1) )^T
        res <= reshaped_step3(r0,r1,r2)

        no overhead of numpy reshape command, as they do NOT copy the data.

        COMMENT: the reshape of a matrix can be viewed as ravel+reshape.
        Let r = (r_ijk) be a 3D matrix of size M*N*O.
        ravel(r) = [r_111, r112, ... , r_MNO] (row major always --> last index runs fastest)
        reshape(ravel(r), (M, N*O)) = [[r_111, r112, ... , r_1NO],
                                       [r_211, r212, ... , r_2NO],
                                       ...,
                                       [r_M11, rM12, ... , r_MNO]]

    Parameters
        ----------
        cvec   : 3 vectors of size (r0),(r1),(r2) defining 3 circulant matrices for each direction,

        rhs   : 3d array of size (r0,r1,r2), right-hand size


        Returns
        -------
        res : 3d array of size (r0,r1,r2), solution

    """
    res = solve_circulant(b_vec, A_LU.solve(rhs).T).T

    return res.flatten()
