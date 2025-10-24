# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to assemble discrete derivatives.
"""

import scipy.sparse as spa

from struphy.utils.arrays import xp as np


# ================== 1d incident matrix =======================
def grad_1d_matrix(spl_kind, NbaseN):
    """
    Returns the 1d strong discrete gradient matrix (V0 --> V1) corresponding either periodic or clamped B-splines (without boundary conditions).

    Parameters
    ----------
    spl_kind : boolean
        periodic (True) or clamped (False) B-splines

    NbaseN : int
        total number of basis functions in V0

    Returns
    -------
    grad : array_like
        strong discrete gradient matrix
    """

    NbaseD = NbaseN - 1 + spl_kind

    grad = np.zeros((NbaseD, NbaseN), dtype=float)

    for i in range(NbaseD):
        grad[i, i] = -1.0
        grad[i, (i + 1) % NbaseN] = 1.0

    return grad


# ===== discrete derivatives in one dimension =================
def discrete_derivatives_1d(space):
    """
    Returns discrete derivatives for a 1d B-spline space.

    Parameters
    ----------
    space : spline_space_1d
        1d B-splines space
    """

    # 1D discrete derivative (full space)
    G = spa.csr_matrix(grad_1d_matrix(space.spl_kind, space.NbaseN))

    # apply boundary operators (reduced space)
    G0 = space.B1.dot(G.dot(space.B0.T))

    return G, G0


# ===== discrete derivatives in three dimensions ==============
def discrete_derivatives_3d(space):
    """
    Returns discrete derivatives for 2d and 3d B-spline spaces.

    Parameters
    ----------
    space : tensor_spline_space
        2d or 3d tensor-product B-splines space
    """

    # discrete derivative in 3rd dimension
    if space.dim == 3:
        grad_1d_3 = space.spaces[2].G.copy()
    else:
        if space.n_tor == 0:
            grad_1d_3 = 0 * spa.identity(1, format="csr")
        else:
            if space.basis_tor == "r":
                grad_1d_3 = 2 * np.pi * space.n_tor * spa.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))
            else:
                grad_1d_3 = 1j * 2 * np.pi * space.n_tor * spa.identity(1, format="csr")

    # standard tensor-product derivatives
    if space.ck == -1:
        # 1d derivatives and number of degrees of freedom in each direction
        grad_1d_1 = space.spaces[0].G.copy()
        grad_1d_2 = space.spaces[1].G.copy()

        n1 = grad_1d_1.shape[1]
        n2 = grad_1d_2.shape[1]
        n3 = grad_1d_3.shape[1]

        d1 = grad_1d_1.shape[0]
        d2 = grad_1d_2.shape[0]
        d3 = grad_1d_3.shape[0]

        # discrete grad (full space)
        G1 = spa.kron(spa.kron(grad_1d_1, spa.identity(n2)), spa.identity(n3))
        G2 = spa.kron(spa.kron(spa.identity(n1), grad_1d_2), spa.identity(n3))
        G3 = spa.kron(spa.kron(spa.identity(n1), spa.identity(n2)), grad_1d_3)

        G = [[G1], [G2], [G3]]
        G = spa.bmat(G, format="csr")

        # discrete curl (full space)
        C12 = spa.kron(spa.kron(spa.identity(n1), spa.identity(d2)), grad_1d_3)
        C13 = spa.kron(spa.kron(spa.identity(n1), grad_1d_2), spa.identity(d3))

        C21 = spa.kron(spa.kron(spa.identity(d1), spa.identity(n2)), grad_1d_3)
        C23 = spa.kron(spa.kron(grad_1d_1, spa.identity(n2)), spa.identity(d3))

        C31 = spa.kron(spa.kron(spa.identity(d1), grad_1d_2), spa.identity(n3))
        C32 = spa.kron(spa.kron(grad_1d_1, spa.identity(d2)), spa.identity(n3))

        C = [[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]]
        C = spa.bmat(C, format="csr")

        # discrete div (full space)
        D1 = spa.kron(spa.kron(grad_1d_1, spa.identity(d2)), spa.identity(d3))
        D2 = spa.kron(spa.kron(spa.identity(d1), grad_1d_2), spa.identity(d3))
        D3 = spa.kron(spa.kron(spa.identity(d1), spa.identity(d2)), grad_1d_3)

        D = [[D1, D2, D3]]
        D = spa.bmat(D, format="csr")

    # C^k polar derivatives
    else:
        # discrete grad (full space)
        G1 = spa.kron(space.polar_splines.G1.copy(), spa.identity(grad_1d_3.shape[1]))
        G2 = spa.kron(space.polar_splines.G2.copy(), spa.identity(grad_1d_3.shape[1]))
        G3 = spa.kron(spa.identity(space.polar_splines.Nbase0), grad_1d_3)

        G = [[G1], [G2], [G3]]
        G = spa.bmat(G, format="csr")

        # discrete curl (full space)
        C12 = spa.kron(space.polar_splines.VC.copy(), spa.identity(grad_1d_3.shape[0]))
        C21 = spa.kron(space.polar_splines.SC.copy(), spa.identity(grad_1d_3.shape[1]))

        C11_12 = -spa.kron(spa.identity(space.polar_splines.Nbase1C_2), grad_1d_3)
        C11_21 = spa.kron(spa.identity(space.polar_splines.Nbase1C_1), grad_1d_3)

        C11 = spa.bmat([[None, C11_12], [C11_21, None]])

        C = [[C11, C12], [C21, None]]
        C = spa.bmat(C, format="csr")

        # discrete div (full space)
        D1 = spa.kron(space.polar_splines.D1.copy(), spa.identity(grad_1d_3.shape[0]))
        D2 = spa.kron(space.polar_splines.D2.copy(), spa.identity(grad_1d_3.shape[0]))
        D3 = spa.kron(spa.identity(space.polar_splines.Nbase2), grad_1d_3)

        D = [[D1, D2, D3]]
        D = spa.bmat(D, format="csr")

    # apply boundary operators
    G0 = space.B1.dot(G.dot(space.B0.T))
    C0 = space.B2.dot(C.dot(space.B1.T))
    D0 = space.B3.dot(D.dot(space.B2.T))

    return G, G0, C, C0, D, D0
