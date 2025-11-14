import cunumpy as xp
import scipy as sc
import scipy.sparse as spa
import scipy.special as sp

import struphy.bsplines.bsplines as bsp
import struphy.eigenvalue_solvers.derivatives as der
import struphy.eigenvalue_solvers.kernels_projectors_global_mhd as ker
import struphy.eigenvalue_solvers.legacy.inner_products_1d as inner
import struphy.eigenvalue_solvers.mass_matrices_1d as mass
import struphy.eigenvalue_solvers.projectors_global as pro
import struphy.eigenvalue_solvers.spline_space as spl


# numerical solution of the general ideal MHD eigenvalue problem in a cylinder using 1d B-splines in radial direction
def solve_ev_problem(rho, B_phi, dB_phi, B_z, p, gamma, a, k, m, num_params, bcZ):
    # 1d clamped B-spline space in [0, 1]
    splines = spl.Spline_space_1d(num_params[0], num_params[1], False, num_params[2])

    # mapping of radial coordinate from [0, 1] to [0, a]
    r = lambda eta: a * eta

    # jacobian for integration
    jac = lambda eta1: a * xp.ones(eta1.shape, dtype=float)

    # ========================== kinetic energy functional ==============================
    # integrands (multiplied by -2/omega**2)
    K_XX = lambda eta: rho(r(eta)) / r(eta)
    K_VV = lambda eta: rho(r(eta)) * r(eta)
    K_ZZ = lambda eta: rho(r(eta)) / r(eta) * (B_z(r(eta)) ** 2 + B_phi(r(eta)) ** 2)

    K_VZ = lambda eta: rho(r(eta)) * B_phi(r(eta))
    K_ZV = lambda eta: rho(r(eta)) * B_phi(r(eta))

    # compute matrices
    K_11 = mass.get_M(splines, 0, 0, K_XX, jac)[1:-1, 1:-1]
    K_22 = mass.get_M(splines, 2, 2, K_VV, jac)
    K_33 = mass.get_M(splines, 2, 2, K_ZZ, jac)[bcZ:, bcZ:]

    K_23 = mass.get_M(splines, 2, 2, K_VZ, jac)[:, bcZ:]
    K_32 = mass.get_M(splines, 2, 2, K_ZV, jac)[bcZ:, :]

    K = spa.bmat([[K_11, None, None], [None, K_22, K_23], [None, K_32, K_33]]).toarray()

    ## test correct computation
    # Bspline_A  = Bsp.Bspline(splines.T, splines.p    )
    # Bspline_B  = Bsp.Bspline(splines.t, splines.p - 1)
    #
    # K_11_scipy = xp.zeros((splines.NbaseN, splines.NbaseN), dtype=float)
    # K_22_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # K_33_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # K_23_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # K_32_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    #
    # for i in range(1, Bspline_A.N - 1):
    #    for j in range(1, Bspline_A.N - 1):
    #        integrand        = lambda eta : a*K_XX(eta)*Bspline_A(eta, i)*Bspline_A(eta, j)
    #        K_11_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]
    #
    # for i in range(Bspline_B.N):
    #    for j in range(Bspline_B.N):
    #        integrand        = lambda eta : a*K_VV(eta)*Bspline_B(eta, i)*Bspline_B(eta, j)
    #        K_22_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]
    #
    #        if bcZ == 0:
    #            integrand        = lambda eta : a*K_ZZ(eta)*Bspline_B(eta, i)*Bspline_B(eta, j)
    #            K_33_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]
    #        else:
    #            if i != 0 and j != 0:
    #                integrand        = lambda eta : a*K_ZZ(eta)*Bspline_B(eta, i)*Bspline_B(eta, j)
    #                K_33_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]
    #
    #        integrand        = lambda eta : a*K_VZ(eta)*Bspline_B(eta, i)*Bspline_B(eta, j)
    #        K_23_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]
    #
    #        integrand        = lambda eta : a*K_ZV(eta)*Bspline_B(eta, i)*Bspline_B(eta, j)
    #        K_32_scipy[i, j] = integrate.quad(integrand, 0., 1.)[0]

    # assert xp.allclose(K_11.toarray(), K_11_scipy[1:-1, 1:-1])
    # assert xp.allclose(K_22.toarray(), K_22_scipy            )
    # assert xp.allclose(K_33.toarray(), K_33_scipy[bcZ:, bcZ:])
    # assert xp.allclose(K_23.toarray(), K_23_scipy[ :  , bcZ:])
    # assert xp.allclose(K_32.toarray(), K_32_scipy[bcZ:,    :])

    # ========================== potential energy functional ===========================
    # integrands (multiplied by 2)
    W_XX = (
        lambda eta: B_phi(r(eta)) ** 2 * m**2 / r(eta) ** 3
        + 2 * B_phi(r(eta)) ** 2 / r(eta) ** 3
        - 2 * B_phi(r(eta)) * dB_phi(r(eta)) / r(eta) ** 2
        + 2 * B_phi(r(eta)) * B_z(r(eta)) * k * m / r(eta) ** 2
        + B_z(r(eta)) ** 2 * k**2 / r(eta)
    )

    W_VV = (
        lambda eta: B_z(r(eta)) ** 2 * k**2 * r(eta)
        + B_z(r(eta)) ** 2 * m**2 / r(eta)
        + gamma * m**2 * p(r(eta)) / r(eta)
    )

    W_ZZ = (
        lambda eta: B_phi(r(eta)) ** 2 * gamma * m**2 * p(r(eta)) / r(eta) ** 3
        + 2 * B_phi(r(eta)) * B_z(r(eta)) * gamma * k * m * p(r(eta)) / r(eta) ** 2
        + B_z(r(eta)) ** 2 * gamma * k**2 * p(r(eta)) / r(eta)
    )

    W_dXdX = lambda eta: B_phi(r(eta)) ** 2 / r(eta) + B_z(r(eta)) ** 2 / r(eta) + gamma * p(r(eta)) / r(eta)

    W_XdX = lambda eta: -2 * B_phi(r(eta)) ** 2 / r(eta) ** 2
    W_dXX = lambda eta: -2 * B_phi(r(eta)) ** 2 / r(eta) ** 2

    W_XV = lambda eta: 2 * B_phi(r(eta)) * B_z(r(eta)) * k / r(eta)
    W_dXV = (
        lambda eta: -B_phi(r(eta)) * B_z(r(eta)) * k + B_z(r(eta)) ** 2 * m / r(eta) + gamma * m * p(r(eta)) / r(eta)
    )

    W_VX = lambda eta: 2 * B_phi(r(eta)) * B_z(r(eta)) * k / r(eta)
    W_VdX = (
        lambda eta: -B_phi(r(eta)) * B_z(r(eta)) * k + B_z(r(eta)) ** 2 * m / r(eta) + gamma * m * p(r(eta)) / r(eta)
    )

    W_dXZ = lambda eta: B_phi(r(eta)) * gamma * m * p(r(eta)) / r(eta) ** 2 + B_z(r(eta)) * gamma * k * p(r(eta)) / r(
        eta,
    )
    W_ZdX = lambda eta: B_phi(r(eta)) * gamma * m * p(r(eta)) / r(eta) ** 2 + B_z(r(eta)) * gamma * k * p(r(eta)) / r(
        eta,
    )

    W_VZ = lambda eta: B_phi(r(eta)) * gamma * m**2 * p(r(eta)) / r(eta) ** 2 + B_z(r(eta)) * gamma * k * m * p(
        r(eta),
    ) / r(eta)
    W_ZV = lambda eta: B_phi(r(eta)) * gamma * m**2 * p(r(eta)) / r(eta) ** 2 + B_z(r(eta)) * gamma * k * m * p(
        r(eta),
    ) / r(eta)

    # compute matrices
    W_11 = mass.get_M(splines, 0, 0, W_XX, jac)[1:-1, 1:-1]
    W_11 += mass.get_M(splines, 1, 1, W_dXdX, jac)[1:-1, 1:-1]
    W_11 += mass.get_M(splines, 1, 0, W_dXX, jac)[1:-1, 1:-1]
    W_11 += mass.get_M(splines, 0, 1, W_XdX, jac)[1:-1, 1:-1]

    W_22 = mass.get_M(splines, 2, 2, W_VV, jac)
    W_33 = mass.get_M(splines, 2, 2, W_ZZ, jac)[bcZ:, bcZ:]

    W_12 = mass.get_M(splines, 0, 2, W_XV, jac)[1:-1, :]
    W_12 += mass.get_M(splines, 1, 2, W_dXV, jac)[1:-1, :]

    W_21 = mass.get_M(splines, 2, 0, W_VX, jac)[:, 1:-1]
    W_21 += mass.get_M(splines, 2, 1, W_VdX, jac)[:, 1:-1]

    W_13 = mass.get_M(splines, 1, 2, W_dXZ, jac)[1:-1, bcZ:]
    W_31 = mass.get_M(splines, 2, 1, W_ZdX, jac)[bcZ:, 1:-1]

    W_23 = mass.get_M(splines, 2, 2, W_VZ, jac)[:, bcZ:]
    W_32 = mass.get_M(splines, 2, 2, W_ZV, jac)[bcZ:, :]

    # W_11[0, :] = 0.00000001
    # W_11[:, 0] = 0.00000001

    # W_22[0, 0] = 0.00000001

    W = spa.bmat([[W_11, W_12, W_13], [W_21, W_22, W_23], [W_31, W_32, W_33]]).toarray()

    # return W_22
    ## test correct computation
    # W_11_scipy = xp.zeros((splines.NbaseN, splines.NbaseN), dtype=float)
    # W_22_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # W_33_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # W_12_scipy = xp.zeros((splines.NbaseN, splines.NbaseD), dtype=float)
    # W_21_scipy = xp.zeros((splines.NbaseD, splines.NbaseN), dtype=float)
    # W_13_scipy = xp.zeros((splines.NbaseN, splines.NbaseD), dtype=float)
    # W_31_scipy = xp.zeros((splines.NbaseD, splines.NbaseN), dtype=float)
    # W_23_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    # W_32_scipy = xp.zeros((splines.NbaseD, splines.NbaseD), dtype=float)
    #
    # for i in range(1, Bspline_A.N - 1):
    #    for j in range(1, Bspline_A.N - 1):
    #        integrand         = lambda eta :   a * W_XX(eta)   * Bspline_A(eta, i, 0) * Bspline_A(eta, j, 0)
    #        W_11_scipy[i, j] += integrate.quad(integrand, 0., 1.)[0]
    #
    #        integrand         = lambda eta : 1/a * W_dXdX(eta) * Bspline_A(eta, i, 1) * Bspline_A(eta, j, 1)
    #        W_11_scipy[i, j] += integrate.quad(integrand, 0., 1.)[0]
    #
    #        integrand         = lambda eta :       W_dXX(eta)  * Bspline_A(eta, i, 1) * Bspline_A(eta, j, 0)
    #        W_11_scipy[i, j] += integrate.quad(integrand, 0., 1.)[0]
    #
    #        integrand         = lambda eta :       W_XdX(eta)  * Bspline_A(eta, i, 0) * Bspline_A(eta, j, 1)
    #        W_11_scipy[i, j] += integrate.quad(integrand, 0., 1.)[0]
    #
    # assert xp.allclose(W_11.toarray(), W_11_scipy[1:-1, 1:-1])

    # print(xp.allclose(K, K.T))
    # print(xp.allclose(W, W.T))

    # solve eigenvalue problem omega**2*K*xi = W*xi
    A = xp.linalg.inv(K).dot(W)

    omega2, XVZ_eig = xp.linalg.eig(A)

    # extract components
    X_eig = XVZ_eig[: (splines.NbaseN - 2), :]
    V_eig = XVZ_eig[(splines.NbaseN - 2) : (splines.NbaseN - 2 + splines.NbaseD), :]
    Z_eig = XVZ_eig[(splines.NbaseN - 2 + splines.NbaseD) :, :]

    # add boundary conditions X(0) = X(1) = 0
    X_eig = xp.vstack((xp.zeros(X_eig.shape[1], dtype=float), X_eig, xp.zeros(X_eig.shape[1], dtype=float)))

    # add boundary condition Z(0) = 0
    if bcZ == 1:
        Z_eig = xp.vstack((xp.zeros(Z_eig.shape[1], dtype=float), Z_eig))

    return omega2, X_eig, V_eig, Z_eig


# numerical solution of the general ideal MHD eigenvalue problem in a cylinder using a 1d commuting diagram with B-splines in radial direction
def solve_ev_problem_FEEC(Rho, B_phi, dB_phi, B_z, dB_z, P, gamma, a, R0, n, m, num_params):
    # 1d clamped B-spline space in [0, 1]
    splines = spl.Spline_space_1d(num_params[0], num_params[1], False, num_params[2])
    proj = pro.projectors_global_1d(splines, num_params[3])
    GRAD = der.grad_1d_matrix(splines)[:, 1:-1]
    GRAD_all = der.grad_1d_matrix(splines)

    # mapping of radial coordinate from [0, 1] to [0, a]
    r = lambda eta: a * eta

    # components of metric tensor and Jacobian determinant
    G_r = a**2
    G_phi = lambda eta: 4 * xp.pi**2 * r(eta) ** 2
    G_z = 4 * xp.pi**2 * R0**2
    J = lambda eta: 4 * xp.pi**2 * R0 * a * r(eta)

    # 2-from components of equilibrium magnetic field and its projection
    B2_phi = lambda eta: 2 * xp.pi * R0 * a * B_phi(r(eta))
    B2_z = lambda eta: 2 * xp.pi * a * r(eta) * B_z(r(eta))

    b2_eq_phi = xp.linalg.solve(proj.D.toarray(), proj.rhs_1(B2_phi))
    b2_eq_z = xp.append(xp.array([0.0]), xp.linalg.solve(proj.D.toarray()[1:, 1:], proj.rhs_1(B2_z)[1:]))

    # 3-form components of equilibrium density and pessure and its projection
    Rho3 = lambda eta: J(eta) * Rho(r(eta))
    P3 = lambda eta: J(eta) * P(r(eta))

    rho3_eq = xp.append(xp.array([0.0]), xp.linalg.solve(proj.D.toarray()[1:, 1:], proj.rhs_1(Rho3)[1:]))
    p3_eq = xp.append(xp.array([0.0]), xp.linalg.solve(proj.D.toarray()[1:, 1:], proj.rhs_1(P3)[1:]))

    # 2-form components of initial velocity and its projection
    U2_r = lambda eta: J(eta) * eta * (1 - eta)

    u2_r = proj.pi_0(U2_r)
    u2_phi = -1 / (2 * xp.pi * m) * GRAD_all.dot(u2_r)
    u2_z = xp.zeros(len(u2_phi), dtype=float)

    b2_r = xp.zeros(len(u2_r), dtype=float)
    b2_phi = xp.zeros(len(u2_phi), dtype=float)
    b2_z = xp.zeros(len(u2_z), dtype=float)

    p3 = xp.zeros(len(u2_z), dtype=float)

    # projection matrices
    pi0_N_i, pi0_D_i, pi1_N_i, pi1_D_i = proj.projection_matrices_1d_reduced()
    pi0_NN_i, pi0_DN_i, pi0_ND_i, pi0_DD_i, pi1_NN_i, pi1_DN_i, pi1_ND_i, pi1_DD_i = proj.projection_matrices_1d()

    # 1D collocation matrices for interpolation in format (point, global basis function)
    x_int = xp.copy(proj.x_int)

    kind_splines = [False, True]

    basis_int_N = bsp.collocation_matrix(splines.T, splines.p, x_int, False, normalize=kind_splines[0])
    basis_int_D = bsp.collocation_matrix(splines.t, splines.p - 1, x_int, False, normalize=kind_splines[1])

    # 1D integration sub-intervals, quadrature points and weights
    if splines.p % 2 == 0:
        x_his = xp.union1d(x_int, splines.el_b)
    else:
        x_his = xp.copy(x_int)

    pts, wts = bsp.quadrature_grid(x_his, proj.pts_loc, proj.wts_loc)

    # compute number of sub-intervals for integrations (even degree)
    if splines.p % 2 == 0:
        subs = 2 * xp.ones(proj.pts.shape[0], dtype=int)

        subs[: splines.p // 2] = 1
        subs[-splines.p // 2 :] = 1

    # compute number of sub-intervals for integrations (odd degree)
    else:
        subs = xp.ones(proj.pts.shape[0], dtype=int)

    # evaluate basis functions on quadrature points in format (interval, local quad. point, global basis function)
    basis_his_N = bsp.collocation_matrix(splines.T, splines.p, pts.flatten(), False, normalize=kind_splines[0]).reshape(
        pts.shape[0],
        pts.shape[1],
        splines.NbaseN,
    )
    basis_his_D = bsp.collocation_matrix(
        splines.t,
        splines.p - 1,
        pts.flatten(),
        False,
        normalize=kind_splines[1],
    ).reshape(pts.shape[0], pts.shape[1], splines.NbaseD)

    # shift first interpolation point away from pole
    x_int[0] += 0.0001

    return basis_his_D

    # ====== mass matrices ============================================================
    M3 = mass.get_M1(splines, mapping=J).toarray()

    M2_r = mass.get_M0(splines, mapping=lambda eta: G_r / J(eta)).toarray()[1:-1, 1:-1]
    M2_phi = mass.get_M1(splines, mapping=lambda eta: J(eta) / G_phi(eta)).toarray()
    M2_z = mass.get_M1(splines, mapping=lambda eta: J(eta) / G_z).toarray()

    # === matrices for curl of equilibrium field (with integration by parts) ==========
    MB_12_eq = xp.empty((splines.NbaseN, splines.NbaseD), dtype=float)
    MB_13_eq = xp.empty((splines.NbaseN, splines.NbaseD), dtype=float)

    MB_21_eq = xp.empty((splines.NbaseD, splines.NbaseN), dtype=float)
    MB_31_eq = xp.empty((splines.NbaseD, splines.NbaseN), dtype=float)

    f_phi = xp.linalg.inv(proj.N.toarray()).T.dot(GRAD_all.T.dot(M2_phi.dot(b2_eq_phi)))
    f_z = xp.linalg.inv(proj.N.toarray()).T.dot(GRAD_all.T.dot(M2_z.dot(b2_eq_z)))

    pi0_ND_phi = xp.empty(pi0_ND_i[3].max() + 1, dtype=float)
    pi0_ND_z = xp.empty(pi0_ND_i[3].max() + 1, dtype=float)

    row_ND = xp.empty(pi0_ND_i[3].max() + 1, dtype=int)
    col_ND = xp.empty(pi0_ND_i[3].max() + 1, dtype=int)

    pi0_DN_phi = xp.empty(pi0_DN_i[3].max() + 1, dtype=float)
    pi0_DN_z = xp.empty(pi0_DN_i[3].max() + 1, dtype=float)

    row_DN = xp.empty(pi0_DN_i[3].max() + 1, dtype=int)
    col_DN = xp.empty(pi0_DN_i[3].max() + 1, dtype=int)

    ker.rhs0_f_1d(pi0_ND_i, basis_int_N, basis_int_D, 1 / J(x_int), f_phi, pi0_ND_phi, row_ND, col_ND)
    ker.rhs0_f_1d(pi0_ND_i, basis_int_N, basis_int_D, 1 / J(x_int), f_z, pi0_ND_z, row_ND, col_ND)

    ker.rhs0_f_1d(pi0_DN_i, basis_int_D, basis_int_N, 1 / J(x_int), f_phi, pi0_DN_phi, row_DN, col_DN)
    ker.rhs0_f_1d(pi0_DN_i, basis_int_D, basis_int_N, 1 / J(x_int), f_z, pi0_DN_z, row_DN, col_DN)

    pi0_ND_phi = spa.csr_matrix((pi0_ND_phi, (row_ND, col_ND)), shape=(splines.NbaseN, splines.NbaseD)).toarray()
    pi0_ND_z = spa.csr_matrix((pi0_ND_z, (row_ND, col_ND)), shape=(splines.NbaseN, splines.NbaseD)).toarray()

    pi0_DN_phi = spa.csr_matrix((pi0_DN_phi, (row_DN, col_DN)), shape=(splines.NbaseD, splines.NbaseN)).toarray()
    pi0_DN_z = spa.csr_matrix((pi0_DN_z, (row_DN, col_DN)), shape=(splines.NbaseD, splines.NbaseN)).toarray()

    MB_12_eq[:, :] = -pi0_ND_phi
    MB_13_eq[:, :] = -pi0_ND_z

    MB_21_eq[:, :] = -pi0_DN_phi
    MB_31_eq[:, :] = -pi0_DN_z

    # === matrices for curl of equilibrium field (without integration by parts) ======
    MB_12_eq = xp.empty((splines.NbaseN, splines.NbaseD), dtype=float)
    MB_13_eq = xp.empty((splines.NbaseN, splines.NbaseD), dtype=float)

    MB_21_eq = xp.empty((splines.NbaseD, splines.NbaseN), dtype=float)
    MB_31_eq = xp.empty((splines.NbaseD, splines.NbaseN), dtype=float)

    cN = xp.empty(splines.NbaseN, dtype=float)
    cD = xp.empty(splines.NbaseD, dtype=float)

    for j in range(splines.NbaseD):
        cD[:] = 0.0
        cD[j] = 1.0

        integrand2 = (
            lambda eta: splines.evaluate_D(eta, cD) / J(eta) * 2 * xp.pi * a * (B_phi(r(eta)) + r(eta) * dB_phi(r(eta)))
        )
        integrand3 = lambda eta: splines.evaluate_D(eta, cD) / J(eta) * 2 * xp.pi * a * R0 * dB_z(r(eta))

        MB_12_eq[:, j] = inner.inner_prod_V0(splines, integrand2)
        MB_13_eq[:, j] = inner.inner_prod_V0(splines, integrand3)

    for j in range(splines.NbaseN):
        cN[:] = 0.0
        cN[j] = 1.0

        integrand2 = (
            lambda eta: splines.evaluate_N(eta, cN) / J(eta) * 2 * xp.pi * a * (B_phi(r(eta)) + r(eta) * dB_phi(r(eta)))
        )
        integrand3 = lambda eta: splines.evaluate_N(eta, cN) / J(eta) * 2 * xp.pi * a * R0 * dB_z(r(eta))

        MB_21_eq[:, j] = inner.inner_prod_V1(splines, integrand2)
        MB_31_eq[:, j] = inner.inner_prod_V1(splines, integrand3)

    # ===== right-hand sides of projection matrices ===============
    rhs0_N_phi = xp.empty(pi0_N_i[0].size, dtype=float)
    rhs0_N_z = xp.empty(pi0_N_i[0].size, dtype=float)

    rhs1_D_phi = xp.empty(pi1_D_i[0].size, dtype=float)
    rhs1_D_z = xp.empty(pi1_D_i[0].size, dtype=float)

    rhs0_N_pr = xp.empty(pi0_N_i[0].size, dtype=float)
    rhs1_D_pr = xp.empty(pi1_D_i[0].size, dtype=float)

    rhs0_N_rho = xp.empty(pi0_N_i[0].size, dtype=float)
    rhs1_D_rho = xp.empty(pi1_D_i[0].size, dtype=float)

    # ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, splines.evaluate_D(x_int, b2_eq_phi)/J(x_int), rhs0_N_phi)
    # ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, splines.evaluate_D(x_int, b2_eq_z  )/J(x_int), rhs0_N_z  )
    #
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, (splines.evaluate_D(pts.flatten(), b2_eq_z  )/J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]), rhs1_D_z)
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, (splines.evaluate_D(pts.flatten(), b2_eq_phi)/J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]), rhs1_D_phi)
    #
    # ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, splines.evaluate_D(x_int, p3_eq)/J(x_int), rhs0_N_pr)
    # temp    = xp.empty(pi0_N_i[0].size, dtype=float)
    # temp[:] = rhs0_N_pr
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, (splines.evaluate_D(pts.flatten(), p3_eq)/J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]), rhs1_D_pr)
    #
    # ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, splines.evaluate_D(x_int, rho3)/J(x_int), rhs0_N_rho)
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, (splines.evaluate_D(pts.flatten(), rho3)/J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]), rhs1_D_rho)

    ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, B2_phi(x_int) / J(x_int), rhs0_N_phi)
    ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, B2_z(x_int) / J(x_int), rhs0_N_z)

    ker.rhs1_1d(
        pi1_D_i[0],
        pi1_D_i[1],
        subs,
        xp.append(0, xp.cumsum(subs - 1)[:-1]),
        wts,
        basis_his_D,
        (B2_phi(pts.flatten()) / J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]),
        rhs1_D_phi,
    )

    ker.rhs1_1d(
        pi1_D_i[0],
        pi1_D_i[1],
        subs,
        xp.append(0, xp.cumsum(subs - 1)[:-1]),
        wts,
        basis_his_D,
        (B2_z(pts.flatten()) / J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]),
        rhs1_D_z,
    )
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, xp.ones(pts.shape, dtype=float), rhs1_D_z)

    ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, P3(x_int) / J(x_int), rhs0_N_pr)
    ker.rhs1_1d(
        pi1_D_i[0],
        pi1_D_i[1],
        subs,
        xp.append(0, xp.cumsum(subs - 1)[:-1]),
        wts,
        basis_his_D,
        (P3(pts.flatten()) / J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]),
        rhs1_D_pr,
    )

    ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, Rho3(x_int) / J(x_int), rhs0_N_rho)
    ker.rhs1_1d(
        pi1_D_i[0],
        pi1_D_i[1],
        subs,
        xp.append(0, xp.cumsum(subs - 1)[:-1]),
        wts,
        basis_his_D,
        (Rho3(pts.flatten()) / J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]),
        rhs1_D_rho,
    )

    rhs0_N_phi = spa.csr_matrix(
        (rhs0_N_phi, (pi0_N_i[0], pi0_N_i[1])),
        shape=(splines.NbaseN, splines.NbaseN),
    ).toarray()
    rhs0_N_z = spa.csr_matrix((rhs0_N_z, (pi0_N_i[0], pi0_N_i[1])), shape=(splines.NbaseN, splines.NbaseN)).toarray()

    rhs1_D_phi = spa.csr_matrix(
        (rhs1_D_phi, (pi1_D_i[0], pi1_D_i[1])),
        shape=(splines.NbaseD, splines.NbaseD),
    ).toarray()
    rhs1_D_z = spa.csr_matrix((rhs1_D_z, (pi1_D_i[0], pi1_D_i[1])), shape=(splines.NbaseD, splines.NbaseD)).toarray()

    # return rhs1_D_z

    rhs0_N_pr = spa.csr_matrix((rhs0_N_pr, (pi0_N_i[0], pi0_N_i[1])), shape=(splines.NbaseN, splines.NbaseN)).toarray()
    rhs1_D_pr = spa.csr_matrix((rhs1_D_pr, (pi1_D_i[0], pi1_D_i[1])), shape=(splines.NbaseD, splines.NbaseD)).toarray()

    rhs0_N_rho = spa.csr_matrix(
        (rhs0_N_rho, (pi0_N_i[0], pi0_N_i[1])),
        shape=(splines.NbaseN, splines.NbaseN),
    ).toarray()
    rhs1_D_rho = spa.csr_matrix(
        (rhs1_D_rho, (pi1_D_i[0], pi1_D_i[1])),
        shape=(splines.NbaseD, splines.NbaseD),
    ).toarray()

    pi0_N_phi = xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_phi[1:-1, 1:-1])
    pi0_N_z = xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_z[1:-1, 1:-1])

    pi1_D_phi = xp.linalg.inv(proj.D.toarray()).dot(rhs1_D_phi)
    pi1_D_z = xp.linalg.inv(proj.D.toarray()).dot(rhs1_D_z)

    pi0_N_pr = xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_pr[1:-1, 1:-1])
    pi1_D_pr = xp.linalg.inv(proj.D.toarray()).dot(rhs1_D_pr)

    pi0_N_rho = xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_rho[1:-1, 1:-1])
    pi1_D_rho = xp.linalg.inv(proj.D.toarray()).dot(rhs1_D_rho)

    # ======= matrices in strong induction equation ================
    # 11 block
    I_11 = -2 * xp.pi * m * pi0_N_phi - 2 * xp.pi * n * pi0_N_z

    # 21 block and 31 block
    I_21 = -GRAD.dot(pi0_N_phi)
    I_31 = -GRAD.dot(pi0_N_z)

    # 22 block and 32 block
    I_22 = 2 * xp.pi * n * pi1_D_z
    I_32 = -2 * xp.pi * m * pi1_D_z

    # 23 block and 33 block
    I_23 = -2 * xp.pi * n * pi1_D_phi
    I_33 = 2 * xp.pi * m * pi1_D_phi

    # total
    I_all = xp.block(
        [
            [I_11, xp.zeros((len(u2_r) - 2, len(u2_phi))), xp.zeros((len(u2_r) - 2, len(u2_z) - 1))],
            [I_21, I_22, I_23[:, 1:]],
            [I_31[1:, :], I_32[1:, :], I_33[1:, 1:]],
        ],
    )

    # ======= matrices in strong pressure equation ================
    P_1 = -GRAD.dot(pi0_N_pr) - (gamma - 1) * pi1_D_pr.dot(GRAD)
    P_2 = -2 * xp.pi * m * gamma * pi1_D_pr
    P_3 = -2 * xp.pi * n * gamma * pi1_D_pr

    P_all = xp.block([[P_1[1:, :], P_2[1:, :], P_3[1:, 1:]]])

    # ========== matrices in weak momentum balance equation ======
    A_1 = 1 / 2 * (pi0_N_rho.T.dot(M2_r) + M2_r.dot(pi0_N_rho))
    A_2 = 1 / 2 * (pi1_D_rho.T.dot(M2_phi) + M2_phi.dot(pi1_D_rho))
    A_3 = 1 / 2 * (pi1_D_rho.T.dot(M2_z) + M2_z.dot(pi1_D_rho))[:, :]

    A_all = xp.block(
        [
            [A_1, xp.zeros((A_1.shape[0], A_2.shape[1])), xp.zeros((A_1.shape[0], A_3.shape[1]))],
            [xp.zeros((A_2.shape[0], A_1.shape[1])), A_2, xp.zeros((A_2.shape[0], A_3.shape[1]))],
            [xp.zeros((A_3.shape[0], A_1.shape[1])), xp.zeros((A_3.shape[0], A_2.shape[1])), A_3],
        ],
    )

    MB_11 = 2 * xp.pi * n * pi0_N_z.T.dot(M2_r) + 2 * xp.pi * m * pi0_N_phi.T.dot(M2_r)
    MB_12 = pi0_N_phi.T.dot(GRAD.T.dot(M2_phi)) - MB_12_eq[1:-1, :]
    MB_13 = pi0_N_z.T.dot(GRAD.T.dot(M2_z)) - MB_13_eq[1:-1, :]
    MB_14 = GRAD.T.dot(M3)

    MB_21 = MB_21_eq[:, 1:-1]
    MB_22 = -2 * xp.pi * n * pi1_D_z.T.dot(M2_phi)
    MB_23 = 2 * xp.pi * m * pi1_D_z.T.dot(M2_z)
    MB_24 = 2 * xp.pi * m * M3

    MB_31 = MB_31_eq[:, 1:-1]
    MB_32 = 2 * xp.pi * n * pi1_D_phi.T.dot(M2_phi)
    MB_33 = -2 * xp.pi * m * pi1_D_phi.T.dot(M2_z)
    MB_34 = 2 * xp.pi * n * M3

    MB_b_all = xp.block(
        [[MB_11, MB_12, MB_13[:, 1:]], [MB_21, MB_22, MB_23[:, 1:]], [MB_31[1:, :], MB_32[1:, :], MB_33[1:, 1:]]],
    )
    MB_p_all = xp.block([[MB_14[:, 1:]], [MB_24[:, 1:]], [MB_34[1:, 1:]]])

    ## ======= matrices in strong induction equation ================
    ## 11 block
    # I_11 = xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(-2*xp.pi*m*rhs0_N_phi[1:-1, 1:-1] - 2*xp.pi*n*rhs0_N_z[1:-1, 1:-1])
    #
    ## 21 block and 31 block
    # I_21 = -GRAD[: , 1:-1].dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_phi[1:-1, 1:-1]))
    # I_31 = -GRAD[1:, 1:-1].dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_z[1:-1, 1:-1]))
    #
    ## 22 block and 32 block
    # I_22 =  2*xp.pi*n*xp.linalg.inv(proj.D.toarray()[ :,  :]).dot(rhs1_D_z[ :, :])
    # I_32 = -2*xp.pi*m*xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_z[1:, :])
    #
    ## 23 block and 33 block
    # I_23 = -2*xp.pi*n*xp.linalg.inv(proj.D.toarray()[ :,  :]).dot(rhs1_D_phi[ :, 1:])
    # I_33 =  2*xp.pi*m*xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_phi[1:, 1:])
    #
    #
    ## ======= matrices in strong pressure equation ================
    # P_1 = -GRAD[1:, 1:-1].dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_pr[1:-1, 1:-1])) - (gamma - 1)*xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_pr[1:, :].dot(GRAD[:, 1:-1]))
    # P_2 = -2*xp.pi*m*gamma*xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_pr[1:,  :])
    # P_3 = -2*xp.pi*n*gamma*xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_pr[1:, 1:])
    #
    #
    ## ========== matrices in weak momentum balance equation ======
    # rhs0_N_rho = xp.empty(pi0_N_i[0].size, dtype=float)
    # ker.rhs0_1d(pi0_N_i[0], pi0_N_i[1], basis_int_N, splines.evaluate_D(x_int, rho3)/J(x_int), rhs0_N_rho)
    #
    #
    # rhs1_D_rho = xp.empty(pi1_D_i[0].size, dtype=float)
    # ker.rhs1_1d(pi1_D_i[0], pi1_D_i[1], subs, xp.append(0, xp.cumsum(subs - 1)[:-1]), wts, basis_his_D, (splines.evaluate_D(pts.flatten(), rho3)/J(pts.flatten())).reshape(pts.shape[0], pts.shape[1]), rhs1_D_rho)
    #
    #
    #
    # A_1 = 1/2*(rhs0_N_rho[1:-1, 1:-1].T.dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).T.dot(M2_r[1:-1, 1:-1])) + M2_r[1:-1, 1:-1].dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).dot(rhs0_N_rho[1:-1, 1:-1])))
    # A_2 = 1/2*(rhs1_D_rho.T.dot(xp.linalg.inv(proj.D.toarray()[:, :]).T.dot(M2_phi)) + M2_phi.dot(xp.linalg.inv(proj.D.toarray()[:, :]).dot(rhs1_D_rho)))
    # A_3 = 1/2*(rhs1_D_rho[1:, 1:].T.dot(xp.linalg.inv(proj.D.toarray()[1:, 1:]).T.dot(M2_z[1:, 1:])) + M2_z[1:, 1:].dot(xp.linalg.inv(proj.D.toarray()[1:, 1:]).dot(rhs1_D_rho[1:, 1:])))
    #
    #
    # MB_11 = 2*xp.pi*n*rhs0_N_z[1:-1, 1:-1].T.dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).T.dot(M2_r[1:-1, 1:-1])) + 2*xp.pi*m*rhs0_N_phi[1:-1, 1:-1].T.dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).T.dot(M2_r[1:-1, 1:-1]))
    #
    # MB_12 = rhs0_N_phi[1:-1, 1:-1].T.dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).T.dot(GRAD[:, 1:-1].T.dot(M2_phi)))
    # MB_13 = rhs0_N_z[1:-1, 1:-1].T.dot(xp.linalg.inv(proj.N.toarray()[1:-1, 1:-1]).T.dot(GRAD[1:, 1:-1].T.dot(M2_z[1:, 1:])))
    #
    # MB_14 = GRAD[1:, 1:-1].T.dot(M3[1:, 1:])
    #
    #
    # MB_22 = -2*xp.pi*n*rhs1_D_z.T.dot(xp.linalg.inv(proj.D.toarray()).T.dot(M2_phi))
    # MB_23 =  2*xp.pi*m*rhs1_D_z[1:, :].T.dot(xp.linalg.inv(proj.D.toarray()[1:, 1:]).T.dot(M2_z[1:, 1:]))
    # MB_24 =  2*xp.pi*m*M3[ :, 1:]
    #
    # MB_32 =  2*xp.pi*n*rhs1_D_phi[:,  1:].T.dot(xp.linalg.inv(proj.D.toarray()).T.dot(M2_phi))
    # MB_33 = -2*xp.pi*m*rhs1_D_phi[1:, 1:].T.dot(xp.linalg.inv(proj.D.toarray()[1:, 1:]).T.dot(M2_z[1:, 1:]))
    # MB_34 =  2*xp.pi*n*M3[1:, 1:]
    #
    #
    # ==== matrices in eigenvalue problem ========
    W_11 = MB_11.dot(I_11) + MB_12.dot(I_21) + MB_13.dot(I_31) + MB_14.dot(P_1)
    W_12 = MB_12.dot(I_22) + MB_13.dot(I_32) + MB_14.dot(P_2)
    W_13 = MB_12.dot(I_23) + MB_13.dot(I_33) + MB_14.dot(P_3)

    W_21 = MB_21.dot(I_11) + MB_22.dot(I_21) + MB_23.dot(I_31) + MB_24.dot(P_1)
    W_22 = MB_22.dot(I_22) + MB_23.dot(I_32) + MB_24.dot(P_2)
    W_23 = MB_22.dot(I_23) + MB_23.dot(I_33) + MB_24.dot(P_3)

    W_31 = MB_31.dot(I_11) + MB_32.dot(I_21) + MB_33.dot(I_31) + MB_34.dot(P_1)
    W_32 = MB_32.dot(I_22) + MB_33.dot(I_32) + MB_34.dot(P_2)
    W_33 = MB_32.dot(I_23) + MB_33.dot(I_33) + MB_34.dot(P_3)

    # W = xp.block([[W_11, W_12, W_13[:, 1:]], [W_21, W_22, W_23[:, 1:]], [W_31[1:, :], W_32[1:, :], W_33[1:, 1:]]])
    W = xp.block([[W_11, W_12, W_13[:, :]], [W_21, W_22, W_23[:, :]], [W_31[:, :], W_32[:, :], W_33[:, :]]])

    # print(xp.allclose(K, K.T))
    # print(xp.allclose(W, W.T))

    # solve eigenvalue problem omega**2*K*xi = W*xi
    MAT = xp.linalg.inv(-A_all).dot(W)

    omega2, XYZ_eig = xp.linalg.eig(MAT)
    # omega2, XYZ_eig = xp.linalg.eig(xp.linalg.inv(-A_all).dot(MB_b_all.dot(I_all) + MB_p_all.dot(P_all)))

    # extract components
    X_eig = XYZ_eig[: (splines.NbaseN - 2), :]
    Y_eig = XYZ_eig[(splines.NbaseN - 2) : (splines.NbaseN - 2 + splines.NbaseD), :]
    Z_eig = XYZ_eig[(splines.NbaseN - 2 + splines.NbaseD) :, :]

    # add boundary conditions X(0) = X(1) = 0
    X_eig = xp.vstack((xp.zeros(X_eig.shape[1], dtype=float), X_eig, xp.zeros(X_eig.shape[1], dtype=float)))

    # add boundary condition Z(0) = 0
    Z_eig = xp.vstack((xp.zeros(Z_eig.shape[1], dtype=float), Z_eig))

    return omega2, X_eig, Y_eig, Z_eig

    ## ========== matrices in initial value problem ===
    LHS = xp.block(
        [
            [A_all, xp.zeros((A_all.shape[0], A_all.shape[1])), xp.zeros((A_all.shape[0], len(p3) - 1))],
            [
                xp.zeros((A_all.shape[0], A_all.shape[1])),
                xp.identity(A_all.shape[0]),
                xp.zeros((A_all.shape[0], len(p3) - 1)),
            ],
            [
                xp.zeros((len(p3) - 1, A_all.shape[1])),
                xp.zeros((len(p3) - 1, A_all.shape[1])),
                xp.identity(len(p3) - 1),
            ],
        ],
    )

    RHS = xp.block(
        [
            [xp.zeros((MB_b_all.shape[0], I_all.shape[1])), MB_b_all, MB_p_all],
            [I_all, xp.zeros((I_all.shape[0], MB_b_all.shape[1])), xp.zeros((I_all.shape[0], MB_p_all.shape[1]))],
            [P_all, xp.zeros((P_all.shape[0], MB_b_all.shape[1])), xp.zeros((P_all.shape[0], MB_p_all.shape[1]))],
        ],
    )

    dt = 0.05
    T = 200.0
    Nt = int(T / dt)

    UPDATE = xp.linalg.inv(LHS - dt / 2 * RHS).dot(LHS + dt / 2 * RHS)
    ##UPDATE = xp.linalg.inv(LHS).dot(LHS + dt*RHS)
    #
    # lambdas, eig_vecs = xp.linalg.eig(UPDATE)

    # return lambdas
    #
    # return lambdas
    #
    u2_r_all = xp.zeros((Nt + 1, len(u2_r)), dtype=float)
    u2_phi_all = xp.zeros((Nt + 1, len(u2_phi)), dtype=float)
    u2_z_all = xp.zeros((Nt + 1, len(u2_z)), dtype=float)

    b2_r_all = xp.zeros((Nt + 1, len(b2_r)), dtype=float)
    b2_phi_all = xp.zeros((Nt + 1, len(b2_phi)), dtype=float)
    b2_z_all = xp.zeros((Nt + 1, len(b2_z)), dtype=float)

    p3_all = xp.zeros((Nt + 1, len(p3)), dtype=float)

    # initialization
    # u2_r_all[0, :]   = u2_r
    # u2_phi_all[0, :] = u2_phi

    u2_r_all[0, 1:-1] = xp.random.rand(len(u2_r) - 2)
    p3_all[0, 1:] = xp.random.rand(len(p3) - 1)

    # time integration
    for n in range(Nt):
        old = xp.concatenate(
            (
                u2_r_all[n, 1:-1],
                u2_phi_all[n, :],
                u2_z_all[n, 1:],
                b2_r_all[n, 1:-1],
                b2_phi_all[n, :],
                b2_z_all[n, 1:],
                p3_all[n, 1:],
            ),
        )
        new = UPDATE.dot(old)

        # extract components
        unew, bnew, pnew = xp.split(
            new,
            [len(u2_r) - 2 + len(u2_phi) + len(u2_z) - 1, 2 * (len(u2_r) - 2 + len(u2_phi) + len(u2_z) - 1)],
        )

        u2_r_all[n + 1, :] = xp.array([0.0] + list(unew[: (splines.NbaseN - 2)]) + [0.0])
        u2_phi_all[n + 1, :] = unew[(splines.NbaseN - 2) : (splines.NbaseN - 2 + splines.NbaseD)]
        u2_z_all[n + 1, :] = xp.array([0.0] + list(unew[(splines.NbaseN - 2 + splines.NbaseD) :]))

        b2_r_all[n + 1, :] = xp.array([0.0] + list(bnew[: (splines.NbaseN - 2)]) + [0.0])
        b2_phi_all[n + 1, :] = bnew[(splines.NbaseN - 2) : (splines.NbaseN - 2 + splines.NbaseD)]
        b2_z_all[n + 1, :] = xp.array([0.0] + list(bnew[(splines.NbaseN - 2 + splines.NbaseD) :]))

        p3_all[n + 1, :] = xp.array([0.0] + list(pnew))

    return u2_r_all, u2_phi_all, u2_z_all, b2_r_all, b2_phi_all, b2_z_all, p3_all, omega2
