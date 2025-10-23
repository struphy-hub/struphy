# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


import cunumpy as xp
import scipy.sparse as spa

import struphy.eigenvalue_solvers.kernels_projectors_global_mhd as ker
import struphy.eigenvalue_solvers.mass_matrices_2d as mass_2d
import struphy.eigenvalue_solvers.mass_matrices_3d as mass_3d


class MHDOperatorsCore:
    """
    Core class for degree of freedom matrices related to ideal MHD equations.

    Parameters
    ----------
        space : tensor_spline_space
            2D or 3D B-spline finite element space.

        equilibrium : equilibrium_mhd
            MHD equilibrium object.

        basis_u : int
            representation of velocity field (0 : vector field where all components are treated as 0-forms, 2 : 2-form).
    """

    def __init__(self, space, equilibrium, basis_u):
        # tensor-product spline space (either 3D or 2D x Fourier)
        self.space = space

        # MHD equilibrium object for evaluation of equilibrium fields
        self.equilibrium = equilibrium

        # bulk veloctiy formulation (either vector field or 2-form)
        assert basis_u == 0 or basis_u == 2
        self.basis_u = basis_u

        # get 1D interpolation points (copies) and shift first point in raidla (eta_1) direction for polar domains
        self.eta_int = [space.projectors.x_int.copy() for space in self.space.spaces]

        if self.space.ck == 0 or self.space.ck == 1:
            self.eta_int[0][0] += 0.00001

        self.nint = [eta_int.size for eta_int in self.eta_int]

        # get 1D quadrature points and weights
        self.eta_his = [space.projectors.pts for space in self.space.spaces]
        self.wts = [space.projectors.wts for space in self.space.spaces]

        self.nhis = [eta_his.shape[0] for eta_his in self.eta_his]
        self.nq = [eta_his.shape[1] for eta_his in self.eta_his]

        # get 1D number of sub-integration intervals
        self.subs = [space.projectors.subs for space in self.space.spaces]
        self.subs_cum = [space.projectors.subs_cum for space in self.space.spaces]

        # get 1D indices of non-vanishing values of expressions dofs_0(N), dofs_0(D), dofs_1(N) and dofs_1(D)
        self.dofs_0_N_i = [list(xp.nonzero(space.projectors.I.toarray())) for space in self.space.spaces]
        self.dofs_1_D_i = [list(xp.nonzero(space.projectors.H.toarray())) for space in self.space.spaces]

        self.dofs_0_D_i = [list(xp.nonzero(space.projectors.ID.toarray())) for space in self.space.spaces]
        self.dofs_1_N_i = [list(xp.nonzero(space.projectors.HN.toarray())) for space in self.space.spaces]

        for i in range(self.space.dim):
            for j in range(2):
                self.dofs_0_N_i[i][j] = self.dofs_0_N_i[i][j].copy()
                self.dofs_1_D_i[i][j] = self.dofs_1_D_i[i][j].copy()

                self.dofs_0_D_i[i][j] = self.dofs_0_D_i[i][j].copy()
                self.dofs_1_N_i[i][j] = self.dofs_1_N_i[i][j].copy()

        # get 1D collocation matrices for interpolation and histopolation
        self.basis_int_N = [space.projectors.N_int.toarray() for space in self.space.spaces]
        self.basis_int_D = [space.projectors.D_int.toarray() for space in self.space.spaces]

        self.basis_his_N = [
            space.projectors.N_pts.toarray().reshape(nhis, nq, space.NbaseN)
            for space, nhis, nq in zip(self.space.spaces, self.nhis, self.nq)
        ]
        self.basis_his_D = [
            space.projectors.D_pts.toarray().reshape(nhis, nq, space.NbaseD)
            for space, nhis, nq in zip(self.space.spaces, self.nhis, self.nq)
        ]

        # number of basis functions in third dimension
        self.N3 = self.space.NbaseN[2]
        self.D3 = self.space.NbaseD[2]

    # =================================================================
    def get_blocks_EF(self, pol=True):
        """
        Returns blocks related to the degree of freedom (DOF) matrix

        basis_u = 0 : EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^0_(b,lmn) ),
        basis_u = 2 : EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^2_(b,lmn) / sqrt(g) ).

        Parameters
        ----------
            pol : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).

        Returns
        -------
            EF : list of six scipy.sparse.csr_matrices
                the DOF matrices.
        """

        if self.basis_u == 0:
            if pol or self.space.dim == 2:
                # ---------- 12 - block ([his, int] of NN) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_3_pts,
                    val,
                    row,
                    col,
                )

                EF_12 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[0] // self.N3, self.space.Ntot_0form // self.N3)
                )
                EF_12.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 13 - block ([his, int] of NN) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_2_pts,
                    val,
                    row,
                    col,
                )

                EF_13 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[0] // self.N3, self.space.Ntot_0form // self.N3)
                )
                EF_13.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 21 - block ([int, his] of NN) ----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_3_pts,
                    val,
                    row,
                    col,
                )

                EF_21 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[1] // self.N3, self.space.Ntot_0form // self.N3)
                )
                EF_21.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 23 - block ([int, his] of NN) ----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_1_pts,
                    val,
                    row,
                    col,
                )

                EF_23 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[1] // self.N3, self.space.Ntot_0form // self.N3)
                )
                EF_23.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 31 - block ([int, int] of NN) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_int[0], self.eta_int[1], 0.0)

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.basis_int_N[0],
                    self.basis_int_N[1],
                    -B2_2_pts,
                    val,
                    row,
                    col,
                )

                EF_31 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[2] // self.D3, self.space.Ntot_0form // self.N3)
                )
                EF_31.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 32 - block ([int, int] of NN) ----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_int[1], 0.0)

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.basis_int_N[0],
                    self.basis_int_N[1],
                    B2_1_pts,
                    val,
                    row,
                    col,
                )

                EF_32 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[2] // self.D3, self.space.Ntot_0form // self.N3)
                )
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

            else:
                # ------- 12 - block ([his, int, int] of NNN) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs11(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_3_pts,
                    val,
                    row,
                    col,
                )

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_0form))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 13 - block ([his, int, int] of NNN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs11(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_2_pts,
                    val,
                    row,
                    col,
                )

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_0form))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 21 - block ([int, his, int] of NNN) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs12(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_3_pts,
                    val,
                    row,
                    col,
                )

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_0form))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 23 - block ([int, his, int] of NNN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs12(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_1_pts,
                    val,
                    row,
                    col,
                )

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_0form))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 31 - block ([int, int, his] of NNN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_2_pts = B2_2_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )

                ker.rhs13(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_1_N_i[2][1],
                    self.subs[2],
                    self.subs_cum[2],
                    self.wts[2],
                    self.basis_int_N[0],
                    self.basis_int_N[1],
                    self.basis_his_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_2_pts,
                    val,
                    row,
                    col,
                )

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_0form))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 32 - block ([int, int, his] of NNN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )

                ker.rhs13(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_1_N_i[2][1],
                    self.subs[2],
                    self.subs_cum[2],
                    self.wts[2],
                    self.basis_int_N[0],
                    self.basis_int_N[1],
                    self.basis_his_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_1_pts,
                    val,
                    row,
                    col,
                )

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_0form))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

        elif self.basis_u == 2:
            if pol or self.space.dim == 2:
                # ---------- 12 - block ([his, int] of DN) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], 0.0))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_D[0],
                    self.basis_int_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_3_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_12 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[0] // self.N3, self.space.Ntot_2form[1] // self.D3)
                )
                EF_12.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 13 - block ([his, int] of DD) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], 0.0))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_D_i[1][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_D_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_D[0],
                    self.basis_int_D[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_2_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_13 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[0] // self.N3, self.space.Ntot_2form[2] // self.N3)
                )
                EF_13.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 21 - block ([int, his] of ND) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), 0.0))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_D[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_3_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_21 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[1] // self.N3, self.space.Ntot_2form[0] // self.D3)
                )
                EF_21.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 23 - block ([int, his] of DD) -----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), 0.0))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_D_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_D_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_D[0],
                    self.basis_his_D[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_1_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_23 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[1] // self.N3, self.space.Ntot_2form[2] // self.N3)
                )
                EF_23.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 31 - block ([int, int] of ND) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_int[0], self.eta_int[1], 0.0)

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_int[1], 0.0))

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size, dtype=int)

                ker.rhs0_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_D_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_D_i[1][1],
                    self.basis_int_N[0],
                    self.basis_int_D[1],
                    -B2_2_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_31 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[2] // self.D3, self.space.Ntot_2form[0] // self.D3)
                )
                EF_31.eliminate_zeros()
                # ----------------------------------------------------

                # ---------- 32 - block ([int, int] of DN) -----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_int[1], 0.0)

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_int[1], 0.0))

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(
                    self.dofs_0_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.basis_int_D[0],
                    self.basis_int_N[1],
                    B2_1_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_32 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_1form[2] // self.D3, self.space.Ntot_2form[1] // self.D3)
                )
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

            else:
                # ------- 12 - block ([his, int, int] of DND) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                )
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=int
                )

                ker.rhs11(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_0_D_i[2][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_0_D_i[2][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_D[0],
                    self.basis_int_N[1],
                    self.basis_int_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_3_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_2form[1]))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 13 - block ([his, int, int] of DDN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                )
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs11(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_D_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_D_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_D[0],
                    self.basis_int_D[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_2_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_2form[2]))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 21 - block ([int, his, int] of NDD) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                )
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_D_i[2][0].size, dtype=int
                )

                ker.rhs12(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_D_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.dofs_0_D_i[2][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_D[1],
                    self.basis_int_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_3_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_2form[0]))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 23 - block ([int, his, int] of DDN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                )
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs12(
                    self.dofs_0_D_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_0_D_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_D[0],
                    self.basis_his_D[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_1_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_2form[2]))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 31 - block ([int, int, his] of NDD) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_2(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_2_pts = B2_2_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                )
                det_dF = det_dF.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_0_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )

                ker.rhs13(
                    self.dofs_0_N_i[0][0],
                    self.dofs_0_D_i[1][0],
                    self.dofs_1_D_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_0_D_i[1][1],
                    self.dofs_1_D_i[2][1],
                    self.subs[2],
                    self.subs_cum[2],
                    self.wts[2],
                    self.basis_int_N[0],
                    self.basis_int_D[1],
                    self.basis_his_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    -B2_2_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_2form[0]))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------

                # ------- 32 - block ([int, int, his] of DND) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_1(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                )
                det_dF = det_dF.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )

                ker.rhs13(
                    self.dofs_0_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_D_i[2][0],
                    self.dofs_0_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_1_D_i[2][1],
                    self.subs[2],
                    self.subs_cum[2],
                    self.wts[2],
                    self.basis_int_D[0],
                    self.basis_int_N[1],
                    self.basis_his_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    B2_1_pts / det_dF,
                    val,
                    row,
                    col,
                )

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_2form[1]))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

        return EF_12, EF_13, EF_21, EF_23, EF_31, EF_32

    # =================================================================
    def get_blocks_FL(self, which, pol=True):
        """
        Returns blocks related to the degree of freedom (DOF) matrix

        basis_u = 0 : FL_(aa,ijk lmn) = dofs^2_(a,ijk)( fun * Lambda^0_(a,lmn) ),
        basis_u = 2 : FL_(aa,ijk lmn) = dofs^2_(a,ijk)( fun * Lambda^2_(a,lmn) / sqrt(g) ).

        Parameters
        ----------
            which : string
                * 'm' : fun = n^3_eq
                * 'p' : fun = p^3_eq
                * 'j' : fun = det_df

            pol : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).

        Returns
        -------
            F : list of three scipy.sparse.csr_matrices
                the DOF matrices.
        """

        if self.basis_u == 2:
            assert which == "m" or which == "p"
        else:
            assert which == "m" or which == "p" or which == "j"

        if self.basis_u == 0:
            if pol or self.space.dim == 2:
                # ------------- 11 - block ([int, his] of NN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                else:
                    EQ = self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), 0.0)

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_11 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[0] // self.D3, self.space.Ntot_0form // self.N3)
                )
                F_11.eliminate_zeros()
                # ------------------------------------------------------------

                # ------------- 22 - block ([his, int] of NN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                else:
                    EQ = self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], 0.0)

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_22 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[1] // self.D3, self.space.Ntot_0form // self.N3)
                )
                F_22.eliminate_zeros()
                # ------------------------------------------------------------

                # ------------- 33 - block ([his, his] of NN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
                else:
                    EQ = self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs2_2d(
                    self.dofs_1_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.subs[0],
                    self.subs[1],
                    self.subs_cum[0],
                    self.subs_cum[1],
                    self.wts[0],
                    self.wts[1],
                    self.basis_his_N[0],
                    self.basis_his_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_33 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[2] // self.N3, self.space.Ntot_0form // self.N3)
                )
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

            else:
                # -------- 11 - block ([int, his, his] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.domain.jacobian_det(
                        self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten()
                    )

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )

                ker.rhs21(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_1_N_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.dofs_1_N_i[2][1],
                    self.subs[1],
                    self.subs[2],
                    self.subs_cum[1],
                    self.subs_cum[2],
                    self.wts[1],
                    self.wts[2],
                    self.basis_int_N[0],
                    self.basis_his_N[1],
                    self.basis_his_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0], self.space.Ntot_0form))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------

                # -------- 22 - block ([his, int, his] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.domain.jacobian_det(
                        self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten()
                    )

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_N_i[2][0].size, dtype=int
                )

                ker.rhs22(
                    self.dofs_1_N_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_N_i[2][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_1_N_i[2][1],
                    self.subs[0],
                    self.subs[2],
                    self.subs_cum[0],
                    self.subs_cum[2],
                    self.wts[0],
                    self.wts[2],
                    self.basis_his_N[0],
                    self.basis_int_N[1],
                    self.basis_his_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1], self.space.Ntot_0form))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------

                # -------- 33 - block ([his, his, int] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                elif which == "p":
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                else:
                    EQ = self.equilibrium.domain.jacobian_det(
                        self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2]
                    )

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_N_i[0][0].size * self.dofs_1_N_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs23(
                    self.dofs_1_N_i[0][0],
                    self.dofs_1_N_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_1_N_i[0][1],
                    self.dofs_1_N_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[0],
                    self.subs[1],
                    self.subs_cum[0],
                    self.subs_cum[1],
                    self.wts[0],
                    self.wts[1],
                    self.basis_his_N[0],
                    self.basis_his_N[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ,
                    val,
                    row,
                    col,
                )

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2], self.space.Ntot_0form))
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

        elif self.basis_u == 2:
            if pol or self.space.dim == 2:
                # ------------- 11 - block ([int, his] of ND) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)
                else:
                    EQ = self.equilibrium.p3(self.eta_int[0], self.eta_his[1].flatten(), 0.0)

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_int[0], self.eta_his[1].flatten(), 0.0))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.subs[1],
                    self.subs_cum[1],
                    self.wts[1],
                    self.basis_int_N[0],
                    self.basis_his_D[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_11 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[0] // self.D3, self.space.Ntot_2form[0] // self.D3)
                )
                F_11.eliminate_zeros()
                # ------------------------------------------------------------

                # ------------- 22 - block ([his, int] of DN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)
                else:
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_int[1], 0.0)

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_int[1], 0.0))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.subs[0],
                    self.subs_cum[0],
                    self.wts[0],
                    self.basis_his_D[0],
                    self.basis_int_N[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_22 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[1] // self.D3, self.space.Ntot_2form[1] // self.D3)
                )
                F_22.eliminate_zeros()
                # ------------------------------------------------------------

                # ------------- 33 - block ([his, his] of DD) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
                else:
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
                )
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=float)
                row = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)
                col = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs2_2d(
                    self.dofs_1_D_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.subs[0],
                    self.subs[1],
                    self.subs_cum[0],
                    self.subs_cum[1],
                    self.wts[0],
                    self.wts[1],
                    self.basis_his_D[0],
                    self.basis_his_D[1],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_33 = spa.csr_matrix(
                    (val, (row, col)), shape=(self.space.Ntot_2form[2] // self.N3, self.space.Ntot_2form[2] // self.N3)
                )
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

            else:
                # -------- 11 - block ([int, his, his] of NDD) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.p3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(
                        self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten()
                    )
                )
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_0_N_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )

                ker.rhs21(
                    self.dofs_0_N_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_1_D_i[2][0],
                    self.dofs_0_N_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.dofs_1_D_i[2][1],
                    self.subs[1],
                    self.subs[2],
                    self.subs_cum[1],
                    self.subs_cum[2],
                    self.wts[1],
                    self.wts[2],
                    self.basis_int_N[0],
                    self.basis_his_D[1],
                    self.basis_his_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0], self.space.Ntot_2form[0]))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------

                # -------- 22 - block ([his, int, his] of DND) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(
                        self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten()
                    )
                )
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_0_N_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
                )

                ker.rhs22(
                    self.dofs_1_D_i[0][0],
                    self.dofs_0_N_i[1][0],
                    self.dofs_1_D_i[2][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_0_N_i[1][1],
                    self.dofs_1_D_i[2][1],
                    self.subs[0],
                    self.subs[2],
                    self.subs_cum[0],
                    self.subs_cum[2],
                    self.wts[0],
                    self.wts[2],
                    self.basis_his_D[0],
                    self.basis_int_N[1],
                    self.basis_his_D[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1], self.space.Ntot_2form[1]))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------

                # -------- 33 - block ([his, his, int] of DDN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == "m":
                    EQ = self.equilibrium.n3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                else:
                    EQ = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(
                    self.equilibrium.domain.jacobian_det(
                        self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2]
                    )
                )
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=float
                )
                row = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )
                col = xp.empty(
                    self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_0_N_i[2][0].size, dtype=int
                )

                ker.rhs23(
                    self.dofs_1_D_i[0][0],
                    self.dofs_1_D_i[1][0],
                    self.dofs_0_N_i[2][0],
                    self.dofs_1_D_i[0][1],
                    self.dofs_1_D_i[1][1],
                    self.dofs_0_N_i[2][1],
                    self.subs[0],
                    self.subs[1],
                    self.subs_cum[0],
                    self.subs_cum[1],
                    self.wts[0],
                    self.wts[1],
                    self.basis_his_D[0],
                    self.basis_his_D[1],
                    self.basis_int_N[2],
                    xp.array(self.space.NbaseN),
                    xp.array(self.space.NbaseD),
                    EQ / det_dF,
                    val,
                    row,
                    col,
                )

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2], self.space.Ntot_2form[2]))
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

        return F_11, F_22, F_33

    # =================================================================
    def get_blocks_PR(self, pol=True):
        """
        Returns the degree of freedom (DOF) matrix

        PR_(ijk lmn) = dofs^3_(ijk)( p^3_eq * Lambda^3_(lmn) / sqrt(g) ).

        Parameters
        ----------
            pol : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).

        Returns
        -------
            PR : scipy.sparse.csr_matrix
                the DOF matrix.
        """

        if pol or self.space.dim == 2:
            # ------------ ([his, his] of DD) --------------------
            # evaluate equilibrium pressure at quadrature points
            P3_pts = self.equilibrium.p3(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
            P3_pts = P3_pts.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(
                self.equilibrium.domain.jacobian_det(self.eta_his[0].flatten(), self.eta_his[1].flatten(), 0.0)
            )
            det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

            # assemble sparse matrix
            val = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=float)
            row = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)
            col = xp.empty(self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size, dtype=int)

            ker.rhs2_2d(
                self.dofs_1_D_i[0][0],
                self.dofs_1_D_i[1][0],
                self.dofs_1_D_i[0][1],
                self.dofs_1_D_i[1][1],
                self.subs[0],
                self.subs[1],
                self.subs_cum[0],
                self.subs_cum[1],
                self.wts[0],
                self.wts[1],
                self.basis_his_D[0],
                self.basis_his_D[1],
                xp.array(self.space.NbaseN),
                xp.array(self.space.NbaseD),
                P3_pts / det_dF,
                val,
                row,
                col,
            )

            PR = spa.csr_matrix(
                (val, (row, col)), shape=(self.space.Ntot_3form // self.D3, self.space.Ntot_3form // self.D3)
            )
            PR.eliminate_zeros()
            # -----------------------------------------------------

        else:
            # --------------- ([his, his, his] of DDD) ------------
            # evaluate equilibrium pressure at quadrature points
            P3_pts = self.equilibrium.p3(
                self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_his[2].flatten()
            )

            P3_pts = P3_pts.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(
                self.equilibrium.domain.jacobian_det(
                    self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_his[2].flatten()
                )
            )
            det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

            # assemble sparse matrix
            val = xp.empty(
                self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=float
            )
            row = xp.empty(
                self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
            )
            col = xp.empty(
                self.dofs_1_D_i[0][0].size * self.dofs_1_D_i[1][0].size * self.dofs_1_D_i[2][0].size, dtype=int
            )

            ker.rhs3(
                self.dofs_1_D_i[0][0],
                self.dofs_1_D_i[1][0],
                self.dofs_1_D_i[2][0],
                self.dofs_1_D_i[0][1],
                self.dofs_1_D_i[1][1],
                self.dofs_1_D_i[2][1],
                self.subs[0],
                self.subs[1],
                self.subs[2],
                self.subs_cum[0],
                self.subs_cum[1],
                self.subs_cum[2],
                self.wts[0],
                self.wts[1],
                self.wts[2],
                self.basis_his_D[0],
                self.basis_his_D[1],
                self.basis_his_D[2],
                xp.array(self.space.NbaseN),
                xp.array(self.space.NbaseD),
                P3_pts / det_dF,
                val,
                row,
                col,
            )

            PR = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_3form, self.space.Ntot_3form))
            PR.eliminate_zeros()
            # ----------------------------------------------------

        return PR

    # ====================================================================
    def get_blocks_Mn(self, pol=True):
        """
        Returns the weighted mass matrix

        basis_u = 0 : Mn_(ab, ijk lmn) = integral( Lambda^0_(a,ijk) * G_ab * Lambda^0_(b,lmn) * n^0_eq * sqrt(g) ),
        basis_u = 2 : Mn_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) * n^0_eq / sqrt(g) ).

        Parameters
        ----------
            pol : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).

        Returns
        -------
            Mn : scipy.sparse.csr_matrix
                the weighted mass matrix.
        """

        weight11 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[0, 0]
        )
        weight12 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[0, 1]
        )
        weight13 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[0, 2]
        )

        weight21 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[1, 0]
        )
        weight22 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[1, 1]
        )
        weight23 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[1, 2]
        )

        weight31 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[2, 0]
        )
        weight32 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[2, 1]
        )
        weight33 = (
            lambda s, chi, phi: self.equilibrium.n0(s, chi, phi) * self.equilibrium.domain.metric(s, chi, phi)[2, 2]
        )

        self.weights_Mn = [
            [weight11, weight12, weight13],
            [weight21, weight22, weight23],
            [weight31, weight32, weight33],
        ]

        # ----------- 0-form ----------------------
        if self.basis_u == 0:
            if pol or self.space.dim == 2:
                Mn = mass_2d.get_Mv(self.space, self.equilibrium.domain, True, self.weights_Mn)
            else:
                Mn = mass_3d.get_Mv(self.space, self.equilibrium.domain, True, self.weights_Mn)
        # -----------------------------------------

        # ----------- 2-form ----------------------
        elif self.basis_u == 2:
            if pol or self.space.dim == 2:
                Mn = mass_2d.get_M2(self.space, self.equilibrium.domain, True, self.weights_Mn)
            else:
                Mn = mass_3d.get_M2(self.space, self.equilibrium.domain, True, self.weights_Mn)
        # -----------------------------------------

        return Mn

    # =================================================================
    def get_blocks_MJ(self, pol=True):
        """
        Returns the weighted mass matrix

        basis_u = 0 : not implemented yet --> MJ = 0,
        basis_u = 2 : MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_c_eq / sqrt(g) ).

        Parameters
        ----------
            pol : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).

        Returns
        -------
            MJ : scipy.sparse.csr_matrix
                the weighted mass matrix.
        """

        weight11 = lambda s, chi, phi: 0 * self.equilibrium.j2_1(s, chi, phi)
        weight12 = lambda s, chi, phi: -self.equilibrium.j2_3(s, chi, phi)
        weight13 = lambda s, chi, phi: self.equilibrium.j2_2(s, chi, phi)

        weight21 = lambda s, chi, phi: self.equilibrium.j2_3(s, chi, phi)
        weight22 = lambda s, chi, phi: 0 * self.equilibrium.j2_2(s, chi, phi)
        weight23 = lambda s, chi, phi: -self.equilibrium.j2_1(s, chi, phi)

        weight31 = lambda s, chi, phi: -self.equilibrium.j2_2(s, chi, phi)
        weight32 = lambda s, chi, phi: self.equilibrium.j2_1(s, chi, phi)
        weight33 = lambda s, chi, phi: 0 * self.equilibrium.j2_3(s, chi, phi)

        self.weights_MJ = [
            [weight11, weight12, weight13],
            [weight21, weight22, weight23],
            [weight31, weight32, weight33],
        ]

        # ----------- 0-form ----------------------
        if self.basis_u == 0:
            if pol or self.space.dim == 2:
                MJ = mass_2d.get_M2(self.space, self.equilibrium.domain, True, self.weights_MJ)
            else:
                MJ = mass_3d.get_M2(self.space, self.equilibrium.domain, True, self.weights_MJ)
        # -----------------------------------------

        # ----------- 2-form ----------------------
        elif self.basis_u == 2:
            if pol or self.space.dim == 2:
                MJ = mass_2d.get_M2(self.space, self.equilibrium.domain, True, self.weights_MJ)
            else:
                MJ = mass_3d.get_M2(self.space, self.equilibrium.domain, True, self.weights_MJ)
        # -----------------------------------------

        return MJ
