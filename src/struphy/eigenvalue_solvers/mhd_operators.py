# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)


import cunumpy as xp
import scipy.sparse as spa

import struphy.eigenvalue_solvers.legacy.mass_matrices_3d_pre as mass_3d_pre
from struphy.eigenvalue_solvers.mhd_operators_core import MHDOperatorsCore


class MHDOperators:
    """
    Class for degree of freedom matrices related to ideal MHD equations.

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
        # create MHD operators core object
        self.core = MHDOperatorsCore(space, equilibrium, basis_u)

        # set adiabatic index
        self.gamma = 5 / 3

        # get 1D int_N and int_D matrices in third direction
        if space.dim == 2:
            self.ID_tor = spa.identity(space.NbaseN[2], format="csr")

            self.int_N3 = spa.identity(space.NbaseN[2], format="csr")
            self.int_D3 = spa.identity(space.NbaseN[2], format="csr")

            self.his_N3 = spa.identity(space.NbaseN[2], format="csr")
            self.his_D3 = spa.identity(space.NbaseN[2], format="csr")

        else:
            B0 = space.spaces[2].B0
            B1 = space.spaces[2].B1

            self.int_N3 = B0.dot(space.spaces[2].projectors.I.dot(B0.T))
            self.int_D3 = B0.dot(space.spaces[2].projectors.ID.dot(B1.T))

            self.his_N3 = B1.dot(space.spaces[2].projectors.HN.dot(B0.T))
            self.his_D3 = B1.dot(space.spaces[2].projectors.H.dot(B1.T))

    # =================================================================
    def __assemble_dofs_EF(self, as_tensor=False):
        """
        Sets blocks related to the degree of freedom (DOF) matrix

        basis_u = 0 : dofs_EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^0_(b,lmn) ),
        basis_u = 2 : dofs_EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^2_(b,lmn) / sqrt(g) ),

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        self.EF_as_tensor = as_tensor

        EF_12, EF_13, EF_21, EF_23, EF_31, EF_32 = self.core.get_blocks_EF(self.EF_as_tensor)

        # ------------ full operator : 0-form --------
        if self.core.basis_u == 0:
            if self.EF_as_tensor:
                EF_11 = spa.bmat([[None, EF_12], [EF_21, None]])

                EF_12 = spa.bmat([[EF_13], [EF_23]])
                EF_21 = spa.bmat([[EF_31, EF_32]])

                self.dofs_EF = []

                # self.dofs_EF_pol_11 = self.core.space.projectors.P1_pol_0.dot(EF_11.dot(self.core.space.Ev_pol_0.T)).tocsr()
                # self.dofs_EF_pol_12 = self.core.space.projectors.P1_pol_0.dot(EF_12.dot(self.core.space.E0_pol.T  )).tocsr()
                # self.dofs_EF_pol_21 = self.core.space.projectors.P0_pol_0.dot(EF_21.dot(self.core.space.Ev_pol_0.T)).tocsr()

                self.dofs_EF += [self.core.space.projectors.P1_pol_0.dot(EF_11.dot(self.core.space.Ev_pol_0.T)).tocsr()]
                self.dofs_EF += [self.core.space.projectors.P1_pol_0.dot(EF_12.dot(self.core.space.E0_pol.T)).tocsr()]
                self.dofs_EF += [self.core.space.projectors.P0_pol_0.dot(EF_21.dot(self.core.space.Ev_pol_0.T)).tocsr()]

            else:
                EF = spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]])

                if self.core.space.dim == 2:
                    EF = spa.kron(EF, self.ID_tor, format="csr")

                self.dofs_EF = self.core.space.projectors.P1_0.dot(EF.dot(self.core.space.Ev_0.T)).tocsr()
        # --------------------------------------------

        # ------------ full operator : 2-form --------
        elif self.core.basis_u == 2:
            if self.EF_as_tensor:
                EF_11 = spa.bmat([[None, EF_12], [EF_21, None]])

                EF_12 = spa.bmat([[EF_13], [EF_23]])
                EF_21 = spa.bmat([[EF_31, EF_32]])

                self.dofs_EF = []

                # self.dofs_EF_pol_11 = self.core.space.projectors.P1_pol_0.dot(EF_11.dot(self.core.space.E2_pol_0.T)).tocsr()
                # self.dofs_EF_pol_12 = self.core.space.projectors.P1_pol_0.dot(EF_12.dot(self.core.space.E3_pol_0.T)).tocsr()
                # self.dofs_EF_pol_21 = self.core.space.projectors.P0_pol_0.dot(EF_21.dot(self.core.space.E2_pol_0.T)).tocsr()

                self.dofs_EF += [self.core.space.projectors.P1_pol_0.dot(EF_11.dot(self.core.space.E2_pol_0.T)).tocsr()]
                self.dofs_EF += [self.core.space.projectors.P1_pol_0.dot(EF_12.dot(self.core.space.E3_pol_0.T)).tocsr()]
                self.dofs_EF += [self.core.space.projectors.P0_pol_0.dot(EF_21.dot(self.core.space.E2_pol_0.T)).tocsr()]

            else:
                EF = spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]])

                if self.core.space.dim == 2:
                    EF = spa.kron(EF, self.ID_tor, format="csr")

                self.dofs_EF = self.core.space.projectors.P1_0.dot(EF.dot(self.core.space.E2_0.T)).tocsr()
        # --------------------------------------------

    # =================================================================
    def __assemble_dofs_FL(self, which, as_tensor=False):
        """
        Sets blocks related to the degree of freedom (DOF) matrix

        basis_u = 0 : dofs_FL_(aa,ijk lmn) = dofs^2_(a,ijk)( fun * Lambda^0_(a,lmn) ),
        basis_u = 2 : dofs_FL_(aa,ijk lmn) = dofs^2_(a,ijk)( fun * Lambda^2_(a,lmn) / sqrt(g) ),

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            which : string
                * 'm' : fun = n^3_eq
                * 'p' : fun = p^3_eq
                * 'j' : fun = det_df

            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        assert which == "m" or which == "p" or which == "j"

        if which == "m":
            self.MF_as_tensor = as_tensor
        elif which == "p":
            self.PF_as_tensor = as_tensor
        elif which == "j":
            self.JF_as_tensor = as_tensor

        FL_11, FL_22, FL_33 = self.core.get_blocks_FL(which, as_tensor)

        # ------------ full operator : 0-form --------
        if self.core.basis_u == 0:
            if as_tensor:
                FL_11 = spa.bmat([[FL_11, None], [None, FL_22]])

                dofs_FL_pol_11 = self.core.space.projectors.P2_pol_0.dot(FL_11.dot(self.core.space.Ev_pol_0.T)).tocsr()
                dofs_FL_pol_22 = self.core.space.projectors.P3_pol_0.dot(FL_33.dot(self.core.space.E0_pol.T)).tocsr()

                if which == "m":
                    self.dofs_MF = []

                    # self.dofs_MF_pol_11 = dofs_FL_pol_11
                    # self.dofs_MF_pol_22 = dofs_FL_pol_22

                    self.dofs_MF[0] += [dofs_FL_pol_11]
                    self.dofs_MF[1] += [dofs_FL_pol_22]

                if which == "p":
                    self.dofs_PF = []

                    # self.dofs_PF_pol_11 = dofs_FL_pol_11
                    # self.dofs_PF_pol_22 = dofs_FL_pol_22

                    self.dofs_PF[0] += [dofs_FL_pol_11]
                    self.dofs_PF[1] += [dofs_FL_pol_22]

                if which == "j":
                    self.dofs_JF = []

                    # self.dofs_JF_pol_11 = dofs_FL_pol_11
                    # self.dofs_JF_pol_22 = dofs_FL_pol_22

                    self.dofs_JF[0] = [dofs_FL_pol_11]
                    self.dofs_JF[1] = [dofs_FL_pol_22]

            else:
                FL = spa.bmat([[FL_11, None, None], [None, FL_22, None], [None, None, FL_33]])

                if self.core.space.dim == 2:
                    FL = spa.kron(FL, self.ID_tor, format="csr")

                if which == "m":
                    self.dofs_MF = self.core.space.projectors.P2_0.dot(FL.dot(self.core.space.Ev_0.T)).tocsr()

                if which == "p":
                    self.dofs_PF = self.core.space.projectors.P2_0.dot(FL.dot(self.core.space.Ev_0.T)).tocsr()

                if which == "j":
                    self.dofs_JF = self.core.space.projectors.P2_0.dot(FL.dot(self.core.space.Ev_0.T)).tocsr()
        # --------------------------------------------

        # ------------ full operator : 2-form --------
        elif self.core.basis_u == 2:
            if as_tensor:
                FL_11 = spa.bmat([[FL_11, None], [None, FL_22]])

                dofs_FL_pol_11 = self.core.space.projectors.P2_pol_0.dot(FL_11.dot(self.core.space.E2_pol_0.T)).tocsr()
                dofs_FL_pol_22 = self.core.space.projectors.P3_pol_0.dot(FL_33.dot(self.core.space.E3_pol_0.T)).tocsr()

                if which == "m":
                    self.dofs_MF = []

                    # self.dofs_MF_pol_11 = dofs_FL_pol_11
                    # self.dofs_MF_pol_22 = dofs_FL_pol_22

                    self.dofs_MF[0] += [dofs_FL_pol_11]
                    self.dofs_MF[1] += [dofs_FL_pol_22]

                if which == "p":
                    self.dofs_PF = []

                    # self.dofs_PF_pol_11 = dofs_FL_pol_11
                    # self.dofs_PF_pol_22 = dofs_FL_pol_22

                    self.dofs_PF[0] += [dofs_FL_pol_11]
                    self.dofs_PF[1] += [dofs_FL_pol_22]

            else:
                FL = spa.bmat([[FL_11, None, None], [None, FL_22, None], [None, None, FL_33]])

                if self.core.space.dim == 2:
                    FL = spa.kron(FL, self.ID_tor, format="csr")

                if which == "m":
                    self.dofs_MF = self.core.space.projectors.P2_0.dot(FL.dot(self.core.space.E2_0.T)).tocsr()

                elif which == "p":
                    self.dofs_PF = self.core.space.projectors.P2_0.dot(FL.dot(self.core.space.E2_0.T)).tocsr()
        # --------------------------------------------

    # =================================================================
    def __assemble_dofs_PR(self, as_tensor=False):
        """
        Sets degree of freedom (DOF) matrix

        PR_(ijk lmn) = dofs^3_(ijk)( p^3_eq * Lambda^3_(lmn) / sqrt(g) ),

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        self.PR_as_tensor = as_tensor

        PR = self.core.get_blocks_PR(self.PR_as_tensor)

        if self.PR_as_tensor:
            self.dofs_PR = self.core.space.projectors.P3_pol_0.dot(PR.dot(self.core.space.E3_pol_0.T)).tocsr()

        else:
            if self.core.space.dim == 2:
                PR = spa.kron(PR, self.ID_tor, format="csr")

            self.dofs_PR = self.core.space.projectors.P3_0.dot(PR.dot(self.core.space.E3_0.T)).tocsr()

    # =================================================================
    def assemble_dofs(self, which, as_tensor=False):
        """
        Sets degree of freedom (DOF) matrix

        which = EF : dofs_EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^0_(b,lmn) )           if basis_u = 0,
        which = EF : dofs_EF_(ab,ijk lmn) = dofs^1_(a,ijk)( B^2_eq x Lambda^2_(b,lmn) / sqrt(g))  if basis_u = 2,

        which = MF : dofs_MF_(aa,ijk lmn) = dofs^2_(a,ijk)( n^3_eq * Lambda^0_(a,lmn) )           if basis_u = 0,
        which = MF : dofs_MF_(aa,ijk lmn) = dofs^2_(a,ijk)( n^3_eq * Lambda^2_(a,lmn) / sqrt(g) ) if basis_u = 2,

        which = PF : dofs_PF_(aa,ijk lmn) = dofs^2_(a,ijk)( p^3_eq * Lambda^0_(a,lmn) )           if basis_u = 0,
        which = PF : dofs_PF_(aa,ijk lmn) = dofs^2_(a,ijk)( p^3_eq * Lambda^2_(a,lmn) / sqrt(g) ) if basis_u = 2,

        which = JF : dofs_JF_(aa,ijk lmn) = dofs^2_(a,ijk)( det_df * Lambda^0_(a,lmn) ),

        which = PR : PR_(ijk lmn) = dofs^3_(ijk)( p^3_eq * Lambda^3_(lmn) / sqrt(g) ),

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        if which == "EF":
            self.__assemble_dofs_EF(as_tensor)
        elif which == "MF":
            self.__assemble_dofs_FL("m", as_tensor)
        elif which == "PF":
            self.__assemble_dofs_FL("p", as_tensor)
        elif which == "JF":
            self.__assemble_dofs_FL("j", as_tensor)
        elif which == "PR":
            self.__assemble_dofs_PR(as_tensor)

    # =================================================================
    def assemble_Mn(self, as_tensor=False):
        """
        Sets the weighted mass matrix

        basis_u = 0 : Mn_(ab, ijk lmn) = integral( Lambda^0_(a,ijk) * G_ab * Lambda^0_(b,lmn) * n^0_eq * sqrt(g) ),
        basis_u = 2 : Mn_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) * n^0_eq / sqrt(g) ).

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        self.Mn_as_tensor = as_tensor

        Mn = self.core.get_blocks_Mn(self.Mn_as_tensor)

        if self.Mn_as_tensor:
            self.Mn_mat = Mn

        else:
            if self.core.space.dim == 2:
                if self.core.basis_u == 0:
                    M11 = spa.kron(Mn[0], self.core.space.M0_tor)
                    M22 = spa.kron(Mn[1], self.core.space.M0_tor)

                if self.core.basis_u == 2:
                    M11 = spa.kron(Mn[0], self.core.space.M1_tor)
                    M22 = spa.kron(Mn[1], self.core.space.M0_tor)

                self.Mn_mat = spa.bmat([[M11, None], [None, M22]], format="csr")

            else:
                self.Mn_mat = Mn

    # =================================================================
    def assemble_MJ(self, as_tensor=False):
        """
        Sets the weighted mass matrix

        basis_u = 0 : not implemented yet --> MJ = 0,
        basis_u = 2 : MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_c_eq / sqrt(g) ).

        while taking into account polar extraction operators and boundary operators.

        Parameters
        ----------
            as_tensor : boolean
                wheather to assemble the matrices in the form (poloidal x toroidal) (True).
        """

        self.MJ_as_tensor = as_tensor

        MJ = self.core.get_blocks_MJ(self.MJ_as_tensor)

        if self.core.basis_u == 2:
            if self.MJ_as_tensor:
                self.MJ_mat = MJ

            else:
                if self.core.space.dim == 2:
                    M11 = spa.kron(MJ[0], self.core.space.M1_tor)
                    M22 = spa.kron(MJ[1], self.core.space.M0_tor)

                    self.MJ_mat = spa.bmat([[M11, None], [None, M22]], format="csr")

                else:
                    self.MJ_mat = MJ

        else:
            print("MJ matrix for vector MHD is not yet implemented (and is therefore set to zero)!")

    # ======================================
    def __EF(self, u):
        """
        TODO
        """

        if self.EF_as_tensor:
            if self.core.basis_u == 0:
                u1, u3 = self.core.space.reshape_pol_v(u)

                out1 = self.int_N3.dot(self.dofs_EF[0].dot(u1).T).T + self.int_N3.dot(self.dofs_EF[1].dot(u3).T).T
                out3 = self.his_N3.dot(self.dofs_EF[2].dot(u1).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                u1, u3 = self.core.space.reshape_pol_2(u)

                out1 = self.int_D3.dot(self.dofs_EF[0].dot(u1).T).T + self.int_N3.dot(self.dofs_EF[1].dot(u3).T).T
                out3 = self.his_D3.dot(self.dofs_EF[2].dot(u1).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_EF.dot(u)

        return self.core.space.projectors.solve_V1(out, False)

    # ======================================
    def __EF_transposed(self, e):
        """
        TODO
        """

        e = self.core.space.projectors.apply_IinvT_V1(e)

        if self.EF_as_tensor:
            e1, e3 = self.core.space.reshape_pol_1(e)

            if self.core.basis_u == 0:
                out1 = (
                    self.int_N3.T.dot(self.dofs_EF[0].T.dot(e1).T).T + self.his_N3.T.dot(self.dofs_EF[2].T.dot(e3).T).T
                )
                out3 = self.int_N3.T.dot(self.dofs_EF[1].T.dot(e1).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                out1 = (
                    self.int_D3.T.dot(self.dofs_EF[0].T.dot(e1).T).T + self.his_D3.T.dot(self.dofs_EF[2].T.dot(e3).T).T
                )
                out3 = self.int_N3.T.dot(self.dofs_EF[1].T.dot(e1).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_EF.T.dot(e)

        return out

    # ======================================
    def __MF(self, u):
        """
        TODO
        """

        if self.MF_as_tensor:
            if self.core.basis_u == 0:
                u1, u3 = self.core.space.reshape_pol_v(u)

                out1 = self.his_N3.dot(self.dofs_MF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_MF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                u1, u3 = self.core.space.reshape_pol_2(u)

                out1 = self.his_D3.dot(self.dofs_MF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_MF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_MF.dot(u)

        return self.core.space.projectors.solve_V2(out, False)

    # ======================================
    def __MF_transposed(self, f):
        """
        TODO
        """

        f = self.core.space.projectors.apply_IinvT_V2(f)

        if self.MF_as_tensor:
            f1, f3 = self.core.space.reshape_pol_2(f)

            if self.core.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_MF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_MF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_MF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_MF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_MF.T.dot(f)

        return out

    # ======================================
    def __PF(self, u):
        """
        TODO
        """

        if self.PF_as_tensor:
            if self.core.basis_u == 0:
                u1, u3 = self.core.space.reshape_pol_v(u)

                out1 = self.his_N3.dot(self.dofs_PF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_PF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                u1, u3 = self.core.space.reshape_pol_2(u)

                out1 = self.his_D3.dot(self.dofs_PF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_PF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_PF.dot(u)

        return self.core.space.projectors.solve_V2(out, False)

    # ======================================
    def __PF_transposed(self, f):
        """
        TODO
        """

        f = self.core.space.projectors.apply_IinvT_V2(f)

        if self.PF_as_tensor:
            f1, f3 = self.core.space.reshape_pol_2(f)

            if self.core.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_PF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_PF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_PF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_PF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_PF.T.dot(f)

        return out

    # ======================================
    def __JF(self, u):
        """
        TODO
        """

        if self.JF_as_tensor:
            if self.core.basis_u == 0:
                u1, u3 = self.core.space.reshape_pol_v(u)

                out1 = self.his_N3.dot(self.dofs_JF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_JF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                u1, u3 = self.core.space.reshape_pol_2(u)

                out1 = self.his_D3.dot(self.dofs_JF[0].dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_JF[1].dot(u3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_JF.dot(u)

        return self.core.space.projectors.solve_V2(out, False)

    # ======================================
    def __JF_transposed(self, f):
        """
        TODO
        """

        f = self.core.space.projectors.apply_IinvT_V2(f)

        if self.JF_as_tensor:
            f1, f3 = self.core.space.reshape_pol_2(f)

            if self.core.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_JF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_JF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

            elif self.core.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_JF[0].T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_JF[1].T.dot(f3).T).T

                out = xp.concatenate((out1.flatten(), out3.flatten()))

        else:
            out = self.dofs_JF.T.dot(f)

        return out

    # ======================================
    def __PR(self, d):
        """
        TODO
        """

        if self.PR_as_tensor:
            d = self.core.space.reshape_pol_3(d)
            out = self.his_D3.dot(self.dofs_PR.dot(d).T).T.flatten()

        else:
            out = self.dofs_PR.dot(d)

        return self.core.space.projectors.solve_V3(out, False)

    # ======================================
    def __PR_transposed(self, d):
        """
        TODO
        """

        d = self.core.space.projectors.apply_IinvT_V3(d)

        if self.PR_as_tensor:
            d = self.core.space.reshape_pol_3(d)
            out = self.his_D3.T.dot(self.dofs_PR.T.dot(d).T).T.flatten()

        else:
            out = self.dofs_PR.T.dot(d)

        return out

    # ======================================
    def __Mn(self, u):
        """
        TODO
        """

        if self.Mn_as_tensor:
            if self.core.basis_u == 0:
                out = self.core.space.apply_Mv_ten(
                    u, [[self.Mn_mat[0], self.core.space.M0_tor], [self.Mn_mat[1], self.core.space.M0_tor]]
                )
            elif self.core.basis_u == 2:
                out = self.core.space.apply_M2_ten(
                    u, [[self.Mn_mat[0], self.core.space.M1_tor], [self.Mn_mat[1], self.core.space.M0_tor]]
                )

        else:
            out = self.Mn_mat.dot(u)

        return out

    # ======================================
    def __MJ(self, b):
        """
        TODO
        """

        if self.MJ_as_tensor:
            if self.core.basis_u == 0:
                out = xp.zeros(self.core.space.Ev_0.shape[0], dtype=float)
            elif self.core.basis_u == 2:
                out = self.core.space.apply_M2_ten(
                    b, [[self.MJ_mat[0], self.core.space.M1_tor], [self.MJ_mat[1], self.core.space.M0_tor]]
                )

        else:
            if self.core.basis_u == 0:
                out = xp.zeros(self.core.space.Ev_0.shape[0], dtype=float)
            elif self.core.basis_u == 2:
                out = self.MJ_mat.dot(b)

        return out

    # ======================================
    def __L(self, u):
        """
        TODO
        """

        if self.core.basis_u == 0:
            out = -self.core.space.D0.dot(self.__PF(u)) - (self.gamma - 1) * self.__PR(
                self.core.space.D0.dot(self.__JF(u))
            )
        elif self.core.basis_u == 2:
            out = -self.core.space.D0.dot(self.__PF(u)) - (self.gamma - 1) * self.__PR(self.core.space.D0.dot(u))

        return out

    # ======================================
    def __S2(self, u):
        """
        TODO
        """

        bu = self.core.space.C0.dot(self.__EF(u))

        out = self.__Mn(u)
        out += self.dt_2**2 / 4 * self.__EF_transposed(self.core.space.C0.T.dot(self.core.space.M2_0(bu)))

        return out

    # ======================================
    def __S6(self, u):
        """
        TODO
        """

        out = self.__Mn(u)

        if self.core.basis_u == 0:
            out -= self.dt_6**2 / 4 * self.__JF_transposed(self.core.space.D0.T.dot(self.core.space.M3_0(self.__L(u))))
        elif self.core.basis_u == 2:
            out -= self.dt_6**2 / 4 * self.core.space.D0.T.dot(self.core.space.M3_0(self.__L(u)))

        return out

    # ======================================
    def set_operators(self, dt_2=1.0, dt_6=1.0):
        """
        TODO
        """

        self.dt_2 = dt_2
        self.dt_6 = dt_6

        if self.core.basis_u == 0:
            if hasattr(self, "dofs_MF"):
                self.MF = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.Ev_0.shape[0]),
                    matvec=self.__MF,
                    rmatvec=self.__MF_transposed,
                )

            if hasattr(self, "dofs_PF"):
                self.PF = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.Ev_0.shape[0]),
                    matvec=self.__PF,
                    rmatvec=self.__PF_transposed,
                )

            if hasattr(self, "dofs_JF"):
                self.JF = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.Ev_0.shape[0]),
                    matvec=self.__JF,
                    rmatvec=self.__JF_transposed,
                )

            if hasattr(self, "dofs_EF"):
                self.EF = spa.linalg.LinearOperator(
                    (self.core.space.E1_0.shape[0], self.core.space.Ev_0.shape[0]),
                    matvec=self.__EF,
                    rmatvec=self.__EF_transposed,
                )

            if hasattr(self, "dofs_PR"):
                self.PR = spa.linalg.LinearOperator(
                    (self.core.space.E3_0.shape[0], self.core.space.E3_0.shape[0]),
                    matvec=self.__PR,
                    rmatvec=self.__PR_transposed,
                )

            if hasattr(self, "dofs_Mn"):
                self.Mn = spa.linalg.LinearOperator(
                    (self.core.space.Ev_0.shape[0], self.core.space.Ev_0.shape[0]), matvec=self.__Mn
                )

            if hasattr(self, "dofs_MJ"):
                self.MJ = spa.linalg.LinearOperator(
                    (self.core.space.Ev_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__MJ
                )

            if hasattr(self, "dofs_PF") and hasattr(self, "dofs_PR") and hasattr(self, "dofs_JF"):
                self.L = spa.linalg.LinearOperator(
                    (self.core.space.E3_0.shape[0], self.core.space.Ev_0.shape[0]), matvec=self.__L
                )

            if hasattr(self, "Mn_mat") and hasattr(self, "dofs_EF"):
                self.S2 = spa.linalg.LinearOperator(
                    (self.core.space.Ev_0.shape[0], self.core.space.Ev_0.shape[0]), matvec=self.__S2
                )

            if hasattr(self, "Mn_mat") and hasattr(self, "L"):
                self.S6 = spa.linalg.LinearOperator(
                    (self.core.space.Ev_0.shape[0], self.core.space.Ev_0.shape[0]), matvec=self.__S6
                )

        elif self.core.basis_u == 2:
            if hasattr(self, "dofs_MF"):
                self.MF = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]),
                    matvec=self.__MF,
                    rmatvec=self.__MF_transposed,
                )

            if hasattr(self, "dofs_PF"):
                self.PF = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]),
                    matvec=self.__PF,
                    rmatvec=self.__PF_transposed,
                )

            if hasattr(self, "dofs_EF"):
                self.EF = spa.linalg.LinearOperator(
                    (self.core.space.E1_0.shape[0], self.core.space.E2_0.shape[0]),
                    matvec=self.__EF,
                    rmatvec=self.__EF_transposed,
                )

            if hasattr(self, "dofs_PR"):
                self.PR = spa.linalg.LinearOperator(
                    (self.core.space.E3_0.shape[0], self.core.space.E3_0.shape[0]),
                    matvec=self.__PR,
                    rmatvec=self.__PR_transposed,
                )

            if hasattr(self, "Mn_mat"):
                self.Mn = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__Mn
                )

            if hasattr(self, "MJ_mat"):
                self.MJ = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__MJ
                )

            if hasattr(self, "dofs_PF") and hasattr(self, "dofs_PR"):
                self.L = spa.linalg.LinearOperator(
                    (self.core.space.E3_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__L
                )

            if hasattr(self, "Mn_mat") and hasattr(self, "dofs_EF"):
                self.S2 = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__S2
                )

            if hasattr(self, "Mn_mat") and hasattr(self, "L"):
                self.S6 = spa.linalg.LinearOperator(
                    (self.core.space.E2_0.shape[0], self.core.space.E2_0.shape[0]), matvec=self.__S6
                )

    # ======================================
    def rhs2(self, u, b):
        """
        TODO
        """

        bu = self.core.space.C0.dot(self.EF(u))

        out = self.Mn(u)
        out -= self.dt_2**2 / 4 * self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(bu)))
        out += self.dt_2 * self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b)))

        return out

    # ======================================
    def rhs6(self, u, p, b):
        """
        TODO
        """

        out = self.Mn(u)

        if self.core.basis_u == 0:
            out += self.dt_6**2 / 4 * self.JF.T(self.core.space.D0.T.dot(self.core.space.M3_0(self.L(u))))
            out += self.dt_6 * self.JF.T(self.core.space.D0.T.dot(self.core.space.M3_0(p)))
            out += self.dt_6 * self.MJ(b)

        elif self.core.basis_u == 2:
            out += self.dt_6**2 / 4 * self.core.space.D0.T.dot(self.core.space.M3_0(self.L(u)))
            out += self.dt_6 * self.core.space.D0.T.dot(self.core.space.M3_0(p))
            out += self.dt_6 * self.MJ(b)
        # --------------------------------------

        return out

    # ======================================
    def guess_S2(self, u, b, kind):
        """
        TODO
        """

        if kind == "Euler":
            k1_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b))))

            u_guess = u + self.dt_2 * k1_u

        elif kind == "Heun":
            k1_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b))))
            k1_b = -self.core.space.C0.dot(self.EF(u))

            k2_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b + self.dt_2 * k1_b))))

            u_guess = u + self.dt_2 / 2 * (k1_u + k2_u)

        elif kind == "RK4":
            k1_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b))))
            k1_b = -self.core.space.C0.dot(self.EF(u))

            k2_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b + self.dt_2 / 2 * k1_b))))
            k2_b = -self.core.space.C0.dot(self.EF(u + self.dt_2 / 2 * k1_u))

            k3_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b + self.dt_2 / 2 * k2_b))))
            k3_b = -self.core.space.C0.dot(self.EF(u + self.dt_2 / 2 * k2_u))

            k4_u = self.Mn_inv(self.EF.T(self.core.space.C0.T.dot(self.core.space.M2_0(b + self.dt_2 * k3_b))))
            k4_b = -self.core.space.C0.dot(self.EF(u + self.dt_2 * k3_u))

            u_guess = u + self.dt_2 / 6 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

        else:
            u_guess = xp.copy(u)

        return u_guess

    # ======================================
    def guess_S6(self, u, p, b, kind):
        """
        TODO
        """

        u_guess = u.copy()

        return u_guess

    # ======================================
    def set_inverse_Mn(self):
        """
        TODO
        """

        if self.core.basis_u == 0:
            self.Mn_inv = mass_3d_pre.get_Mv_PRE_3(self.core.space, [self.Mn_mat[0], self.Mn_mat[1]])

        if self.core.basis_u == 2:
            self.Mn_inv = mass_3d_pre.get_M2_PRE_3(self.core.space, [self.Mn_mat[0], self.Mn_mat[1]])

    # ======================================
    def set_preconditioner_S2(self, which, tol_inv=1e-15, drop_tol=1e-4, fill_fac=10.0):
        assert which == "LU" or which == "ILU" or which == "FFT"

        # ------------------- LU/ILU preconditioner ------------------------
        if which == "ILU" or which == "LU":
            # assemble full weighted mass matrix Mn
            if self.Mn_as_tensor:
                if self.core.basis_u == 0:
                    Mn = spa.bmat(
                        [
                            [spa.kron(self.Mn_mat[0], self.core.space.M0_tor), None],
                            [None, spa.kron(self.Mn_mat[1], self.core.space.M0_tor)],
                        ],
                        format="csr",
                    )

                if self.core.basis_u == 2:
                    Mn = spa.bmat(
                        [
                            [spa.kron(self.Mn_mat[0], self.core.space.M1_tor), None],
                            [None, spa.kron(self.Mn_mat[1], self.core.space.M0_tor)],
                        ],
                        format="csr",
                    )

            else:
                Mn = self.Mn_mat.copy()

            # assemble approximations for inverse interpolation matrices
            self.core.space.projectors.assemble_approx_inv(tol_inv)

            # assemble approximate EF matrix
            if self.EF_as_tensor:
                if self.core.basis_u == 0:
                    EF_11 = spa.kron(self.dofs_EF[0], self.int_N3)
                    EF_12 = spa.kron(self.dofs_EF[1], self.int_N3)
                    EF_21 = spa.kron(self.dofs_EF[2], self.his_N3)

                if self.core.basis_u == 2:
                    EF_11 = spa.kron(self.dofs_EF[0], self.int_D3)
                    EF_12 = spa.kron(self.dofs_EF[1], self.int_N3)
                    EF_21 = spa.kron(self.dofs_EF[2], self.his_D3)

                EF_approx = spa.bmat([[EF_11, EF_12], [EF_21, None]], format="csr")

                del EF_11, EF_12, EF_21

                EF_approx = self.core.space.projectors.I1_0_inv_approx.dot(EF_approx)

            else:
                EF_approx = self.core.space.projectors.I1_0_inv_approx.dot(self.dofs_EF)

            # assemble full mass matrix M2_0
            if self.core.space.M2_as_tensor:
                M2_11 = spa.kron(self.core.space.M2_pol_mat[0], self.core.space.M1_tor)
                M2_22 = spa.kron(self.core.space.M2_pol_mat[1], self.core.space.M0_tor)

                M2_0 = spa.bmat([[M2_11, None], [None, M2_22]], format="csr")

                del M2_11, M2_22

            else:
                M2_0 = self.core.space.M2_mat

            M2_0 = self.core.space.B2.dot(M2_0.dot(self.core.space.B2.T)).tocsr()

            # assemble approximate S2 matrix
            S2_approx = Mn + self.dt_2**2 / 4 * EF_approx.T.dot(
                self.core.space.C0.T.dot(M2_0.dot(self.core.space.C0.dot(EF_approx)))
            )

            del Mn, EF_approx, M2_0

            # compute LU/ILU of approximate S2 matrix
            if which == "LU":
                S2_LU = spa.linalg.splu(S2_approx.tocsc())
            else:
                S2_LU = spa.linalg.spilu(S2_approx.tocsc(), drop_tol=drop_tol, fill_factor=fill_fac)

            self.S2_PRE = spa.linalg.LinearOperator(S2_approx.shape, S2_LU.solve)
        # ---------------------------------------------------------------------

        # ----------------------- FFT preconditioner --------------------------
        elif which == "FFT":

            def solve_S2(x):
                return self.Mn_inv(x)

            self.S2_PRE = spa.linalg.LinearOperator(self.Mn_inv.shape, solve_S2)
        # ---------------------------------------------------------------------

    # ======================================
    def set_preconditioner_S6(self, which, tol_inv=1e-15, drop_tol=1e-4, fill_fac=10.0):
        assert which == "LU" or which == "ILU" or which == "FFT"

        # -------------------------- LU/ILU preconditioner -------------------------
        if which == "ILU" or which == "LU":
            # assemble full weighted mass matrix Mn
            if self.Mn_as_tensor:
                if self.core.basis_u == 0:
                    Mn = spa.bmat(
                        [
                            [spa.kron(self.Mn_mat[0], self.core.space.M0_tor), None],
                            [None, spa.kron(self.Mn_mat[1], self.core.space.M0_tor)],
                        ],
                        format="csr",
                    )

                if self.core.basis_u == 2:
                    Mn = spa.bmat(
                        [
                            [spa.kron(self.Mn_mat[0], self.core.space.M1_tor), None],
                            [None, spa.kron(self.Mn_mat[1], self.core.space.M0_tor)],
                        ],
                        format="csr",
                    )

            else:
                Mn = self.Mn_mat.copy()

            # assemble approximations for inverse interpolation matrices
            self.core.space.projectors.assemble_approx_inv(tol_inv)

            # assemble approximate PF matrix
            if self.PF_as_tensor:
                if self.core.basis_u == 0:
                    PF_11 = spa.kron(self.dofs_PF[0], self.his_N3)
                    PF_22 = spa.kron(self.dofs_PF[1], self.int_N3)
                if self.core.basis_u == 2:
                    PF_11 = spa.kron(self.dofs_PF[0], self.his_D3)
                    PF_22 = spa.kron(self.dofs_PF[1], self.int_N3)

                PF_approx = spa.bmat([[PF_11, None], [None, PF_22]], format="csr")

                del PF_11, PF_22

                PF_approx = self.core.space.projectors.I2_0_inv_approx.dot(PF_approx)

            else:
                PF_approx = self.core.space.projectors.I2_0_inv_approx.dot(self.dofs_PF)

            # assemble approximate JF matrix (only for 0-form MHD)
            if self.core.basis_u == 0:
                if self.JF_as_tensor:
                    JF_11 = spa.kron(self.dofs_JF[0], self.his_N3)
                    JF_22 = spa.kron(self.dofs_JF[1], self.int_N3)

                    JF_approx = spa.bmat([[JF_11, None], [None, JF_22]], format="csr")

                    del JF_11, JF_22

                    JF_approx = self.core.space.projectors.I2_0_inv_approx.dot(JF_approx)

                else:
                    JF_approx = self.core.space.projectors.I2_0_inv_approx.dot(self.dofs_JF)

            # assemble approximate PR matrix
            if self.PR_as_tensor:
                PR_approx = spa.kron(self.dofs_PF, self.his_D3)
                PR_approx = self.core.space.projectors.I3_0_inv_approx.dot(PR_approx)

            else:
                PR_approx = self.core.space.projectors.I3_0_inv_approx.dot(self.dofs_PR)

            # assemble approximate L matrix
            if self.core.basis_u == 0:
                L_approx = -self.core.space.D0.dot(PF_approx) - (self.gamma - 1) * PR_approx.dot(
                    self.core.space.D0.dot(JF_approx)
                )

                del PF_approx, PR_approx

            if self.core.basis_u == 2:
                L_approx = -self.core.space.D0.dot(PF_approx) - (self.gamma - 1) * PR_approx.dot(self.core.space.D0)

                del PF_approx, PR_approx

            # assemble full mass matrix M3
            if self.core.space.M3_as_tensor:
                M3_0 = spa.kron(self.core.space.M3_pol_mat, self.core.space.M1_tor)
            else:
                M3_0 = self.core.space.M3_mat

            M3_0 = self.core.space.B3.dot(M3_0.dot(self.core.space.B3.T)).tocsr()

            # assemble approximate S6 matrix
            if self.core.basis_u == 0:
                S6_approx = Mn - self.dt_6**2 / 4 * JF_approx.T.dot(self.core.space.D0.T.dot(M3_0.dot(L_approx)))

                del Mn, JF_approx, M3_0, L_approx

            if self.core.basis_u == 2:
                S6_approx = Mn - self.dt_6**2 / 4 * self.core.space.D0.T.dot(M3_0.dot(L_approx))

                del Mn, M3_0, L_approx

            # compute LU/ILU of approximate S6 matrix
            if which == "LU":
                S6_LU = spa.linalg.splu(S6_approx.tocsc())
            else:
                S6_LU = spa.linalg.spilu(S6_approx.tocsc(), drop_tol=drop_tol, fill_factor=fill_fac)

            self.S6_PRE = spa.linalg.LinearOperator(S6_approx.shape, S6_LU.solve)
        # ---------------------------------------------------------------------------

        # -------------------------- FFT preconditioner -----------------------------
        elif which == "FFT":

            def solve_S6(x):
                return self.Mn_inv(x)

            self.S6_PRE = spa.linalg.LinearOperator(self.Mn_inv.shape, solve_S6)
        # ---------------------------------------------------------------------------
