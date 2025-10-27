import time

import cunumpy as xp
import scipy.sparse as spa

import struphy.feec.massless_operators.fB_bb_kernel as bb_kernel
import struphy.feec.massless_operators.fB_bv_kernel as bv_kernel
import struphy.feec.massless_operators.fB_vv_kernel as vv_kernel


class Massless_linear_operators:
    """
    Linear operators in substep vv, bb, and bv of fB formulation
    Parameters
    ----------
        DOMAIN : obj,
            Domain object from geometry/domain_3d.

        SPACES : obj,
            FEEC self.SPACES., store information of tensor products of B-splines

        KIN    : obj,
            obj storing information of particles
    """

    def __init__(self, SPACES, DOMAIN, KIN):
        self.indN = SPACES.indN
        self.indD = SPACES.indD
        self.Np_loc = KIN.Np_loc
        self.Np = KIN.Np
        self.Ntot_1form = SPACES.Ntot_1form
        self.Ntot_2form = SPACES.Ntot_2form
        self.Nel = SPACES.Nel
        self.n_quad = SPACES.n_quad
        self.p = SPACES.p
        self.d = [self.p[0] - 1, self.p[1] - 1, self.p[2] - 1]
        self.basisN = SPACES.basisN
        self.basisD = SPACES.basisD
        self.Nbase_2form = SPACES.Nbase_2form
        self.Ntot_2form = SPACES.Ntot_2form
        self.Nbase_1form = SPACES.Nbase_1form
        self.Ntot_1form = SPACES.Ntot_1form
        self.Nel = SPACES.Nel
        self.NbaseN = SPACES.NbaseN
        self.NbaseD = SPACES.NbaseD

    def linearoperator_step_vv(self, M2_PRE, M2, M1_PRE, M1, TEMP, ACC_VV):
        """
        This function is used in substep vv with L2 projector.
        """

        dft = xp.empty((3, 3), dtype=float)
        generate_weight1 = xp.zeros(3, dtype=float)
        generate_weight2 = xp.zeros(3, dtype=float)
        generate_weight3 = xp.zeros(3, dtype=float)
        # =========================inverse of M1 ===========================
        ACC_VV.temp1[:], ACC_VV.temp2[:], ACC_VV.temp3[:] = xp.split(
            spa.linalg.cg(
                M1,
                1.0 / self.Np * xp.concatenate((ACC_VV.vec1.flatten(), ACC_VV.vec2.flatten(), ACC_VV.vec3.flatten())),
                tol=10 ** (-14),
                M=M1_PRE,
            )[0],
            [self.Ntot_2form[0], self.Ntot_2form[0] + self.Ntot_2form[1]],
        )
        ACC_VV.one_form1[:, :, :] = ACC_VV.temp1.reshape(self.Nbase_1form[0])
        ACC_VV.one_form2[:, :, :] = ACC_VV.temp2.reshape(self.Nbase_1form[1])
        ACC_VV.one_form3[:, :, :] = ACC_VV.temp3.reshape(self.Nbase_1form[2])

        # ==========================Qvv * vector ============================
        vv_kernel.right_hand(
            self.indN[0],
            self.indN[1],
            self.indN[2],
            self.indD[0],
            self.indD[1],
            self.indD[2],
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            self.p[0],
            self.p[1],
            self.p[2],
            self.d[0],
            self.d[1],
            self.d[2],
            self.basisN[0],
            self.basisN[1],
            self.basisN[2],
            self.basisD[0],
            self.basisD[1],
            self.basisD[2],
            TEMP.LO_r1,
            TEMP.LO_r2,
            TEMP.LO_r3,
            ACC_VV.one_form1,
            ACC_VV.one_form2,
            ACC_VV.one_form3,
        )

        vv_kernel.weight(
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            TEMP.LO_b1,
            TEMP.LO_b2,
            TEMP.LO_b3,
            TEMP.LO_r1,
            TEMP.LO_r2,
            TEMP.LO_r3,
            TEMP.LO_w1,
            TEMP.LO_w2,
            TEMP.LO_w3,
        )

        vv_kernel.final(
            self.indN[0],
            self.indN[1],
            self.indN[2],
            self.indD[0],
            self.indD[1],
            self.indD[2],
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            self.p[0],
            self.p[1],
            self.p[2],
            self.d[0],
            self.d[1],
            self.d[2],
            TEMP.LO_w1,
            TEMP.LO_w2,
            TEMP.LO_w3,
            ACC_VV.one_form1,
            ACC_VV.one_form2,
            ACC_VV.one_form3,
            self.basisN[0],
            self.basisN[1],
            self.basisN[2],
            self.basisD[0],
            self.basisD[1],
            self.basisD[2],
        )

        # =========================inverse of M1 ===========================
        ACC_VV.temp1[:], ACC_VV.temp2[:], ACC_VV.temp3[:] = xp.split(
            spa.linalg.cg(
                M1,
                xp.concatenate((ACC_VV.one_form1.flatten(), ACC_VV.one_form2.flatten(), ACC_VV.one_form3.flatten())),
                tol=10 ** (-14),
                M=M1_PRE,
            )[0],
            [self.Ntot_1form[0], self.Ntot_1form[0] + self.Ntot_1form[1]],
        )
        ACC_VV.coe1[:, :, :] = ACC_VV.temp1.reshape(self.Nbase_1form[0])
        ACC_VV.coe2[:, :, :] = ACC_VV.temp2.reshape(self.Nbase_1form[1])
        ACC_VV.coe3[:, :, :] = ACC_VV.temp3.reshape(self.Nbase_1form[2])

    def local_linearoperator_step_vv(indN, indD, ACC_VV, M2_PRE, M2, Np, TEMP):
        """
        This function is used in substep vv with local projector
        """
        # ============= load information about B-splines =============
        p = ACC_VV.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D splin degrees
        Nel = ACC_VV.Nel  # number of elements
        n_quad = ACC_VV.tensor_space_FEM.n_quad  # number of quadrature points per element
        basisN = ACC_VV.tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
        basisD = ACC_VV.tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)

        # ==========================
        vv_kernel.right_hand(
            indN[0],
            indN[1],
            indN[2],
            indD[0],
            indD[1],
            indD[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            basisN[0],
            basisN[1],
            basisN[2],
            basisD[0],
            basisD[1],
            basisD[2],
            TEMP.right_1,
            TEMP.right_2,
            TEMP.right_3,
            ACC_VV.vec1,
            ACC_VV.vec2,
            ACC_VV.vec3,
        )

        vv_kernel.weight(
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            TEMP.b1value,
            TEMP.b2value,
            TEMP.b3value,
            TEMP.right_1,
            TEMP.right_2,
            TEMP.right_3,
            TEMP.weight1,
            TEMP.weight2,
            TEMP.weight3,
        )

        vv_kernel.final(
            indN[0],
            indN[1],
            indN[2],
            indD[0],
            indD[1],
            indD[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            TEMP.weight1,
            TEMP.weight2,
            TEMP.weight3,
            ACC_VV.one_form1,
            ACC_VV.one_form2,
            ACC_VV.one_form3,
            basisN[0],
            basisN[1],
            basisN[2],
            basisD[0],
            basisD[1],
            basisD[2],
        )

        ACC_VV.coe1[:, :, :] = ACC_VV.one_form1
        ACC_VV.coe2[:, :, :] = ACC_VV.one_form2
        ACC_VV.coe3[:, :, :] = ACC_VV.one_form3

    def linearoperator_pre_step_vv(
        self,
        tensor_space_FEM,
        df_det,
        DFIT_11,
        DFIT_12,
        DFIT_13,
        DFIT_21,
        DFIT_22,
        DFIT_23,
        DFIT_31,
        DFIT_32,
        DFIT_33,
        M2_PRE,
        M2,
        b1,
        b2,
        b3,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        uvalue,
        b1value,
        b2value,
        b3value,
    ):
        """
        This function is used in substep vv with L2 projector or local projector.
        """
        # =====we can just calculate 3 matrices=====
        # ============= load information about B-splines =============
        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D splin degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights
        indN = tensor_space_FEM.indN
        indD = tensor_space_FEM.indD

        dft = xp.empty((3, 3), dtype=float)
        generate_weight1 = xp.zeros(3, dtype=float)
        generate_weight2 = xp.zeros(3, dtype=float)
        generate_weight3 = xp.zeros(3, dtype=float)

        vv_kernel.prepre(
            indN[0],
            indN[1],
            indN[2],
            indD[0],
            indD[1],
            indD[2],
            df_det,
            DFIT_11,
            DFIT_12,
            DFIT_13,
            DFIT_21,
            DFIT_22,
            DFIT_23,
            DFIT_31,
            DFIT_32,
            DFIT_33,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            b1value,
            b2value,
            b3value,
            uvalue,
            b1,
            b2,
            b3,
            dft,
            generate_weight1,
            generate_weight3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            pts[0],
            pts[1],
            pts[2],
            wts[0],
            wts[1],
            wts[2],
        )

    def gather(self, index, dt, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np):
        """
        This function is used in substep vv with scatter-gather algorithm.
        """
        if index == 1:
            vv_kernel.piecewise_gather(
                0.0,
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.n_quad,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.Nel,
                particles_loc,
                Np_loc,
                Np,
                acc.gather1_loc,
                acc.gather2_loc,
                acc.gather3_loc,
                acc.mid_particles,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        elif index == 2:
            vv_kernel.piecewise_gather(
                0.5 * dt,
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.n_quad,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.Nel,
                particles_loc,
                Np_loc,
                Np,
                acc.gather1_loc,
                acc.gather2_loc,
                acc.gather3_loc,
                acc.stage1_out_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        elif index == 3:
            vv_kernel.piecewise_gather(
                0.5 * dt,
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.n_quad,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.Nel,
                particles_loc,
                Np_loc,
                Np,
                acc.gather1_loc,
                acc.gather2_loc,
                acc.gather3_loc,
                acc.stage2_out_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        else:
            vv_kernel.piecewise_gather(
                dt,
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.n_quad,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.Nel,
                particles_loc,
                Np_loc,
                Np,
                acc.gather1_loc,
                acc.gather2_loc,
                acc.gather3_loc,
                acc.stage3_out_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )

    def scatter_gather_weight(self, acc, tensor_space_FEM, b1value, b2value, b3value):
        vv_kernel.weight(
            tensor_space_FEM.Nel[0],
            tensor_space_FEM.Nel[1],
            tensor_space_FEM.Nel[2],
            acc.n_quad[0],
            acc.n_quad[1],
            acc.n_quad[2],
            b1value,
            b2value,
            b3value,
            acc.gather1,
            acc.gather2,
            acc.gather3,
            acc.weight1,
            acc.weight2,
            acc.weight3,
        )

    def scatter(self, index, acc, tensor_space_FEM, domain, particles_loc, Np_loc, Np):
        """
        This function is used in substep vv with scatter-gather algorithm.
        """
        if index == 1:
            vv_kernel.piecewise_scatter(
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.stage1_out_loc,
                tensor_space_FEM.Nel,
                Np_loc,
                Np,
                acc.weight1,
                acc.weight2,
                acc.weight3,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[1],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.n_quad,
                particles_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        elif index == 2:
            vv_kernel.piecewise_scatter(
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.stage2_out_loc,
                tensor_space_FEM.Nel,
                Np_loc,
                Np,
                acc.weight1,
                acc.weight2,
                acc.weight3,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[1],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.n_quad,
                particles_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        elif index == 3:
            vv_kernel.piecewise_scatter(
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.stage3_out_loc,
                tensor_space_FEM.Nel,
                Np_loc,
                Np,
                acc.weight1,
                acc.weight2,
                acc.weight3,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[1],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.n_quad,
                particles_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )
        else:
            vv_kernel.piecewise_scatter(
                acc.index_shapex,
                acc.index_shapey,
                acc.index_shapez,
                acc.index_diffx,
                acc.index_diffy,
                acc.index_diffz,
                acc.p_shape,
                acc.p_size,
                acc.stage4_out_loc,
                tensor_space_FEM.Nel,
                Np_loc,
                Np,
                acc.weight1,
                acc.weight2,
                acc.weight3,
                tensor_space_FEM.pts[0],
                tensor_space_FEM.pts[1],
                tensor_space_FEM.pts[2],
                tensor_space_FEM.n_quad,
                particles_loc,
                domain.kind_map,
                domain.params,
                domain.T[0],
                domain.T[1],
                domain.T[2],
                domain.p,
                domain.Nel,
                domain.NbaseN,
                domain.cx,
                domain.cy,
                domain.cz,
            )

    def linearoperator_step3(
        self,
        twoform_temp1_long,
        twoform_temp2_long,
        twoform_temp3_long,
        temp_vector_1,
        temp_vector_2,
        temp_vector_3,
        idn,
        idd,
        tensor_space_FEM,
        dt,
        input_vector,
        b1,
        b2,
        b3,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        uvalue,
        b1value,
        b2value,
        b3value,
    ):
        """
        This function is used in substep bb.
        """
        # ============= load information about B-splines =============
        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
        NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (D)
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        Ntot_2form = tensor_space_FEM.Ntot_2form
        Nbase_2form = tensor_space_FEM.Nbase_2form

        dft = xp.empty((3, 3), dtype=float)
        generate_weight1 = xp.empty(3, dtype=float)
        generate_weight2 = xp.empty(3, dtype=float)
        generate_weight3 = xp.empty(3, dtype=float)

        # ==================================================================
        # ========================= C ===========================
        # time1 = time.time()
        twoform_temp1_long[:], twoform_temp2_long[:], twoform_temp3_long[:] = xp.split(
            tensor_space_FEM.C.dot(input_vector),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        temp_vector_1[:, :, :] = twoform_temp1_long.reshape(Nbase_2form[0])
        temp_vector_2[:, :, :] = twoform_temp2_long.reshape(Nbase_2form[1])
        temp_vector_3[:, :, :] = twoform_temp3_long.reshape(Nbase_2form[2])
        # time2 = time.time()
        # print('curl_time', time2 - time1)
        # ==========================Q5 * vector ============================
        time1 = time.time()
        bb_kernel.right_hand(
            idn[0],
            idn[1],
            idn[2],
            idd[0],
            idd[1],
            idd[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            temp_vector_1,
            temp_vector_2,
            temp_vector_3,
        )
        time2 = time.time()
        # print('right_hand_time', time2 - time1)
        # time1 = time.time()
        bb_kernel.weight(
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        # time2 = time.time()
        # print('weight_time', time2 - time1)
        # time1 = time.time()
        bb_kernel.final(
            idn[0],
            idn[1],
            idn[2],
            idd[0],
            idd[1],
            idd[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            temp_vector_1,
            temp_vector_2,
            temp_vector_3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )
        # time2 = time.time()
        # print('time_final', time2 - time1)
        # ========================= C.T ===========================
        # time1 = time.time()
        temp_final = tensor_space_FEM.M1.dot(input_vector) - dt / 2.0 * tensor_space_FEM.C.T.dot(
            xp.concatenate((temp_vector_1.flatten(), temp_vector_2.flatten(), temp_vector_3.flatten())),
        )
        # time2 = time.time()
        # print('second_curl_time', time2 - time1)
        # print('gmres_number', 1)
        return temp_final

    def linearoperator_pre_step3(self, LO_inv, tensor_space_FEM):
        """
        This function is used in substep bb.
        """
        # ============= load information about B-splines =============
        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
        NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (D)
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
        basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)

        bb_kernel.pre(
            tensor_space_FEM.indN[0],
            tensor_space_FEM.indN[1],
            tensor_space_FEM.indN[2],
            tensor_space_FEM.indD[0],
            tensor_space_FEM.indD[1],
            tensor_space_FEM.indD[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            LO_inv,
            basisN[0],
            basisN[1],
            basisN[2],
        )

    def linearoperator_right_step3(
        self,
        twoform_temp1_long,
        twoform_temp2_long,
        twoform_temp3_long,
        temp_vector_1,
        temp_vector_2,
        temp_vector_3,
        idn,
        idd,
        tensor_space_FEM,
        G_inv_11,
        G_inv_12,
        G_inv_13,
        G_inv_22,
        G_inv_23,
        G_inv_33,
        dt,
        input_vector,
        b1,
        b2,
        b3,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        uvalue,
        b1value,
        b2value,
        b3value,
    ):
        """
        This function is used in substep bb.
        """
        # ============= load information about B-splines =============
        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D splin degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
        NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (D)
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        Ntot_2form = tensor_space_FEM.Ntot_2form
        Nbase_2form = tensor_space_FEM.Nbase_2form

        dft = xp.empty((3, 3), dtype=float)
        generate_weight1 = xp.empty(3, dtype=float)
        generate_weight2 = xp.empty(3, dtype=float)
        generate_weight3 = xp.empty(3, dtype=float)

        # ==================================================================
        # ========================= C ===========================
        twoform_temp1_long[:], twoform_temp2_long[:], twoform_temp3_long[:] = xp.split(
            tensor_space_FEM.C.dot(input_vector),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        temp_vector_1[:, :, :] = twoform_temp1_long.reshape(Nbase_2form[0])
        temp_vector_2[:, :, :] = twoform_temp2_long.reshape(Nbase_2form[1])
        temp_vector_3[:, :, :] = twoform_temp3_long.reshape(Nbase_2form[2])
        # ==========================Q5 * vector ============================
        # time1 = time.time()
        bb_kernel.right_hand(
            idn[0],
            idn[1],
            idn[2],
            idd[0],
            idd[1],
            idd[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            temp_vector_1,
            temp_vector_2,
            temp_vector_3,
        )
        # time2 = time.time()
        # print('right_hand_bb', time2 - time1)
        # time1 = time.time()
        bb_kernel.bvalue(
            idn[0],
            idn[1],
            idn[2],
            idd[0],
            idd[1],
            idd[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            b1,
            b2,
            b3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            b1value,
            b2value,
            b3value,
            temp_vector_1,
            temp_vector_2,
            temp_vector_3,
        )
        # time2 = time.time()
        # print('bvalue_time_bb', time2 - time1)
        # time1 = time.time()
        bb_kernel.right_bwvalue(
            G_inv_11,
            G_inv_12,
            G_inv_13,
            G_inv_22,
            G_inv_23,
            G_inv_33,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            dft,
            pts[0],
            pts[1],
            pts[2],
            wts[0],
            wts[1],
            wts[2],
            generate_weight1,
            generate_weight2,
            generate_weight3,
            b1value,
            b2value,
            b3value,
            uvalue,
            weight1,
            weight2,
            weight3,
            right_1,
            right_2,
            right_3,
        )
        # time2 = time.time()
        # print('bwvalue_time_bb', time2 - time1)
        # time1 = time.time()
        bb_kernel.final(
            idn[0],
            idn[1],
            idn[2],
            idd[0],
            idd[1],
            idd[2],
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            temp_vector_1,
            temp_vector_2,
            temp_vector_3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )
        # time2 = time.time()
        # print('final_bb', time2 - time1)
        # ========================= C.T ===========================
        temp_final = tensor_space_FEM.M1.dot(input_vector) + dt / 2.0 * tensor_space_FEM.C.T.dot(
            xp.concatenate((temp_vector_1.flatten(), temp_vector_2.flatten(), temp_vector_3.flatten())),
        )

        return temp_final

    # ==========================================================================================================
    def substep4_linear_operator(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        input,
        tensor_space_FEM,
        Ntot_2form,
        Ntot_1form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with L2 projector.
        """
        # input: b1value, b2value, b3value (as weight), bb1, bb2, bb3 (as right hand vector)

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            tensor_space_FEM.C.dot(input),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])
        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )
        acc.oneform_temp_long[:] = spa.linalg.gmres(
            M1,
            xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten())),
            tol=10 ** (-10),
            M=M1_PRE,
        )[0]

        acc.oneform_temp1_long[:], acc.oneform_temp2_long[:], acc.oneform_temp3_long[:] = xp.split(
            spa.linalg.gmres(M1, mat.dot(acc.oneform_temp_long), tol=10 ** (-10), M=M1_PRE)[0],
            [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]],
        )
        acc.oneform_temp1[:, :, :] = acc.oneform_temp1_long.reshape(Nbase_1form[0])
        acc.oneform_temp2[:, :, :] = acc.oneform_temp2_long.reshape(Nbase_1form[1])
        acc.oneform_temp3[:, :, :] = acc.oneform_temp3_long.reshape(Nbase_1form[2])
        bv_kernel.right_hand1(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
        )
        bv_kernel.weight_1(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_left(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return M1.dot(input) + dt**2 / 4.0 * tensor_space_FEM.C.T.dot(
            xp.concatenate((acc.twoform_temp1.flatten(), acc.twoform_temp2.flatten(), acc.twoform_temp3.flatten())),
        )

    # ==========================================================================================================
    def substep4_linear_operator_right(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        bb1,
        bb2,
        bb3,
        tensor_space_FEM,
        Ntot_2form,
        Ntot_1form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        vec,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with L2 projector.
        """

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            CURL.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])

        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )
        acc.oneform_temp_long[:] = mat.dot(
            spa.linalg.gmres(
                M1,
                xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten())),
                tol=10 ** (-10),
                M=M1_PRE,
            )[0],
        )

        acc.oneform_temp1_long[:], acc.oneform_temp2_long[:], acc.oneform_temp3_long[:] = xp.split(
            spa.linalg.gmres(M1, dt**2.0 / 4.0 * acc.oneform_temp_long + dt * vec, tol=10 ** (-10), M=M1_PRE)[0],
            [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]],
        )
        acc.oneform_temp1[:, :, :] = acc.oneform_temp1_long.reshape(Nbase_1form[0])
        acc.oneform_temp2[:, :, :] = acc.oneform_temp2_long.reshape(Nbase_1form[1])
        acc.oneform_temp3[:, :, :] = acc.oneform_temp3_long.reshape(Nbase_1form[2])
        bv_kernel.right_hand1(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
        )
        bv_kernel.weight_1(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_left(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return M1.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))) - CURL.T.dot(
            xp.concatenate((acc.twoform_temp1.flatten(), acc.twoform_temp2.flatten(), acc.twoform_temp3.flatten())),
        )

    # ==========================================================================================================
    def substep4_pre(
        self,
        df_det,
        dft,
        generate_weight1,
        generate_weight3,
        G_inv_11,
        G_inv_12,
        G_inv_13,
        G_inv_22,
        G_inv_23,
        G_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        bb1,
        bb2,
        bb3,
        tensor_space_FEM,
        b1value,
        b2value,
        b3value,
        uvalue,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv L2 projector or local projector.
        """
        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        bv_kernel.prepre(
            df_det,
            G_inv_11,
            G_inv_12,
            G_inv_13,
            G_inv_22,
            G_inv_23,
            G_inv_33,
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            b1value,
            b2value,
            b3value,
            uvalue,
            bb1,
            bb2,
            bb3,
            dft,
            generate_weight1,
            generate_weight3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            pts[0],
            pts[1],
            pts[2],
            wts[0],
            wts[1],
            wts[2],
        )

    # ==========================================================================================================
    def substep4_pusher_field(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        bb1,
        bb2,
        bb3,
        tensor_space_FEM,
        Ntot_2form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        vec,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with L2 projector.
        """

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            CURL.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])
        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return spa.linalg.cg(
            M1,
            xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten())),
            tol=10 ** (-13),
            M=M1_PRE,
        )[0]

    # ==========================================================================================================
    def substep4_localproj_linear_operator(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        input,
        tensor_space_FEM,
        Ntot_2form,
        Ntot_1form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with local projector.
        """

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            tensor_space_FEM.C.dot(input),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])
        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        acc.oneform_temp1_long[:], acc.oneform_temp2_long[:], acc.oneform_temp3_long[:] = xp.split(
            mat.dot(
                xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten())),
            ),
            [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]],
        )

        acc.oneform_temp1[:, :, :] = acc.oneform_temp1_long.reshape(Nbase_1form[0])
        acc.oneform_temp2[:, :, :] = acc.oneform_temp2_long.reshape(Nbase_1form[1])
        acc.oneform_temp3[:, :, :] = acc.oneform_temp3_long.reshape(Nbase_1form[2])
        bv_kernel.right_hand1(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
        )
        bv_kernel.weight_1(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_left(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return M1.dot(input) + dt**2 / 4.0 * tensor_space_FEM.C.T.dot(
            xp.concatenate((acc.twoform_temp1.flatten(), acc.twoform_temp2.flatten(), acc.twoform_temp3.flatten())),
        )

    # ==========================================================================================================
    def substep4_localproj_linear_operator_right(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        bb1,
        bb2,
        bb3,
        tensor_space_FEM,
        Ntot_2form,
        Ntot_1form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        vec,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with local projector.
        """

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            CURL.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])

        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )
        acc.oneform_temp_long[:] = mat.dot(
            xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten())),
        )

        acc.oneform_temp1_long[:], acc.oneform_temp2_long[:], acc.oneform_temp3_long[:] = xp.split(
            (dt**2.0 / 4.0 * acc.oneform_temp_long + dt * vec),
            [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]],
        )

        acc.oneform_temp1[:, :, :] = acc.oneform_temp1_long.reshape(Nbase_1form[0])
        acc.oneform_temp2[:, :, :] = acc.oneform_temp2_long.reshape(Nbase_1form[1])
        acc.oneform_temp3[:, :, :] = acc.oneform_temp3_long.reshape(Nbase_1form[2])
        bv_kernel.right_hand1(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
        )
        bv_kernel.weight_1(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_left(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return M1.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))) - CURL.T.dot(
            xp.concatenate((acc.twoform_temp1.flatten(), acc.twoform_temp2.flatten(), acc.twoform_temp3.flatten())),
        )

    # ==========================================================================================================
    def substep4_localproj_pusher_field(
        self,
        acc,
        dft,
        generate_weight1,
        generate_weight3,
        DF_inv_11,
        DF_inv_12,
        DF_inv_13,
        DF_inv_21,
        DF_inv_22,
        DF_inv_23,
        DF_inv_31,
        DF_inv_32,
        DF_inv_33,
        N_index_x,
        N_index_y,
        N_index_z,
        D_index_x,
        D_index_y,
        D_index_z,
        M1,
        M1_PRE,
        CURL,
        mat,
        bb1,
        bb2,
        bb3,
        tensor_space_FEM,
        Ntot_2form,
        Nbase_2form,
        Nbase_1form,
        weight1,
        weight2,
        weight3,
        right_1,
        right_2,
        right_3,
        b1value,
        b2value,
        b3value,
        vec,
        dt,
        kind_map,
        params_map,
    ):
        """
        This function is used in substep bv with local projector.
        """

        p = tensor_space_FEM.p  # spline degrees
        d = [p[0] - 1, p[1] - 1, p[2] - 1]  # D spline degrees
        Nel = tensor_space_FEM.Nel  # number of elements
        n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
        pts = tensor_space_FEM.pts  # global quadrature points
        wts = tensor_space_FEM.wts  # global quadrature weights

        # ==========================================
        acc.twoform_temp1_long[:], acc.twoform_temp2_long[:], acc.twoform_temp3_long[:] = xp.split(
            CURL.dot(xp.concatenate((bb1.flatten(), bb2.flatten(), bb3.flatten()))),
            [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]],
        )
        acc.twoform_temp1[:, :, :] = acc.twoform_temp1_long.reshape(Nbase_2form[0])
        acc.twoform_temp2[:, :, :] = acc.twoform_temp2_long.reshape(Nbase_2form[1])
        acc.twoform_temp3[:, :, :] = acc.twoform_temp3_long.reshape(Nbase_2form[2])
        bv_kernel.right_hand2(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
            right_1,
            right_2,
            right_3,
            acc.twoform_temp1,
            acc.twoform_temp2,
            acc.twoform_temp3,
        )
        bv_kernel.weight_2(
            DF_inv_11,
            DF_inv_12,
            DF_inv_13,
            DF_inv_21,
            DF_inv_22,
            DF_inv_23,
            DF_inv_31,
            DF_inv_32,
            DF_inv_33,
            pts[0],
            pts[1],
            pts[2],
            dft,
            generate_weight1,
            generate_weight3,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            b1value,
            b2value,
            b3value,
            right_1,
            right_2,
            right_3,
            weight1,
            weight2,
            weight3,
        )
        bv_kernel.final_right(
            N_index_x,
            N_index_y,
            N_index_z,
            D_index_x,
            D_index_y,
            D_index_z,
            Nel[0],
            Nel[1],
            Nel[2],
            n_quad[0],
            n_quad[1],
            n_quad[2],
            p[0],
            p[1],
            p[2],
            d[0],
            d[1],
            d[2],
            weight1,
            weight2,
            weight3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            tensor_space_FEM.basisN[0],
            tensor_space_FEM.basisN[1],
            tensor_space_FEM.basisN[2],
            tensor_space_FEM.basisD[0],
            tensor_space_FEM.basisD[1],
            tensor_space_FEM.basisD[2],
        )

        return xp.concatenate((acc.oneform_temp1.flatten(), acc.oneform_temp2.flatten(), acc.oneform_temp3.flatten()))
