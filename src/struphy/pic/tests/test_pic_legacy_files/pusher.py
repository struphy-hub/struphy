import struphy.pic.tests.test_pic_legacy_files.pusher_pos as push_pos
import struphy.pic.tests.test_pic_legacy_files.pusher_vel_2d as push_vel_2d
import struphy.pic.tests.test_pic_legacy_files.pusher_vel_3d as push_vel_3d
from struphy.utils.arrays import xp as np


class Pusher:
    """
    TODO
    """

    def __init__(self, domain, fem_space, b0_eq, b2_eq, basis_u, bc_pos):
        # mapped domain
        self.domain = domain

        # set pseudo-cartesian mapping parameters in case of polar domains
        if self.domain.pole:
            # IGA straight
            if self.domain.kind_map == 1:
                self.map_pseudo, self.R0_pseudo = 20, self.domain.cx[0, 0, 0]

            # IGA toroidal
            if self.domain.kind_map == 2:
                self.map_pseudo, self.R0_pseudo = 22, self.domain.cx[0, 0, 0]

            # analytical hollow cylinder
            if self.domain.kind_map == 20:
                self.map_pseudo, self.R0_pseudo = 20, self.domain.params_numpy[2]

            # analytical hollow torus
            if self.domain.kind_map == 22:
                self.map_pseudo, self.R0_pseudo = 22, self.domain.params_numpy[2]

        # FEM space for perturbed fields
        self.fem_space = fem_space

        # equilibrium magnetic FE coefficients
        assert b0_eq.shape[:2] == (self.fem_space.NbaseN[0], self.fem_space.NbaseN[1])

        self.b0_eq = b0_eq

        assert b2_eq[0].shape[:2] == (self.fem_space.NbaseN[0], self.fem_space.NbaseD[1])
        assert b2_eq[1].shape[:2] == (self.fem_space.NbaseD[0], self.fem_space.NbaseN[1])
        assert b2_eq[2].shape[:2] == (self.fem_space.NbaseD[0], self.fem_space.NbaseD[1])

        self.b2_eq = b2_eq

        # basis of perturbed velocity field
        assert basis_u == 0 or basis_u == 1 or basis_u == 2

        self.basis_u = basis_u

        # boundary condition in s-direction (0 : periodic, 1 : absorbing)
        self.bc_pos = bc_pos

    # ======================================================
    def push_step3(self, particles, dt, b2, up, mu_0, power):
        """
        TODO
        """

        # extract flattened magnetic FE coefficients
        b2 = self.fem_space.extract_2(b2)

        # extract flattened velocity FE coefficients
        if self.basis_u == 0:
            up = self.fem_space.extract_v(up)
        elif self.basis_u == 1:
            up = self.fem_space.extract_1(up)
        elif self.basis_u == 2:
            up = self.fem_space.extract_2(up)

        # push particles
        if self.fem_space.dim == 2:
            push_vel_2d.pusher_step3(
                particles,
                dt,
                self.fem_space.T[0],
                self.fem_space.T[1],
                self.fem_space.p,
                self.fem_space.Nel,
                self.fem_space.NbaseN,
                self.fem_space.NbaseD,
                particles.shape[1],
                self.b2_eq[0],
                self.b2_eq[1],
                self.b2_eq[2],
                b2[0],
                b2[1],
                b2[2],
                self.b0_eq,
                up[0],
                up[1],
                up[2],
                self.basis_u,
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
                mu_0,
                power,
                self.fem_space.n_tor,
            )

        else:
            push_vel_3d.pusher_step3(
                particles,
                dt,
                self.fem_space.T[0],
                self.fem_space.T[1],
                self.fem_space.T[2],
                self.fem_space.p,
                self.fem_space.Nel,
                self.fem_space.NbaseN,
                self.fem_space.NbaseD,
                particles.shape[1],
                self.b2_eq[0] + b2[0],
                self.b2_eq[1] + b2[1],
                self.b2_eq[2] + b2[2],
                self.b0_eq,
                up[0],
                up[1],
                up[2],
                self.basis_u,
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
                mu_0,
                power,
            )

    # ======================================================
    def push_step4(self, particles, dt):
        """
        TODO
        """

        # modified pusher in pseudo cartesian coordinates (for polar domain)
        if self.domain.pole:
            push_pos.pusher_step4_pcart(
                particles,
                dt,
                particles.shape[1],
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
                self.map_pseudo,
                self.R0_pseudo,
            )

        # standard pusher in logical coordinates (for domains without a pole)
        else:
            push_pos.pusher_step4(
                particles,
                dt,
                particles.shape[1],
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
                self.bc_pos,
            )

    # ======================================================
    def push_step5(self, particles, dt, b2):
        """
        TODO
        """

        # extract flattened magnetic FE coefficients
        b2 = self.fem_space.extract_2(b2)

        # push particles
        if self.fem_space.dim == 2:
            push_vel_2d.pusher_step5(
                particles,
                dt,
                self.fem_space.T[0],
                self.fem_space.T[1],
                self.fem_space.p,
                self.fem_space.Nel,
                self.fem_space.NbaseN,
                self.fem_space.NbaseD,
                particles.shape[1],
                self.b2_eq[0],
                self.b2_eq[1],
                self.b2_eq[2],
                b2[0],
                b2[1],
                b2[2],
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
                self.fem_space.n_tor,
            )

        else:
            push_vel_3d.pusher_step5(
                particles,
                dt,
                self.fem_space.T[0],
                self.fem_space.T[1],
                self.fem_space.T[2],
                self.fem_space.p,
                self.fem_space.Nel,
                self.fem_space.NbaseN,
                self.fem_space.NbaseD,
                particles.shape[1],
                self.b2_eq[0] + b2[0],
                self.b2_eq[1] + b2[1],
                self.b2_eq[2] + b2[2],
                self.domain.kind_map,
                self.domain.params_numpy,
                self.domain.T[0],
                self.domain.T[1],
                self.domain.T[2],
                self.domain.p,
                self.domain.Nel,
                self.domain.NbaseN,
                self.domain.cx,
                self.domain.cy,
                self.domain.cz,
            )

    # ======================================================
    def push_eta_pc_full(self, particles, dt, up):
        """
        TODO
        """

        # extract flattened flow field FE coefficients
        if self.basis_u == 1:
            up = self.fem_space.extract_1(up)
        elif self.basis_u == 2:
            up = self.fem_space.extract_2(up)
        else:
            up = self.fem_space.extract_v(up)

        # push particles
        push_pos.pusher_rk4_pc_full(
            particles,
            dt,
            self.fem_space.T[0],
            self.fem_space.T[1],
            self.fem_space.T[2],
            self.fem_space.p,
            self.fem_space.Nel,
            self.fem_space.NbaseN,
            self.fem_space.NbaseD,
            particles.shape[1],
            up[0],
            up[1],
            up[2],
            self.basis_u,
            self.domain.kind_map,
            self.domain.params_numpy,
            self.domain.T[0],
            self.domain.T[1],
            self.domain.T[2],
            self.domain.p,
            self.domain.Nel,
            self.domain.NbaseN,
            self.domain.cx,
            self.domain.cy,
            self.domain.cz,
            self.bc_pos,
        )

    # ======================================================
    def push_eta_pc_perp(self, particles, dt, up):
        """
        TODO
        """

        # extract flattened magnetic FE coefficients
        if self.basis_u == 1:
            up = self.fem_space.extract_1(up)
        elif self.basus_u == 2:
            up = self.fem_space.extract_2(up)
        else:
            up[0] = self.fem_space.extract_0(up[0])
            up[1] = self.fem_space.extract_0(up[1])
            up[2] = self.fem_space.extract_0(up[2])

        # push particles
        push_pos.pusher_rk4_pc_perp(
            particles,
            dt,
            self.fem_space.T[0],
            self.fem_space.T[1],
            self.fem_space.T[2],
            self.fem_space.p,
            self.fem_space.Nel,
            self.fem_space.NbaseN,
            self.fem_space.NbaseD,
            particles.shape[1],
            up[0],
            up[1],
            up[2],
            self.basis_u,
            self.domain.kind_map,
            self.domain.params_numpy,
            self.domain.T[0],
            self.domain.T[1],
            self.domain.T[2],
            self.domain.p,
            self.domain.Nel,
            self.domain.NbaseN,
            self.domain.cx,
            self.domain.cy,
            self.domain.cz,
            self.bc_pos,
        )

    # ======================================================
    def push_vel_pc_full(self, particles, dt, GXu_1, GXu_2, GXu_3):
        """
        TODO
        """

        # extract flattened magnetic FE coefficients
        GXu_1_1, GXu_1_2, GXu_1_3 = self.fem_space.extract_1(GXu_1)
        GXu_2_1, GXu_2_2, GXu_2_3 = self.fem_space.extract_1(GXu_2)
        GXu_3_1, GXu_3_2, GXu_3_3 = self.fem_space.extract_1(GXu_3)

        # push particles
        push_vel_3d.pusher_v_pressure_full(
            particles,
            dt,
            self.fem_space.T[0],
            self.fem_space.T[1],
            self.fem_space.T[2],
            self.fem_space.p,
            self.fem_space.Nel,
            self.fem_space.NbaseN,
            self.fem_space.NbaseD,
            particles.shape[1],
            GXu_1_1,
            GXu_1_2,
            GXu_1_3,
            GXu_2_1,
            GXu_2_2,
            GXu_2_3,
            GXu_3_1,
            GXu_3_2,
            GXu_3_3,
            self.domain.kind_map,
            self.domain.params_numpy,
            self.domain.T[0],
            self.domain.T[1],
            self.domain.T[2],
            self.domain.p,
            self.domain.Nel,
            self.domain.NbaseN,
            self.domain.cx,
            self.domain.cy,
            self.domain.cz,
        )

    # ======================================================
    def push_vel_pc_perp(self, particles, dt, GXu_1, GXu_2, GXu_3):
        """
        TODO
        """

        # extract flattened magnetic FE coefficients
        GXu_1_1, GXu_1_2, GXu_1_3 = self.fem_space.extract_1(GXu_1)
        GXu_2_1, GXu_2_2, GXu_2_3 = self.fem_space.extract_1(GXu_2)
        GXu_3_1, GXu_3_2, GXu_3_3 = self.fem_space.extract_1(GXu_3)

        # push particles
        push_vel_3d.pusher_v_pressure_perp(
            particles,
            dt,
            self.fem_space.T[0],
            self.fem_space.T[1],
            self.fem_space.T[2],
            self.fem_space.p,
            self.fem_space.Nel,
            self.fem_space.NbaseN,
            self.fem_space.NbaseD,
            particles.shape[1],
            GXu_1_1,
            GXu_1_2,
            GXu_1_3,
            GXu_2_1,
            GXu_2_2,
            GXu_2_3,
            GXu_3_1,
            GXu_3_2,
            GXu_3_3,
            self.domain.kind_map,
            self.domain.params_numpy,
            self.domain.T[0],
            self.domain.T[1],
            self.domain.T[2],
            self.domain.p,
            self.domain.Nel,
            self.domain.NbaseN,
            self.domain.cx,
            self.domain.cy,
            self.domain.cz,
        )
