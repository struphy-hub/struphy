# coding: utf-8
#

"""
Classes for local projectors in 1D and 3D based on quasi-spline interpolation and histopolation.
"""

import cunumpy as xp
import scipy.sparse as spa
from psydac.ddm.mpi import mpi as MPI

import struphy.feec.bsplines as bsp
import struphy.feec.projectors.shape_pro_local.shape_local_projector_kernel as ker_loc


# ======================= 3d ====================================
class projectors_local_3d:
    """
    Local commuting projectors pi_0, pi_1, pi_2 and pi_3 in 3d with smoothed delta functions
    only the periodic case is implemented.
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 3d tensor product space of B-splines

    n_quad : list of ints
        number of quadrature points per integration interval for histopolations

    p_shape: list of ints B-spline degrees of shape functions in three directions
    p_size : list of doubles cell size of smoothed delta function in three directions
    NbaseN : list of ints number of N splines in three directions
    NbaseD : list of ints number of D splines in three directions
    mpi_commm: mpi encironment
    """

    def __init__(self, tensor_space, n_quad, p_shape, p_size, NbaseN, NbaseD, mpi_comm):
        self.kind = "local"  # kind of projector

        self.tensor_space = tensor_space  # 3D tensor-product B-splines space
        self.mpi_rank = mpi_comm.Get_rank()
        self.T = tensor_space.T  # knot vector
        self.p = tensor_space.p  # spline degree
        self.bc = tensor_space.spl_kind  # boundary conditions
        self.el_b = tensor_space.el_b  # element boundaries

        self.Nel = tensor_space.Nel  # number of elements
        self.NbaseN = tensor_space.NbaseN  # number of basis functions (N)
        self.NbaseD = tensor_space.NbaseD  # number of basis functions (D)

        self.n_quad = n_quad  # number of quadrature point per integration interval

        self.polar = False  # local projectors for polar splines are not implemented yet

        self.lambdas_0 = xp.zeros((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
        self.potential_lambdas_0 = xp.zeros((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)

        self.lambdas_1_11 = xp.zeros((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
        self.lambdas_1_12 = xp.zeros((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
        self.lambdas_1_13 = xp.zeros((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

        self.lambdas_1_21 = xp.zeros((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
        self.lambdas_1_22 = xp.zeros((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
        self.lambdas_1_23 = xp.zeros((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

        self.lambdas_1_31 = xp.zeros((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
        self.lambdas_1_32 = xp.zeros((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
        self.lambdas_1_33 = xp.zeros((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

        self.lambdas_2_11 = xp.zeros((NbaseN[0], NbaseD[1], NbaseD[2]), dtype=float)
        self.lambdas_2_12 = xp.zeros((NbaseD[0], NbaseN[1], NbaseD[2]), dtype=float)
        self.lambdas_2_13 = xp.zeros((NbaseD[0], NbaseD[1], NbaseN[2]), dtype=float)

        self.lambdas_2_21 = xp.zeros((NbaseN[0], NbaseD[1], NbaseD[2]), dtype=float)
        self.lambdas_2_22 = xp.zeros((NbaseD[0], NbaseN[1], NbaseD[2]), dtype=float)
        self.lambdas_2_23 = xp.zeros((NbaseD[0], NbaseD[1], NbaseN[2]), dtype=float)

        self.lambdas_2_31 = xp.zeros((NbaseN[0], NbaseD[1], NbaseD[2]), dtype=float)
        self.lambdas_2_32 = xp.zeros((NbaseD[0], NbaseN[1], NbaseD[2]), dtype=float)
        self.lambdas_2_33 = xp.zeros((NbaseD[0], NbaseD[1], NbaseN[2]), dtype=float)

        self.lambdas_3 = xp.zeros((NbaseD[0], NbaseD[1], NbaseD[2]), dtype=float)

        self.p_size = p_size
        self.p_shape = p_shape

        self.related = xp.zeros(3, dtype=int)
        for a in range(3):
            # self.related[a] = int(xp.floor(NbaseN[a]/2.0))
            self.related[a] = int(
                xp.floor((3 * int((self.p_size[a] * (self.p_shape[a] + 1)) * self.Nel[a] + 1) + 3 * self.p[a]) / 2.0),
            )
            if (2 * self.related[a] + 1) > NbaseN[a]:
                self.related[a] = int(xp.floor(NbaseN[a] / 2.0))

        self.kernel_0_loc = xp.zeros(
            (
                NbaseN[0],
                NbaseN[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )

        self.kernel_1_11_loc = xp.zeros(
            (
                NbaseD[0],
                NbaseN[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )
        self.kernel_1_12_loc = xp.zeros(
            (
                NbaseD[0],
                NbaseN[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )
        self.kernel_1_13_loc = xp.zeros(
            (
                NbaseD[0],
                NbaseN[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )

        self.kernel_1_22_loc = xp.zeros(
            (
                NbaseN[0],
                NbaseD[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )
        self.kernel_1_23_loc = xp.zeros(
            (
                NbaseN[0],
                NbaseD[1],
                NbaseN[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )

        self.kernel_1_33_loc = xp.zeros(
            (
                NbaseN[0],
                NbaseN[1],
                NbaseD[2],
                2 * self.related[0] + 1,
                2 * self.related[1] + 1,
                2 * self.related[2] + 1,
            ),
            dtype=float,
        )

        self.right_loc_1 = xp.zeros((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
        self.right_loc_2 = xp.zeros((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
        self.right_loc_3 = xp.zeros((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

        if self.mpi_rank == 0:
            self.kernel_0 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseN[1],
                    NbaseN[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )

            self.kernel_1_11 = xp.zeros(
                (
                    NbaseD[0],
                    NbaseN[1],
                    NbaseN[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )
            self.kernel_1_12 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseD[1],
                    NbaseN[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )
            self.kernel_1_13 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseN[1],
                    NbaseD[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )

            self.kernel_1_22 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseD[1],
                    NbaseN[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )
            self.kernel_1_23 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseN[1],
                    NbaseD[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )

            self.kernel_1_33 = xp.zeros(
                (
                    NbaseN[0],
                    NbaseN[1],
                    NbaseD[2],
                    2 * self.related[0] + 1,
                    2 * self.related[1] + 1,
                    2 * self.related[2] + 1,
                ),
                dtype=float,
            )

            self.right_1 = xp.zeros((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
            self.right_2 = xp.zeros((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
            self.right_3 = xp.zeros((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

        else:
            self.kernel_0 = None

            self.kernel_1_11 = None
            self.kernel_1_12 = None
            self.kernel_1_13 = None

            self.kernel_1_22 = None
            self.kernel_1_23 = None

            self.kernel_1_33 = None

            self.right_1 = None
            self.right_2 = None
            self.right_3 = None

        self.num_cell = xp.empty(3, dtype=int)
        for i in range(3):
            if self.p[i] == 1:
                self.num_cell[i] = 1
            else:
                self.num_cell[i] = 2

        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [xp.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [xp.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]

        self.pts = [0, 0, 0]
        self.wts = [0, 0, 0]
        for a in range(3):
            self.pts[a], self.wts[a] = bsp.quadrature_grid(
                [0, 1.0 / 2.0 / self.Nel[a]],
                self.pts_loc[a],
                self.wts_loc[a],
            )
        # print('check_pts', self.pts[0].shape, self.pts[1].shape, self.pts[2].shape)
        # print('check_pts', self.wts)
        # set interpolation and histopolation coefficients
        self.coeff_i = [0, 0, 0]
        self.coeff_h = [0, 0, 0]
        for a in range(3):
            if self.bc[a] == True:
                self.coeff_i[a] = xp.zeros(2 * self.p[a], dtype=float)
                self.coeff_h[a] = xp.zeros(2 * self.p[a], dtype=float)

                if self.p[a] == 1:
                    self.coeff_i[a][:] = xp.array([1.0, 0.0])
                    self.coeff_h[a][:] = xp.array([1.0, 1.0])

                elif self.p[a] == 2:
                    self.coeff_i[a][:] = 1 / 2 * xp.array([-1.0, 4.0, -1.0, 0.0])
                    self.coeff_h[a][:] = 1 / 2 * xp.array([-1.0, 3.0, 3.0, -1.0])

                elif self.p[a] == 3:
                    self.coeff_i[a][:] = 1 / 6 * xp.array([1.0, -8.0, 20.0, -8.0, 1.0, 0.0])
                    self.coeff_h[a][:] = 1 / 6 * xp.array([1.0, -7.0, 12.0, 12.0, -7.0, 1.0])

                elif self.p[a] == 4:
                    self.coeff_i[a][:] = 2 / 45 * xp.array([-1.0, 16.0, -295 / 4, 140.0, -295 / 4, 16.0, -1.0, 0.0])
                    self.coeff_h[a][:] = (
                        2 / 45 * xp.array([-1.0, 15.0, -231 / 4, 265 / 4, 265 / 4, -231 / 4, 15.0, -1.0])
                    )

                else:
                    print("degree > 4 not implemented!")

            else:
                self.coeff_i[a] = xp.zeros((2 * self.p[a] - 1, 2 * self.p[a] - 1), dtype=float)
                self.coeff_h[a] = xp.zeros((2 * self.p[a] - 1, 2 * self.p[a]), dtype=float)

                if self.p[a] == 1:
                    self.coeff_i[a][0, :] = xp.array([1.0])
                    self.coeff_h[a][0, :] = xp.array([1.0, 1.0])

                elif self.p[a] == 2:
                    self.coeff_i[a][0, :] = 1 / 2 * xp.array([2.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 2 * xp.array([-1.0, 4.0, -1.0])
                    self.coeff_i[a][2, :] = 1 / 2 * xp.array([0.0, 0.0, 2.0])

                    self.coeff_h[a][0, :] = 1 / 2 * xp.array([3.0, -1.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 2 * xp.array([-1.0, 3.0, 3.0, -1.0])
                    self.coeff_h[a][2, :] = 1 / 2 * xp.array([0.0, 0.0, -1.0, 3.0])

                elif self.p[a] == 3:
                    self.coeff_i[a][0, :] = 1 / 18 * xp.array([18.0, 0.0, 0.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 18 * xp.array([-5.0, 40.0, -24.0, 8.0, -1.0])
                    self.coeff_i[a][2, :] = 1 / 18 * xp.array([3.0, -24.0, 60.0, -24.0, 3.0])
                    self.coeff_i[a][3, :] = 1 / 18 * xp.array([-1.0, 8.0, -24.0, 40.0, -5.0])
                    self.coeff_i[a][4, :] = 1 / 18 * xp.array([0.0, 0.0, 0.0, 0.0, 18.0])

                    self.coeff_h[a][0, :] = 1 / 18 * xp.array([23.0, -17.0, 7.0, -1.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 18 * xp.array([-8.0, 56.0, -28.0, 4.0, 0.0, 0.0])
                    self.coeff_h[a][2, :] = 1 / 18 * xp.array([3.0, -21.0, 36.0, 36.0, -21.0, 3.0])
                    self.coeff_h[a][3, :] = 1 / 18 * xp.array([0.0, 0.0, 4.0, -28.0, 56.0, -8.0])
                    self.coeff_h[a][4, :] = 1 / 18 * xp.array([0.0, 0.0, -1.0, 7.0, -17.0, 23.0])

                elif self.p[a] == 4:
                    self.coeff_i[a][0, :] = 1 / 360 * xp.array([360.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 360 * xp.array([-59.0, 944.0, -1000.0, 720.0, -305.0, 64.0, -4.0])
                    self.coeff_i[a][2, :] = 1 / 360 * xp.array([23.0, -368.0, 1580.0, -1360.0, 605.0, -128.0, 8.0])
                    self.coeff_i[a][3, :] = 1 / 360 * xp.array([-16.0, 256.0, -1180.0, 2240.0, -1180.0, 256.0, -16.0])
                    self.coeff_i[a][4, :] = 1 / 360 * xp.array([8.0, -128.0, 605.0, -1360.0, 1580.0, -368.0, 23.0])
                    self.coeff_i[a][5, :] = 1 / 360 * xp.array([-4.0, 64.0, -305.0, 720.0, -1000.0, 944.0, -59.0])
                    self.coeff_i[a][6, :] = 1 / 360 * xp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 360.0])

                    self.coeff_h[a][0, :] = 1 / 360 * xp.array([419.0, -525.0, 475.0, -245.0, 60.0, -4.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 360 * xp.array([-82.0, 1230.0, -1350.0, 730.0, -180.0, 12.0, 0.0, 0.0])
                    self.coeff_h[a][2, :] = 1 / 360 * xp.array([39.0, -585.0, 2175.0, -1425.0, 360.0, -24.0, 0.0, 0.0])
                    self.coeff_h[a][3, :] = (
                        1 / 360 * xp.array([-16.0, 240.0, -924.0, 1060.0, 1060.0, -924.0, 240.0, -16.0])
                    )
                    self.coeff_h[a][4, :] = 1 / 360 * xp.array([0.0, 0.0, -24.0, 360.0, -1425.0, 2175.0, -585.0, 39.0])
                    self.coeff_h[a][5, :] = 1 / 360 * xp.array([0.0, 0.0, 12.0, -180.0, 730.0, -1350.0, 1230.0, -82.0])
                    self.coeff_h[a][6, :] = 1 / 360 * xp.array([0.0, 0.0, -4.0, 60.0, -245.0, 475.0, -525.0, 419.0])

                else:
                    print("degree > 4 not implemented!")

    def accumulate_0_form(self, mpi_comm):
        # blocks of global mass matrix

        mpi_comm.Reduce(self.kernel_0_loc, self.kernel_0, op=MPI.SUM, root=0)

    # ================ matrix in V0 ===========================
    def assemble_0_form(self, tensor_space_FEM, mpi_comm):
        """
        Assembles the 3D mass matrix [[NNN NNN]] * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

        Parameters
        ----------
        tensor_space_FEM : tensor_spline_space
            tensor product B-spline space for finite element spaces

        domain : domain
            domain object defining the geometry
        """
        # assembly of global mass matrix
        Ni = tensor_space_FEM.Nbase_0form
        Nj = tensor_space_FEM.Nbase_0form

        # conversion to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Ni[2] * col1 + Ni[2] * col2 + col3

        M = spa.csr_matrix(
            (self.kernel_0.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )

        M.eliminate_zeros()

        return M

    # ================ matrix in V1 ===========================
    def accumulate_1_form(self, mpi_comm):
        # blocks of global mass matrix
        mpi_comm.Reduce(self.kernel_1_11_loc, self.kernel_1_11, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.kernel_1_12_loc, self.kernel_1_12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.kernel_1_13_loc, self.kernel_1_13, op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.kernel_1_22_loc, self.kernel_1_22, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.kernel_1_23_loc, self.kernel_1_23, op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.kernel_1_33_loc, self.kernel_1_33, op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.right_loc_1, self.right_1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.right_loc_2, self.right_2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.right_loc_3, self.right_3, op=MPI.SUM, root=0)

    def assemble_1_form(self, tensor_space_FEM):
        """
        Assembles the 3D mass matrix [[DNN DNN, DNN NDN, DNN NND], [NDN DNN, NDN NDN, NDN NND], [NND DNN, NND NDN, NND NND]] * G^(-1) * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.

        Parameters
        ----------
        tensor_space_FEM : tensor_spline_space
            tensor product B-spline space for finite element spaces

        domain : domain
            domain object defining the geometry
        """
        # === 11 component =====
        a = 0
        b = 0
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M11 = spa.csr_matrix(
            (self.kernel_1_11.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M11.eliminate_zeros()

        # === 12 component =====
        a = 0
        b = 1
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M12 = spa.csr_matrix(
            (self.kernel_1_12.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M12.eliminate_zeros()

        # === 13 component =====
        a = 0
        b = 2
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M13 = spa.csr_matrix(
            (self.kernel_1_13.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M13.eliminate_zeros()

        # === 22 component =====
        a = 1
        b = 1
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M22 = spa.csr_matrix(
            (self.kernel_1_22.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M22.eliminate_zeros()

        # === 23 component =====
        a = 1
        b = 2
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M23 = spa.csr_matrix(
            (self.kernel_1_23.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M23.eliminate_zeros()

        # === 33 component =====
        a = 2
        b = 2
        Ni = tensor_space_FEM.Nbase_1form[a]
        Nj = tensor_space_FEM.Nbase_1form[b]

        # convert to sparse matrix
        indices = xp.indices(
            (Ni[0], Ni[1], Ni[2], 2 * self.related[0] + 1, 2 * self.related[1] + 1, 2 * self.related[2] + 1),
        )

        shift = [xp.arange(Ni) - offset for Ni, offset in zip(Ni, self.related)]

        row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
        col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
        col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

        col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

        M33 = spa.csr_matrix(
            (self.kernel_1_33.flatten(), (row, col.flatten())),
            shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
        )
        M33.eliminate_zeros()

        # final block matrix
        M = spa.bmat([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]], format="csr")
        # print('insider_check', self.kernel_1_33)
        return (M, xp.concatenate((self.right_1.flatten(), self.right_2.flatten(), self.right_3.flatten())))

    def heavy_test(self, test1, test2, test3, acc, particles_loc, Np, domain):
        ker_loc.kernel_1_heavy(
            self.pts[0][0],
            self.pts[1][0],
            self.pts[2][0],
            self.wts[0][0],
            self.wts[1][0],
            self.wts[2][0],
            test1,
            test2,
            test3,
            acc.oneform_temp1,
            acc.oneform_temp2,
            acc.oneform_temp3,
            Np,
            self.n_quad,
            self.p,
            self.Nel,
            self.p_shape,
            self.p_size,
            particles_loc,
            self.lambdas_1_11,
            self.lambdas_1_12,
            self.lambdas_1_13,
            self.lambdas_1_21,
            self.lambdas_1_22,
            self.lambdas_1_23,
            self.lambdas_1_31,
            self.lambdas_1_32,
            self.lambdas_1_33,
            self.num_cell,
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.NbaseN,
            self.NbaseD,
            particles_loc.shape[1],
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

    def potential_pi_0(self, particles_loc, Np, domain, mpi_comm):
        """
        Local projector on the discrete space V0.

        Parameters
        ----------

        Returns
        -------
        kernel_0 matrix
        """
        if self.bc[0] and self.bc[1] and self.bc[2]:
            ker_loc.potential_kernel_0_form(
                Np,
                self.p,
                self.Nel,
                self.p_shape,
                self.p_size,
                particles_loc,
                self.lambdas_0,
                self.kernel_0_loc,
                self.num_cell,
                self.coeff_i[0],
                self.coeff_i[1],
                self.coeff_i[2],
                self.NbaseN,
                self.related,
                particles_loc.shape[1],
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
            print("non-periodic case not implemented!!!")

        mpi_comm.Reduce(self.lambdas_0, self.potential_lambdas_0, op=MPI.SUM, root=0)
        # print('check_lambdas', self.lambdas_0)

    def S_pi_0(self, particles_loc, Np, domain):
        """
        Local projector on the discrete space V0.

        Parameters
        ----------

        Returns
        -------
        kernel_0 matrix
        """
        self.kernel_0[:, :, :, :, :, :] = 0.0
        if self.bc[0] and self.bc[1] and self.bc[2]:
            ker_loc.kernel_0_form(
                Np,
                self.p,
                self.Nel,
                self.p_shape,
                self.p_size,
                particles_loc,
                self.lambdas_0,
                self.kernel_0_loc,
                self.num_cell,
                self.coeff_i[0],
                self.coeff_i[1],
                self.coeff_i[2],
                self.NbaseN,
                self.related,
                particles_loc.shape[1],
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
            print("non-periodic case not implemented!!!")

        # print('check_lambdas', self.lambdas_0)

    def S_pi_1(self, particles_loc, Np, domain):
        """
            Local projector on the discrete space V1.

            Parameters
            ----------
            pts : quadrature point in the first half cell
            wts : quadrature weight in the first half cell
            quad: shape of pts or wts

            Returns
        -------
            lambdas : array_like
                the coefficients in V0 corresponding to the projected function
        """
        self.kernel_1_11_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_12_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_13_loc[:, :, :, :, :, :] = 0.0

        self.kernel_1_22_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_23_loc[:, :, :, :, :, :] = 0.0

        self.kernel_1_33_loc[:, :, :, :, :, :] = 0.0

        self.right_loc_1[:, :, :] = 0.0
        self.right_loc_2[:, :, :] = 0.0
        self.right_loc_3[:, :, :] = 0.0

        if self.bc[0] and self.bc[1] and self.bc[2]:
            ker_loc.kernel_1_form(
                self.right_loc_1,
                self.right_loc_2,
                self.right_loc_3,
                self.pts[0][0],
                self.pts[1][0],
                self.pts[2][0],
                self.wts[0][0],
                self.wts[1][0],
                self.wts[2][0],
                Np,
                self.n_quad,
                self.p,
                self.Nel,
                self.p_shape,
                self.p_size,
                particles_loc,
                self.lambdas_1_11,
                self.lambdas_1_12,
                self.lambdas_1_13,
                self.lambdas_1_21,
                self.lambdas_1_22,
                self.lambdas_1_23,
                self.lambdas_1_31,
                self.lambdas_1_32,
                self.lambdas_1_33,
                self.kernel_1_11_loc,
                self.kernel_1_12_loc,
                self.kernel_1_13_loc,
                self.kernel_1_22_loc,
                self.kernel_1_23_loc,
                self.kernel_1_33_loc,
                self.num_cell,
                self.coeff_i[0],
                self.coeff_i[1],
                self.coeff_i[2],
                self.coeff_h[0],
                self.coeff_h[1],
                self.coeff_h[2],
                self.NbaseN,
                self.NbaseD,
                self.related,
                particles_loc.shape[1],
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
            print("non-periodic case not implemented!!!")

    def S_pi_01(self, particles_loc, Np, domain):
        """
            Local projector on the discrete space V1.

            Parameters
            ----------
            pts : quadrature point in the first half cell
            wts : quadrature weight in the first half cell
            quad: shape of pts or wts

            Returns
        -------
            lambdas : array_like
                the coefficients in V0 corresponding to the projected function
        """
        self.kernel_1_11_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_12_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_13_loc[:, :, :, :, :, :] = 0.0

        self.kernel_1_22_loc[:, :, :, :, :, :] = 0.0
        self.kernel_1_23_loc[:, :, :, :, :, :] = 0.0

        self.kernel_1_33_loc[:, :, :, :, :, :] = 0.0

        self.right_loc_1[:, :, :] = 0.0
        self.right_loc_2[:, :, :] = 0.0
        self.right_loc_3[:, :, :] = 0.0

        if self.bc[0] and self.bc[1] and self.bc[2]:
            ker_loc.kernel_01_form(
                self.right_loc_1,
                self.right_loc_2,
                self.right_loc_3,
                Np,
                self.n_quad,
                self.p,
                self.Nel,
                self.p_shape,
                self.p_size,
                particles_loc,
                self.lambdas_1_11,
                self.lambdas_1_12,
                self.lambdas_1_13,
                self.lambdas_1_21,
                self.lambdas_1_22,
                self.lambdas_1_23,
                self.lambdas_1_31,
                self.lambdas_1_32,
                self.lambdas_1_33,
                self.kernel_1_11_loc,
                self.kernel_1_12_loc,
                self.kernel_1_13_loc,
                self.kernel_1_22_loc,
                self.kernel_1_23_loc,
                self.kernel_1_33_loc,
                self.num_cell,
                self.coeff_i[0],
                self.coeff_i[1],
                self.coeff_i[2],
                self.NbaseN,
                self.NbaseD,
                self.related,
                particles_loc.shape[1],
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
            print("non-periodic case not implemented!!!")

    def vv_S1(self, particles_loc, Np, domain, index_label, accvv, dt, mpi_comm):
        if self.bc[0] and self.bc[1] and self.bc[2]:
            if index_label == 1:
                ker_loc.vv_1_form(
                    self.wts[0][0],
                    self.wts[1][0],
                    self.wts[2][0],
                    self.pts[0][0],
                    self.pts[1][0],
                    self.pts[2][0],
                    0.0,
                    self.right_loc_1,
                    self.right_loc_2,
                    self.right_loc_3,
                    Np,
                    self.n_quad,
                    self.p,
                    self.Nel,
                    self.p_shape,
                    self.p_size,
                    particles_loc,
                    accvv.mid_particles,
                    self.lambdas_1_11,
                    self.lambdas_1_12,
                    self.lambdas_1_13,
                    self.lambdas_1_21,
                    self.lambdas_1_22,
                    self.lambdas_1_23,
                    self.lambdas_1_31,
                    self.lambdas_1_32,
                    self.lambdas_1_33,
                    self.num_cell,
                    self.coeff_i[0],
                    self.coeff_i[1],
                    self.coeff_i[2],
                    self.coeff_h[0],
                    self.coeff_h[1],
                    self.coeff_h[2],
                    self.NbaseN,
                    self.NbaseD,
                    self.related,
                    particles_loc.shape[1],
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
            elif index_label == 2:
                ker_loc.vv_1_form(
                    self.wts[0][0],
                    self.wts[1][0],
                    self.wts[2][0],
                    self.pts[0][0],
                    self.pts[1][0],
                    self.pts[2][0],
                    0.5 * dt,
                    self.right_loc_1,
                    self.right_loc_2,
                    self.right_loc_3,
                    Np,
                    self.n_quad,
                    self.p,
                    self.Nel,
                    self.p_shape,
                    self.p_size,
                    particles_loc,
                    accvv.stage1_out_loc,
                    self.lambdas_1_11,
                    self.lambdas_1_12,
                    self.lambdas_1_13,
                    self.lambdas_1_21,
                    self.lambdas_1_22,
                    self.lambdas_1_23,
                    self.lambdas_1_31,
                    self.lambdas_1_32,
                    self.lambdas_1_33,
                    self.num_cell,
                    self.coeff_i[0],
                    self.coeff_i[1],
                    self.coeff_i[2],
                    self.coeff_h[0],
                    self.coeff_h[1],
                    self.coeff_h[2],
                    self.NbaseN,
                    self.NbaseD,
                    self.related,
                    particles_loc.shape[1],
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
            elif index_label == 3:
                ker_loc.vv_1_form(
                    self.wts[0][0],
                    self.wts[1][0],
                    self.wts[2][0],
                    self.pts[0][0],
                    self.pts[1][0],
                    self.pts[2][0],
                    0.5 * dt,
                    self.right_loc_1,
                    self.right_loc_2,
                    self.right_loc_3,
                    Np,
                    self.n_quad,
                    self.p,
                    self.Nel,
                    self.p_shape,
                    self.p_size,
                    particles_loc,
                    accvv.stage2_out_loc,
                    self.lambdas_1_11,
                    self.lambdas_1_12,
                    self.lambdas_1_13,
                    self.lambdas_1_21,
                    self.lambdas_1_22,
                    self.lambdas_1_23,
                    self.lambdas_1_31,
                    self.lambdas_1_32,
                    self.lambdas_1_33,
                    self.num_cell,
                    self.coeff_i[0],
                    self.coeff_i[1],
                    self.coeff_i[2],
                    self.coeff_h[0],
                    self.coeff_h[1],
                    self.coeff_h[2],
                    self.NbaseN,
                    self.NbaseD,
                    self.related,
                    particles_loc.shape[1],
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
            elif index_label == 4:
                ker_loc.vv_1_form(
                    self.wts[0][0],
                    self.wts[1][0],
                    self.wts[2][0],
                    self.pts[0][0],
                    self.pts[1][0],
                    self.pts[2][0],
                    dt,
                    self.right_loc_1,
                    self.right_loc_2,
                    self.right_loc_3,
                    Np,
                    self.n_quad,
                    self.p,
                    self.Nel,
                    self.p_shape,
                    self.p_size,
                    particles_loc,
                    accvv.stage3_out_loc,
                    self.lambdas_1_11,
                    self.lambdas_1_12,
                    self.lambdas_1_13,
                    self.lambdas_1_21,
                    self.lambdas_1_22,
                    self.lambdas_1_23,
                    self.lambdas_1_31,
                    self.lambdas_1_32,
                    self.lambdas_1_33,
                    self.num_cell,
                    self.coeff_i[0],
                    self.coeff_i[1],
                    self.coeff_i[2],
                    self.coeff_h[0],
                    self.coeff_h[1],
                    self.coeff_h[2],
                    self.NbaseN,
                    self.NbaseD,
                    self.related,
                    particles_loc.shape[1],
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

        mpi_comm.Reduce(self.right_loc_1, accvv.vec1, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.right_loc_2, accvv.vec2, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.right_loc_3, accvv.vec3, op=MPI.SUM, root=0)
