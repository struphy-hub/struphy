# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

import time

import numpy as np
import scipy.sparse as spa
from mpi4py import MPI

import struphy.pic.tests.test_pic_legacy_files.accumulation_kernels_3d as pic_ker_3d

# import struphy.pic.tests.test_pic_legacy_files.accumulation_kernels_2d as pic_ker_2d

# from struphy.pic.tests.test_pic_legacy_files.control_variate import TermsControlVariate


class Accumulator:
    """
    Class for computing charge and current densities from particles.

    Parameters
    ---------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space

    domain : domain object
        domain object from hylife.geometry.domain_3d defining the mapping

    basis_u : int
        bulk velocity representation (0 : vector-field, 1 : 1-form , 2 : 2-form)

    mpi_comm : MPI.COMM_WORLD
        MPI communicator

    control : boolean
        whether a full-f (False) of delta-f approach is used

    cv_ep : control variate object
        the distribution function that serves as a control variate (only necessary in case of use_control = True)
    """

    # ===============================================================
    def __init__(self, tensor_space_FEM, domain, basis_u, mpi_comm, use_control, cv_ep=None):
        self.space = tensor_space_FEM
        self.domain = domain
        self.basis_u = basis_u
        self.mpi_rank = mpi_comm.Get_rank()
        self.use_control = use_control

        # intialize delta-f correction terms
        if self.use_control and self.mpi_rank == 0:
            self.cont = TermsControlVariate(self.space, self.domain, self.basis_u, cv_ep)

        # reserve memory for implicit particle-coupling sub-steps
        self.blocks_loc = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.blocks_glo = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.vecs_loc = [0, 0, 0]
        self.vecs_glo = [0, 0, 0]

        for a in range(3):
            if self.basis_u == 0:
                Ni = self.space.Nbase_0form
            else:
                Ni = getattr(self.space, "Nbase_" + str(self.basis_u) + "form")[a]

            self.vecs_loc[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
            self.vecs_glo[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)

            for b in range(3):
                if self.space.dim == 2:
                    self.blocks_loc[a][b] = np.empty(
                        (Ni[0], Ni[1], Ni[2], 2 * self.space.p[0] + 1, 2 * self.space.p[1] + 1, self.space.NbaseN[2]),
                        dtype=float,
                    )
                    self.blocks_glo[a][b] = np.empty(
                        (Ni[0], Ni[1], Ni[2], 2 * self.space.p[0] + 1, 2 * self.space.p[1] + 1, self.space.NbaseN[2]),
                        dtype=float,
                    )

                else:
                    self.blocks_loc[a][b] = np.empty(
                        (
                            Ni[0],
                            Ni[1],
                            Ni[2],
                            2 * self.space.p[0] + 1,
                            2 * self.space.p[1] + 1,
                            2 * self.space.p[2] + 1,
                        ),
                        dtype=float,
                    )
                    self.blocks_glo[a][b] = np.empty(
                        (
                            Ni[0],
                            Ni[1],
                            Ni[2],
                            2 * self.space.p[0] + 1,
                            2 * self.space.p[1] + 1,
                            2 * self.space.p[2] + 1,
                        ),
                        dtype=float,
                    )

    # ===============================================================
    def to_sparse_step1(self):
        """Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering

        Returns
        -------
        M : sparse matrix in csr-format
            anti-symmetric, sparse block matrix [[0, M12, M13], [-M12.T, 0, M23], [-M13.T, -M23.T, 0]]
        """

        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for a in range(2):
            for b in range(a + 1, 3):
                if self.basis_u == 0:
                    Ni = self.space.Nbase_0form
                    Nj = self.space.Nbase_0form

                elif self.basis_u == 1:
                    Ni = self.space.Nbase_1form[a]
                    Nj = self.space.Nbase_1form[b]

                elif self.basis_u == 2:
                    Ni = self.space.Nbase_2form[a]
                    Nj = self.space.Nbase_2form[b]

                indices = np.indices(self.blocks_glo[a][b].shape)

                row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

                shift = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]

                if self.space.dim == 2:
                    shift += [np.zeros(self.space.NbaseN[2], dtype=int)]
                else:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]

                col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
                col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
                col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

                col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

                M[a][b] = spa.csr_matrix(
                    (self.blocks_glo[a][b].flatten(), (row, col.flatten())),
                    shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
                )
                M[a][b].eliminate_zeros()

        # final block matrix
        M = spa.bmat(
            [[None, M[0][1], M[0][2]], [-M[0][1].T, None, M[1][2]], [-M[0][2].T, -M[1][2].T, None]], format="csr"
        )

        # apply extraction operator
        if self.basis_u == 0:
            M = self.space.Ev_0.dot(M.dot(self.space.Ev_0.T)).tocsr()

        elif self.basis_u == 1:
            M = self.space.E1_0.dot(M.dot(self.space.E1_0.T)).tocsr()

        elif self.basis_u == 2:
            M = self.space.E2_0.dot(M.dot(self.space.E2_0.T)).tocsr()

        return M

    # ===============================================================
    def to_sparse_step3(self):
        """Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering

        Returns
        -------
        M : sparse matrix in csr-format
            symmetric, sparse block matrix [[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]]
        """

        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for a in range(3):
            for b in range(a, 3):
                if self.basis_u == 0:
                    Ni = self.space.Nbase_0form
                    Nj = self.space.Nbase_0form

                elif self.basis_u == 1:
                    Ni = self.space.Nbase_1form[a]
                    Nj = self.space.Nbase_1form[b]

                elif self.basis_u == 2:
                    Ni = self.space.Nbase_2form[a]
                    Nj = self.space.Nbase_2form[b]

                indices = np.indices(self.blocks_glo[a][b].shape)

                row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

                shift = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]

                if self.space.dim == 2:
                    shift += [np.zeros(self.space.NbaseN[2], dtype=int)]
                else:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]

                col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
                col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
                col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

                col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

                M[a][b] = spa.csr_matrix(
                    (self.blocks_glo[a][b].flatten(), (row, col.flatten())),
                    shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2]),
                )
                M[a][b].eliminate_zeros()

        # final block matrix
        M = spa.bmat(
            [[M[0][0], M[0][1], M[0][2]], [M[0][1].T, M[1][1], M[1][2]], [M[0][2].T, M[1][2].T, M[2][2]]], format="csr"
        )

        # apply extraction operator
        if self.basis_u == 0:
            M = self.space.Ev_0.dot(M.dot(self.space.Ev_0.T)).tocsr()

        elif self.basis_u == 1:
            M = self.space.E1_0.dot(M.dot(self.space.E1_0.T)).tocsr()

        elif self.basis_u == 2:
            M = self.space.E2_0.dot(M.dot(self.space.E2_0.T)).tocsr()

        return M

    # ===============================================================
    def accumulate_step1(self, particles_loc, Np, b2_eq, b2, mpi_comm):
        """TODO"""

        b2_1, b2_2, b2_3 = self.space.extract_2(b2)

        if self.space.dim == 2:
            pic_ker_2d.kernel_step1(
                particles_loc,
                self.space.T[0],
                self.space.T[1],
                self.space.p,
                self.space.Nel,
                self.space.NbaseN,
                self.space.NbaseD,
                particles_loc.shape[0],
                b2_eq[0],
                b2_eq[1],
                b2_eq[2],
                b2_1,
                b2_2,
                b2_3,
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
                self.blocks_loc[0][1],
                self.blocks_loc[0][2],
                self.blocks_loc[1][2],
                self.basis_u,
                self.space.n_tor,
            )

        else:
            pic_ker_3d.kernel_step1(
                particles_loc,
                self.space.T[0],
                self.space.T[1],
                self.space.T[2],
                self.space.p,
                self.space.Nel,
                self.space.NbaseN,
                self.space.NbaseD,
                particles_loc.shape[0],
                b2_1,
                b2_2,
                b2_3,
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
                self.blocks_loc[0][1],
                self.blocks_loc[0][2],
                self.blocks_loc[1][2],
                self.basis_u,
            )

        mpi_comm.Allreduce(self.blocks_loc[0][1], self.blocks_glo[0][1], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[0][2], self.blocks_glo[0][2], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[1][2], self.blocks_glo[1][2], op=MPI.SUM)

        self.blocks_glo[0][1] /= Np
        self.blocks_glo[0][2] /= Np
        self.blocks_glo[1][2] /= Np

    # ===============================================================
    def accumulate_step3(self, particles_loc, Np, b2_eq, b2, mpi_comm):
        """TODO"""

        b2_1, b2_2, b2_3 = self.space.extract_2(b2)

        if self.space.dim == 2:
            pic_ker_2d.kernel_step3(
                particles_loc,
                self.space.T[0],
                self.space.T[1],
                self.space.p,
                self.space.Nel,
                self.space.NbaseN,
                self.space.NbaseD,
                particles_loc.shape[0],
                b2_eq[0],
                b2_eq[1],
                b2_eq[2],
                b2_1,
                b2_2,
                b2_3,
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
                self.blocks_loc[0][0],
                self.blocks_loc[0][1],
                self.blocks_loc[0][2],
                self.blocks_loc[1][1],
                self.blocks_loc[1][2],
                self.blocks_loc[2][2],
                self.vecs_loc[0],
                self.vecs_loc[1],
                self.vecs_loc[2],
                self.basis_u,
                self.space.n_tor,
            )

        else:
            pic_ker_3d.kernel_step3(
                particles_loc,
                self.space.T[0],
                self.space.T[1],
                self.space.T[2],
                self.space.p,
                self.space.Nel,
                self.space.NbaseN,
                self.space.NbaseD,
                particles_loc.shape[0],
                b2_1,
                b2_2,
                b2_3,
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
                self.blocks_loc[0][0],
                self.blocks_loc[0][1],
                self.blocks_loc[0][2],
                self.blocks_loc[1][1],
                self.blocks_loc[1][2],
                self.blocks_loc[2][2],
                self.vecs_loc[0],
                self.vecs_loc[1],
                self.vecs_loc[2],
                self.basis_u,
            )

        mpi_comm.Allreduce(self.blocks_loc[0][0], self.blocks_glo[0][0], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[0][1], self.blocks_glo[0][1], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[0][2], self.blocks_glo[0][2], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[1][1], self.blocks_glo[1][1], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[1][2], self.blocks_glo[1][2], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[2][2], self.blocks_glo[2][2], op=MPI.SUM)

        mpi_comm.Allreduce(self.vecs_loc[0], self.vecs_glo[0], op=MPI.SUM)
        mpi_comm.Allreduce(self.vecs_loc[1], self.vecs_glo[1], op=MPI.SUM)
        mpi_comm.Allreduce(self.vecs_loc[2], self.vecs_glo[2], op=MPI.SUM)

        self.blocks_glo[0][0] /= Np
        self.blocks_glo[0][1] /= Np
        self.blocks_glo[0][2] /= Np
        self.blocks_glo[1][1] /= Np
        self.blocks_glo[1][2] /= Np
        self.blocks_glo[2][2] /= Np

        self.vecs_glo[0] /= Np
        self.vecs_glo[1] /= Np
        self.vecs_glo[2] /= Np

    # ===============================================================
    def accumulate_step_ph_full(self, particles_loc, Np, mpi_comm):
        """TODO"""

        if self.space.dim == 2:
            raise NotImplementedError("2d not implemented")

        else:
            pic_ker_3d.kernel_step_ph_full(
                particles_loc,
                self.space.T[0],
                self.space.T[1],
                self.space.T[2],
                self.space.p,
                self.space.Nel,
                self.space.NbaseN,
                self.space.NbaseD,
                particles_loc.shape[0],
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
                self.blocks_loc[0][0],
                self.blocks_loc[0][1],
                self.blocks_loc[0][2],
                self.blocks_loc[1][1],
                self.blocks_loc[1][2],
                self.blocks_loc[2][2],
                self.vecs_loc[0],
                self.vecs_loc[1],
                self.vecs_loc[2],
                self.basis_u,
            )

        mpi_comm.Allreduce(self.blocks_loc[0][0], self.blocks_glo[0][0], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[0][1], self.blocks_glo[0][1], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[0][2], self.blocks_glo[0][2], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[1][1], self.blocks_glo[1][1], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[1][2], self.blocks_glo[1][2], op=MPI.SUM)
        mpi_comm.Allreduce(self.blocks_loc[2][2], self.blocks_glo[2][2], op=MPI.SUM)

        mpi_comm.Allreduce(self.vecs_loc[0], self.vecs_glo[0], op=MPI.SUM)
        mpi_comm.Allreduce(self.vecs_loc[1], self.vecs_glo[1], op=MPI.SUM)
        mpi_comm.Allreduce(self.vecs_loc[2], self.vecs_glo[2], op=MPI.SUM)

        self.blocks_glo[0][0] /= Np
        self.blocks_glo[0][1] /= Np
        self.blocks_glo[0][2] /= Np
        self.blocks_glo[1][1] /= Np
        self.blocks_glo[1][2] /= Np
        self.blocks_glo[2][2] /= Np

        self.vecs_glo[0] /= Np
        self.vecs_glo[1] /= Np
        self.vecs_glo[2] /= Np

    # ===============================================================
    def assemble_step1(self, b2_eq, b2):
        """TODO"""

        # delta-f correction
        if self.use_control:
            b2_1, b2_2, b2_3 = self.space.extract_2(b2)

            if self.space.dim == 2:
                self.cont.correct_step1(b2_eq[0], b2_eq[1], b2_eq[2])
            else:
                self.cont.correct_step1(b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3)

            self.blocks_glo[0][1] += self.cont.M12
            self.blocks_glo[0][2] += self.cont.M13
            self.blocks_glo[1][2] += self.cont.M23

        # build global sparse matrix
        return self.to_sparse_step1()

    # ===============================================================
    def assemble_step3(self, b2_eq, b2):
        """TODO"""

        # delta-f correction
        if self.use_control:
            b2_1, b2_2, b2_3 = self.space.extract_2(b2)

            if self.space.dim == 2:
                self.cont.correct_step3(b2_1, b2_2, b2_3)
            else:
                self.cont.correct_step3(b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3)

            self.vecs_glo[0] += self.cont.F1
            self.vecs_glo[1] += self.cont.F2
            self.vecs_glo[2] += self.cont.F3

        # build global sparse matrix and global vector
        if self.basis_u == 0:
            return self.to_sparse_step3(), self.space.Ev_0.dot(
                np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))
            )

        elif self.basis_u == 1:
            return self.to_sparse_step3(), self.space.E1_0.dot(
                np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))
            )

        elif self.basis_u == 2:
            return self.to_sparse_step3(), self.space.E2_0.dot(
                np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))
            )
