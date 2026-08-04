import logging
from typing import Callable

import cunumpy as xp
from feectools.api.settings import PSYDAC_BACKEND_GPYCCEL
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilMatrix, StencilVector

from struphy.feec import mass_kernels
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.geometry.base import Domain
from struphy.utils.pyccel import Pyccelkernel

from feectools.linalg.block import BlockLinearOperator, BlockVector


class BoundaryIntegralOperators:
    """
    Collection of boundary integral operators and boundary mass operators
    for the H1, H(curl) and H(div) spaces.

    Analogous to WeightedMassOperators but for surface integrals.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        Mass operators object, contains geometry and derham.
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool] | None = None,
    ):
        
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._domain = mass_ops.domain

        # shared surface setup for all spaces
        # active faces based on bcs
        if active_faces is not None:
            # use provided active faces directly
            self._active_faces = active_faces
        else:
            # default: integrate on free faces based on bcs
            self._active_faces = []
            for face_idx in range(6):
                normal_dir = face_idx % 3
                bc = self._derham.bcs[normal_dir]
                if bc is None:
                    self._active_faces.append(False)
                elif face_idx < 3:
                    self._active_faces.append(bc[0] == "free")
                else:
                    self._active_faces.append(bc[1] == "free")

        # TODO: shared surface quad grids, geom weights, spans, wts, bases
        # for each space (H1, Hcurl, Hdiv) — these differ because the
        # quadrature grids are different for each space

    ##################################################
    # H1 boundary operators (scalar, normal trace)   #
    ##################################################

    @property
    def S0(self) -> "BoundaryMassOperatorH1":
        """
        Boundary mass matrix for H1:

            S0_{ijk,lmn} = int_{partial Omega} Lambda^0_{ijk} Lambda^0_{lmn} sqrt(g) |DF^-T n| dS
        """
        if not hasattr(self, "_S0"):
            self._S0 = BoundaryMassOperatorH1(self._mass_ops, self._active_faces)
        return self._S0

    ##################################################
    # H(curl) boundary operators (tangential trace)  #
    ##################################################

    @property
    def S1(self) -> "BoundaryMassOperatorHCurl":
        """
        Boundary mass matrix for H(curl):

            S1_{(mu,ijk),(nu,lmn)} = int_{partial Omega} (Lambda^1_{mu,ijk} x n) . Lambda^1_{nu,lmn} sqrt(g) |DF^-T n| dS

        Encodes the bilinear form for the tangential trace u x n against H(curl) test functions.
        """
        if not hasattr(self, "_S1"):
            self._S1 = BoundaryMassOperatorHCurl(self._mass_ops, self._active_faces)
        return self._S1


class BoundaryMassOperatorH1(LinOpWithTransp):
    """
    Assembles the boundary mass matrix for H1 basis functions.

    Computes the six surface integrals

        S_i'_{ijk,lmn} = int_{partial Omega_i'} Lambda^0_{ijk} Lambda^0_{lmn} sqrt(g) |DF^-T n_hat_i| dS

    and adds them together into a single StencilMatrix S such that

        I = psi^T S alpha

    for any discrete test function psi_h and spline function alpha_h in V^0_h.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        Mass operators object, contains geometry and derham.
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool]
    ):
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._domain_obj = mass_ops.domain

        # H1 space info
        self._space = self._derham.fem_spaces["0"]
        self._space_key = "0"

        # 3D quadrature grid info for H1 space
        self._quad_grid_pts = self._derham.spline_attributes[self._space_key].quad_grid_pts
        self._spans_l = self._derham.spline_attributes[self._space_key].quad_grid_spans
        self._wts_l = self._derham.spline_attributes[self._space_key].quad_grid_wts
        self._bases_l = self._derham.spline_attributes[self._space_key].quad_grid_bases
        self._tensor_fem_spaces = self._derham.spline_attributes[self._space_key].tensor_spaces

        # boundary and extraction operators
        self._V_extraction_op = self._derham.extraction_ops[self._space_key]
        self._W_extraction_op = self._derham.extraction_ops[self._space_key]
        self._V_boundary_op = self._derham.boundary_ops[self._space_key]
        self._W_boundary_op = self._derham.boundary_ops[self._space_key]

        self._V_extraction_op_T = self._V_extraction_op.T
        self._W_extraction_op_T = self._W_extraction_op.T
        self._V_boundary_op_T = self._V_boundary_op.T
        self._W_boundary_op_T = self._W_boundary_op.T

        # initialize StencilMatrix
        fem_space = self._tensor_fem_spaces[0]
        self._mat = StencilMatrix(
            fem_space.coeff_space,
            fem_space.coeff_space,
            backend=PSYDAC_BACKEND_GPYCCEL,
            precompiled=True,
        )

        # build composite operator B * E * M * E^T * B^T
        self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op_T
        self._M0 = self._W_boundary_op @ self._M @ self._V_boundary_op_T

        # set domain and codomain
        self._domain = self._M0.domain
        self._codomain = self._M0.codomain
        self._dtype = fem_space.coeff_space.dtype

        # allocate temporaries
        self._temp_WB = self._W_boundary_op.domain.zeros()
        self._temp_WE = self._W_extraction_op.domain.zeros()
        self._temp_VB = self._V_boundary_op.domain.zeros()
        self._temp_VE = self._V_extraction_op.domain.zeros()
        self._temp_mat = self._mat.domain.zeros()

        # determine which faces to integrate over based on bcs
        self._active_faces = active_faces

        # for each active face, extract surface quadrature grid and geometric weights
        self._surface_quad_grid_meshes = []
        self._surface_geom_weights = []
        self._surface_spans = []
        self._surface_wts = []
        self._surface_bases = []

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                self._surface_quad_grid_meshes.append(None)
                self._surface_geom_weights.append(None)
                self._surface_spans.append(None)
                self._surface_wts.append(None)
                self._surface_bases.append(None)
                continue

            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]
            fixed_val = 0.0 if face_idx < 3 else 1.0

            surf_pts = [self._quad_grid_pts[0][d].flatten() for d in surf_dirs]
            self._surface_quad_grid_meshes.append(xp.meshgrid(*surf_pts, indexing="ij"))

            surf_pts_1d = [self._quad_grid_pts[0][d].flatten() for d in surf_dirs]
            e_1d = [None, None, None]
            e_1d[surf_dirs[0]] = surf_pts_1d[0]
            e_1d[surf_dirs[1]] = surf_pts_1d[1]
            e_1d[normal_dir] = xp.array([fixed_val])

            sqrt_g = xp.abs(self._domain_obj.jacobian_det(*e_1d))
            DFinv = self._domain_obj.jacobian_inv(*e_1d, change_out_order=True)
            DFinv_n = DFinv[..., normal_dir, :]
            norm_DFinv_n = xp.sqrt(xp.sum(DFinv_n**2, axis=-1))

            surface_geom_weights = xp.squeeze(sqrt_g * norm_DFinv_n)
            self._surface_geom_weights.append(surface_geom_weights)

            self._surface_spans.append([self._spans_l[0][d] for d in surf_dirs])
            self._surface_wts.append([self._wts_l[0][d] for d in surf_dirs])
            self._surface_bases.append([self._bases_l[0][d] for d in surf_dirs])

        # load assembly kernel
        self._assembly_kernel = Pyccelkernel(mass_kernels.surface_kernel_3d_mat_h1)

        self.assemble()

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def _assemble_face(
        self,
        face_idx: int,
        mat: StencilMatrix,
    ):
        """
        Assembles the contribution of a single face to the boundary mass matrix.

        Parameters
        ----------
        face_idx : int
            Index of the face (0 to 5).

        mat : StencilMatrix
            Output matrix to accumulate into.
        """
        normal_dir = face_idx % 3
        fem_space = self._tensor_fem_spaces[0]
        starts = [int(start) for start in fem_space.coeff_space.starts]
        ends = [int(end) for end in fem_space.coeff_space.ends]
        pads = fem_space.coeff_space.pads

        boundary_index = starts[normal_dir] if face_idx < 3 else ends[normal_dir]

        fem_space = self._tensor_fem_spaces[0]
        starts = [int(start) for start in fem_space.coeff_space.starts]
        pads = fem_space.coeff_space.pads

        self._assembly_kernel(
            *self._surface_spans[face_idx],
            *fem_space.degree,
            *fem_space.degree,
            *starts,
            *pads,
            *self._surface_wts[face_idx],
            *self._surface_bases[face_idx],
            *self._surface_bases[face_idx],
            boundary_index,
            normal_dir,
            self._surface_geom_weights[face_idx],
            mat._data,
        )

    def assemble(
        self,
        clear: bool = True,
    ):
        """
        Assembles the boundary mass matrix.

        Parameters
        ----------
        clear : bool, optional
            Whether to zero the matrix before assembly.
        """
        if clear:
            self._mat._data[:] = 0.0

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                continue
            self._assemble_face(face_idx, self._mat)

        self._mat.exchange_assembly_data()
        self._mat.update_ghost_regions()

    def dot(self, v, out=None, apply_bc=True):
        """
        Applies the boundary mass matrix to a vector.

        Parameters
        ----------
        v : StencilVector
            Input vector (spline coefficients of alpha_h).

        out : StencilVector, optional
            Output vector. If None, a new zero vector is created.

        apply_bc : bool
            Whether to apply boundary operators.

        Returns
        -------
        out : StencilVector
            The result S * v.
        """
        if out is None:
            out = self.codomain.zeros()

        if apply_bc:
            self._V_boundary_op_T.dot(v, out=self._temp_VB)
            self._V_extraction_op_T.dot(self._temp_VB, out=self._temp_mat)
            self._mat.dot(self._temp_mat, out=self._temp_WE)
            self._W_extraction_op.dot(self._temp_WE, out=self._temp_WB)
            self._W_boundary_op.dot(self._temp_WB, out=out)
        else:
            self._V_extraction_op_T.dot(v, out=self._temp_mat)
            self._mat.dot(self._temp_mat, out=self._temp_WE)
            self._W_extraction_op.dot(self._temp_WE, out=out)

        return out

    def transpose(self, conjugate=False):
        """
        Returns self since the boundary mass matrix is symmetric.
        """
        return self

    def __call__(
        self,
        clear: bool = True,
    ):
        """
        Assembles the boundary mass matrix.

        Parameters
        ----------
        clear : bool, optional
            Whether to zero the matrix before assembly.
        """
        self.assemble(clear=clear)
        return self


    def toarray(self):
        return self._M0.toarray()


    def tosparse(self):
        return self._M0.tosparse()


class BoundaryMassOperatorHCurl(LinOpWithTransp):
    """
    Assembles the boundary mass matrix for H(curl) basis functions.

    Computes the surface integrals

        W^{mu,nu}_{ijk,lmn} = int_{partial Omega} hat_Lambda^1_{mu,ijk} hat_R_n^{mu,nu} hat_Lambda^1_{nu,lmn} dS

    where hat_R_n is the pullback of [n]_x to logical coordinates.

    The result is a 3x3 BlockLinearOperator where diagonal blocks are zero
    (skew-symmetry of [n]_x) and off-diagonal blocks are assembled via
    surface_kernel_3d_mat_h1.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        Mass operators object, contains geometry and derham.
    active_faces : list[bool]
        Which of the six faces to integrate over.
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool],
    ):
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._domain_obj = mass_ops.domain
        self._active_faces = active_faces

        self._space_key = "1"
        self._space = self._derham.fem_spaces[self._space_key]
        self._quad_grid_pts = self._derham.spline_attributes[self._space_key].quad_grid_pts
        self._spans_l = self._derham.spline_attributes[self._space_key].quad_grid_spans
        self._wts_l = self._derham.spline_attributes[self._space_key].quad_grid_wts
        self._bases_l = self._derham.spline_attributes[self._space_key].quad_grid_bases
        self._tensor_fem_spaces = self._derham.spline_attributes[self._space_key].tensor_spaces

        self._V_extraction_op = self._derham.extraction_ops[self._space_key]
        self._W_extraction_op = self._derham.extraction_ops[self._space_key]
        self._V_boundary_op = self._derham.boundary_ops[self._space_key]
        self._W_boundary_op = self._derham.boundary_ops[self._space_key]

        self._V_extraction_op_T = self._V_extraction_op.T
        self._W_extraction_op_T = self._W_extraction_op.T
        self._V_boundary_op_T = self._V_boundary_op.T
        self._W_boundary_op_T = self._W_boundary_op.T

        V = self._space
        W = self._space

        blocks = [
            [
                StencilMatrix(
                    Vs.coeff_space,
                    Ws.coeff_space,
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )
                if i != j
                else None
                for j, Vs in enumerate(V.spaces)
            ]
            for i, Ws in enumerate(W.spaces)
        ]

        self._mat = BlockLinearOperator(
            V.coeff_space,
            W.coeff_space,
            blocks=blocks,
        )

        self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op_T
        self._M0 = self._W_boundary_op @ self._M @ self._V_boundary_op_T

        self._domain = self._M0.domain
        self._codomain = self._M0.codomain
        self._dtype = self._tensor_fem_spaces[0].coeff_space.dtype

        # allocate temporaries
        self._temp_VB = self._V_boundary_op.domain.zeros()
        self._temp_VE = self._V_extraction_op.domain.zeros()
        self._temp_WB = self._W_boundary_op.domain.zeros()
        self._temp_WE = self._W_extraction_op.domain.zeros()
        self._temp_mat = self._mat.domain.zeros()

        # for each active face extract surface data
        self._surface_R_n = []
        self._surface_spans = []
        self._surface_wts = []
        self._surface_bases = []

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                self._surface_R_n.append(None)
                self._surface_spans.append(None)
                self._surface_wts.append(None)
                self._surface_bases.append(None)
                continue

            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]

            sign = 1.0 if face_idx < 3 else -1.0
            n_hat = xp.zeros(3)
            n_hat[normal_dir] = sign

            # constant skew-symmetric cross-product matrix R_n such that R_n v = n_hat x v
            R_n_const = xp.zeros((3, 3))
            R_n_const[0, 1] = -n_hat[2]
            R_n_const[0, 2] =  n_hat[1]
            R_n_const[1, 0] =  n_hat[2]
            R_n_const[1, 2] = -n_hat[0]
            R_n_const[2, 0] = -n_hat[1]
            R_n_const[2, 1] =  n_hat[0]

            # store R_n per component mu on its own quadrature grid shape
            surface_R_n_per_mu = [None, None, None]
            for mu in surf_dirs:
                nq1 = self._spans_l[mu][surf_dirs[0]].size * self._wts_l[mu][surf_dirs[0]].shape[1]
                nq2 = self._spans_l[mu][surf_dirs[1]].size * self._wts_l[mu][surf_dirs[1]].shape[1]
                R_n_mu = xp.zeros((nq1, nq2, 3, 3))
                R_n_mu[..., :, :] = R_n_const
                surface_R_n_per_mu[mu] = R_n_mu

            self._surface_R_n.append(surface_R_n_per_mu)
            surface_spans_per_mu = [None, None, None]
            surface_wts_per_mu = [None, None, None]
            surface_bases_per_mu = [None, None, None]

            for mu in surf_dirs:
                surface_spans_per_mu[mu] = [self._spans_l[mu][d] for d in surf_dirs]
                surface_wts_per_mu[mu] = [self._wts_l[mu][d] for d in surf_dirs]
                surface_bases_per_mu[mu] = [self._bases_l[mu][d] for d in surf_dirs]

            self._surface_spans.append(surface_spans_per_mu)
            self._surface_wts.append(surface_wts_per_mu)
            self._surface_bases.append(surface_bases_per_mu)

        self._assembly_kernel = Pyccelkernel(mass_kernels.surface_kernel_3d_mat_h1)

        self.assemble()

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def _assemble_face(self, face_idx: int, mat):
        normal_dir = face_idx % 3
        surf_dirs = [d for d in range(3) if d != normal_dir]

        mu, nu = surf_dirs[0], surf_dirs[1]

        fem_space_mu = self._tensor_fem_spaces[mu]
        fem_space_nu = self._tensor_fem_spaces[nu]

        starts_mu = [int(s) for s in fem_space_mu.coeff_space.starts]
        ends_mu = [int(e) for e in fem_space_mu.coeff_space.ends]
        pads_mu = fem_space_mu.coeff_space.pads

        starts_nu = [int(s) for s in fem_space_nu.coeff_space.starts]
        ends_nu = [int(e) for e in fem_space_nu.coeff_space.ends]
        pads_nu = fem_space_nu.coeff_space.pads

        boundary_index_mu = starts_mu[normal_dir] if face_idx < 3 else ends_mu[normal_dir]
        boundary_index_nu = starts_nu[normal_dir] if face_idx < 3 else ends_nu[normal_dir]

        mat_fun_mu_nu = self._surface_R_n[face_idx][mu][..., mu, nu]
        mat_fun_nu_mu = self._surface_R_n[face_idx][nu][..., nu, mu]

        self._assembly_kernel(
            *self._surface_spans[face_idx][mu],
            *fem_space_mu.degree,
            *fem_space_nu.degree,
            *starts_mu,
            *pads_mu,
            *self._surface_wts[face_idx][mu],
            *self._surface_bases[face_idx][mu],
            *self._surface_bases[face_idx][nu],
            boundary_index_mu,
            normal_dir,
            mat_fun_mu_nu,
            mat.blocks[mu][nu]._data,
        )

        self._assembly_kernel(
            *self._surface_spans[face_idx][nu],
            *fem_space_nu.degree,
            *fem_space_mu.degree,
            *starts_nu,
            *pads_nu,
            *self._surface_wts[face_idx][nu],
            *self._surface_bases[face_idx][nu],
            *self._surface_bases[face_idx][mu],
            boundary_index_nu,
            normal_dir,
            mat_fun_nu_mu,
            mat.blocks[nu][mu]._data,
        )

    def assemble(self, clear: bool = True):
        """Assembles the H(curl) boundary mass matrix."""
        if clear:
            for mu in range(3):
                for nu in range(3):
                    if mu != nu:
                        self._mat.blocks[mu][nu]._data[:] = 0.0

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                continue
            self._assemble_face(face_idx, self._mat)

        for mu in range(3):
            for nu in range(3):
                if mu != nu:
                    self._mat.blocks[mu][nu].exchange_assembly_data()
                    self._mat.blocks[mu][nu].update_ghost_regions()

    def dot(self, v, out=None, apply_bc=True):
        """Applies the H(curl) boundary mass matrix to a BlockVector."""
        if out is None:
            out = self.codomain.zeros()

        if apply_bc:
            self._V_boundary_op_T.dot(v, out=self._temp_VB)
            self._V_extraction_op_T.dot(self._temp_VB, out=self._temp_mat)
            self._mat.dot(self._temp_mat, out=self._temp_WE)
            self._W_extraction_op.dot(self._temp_WE, out=self._temp_WB)
            self._W_boundary_op.dot(self._temp_WB, out=out)
        else:
            self._V_extraction_op_T.dot(v, out=self._temp_mat)
            self._mat.dot(self._temp_mat, out=self._temp_WE)
            self._W_extraction_op.dot(self._temp_WE, out=out)

        return out

    def transpose(self, conjugate=False):
        return self

    def toarray(self):
        return self._M0.toarray()

    def tosparse(self):
        return self._M0.tosparse()