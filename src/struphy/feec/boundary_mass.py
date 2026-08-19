import logging
from typing import Literal

import cunumpy as xp
from cunumpy import PyccelKernel
from cunumpy import PyccelKernel
from feectools.api.settings import PSYDAC_BACKEND_GPYCCEL
from feectools.linalg.block import BlockLinearOperator
from feectools.linalg.stencil import StencilMatrix

from struphy.feec import mass_kernels
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.feec.mass import WeightedMassOperators

logger = logging.getLogger("struphy")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ScalarSpace = Literal["H1", "L2"]
VectorSpace = Literal["Hcurl", "Hdiv"]

_SPACE_KEY = {"H1": "0", "Hcurl": "1", "Hdiv": "2", "L2": "3"}


# ---------------------------------------------------------------------------
# Collection class
# ---------------------------------------------------------------------------


class BoundaryIntegralOperators:
    """
    Collection of boundary integral operators for scalar and vector fields.

    Three operators are exposed via methods:

    ``scalar(test_space)``
        int_{dOmega} alpha * beta dS
        data: H1, test: H1 or L2

    ``normal(data_space, test_space)``
        int_{dOmega} (u . n) * alpha dS
        data: Hdiv (canonical) or Hcurl, test: H1 or L2

    ``tangential(data_space, test_space)``
        int_{dOmega} (u x n) . v dS
        data: Hcurl (canonical) or Hdiv, test: Hcurl or Hdiv

    Parameters
    ----------
    mass_ops : WeightedMassOperators
    active_faces : list[bool] or None
        Which of the six faces to integrate over.
        If None, inferred from boundary conditions.
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool] | None = None,
    ):
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._cache: dict = {}

        if active_faces is not None:
            self._active_faces = active_faces
        else:
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

    def scalar(self, test_space: ScalarSpace = "H1") -> "ScalarBoundaryMass":
        """Scalar boundary mass: int_{dOmega} alpha * beta dS. Data: H1."""
        key = ("scalar", test_space)
        if key not in self._cache:
            self._cache[key] = ScalarBoundaryMass(
                self._mass_ops, self._active_faces, test_space=test_space
            )
        return self._cache[key]

    def normal(
        self,
        data_space: VectorSpace = "Hdiv",
        test_space: ScalarSpace = "H1",
    ) -> "NormalBoundaryMass":
        """Normal trace boundary mass: int_{dOmega} (u.n) * alpha dS."""
        key = ("normal", data_space, test_space)
        if key not in self._cache:
            self._cache[key] = NormalBoundaryMass(
                self._mass_ops, self._active_faces,
                data_space=data_space, test_space=test_space,
            )
        return self._cache[key]

    def tangential(
        self,
        data_space: VectorSpace = "Hcurl",
        test_space: VectorSpace = "Hcurl",
    ) -> "TangentialBoundaryMass":
        """Tangential trace boundary mass: int_{dOmega} (u x n).v dS."""
        key = ("tangential", data_space, test_space)
        if key not in self._cache:
            self._cache[key] = TangentialBoundaryMass(
                self._mass_ops, self._active_faces,
                data_space=data_space, test_space=test_space,
            )
        return self._cache[key]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BoundaryMassOperator(LinOpWithTransp):
    """
    Base class for boundary mass operators.

    Subclasses set ``_data_space_key`` and ``_test_space_key`` before calling
    ``super().__init__``, and implement:
        _build_mat, _setup_surface_data, _assemble_face, _clear_mat, _finalize_mat, transpose.

    The data (trial/column) space provides the basis functions for the field being integrated.
    The test (row) space provides the basis functions for the test functions.
    """

    _data_space_key: str  # set by subclass before super().__init__
    _test_space_key: str  # set by subclass before super().__init__

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool],
    ):
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._domain_obj = mass_ops.domain
        self._active_faces = active_faces

        # --- data (trial) space ---
        self._data_space = self._derham.fem_spaces[self._data_space_key]
        self._data_spans_l = self._derham.spline_attributes[self._data_space_key].quad_grid_spans
        self._data_wts_l = self._derham.spline_attributes[self._data_space_key].quad_grid_wts
        self._data_bases_l = self._derham.spline_attributes[self._data_space_key].quad_grid_bases
        self._data_tensor_spaces = self._derham.spline_attributes[self._data_space_key].tensor_spaces
        self._data_nbasis = self._derham.spline_attributes[self._data_space_key].nbasis

        # --- test space ---
        self._test_space = self._derham.fem_spaces[self._test_space_key]
        self._test_spans_l = self._derham.spline_attributes[self._test_space_key].quad_grid_spans
        self._test_wts_l = self._derham.spline_attributes[self._test_space_key].quad_grid_wts
        self._test_bases_l = self._derham.spline_attributes[self._test_space_key].quad_grid_bases
        self._test_tensor_spaces = self._derham.spline_attributes[self._test_space_key].tensor_spaces
        self._test_nbasis = self._derham.spline_attributes[self._test_space_key].nbasis

        # --- extraction and boundary operators ---
        self._V_extraction_op = self._derham.extraction_ops[self._data_space_key]
        self._W_extraction_op = self._derham.extraction_ops[self._test_space_key]
        self._V_boundary_op = self._derham.boundary_ops[self._data_space_key]
        self._W_boundary_op = self._derham.boundary_ops[self._test_space_key]

        self._V_extraction_op_T = self._V_extraction_op.T
        self._V_boundary_op_T = self._V_boundary_op.T

        # --- raw matrix and composite operator S = W_bnd @ W_ext @ M @ V_ext^T @ V_bnd^T ---
        self._mat = self._build_mat()
        self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op_T
        self._M0 = self._W_boundary_op @ self._M @ self._V_boundary_op_T

        self._domain = self._M0.domain
        self._codomain = self._M0.codomain
        self._dtype = self._data_tensor_spaces[0].coeff_space.dtype

        # --- temporaries ---
        self._temp_WB = self._W_boundary_op.domain.zeros()
        self._temp_WE = self._W_extraction_op.domain.zeros()
        self._temp_VB = self._V_boundary_op.domain.zeros()
        self._temp_mat = self._mat.domain.zeros()

        self._setup_surface_data()
        self._assembly_kernel = PyccelKernel(mass_kernels.surface_kernel_3d_mat)
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

    def _build_mat(self):
        raise NotImplementedError

    def _setup_surface_data(self):
        raise NotImplementedError

    def _assemble_face(self, face_idx: int, mat):
        raise NotImplementedError

    def _clear_mat(self):
        raise NotImplementedError

    def _finalize_mat(self):
        raise NotImplementedError

    def assemble(self, clear: bool = True):
        if clear:
            self._clear_mat()
        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                continue
            self._assemble_face(face_idx, self._mat)
        self._finalize_mat()

    def dot(self, v, out=None, apply_bc=True):
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

    def dot_inner(self, u, v) -> float:
        """Compute u^T (S v) summed over all components."""
        Sv = self.dot(v)
        if hasattr(Sv, "blocks"):
            total = 0.0
            for mu in range(len(Sv.blocks)):
                u_mu = u.blocks[mu] if hasattr(u, "blocks") else u[mu]
                Sv_mu = Sv.blocks[mu]
                total += float(xp.sum(u_mu.toarray() * Sv_mu.toarray()))
            return total
        u_arr = u.toarray() if hasattr(u, "toarray") else xp.asarray(u)
        Sv_arr = Sv.toarray() if hasattr(Sv, "toarray") else xp.asarray(Sv)
        return float(xp.sum(u_arr * Sv_arr))

    def toarray(self):
        return self._M0.toarray()

    def tosparse(self):
        return self._M0.tosparse()


# ---------------------------------------------------------------------------
# ScalarBoundaryMass: int_{dOmega} alpha * beta dS
# data: H1,  test: H1 or L2
# ---------------------------------------------------------------------------


class ScalarBoundaryMass(BoundaryMassOperator):
    """
    Scalar boundary mass operator.

        int_{dOmega} alpha * beta dS

    Data space : H1
    Test space : H1 (default) or L2

    Parameters
    ----------
    mass_ops : WeightedMassOperators
    active_faces : list[bool]
    test_space : "H1" or "L2"
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool],
        test_space: ScalarSpace = "H1",
    ):
        self._data_space_key = _SPACE_KEY["H1"]
        self._test_space_key = _SPACE_KEY[test_space]
        super().__init__(mass_ops, active_faces)

    def _build_mat(self) -> StencilMatrix:
        data_fem = self._data_tensor_spaces[0]
        test_fem = self._test_tensor_spaces[0]
        return StencilMatrix(
            data_fem.coeff_space,
            test_fem.coeff_space,
            backend=PSYDAC_BACKEND_GPYCCEL,
            precompiled=True,
        )

    def _setup_surface_data(self):
        self._surface_geom_weights = []
        self._surface_data_spans = []
        self._surface_data_wts = []
        self._surface_data_bases = []
        self._surface_test_bases = []

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                for lst in (
                    self._surface_geom_weights,
                    self._surface_data_spans, self._surface_data_wts,
                    self._surface_data_bases, self._surface_test_bases,
                ):
                    lst.append(None)
                continue

            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]
            fixed_val = 0.0 if face_idx < 3 else 1.0

            surf_pts_1d = [self._data_spans_l[0][d].flatten() for d in surf_dirs]
            e_1d = [None, None, None]
            e_1d[surf_dirs[0]] = surf_pts_1d[0]
            e_1d[surf_dirs[1]] = surf_pts_1d[1]
            e_1d[normal_dir] = xp.array([fixed_val])

            sqrt_g = xp.abs(self._domain_obj.jacobian_det(*e_1d))
            DFinv = self._domain_obj.jacobian_inv(*e_1d, change_out_order=True)
            DFinv_n = DFinv[..., normal_dir, :]
            norm_DFinv_n = xp.sqrt(xp.sum(DFinv_n**2, axis=-1))
            self._surface_geom_weights.append(xp.squeeze(sqrt_g * norm_DFinv_n))

            self._surface_data_spans.append([self._data_spans_l[0][d] for d in surf_dirs])
            self._surface_data_wts.append([self._data_wts_l[0][d] for d in surf_dirs])
            self._surface_data_bases.append([self._data_bases_l[0][d] for d in surf_dirs])
            self._surface_test_bases.append([self._test_bases_l[0][d] for d in surf_dirs])

    def _assemble_face(self, face_idx: int, mat: StencilMatrix):
        normal_dir = face_idx % 3
        data_fem = self._data_tensor_spaces[0]
        starts = [int(s) for s in data_fem.coeff_space.starts]
        ends = [int(e) for e in data_fem.coeff_space.ends]
        pads = data_fem.coeff_space.pads
        boundary_index = 0 if face_idx < 3 else self._data_nbasis[0][normal_dir] - 1

        logger.debug(f"{normal_dir=}, {face_idx=}, {boundary_index=}, {starts=}, {ends=}, {pads=}")

        if starts[normal_dir] == boundary_index or ends[normal_dir] == boundary_index:
            self._assembly_kernel(
                *self._surface_data_spans[face_idx],
                *data_fem.degree,
                *data_fem.degree,
                *starts,
                *pads,
                *self._surface_data_wts[face_idx],
                *self._surface_data_bases[face_idx],
                *self._surface_test_bases[face_idx],
                boundary_index,
                normal_dir,
                self._surface_geom_weights[face_idx],
                mat._data,
            )

    def _clear_mat(self):
        self._mat._data[:] = 0.0

    def _finalize_mat(self):
        self._mat.exchange_assembly_data()
        self._mat.update_ghost_regions()

    def transpose(self, conjugate=False):
        return self  # symmetric when data == test space


# ---------------------------------------------------------------------------
# NormalBoundaryMass: int_{dOmega} (u . n) * alpha dS
# data: Hdiv or Hcurl (vector),  test: H1 or L2 (scalar)
# ---------------------------------------------------------------------------


class NormalBoundaryMass(BoundaryMassOperator):
    """
    Normal trace boundary mass operator.

        int_{dOmega} (u . n) * alpha dS

    The normal trace (u.n) is scalar, so the test space is always scalar.

    Data space : Hdiv (canonical) or Hcurl
    Test space : H1 (default) or L2

    The raw matrix is a BlockLinearOperator with 1 test block x 3 data blocks.
    On each face only the component aligned with the normal contributes,
    since (e_mu . n) = 0 for tangential components.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
    active_faces : list[bool]
    data_space : "Hdiv" or "Hcurl"
    test_space : "H1" or "L2"
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool],
        data_space: VectorSpace = "Hdiv",
        test_space: ScalarSpace = "H1",
    ):
        self._data_space_key = _SPACE_KEY[data_space]
        self._test_space_key = _SPACE_KEY[test_space]
        super().__init__(mass_ops, active_faces)

    def _build_mat(self) -> BlockLinearOperator:
        # 1 x 3 block operator: scalar test, vector data
        test_fem = self._test_tensor_spaces[0]
        blocks = [
            [
                StencilMatrix(
                    self._data_tensor_spaces[mu].coeff_space,
                    test_fem.coeff_space,
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )
                for mu in range(3)
            ]
        ]
        return BlockLinearOperator(
            self._data_space.coeff_space,
            test_fem.coeff_space,
            blocks=blocks,
        )

    def _setup_surface_data(self):
        self._surface_sign = []
        self._surface_normal_dir = []
        self._surface_data_spans = []
        self._surface_data_wts = []
        self._surface_data_bases = []
        self._surface_test_bases = []

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                for lst in (
                    self._surface_sign, self._surface_normal_dir,
                    self._surface_data_spans, self._surface_data_wts,
                    self._surface_data_bases, self._surface_test_bases,
                ):
                    lst.append(None)
                continue

            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]
            sign = - 1.0 if face_idx < 3 else 1.0
    
            self._surface_sign.append(sign)
            self._surface_normal_dir.append(normal_dir)

            # only the normal_dir component of the data field contributes
            mu = normal_dir
            self._surface_data_spans.append([self._data_spans_l[mu][d] for d in surf_dirs])
            self._surface_data_wts.append([self._data_wts_l[mu][d] for d in surf_dirs])
            self._surface_data_bases.append([self._data_bases_l[mu][d] for d in surf_dirs])
            self._surface_test_bases.append([self._test_bases_l[0][d] for d in surf_dirs])

    def _assemble_face(self, face_idx: int, mat: BlockLinearOperator):
        normal_dir = self._surface_normal_dir[face_idx]
        sign = self._surface_sign[face_idx]
        mu = normal_dir  # only normal component contributes

        data_fem_mu = self._data_tensor_spaces[mu]
        starts_mu = [int(s) for s in data_fem_mu.coeff_space.starts]
        ends_mu = [int(e) for e in data_fem_mu.coeff_space.ends]
        pads_mu = data_fem_mu.coeff_space.pads
        boundary_index_mu = 0 if face_idx < 3 else self._data_nbasis[mu][normal_dir] - 1

        nq1 = self._surface_data_spans[face_idx][0].size * self._surface_data_wts[face_idx][0].shape[1]
        nq2 = self._surface_data_spans[face_idx][1].size * self._surface_data_wts[face_idx][1].shape[1]
        geom_weight = xp.full((nq1, nq2), sign)

        logger.debug(f"{normal_dir=}, {face_idx=}, {boundary_index_mu=}, {starts_mu=}, {ends_mu=}, {pads_mu=}")

        if starts_mu[normal_dir] == boundary_index_mu or ends_mu[normal_dir] == boundary_index_mu:
            self._assembly_kernel(
                *self._surface_data_spans[face_idx],
                *data_fem_mu.degree,
                *self._test_tensor_spaces[0].degree,
                *starts_mu,
                *pads_mu,
                *self._surface_data_wts[face_idx],
                *self._surface_data_bases[face_idx],
                *self._surface_test_bases[face_idx],
                boundary_index_mu,
                normal_dir,
                geom_weight,
                mat.blocks[0][mu]._data,
            )

    def _clear_mat(self):
        for mu in range(3):
            self._mat.blocks[0][mu]._data[:] = 0.0

    def _finalize_mat(self):
        for mu in range(3):
            self._mat.blocks[0][mu].exchange_assembly_data()
            self._mat.blocks[0][mu].update_ghost_regions()

    def transpose(self, conjugate=False):
        raise NotImplementedError(
            "Transpose of NormalBoundaryMass maps scalar -> vector; not implemented."
        )


# ---------------------------------------------------------------------------
# TangentialBoundaryMass: int_{dOmega} (u x n) . v dS
# data: Hcurl or Hdiv (vector),  test: Hcurl or Hdiv (vector)
# ---------------------------------------------------------------------------


class TangentialBoundaryMass(BoundaryMassOperator):
    """
    Tangential trace boundary mass operator.

        int_{dOmega} (u x n) . v dS

    The tangential trace (u x n) is a vector on the boundary, so the test
    space is also vector-valued.

    Data space : Hcurl (canonical) or Hdiv
    Test space : Hcurl (default) or Hdiv

    The raw matrix is a 3x3 BlockLinearOperator. The skew-symmetry of (n x .)
    means diagonal blocks are zero; only the two off-diagonal blocks per face
    (corresponding to the two surface directions) are assembled.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
    active_faces : list[bool]
    data_space : "Hcurl" or "Hdiv"
    test_space : "Hcurl" or "Hdiv"
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
        active_faces: list[bool],
        data_space: VectorSpace = "Hcurl",
        test_space: VectorSpace = "Hcurl",
    ):
        self._data_space_key = _SPACE_KEY[data_space]
        self._test_space_key = _SPACE_KEY[test_space]
        super().__init__(mass_ops, active_faces)

    def _build_mat(self) -> BlockLinearOperator:
        # 3x3 block matrix, off-diagonal blocks only (diagonal zero by skew-symmetry)
        blocks = [
            [
                StencilMatrix(
                    self._data_tensor_spaces[j].coeff_space,
                    self._test_tensor_spaces[i].coeff_space,
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )
                if i != j
                else None
                for j in range(3)
            ]
            for i in range(3)
        ]
        return BlockLinearOperator(
            self._data_space.coeff_space,
            self._test_space.coeff_space,
            blocks=blocks,
        )

    def _setup_surface_data(self):
        self._surface_R_n = []
        self._surface_data_spans = []
        self._surface_data_wts = []
        self._surface_data_bases = []
        self._surface_test_bases = []

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                for lst in (
                    self._surface_R_n,
                    self._surface_data_spans, self._surface_data_wts,
                    self._surface_data_bases, self._surface_test_bases,
                ):
                    lst.append(None)
                continue

            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]
            sign = 1.0 if face_idx < 3 else -1.0

            n_hat = xp.zeros(3)
            n_hat[normal_dir] = sign

            # skew-symmetric cross-product matrix: R_n v = n x v
            R_n = xp.zeros((3, 3))
            R_n[0, 1] = -n_hat[2]
            R_n[0, 2] = n_hat[1]
            R_n[1, 0] = n_hat[2]
            R_n[1, 2] = -n_hat[0]
            R_n[2, 0] = -n_hat[1]
            R_n[2, 1] = n_hat[0]

            # broadcast R_n onto each data component's quadrature grid
            surface_R_n_per_mu = [None, None, None]
            for mu in surf_dirs:
                nq1 = self._data_spans_l[mu][surf_dirs[0]].size * self._data_wts_l[mu][surf_dirs[0]].shape[1]
                nq2 = self._data_spans_l[mu][surf_dirs[1]].size * self._data_wts_l[mu][surf_dirs[1]].shape[1]
                R_n_mu = xp.zeros((nq1, nq2, 3, 3))
                R_n_mu[..., :, :] = R_n
                surface_R_n_per_mu[mu] = R_n_mu
            self._surface_R_n.append(surface_R_n_per_mu)

            data_spans_per_mu = [None, None, None]
            data_wts_per_mu = [None, None, None]
            data_bases_per_mu = [None, None, None]
            test_bases_per_mu = [None, None, None]

            for mu in surf_dirs:
                data_spans_per_mu[mu] = [self._data_spans_l[mu][d] for d in surf_dirs]
                data_wts_per_mu[mu] = [self._data_wts_l[mu][d] for d in surf_dirs]
                data_bases_per_mu[mu] = [self._data_bases_l[mu][d] for d in surf_dirs]
                test_bases_per_mu[mu] = [self._test_bases_l[mu][d] for d in surf_dirs]

            self._surface_data_spans.append(data_spans_per_mu)
            self._surface_data_wts.append(data_wts_per_mu)
            self._surface_data_bases.append(data_bases_per_mu)
            self._surface_test_bases.append(test_bases_per_mu)

    def _assemble_face(self, face_idx: int, mat: BlockLinearOperator):
        normal_dir = face_idx % 3
        surf_dirs = [d for d in range(3) if d != normal_dir]
        mu, nu = surf_dirs[0], surf_dirs[1]

        data_fem_mu = self._data_tensor_spaces[mu]
        data_fem_nu = self._data_tensor_spaces[nu]

        starts_mu = [int(s) for s in data_fem_mu.coeff_space.starts]
        ends_mu = [int(e) for e in data_fem_mu.coeff_space.ends]
        pads_mu = data_fem_mu.coeff_space.pads

        starts_nu = [int(s) for s in data_fem_nu.coeff_space.starts]
        ends_nu = [int(e) for e in data_fem_nu.coeff_space.ends]
        pads_nu = data_fem_nu.coeff_space.pads

        boundary_index_mu = 0 if face_idx < 3 else self._data_nbasis[mu][normal_dir] - 1
        boundary_index_nu = 0 if face_idx < 3 else self._data_nbasis[nu][normal_dir] - 1

        logger.debug(f"{normal_dir=}, {face_idx=}, {boundary_index_mu=}, {starts_mu=}, {ends_mu=}, {pads_mu=}")
        logger.debug(f"{normal_dir=}, {face_idx=}, {boundary_index_nu=}, {starts_nu=}, {ends_nu=}, {pads_nu=}")

        mat_fun_mu_nu = self._surface_R_n[face_idx][mu][..., mu, nu]
        mat_fun_nu_mu = self._surface_R_n[face_idx][nu][..., nu, mu]

        if starts_mu[normal_dir] == boundary_index_mu or ends_mu[normal_dir] == boundary_index_mu:
            self._assembly_kernel(
                *self._surface_data_spans[face_idx][mu],
                *data_fem_mu.degree,
                *self._test_tensor_spaces[nu].degree,
                *starts_mu,
                *pads_mu,
                *self._surface_data_wts[face_idx][mu],
                *self._surface_data_bases[face_idx][mu],
                *self._surface_test_bases[face_idx][nu],
                boundary_index_mu,
                normal_dir,
                mat_fun_mu_nu,
                mat.blocks[mu][nu]._data,
            )

        if starts_nu[normal_dir] == boundary_index_nu or ends_nu[normal_dir] == boundary_index_nu:
            self._assembly_kernel(
                *self._surface_data_spans[face_idx][nu],
                *data_fem_nu.degree,
                *self._test_tensor_spaces[mu].degree,
                *starts_nu,
                *pads_nu,
                *self._surface_data_wts[face_idx][nu],
                *self._surface_data_bases[face_idx][nu],
                *self._surface_test_bases[face_idx][mu],
                boundary_index_nu,
                normal_dir,
                mat_fun_nu_mu,
                mat.blocks[nu][mu]._data,
            )

    def _clear_mat(self):
        for mu in range(3):
            for nu in range(3):
                if mu != nu:
                    self._mat.blocks[mu][nu]._data[:] = 0.0

    def _finalize_mat(self):
        for mu in range(3):
            for nu in range(3):
                if mu != nu:
                    self._mat.blocks[mu][nu].exchange_assembly_data()
                    self._mat.blocks[mu][nu].update_ghost_regions()

    def transpose(self, conjugate=False):
        return -self