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

class BoundaryIntegralOperator:
    """
    Assembles the boundary integral vector for H1 basis functions.

    Computes the surface integrals

        I_i' = int_{partial Omega_i'} psi_h Tr(alpha) sqrt(g) |DF^-T n_hat_i| dS

    and adds them together into a single StencilVector v such that

        I = psi^T v

    for any discrete test function psi_h in V^0_h.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        Mass operators object, contains geometry and derham.
    """

    def __init__(
        self,
        mass_ops: WeightedMassOperators,
    ):
        self._mass_ops = mass_ops
        self._derham = mass_ops.derham
        self._domain = mass_ops.domain

        # H1 space info
        self._space = self._derham.fem_spaces["0"]
        self._space_key = "0"

        # 3D quadrature grid info for H1 space
        self._quad_grid_pts = self._derham.spline_attributes[self._space_key].quad_grid_pts
        self._spans_l = self._derham.spline_attributes[self._space_key].quad_grid_spans
        self._wts_l = self._derham.spline_attributes[self._space_key].quad_grid_wts
        self._bases_l = self._derham.spline_attributes[self._space_key].quad_grid_bases
        self._tensor_fem_spaces = self._derham.spline_attributes[self._space_key].tensor_spaces

        # for each of the 6 faces, extract surface quadrature grid and geometric weights
        self._surface_quad_grid_meshes = []
        self._surface_geom_weights = []
        self._surface_spans = []
        self._surface_wts = []
        self._surface_bases = []

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

            # take quadrature points in the two surface directions
            surf_pts = [self._quad_grid_pts[0][d].flatten() for d in surf_dirs]

            # build 2D meshgrid over surface
            self._surface_quad_grid_meshes.append(xp.meshgrid(*surf_pts, indexing="ij"))

            # compute geometric weights         
            fixed_val = 0.0 if face_idx < 3 else 1.0

            surf_pts_1d = [self._quad_grid_pts[0][d].flatten() for d in surf_dirs]

            e_1d = [None, None, None]
            e_1d[surf_dirs[0]] = surf_pts_1d[0]
            e_1d[surf_dirs[1]] = surf_pts_1d[1]
            e_1d[normal_dir] = xp.array([fixed_val])

            sqrt_g = xp.abs(self._domain.jacobian_det(*e_1d))  # metric

            DFinv = self._domain.jacobian_inv(*e_1d, change_out_order=True)
            DFinv_n = DFinv[..., normal_dir, :]
            norm_DFinv_n = xp.sqrt(xp.sum(DFinv_n**2, axis=-1))  # jacobian

            surface_geom_weights = sqrt_g * norm_DFinv_n
            surface_geom_weights = xp.squeeze(surface_geom_weights)
            self._surface_geom_weights.append(surface_geom_weights)
                        

            # extract surface spans, weights, bases for 2D quadrature
            self._surface_spans.append([self._spans_l[0][d] for d in surf_dirs])  # global index of the last nonzero spline
            self._surface_wts.append([self._wts_l[0][d] for d in surf_dirs])  # quadrature weights
            self._surface_bases.append([self._bases_l[0][d] for d in surf_dirs])  # spline values

    def _assemble_face(
        self,
        face_idx: int,
        fun_weights: xp.ndarray,
        dofs: StencilVector,
    ):
        """
        Assembles the contribution of a single face to the boundary integral vector.

        Parameters
        ----------
        face_idx : int
            Index of the face (0 to 5).

        fun_weights : xp.ndarray
            Function alpha evaluated at the surface quadrature points,
            already multiplied by the surface Jacobian.

        dofs : StencilVector
            Output vector to accumulate into.
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

        mass_kernels.surface_kernel_3d_vec(
            *self._surface_spans[face_idx],
            *fem_space.degree,
            *starts,
            *pads,
            *self._surface_wts[face_idx],
            *self._surface_bases[face_idx],
            boundary_index,
            fun_weights,
            dofs._data,
        )


    def assemble_callable(
        self,
        fun: Callable,
        dofs: StencilVector = None,
        clear: bool = True,
    ) -> StencilVector:
        """
        Assembles the boundary integral vector for a callable function alpha.

        Parameters
        ----------
        fun : Callable
            The function alpha(eta1, eta2, eta3) in logical coordinates.

        dofs : StencilVector, optional
            Output vector. If None, a new zero vector is created.

        clear : bool, optional
            Whether to zero the output vector before assembly.

        Returns
        -------
        dofs : StencilVector
            The assembled boundary integral vector v.
        """
        if dofs is None:
            dofs = self._space.coeff_space.zeros()

        if clear:
            dofs._data[:] = 0.0

        for face_idx in range(6):
            if not self._active_faces[face_idx]:
                continue

            normal_dir = face_idx % 3
            # fix the normal coordinate to 0.0 or 1.0
            fixed_val = 0.0 if face_idx < 3 else 1.0

            surface_mesh = self._surface_quad_grid_meshes[face_idx]
            normal_dir = face_idx % 3
            surf_dirs = [d for d in range(3) if d != normal_dir]

            e = [None, None, None]
            e[surf_dirs[0]] = surface_mesh[0]
            e[surf_dirs[1]] = surface_mesh[1]
            e[normal_dir] = xp.full_like(surface_mesh[0], fixed_val)
            e1, e2, e3 = e

            fun_weights = fun(e1, e2, e3)

            # multiply by surface Jacobian
            fun_weights = fun_weights * self._surface_geom_weights[face_idx]
            fun_weights = xp.squeeze(fun_weights)

            self._assemble_face(face_idx, fun_weights, dofs)

            tmp = self._space.coeff_space.zeros()
            self._assemble_face(face_idx, fun_weights, tmp)
            tmp.exchange_assembly_data()
            tmp.update_ghost_regions()

        dofs.exchange_assembly_data()
        dofs.update_ghost_regions()
        
        return dofs

    def __call__(
        self,
        fun: Callable | SplineFunction,
        dofs: StencilVector = None,
        clear: bool = True,
    ) -> StencilVector:
        """
        Assembles the boundary integral vector for a callable or SplineFunction alpha.

        Parameters
        ----------
        fun : Callable | SplineFunction
            The function alpha, either a callable or a SplineFunction.

        dofs : StencilVector, optional
            Output vector. If None, a new zero vector is created.

        clear : bool, optional
            Whether to zero the output vector before assembly.

        Returns
        -------
        dofs : StencilVector
            The assembled boundary integral vector v.
        """
        if callable(fun):
            return self.assemble_callable(fun, dofs=dofs, clear=clear)
        else:
            raise ValueError(
                f"Expected callable, got {type(fun)} instead."
            )


class BoundaryMassOperator(LinOpWithTransp):
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
        self._assembly_kernel = Pyccelkernel(mass_kernels.surface_kernel_3d_mat)

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