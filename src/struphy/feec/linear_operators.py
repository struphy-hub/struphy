import cunumpy as xp
from feectools.linalg.basic import LinearOperator, Vector, VectorSpace
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector, StencilVectorSpace

from struphy.feec.utilities import apply_essential_bc_to_array
from struphy.polar.basic import PolarDerhamSpace


def _blocks(v):
    """The StencilVector blocks of a StencilVector or BlockVector (of StencilVectors), else None."""
    if isinstance(v, StencilVector):
        return [v]
    if isinstance(v, BlockVector) and all(isinstance(b, StencilVector) for b in v.blocks):
        return list(v.blocks)
    return None


_MASKED_COPY_SRC = r"""
extern "C" __global__ void masked_copy(const unsigned long long* __restrict__ ptrs, const long long* __restrict__ offsets,
                                       int n_comp, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = 0;
    while (c < n_comp - 1 && i >= offsets[c + 1]) ++c;
    long long j = i - offsets[c];
    const double* v = (const double*)ptrs[3 * c];
    double* out = (double*)ptrs[3 * c + 1];
    const double* mask = (const double*)ptrs[3 * c + 2];
    out[j] = v[j] * mask[j];
}
"""


def _masked_copy_fused(op, v_blocks, out_blocks, masks) -> bool:
    """``out_blocks[c] = v_blocks[c] * masks[c]`` for all blocks with one kernel (pointer tables cached on ``op``).

    False (nothing done) unless every block is C-contiguous float64 of the shape of its mask."""
    arrays = [(vb._data, ob._data, m) for vb, ob, (m, _) in zip(v_blocks, out_blocks, masks)]
    for a, b, m in arrays:
        if not (
            a.dtype == b.dtype == m.dtype == xp.float64
            and a.shape == b.shape == m.shape
            and a.flags.c_contiguous
            and b.flags.c_contiguous
            and m.flags.c_contiguous
        ):
            return False
    import cupy as cp
    import numpy as np

    cache = op.__dict__.setdefault("_masked_copy_cache", {})
    key = tuple(int(x.data.ptr) for triple in arrays for x in triple)
    entry = cache.get(key)
    if entry is None:
        if "kernel" not in cache:
            cache["kernel"] = cp.RawKernel(_MASKED_COPY_SRC, "masked_copy")
        sizes = [a.size for a, _, _ in arrays]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        entry = (cp.asarray(np.array(key, dtype=np.uint64)), cp.asarray(offsets), int(offsets[-1]))
        if len(cache) > 64:
            kernel = cache["kernel"]
            cache.clear()
            cache["kernel"] = kernel
        cache[key] = entry
    ptrs, offsets, total = entry
    if total:
        cache["kernel"](((total + 255) // 256,), (256,), (ptrs, offsets, np.int32(len(arrays)), np.int64(total)))
    return True


class BoundaryOperator(LinearOperator):
    r"""
    Applies homogeneous Dirichlet boundary conditions to a vector.

    Parameters
    ----------
    vector_space : feectools.linalg.basic.VectorSpace
        The vector space associated to the operator.

    space_id : str
        Symbolic space ID of vector_space (H1, Hcurl, Hdiv, L2 or H1vec).

    dirichlet_bc : tuple[tuple[bool]]
        Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).
    """

    def __init__(self, vector_space, space_id, dirichlet_bc):
        assert isinstance(vector_space, VectorSpace)
        assert isinstance(space_id, str)

        self._domain = vector_space
        self._codomain = vector_space
        self._dtype = vector_space.dtype

        self._space_id = space_id
        self._bc = dirichlet_bc

        assert isinstance(dirichlet_bc, tuple)
        assert len(dirichlet_bc) == 3

        # number of non-zero elements in poloidal/toroidal direction
        if isinstance(vector_space, PolarDerhamSpace):
            vec_space_ten = vector_space.parent_space
        else:
            vec_space_ten = vector_space

        if isinstance(vec_space_ten, StencilVectorSpace):
            n_pts = vec_space_ten.npts
        else:
            n_pts = [comp.npts for comp in vec_space_ten.spaces]

        dim_nz1_pol = 1
        dim_nz2_pol = 1
        dim_nz3_pol = 1

        dim_nz1_tor = 1
        dim_nz2_tor = 1
        dim_nz3_tor = 1

        if space_id == "H1":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0] - self.bc[0][1]) * n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor,)

        elif space_id == "Hcurl":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] - vector_space.n_rings[0]) * n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] - vector_space.n_rings[1] - self.bc[0][1]) * n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] - vector_space.n_rings[2] - self.bc[0][1]) * n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0]
                dim_nz1_pol *= n_pts[0][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz2_pol *= n_pts[1][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz2_pol *= n_pts[1][1]

                dim_nz3_pol *= n_pts[2][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz3_pol *= n_pts[2][1] - self.bc[1][0] - self.bc[1][1]

            dim_nz1_tor *= n_pts[0][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz2_tor *= n_pts[1][2] - self.bc[2][0] - self.bc[2][1]
            dim_nz3_tor *= n_pts[2][2]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor, dim_nz2_pol * dim_nz2_tor, dim_nz3_pol * dim_nz3_tor)

        elif space_id == "Hdiv" or space_id == "H1vec":
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0][0] - vector_space.n_rings[0] - self.bc[0][1]) * n_pts[0][1]
                dim_nz1_pol += vector_space.n_polar[0]

                dim_nz2_pol *= (n_pts[1][0] - vector_space.n_rings[1]) * n_pts[1][1]
                dim_nz2_pol += vector_space.n_polar[1]

                dim_nz3_pol *= (n_pts[2][0] - vector_space.n_rings[2]) * n_pts[2][1]
                dim_nz3_pol += vector_space.n_polar[2]
            else:
                dim_nz1_pol *= n_pts[0][0] - self.bc[0][0] - self.bc[0][1]
                dim_nz1_pol *= n_pts[0][1]

                dim_nz2_pol *= n_pts[1][0]
                dim_nz2_pol *= n_pts[1][1] - self.bc[1][0] - self.bc[1][1]

                dim_nz3_pol *= n_pts[2][0]
                dim_nz3_pol *= n_pts[2][1]

            dim_nz1_tor *= n_pts[0][2]
            dim_nz2_tor *= n_pts[1][2]
            dim_nz3_tor *= n_pts[2][2] - self.bc[2][0] - self.bc[2][1]

            self._dim_nz_pol = (dim_nz1_pol, dim_nz2_pol, dim_nz3_pol)
            self._dim_nz_tor = (dim_nz1_tor, dim_nz2_tor, dim_nz3_tor)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor, dim_nz2_pol * dim_nz2_tor, dim_nz3_pol * dim_nz3_tor)

        else:
            if isinstance(vector_space, PolarDerhamSpace):
                dim_nz1_pol *= (n_pts[0] - vector_space.n_rings[0]) * n_pts[1]
                dim_nz1_pol += vector_space.n_polar[0]
            else:
                dim_nz1_pol *= n_pts[0]
                dim_nz1_pol *= n_pts[1]

            dim_nz1_tor *= n_pts[2]

            self._dim_nz_pol = (dim_nz1_pol,)
            self._dim_nz_tor = (dim_nz1_tor,)

            self._dim_nz = (dim_nz1_pol * dim_nz1_tor,)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def bc(self):
        return self._bc

    @property
    def dim_nz_pol(self):
        return self._dim_nz_pol

    @property
    def dim_nz_tor(self):
        return self._dim_nz_tor

    @property
    def dim_nz(self):
        return self._dim_nz

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            The input (domain) vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
        out : feectools.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self._domain

        if out is None:
            out = v.copy()
        else:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            masks = self._device_masks(v)
            if masks is not None:
                # copy and boundary conditions in one kernel per block (see _device_masks)
                in_sync = True
                v_blocks, out_blocks = _blocks(v), _blocks(out)
                if len(v_blocks) > 1 and _masked_copy_fused(self, v_blocks, out_blocks, masks):
                    pass
                else:
                    for v_blk, out_blk, (mask, _) in zip(v_blocks, out_blocks, masks):
                        xp.multiply(v_blk._data, mask, out=out_blk._data)
                for v_blk, (_, stale) in zip(v_blocks, masks):
                    in_sync = in_sync and v_blk.ghost_regions_in_sync and not stale
                # one flag for all blocks: a BlockVector flagged in sync with stale blocks would have them updated
                # one at a time by the operators that check each block, instead of together
                out.ghost_regions_in_sync = in_sync
                return out
            v.copy(out=out)

        # apply boundary conditions to output vector
        apply_essential_bc_to_array(self._space_id, out, self.bc)

        return out

    def _device_masks(self, v):
        """Per block of ``v``: (0/1 mask of the entries the boundary conditions keep, whether they stale the ghosts).

        For device (CuPy) data of Stencil/Block vectors, ``dot`` computes ``out = v * mask`` (one kernel per block)
        instead of a copy and one fill per boundary and component. The masks are found once, by applying
        :func:`apply_essential_bc_to_array` to a vector of ones, so they zero exactly the entries it zeroes (ghost
        entries included) and flag the same blocks as having stale ghost regions. None on the host backend and for
        other vectors (e.g. polar vectors), which use the copy and :func:`apply_essential_bc_to_array`.
        """
        if not hasattr(self, "_masks_cache"):
            self._masks_cache = None
            blocks = _blocks(v)
            if blocks is not None and all(xp.is_gpu(b._data) for b in blocks):
                probe = v.space.zeros()
                probe_blocks = _blocks(probe)
                for b in probe_blocks:
                    b._data[...] = 1.0
                    b.ghost_regions_in_sync = True
                apply_essential_bc_to_array(self._space_id, probe, self.bc)
                self._masks_cache = [(b._data.copy(), not b.ghost_regions_in_sync) for b in probe_blocks]
        return self._masks_cache

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return BoundaryOperator(self._domain, self._space_id, self.bc)
