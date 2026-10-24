"""The discrete gradient with its boundary operators as one CUDA kernel.

``Derham.grad`` is ``B1 @ G @ B0.T``: the boundary operator of the 0-forms (a 0/1 mask), the derivatives of the
three directions (feectools ``DirectionalDerivativeOperator``, a difference of neighbouring coefficients) and the
boundary operator of the 1-forms. Applied as composed operators, each product makes three passes over the vectors
(and one more over the input for the masked copy), and its transpose four. :func:`fuse_boundary_gradient` returns an
operator that applies the whole chain with one kernel on the GPU (``fused_derivatives_cuda.cu``), with the same
operations in the same order, so the result is bit for bit the same, and the composed operator otherwise.
"""

from functools import cache
from pathlib import Path

import cunumpy as xp
import numpy as np
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.linalg.basic import ComposedLinearOperator, LinearOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.linear_operators import BoundaryOperator

# entries of the kernel metadata per component, see fused_derivatives_cuda.cu
_META = 8


@cache
def _kernels():
    """The forward and transposed CUDA kernels (compiled on first launch)."""
    from cunumpy.kernels import CudaKernel

    path = Path(__file__).with_name("fused_derivatives_cuda.cu")
    return (
        CudaKernel.from_file(path, "fused_boundary_gradient"),
        CudaKernel.from_file(path, "fused_boundary_gradient_t"),
    )


def _derivative_blocks(block: BlockLinearOperator):
    """The derivatives of a gradient block (one block column) or its transpose (one block row), else None."""
    if not isinstance(block, BlockLinearOperator):
        return None
    n_rows, n_cols = block.n_block_rows, block.n_block_cols
    if min(n_rows, n_cols) != 1 or not 1 <= max(n_rows, n_cols) <= 3:
        return None
    ops = [block[k, 0] if n_cols == 1 else block[0, k] for k in range(max(n_rows, n_cols))]
    if not all(isinstance(op, DirectionalDerivativeOperator) and op.domain.ndim == 3 for op in ops):
        return None
    transposed = n_cols > 1 or (n_rows == 1 and ops[0].transposed)
    if any(op.transposed != transposed for op in ops):
        return None
    return ops, transposed


def fuse_boundary_gradient(op: LinearOperator) -> LinearOperator:
    """``op`` as a :class:`FusedBoundaryGradient` if it is a gradient with boundary operators (``B_out @ G @ B_in``,
    ``G`` a block of directional derivatives) or the transpose of one, else ``op`` itself.

    Parameters
    ----------
    op : LinearOperator
        E.g. ``derham.grad`` or ``derham.grad.T``.

    Returns
    -------
    LinearOperator
        The same operator; on the GPU applied with one kernel.
    """
    if isinstance(op, FusedBoundaryGradient) or not isinstance(op, ComposedLinearOperator):
        return op
    factors = op.multiplicands
    if len(factors) != 3:
        return op
    b_out, block, b_in = factors
    if not (isinstance(b_out, BoundaryOperator) and isinstance(b_in, BoundaryOperator)):
        return op
    found = _derivative_blocks(block)
    if found is None:
        return op
    derivatives, transposed = found
    return FusedBoundaryGradient(op, b_out, derivatives, b_in, transposed)


class FusedBoundaryGradient(LinearOperator):
    """``B_out @ G @ B_in`` (a gradient with its boundary operators, or its transpose) with one kernel on the GPU.

    The kernel computes the masked differences on the owned entries of the output from the input with its ghost
    regions, as the composed operators do (see ``fused_derivatives_cuda.cu``). It is used for device data on a GPU; the
    composed operator is applied in every other case (host data, cunumpy's fake CuPy, polar or other vectors).

    Parameters
    ----------
    op : ComposedLinearOperator
        The composed operator, applied when the kernel does not apply.
    b_out, b_in : BoundaryOperator
        The boundary operators on the output and input side.
    derivatives : list[DirectionalDerivativeOperator]
        The derivatives per component (of the output, or of the input if transposed).
    transposed : bool
        Whether ``op`` is the transpose of a gradient (1-forms to 0-forms).
    """

    def __init__(self, op, b_out, derivatives, b_in, transposed: bool):
        self._op = op
        self._b_out = b_out
        self._b_in = b_in
        self._derivatives = derivatives
        self._transposed = transposed
        self._meta = None
        self._input_masks = None

    @property
    def operator(self) -> LinearOperator:
        """The composed operator."""
        return self._op

    @property
    def domain(self):
        return self._op.domain

    @property
    def codomain(self):
        return self._op.codomain

    @property
    def dtype(self):
        return self._op.dtype

    def tosparse(self):
        return self._op.tosparse()

    def toarray(self):
        return self._op.toarray()

    def transpose(self, conjugate=False):
        return fuse_boundary_gradient(self._op.transpose(conjugate=conjugate))

    def dot(self, v, out=None):
        if self._use_kernel(v):
            return self._dot_fused(v, out)
        return self._op.dot(v, out=out)

    def _use_kernel(self, v) -> bool:
        """Whether :meth:`dot` uses the kernel: device data on a real GPU (not cunumpy's fake CuPy)."""
        from cunumpy.kernel_testing import fake_cupy_active

        blocks = v.blocks if isinstance(v, BlockVector) else [v]
        if not all(isinstance(b, StencilVector) and xp.is_gpu(b._data) for b in blocks):
            return False
        return not fake_cupy_active()

    def _metadata(self):
        """Owned region, direction and sign of each derivative, as the kernels take them (device array, cached)."""
        if self._meta is None:
            rows = []
            for d in self._derivatives:
                # the owned entries of the derivative's codomain, the same padded indices in its domain
                idslice = d._idslice
                rows.append(
                    [s.start for s in idslice] + [s.stop - s.start for s in idslice] + [d.diffdir, int(d.negative)]
                )
            assert all(len(r) == _META for r in rows)
            self._meta = xp.asarray(np.array(rows, dtype=np.int64).ravel())
        return self._meta

    def _input_mask_arrays(self):
        """The 0/1 mask of the input boundary operator per block, with its ghost regions exchanged (cached).

        The composed operators mask the input and then exchange the ghost regions of the masked vector (the boundary
        operator leaves copies of boundary coefficients in the ghost regions unmasked). The kernels read the input with
        its ghost regions exchanged instead, times this mask, whose ghost entries are then those of the owners: the
        products are the exchanged values of the masked input, in serial and in parallel.
        """
        if self._input_masks is None:
            ones = self.domain.zeros()
            for b in getattr(ones, "blocks", [ones]):
                b._data[...] = 1.0
            ones.ghost_regions_in_sync = True
            mask = self._b_in.dot(ones)
            mask.update_ghost_regions()
            self._input_masks = [b._data for b in getattr(mask, "blocks", [mask])]
        return self._input_masks

    def _dot_fused(self, v, out=None):
        """The product with the kernels (device data)."""
        if out is None:
            out = self.codomain.zeros()
        masks_out = self._b_out._device_masks(out)
        if masks_out is None:
            return self._op.dot(v, out=out)
        masks_in = self._input_mask_arrays()

        # the kernels read the ghost regions of the input
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        forward, transposed = _kernels()
        meta = self._metadata()
        n_comp = len(self._derivatives)

        def three(arrays):
            return [arrays[min(k, len(arrays) - 1)] for k in range(3)]

        if not self._transposed:
            outs = three([b._data for b in out.blocks])
            m_out = three([m for m, _ in masks_out])
            n_threads = sum(int(np.prod([s.stop - s.start for s in d._idslice])) for d in self._derivatives)
            forward(v._data, masks_in[0], *outs, *m_out, meta, n_comp, n_threads=n_threads)
        else:
            ys = three([b._data for b in v.blocks])
            m_in = three(masks_in)
            n_threads = int(np.prod([s.stop - s.start for s in self._derivatives[0]._idslice]))
            transposed(*ys, *m_in, out._data, masks_out[0][0], meta, n_comp, n_threads=n_threads)

        # as the composed operators: the ghost regions of the output are not written
        out.ghost_regions_in_sync = False
        return out
