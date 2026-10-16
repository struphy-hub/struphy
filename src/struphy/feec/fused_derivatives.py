"""The gradient with its boundary operators, ``B1 G B0^T``, and its transpose as one CUDA kernel each (CuPy backend).

``derham.grad`` is the composition of a boundary operator (a 0/1 mask), the derivative operators (one subtraction per
component) and another boundary operator; its transpose also adds the three components. Applied one by one, that is
up to ten kernels and about twenty passes over the vectors per product, for a few arithmetic operations per entry.
:func:`fuse_boundary_gradient` returns an operator that does the same in one kernel: each output entry is computed from
the (masked) input entries it depends on, with the same operations in the same order, so the results are the same bit
for bit.

The masks are the 0/1 masks of the boundary operators (:meth:`~struphy.feec.linear_operators.BoundaryOperator.dot`
applied to a vector of ones). Their ghost regions are updated once: the original applies the input mask to a temporary
whose ghost regions are then exchanged if the mask changed entries other processes hold as ghosts, so a ghost entry
is masked with the mask of the process that owns it, as here.
"""

import cunumpy as xp
import numpy as np
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.linalg.basic import ComposedLinearOperator, IdentityOperator, LinearOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.linear_operators import BoundaryOperator

_SRC = r"""
// per component c, P[16 * c + k]: ext0, ext1, ext2, out_base, os0, os1, os2, a_base, b_base, vs0, vs1, vs2, negative,
// first flat index of the component (grad) or unused (grad.T)
// ptrs[4 * c + k]: out, out_mask (0: none), in, in_mask (0: none)

__device__ __forceinline__ double diff_term(const long long* p, const unsigned long long* q, long long i0, long long i1,
                                            long long i2)
{
    const double* in = (const double*)q[2];
    const double* m = (const double*)q[3];
    long long a = p[7] + i0 * p[9] + i1 * p[10] + i2 * p[11];
    long long b = p[8] + i0 * p[9] + i1 * p[10] + i2 * p[11];
    double va = in[a], vb = in[b];
    if (m) { va = va * m[a]; vb = vb * m[b]; }
    return p[12] ? vb - va : va - vb;
}

extern "C" __global__ void fused_grad(const long long* __restrict__ P, const unsigned long long* __restrict__ ptrs,
                                      int n_comp, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = 0;
    while (c < n_comp - 1 && i >= P[16 * (c + 1) + 13]) ++c;
    const long long* p = P + 16 * c;
    const unsigned long long* q = ptrs + 4 * c;
    long long j = i - p[13];
    long long i2 = j % p[2]; j /= p[2];
    long long i1 = j % p[1]; long long i0 = j / p[1];
    double d = diff_term(p, q, i0, i1, i2);
    const double* mo = (const double*)q[1];
    long long o = p[3] + i0 * p[4] + i1 * p[5] + i2 * p[6];
    if (mo) d = d * mo[o];
    ((double*)q[0])[o] = d;
}

extern "C" __global__ void fused_grad_t(const long long* __restrict__ P, const unsigned long long* __restrict__ ptrs,
                                        int n_comp, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // the output region is the same for all components (that of the codomain)
    long long j = i;
    long long i2 = j % P[2]; j /= P[2];
    long long i1 = j % P[1]; long long i0 = j / P[1];
    double acc = diff_term(P, ptrs, i0, i1, i2);
    for (int c = 1; c < n_comp; ++c) acc = acc + diff_term(P + 16 * c, ptrs + 4 * c, i0, i1, i2);
    const double* mo = (const double*)ptrs[1];
    long long o = P[3] + i0 * P[4] + i1 * P[5] + i2 * P[6];
    if (mo) acc = acc * mo[o];
    ((double*)ptrs[0])[o] = acc;
}
"""


def _kernels():
    import cupy as cp

    if not hasattr(_kernels, "module"):
        _kernels.module = cp.RawModule(code=_SRC)
    return _kernels.module.get_function("fused_grad"), _kernels.module.get_function("fused_grad_t")


def _blocks(v):
    return list(v.blocks) if isinstance(v, BlockVector) else [v]


def _mask(op, space):
    """The 0/1 mask arrays (per block, ghost regions of the owners) of a boundary operator, or None for an identity."""
    if isinstance(op, IdentityOperator):
        return None
    ones = space.zeros()
    for b in _blocks(ones):
        b._data[...] = 1.0
    ones.ghost_regions_in_sync = True
    probe = space.zeros()
    op.dot(ones, out=probe)
    probe.update_ghost_regions()
    return [b._data.copy() for b in _blocks(probe)]


def _derivative_params(d: DirectionalDerivativeOperator, in_data, out_data):
    """The 16 parameters of one derivative ``out[id] = in[diff] - in[id]`` (or negated) for the kernels."""
    cod = d.codomain
    idslice = [slice(pad * m, e - s + 1 + pad * m) for pad, s, e, m in zip(cod.pads, cod.starts, cod.ends, cod.shifts)]
    k = d._diffdir
    pad, s, e, m = cod.pads[k], cod.starts[k], cod.ends[k], cod.shifts[k]
    if d._transposed:
        diff_k = slice(m * pad - 1, e - s + 1 + pad * m - 1)
    else:
        diff_k = slice(m * pad + 1, e - s + 1 + m * pad + 1)
    diffslice = [diff_k if i == k else idslice[i] for i in range(3)]
    ext = [sl.stop - sl.start for sl in idslice]
    os_ = [st // out_data.itemsize for st in out_data.strides]
    vs = [st // in_data.itemsize for st in in_data.strides]
    out_base = sum(sl.start * st for sl, st in zip(idslice, os_))
    a_base = sum(sl.start * st for sl, st in zip(diffslice, vs))
    b_base = sum(sl.start * st for sl, st in zip(idslice, vs))
    return [*ext, out_base, *os_, a_base, b_base, *vs, int(d._negative), 0, 0, 0]


class FusedBoundaryGradient(LinearOperator):
    """``B_out G B_in`` (``G`` the 3x1 gradient or its 1x3 transpose of directional derivatives) in one kernel.

    Use :func:`fuse_boundary_gradient`. Same domain, codomain and results as the composed operator.
    """

    def __init__(self, composed: ComposedLinearOperator, B_out, G: BlockLinearOperator, B_in):
        self._composed = composed
        self._G = G
        self._transposed_grad = G.n_block_rows == 1
        self._derivs = (
            [G[i, 0] for i in range(G.n_block_rows)]
            if not self._transposed_grad
            else [G[0, j] for j in range(G.n_block_cols)]
        )
        self._mask_in = _mask(B_in, G.domain)
        self._mask_out = _mask(B_out, G.codomain)
        self._cache = {}

    @property
    def domain(self):
        return self._composed.domain

    @property
    def codomain(self):
        return self._composed.codomain

    @property
    def dtype(self):
        return self._composed.dtype

    def tosparse(self):
        return self._composed.tosparse()

    def toarray(self):
        return self._composed.toarray()

    def transpose(self, conjugate=False):
        return fuse_boundary_gradient(self._composed.transpose(conjugate=conjugate))

    def _setup(self, v, out):
        """Parameter and pointer tables for these vectors (cached by their data pointers)."""
        in_blocks, out_blocks = _blocks(v), _blocks(out)
        key = tuple(int(b._data.data.ptr) for b in in_blocks + out_blocks)
        entry = self._cache.get(key)
        if entry is not None:
            return entry
        P, ptrs = [], []
        if self._transposed_grad:
            o = out_blocks[0]._data
            mo = self._mask_out[0] if self._mask_out is not None else None
            for c, d in enumerate(self._derivs):
                x = in_blocks[c]._data
                P += _derivative_params(d, x, o)
                mi = self._mask_in[c] if self._mask_in is not None else None
                ptrs += [
                    o.data.ptr,
                    mo.data.ptr if mo is not None else 0,
                    x.data.ptr,
                    mi.data.ptr if mi is not None else 0,
                ]
            total = int(np.prod(P[:3]))
        else:
            x = in_blocks[0]._data
            mi = self._mask_in[0] if self._mask_in is not None else None
            first = 0
            for c, d in enumerate(self._derivs):
                o = out_blocks[c]._data
                p = _derivative_params(d, x, o)
                p[13] = first
                first += int(np.prod(p[:3]))
                P += p
                mo = self._mask_out[c] if self._mask_out is not None else None
                ptrs += [
                    o.data.ptr,
                    mo.data.ptr if mo is not None else 0,
                    x.data.ptr,
                    mi.data.ptr if mi is not None else 0,
                ]
            total = first
        entry = (xp.asarray(np.array(P, dtype=np.int64)), xp.asarray(np.array(ptrs, dtype=np.uint64)), total)
        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[key] = entry
        return entry

    def dot(self, v, out=None):
        assert v.space is self.domain
        if out is None:
            out = self.codomain.zeros()
        assert out.space is self.codomain
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()
        P, ptrs, total = self._setup(v, out)
        grad, grad_t = _kernels()
        kernel = grad_t if self._transposed_grad else grad
        if total:
            kernel(((total + 255) // 256,), (256,), (P, ptrs, np.int32(len(self._derivs)), np.int64(total)))
        out.ghost_regions_in_sync = False
        return out


def fuse_boundary_gradient(op):
    """A :class:`FusedBoundaryGradient` for ``op``, or ``op`` itself if it is not of that form or not on the GPU.

    ``op`` must be ``B_out @ G @ B_in`` with boundary (or identity) operators ``B_out``, ``B_in`` (each may be left
    out) and ``G`` the 3x1 gradient (or 1x3 transpose) of
    :class:`~feectools.feec.derivatives.DirectionalDerivativeOperator` blocks on C-contiguous float64 device arrays,
    as ``derham.grad`` and ``derham.grad.T`` without polar splines.
    """
    factors = list(op.multiplicands) if isinstance(op, ComposedLinearOperator) else [op]
    G_pos = [k for k, a in enumerate(factors) if isinstance(a, BlockLinearOperator)]
    if len(G_pos) != 1 or len(factors) > 3:
        return op
    k = G_pos[0]
    G = factors[k]
    outer, inner = factors[:k], factors[k + 1 :]
    if len(outer) > 1 or len(inner) > 1:
        return op
    B_out = outer[0] if outer else IdentityOperator(G.codomain)
    B_in = inner[0] if inner else IdentityOperator(G.domain)
    if not all(isinstance(B, (BoundaryOperator, IdentityOperator)) for B in (B_out, B_in)):
        return op
    if not isinstance(G, BlockLinearOperator) or (G.n_block_rows, G.n_block_cols) not in ((3, 1), (1, 3)):
        return op
    blocks = [G[i, j] for i in range(G.n_block_rows) for j in range(G.n_block_cols)]
    if not all(isinstance(d, DirectionalDerivativeOperator) and d.domain.ndim == 3 for d in blocks):
        return op
    probe = G.domain.zeros()
    if not all(xp.is_gpu(b._data) and b._data.dtype == np.float64 for b in _blocks(probe)):
        return op
    if not all(isinstance(b, StencilVector) for b in _blocks(probe) + _blocks(G.codomain.zeros())):
        return op
    return FusedBoundaryGradient(op, B_out, G, B_in)
