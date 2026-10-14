"""Fast application of assembled mass matrices that are Kronecker products of 1D banded matrices.

A mass matrix block ``M[i, j] = int w Lambda_i Lambda_j`` of tensor-product splines factorizes as
``M = c * M_1 (x) M_2 (x) M_3`` when its weight ``w`` is constant (e.g. on a Cuboid). Applied as a
:class:`~feectools.linalg.stencil.StencilMatrix`, every row reads all ``n_1 * n_2 * n_3`` diagonals (105 for
degrees (1, 2, 3)) from memory; as a Kronecker product it is three 1D banded passes over the vector, ``n_1 + n_2 +
n_3`` operations per entry and no 3D matrix data. :func:`kronecker_apply_operator` checks each assembled block and
returns an operator for the matrix-vector products in which the blocks that are Kronecker products (to round-off)
are replaced by :class:`KroneckerBandMatrix`.

The 1D factors are read from the assembled data itself (rows owned by this process), so they hold exactly what the
assembly computed, on any domain decomposition, and the check is the assembled data, not an assumption about the
mapping or the weights.
"""

import logging
import os
from functools import cache
from pathlib import Path

import cunumpy as xp
from feectools.linalg.basic import LinearOperator
from feectools.linalg.block import BlockLinearOperator
from feectools.linalg.stencil import StencilMatrix, StencilVector

logger = logging.getLogger("struphy")

# relative tolerance of the Kronecker check: the 3D assembly and the product of the factors differ by round-off only
KRONECKER_RTOL = 1e-12


def kronecker_mass_enabled() -> bool:
    """Whether :func:`kronecker_apply_operator` may replace blocks (``STRUPHY_KRONECKER_MASS=0`` switches it off)."""
    return os.environ.get("STRUPHY_KRONECKER_MASS", "1").lower() not in ("0", "false", "no")


@cache
def _band_pass_cuda():
    """The CUDA kernel of one banded pass (compiled on first call)."""
    from cunumpy.kernels import CudaKernel
    from feectools.utilities.cached_launch import CachedLaunch

    # launched with the same buffers in every solver iteration: cache the launch preparation
    return CachedLaunch(CudaKernel.from_file(Path(__file__).with_name("kron_band_pass_cuda.cu"), "kron_band_pass"))


def _band_pass(A, inp, out, axis, offset):
    """``out[..., i, ...] = sum_d A[i, d] * inp[..., i + offset + d, ...]`` along ``axis`` (other axes unchanged)."""
    if xp.is_gpu(out):
        _band_pass_cuda()(A, inp, out, axis, offset, n_threads=out.size)
        return
    m = out.shape[axis]
    shape = [1, 1, 1]
    shape[axis] = m
    out[...] = 0.0
    for d in range(A.shape[1]):
        idx = [slice(None)] * 3
        idx[axis] = slice(offset + d, offset + d + m)
        out += A[:, d].reshape(shape) * inp[tuple(idx)]


class KroneckerBandMatrix(LinearOperator):
    """A square 3D :class:`~feectools.linalg.stencil.StencilMatrix` applied as a Kronecker product of 1D banded
    matrices, ``(A_1 (x) A_2 (x) A_3) x`` on the rows owned by this process.

    ``A_k[i, d]`` is the entry of owned row ``i`` (along direction ``k``) and diagonal ``d``, column
    ``i - q_k + d`` with ``q_k = (n_k - 1) // 2``, as in the stencil matrix. The product is computed in three passes,
    each over the owned rows of its direction and the full (padded) extent of the directions after it, so the passes
    need no communication beyond the ghost regions of the input. Use :meth:`from_stencil`.

    Parameters
    ----------
    stencil : StencilMatrix
        The assembled matrix (kept for ``toarray``, ``tosparse`` and ``transpose``).
    factors : tuple of arrays
        ``A_1, A_2, A_3``, shapes ``(m_k, n_k)``, C-contiguous backend arrays.
    """

    def __init__(self, stencil: StencilMatrix, factors):
        self._stencil = stencil
        self._factors = factors

        V = stencil.domain
        self._m = tuple(int(e - s + 1) for s, e in zip(V.starts, V.ends))
        self._pads = tuple(int(p) for p in V.pads)
        self._q = tuple((int(n) - 1) // 2 for n in stencil._data.shape[3:])

        # temporaries of the first two passes: owned rows along the passes done, padded extent along the others
        x_shape = V.zeros()._data.shape
        self._t1 = xp.empty((self._m[0], x_shape[1], x_shape[2]), dtype=stencil._data.dtype)
        self._t2 = xp.empty((self._m[0], self._m[1], x_shape[2]), dtype=stencil._data.dtype)

    @classmethod
    def from_stencil(cls, S: StencilMatrix, rtol: float = KRONECKER_RTOL):
        """The Kronecker form of ``S`` if it is one (to ``rtol``, relative to its largest entry), else None.

        Only square 3D matrices on one space (domain is codomain) with shifts 1 qualify, e.g. the diagonal blocks of
        the mass matrices. The factors are read from the owned rows of ``S``: with reference rows ``r_k`` (the first
        owned row along each direction) and central diagonals ``q_k``,
        ``A_1[i, :] = S[i, r_2, r_3, :, q_2, q_3]``, ``A_2[i, :] = S[r_1, i, r_3, q_1, :, q_3] / S[r, q]``,
        ``A_3[i, :] = S[r_1, r_2, i, q_1, q_2, :] / S[r, q]``; the product is then compared with all owned rows.
        """
        if not isinstance(S, StencilMatrix) or S.domain is not S.codomain or S.domain.ndim != 3:
            return None
        V = S.domain
        if any(int(s) != 1 for s in V.shifts) or S._data.ndim != 6:
            return None

        p = [int(p) for p in V.pads]
        m = [int(e - s + 1) for s, e in zip(V.starts, V.ends)]
        q = [(int(n) - 1) // 2 for n in S._data.shape[3:]]
        if min(m) < 1:
            return None
        data = S._data

        den = data[p[0], p[1], p[2], q[0], q[1], q[2]]
        if float(den) == 0.0:
            return None
        A1 = xp.ascontiguousarray(data[p[0] : p[0] + m[0], p[1], p[2], :, q[1], q[2]])
        A2 = xp.ascontiguousarray(data[p[0], p[1] : p[1] + m[1], p[2], q[0], :, q[2]] / den)
        A3 = xp.ascontiguousarray(data[p[0], p[1], p[2] : p[2] + m[2], q[0], q[1], :] / den)

        # compare with every owned row, one plane (row along direction 1) at a time
        owned = data[p[0] : p[0] + m[0], p[1] : p[1] + m[1], p[2] : p[2] + m[2]]
        A23 = A2[:, None, None, :, None] * A3[None, :, None, None, :]  # (m2, m3, 1, n2, n3)
        max_diff = 0.0
        max_abs = 0.0
        for i in range(m[0]):
            recon = A1[i][None, None, :, None, None] * A23
            max_diff = max(max_diff, float(xp.max(xp.abs(owned[i] - recon))))
            max_abs = max(max_abs, float(xp.max(xp.abs(owned[i]))))
        if max_abs == 0.0 or max_diff > rtol * max_abs:
            return None
        return cls(S, (A1, A2, A3))

    # ------------------------------------------------------------------
    # LinearOperator interface
    # ------------------------------------------------------------------
    @property
    def domain(self):
        return self._stencil.domain

    @property
    def codomain(self):
        return self._stencil.codomain

    @property
    def dtype(self):
        return self._stencil.dtype

    @property
    def factors(self):
        """The 1D factors ``A_1, A_2, A_3`` (rows owned by this process)."""
        return self._factors

    def tosparse(self):
        return self._stencil.tosparse()

    def toarray(self):
        return self._stencil.toarray()

    def transpose(self, conjugate=False):
        return self._stencil.transpose(conjugate=conjugate)

    def dot(self, v, out=None):
        assert isinstance(v, StencilVector)
        assert v.space is self.domain
        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain

        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        A1, A2, A3 = self._factors
        p, q, m = self._pads, self._q, self._m
        owned = out._data[p[0] : p[0] + m[0], p[1] : p[1] + m[1], p[2] : p[2] + m[2]]

        _band_pass(A1, v._data, self._t1, 0, p[0] - q[0])
        _band_pass(A2, self._t1, self._t2, 1, p[1] - q[1])
        _band_pass(A3, self._t2, owned, 2, p[2] - q[2])

        out.ghost_regions_in_sync = False
        return out


def kronecker_apply_operator(mat, name: str = ""):
    """The operator to apply ``mat`` with: ``mat`` itself, or a copy whose Kronecker blocks are replaced.

    Parameters
    ----------
    mat : StencilMatrix or BlockLinearOperator
        An assembled (block) mass matrix.
    name : str
        Name of the operator, for the log.

    Returns
    -------
    LinearOperator
        ``mat`` if no block is a Kronecker product (or :func:`kronecker_mass_enabled` is False); otherwise a
        :class:`KroneckerBandMatrix` (for a StencilMatrix) or a BlockLinearOperator with the same blocks except the
        Kronecker ones. It is valid only as long as ``mat`` is not modified.
    """
    if not kronecker_mass_enabled():
        return mat

    if isinstance(mat, StencilMatrix):
        kron = KroneckerBandMatrix.from_stencil(mat)
        if kron is not None:
            logger.info(f"Mass matrix {name}: applied as a Kronecker product of 1D banded matrices")
        return mat if kron is None else kron

    if isinstance(mat, BlockLinearOperator):
        blocks = {}
        replaced = []
        for key, block in mat._blocks.items():
            kron = KroneckerBandMatrix.from_stencil(block)
            if kron is None:
                blocks[key] = block
            else:
                blocks[key] = kron
                replaced.append(key)
        if not replaced:
            return mat
        logger.info(f"Mass matrix {name}: blocks {replaced} applied as Kronecker products of 1D banded matrices")
        return BlockLinearOperator(mat.domain, mat.codomain, blocks=blocks)

    return mat
