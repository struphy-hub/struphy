"""Transform the pyccelized kernel argument classes into arguments for CUDA kernels.

Pyccel kernels take the argument classes of :mod:`struphy.kernel_arguments.pusher_args_kernels`
as arguments. A ``cupy.RawKernel`` can only take pointers and scalars, hence :func:`transform`
collects the attributes of such a class into a :class:`CudaArguments` object holding CuPy arrays.

The transform is meant to be done **once** at setup (not at every kernel call). Arrays that are
already CuPy arrays are used as they are (no copy); NumPy arrays are copied to the device once,
after which the device copy is the one updated by CUDA kernels.

The field order of each :class:`CudaArguments` subclass defines the corresponding part
of the CUDA kernel signature (``double*`` for float arrays, ``long long*`` for int arrays,
``bool*`` for bool arrays, ``int`` for int scalars).
"""

from dataclasses import dataclass, fields
from functools import singledispatch
from typing import Any

import numpy as np

from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments


@dataclass(frozen=True)
class CudaArguments:
    """Base class for CUDA kernel arguments; :attr:`values` is the flat tuple passed to the kernel."""

    def __post_init__(self):
        object.__setattr__(self, "_values", tuple(getattr(self, f.name) for f in fields(self)))

    @property
    def values(self) -> tuple:
        """Flat CUDA kernel arguments, in field order."""
        return self._values

    @property
    def n_threads(self) -> int | None:
        """Number of CUDA threads needed for this argument, None if no preference."""
        return None


@dataclass(frozen=True)
class CudaMarkerArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments`."""

    markers: Any
    valid_mks: Any
    n_markers: np.int32
    n_cols: np.int32
    Np: np.int32
    vdim: np.int32
    weight_idx: np.int32
    first_diagnostics_idx: np.int32
    first_init_idx: np.int32
    first_shift_idx: np.int32
    residual_idx: np.int32
    first_free_idx: np.int32
    mu_idx: np.int32
    bc_type: Any

    @property
    def n_threads(self) -> int:
        return int(self.n_markers)


@dataclass(frozen=True)
class CudaDomainArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DomainArguments`."""

    kind_map: np.int32
    params: Any
    degree: Any
    t1: Any
    t2: Any
    t3: Any
    ind1: Any
    ind2: Any
    ind3: Any
    cx: Any
    cy: Any
    cz: Any


@dataclass(frozen=True)
class CudaDerhamArguments(CudaArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments`."""

    pn: Any
    tn1: Any
    tn2: Any
    tn3: Any
    starts: Any


def _cupy(arr, dtype):
    """C-contiguous CuPy array of ``arr`` (no copy if it already is one)."""
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(arr, dtype=dtype))


@singledispatch
def transform(args: Any) -> CudaArguments:
    """Transform a pyccelized kernel argument class into its :class:`CudaArguments` counterpart."""
    raise TypeError(f"No CUDA transform for arguments of type {type(args)}.")


@transform.register
def _(args: MarkerArguments) -> CudaMarkerArguments:
    return CudaMarkerArguments(
        markers=_cupy(args.markers, np.float64),
        valid_mks=_cupy(args.valid_mks, np.bool_),
        n_markers=np.int32(args.n_markers),
        n_cols=np.int32(args.markers.shape[1]),
        Np=np.int32(args.Np),
        vdim=np.int32(args.vdim),
        weight_idx=np.int32(args.weight_idx),
        first_diagnostics_idx=np.int32(args.first_diagnostics_idx),
        first_init_idx=np.int32(args.first_init_idx),
        first_shift_idx=np.int32(args.first_shift_idx),
        residual_idx=np.int32(args.residual_idx),
        first_free_idx=np.int32(args.first_free_idx),
        mu_idx=np.int32(args.mu_idx),
        bc_type=_cupy(args.bc_type, np.int64),
    )


@transform.register
def _(args: DomainArguments) -> CudaDomainArguments:
    return CudaDomainArguments(
        kind_map=np.int32(args.kind_map),
        params=_cupy(args.params, np.float64),
        degree=_cupy(args.degree, np.int64),
        t1=_cupy(args.t1, np.float64),
        t2=_cupy(args.t2, np.float64),
        t3=_cupy(args.t3, np.float64),
        ind1=_cupy(args.ind1, np.int64),
        ind2=_cupy(args.ind2, np.int64),
        ind3=_cupy(args.ind3, np.int64),
        cx=_cupy(args.cx, np.float64),
        cy=_cupy(args.cy, np.float64),
        cz=_cupy(args.cz, np.float64),
    )


@transform.register
def _(args: DerhamArguments) -> CudaDerhamArguments:
    return CudaDerhamArguments(
        pn=_cupy(args.pn, np.int64),
        tn1=_cupy(args.tn1, np.float64),
        tn2=_cupy(args.tn2, np.float64),
        tn3=_cupy(args.tn3, np.float64),
        starts=_cupy(args.starts, np.int64),
    )
