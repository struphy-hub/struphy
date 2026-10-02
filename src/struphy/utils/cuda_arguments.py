"""CUDA counterparts of the pyccelized kernel argument classes.

The compiled classes in :mod:`struphy.kernel_arguments.pusher_args_kernels` only accept NumPy
arrays. The classes here take the same constructor arguments, keep references to **CuPy** arrays
(no copies, other arrays raise) and pack them into one C struct each, which CUDA kernels take by value.
The structs are declared in ``struphy/kernel_arguments/pusher_args.cuh``; :attr:`Argument.fields` lists
their members in declaration order, with the C types.
"""

import operator
from abc import ABC

import numpy as np

# NumPy types of the struct members, by C type; pointers are passed as device addresses
C_TYPES = {
    "int": np.int32,
    "double": np.float64,
    "double*": np.uint64,
    "bool*": np.uint64,
    "long long*": np.uint64,
}


class Argument(ABC):
    """Base class for objects that are passed to a CUDA kernel as one C struct.

    A subclass names the struct (:attr:`struct_name`), lists its members (:attr:`fields`) and stores each member
    as an attribute of the same name, then calls :meth:`_pack` at the end of its constructor. The struct is packed
    once; kernel calls pass it as it is.
    """

    struct_name: str
    """Name of the C struct in ``pusher_args.cuh``."""

    fields: tuple[tuple[str, str], ...]
    """``(C type, name)`` of each struct member, in declaration order."""

    @classmethod
    def struct_dtype(cls) -> np.dtype:
        """NumPy dtype with the memory layout of the C struct (C alignment and padding).

        Returns
        -------
        numpy.dtype
            Structured dtype, one field per struct member.
        """
        return np.dtype(
            {"names": [name for _, name in cls.fields], "formats": [C_TYPES[ctype] for ctype, _ in cls.fields]},
            align=True,
        )

    def _pack(self):
        """Pack the members into the struct; pointer members hold the device address of the array attribute.

        Scalars are checked against the C type of their member: a non-integer for an ``int`` or a value that
        does not fit raises, instead of arriving in the kernel truncated or wrapped around.
        """
        struct = np.zeros((), dtype=self.struct_dtype())
        for ctype, name in self.fields:
            value = getattr(self, name)
            if ctype.endswith("*"):
                struct[name] = value.data.ptr
            elif ctype == "int":
                value = operator.index(value)  # raises TypeError for floats
                info = np.iinfo(C_TYPES[ctype])
                if not info.min <= value <= info.max:
                    raise OverflowError(f"{name} = {value} does not fit into a C {ctype}.")
                struct[name] = value
            else:
                struct[name] = float(value)
        self._struct = struct[()]

    def __getstate__(self):
        # the struct holds device addresses, which are not valid for the arrays of a copy
        state = self.__dict__.copy()
        state.pop("_struct", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pack()

    def get_cuda_args(self) -> tuple:
        """Return this object's arguments in CUDA kernel signature order: the packed struct.

        Returns
        -------
        tuple
            One ``numpy.void`` with the bytes of the C struct.
        """
        return (self._struct,)


def _cupy_array(name: str, arr, dtype):
    """Check that ``arr`` is a C-contiguous CuPy array of ``dtype`` (never converts or copies).

    Parameters
    ----------
    name : str
        Name of the argument, for error messages.

    arr : cupy.ndarray
        The array to check.

    dtype : type
        Expected dtype.

    Returns
    -------
    cupy.ndarray
        ``arr`` itself.
    """
    if not hasattr(arr, "__cuda_array_interface__"):
        raise TypeError(f"{name} must be a CuPy array, got {type(arr)}.")
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous array of dtype {np.dtype(dtype)}.")
    return arr


class CudaMarkerArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.MarkerArguments`.

    Passed to CUDA kernels as ``struct MarkerArgs``, see :attr:`fields`.

    Parameters
    ----------
    markers : cupy.ndarray[float]
        Markers array (C-contiguous, float64).

    valid_mks : cupy.ndarray[bool]
        True for valid markers (not holes or ghosts).

    Np : int
        Total number of particles.

    vdim : int
        Dimension of velocity space.

    weight_idx : int
        Column index of particle weight.

    first_diagnostics_idx : int
        Starting index for diagnostics columns.

    first_pusher_idx : int
        Starting buffer marker index number for pusher.

    first_shift_idx : int
        First index for storing shifts due to boundary conditions in eta-space.

    residual_idx : int
        Column for storing the residual in iterative pushers.

    first_free_idx : int
        First index for storing auxiliary quantities for each particle.

    mu_idx : int
        Column index of particle magnetic moment.

    bc_type : cupy.ndarray[int]
        Kinetic boundary condition in each logical direction (int64).

    Attributes
    ----------
    markers : cupy.ndarray[float]
        The markers array passed in (no copy).

    valid_mks : cupy.ndarray[bool]
        The array of valid markers passed in (no copy).

    n_markers : int
        Number of rows of ``markers``, e.g. for the number of CUDA threads.
    """

    struct_name = "MarkerArgs"
    fields = (
        ("double*", "markers"),
        ("bool*", "valid_mks"),
        ("int", "n_markers"),
        ("int", "n_cols"),
        ("int", "Np"),
        ("int", "vdim"),
        ("int", "weight_idx"),
        ("int", "first_diagnostics_idx"),
        ("int", "first_init_idx"),
        ("int", "first_shift_idx"),
        ("int", "residual_idx"),
        ("int", "first_free_idx"),
        ("int", "mu_idx"),
        ("long long*", "bc_type"),
    )

    def __init__(
        self,
        markers,
        valid_mks,
        Np: int,
        vdim: int,
        weight_idx: int,
        first_diagnostics_idx: int,
        first_pusher_idx: int,
        first_shift_idx: int,
        residual_idx: int,
        first_free_idx: int,
        mu_idx: int,
        bc_type,
    ):
        self.markers = _cupy_array("markers", markers, np.float64)
        self.valid_mks = _cupy_array("valid_mks", valid_mks, np.bool_)
        self.n_markers = markers.shape[0]
        self.n_cols = markers.shape[1]
        self.Np = Np
        self.vdim = vdim
        self.weight_idx = weight_idx
        self.first_diagnostics_idx = first_diagnostics_idx
        self.first_init_idx = first_pusher_idx
        self.first_shift_idx = first_shift_idx
        self.residual_idx = residual_idx
        self.first_free_idx = first_free_idx
        self.mu_idx = mu_idx
        self.bc_type = _cupy_array("bc_type", bc_type, np.int64)
        self._pack()


class CudaDerhamArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments`.

    Passed to CUDA kernels as ``struct DerhamArgs``, see :attr:`fields`. The scratch arrays of the pyccel class
    (``bn1``, ..., ``bd3``) are not part of it; CUDA kernels use per-thread local arrays instead.

    Parameters
    ----------
    pn : cupy.ndarray[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham` (int64).

    tn1, tn2, tn3 : cupy.ndarray[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.

    starts : cupy.ndarray[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham` (int64).
    """

    struct_name = "DerhamArgs"
    fields = (
        ("long long*", "pn"),
        ("double*", "tn1"),
        ("double*", "tn2"),
        ("double*", "tn3"),
        ("long long*", "starts"),
    )

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _cupy_array("pn", pn, np.int64)
        self.tn1, self.tn2, self.tn3 = (_cupy_array("tn", t, np.float64) for t in (tn1, tn2, tn3))
        self.starts = _cupy_array("starts", starts, np.int64)
        self._pack()


class CudaDerhamArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DerhamArguments`.

    CUDA signature of :meth:`get_cuda_args`: ``long long* pn, double* tn1, double* tn2, double* tn3, long long* starts``

    The scratch arrays of the pyccel class (``bn1``, ..., ``bd3``) are not part of it; CUDA kernels use
    per-thread local arrays instead.

    Parameters
    ----------
    pn : cupy.ndarray[int]
        Spline degrees of :class:`~struphy.feec.psydac_derham.Derham` (int64).

    tn1, tn2, tn3 : cupy.ndarray[float]
        Knot sequences of :class:`~struphy.feec.psydac_derham.Derham`.

    starts : cupy.ndarray[int]
        Start indices (current MPI process) of :class:`~struphy.feec.psydac_derham.Derham` (int64).
    """

    def __init__(self, pn, tn1, tn2, tn3, starts):
        self.pn = _cupy_array("pn", pn, np.int64)
        self.tn1, self.tn2, self.tn3 = (_cupy_array("tn", t, np.float64) for t in (tn1, tn2, tn3))
        self.starts = _cupy_array("starts", starts, np.int64)

    def get_cuda_args(self) -> tuple:
        return (self.pn, self.tn1, self.tn2, self.tn3, self.starts)


class CudaDomainArguments(Argument):
    """CUDA version of :class:`~struphy.kernel_arguments.pusher_args_kernels.DomainArguments`.

    Passed to CUDA kernels as ``struct DomainArgs``, see :attr:`fields`.

    Parameters
    ----------
    kind_map : int
        Mapping identifier of :class:`~struphy.geometry.base.Domain`.

    params : cupy.ndarray[float]
        Mapping parameters.

    degree : cupy.ndarray[int]
        Spline degrees of the mapping.

    t1, t2, t3 : cupy.ndarray[float]
        Knot sequences of the mapping.

    ind1, ind2, ind3 : cupy.ndarray[int]
        Indices of non-vanishing splines in format (number of mapping grid cells, degree + 1).

    cx, cy, cz : cupy.ndarray[float]
        Spline coefficients (control points) of the mapping.
    """

    struct_name = "DomainArgs"
    fields = (
        ("int", "kind_map"),
        ("double*", "params"),
        ("long long*", "degree"),
        ("double*", "t1"),
        ("double*", "t2"),
        ("double*", "t3"),
        ("long long*", "ind1"),
        ("long long*", "ind2"),
        ("long long*", "ind3"),
        ("double*", "cx"),
        ("double*", "cy"),
        ("double*", "cz"),
    )

    def __init__(self, kind_map: int, params, degree, t1, t2, t3, ind1, ind2, ind3, cx, cy, cz):
        self.kind_map = kind_map
        self.params = _cupy_array("params", params, np.float64)
        self.degree = _cupy_array("degree", degree, np.int64)
        self.t1, self.t2, self.t3 = (_cupy_array("t", t, np.float64) for t in (t1, t2, t3))
        self.ind1, self.ind2, self.ind3 = (_cupy_array("ind", ind, np.int64) for ind in (ind1, ind2, ind3))
        self.cx, self.cy, self.cz = (_cupy_array("c", c, np.float64) for c in (cx, cy, cz))
        self._pack()
