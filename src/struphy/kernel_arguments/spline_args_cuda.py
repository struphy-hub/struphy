"""CUDA version of :class:`~struphy.kernel_arguments.spline_args_kernels.SplineArguments`.

Takes the same constructor arguments and has the same attributes as the pyccel class, but holds CuPy arrays and
is passed to CUDA kernels as one C struct (``kernel_arguments/spline_args.cuh``, generated from
:attr:`CudaSplineArguments.fields`).
"""

import numpy as np
from cunumpy.arguments import CudaStructArguments

from struphy.kernel_arguments.pusher_args_cuda import _device_array


class CudaSplineArguments(CudaStructArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.spline_args_kernels.SplineArguments` (``SplineArgs``).

    The knot vectors are array views (pointer, shape, strides), because ``find_span`` needs the number of knots.
    """

    struct_name = "SplineArgs"
    fields = (
        ("kind", "long long*"),
        ("pn", "long long*"),
        ("tn1", "Array1D<double>"),
        ("tn2", "Array1D<double>"),
        ("tn3", "Array1D<double>"),
        ("starts", "long long*"),
    )

    def __init__(self, kind, pn, tn1, tn2, tn3, starts):
        self.kind = _device_array("kind", kind, np.int64)
        self.pn = _device_array("pn", pn, np.int64)
        if bool(((pn < 1) | (pn > 8)).any()):
            raise ValueError("CUDA spline degrees must be between 1 and 8.")
        self.tn1, self.tn2, self.tn3 = (_device_array("tn", t, np.float64, ndim=1) for t in (tn1, tn2, tn3))
        self.starts = _device_array("starts", starts, np.int64)
        self.pack()
