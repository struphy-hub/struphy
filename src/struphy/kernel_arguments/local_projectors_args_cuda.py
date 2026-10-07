"""CUDA version of :class:`~struphy.kernel_arguments.local_projectors_args_kernels.LocalProjectorsArguments`.

Takes the same constructor arguments and has the same attributes as the pyccel class, but holds CuPy arrays and
is passed to CUDA kernels as one C struct (``kernel_arguments/local_projectors_args.cuh``, generated from
:attr:`CudaLocalProjectorsArguments.fields`). No CUDA kernel uses it yet: local projectors are not supported on
the CuPy backend.
"""

import numpy as np
from cunumpy.cuda import CudaStructArguments

from struphy.kernel_arguments.pusher_args_cuda import _device_array


class CudaLocalProjectorsArguments(CudaStructArguments):
    """CUDA version of :class:`~struphy.kernel_arguments.local_projectors_args_kernels.LocalProjectorsArguments`."""

    struct_name = "LocalProjectorsArgs"
    fields = (
        ("space_key", "int"),
        ("IoH", "bool*"),
        ("shift", "long long*"),
        ("original_size", "long long*"),
        ("index_translation1", "long long*"),
        ("index_translation2", "long long*"),
        ("index_translation3", "long long*"),
        ("starts", "long long*"),
        ("ends", "long long*"),
        ("pds", "long long*"),
        ("B_nbasis", "long long*"),
        ("periodic", "bool*"),
        ("degree", "long long*"),
        ("wij0", "Array2D<double>"),
        ("wij1", "Array2D<double>"),
        ("wij2", "Array2D<double>"),
        ("wts0", "Array2D<double>"),
        ("wts1", "Array2D<double>"),
        ("wts2", "Array2D<double>"),
        ("inv_index_translation0", "long long*"),
        ("inv_index_translation1", "long long*"),
        ("inv_index_translation2", "long long*"),
    )

    def __init__(
        self,
        space_key: int,
        IoH,
        shift,
        original_size,
        index_translation1,
        index_translation2,
        index_translation3,
        starts,
        ends,
        pds,
        B_nbasis,
        periodic,
        degree,
        wij0,
        wij1,
        wij2,
        wts0,
        wts1,
        wts2,
        inv_index_translation0,
        inv_index_translation1,
        inv_index_translation2,
    ):
        self.space_key = space_key
        self.IoH = _device_array("IoH", IoH, np.bool_)
        self.shift = _device_array("shift", shift, np.int64)
        self.original_size = _device_array("original_size", original_size, np.int64)
        self.index_translation1 = _device_array("index_translation1", index_translation1, np.int64)
        self.index_translation2 = _device_array("index_translation2", index_translation2, np.int64)
        self.index_translation3 = _device_array("index_translation3", index_translation3, np.int64)
        self.starts = _device_array("starts", starts, np.int64)
        self.ends = _device_array("ends", ends, np.int64)
        self.pds = _device_array("pds", pds, np.int64)
        self.B_nbasis = _device_array("B_nbasis", B_nbasis, np.int64)
        self.periodic = _device_array("periodic", periodic, np.bool_)
        self.degree = _device_array("degree", degree, np.int64)
        self.wij0 = _device_array("wij0", wij0, np.float64)
        self.wij1 = _device_array("wij1", wij1, np.float64)
        self.wij2 = _device_array("wij2", wij2, np.float64)
        self.wts0 = _device_array("wts0", wts0, np.float64)
        self.wts1 = _device_array("wts1", wts1, np.float64)
        self.wts2 = _device_array("wts2", wts2, np.float64)
        self.inv_index_translation0 = _device_array("inv_index_translation0", inv_index_translation0, np.int64)
        self.inv_index_translation1 = _device_array("inv_index_translation1", inv_index_translation1, np.int64)
        self.inv_index_translation2 = _device_array("inv_index_translation2", inv_index_translation2, np.int64)
        self.pack()
