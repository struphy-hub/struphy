"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: values
OUTPUTS = (8,)

eval_spline_mpi_markers = Kernel.from_folder(__name__, structs=CUDA_STRUCTS, host_options={"outputs": OUTPUTS})
