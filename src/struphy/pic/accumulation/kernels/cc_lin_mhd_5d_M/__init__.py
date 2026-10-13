"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: vec1, vec2, vec3
OUTPUTS = (9, 10, 11)

cc_lin_mhd_5d_M = Kernel.from_folder(__name__, structs=CUDA_STRUCTS, host_options={"outputs": OUTPUTS})
