"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: vec1, vec2, vec3
OUTPUTS = (3, 4, 5)

div_u_weak_1form = Kernel.from_folder(__name__, structs=CUDA_STRUCTS, host_options={"outputs": OUTPUTS})
