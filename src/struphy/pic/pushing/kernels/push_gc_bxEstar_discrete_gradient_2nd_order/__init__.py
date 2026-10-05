"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

push_gc_bxEstar_discrete_gradient_2nd_order = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)
