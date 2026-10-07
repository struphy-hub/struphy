"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

cc_lin_mhd_5d_gradB_dg_init = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)
