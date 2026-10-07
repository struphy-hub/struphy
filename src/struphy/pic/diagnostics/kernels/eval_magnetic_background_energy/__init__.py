"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

eval_magnetic_background_energy = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)
