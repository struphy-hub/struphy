"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported.

No CUDA version yet: on the CuPy backend the pyccel kernel runs on host copies of the arrays (a warning is
emitted once), as before the move into this folder.
"""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

kernel_3d_diag = Kernel.from_folder(__name__, structs=CUDA_STRUCTS, missing_cuda="fallback")
