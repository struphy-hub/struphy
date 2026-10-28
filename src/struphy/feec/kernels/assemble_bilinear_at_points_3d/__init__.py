"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported.

No CUDA version yet: on the CuPy backend the pyccel kernel runs on host copies of the arrays (a warning is
emitted once).
"""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: mat
OUTPUTS = (0,)

assemble_bilinear_at_points_3d = Kernel.from_folder(
    __name__, structs=CUDA_STRUCTS, missing_cuda="fallback", host_options={"outputs": OUTPUTS}
)
