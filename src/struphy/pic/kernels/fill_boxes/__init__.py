"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported.

No CUDA version yet (#694): on the CuPy backend the pyccel kernel runs on host copies of the arrays (a warning is
emitted once), and only the arrays in ``OUTPUTS`` are copied back to the device.
"""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: boxes, next_index
OUTPUTS = (2, 3)

fill_boxes = Kernel.from_folder(
    __name__, structs=CUDA_STRUCTS, missing_cuda="fallback", host_options={"outputs": OUTPUTS}
)
