"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: mat12, mat13, mat23, vec1, vec2, vec3
OUTPUTS = tuple(range(3, 9))

gk_drift_coupling = Kernel.from_folder(
    __name__, structs=CUDA_STRUCTS, missing_cuda="fallback", host_options={"outputs": OUTPUTS}
)
