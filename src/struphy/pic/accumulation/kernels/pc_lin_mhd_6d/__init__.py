"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: mat11_11, mat12_11, mat13_11, mat22_11, mat23_11, mat33_11,
# mat11_12, mat12_12, mat13_12, mat22_12, mat23_12, mat33_12, mat11_22, mat12_22, mat13_22, mat22_22, mat23_22,
# mat33_22, vec1_1, vec2_1, vec3_1, vec1_2, vec2_2, vec3_2
OUTPUTS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 39, 40, 41, 42, 43, 44)

pc_lin_mhd_6d = Kernel.from_folder(__name__, structs=CUDA_STRUCTS, host_options={"outputs": OUTPUTS})
