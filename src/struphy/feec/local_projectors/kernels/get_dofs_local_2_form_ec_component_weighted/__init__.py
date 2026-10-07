"""The kernel of this folder: pyccel in ``<name>_kernels.py``, CUDA in ``<name>_cuda.cu`` once ported."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

get_dofs_local_2_form_ec_component_weighted = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)
