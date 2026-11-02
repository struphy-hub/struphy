"""The kernel of this folder: pyccel in ``kernel_3d_mat_kernels.py``, CUDA in ``kernel_3d_mat_cuda.cu``."""

from cunumpy.kernels import Kernel

from struphy.utils.cuda_arguments import CUDA_STRUCTS

# the arguments the kernel writes to, by position: data
OUTPUTS = (25,)


def _n_threads(args):
    """One thread per (element, codomain basis function, domain basis function), see kernel_3d_mat_cuda.cu."""
    n_elements = args[0].size * args[1].size * args[2].size
    n_pairs = 1
    for degree in args[3:9]:  # pi1, pi2, pi3, pj1, pj2, pj3
        n_pairs *= int(degree) + 1
    return n_elements * n_pairs


kernel_3d_mat = Kernel.from_folder(
    __name__, structs=CUDA_STRUCTS, host_options={"outputs": OUTPUTS}, n_threads_from=_n_threads
)
