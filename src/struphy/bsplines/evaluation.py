"""Spline evaluation kernel pairs with identical Python arguments on both backends."""

from pathlib import Path

from cunumpy.cuda import CudaKernel
from cunumpy.kernels import Kernel, PyccelKernel

from struphy.bsplines import evaluation_kernels_3d
from struphy.utils.cuda_arguments import CUDA_OPTIONS


_SOURCE = Path(__file__).with_name("evaluate_spline_cuda.cu")


def _pair(name):
    return Kernel(
        PyccelKernel(getattr(evaluation_kernels_3d, name)),
        CudaKernel.from_file(_SOURCE, name=name, **CUDA_OPTIONS),
    )


eval_spline_mpi_markers = _pair("eval_spline_mpi_markers")
eval_spline_mpi_matrix = _pair("eval_spline_mpi_matrix")
eval_spline_mpi_sparse_meshgrid = _pair("eval_spline_mpi_sparse_meshgrid")
