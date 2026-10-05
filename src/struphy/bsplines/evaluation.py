"""Spline evaluation kernel pairs with identical Python arguments on both backends."""

from pathlib import Path

from cunumpy import PyccelKernel

from struphy.bsplines import evaluation_kernels_3d
from struphy.utils.kernel_backends import CudaKernel, Kernel


_SOURCE = Path(__file__).with_name("evaluate_spline_cuda.cu")


def _pair(name):
    return Kernel(
        PyccelKernel(getattr(evaluation_kernels_3d, name)),
        CudaKernel.from_file(_SOURCE, name=name),
    )


eval_spline_mpi_markers = _pair("eval_spline_mpi_markers")
eval_spline_mpi_matrix = _pair("eval_spline_mpi_matrix")
eval_spline_mpi_sparse_meshgrid = _pair("eval_spline_mpi_sparse_meshgrid")
