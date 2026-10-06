"""Spline evaluation kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` holds the pyccel kernel ``<name>_kernels.py``, its CUDA version ``<name>_cuda.cu`` once
ported, and the arguments of its parity test ``<name>_test_args.py`` once it has a CUDA version. The folder's
``__init__.py`` declares the kernel, a :class:`cunumpy.kernels.Kernel` that runs the version of the active
backend::

    from struphy.bsplines.kernels.eval_spline_mpi_markers import eval_spline_mpi_markers
"""
