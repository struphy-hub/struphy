"""Mapping evaluation and pull-back/push-forward kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` holds the pyccel kernel ``<name>_kernels.py`` and, once ported, its CUDA version
``<name>_cuda.cu``. The folder's
``__init__.py`` declares the kernel, a :class:`cunumpy.kernels.Kernel` that runs the version of the active
backend::

    from struphy.geometry.kernels.kernel_evaluate_pic import kernel_evaluate_pic
"""
