"""Box sorting kernels of :class:`~struphy.pic.base.Particles`, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` holds the pyccel kernel ``<name>_kernels.py`` and, once ported (#694), its CUDA version
``<name>_cuda.cu``. The folder's ``__init__.py`` declares the kernel, a :class:`cunumpy.kernels.Kernel` that runs the
version of the active backend::

    from struphy.pic.kernels.box_particles import box_particles

The helpers they call are in :mod:`struphy.pic.sorting_kernels`.
"""
