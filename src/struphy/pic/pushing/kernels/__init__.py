"""Pusher and marker evaluation kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` holds the pyccel kernel ``<name>_kernels.py`` and, once ported, its CUDA version
``<name>_cuda.cu``. The folder's ``__init__.py`` declares the
kernel, a :class:`cunumpy.kernels.Kernel` that runs the version of the active backend::

    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
"""
