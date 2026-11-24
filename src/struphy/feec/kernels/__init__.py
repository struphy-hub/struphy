"""FEEC kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` holds the pyccel kernel ``<name>_kernels.py`` and, once ported, its CUDA version
``<name>_cuda.cu``. The folder's
``__init__.py`` declares the kernel, a :class:`cunumpy.kernels.Kernel` that runs the version of the active
backend::

    from struphy.feec.kernels.hybrid_weight import hybrid_weight

Besides the utility kernel ``hybrid_weight``, the folders hold the assembly kernels of
:mod:`struphy.feec.mass` (``kernel_{1,2,3}d_mat``, ``kernel_{1,2,3}d_vec``, ``kernel_{1,2,3}d_eval``,
``kernel_3d_matrixfree``, ``kernel_3d_diag``), of :mod:`struphy.feec.boundary_mass`
(``surface_kernel_3d_mat``, ``surface_kernel_3d_vec``) and of :mod:`struphy.feec.basis_projection_ops`
(``assemble_dofs_for_weighted_basisfuns_{1,2,3}d``). None of the assembly kernels has a CUDA version yet;
they are declared with ``missing_cuda="fallback"``, so on the CuPy backend they run on host copies of the
arrays, as before.
"""
