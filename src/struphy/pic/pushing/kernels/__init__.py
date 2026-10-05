"""Pusher and marker evaluation kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` contains the pyccel kernel ``<name>_kernels.py`` and, once ported, its CUDA version
``<name>_cuda.cu``. Get the kernel for the active backend from the catalog::

    from struphy.pic.pushing.kernels import catalog

    kernel = catalog["push_eta_stage"]
"""

from struphy.utils.kernel_backends import KernelCatalog

catalog = KernelCatalog.from_package(__name__)
