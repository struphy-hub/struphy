"""Accumulation kernels, one folder per kernel (see ``CUDA_STRATEGY.md``).

Each folder ``<name>`` contains the pyccel kernel ``<name>_kernels.py`` and, once ported, its CUDA version
``<name>_cuda.cu``. Get the kernel for the active backend from the catalog::

    from struphy.pic.accumulation.kernels import catalog

    kernel = catalog["charge_density_0form"]
"""

from cunumpy.kernels import KernelCatalog

from struphy.utils.cuda_arguments import CUDA_STRUCTS

catalog = KernelCatalog.from_package(__name__, structs=CUDA_STRUCTS)
