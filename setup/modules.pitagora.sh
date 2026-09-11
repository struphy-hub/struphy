MODULES_INTEL="intel-oneapi-compilers-classic/2021.10.0 \
intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1 \
python/3.11.7"

MODULES_GCC="gcc/12.3.0 \
openmpi/4.1.6--gcc--12.3.0-ucx1.20 \
petsc/3.22.1--openmpi--4.1.6--gcc--12.3.0-ucx1.20-complex-mumps \
python/3.11.7"

# The petsc module above is built with CUDA support, so petsc4py's import dlopens
# libcuda.so.1 even on nodes without a GPU driver. Point at a stub so it doesn't fail.
export LD_LIBRARY_PATH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.venv/petsc_cuda_stub:${LD_LIBRARY_PATH:-}"
