MODULES_INTEL="intel-oneapi-compilers-classic/2021.10.0 \
intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1 \
python/3.11.7"


# openmpi/4.1.6--gcc--12.3.0
MODULES_GCC="gcc/12.3.0 \
openmpi/4.1.6--gcc--12.3.0 \
python/3.11.7 \
hdf5/1.14.3--gcc--12.3.0 \
cmake/3.27.9 \
netcdf-fortran/4.6.1--gcc--12.3.0 \
netlib-scalapack/2.2.0--openmpi--4.1.6--gcc--12.3.0-ucx1.20"

# On the Booster (GPU) partition, ARRAY_BACKEND=cupy runs need libnvrtc.so.12 for
# cupy's RawKernel/JIT compilation -- otherwise every cupy import fails as soon as
# it touches the GPU (e.g. `xp.tri()` at struphy import time). SLURM_JOB_PARTITION
# is only set inside a submitted job, so this is a no-op on the DCGP (CPU) partition
# or outside SLURM.
if [[ "${SLURM_JOB_PARTITION:-}" == *boost* ]]; then
    MODULES_INTEL="$MODULES_INTEL cuda/12.6"
    MODULES_GCC="$MODULES_GCC cuda/12.6"
fi

# For GVEC
# Should be fixed so it works with both gcc and intel
export FC=`which gfortran`
export CC=`which gcc`
export CXX=`which g++`
