MODULES_INTEL="intel-oneapi-compilers-classic/2021.10.0 \
intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1 \
python/3.11.7"

MODULES_GCC="gcc/12.3.0 \
openmpi/4.1.6--gcc--12.3.0 \
python/3.11.7"

# On the Booster (GPU) partition, ARRAY_BACKEND=cupy runs need libnvrtc.so.12 for
# cupy's RawKernel/JIT compilation -- otherwise every cupy import fails as soon as
# it touches the GPU (e.g. `xp.tri()` at struphy import time). SLURM_JOB_PARTITION
# is only set inside a submitted job, so this is a no-op on the DCGP (CPU) partition
# or outside SLURM.
if [[ "${SLURM_JOB_PARTITION:-}" == *boost* ]]; then
    MODULES_INTEL="$MODULES_INTEL cuda/12.6"
    MODULES_GCC="$MODULES_GCC cuda/12.6"
fi
