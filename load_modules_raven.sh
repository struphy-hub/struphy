module purge
module load nvhpcsdk/25 cuda/12.6-nvhpcsdk_25 openmpi/5.0 openmpi_gpu/5.0 gcc/13

source ~/virtual_envs/env_struphy_gpu/bin/activate

# Add likwid module

module load likwid/5.3
LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib:$LD_LIBRARY_PATH
export PMIX_MCA_pcompress_base_silence_warning=1
