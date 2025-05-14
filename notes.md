#export LD_LIBRARY_PATH=/mpcdf/soft/RHEL_9/packages/x86_64/gcc/14.1.0/lib64:$LD_LIBRARY_PATH
module purge
module load gcc/14 rocm/6.3 openmpi/5.0 likwid/5.3 python-waterboa/2024.06
module load amd-llvm/5.1
source ~/virtual_envs/env_struphy/bin/activate
export PYTHONPATH=/viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
export HSA_XNACK=1
export OFFLOAD_TRACK_ALLOCATION_TRACES=true
export LD_LIBRARY_PATH=/mpcdf/soft/RHEL_9/packages/x86_64/gcc/14.1.0/lib64:$LD_LIBRARY_PATH

# Check the kernel in src/struphy/propagators/propagators_markers.py




