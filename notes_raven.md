# Setup

```
module purge
module load nvhpcsdk/25 cuda/12.6-nvhpcsdk_25 openmpi/5.0 openmpi_gpu/5.0 gcc/13

source ~/virtual_envs/env_struphy_gpu/bin/activate

# Add likwid module

module load likwid/5.3
LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib:$LD_LIBRARY_PATH
```

Write the GPU kernels

```
vim src/struphy/pic/pushing/pusher_kernels_gpu.py
```

# Compiling

```
struphy compile --language fortran --compiler /u/maxlin/git_repos/struphy/compiler_nvidia.json --omp-pic

# or

pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_nvidia.json --conda-warnings=off --verbose  --openmp  /raven/u/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.py
```

# Running

```
python src/struphy/gpu/test_cupy_timings.py
python src/struphy/gpu/test_pyccel_timings.py
```

Run with CPU/GPU

```
struphy run Vlasov       --time-trace -o sim_Vlasov_cpu # CPU
struphy run Vlasov --gpu --time-trace -o sim_Vlasov_gpu # GPU


struphy run IsothermalEulerSPH       --time-trace -i verification/IsothermalEulerSPH_soundwave.yml -o sim_IsothermalEulerSPH_cpu # CPU
struphy run IsothermalEulerSPH --gpu --time-trace -i verification/IsothermalEulerSPH_soundwave.yml -o sim_IsothermalEulerSPH_gpu # GPU

struphy run IsothermalEulerSPH       --time-trace --nsys-profile -i params_IsothermalEulerSPH_amin.yml -o sim_IsothermalEulerSPH_soundwave_cpu # CPU
struphy run IsothermalEulerSPH --gpu --time-trace --nsys-profile -i params_IsothermalEulerSPH_amin.yml -o sim_IsothermalEulerSPH_soundwave_gpu # GPU
```

## Submit jobs

```
struphy run IsothermalEulerSPH       --time-trace -i params_IsothermalEulerSPH_amin.yml -o sim_IsothermalEulerSPH_soundwave_cpu -b slurm_raven_gpu.sh # CPU
struphy run IsothermalEulerSPH --gpu --time-trace -i params_IsothermalEulerSPH_amin.yml -o sim_IsothermalEulerSPH_soundwave_gpu -b slurm_raven_gpu.sh # GPU
```

## post processing
****
```
struphy pproc -d sim_Vlasov_cpu --time-trace
struphy pproc -d sim_Vlasov_gpu --time-trace

struphy pproc -d sim_IsothermalEulerSPH_cpu --time-trace
struphy pproc -d sim_IsothermalEulerSPH_gpu --time-trace

struphy pproc -d sim_IsothermalEulerSPH_soundwave_cpu --time-trace
struphy pproc -d sim_IsothermalEulerSPH_soundwave_gpu --time-trace
```
