# Salloc

```
salloc --partition=gpudev --ntasks=1 --time=00:15:00 --mem=12500 --gres=gpu:a100:1
```

# Setup

```
module purge
module load nvhpcsdk/25 cuda/12.6-nvhpcsdk_25 openmpi/5.0 openmpi_gpu/5.0

source ~/virtual_envs/env_struphy/bin/activate
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

Run with CPU

```
struphy run Vlasov --time-trace -o sim_cpu
```

Run with GPU

```
struphy run Vlasov --gpu --time-trace -o sim_gpu
```

post processing

```
struphy pproc -d sim_cpu --time-trace
struphy pproc -d sim_gpu --time-trace
```
