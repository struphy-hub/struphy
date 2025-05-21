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


# Compiling

```
struphy compile --language fortran --compiler /u/maxlin/git_repos/struphy/compiler_nvidia.json
```

# Running

```
python src/struphy/test_cupy_timings.py
python src/struphy/test_pyccel_timings.py
```

```
struphy run Vlasov
```

