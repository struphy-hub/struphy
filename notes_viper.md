# Load modules

```
module purge
module load gcc/14 rocm/6.4 openmpi/5.0 python-waterboa/2024.06 cupy/13.4
module load amd-llvm/6.1
source ~/virtual_envs/env_struphy/bin/activate
```

## Compile all the kernels

Only this should be needed

```
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/linear_algebra/linalg_kernels.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/bsplines/bsplines_kernels.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/bsplines/evaluation_kernels_1d.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/bsplines/evaluation_kernels_2d.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/bsplines/evaluation_kernels_3d.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/geometry/mappings_kernels.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/geometry/evaluation_kernels.py
```

Compile all the kernels the usual way

```
struphy compile --language fortran --compiler /u/maxlin/git_repos/struphy/compiler_llvm.json
```


Compile the GPU kernels with `--verbose` and `--openmp`, then copy the compile commands:

```
# pusher_args_kernels
pyccel --libdir /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib --language=fortran --compiler=/u/maxlin/git_repos/struphy/testing/compiler_llvm.json --conda-warnings=off --verbose  --openmp  /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_args_kernels.py --verbose

# pusher_kernels_gpu.py
pyccel --libdir /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib --language=fortran --compiler=/u/maxlin/git_repos/struphy/testing/compiler_llvm.json --conda-warnings=off --verbose  --openmp  /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.py --verbose
```

Copy the compiler commands and uncomment the openmp pragmas.

## Add to pusher_kernels_gpu.f90

```
module pusher_kernels_gpu

  !use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
  !      C_DOUBLE
  use pusher_args_kernels
  use pusher_args_kernels, only: DerhamArguments
  use pusher_args_kernels, only: DomainArguments
  use pusher_args_kernels, only: MarkerArguments
  use pyc_math_f90

  implicit none
  
  integer, parameter, private :: f64 = 8
  integer, parameter, private :: i64 = 8
  
  ! ...
```

# Add to pusher_args_kernels.f90

```
module pusher_args_kernels

  !use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
  !      C_INT64_T , b1 => C_BOOL

  implicit none
  
  integer, parameter, private :: f64 = 8
  integer, parameter, private :: i64 = 8
  integer, parameter, private :: b1 = 1
```


# Compile again

```
# pusher_args_kernels
# pyccel --libdir /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib --language=fortran --compiler=/u/maxlin/git_repos/struphy/testing/compiler_llvm.json --conda-warnings=off --verbose  --openmp  /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_args_kernels.py --verbose

# pusher_kernels_gpu.py
pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.py

# Fix pusher_args_kernels.f90
# sed -i '1,7c\
# module pusher_args_kernels\
# \
#   !use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &\
#   !      C_INT64_T , b1 => C_BOOL\
# \
#   implicit none\n\
#   \
#   integer, parameter, private :: f64 = 8\
#   integer, parameter, private :: i64 = 8\
#   integer, parameter, private :: b1 = 1' \
#/viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_args_kernels.f90


# Fix pusher_kernels_gpu.py
sed -i '1,10c\
module pusher_kernels_gpu\
\
!   use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &\
!         C_INT64_T\
  use pusher_args_kernels\
  use pusher_args_kernels, only: DerhamArguments\
  use pusher_args_kernels, only: DomainArguments\
  use pusher_args_kernels, only: MarkerArguments\
\
  implicit none\
\
  integer, parameter, private :: f64 = 8\
  integer, parameter, private :: i64 = 8' \
/viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.f90

# Compile pusher_args_kernels.py
# $ pyccel --libdir /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib --language=fortran --compiler=/u/maxlin/git_repos/struphy/testing/compiler_llvm.json --conda-warnings=off --verbose  --openmp  /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_args_kernels.py --verbose
#/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -O3 -fPIC -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.f90 -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.o -J /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
#/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -O3 -fPIC -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.f90 -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.o -J /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
#/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdclang -O3 -funroll-loops -fPIC -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include -fPIC -O2 -isystem /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include -pthread -B /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/compiler_compat -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/cwrapper -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ -I /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include/python3.12 -I /viper/u2/maxlin/virtual_envs/env_struphy/lib/python3.12/site-packages/numpy/_core/include /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.c -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.o
#/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -shared -O3 -fPIC -L /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib -Wl,-rpath /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math/pyc_math_f90.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/cwrapper/cwrapper.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.o /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib/libpython3.12.so -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.cpython-312-x86_64-linux-gnu.so -lm
# > Shared library has been created: /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_args_kernels.cpython-312-x86_64-linux-gnu.so

# Compile pusher_kernels_gpu.py
# $ pyccel --language=fortran --compiler=/u/maxlin/git_repos/struphy/compiler_llvm.json --conda-warnings=off --verbose    /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.py --openmp
/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -O3 -fPIC -fopenmp --offload-arch=gfx942 -fopenmp-force-usm -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.f90 -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.o -J /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -O3 -fPIC -fopenmp --offload-arch=gfx942 -fopenmp-force-usm -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.f90 -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.o -J /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdclang -O3 -funroll-loops -fPIC -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include -fPIC -O2 -isystem /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include -pthread -B /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/compiler_compat -fopenmp -c -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/cwrapper -I /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__ -I /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/include/python3.12 -I /viper/u2/maxlin/virtual_envs/env_struphy/lib/python3.12/site-packages/numpy/_core/include /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.c -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.o
/mpcdf/soft/RHEL_9/packages/x86_64/rocm/6.4.0/bin/amdflang -shared -O3 -fPIC -fopenmp --offload-arch=gfx942 -fopenmp-force-usm /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu_wrapper.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/bind_c_pusher_kernels_gpu.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/cwrapper/cwrapper.o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/math/pyc_math_f90.o /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/lib/libpython3.12.so -o /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.cpython-312-x86_64-linux-gnu.so -lm -lomptarget

# > Shared library has been created: /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/pusher_kernels_gpu.cpython-312-x86_64-linux-gnu.so

#cp /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_args_kernels.cpython-312-x86_64-linux-gnu.so /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing

cp /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__/pusher_kernels_gpu.cpython-312-x86_64-linux-gnu.so /viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing

```


# Run with GPU

salloc the apudev partition for 15 minutes

```
salloc --time=00:15:00 --partition=apudev --ntasks=1 --gres=gpu:1 --cpus-per-task=1 --mem=40000
```

Setup magic commands

```
module purge
module load gcc/14 rocm/6.4 openmpi/5.0 python-waterboa/2024.06
module load amd-llvm/6.1
module load cupy_rocm/13.4
source ~/virtual_envs/env_struphy/bin/activate
# export PYTHONPATH=/viper/u2/maxlin/git_repos/struphy/src/struphy/pic/pushing/__pyccel__
export HSA_XNACK=1
export OFFLOAD_TRACK_ALLOCATION_TRACES=true
export LD_LIBRARY_PATH=/mpcdf/soft/RHEL_9/packages/x86_64/gcc/14.1.0/lib64:$LD_LIBRARY_PATH
```

* GPU/CPU kernel is set in src/struphy/propagators/propagators_markers.py
* Now, the model takes a parameter "gpu" from main.py


Run tests

```
struphy run Vlasov
```
