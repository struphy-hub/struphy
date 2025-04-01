# Compile with:
# pyccel --libdir /usr/lib/x86_64-linux-gnu --language=fortran --compiler=/home/maxlinadmin/git_repos/struphy/compiler_nvfortran_maxlinpc.json --conda-warnings=off --openmp /home/maxlinadmin/git_repos/struphy/src/struphy/pyccel_kernels.py --verbose
# pyccel --language fortran --openmp --compiler compiler_clang_maxlinpc.json --verbose /home/maxlinadmin/git_repos/struphy/src/struphy/pyccel_kernels.py
# Run with
# nsys profile --stats true python3 src/struphy/compute_gpu_pyccel.py

import numpy as np

def axpy(a: float, x: 'float[:]', y: 'float[:]'):
    N: int = x.shape[0]
    for i in range(N):
        y[i] = a * x[i] + y[i]

def axpy_gpu(a: float, x: 'float[:]', y: 'float[:]'):
    N: int = x.shape[0]
    #$ omp target teams distribute parallel for schedule(static)
    for i in range(N):
        y[i] = a * x[i] + y[i]
            
def heavy_compute_cpu(x: 'float[:]', y: 'float[:]'):
    N: int = x.shape[0]
    temp: float = 0.0
    for i in range(N):
        temp = x[i]
        # A heavy inner loop to increase arithmetic intensity
        for j in range(1000):
            temp = np.sqrt(temp + 1.0)
        y[i] = temp

# GPU version: offloaded using OpenMP target directive
def heavy_compute_gpu(x: 'float[:]', y: 'float[:]'):
    N: int = x.shape[0]
    temp: float = 0.0
    #$ omp target teams distribute parallel for schedule(static)
    for i in range(N):
        temp = x[i]
        for j in range(1000):
            temp =  np.sqrt(temp + 1.0)
        y[i] = temp

def matmul_cpu(A: 'float[:,:]', B: 'float[:,:]', C: 'float[:,:]'):
    N: int = A.shape[0]
    s: float = 0.0
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s

def matmul_gpu(A: 'float[:,:]', B: 'float[:,:]', C: 'float[:,:]'):
    N: int = A.shape[0]
    s: float = 0.0
    #$ omp target teams distribute parallel for collapse(2)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j] 
            C[i, j] = s
