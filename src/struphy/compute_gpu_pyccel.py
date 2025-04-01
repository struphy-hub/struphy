from pyccel.stdlib.internal.openmp import (
    omp_get_num_devices,
    omp_get_thread_num,
    omp_get_num_threads,
    omp_get_team_num
)
import numpy as np
from struphy.pyccel_kernels import axpy, axpy_gpu
from struphy.pyccel_kernels import heavy_compute_cpu, heavy_compute_gpu
from struphy.pyccel_kernels import matmul_cpu, matmul_gpu
# from struphy.pic.pushing.pusher_kernels import matmul_gpu

import time



def time_axpy():
    print("time_axpy")
    N: int = 100_000_000
    a: float = 2.0
    x: 'float[:]' = np.random.random(N)
    y: 'float[:]' = np.random.random(N)
    
    a = time.perf_counter()
    axpy(a, x, y)
    b = time.perf_counter()
    elapsed_time_axpy = b - a
    
    axpy_gpu(a, x, y)
    c = time.perf_counter()
    axpy_gpu(a, x, y)
    d = time.perf_counter()
    elapsed_time_axpy_gpu = d - c
    
    print(f"CPU: {elapsed_time_axpy} seconds")
    print(f"GPU: {elapsed_time_axpy_gpu} seconds")

def time_heavy_compute():
    print("time_heavy_compute")
    N: int = 1000_000  
    x: 'float[:]' = np.random.random(N).astype(np.float64)
    y_cpu: 'float[:]' = np.empty_like(x)
    y_gpu: 'float[:]' = np.empty_like(x)
    
    # Warm-up the GPU (optional)
    heavy_compute_gpu(x, y_gpu)
    
    # Time the CPU version.
    start_cpu = time.time()
    heavy_compute_cpu(x, y_cpu)
    elapsed_cpu = time.time() - start_cpu
    
    # Time the GPU version.
    start_gpu = time.time()
    heavy_compute_gpu(x, y_gpu)
    elapsed_gpu = time.time() - start_gpu
    
    print(f"CPU: {elapsed_cpu:.6f} seconds")
    print(f"GPU: {elapsed_gpu:.6f} seconds")

def time_matmul_compute():
    print("time_matmul_compute")
    # Set a problem size that is large enough so that the offloading overhead is amortized.
    N: int = 2**10 # Adjust N as needed
    A = np.random.random((N, N)).astype(np.float64)
    B = np.random.random((N, N)).astype(np.float64)
    C_cpu = np.empty((N, N), dtype=np.float64)
    C_gpu = np.empty((N, N), dtype=np.float64)

    # Warm-up GPU offloading (optional)
    # matmul_gpu(A, B, C_gpu)
    
    print('Start matmul_cpu')
    # Time CPU matrix multiplication.
    start_cpu = time.time()
    matmul_cpu(A, B, C_cpu)
    elapsed_cpu = time.time() - start_cpu

    print('Start matmul_gpu')
    # Time GPU matrix multiplication.
    start_gpu = time.time()
    matmul_gpu(A, B, C_gpu)
    elapsed_gpu = time.time() - start_gpu

    print(f"CPU: {elapsed_cpu:.6f} seconds")
    print(f"GPU: {elapsed_gpu:.6f} seconds")

if __name__ == '__main__':
    print("Number of available GPUs:", omp_get_num_devices())

    from mpi4py import MPI
    time_matmul_compute()
    


    # time_axpy()
    # time_heavy_compute()