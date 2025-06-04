import time

import numpy as np
import struphy.pic.pushing.pusher_kernels
from struphy.pic.pushing.pusher_kernels_gpu import matmul_cpu, matmul_gpu
# from pusher_kernels_gpu import matmul_cpu, matmul_gpu


def compare_pyccel_cpu_gpu(N=2000):
    # compare_np_cp()

    A = np.random.random((N, N))
    B = np.random.random((N, N))

    C_cpu = np.empty((N, N))
    C_gpu = np.empty((N, N))

    # Warm-up GPU offloading (optional)
    # matmul_gpu(A, B, C_gpu)
    # print(f"matrix size: {N}")

    # ------------------------ CPU --------------------------- #
    print("Start matmul_cpu")
    # Time CPU matrix multiplication.
    start_cpu = time.time()
    matmul_cpu(A, B, C_cpu)
    elapsed_cpu = time.time() - start_cpu

    # ------------------------ GPU --------------------------- #
    matmul_gpu(A, B, C_gpu)
    print("End matmul_cpu")
    # print("warming up gpu")
    # matmul_gpu(A, B, C_gpu)
    print("Start matmul_gpu")
    # Time GPU matrix multiplication.
    start_gpu = time.time()
    # for i in range(1000):
    matmul_gpu(A, B, C_gpu)
    elapsed_gpu = time.time() - start_gpu
    print("End matmul_gpu")

    # ------------------------ Output --------------------------- #

    # print(f"{A = }")
    # print(f"{B = }")
    # print(f"{C_cpu = }")
    # print(f"{C_gpu = }")
    print(f"{np.allclose(C_cpu, C_gpu) = }")
    print(f"{elapsed_cpu = }")
    print(f"{elapsed_gpu = }")
    print(f"Speedup: {elapsed_cpu / elapsed_gpu} for matrix size: {N}x{N}")


if __name__ == "__main__":
    compare_pyccel_cpu_gpu(N=2000)
