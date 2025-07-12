import time

import numpy as np

import struphy.pic.pushing.pusher_kernels
from struphy.pic.pushing.pusher_kernels_gpu import matmul_cpu, matmul_gpu

# from pusher_kernels_gpu import matmul_cpu, matmul_gpu


def compare_pyccel_cpu_gpu(Nvec=[2000], iterations=1):
    elapsed_cpu_vec = []
    elapsed_gpu_vec = []

    for N in Nvec:
        print(f"{N = }")

        min_cpu = 1e17
        max_cpu = 0.0
        avg_cpu = 0.0

        min_gpu = 1e17
        max_gpu = 0.0
        avg_gpu = 0.0

        for iteration in range(iterations):
            A = np.random.random((N, N))
            B = np.random.random((N, N))

            C_cpu = np.empty((N, N))
            C_gpu = np.empty((N, N))

            # ------------------------ CPU --------------------------- #
            # Time CPU matrix multiplication.
            start_cpu = time.time()
            matmul_cpu(A, B, C_cpu)
            el_cpu = time.time() - start_cpu

            # ------------------------ GPU --------------------------- #
            matmul_gpu(A, B, C_gpu)
            # Time GPU matrix multiplication.
            start_gpu = time.time()
            matmul_gpu(A, B, C_gpu)
            el_gpu = time.time() - start_gpu

            min_cpu = min(min_cpu, el_cpu)
            max_cpu = max(max_cpu, el_cpu)
            avg_cpu += el_cpu

            min_gpu = min(min_gpu, el_gpu)
            max_gpu = max(max_gpu, el_gpu)
            avg_gpu += el_gpu

            # ------------------------ Output --------------------------- #
            assert np.allclose(C_cpu, C_gpu)

        avg_cpu /= iterations
        avg_gpu /= iterations

        elapsed_cpu_vec.append([N, avg_cpu, min_cpu, max_cpu])
        elapsed_gpu_vec.append([N, avg_gpu, min_gpu, max_gpu])
        # print(f"{elapsed_cpu_vec[-1] = } {elapsed_gpu_vec[-1] = }")
        # print(f"{A = }")
        # print(f"{B = }")
        # print(f"{C_cpu = }")
        # print(f"{C_gpu = }")
        # print(f"{np.allclose(C_cpu, C_gpu) = }")
        # print("\n\n\n#" + "-"*40 + "#")
        # print("Comparing pyccel with and w/o GPU...")
        # print(f"{elapsed_cpu = }")
        # print(f"{elapsed_gpu = }")
        # print(f"pyccel+OpenMP speedup: {elapsed_cpu / elapsed_gpu} for matrix size: {N}x{N}")
        # print(f"{N},{elapsed_cpu / elapsed_gpu}")
        # print("#" + "-"*40 + "#")
    return elapsed_cpu_vec, elapsed_gpu_vec


if __name__ == "__main__":
    compare_pyccel_cpu_gpu(N=2000)
