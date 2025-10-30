import time

import cunumpy as xp


def main(N=8192):
    print(f"Creating {N}x{N} random matrices...")

    A = xp.random.rand(N, N)
    B = xp.random.rand(N, N)

    print("Running matrix multiplication: C = A @ B...")
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    print(f"Matrix multiplication took {t1 - t0:.3f} seconds")

    print("Running D = xp.tanh(C * 0.01) + xp.exp(-C * 0.001)...")
    t0 = time.perf_counter()
    D = xp.tanh(C * 0.01) + xp.exp(-C * 0.001)

    t1 = time.perf_counter()
    print(f"Transformation took {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
