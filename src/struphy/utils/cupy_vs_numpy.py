import logging


import time

import cunumpy as xp

logger = logging.getLogger("struphy")

def main(N=8192):
    logger.info(f"Creating {N}x{N} random matrices...")

    A = xp.random.rand(N, N)
    B = xp.random.rand(N, N)

    logger.info("Running matrix multiplication: C = A @ B...")
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    logger.info(f"Matrix multiplication took {t1 - t0:.3f} seconds")

    logger.info("Running D = xp.tanh(C * 0.01) + xp.exp(-C * 0.001)...")
    t0 = time.perf_counter()
    D = xp.tanh(C * 0.01) + xp.exp(-C * 0.001)

    t1 = time.perf_counter()
    logger.info(f"Transformation took {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
