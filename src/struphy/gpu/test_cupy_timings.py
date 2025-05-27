# https://docs.cupy.dev/en/stable/user_guide/basic.html
import time
try:
    import cupy as cp
except:
    print(f"cupy not installed, falling back to numpy")
    import numpy as cp
import numpy as np

def test_xp(xp, N = 100_000_000, num_iterations = 1):
    print(f"Running with {xp = }")
    elapsed_time = 0.0

    # Warm-up (important for CuPy)
    _ = xp.linalg.norm(xp.random.random((1000_000,)))

    # ---- Timing ---- #
    for _ in range(num_iterations):
        x_arr = xp.random.random((N))

        t0 = time.time()
        l2 = xp.linalg.norm(x_arr ** 6.123 - xp.sqrt(x_arr))
        if xp == cp:
            cp.cuda.Device().synchronize()
        t1 = time.time()
        elapsed_time += (t1 - t0)
        print(f"{t1 - t0 = }")
    elapsed_time /= num_iterations
    return elapsed_time

def compare_np_cp():
    print("Comparing numpy and cupy...")
    time_cpu = test_xp(np)
    time_gpu = test_xp(cp)
    # time_cpu = test_numpy()
    # time_gpu = test_cupy()    

    # ---- Print output ---- #
    print(f"Time Numpy: {time_cpu}")
    print(f"Time CuPy: {time_gpu}")
    print(f"Speedup: {time_cpu / time_gpu}")

if __name__ == '__main__':
    compare_np_cp()