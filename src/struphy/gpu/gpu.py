try:
    import cupy as cp
    # Try running a simple GPU command to make sure GPU is really available
    _ = cp.random.random(1)  # May raise an error if GPU is not usable
    gpu_active = True
    print("GPU active (CuPy)")
except Exception:
    import numpy as cp
    gpu_active = False
    print("GPU not active, falling back to NumPy")


def import_xp():
    if gpu_active:
        import cupy as xp
    else:
        import numpy as xp
    return xp