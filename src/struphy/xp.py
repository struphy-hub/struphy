import os

# TODO: Make this configurable via environment variable or config file.
# Default to numpy
_backend = os.getenv("ARRAY_BACKEND", "numpy").lower()


# TODO: Write an ArrayBackend class 

def _import_numpy():
    # print("importing numpy...")
    import numpy as np

    return np


def _import_cupy():
    # print("importing cupy...")
    try:
        import cupy as cp

        return cp
    except ImportError:
        print("CuPy not available, falling back to NumPy.")
        return _import_numpy()

# Import numpy/cupy
if _backend == "cupy":
    xp = _import_cupy()
else:
    xp = _import_numpy()
