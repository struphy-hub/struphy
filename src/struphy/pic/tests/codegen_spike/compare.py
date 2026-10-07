"""Run the numba kernel generated from ``bstar_parallel_3form`` on its parity cases and compare it with pyccel.

Usage (numba is not a struphy dependency)::

    NUMBA_ENABLE_CUDASIM=1 python -m struphy.pic.tests.codegen_spike.compare cuda   # numba.cuda simulator
    python -m struphy.pic.tests.codegen_spike.compare cpu                           # same translation, numba.njit

The simulator interprets the generated Python thread by thread (it does not compile anything); ``cpu`` compiles the
same translation in nopython mode. Prints one line per case and a summary; exits with 1 if a case disagrees.
"""

import sys
import time

import cunumpy as xp
import numpy as np

from struphy.pic.tests.codegen_spike.numba_codegen import flatten, load

MODULE = "struphy.pic.pushing.kernels.bstar_parallel_3form.bstar_parallel_3form_kernels"
NAME = "bstar_parallel_3form"
BLOCK = 128


def compare(target):
    from struphy.pic.pushing.kernels.bstar_parallel_3form import bstar_parallel_3form
    from struphy.pic.tests.cuda_parity_cases import PARITY_CASES
    from struphy.pic.tests.test_cuda_emulation import arrays

    kernel, path = load(MODULE, NAME, target)
    print(f"generated {path} ({len(path.read_text().splitlines())} lines)")
    spec = PARITY_CASES[NAME]
    failures, bitwise = 0, 0
    with xp.use_backend("numpy"):
        for case in spec.cases:
            reference_args, generated_args = spec.build(case), spec.build(case)
            bstar_parallel_3form(*reference_args)
            flat = flatten(generated_args)
            start = time.perf_counter()
            if target == "cuda":
                n_markers = generated_args[2].n_markers
                kernel[(n_markers + BLOCK - 1) // BLOCK, BLOCK](*flat)
            else:
                kernel(*flat)
            seconds = time.perf_counter() - start
            reference, generated = arrays(reference_args), arrays(generated_args)
            diff = max(
                float(np.max(np.abs(r.astype(float) - g.astype(float)), initial=0.0))
                for r, g in zip(reference, generated)
            )
            same = all(np.array_equal(r, g, equal_nan=True) for r, g in zip(reference, generated))
            ok = all(np.allclose(g, r, rtol=spec.rtol, atol=spec.atol) for r, g in zip(reference, generated))
            failures += not ok
            bitwise += same
            print(f"domain {case[0]:2d}: max |diff| = {diff:.2e}, bitwise equal: {same}, {seconds:.2f} s")
    print(f"{target}: {len(spec.cases) - failures}/{len(spec.cases)} cases agree, {bitwise} bitwise")
    return failures


if __name__ == "__main__":
    sys.exit(1 if compare(sys.argv[1] if len(sys.argv) > 1 else "cpu") else 0)
