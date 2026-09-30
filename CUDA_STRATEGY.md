# CUDA strategy for Struphy kernels

Plan for running Struphy's compute kernels on NVIDIA GPUs, next to the existing pyccel (CPU) kernels.
The work is split into small PRs that can be reviewed and merged one at a time. Nothing here has to be done in one go.

## PR checklist

- [ ] **PR 1: Proof of concept** (branch `cuda-kernel-proof-of-concept`)
  `Kernel` (pyccel/CUDA pair), `CudaKernel`, `CudaMarkerArguments`/`CudaDomainArguments`, and one test file with a demo kernel pair run on both backends. Also adds this document and the `gpu` optional dependency.
- [ ] **PR 2: CUDA source files** — `CudaKernel` loads CUDA source from a `<name>_cuda.cu` file next to the pyccel file; `.cu`/`.cuh` files are shipped as package data.
- [ ] **PR 3: Kernel catalog** — kernels are defined once in the `__init__.py` of the folder that contains them; a missing CUDA kernel raises an error on the GPU backend.
- [ ] **PR 4: `Pusher` and propagators accept `Kernel`** — replace `PyccelKernel(...)` in the propagators by catalog lookups (no CUDA kernels yet, so no behaviour change on CPU).
- [ ] **PR 5: `Domain` on the GPU** — `Domain.cuda_args_domain`, plus the separate fix for `Domain` deepcopy on the CuPy backend (`_build_args_domain` passes `params_numpy` without `_to_numpy_for_kernel`).
- [ ] **PR 6: `Particles` on the GPU** — `Particles` can be created on the CuPy backend (e.g. `xp.prod` on Python lists in `pic/base.py`), plus `Particles.cuda_args_markers`.
- [ ] **PR 7: `Derham` on the GPU** — `Derham` can be created on the CuPy backend, plus `Derham.cuda_args_derham`.
- [ ] **PR 8: Shared CUDA headers for the argument classes** — one `.cuh` per argument class instead of long flat kernel signatures.
- [ ] **PR 9: One folder per kernel, starting with `pic/pushing`** — pure refactor, no behaviour change.
- [ ] **PR 10: Device versions of helper kernels** — B-spline evaluation, mapping evaluation (per domain), small linear algebra, as `__device__` functions in `.cuh` headers.
- [ ] **PR 11: First real CUDA kernel** — `push_eta_stage` with a pyccel/CUDA parity test and an end-to-end run on the GPU.
- [ ] **PR 12+: Port kernels one by one**, in the order they are needed by the models we want on the GPU (see [Porting order](#porting-order)).
- [ ] **CI**: a GPU runner that runs the CUDA tests (can happen any time after PR 1).

Unrelated bugs found along the way go into their own PRs, not into these ones.

## Goal

A developer who adds a new kernel (e.g. for a new model) should only have to:

1. write the pyccel kernel `<name>_kernels.py` as today, and
2. optionally write `<name>_cuda.cu` **in the same folder**.

Everything else (loading, dispatch, argument passing, tests for agreement between the two versions) is done by the infrastructure.
CUDA kernels can be added one by one. If the code runs on the GPU and needs a kernel that has no CUDA version yet, it raises a clear error instead of silently falling back to the CPU.

## Principles

- **1:1 correspondence.** Each CUDA kernel has the same name and the same arguments (in the same order) as its pyccel kernel. The CUDA kernel takes the CUDA versions of the argument classes, plus the number of threads (`n_threads`).
- **The backend decides.** The cunumpy backend (`ARRAY_BACKEND=cupy` or `cunumpy.set_backend("cupy")`, queried with `cunumpy.get_backend()`, cunumpy ≥ 0.2.0) selects the CUDA kernels; with NumPy the pyccel kernels run as today.
- **No conversions at call time.** When a kernel is called, its arguments are already in the right format. There are no host/device copies per kernel call.
- **Data already lives on the GPU.** On the CuPy backend, `xp` is `cupy`, so markers, spline coefficients etc. are CuPy arrays from the start. The CUDA argument objects only collect *references* to these arrays and raise if they get host arrays.
- **No silent CPU fallback on the GPU.** A kernel without a CUDA version raises an error on the GPU backend. Falling back would mean copying data to the host and back at every call.
- **Small steps.** Every PR keeps the CPU code path working and tested.

## Current state (PR 1)

| File | Content |
|---|---|
| `src/struphy/utils/kernel_backends.py` | `is_cuda_backend()`, `CudaKernel` (wraps a `cupy.RawKernel`, compiled lazily; replaces the argument classes by their `values` at each call, takes `n_threads`), `Kernel` (pyccel/CUDA pair, `get_kernel()` picks by backend) |
| `src/struphy/utils/cuda_arguments.py` | `CudaMarkerArguments`, `CudaDomainArguments`: same constructor arguments as the pyccel classes, hold CuPy arrays, flatten them into the CUDA kernel arguments |
| `src/struphy/pic/tests/test_kernel_backends.py` | the demo kernel pair `push_eta_linear` (pyccel function compiled with `epyccel` at test time, CUDA source string) and tests on both backends |

Things we learned in the proof of concept:

- The pyccel-compiled argument classes (`MarkerArguments`, `DomainArguments`, `DerhamArguments`) hold references to their owner's arrays, but only accept **NumPy** arrays. Hence the CUDA counterparts in `cuda_arguments.py`.
- Today, `Particles` builds `args_markers` from `_to_numpy_for_kernel(self.markers)`, i.e. from a **host copy** when the backend is CuPy. The same holds for `Domain` and `Derham`.
- `Particles6D` and `Derham` cannot be created on the CuPy backend yet (PR 6, PR 7). `Domain` (e.g. `Cuboid`) can, and all its arrays are already CuPy arrays.
- `cupy.RawKernel` accepts only device arrays (host arrays raise) and does **not** check the kernel signature. Each argument is read with the size declared in the signature, so Python `int`/`float` arrive correctly in `int`/`double` parameters, but a wrongly typed scalar (e.g. an integer for a `double`, or a value that overflows an `int`) gives a wrong value **without an error**. Casting Python scalars in `CudaKernel` does not prevent this, so it is not done; see the follow-up in [Open questions](#open-questions).
- Flattening the argument classes at each call (joining their `values`) costs well under 1 µs, compared to about 70 µs for launching the kernel.
- `struphy compile` compiles every `.py` file whose name contains `kernels`. Non-pyccel modules must not contain `kernels` in their name; `.cu` files are ignored by it.
- On an H100, the demo kernel pushes 10⁶ markers in about 0.13 ms per step.

## Target layout

Each kernel gets its own folder with the pyccel and the CUDA version side by side. The `__init__.py` of the parent folder defines the catalog of the kernels in it:

```
src/struphy/pic/pushing/kernels/
├── __init__.py                          # catalog = KernelCatalog.from_package(__name__)
├── push_eta_stage/
│   ├── __init__.py
│   ├── push_eta_stage_kernels.py        # pyccel (compiled by `struphy compile`)
│   └── push_eta_stage_cuda.cu           # CUDA (compiled at runtime by CuPy/NVRTC)
├── push_vxb_analytic/
│   ├── __init__.py
│   └── push_vxb_analytic_kernels.py     # no CUDA version yet -> error on the GPU
└── ...
```

Conventions:

- The folder name, the pyccel function name, the CUDA `extern "C" __global__` function name and the catalog key are all the same.
- The pyccel file is `<name>_kernels.py` (so `struphy compile` picks it up); the CUDA file is `<name>_cuda.cu`.
- Shared CUDA code (device helper functions, argument structs) lives in `.cuh` headers next to the pyccel module it mirrors, e.g. `bsplines/bsplines_kernels.cuh` for `bsplines/bsplines_kernels.py`.

The geometry domains already follow a similar layout (`geometry/domains/cuboid/cuboid_kernels.py`), which can be extended with `cuboid_cuda.cuh` for the device version of the mapping.

Usage at a call site (e.g. in a propagator):

```python
from struphy.pic.pushing.kernels import catalog

kernel = catalog["push_eta_stage"]  # Kernel: pyccel or CUDA depending on the backend
```

## Details per PR

### PR 2: CUDA source files

- `CudaKernel.from_file(path, name)` reads `<name>_cuda.cu`. The kernel is compiled lazily on first call. CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), so the compile cost is paid once per machine.
- Headers are found through NVRTC include paths (`cupy.RawModule(code=..., options=("-I<struphy src>",))`), so a `.cu` file can `#include "struphy/bsplines/bsplines_kernels.cuh"`.
- Add `"**/*.cu"` and `"**/*.cuh"` to `[tool.setuptools.package-data]` in `pyproject.toml`.
- Move the test kernel's CUDA source from the string in `test_kernel_backends.py` into a `.cu` file.

### PR 3: Kernel catalog

- `KernelCatalog.from_package(package)` scans the subfolders of a package. For every `<name>/<name>_kernels.py` it creates a `Kernel` with `PyccelKernel(<name>_kernels.<name>)` and, if `<name>/<name>_cuda.cu` exists, a `CudaKernel`. This is what keeps the work for developers minimal: they add files, not registration code.
- `Kernel` without a CUDA version: on the CuPy backend, `get_kernel()` raises

  ```
  NotImplementedError: No CUDA version of kernel 'push_vxb_analytic' (expected .../push_vxb_analytic/push_vxb_analytic_cuda.cu).
  ```

- The error should come as early as possible: propagators/pushers call `get_kernel()` when they are set up, not only at the first time step. Then a GPU run fails right away instead of after the initialization.
- A small overview, e.g. `struphy compile --status` also printing "CUDA kernels: 3 of 60", helps to see what is left to port.

### PR 4: `Pusher` and propagators use `Kernel`

- `Pusher` currently asserts `isinstance(kernel, PyccelKernel)` (`pic/pushing/pusher.py`). Allow `Kernel` there and use `kernel.name` for profiling as today.
- On the GPU, the pusher passes the CUDA argument objects (`particles.cuda_args_markers`, `domain.cuda_args_domain`, ...) instead of the pyccel ones. The choice is made once when the pusher is set up, together with the kernel. This avoids calling a CUDA kernel with pyccel arguments or vice versa.

### PR 5–7: Owners build their CUDA arguments

- `Particles`, `Domain` and `Derham` own the arrays, so they build the CUDA argument objects from their own `xp` arrays (`cuda_args_markers`, `cuda_args_domain`, `cuda_args_derham`), in the same place where the pyccel argument objects are built today. The pyccel `args_*` stay as they are: they are used by 30+ modules through pyccel kernels.
- The CUDA argument objects hold references. If an owner reallocates an array (today the markers are allocated once), it must rebuild its CUDA arguments at the same place, exactly like for the pyccel arguments.
- First these classes must be creatable on the CuPy backend at all:
  - `Particles`: `xp.prod`/`xp.sum` on Python lists and similar (`pic/base.py`), probably more.
  - `Derham`: NumPy arrays from feectools reach `cupy.ascontiguousarray`.
  - `Domain`: deepcopy on CuPy fails (see PR 5 above).

### PR 8: Argument structs in shared headers

Today every CUDA kernel repeats the full flat signature (26 parameters for markers and domain alone). CuPy does not check it, so adding a field to `MarkerArguments` would shift all following arguments of all CUDA kernels **without an error**.

- Define `struct MarkerArgs { double* markers; bool* valid_mks; int n_markers; ... };` etc. in `kernel_arguments/pusher_args.cuh`, and pass one struct per argument class.
- To check first: how to pass a struct to a `cupy.RawKernel` (e.g. as a NumPy structured scalar with pointer fields). If this does not work well, keep the flat signature, but generate it from the Python class so that it is defined in one place.
- A test compares the struct layout (field names, types, order) with the Python class.

### PR 9: One folder per kernel

- Start with `pic/pushing` (`pusher_kernels.py`: 19 kernels, `pusher_kernels_gc.py`: 15, `pusher_kernels_sph.py`: 3, `eval_kernels_gc.py`: 5), later `pic/accumulation` (`accum_kernels.py`: 8, `accum_kernels_gc.py`: 8), then the remaining modules as needed.
- Pure refactor: move each kernel into `<name>/<name>_kernels.py`, update the imports at the call sites and in tests. No CUDA code in this PR.
- Things to check:
  - pyccel dependencies between kernel modules are found through imports (`# do not remove; needed to identify dependencies`); the new modules must keep these imports.
  - Many small pyccel modules instead of a few large ones: compile time with `struphy compile -j N`, and the import time of many `.so` files.
  - A re-export module for the old import paths must not have `kernels` in its name, otherwise `struphy compile` tries to compile it.

### PR 10: Device helper functions

The pusher kernels call helpers from other pyccel modules: B-spline evaluation (`bsplines_kernels`, `evaluation_kernels_3d`), mapping evaluation (`geometry/evaluation_kernels`, one module per domain), small linear algebra (`linalg_kernels`), boundary conditions (`pusher_utilities_kernels`). Each needs a `__device__` version in a `.cuh` header before the kernels using it can be ported.

- Port only what the next kernel needs, not whole modules at once.
- Scratch arrays that the pyccel classes allocate once (e.g. `DerhamArguments.bn1`, ..., `bd3`) become per-thread local arrays in CUDA (fixed maximum spline degree, or template parameter).
- Each device helper gets a test against its pyccel version through a small test kernel.

### PR 11: First real kernel: `push_eta_stage`

- Write `push_eta_stage_cuda.cu`, using the device helpers from PR 10.
- Parity test: same markers, both backends, results agree to round-off (`rtol ~ 1e-13`).
- End-to-end: run a propagator that only needs this kernel with `ARRAY_BACKEND=cupy`, and check that no host/device transfers happen inside the time loop (e.g. with `nsys` or by counting CuPy memory copies).

### PR 12+: Port kernels one by one

For each kernel: add `<name>_cuda.cu`, a parity test is added automatically by the catalog (every kernel with a CUDA version is run on both backends with the same inputs), and the kernel is removed from the "missing" list.

## Porting order

Port the kernels in the order the target models need them, so that complete models can run on the GPU as early as possible. Proposed:

1. `push_eta_stage` (PR 11) and the helpers it needs.
2. The remaining 6D full-orbit pushers (`pusher_kernels.py`), e.g. `push_vxb_analytic`, `push_v_with_efield`.
3. The accumulation kernels these models need (`accum_kernels.py`). Note: accumulation writes to shared grid arrays from many threads, so it needs atomics or a sort-then-reduce strategy. This is a design question of its own.
4. Guiding-center pushers and evaluations (`pusher_kernels_gc.py`, `eval_kernels_gc.py`, `accum_kernels_gc.py`).
5. SPH kernels.

## Testing

- Every kernel with a CUDA version has a parity test (pyccel vs. CUDA, same inputs), generated from the catalog.
- CUDA tests are skipped when no GPU is available (`cunumpy.cupy_available()`), so the normal CI keeps working. A GPU runner in CI runs them.
- Regression tests on the CPU (`pic/tests/test_pushers.py`, `pic/tests/test_kernel_setup.py`, ...) must pass in every PR.

## Open questions

- **Scalar types.** Scalars are not checked against the kernel signature (see [Current state](#current-state-pr-1)). `CudaKernel` could read the parameter types from the `extern "C"` signature once, when it is created, and cast each scalar to its declared type or raise if it does not fit (e.g. a Python `float` for an `int`, or an overflowing integer). This could go together with PR 8.

- **Marker layout.** The markers array is row-major (`n_markers × n_cols`). With one thread per marker, the memory accesses are strided. This is fine for now (each thread reads a few neighbouring columns), but a column-major or struct-of-arrays layout may be faster later. This would affect the CPU code too, so it is out of scope here.
- **MPI + GPUs.** One GPU per MPI rank (`cunumpy.set_device(rank % n_gpus)`), and GPU-aware MPI for the marker exchange, so markers do not go through the host.
- **Single-source alternatives.** Before porting a large number of kernels by hand, it may be worth checking whether some of them can be generated from the Python source (e.g. `cupyx.jit` or numba-cuda) instead of written twice. Hand-written CUDA stays the default.
- **Kernel launch configuration.** One thread per marker with a fixed block size for now. Kernels over grid points (accumulation, FEEC) will need their own launch sizes, so `CudaKernel` will need a way to set them.
