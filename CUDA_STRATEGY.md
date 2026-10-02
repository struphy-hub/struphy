# CUDA strategy for Struphy kernels

Plan for running Struphy's compute kernels on NVIDIA GPUs, next to the existing pyccel (CPU) kernels.
The work is split into small PRs that can be reviewed and merged one at a time. Nothing here has to be done in one go.

## PR checklist

- [x] **PR 1: Proof of concept** (branch `cuda-kernel-proof-of-concept`)
  `Kernel` (pyccel/CUDA pair), `CudaKernel`, `CudaMarkerArguments`/`CudaDomainArguments`, and one test file with a demo kernel pair run on both backends. Also adds this document and the `gpu` optional dependency.
- [x] **PR 2: CUDA source files** — `CudaKernel` loads CUDA source from a `<name>_cuda.cu` file next to the pyccel file; `.cu`/`.cuh` files are shipped as package data.
- [x] **PR 3: Kernel catalog** — kernels are defined once in the `__init__.py` of the folder that contains them; a missing CUDA kernel raises an error on the GPU backend.
- [x] **PR 4: `Pusher` accepts `Kernel`** — the kernel for the active backend is chosen once, when the pusher is created; a plain `PyccelKernel` is wrapped, so the propagators do not change (no behaviour change on CPU).
- [x] **PR 5: `Domain` on the GPU** — domain arguments are selected and stored at domain construction; CUDA arguments reference device arrays, and deepcopy/unpickling rebuilds the arguments from the copied or restored arrays.
- [x] **PR 6: `Particles` on the GPU** — `Particles` can be created on the CuPy backend, and `Particles.args_markers` is selected as the CUDA or Pyccel argument bundle at construction.
- [x] **PR 7: `Derham` on the GPU** — `Derham` can be created on the CuPy backend, plus `Derham.cuda_args_derham`.
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

## Current state (PR 7)

| File | Content |
|---|---|
| `src/struphy/utils/kernel_backends.py` | `is_cuda_backend()`, `CudaKernel` (wraps a `cupy.RawKernel`, compiled lazily; expands `Argument.get_cuda_args()` and takes `n_threads`), `Kernel` and `KernelCatalog` for backend selection and discovery |
| `src/struphy/utils/cuda_arguments.py` | `Argument` contract plus `CudaMarkerArguments`, `CudaDerhamArguments` and `CudaDomainArguments`; CUDA arrays are stored individually and returned in signature order by `get_cuda_args()` |
| `src/struphy/geometry/base.py` | `Domain.args_domain` is selected once at construction; CUDA domains use device arrays, while direct Pyccel geometry calls retain a host argument bundle |
| `src/struphy/pic/base.py` | `Particles` arrays and `args_markers` use the backend selected at construction; direct Pyccel methods retain a private host bundle |
| `src/struphy/pic/tests/test_kernel_backends.py` | the demo kernel pair `push_eta_linear` (pyccel function compiled with `epyccel` at test time, CUDA source string) and tests on both backends |

Things we learned in the proof of concept:

- The pyccel-compiled argument classes (`MarkerArguments`, `DomainArguments`, `DerhamArguments`) hold references to their owner's arrays, but only accept **NumPy** arrays. Hence the CUDA counterparts in `cuda_arguments.py`.
- `Particles.args_markers` is the CUDA or Pyccel bundle selected at construction. Existing direct Pyccel methods use a private host bundle. `Derham` still needs its CUDA creation work (PR 7).
- `Domain.args_domain` returns the argument type selected at domain construction; on CuPy it is a `CudaDomainArguments` object. Direct Pyccel methods on `Domain` use a separate internal host bundle.
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

- `CudaKernel.from_file(path)` reads `<name>_cuda.cu`; the kernel name is taken from the file name. The kernel is compiled lazily on first call. CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), so the compile cost is paid once per machine.
- Add `"**/*.cu"` and `"**/*.cuh"` to `[tool.setuptools.package-data]` in `pyproject.toml`.
- Later (PR 10), when the first shared header is needed: headers are found through NVRTC include paths (`cupy.RawModule(code=..., options=("-I<struphy src>",))`), so a `.cu` file can `#include "struphy/bsplines/bsplines_kernels.cuh"`.

### PR 3: Kernel catalog

- `KernelCatalog.from_package(package)` scans the subfolders of a package. For every `<name>/<name>_kernels.py` it creates a `Kernel` with `PyccelKernel(<name>_kernels.<name>)` and, if `<name>/<name>_cuda.cu` exists, a `CudaKernel`. This is what keeps the work for developers minimal: they add files, not registration code.
- `Kernel` without a CUDA version: on the CuPy backend, `get_kernel()` raises

  ```
  NotImplementedError: No CUDA version of kernel 'push_vxb_analytic' (expected .../push_vxb_analytic/push_vxb_analytic_cuda.cu).
  ```

- The error should come as early as possible: propagators/pushers call `get_kernel()` when they are set up, not only at the first time step. Then a GPU run fails right away instead of after the initialization.
- `catalog.missing_cuda` lists the kernels without a CUDA version. Later, a small overview, e.g. `struphy compile --status` also printing "CUDA kernels: 3 of 60", helps to see what is left to port.

### PR 4: `Pusher` accepts `Kernel`

- `Pusher` takes a `Kernel` or, as before, a `PyccelKernel` (wrapped into a `Kernel` without CUDA version). It calls `get_kernel()` once in its constructor, so on the CuPy backend a pusher whose kernel has no CUDA version fails when it is created, not in the time loop. Since no pusher kernel has a CUDA version yet, this is the case for all pushers.
- The propagators do not change. They switch to catalog lookups once the kernels are split into folders (PR 9).
- The pusher already passes the backend-selected `particles.args_markers` and `domain.args_domain`. PR 11 still needs to provide device arrays in `args_kernel` and the CUDA implementation of the first real pusher kernel.

### PR 5–7: Owners build their CUDA arguments

- `Particles`, `Domain` and `Derham` own the arrays, so they build CUDA argument objects from their own `xp` arrays. `Particles.args_markers` and `Domain.args_domain` are selected once at construction; private host bundles remain available for existing direct Pyccel calls.
- The CUDA argument objects hold references. If an owner reallocates an array (today the markers are allocated once), it must rebuild its CUDA arguments at the same place, exactly like for the pyccel arguments.
- First these classes must be creatable on the CuPy backend at all:
  - `Particles`: wrap Python lists in `xp.array` before reductions, and use host buffers for scalar MPI gathers (`pic/base.py`).
  - `Derham`: feectools and struphy moved host data (knots, grids, collocation matrices) to CuPy before calling pyccel kernels (see PR 7 below).
  - `Domain`: deepcopy and unpickling on CuPy failed (see PR 5 below).

### PR 5: `Domain` on the GPU (complete)

- `Domain.args_domain` is selected when the domain is created: `DomainArguments` for NumPy or `CudaDomainArguments` for CuPy. CUDA arguments are built from the domain's device arrays; NumPy data is never copied to the device.
- Arrays that already have the dtype and layout the CUDA kernels expect (`float64`/`int64`, C-contiguous) are referenced, not copied, e.g. the knot vectors `T` and `indN`. Otherwise one device copy is made when the arguments are built (e.g. `degree`, which is a tuple, or broadcast control points), never at kernel call time.
- The selected arguments and the internal host bundle are recreated after deepcopy or unpickling so they refer to the new domain's arrays.
- Direct Pyccel geometry methods use the host bundle; the public `args_domain` remains the backend-specific bundle selected at construction.
- Spline mappings (e.g. `IGAPolarCylinder`) cannot be created on the CuPy backend yet: `interp_mapping` passes CuPy arrays to `scipy.sparse.csc_matrix`. Consequently, CUDA `args_domain` is available only for analytic mappings for now; making spline mappings work on the GPU is left for when the first one is needed there.

### PR 6: `Particles` on the GPU (complete)

- Particle arrays, validity masks, and boundary-condition codes are allocated through `cunumpy`, so they live on CuPy when the CuPy backend is active.
- `args_markers` is built as `CudaMarkerArguments` from device arrays on CuPy, or as `MarkerArguments` on NumPy. A private host bundle remains for direct Pyccel calls.
- Domain decomposition now wraps the Python `nprocs` list with `xp.array` before calling `xp.prod`. Scalar MPI gathers use small NumPy buffers and copy the results back to the active array backend, avoiding unsupported CuPy buffers in MPI calls.
- Full GPU particle pushing still depends on CUDA versions of the required kernels.

### PR 7: `Derham` on the GPU (complete)

- feectools and the struphy code that builds `Derham` followed `xp` everywhere. So on CuPy, the knots, quadrature grids and decomposition metadata became device arrays and then reached pyccel kernels, SciPy or MPI, which only take host arrays.
  - feectools: struphy-hub/feectools#85, the first of the feectools CUDA PRs, makes feectools run on the CuPy backend. It must be merged, and the submodule or the feectools version bumped, before `Derham` can be created on CuPy.
  - struphy: data that describes the spline spaces is host data on every backend. Only the coefficients (`StencilVector` data) and stencil matrices live on the device. The projection and quadrature grids of `Derham` (`get_pts_and_wts`, ...) and `spline_types_pyccel` are NumPy. `domain_array`, `index_array(_N/_D)` and `neighbours` are gathered with NumPy MPI buffers and then converted with `xp.asarray`, so they are device arrays on CuPy like `Particles.domain_array`.
- `Derham.args_derham` is built from the host knots, degrees and starts on both backends. `Derham.cuda_args_derham` lazily builds `CudaDerhamArguments` with one device copy of these small arrays. On NumPy it raises (host arrays are never copied to the device). The pyccel scratch arrays (`bn1`, ..., `bd3`) are not part of it; they become per-thread local arrays in CUDA (PR 10).
- Not supported on CuPy yet:
  - Local projectors (`DerhamOptions.local_projectors=True`) raise `NotImplementedError` when the `Derham` is created. `CommutingProjectorLocal` builds its data with `xp` and calls pyccel kernels on it, like `Derham` did.
  - Polar splines need a spline mapping, which cannot be created on CuPy yet (see PR 5).
  - Field evaluation (`SplineFunction.__call__`, ...) still calls pyccel kernels with the coefficients, which are device arrays on CuPy. It needs CUDA evaluation kernels (PR 10+).
- Tests: `feec/tests/test_derham_gpu.py`. Without a GPU, a strict host stand-in for CuPy (rejects host/device mixing, cannot run kernels; not part of the repository) was used, on top of struphy-hub/feectools#85. With it, a `Derham` created on the "CuPy" backend matches the NumPy one on 1, 2 and 4 MPI processes.

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

### PR 12+: Port kernels and particle boundary handling

For each kernel: add `<name>_cuda.cu`, a parity test is added automatically by the catalog (every kernel with a CUDA version is run on both backends with the same inputs), and the kernel is removed from the "missing" list.

- Port particle kinetic boundary handling to CUDA, including the `reflect` helper currently called from `Particles.apply_kinetic_bc`. The CUDA path must use `Particles.args_markers` and `Domain.args_domain`; remove the temporary direct-Pyccel use of `Domain._pyccel_args_domain` from this path once reflection runs in a CUDA kernel.

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
