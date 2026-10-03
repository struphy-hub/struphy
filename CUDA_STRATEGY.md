# CUDA strategy for Struphy kernels

Plan for running Struphy's compute kernels on NVIDIA GPUs, next to the existing pyccel (CPU) kernels.
The work is split into small PRs that can be reviewed and merged one at a time. Nothing here has to be done in one go.

Progress is tracked in struphy-hub/struphy#650.

## PR checklist

- [x] **PR 1: Proof of concept** (#643)
  `Kernel` (pyccel/CUDA pair), `CudaKernel`, `CudaMarkerArguments`/`CudaDomainArguments`, and one test file with a demo kernel pair run on both backends. Also adds this document and the `gpu` optional dependency.
- [x] **PR 2: CUDA source files** (#644) — `CudaKernel` loads CUDA source from a `<name>_cuda.cu` file next to the pyccel file; `.cu`/`.cuh` files are shipped as package data.
- [x] **PR 3: Kernel catalog** (#645) — kernels are defined once in the `__init__.py` of the folder that contains them; a missing CUDA kernel raises an error on the GPU backend.
- [x] **PR 4: `Pusher` accepts `Kernel`** (#646) — the kernel for the active backend is chosen once, when the pusher is created; a plain `PyccelKernel` is wrapped, so the propagators do not change (no behaviour change on CPU).
- [x] **PR 5: `Domain` on the GPU** (#649) — `Domain.args_domain` is selected at construction; CUDA arguments reference device arrays, and deepcopy/unpickling rebuilds them from the copied or restored arrays.
- [x] **PR 6: `Particles` on the GPU** (#651) — `Particles` can be created on the CuPy backend, and `Particles.args_markers` is selected at construction.
- [x] **PR 7: `Derham` on the GPU** (#652) — `Derham` can be created on the CuPy backend, and `Derham.args_derham` is selected at construction.
- [x] **PR 8: Shared CUDA header for the argument classes** (#653, open) — one struct per argument class in `kernel_arguments/pusher_args.cuh`, passed by value, instead of long flat kernel signatures.
- [x] **PR 9: One folder per kernel, starting with `pic/pushing`** (#663, open) — pure refactor; the propagators take their pusher and evaluation kernels from `struphy.pic.pushing.kernels.catalog`.
- [ ] **PR 10: Device versions of helper kernels** — B-spline evaluation, mapping evaluation (per domain), small linear algebra, boundary conditions, as `__device__` functions in `.cuh` headers, each tested against its pyccel version. Starts with an H100 run of the PR 8 struct tests.
- [ ] **PR 11: First real CUDA kernel** — `push_eta_stage` with a pyccel/CUDA parity test and an end-to-end run on the GPU without host/device transfers in the time loop.
- [ ] **PR 12+: Port kernels one by one**, in the order they are needed by the models we want on the GPU (see [Porting order](#porting-order)). Accumulation needs its own design step first.
- [ ] **Later: move the kernel infrastructure to cunumpy** — once a few kernels run on the GPU and the struphy classes are understood and stable, replace `Kernel`, `KernelCatalog`, `CudaKernel` and `Argument` by their cunumpy counterparts (see [Moving to cunumpy](#moving-to-cunumpy-later)). Kernels and device helpers do not change.
- [ ] **CI**: a GPU runner that runs the CUDA tests (can happen any time; until then the GPU tests are run by hand on an H100 before each PR that touches CUDA code is merged).

Unrelated bugs found along the way go into their own PRs, not into these ones.

## Goal

A developer who adds a new kernel (e.g. for a new model) should only have to:

1. write the pyccel kernel `<name>_kernels.py` as today, and
2. optionally write `<name>_cuda.cu` **in the same folder**.

Everything else (loading, dispatch, argument passing, tests for agreement between the two versions) is done by the infrastructure.
CUDA kernels can be added one by one. If the code runs on the GPU and needs a kernel that has no CUDA version yet, it raises a clear error instead of silently falling back to the CPU.

## Principles

- **1:1 correspondence.** Each CUDA kernel has the same name and the same arguments (in the same order) as its pyccel kernel. The CUDA kernel takes the CUDA versions of the argument classes, plus the number of threads (`n_threads`).
- **The backend decides.** The cunumpy backend (`ARRAY_BACKEND=cupy` or `cunumpy.set_backend("cupy")`, queried with `cunumpy.get_backend()`) selects the CUDA kernels; with NumPy the pyccel kernels run as today.
- **No conversions at call time.** When a kernel is called, its arguments are already in the right format. There are no host/device copies per kernel call.
- **Data already lives on the GPU.** On the CuPy backend, `xp` is `cupy`, so markers, spline coefficients etc. are CuPy arrays from the start. The CUDA argument objects only collect *references* to these arrays and raise if they get host arrays.
- **No silent CPU fallback on the GPU.** A kernel without a CUDA version raises an error on the GPU backend. Falling back would mean copying data to the host and back at every call.
- **Kernels see only the C structs.** CUDA kernels and `__device__` helpers take `MarkerArgs`, `DomainArgs` and `DerhamArgs` from `pusher_args.cuh` and nothing else of the Python side. The Python classes that fill the structs can change (and will, when they move to cunumpy) without touching a single kernel.
- **Own infrastructure first, cunumpy later.** The kernel infrastructure (`Kernel`, `KernelCatalog`, `CudaKernel`, `Argument`) stays in struphy while it is being designed and while we learn what the kernels need. [cunumpy](https://github.com/struphy-hub/cunumpy), which we maintain and release on PyPI ourselves, already has equivalents (see [Moving to cunumpy](#moving-to-cunumpy-later)); the move happens when the struphy classes are stable, as a mechanical replacement.
- **Small steps.** Every PR keeps the CPU code path working and tested.

## Current state (after PR 9)

| File | Content |
|---|---|
| `src/struphy/utils/kernel_backends.py` | `is_cuda_backend()`, `CudaKernel` (wraps a `cupy.RawKernel`, compiled lazily with the struphy headers on the include path; expands `Argument.get_cuda_args()` and takes `n_threads`), `Kernel` and `KernelCatalog` for backend selection and discovery |
| `src/struphy/utils/cuda_arguments.py` | `Argument` base class plus `CudaMarkerArguments`, `CudaDerhamArguments` and `CudaDomainArguments`; each references its device arrays and packs them once into its C struct, which `get_cuda_args()` returns |
| `src/struphy/kernel_arguments/pusher_args.cuh` | the C structs `MarkerArgs`, `DerhamArgs` and `DomainArgs` that CUDA kernels take in place of the pyccel argument classes |
| `src/struphy/geometry/base.py`, `src/struphy/pic/base.py`, `src/struphy/feec/psydac_derham.py` | `Domain.args_domain`, `Particles.args_markers` and `Derham.args_derham` are the pyccel bundle on NumPy and the CUDA bundle on CuPy, selected once at construction; each owner also keeps a private pyccel bundle (`_pyccel_args_*`) for direct pyccel calls |
| `src/struphy/pic/pushing/kernels/` | the 43 pusher and marker evaluation kernels of `pic/pushing`, one folder each, and their `catalog` (none has a CUDA version yet) |
| `src/struphy/pic/tests/test_kernel_backends.py` | the demo kernel pair `push_eta_linear` (pyccel function compiled with `epyccel` at test time, CUDA source string) and tests on both backends |

Things we learned so far:

- The pyccel-compiled argument classes (`MarkerArguments`, `DomainArguments`, `DerhamArguments`) hold references to their owner's arrays, but only accept **NumPy** arrays, and a pyccel class cannot inherit from a Python base class. Hence the CUDA counterparts are separate Python classes that hold the same arrays.
- `cupy.RawKernel` accepts only device arrays (host arrays raise) and does **not** check the kernel signature: a wrongly typed scalar or a shifted argument gives a wrong value **without an error**. This is why the argument classes are passed as structs (PR 8). Scalars passed directly to a kernel (`dt`, `stage`) are still unchecked; see [Open questions](#open-questions).
- Flattening the argument classes at each call costs well under 1 µs, compared to about 70 µs for launching the kernel. Since PR 8 each class is one struct, packed once when the argument object is created.
- `struphy compile` compiles every `.py` file whose name contains `kernels`. Non-pyccel modules must not contain `kernels` in their name; `.cu` files are ignored by it.
- Kernel names can have at most 48 characters: with the Fortran backend, pyccel names the wrapper module `bind_c_<name>_kernels`, and Fortran names have at most 63 characters (PR 9).
- Running the particle tests without an MPI launcher uses feectools' `MockComm`, whose collectives do nothing. Code that gathers into a buffer must fill its own entry first (fixed in PR 6).
- On an H100, the demo kernel pushes 10⁶ markers in about 0.13 ms per step (PR 1, flat arguments). Passing the structs by value (PR 8) has **not** run on a GPU yet; this is the first thing PR 10 does.

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
- Shared CUDA code (device helper functions, argument structs) lives in `.cuh` headers next to the pyccel module it mirrors, e.g. `bsplines/bsplines_kernels.cuh` for `bsplines/bsplines_kernels.py`. A `.cu` file includes them relative to the source root, e.g. `#include "struphy/kernel_arguments/pusher_args.cuh"`.
- Kernels read the markers array only through the macro `MARKER(args, ip, j)`, to be added to `pusher_args.cuh` in PR 10 (`args.markers[ip * args.n_cols + j]`). If the layout of `MarkerArgs.markers` changes later (strided view, struct of arrays), only the macro changes.

The geometry domains already follow a similar layout (`geometry/domains/cuboid/cuboid_kernels.py`), which can be extended with `cuboid_cuda.cuh` for the device version of the mapping.

Usage at a call site (e.g. in a propagator):

```python
from struphy.pic.pushing.kernels import catalog

kernel = catalog["push_eta_stage"]  # Kernel: pyccel or CUDA depending on the backend
```

## Details per PR

### PR 2: CUDA source files (complete)

- `CudaKernel.from_file(path)` reads `<name>_cuda.cu`; the kernel name is taken from the file name. The kernel is compiled lazily on first call. CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), so the compile cost is paid once per machine. Note: CuPy keys the cache on the source text and options only, so a changed `.cuh` header does not trigger a recompile by itself; delete `~/.cupy/kernel_cache` after editing a header (or add an include hash to the options, which cunumpy does).
- `"**/*.cu"` and `"**/*.cuh"` are package data in `pyproject.toml`.
- Shared headers are found through the NVRTC include path (`-I<folder containing the struphy package>`, added in PR 8).

### PR 3: Kernel catalog (complete)

- `KernelCatalog.from_package(package)` scans the subfolders of a package. For every `<name>/<name>_kernels.py` it creates a `Kernel` with `PyccelKernel(<name>_kernels.<name>)` and, if `<name>/<name>_cuda.cu` exists, a `CudaKernel`. Developers add files, not registration code.
- `Kernel` without a CUDA version: on the CuPy backend, `get_kernel()` raises `NotImplementedError: No CUDA version of kernel 'push_vxb_analytic' (expected .../push_vxb_analytic_cuda.cu)`.
- The error comes as early as possible: propagators/pushers call `get_kernel()` when they are set up, not at the first time step.
- `catalog.missing_cuda` lists the kernels without a CUDA version.

### PR 4: `Pusher` accepts `Kernel` (complete)

- `Pusher` takes a `Kernel` or a `PyccelKernel` (wrapped into a `Kernel` without CUDA version) and calls `get_kernel()` once in its constructor.
- The propagators did not change; they switched to catalog lookups in PR 9.

### PR 5–7: Owners build their CUDA arguments (complete)

- `Particles`, `Domain` and `Derham` own the arrays, so they build the CUDA argument objects from their own `xp` arrays. `args_markers`, `args_domain` and `args_derham` are the pyccel bundle on NumPy and the CUDA bundle on CuPy, selected once at construction; each owner keeps a private pyccel bundle (`_pyccel_args_*`) for existing direct pyccel calls.
- The CUDA argument objects hold references. If an owner reallocates an array, it must rebuild its arguments at the same place, exactly like for the pyccel arguments.
- `Domain` (PR 5): arrays that already have the dtype and layout the kernels expect (`float64`/`int64`, C-contiguous) are referenced, not copied; otherwise one device copy is made when the arguments are built (e.g. `degree`, a tuple). Arguments are recreated after deepcopy or unpickling. Spline mappings (e.g. `IGAPolarCylinder`) cannot be created on CuPy yet (`interp_mapping` passes CuPy arrays to `scipy.sparse`), so CUDA `args_domain` exists only for analytic mappings for now.
- `Particles` (PR 6): particle arrays, validity masks and boundary-condition codes are allocated through `xp`. Scalar MPI gathers use small NumPy buffers (the own entry filled first, see `MockComm` above) and copy the result back to the active backend.
- `Derham` (PR 7): depends on feectools running on the CuPy backend (struphy-hub/feectools#85, merged). Data that describes the spline spaces (knots, quadrature and projection grids, `spline_types_pyccel`) is host data on every backend; only coefficients and stencil matrices live on the device. `domain_array`, `index_array(_N/_D)` and `neighbours` are gathered with NumPy MPI buffers and converted with `xp.asarray`. `args_derham` is built from the host knots, degrees and starts on NumPy, and from one device copy of them on CuPy. The pyccel scratch arrays (`bn1`, ..., `bd3`) are not part of the CUDA arguments; they become per-thread local arrays in CUDA (PR 10). Not supported on CuPy yet: local projectors (`NotImplementedError` at construction), polar splines (need a spline mapping), and field evaluation (`SplineFunction.__call__` still calls pyccel kernels with device coefficients).

### PR 8: Argument structs in a shared header (complete, open)

Before, every CUDA kernel repeated the full flat signature (26 parameters for markers and domain alone). CuPy does not check it, so adding a field to `MarkerArguments` would have shifted all following arguments of all CUDA kernels **without an error**.

- `kernel_arguments/pusher_args.cuh` defines `struct MarkerArgs`, `DerhamArgs` and `DomainArgs`. A kernel takes them by value, e.g. `void push_eta_linear(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain)`, and reads `args_markers.markers`, `args_markers.n_markers`, ... The member names are the attribute names of the pyccel classes; the only CUDA-specific member is `MarkerArgs.n_cols`.
- Each `Argument` subclass lists its members as `fields = (("double*", "markers"), ...)`, from which `struct_dtype()` builds a NumPy structured dtype with `align=True` (C alignment and padding). Pointer members are `uint64` holding `array.data.ptr`. The struct (a `numpy.void`, which CuPy passes by value) is packed once, in the constructor, so a kernel call does no extra work.
- The struct holds device addresses: it is repacked when an argument object is deepcopied or unpickled (`__getstate__`/`__setstate__`), and owners that reallocate an array must rebuild their argument object, as before.
- Scalar members are range-checked when packing: a `float` for an `int` member raises `TypeError`, a value that does not fit raises `OverflowError`.
- Tests: the header is parsed and compared with `fields` (names, C types, order) without a GPU; on a GPU, an NVRTC-compiled kernel reports `sizeof` and the member offsets, which are compared with the dtype. On the host, `offsetof`/`sizeof` from a C++ compiler agree with the dtypes (x86-64/arm64 lay these structs out like CUDA).
- Why structs and not flat parameters: see the discussion in #653. The struct is what lets a `__device__` helper take one `const DomainArgs&` instead of 12+ forwarded parameters (PR 10), and it mirrors how the pyccel helpers take `args_domain`. What is given up: CuPy's own rejection of host arrays at launch (`_cupy_array()` does it at construction instead) and compatibility with `cupyx.jit`-style kernels (which could take the flat tuple from `fields` if ever needed).
- To add in PR 10: the `MARKER(args, ip, j)` macro, and a check in `_cupy_array()` that the array is on the current device (neither CuPy's `RawKernel` nor the struct checks it; it matters with one GPU per MPI rank).

### PR 9: One folder per kernel (complete, open)

- The 43 kernels of `pic/pushing` (`pusher_kernels.py`: 16, `pusher_kernels_gc.py`: 15, `pusher_kernels_sph.py`: 3, `eval_kernels_gc.py`: 5, `eval_kernels_sph.py`: 4) are now in `pic/pushing/kernels/<name>/<name>_kernels.py`; the five old modules are removed. The function bodies are unchanged (checked by comparing the ASTs with the old modules).
- `pusher_utilities_kernels.py` stays a shared module: it holds helpers called by the kernels (boundary conditions), not kernels. It gets its device version in PR 10.
- Each new module imports only what its kernel uses, plus `pusher_args_kernels` as a module (`# do not remove; needed to identify dependencies`): `dependencies.py` only sees imported modules.
- Call sites use the catalog: `catalog["push_eta_stage"]` is passed to `Pusher`, and `KernelSetup` also accepts a `Kernel`, resolved with `get_kernel()` when the setup is created. `CurrentCoupling5DGradB` calls its pusher kernels itself and resolves them once. `Particles` runs the SPH evaluation kernels on its host bundle with `catalog[...].pyccel_kernel`.
- No re-export modules for the old import paths (they would have to contain `kernels` in their name). Code outside struphy that imports e.g. `struphy.pic.pushing.pusher_kernels` has to switch to the catalog.
- `push_gc_bxEstar_discrete_gradient_1st_order_newton` was renamed to `push_gc_bxEstar_dg_1st_order_newton` (Fortran name length); `test_pushing_catalog` checks the layout and the name lengths.
- Import time: loading the catalog (43 compiled modules) takes about 75 ms, once per process.
- `pic/accumulation` (`accum_kernels.py`: 8, `accum_kernels_gc.py`: 8) is split the same way when the first accumulation kernel is ported (PR 12+); the remaining kernel modules as needed.

### PR 10: Device helper functions

The pusher kernels call helpers from other pyccel modules: B-spline evaluation (`bsplines_kernels`, `evaluation_kernels_3d`), mapping evaluation (`geometry/evaluation_kernels`, one module per domain), small linear algebra (`linalg_kernels`), boundary conditions (`pusher_utilities_kernels`). Each needs a `__device__` version in a `.cuh` header before the kernels using it can be ported.

- First: run the PR 8 GPU tests (`test_cuda_struct_layout`, `test_cuda_struct_scalars_are_checked`, the demo kernels) on an H100. Everything below assumes structs arrive by value correctly.
- Port only what `push_eta_stage` needs: `get_spans` and the basis functions for the spline degrees in use, the analytic mapping evaluation (`kind_map` switch, starting with `Cuboid`), the 3×3 linear algebra, and the kinetic boundary conditions. Everything else comes with the kernel that needs it.
- Device helpers take the argument structs by reference (`const DomainArgs&`), exactly like the pyccel helpers take `args_domain`, and read markers through `MARKER(args, ip, j)`.
- Scratch arrays that the pyccel classes allocate once (e.g. `DerhamArguments.bn1`, ..., `bd3`) become per-thread local arrays in CUDA with a fixed maximum spline degree (compile-time constant, checked against `pn` when the arguments are built) or a template parameter.
- Tests: each device helper is called from a small elementwise test kernel (one thread per evaluation point, inputs and outputs as arrays) and compared with its pyccel version from Python. One generic test-kernel template in `test_kernel_backends.py` (or a new `test_device_helpers.py`) serves all scalar-returning helpers. Tests are skipped without a GPU.
- Headers are included through the source root (`#include "struphy/bsplines/bsplines_kernels.cuh"`). Remember the CuPy cache caveat from PR 2 when editing headers.

### PR 11: First real kernel: `push_eta_stage`

- Write `push_eta_stage_cuda.cu`, using the device helpers from PR 10; one thread per marker, `if (ip >= args_markers.n_markers) return;`.
- Parity: same markers on both backends, results agree to round-off (`rtol ~ 1e-13`). The test is written once against the catalog: it parametrizes over every kernel that has a CUDA version (`[n for n in catalog.names if n not in catalog.missing_cuda]`), builds the arguments with a per-kernel factory, runs the pyccel and the CUDA kernel, and compares every array argument. Later kernels then only add their factory.
- `Pusher` on the CuPy backend passes `n_threads=particles.markers.shape[0]` and device arrays in `args_kernel` (the Butcher tableau arrays `a_stage`, `b`, `c` are tiny NumPy arrays today; `PushEta` makes device copies once at setup).
- End-to-end: run `PushEta` (a propagator that only needs this kernel) with `ARRAY_BACKEND=cupy` on 1 and 2 ranks, and check that no host/device transfers happen inside the time loop: `nsys profile` (look for `cudaMemcpy` between the kernel launches) or monkeypatch `cupy.ndarray.get`/`cupy.asarray` to count calls during the loop.

### PR 12+: Port kernels and particle boundary handling

For each kernel: add `<name>_cuda.cu`; the parity test from PR 11 picks it up; `catalog.missing_cuda` shrinks (a `struphy compile --status` line "CUDA kernels: 3 of 43" would be a nice addition).

- Port the remaining particle boundary handling, including the `reflect` helper called from `Particles.apply_kinetic_bc`, so the whole push stays on the device. The CUDA path must use `Particles.args_markers` and `Domain.args_domain`; remove the temporary direct-pyccel use of `Domain._pyccel_args_domain` from this path once reflection runs in a CUDA kernel.
- Accumulation (before the first accumulation kernel): many markers write to the same grid cells. Options are atomics (`atomicAdd` on `double`, native since sm_60) and sort-then-reduce. Start with atomics, which match the pyccel structure one-to-one; measure before optimizing. The accumulated arrays are owned by feectools stencil vectors, which are host arrays on the NumPy backend and device arrays on CuPy (feectools#85), so the kernel writes into `vec._data` directly on both. Split `pic/accumulation` into one folder per kernel at the same time. Accumulation kernels over grid points need their own launch sizes; `CudaKernel` gets a `grid`/`block` override then.
- Field evaluation on the device (`SplineFunction.__call__` and friends) is needed by diagnostics and by some propagators; it reuses the B-spline device helpers from PR 10 with 3D launch shapes.

## Moving to cunumpy (later)

cunumpy `devel` (version 0.4.0, unreleased; PyPI has 0.3.0) contains equivalents of everything struphy built in PRs 1–8, with more checks. The move is deliberately postponed until a few kernels run on the GPU: keeping the small struphy classes while the design settles makes them easier to understand and change, and the kernels never see the Python side, so nothing written in PRs 10–12 has to change.

| struphy | cunumpy 0.4.0 |
|---|---|
| `utils.kernel_backends.Kernel`, `KernelCatalog`, `CudaKernel` | `xp.Kernel`, `xp.KernelCatalog.from_package` (same folder convention, source root on the include path by default), `xp.CudaKernel` (parses the `__global__` signature once and checks every call: argument count, array dtypes and contiguity, scalars cast and range-checked; 1D–3D launches via `n_threads` or `grid`; header-aware compile cache; `compile()` at setup; debug mode with `-lineinfo`, bounds checks and synchronized launches) |
| `utils.cuda_arguments.Argument` + `_pack()` | `xp.CudaStructArguments` (`struct_name`, `fields`, `pack()`, repack on copy/unpickle, `cls.struct.declaration`/`to_header()`), `xp.CudaStruct.from_signature` to derive a struct from pyccel-style annotations |
| the two bundles per owner (`args_*` and `_pyccel_args_*`) | `xp.KernelArguments`: one object with `__host_args__()` (the pyccel instance) and `__cuda_args__()` (the packed struct), resolved by `Kernel` and `PyccelKernel`, so call sites never branch on the backend |
| `double* markers` + `n_cols` + `MARKER()` macro | `Array2D<double>` view (pointer, shape, strides) from `cunumpy/array_view.cuh`; kernels write `markers(ip, j)` |
| the "reference or copy once" rule, written by hand | `xp.as_device_array(value, dtype, ndim)` |
| the parity test of PR 11 | `cunumpy.testing.assert_kernels_agree`, `KernelCatalog.parity_cases()`, `requires_cupy`, `backend` fixture |
| the device-helper test kernels of PR 10 | `cunumpy.testing.device_function_kernel(header, prototype)` |
| transfer check of PR 11 | `xp.count_transfers()`, `xp.assert_no_transfers()` |
| atomics, thread-index macros | `cunumpy/atomic.cuh`, `cunumpy/index.cuh` |
| one GPU per MPI rank | `xp.bind_local_device()`, `xp.require_cuda_aware_mpi()`, `xp.synchronize_for_mpi()` |

When the time comes:

1. Release cunumpy 0.4.0 on PyPI (its `[Unreleased]` changelog block) and pin it in `pyproject.toml`.
2. Delete `utils/kernel_backends.py`; the catalog `__init__.py` files, `Pusher` and `KernelSetup` use the cunumpy classes.
3. The three argument classes subclass `xp.CudaStructArguments` and implement `__host_args__()`, becoming the single `args_*` object of their owner on both backends; the `_pyccel_args_*` bundles and the backend branches that select between the bundles go away. The pyccel class stays a plain compiled class held as an attribute (it cannot inherit from anything).
4. `pusher_args.cuh` is generated from the Python `fields` (`to_header()`); a test asserts the committed header equals the generated one. Optionally switch `markers` to `Array2D<double>` by changing the `MARKER` macro.
5. Replace the hand-written parity, device-helper and transfer tests by the `cunumpy.testing` helpers.

What this buys, beyond less code in struphy: scalar checks against the kernel signature, launch shapes for grid kernels, the include-hash compile cache, debug mode, and the `KernelArguments` cleanup of the duplicate bundles.

Until then, pieces of cunumpy that are independent of the dispatch classes can be used as soon as they are released: `xp.bind_local_device()` and `xp.require_cuda_aware_mpi()` for MPI runs, `xp.as_device_array` in the argument classes.

## Porting order

The first target is **Vlasov** (`models/vlasov.py`), with `PushEta` and both
`PushVxB` algorithms. Its complete pusher set is `push_eta_stage`,
`push_vxb_analytic`, and `push_vxb_implicit`. CUDA support initially uses Cuboid
mappings and spline degrees 1–8. GPU execution is to be tested by the maintainer
before merging.

Next, in order:

1. Extend Vlasov to additional analytic mappings, then spline mappings; validate
   multi-rank device execution and boundary handling on the GPU.
2. Vlasov–Ampere and Vlasov–Maxwell: `push_v_with_efield`,
   `charge_density_0form`, and `vlasov_maxwell` accumulation. First split
   accumulation into a catalog and implement fp64 atomics into stencil storage;
   measure collisions and summation error before considering sort/reduce.
3. Linear Vlasov variants: weight pushers and `linear_vlasov_ampere` accumulation.
4. Remaining 6D current/pressure coupling pushers and accumulators.
5. Guiding-center pushers, evaluations, and accumulation.
6. SPH kernels, after device box sorting.

## Testing

- Every kernel with a CUDA version has a parity test (pyccel vs. CUDA, same inputs), generated from the catalog (PR 11).
- Each device helper is tested against its pyccel version through a small test kernel (PR 10).
- CUDA tests are skipped when no GPU is available (`cunumpy.cupy_available()`), so the normal CI keeps working. Until a GPU runner exists, the GPU tests are run by hand on an H100 before a PR that touches CUDA code is merged, and the PR description says so.
- Regression tests on the CPU (`pic/tests/test_pushers.py`, `pic/tests/test_kernel_setup.py`, the model tests, ...) must pass in every PR. Particle tests need an MPI launcher (`mpirun -n 1 pytest ...`).

## Open questions

- **Scalar types.** Members of the argument structs are checked when they are packed (PR 8). Scalars passed directly to a kernel (`dt`, `stage`) are not checked against the kernel signature: a Python `float` for an `int` parameter arrives wrong without an error. Either `CudaKernel` parses the `extern "C"` signature once and casts/checks (what cunumpy's `CudaKernel` does), or we accept it until the move to cunumpy. Until then: kernels take `double dt, int stage` in exactly this order, and the pyccel signature is the reference.
- **Kernel launch configuration.** One thread per marker with a fixed block size for now. Kernels over grid points (accumulation, FEEC) need their own launch sizes; `CudaKernel` needs a `grid`/`block` override (PR 12+), or cunumpy's.
- **Accumulation strategy.** Atomics vs. sort-then-reduce; see PR 12+. Decided by measurement on the first accumulation kernel.
- **Marker layout.** The markers array is row-major (`n_markers × n_cols`). With one thread per marker, the memory accesses are strided. This is fine for now (each thread reads a few neighbouring columns), but a column-major or struct-of-arrays layout may be faster later. This would affect the CPU code too, so it is out of scope here. The `MARKER()` macro keeps it open on the CUDA side.
- **MPI + GPUs.** One GPU per MPI rank (`xp.bind_local_device()` before `MPI_Init`, with feectools#86/#87), and GPU-aware MPI for the marker exchange, so markers do not go through the host. `xp.mpi_is_cuda_aware()` detects it; the exchange in `Particles.mpi_sort_markers` has to be checked for host staging buffers.
- **Single-source alternatives.** Hand-written CUDA stays the default. Generating whole kernels from the Python source (`cupyx.jit`, numba-cuda, or a pyccel CUDA backend) is worth a look before the guiding-center kernels (the largest ones) are ported. Those tools take flat arguments, which `fields` also provides.
- **Spline mappings and polar splines on the GPU** (PR 5/7 leftovers): needed once a model with an IGA mapping is run on the GPU.

## PR 10 implementation notes

Device headers now provide Cuboid Jacobians, matrix inversion/vector products,
spline span and N/D basis evaluation (degrees 1–8), and per-marker boundary
conditions. `DerhamArgs` carries knot lengths because raw device pointers have
no shape. Argument construction checks spline degrees and the current device.
GPU parity tests cover non-block-aligned batches, domain endpoints, and mixed
boundaries. H100 struct and helper validation is still required; the development
workspace is macOS and has no CUDA device.

## PR 11 implementation notes

`push_eta_stage` now has a CUDA implementation for Cuboid. Both signatures take
an explicit final `n_stages` argument because a raw CUDA tableau pointer has no
length. PushEta prepares tableau arrays once; Pusher passes the current marker
count at launch and rejects unsupported mappings during setup. Empty launches
are no-ops. Catalog parity tests require a factory for every ported kernel and
cover holes, boundary particles, Euler/RK4 and periodic/reflect/remove boundaries.
The single-rank pusher test guards host array conversions after warmup.
H100 execution and the two-rank no-transfer requirement remain unverified;
existing MPI sorting still synchronizes dynamic counts with the host.

## PR 12 implementation notes

Scoped to the Vlasov model: both magnetic rotation algorithms now have CUDA
versions, using tensor-product N/D spline evaluation and strided coefficient
views. The reflection path in Particles uses device marker/domain arguments.
SplineFunction has device evaluation for marker and meshgrid point sets.
Accumulation is deliberately left for the next Vlasov–Ampere/Maxwell step.
GPU tests are provided but not run here, at the maintainer's request.
