"""Micro-benchmark suite for the kernels ported to CUDA on this branch:
every ``*_general_gpu`` pusher (:mod:`struphy.pic.pushing.pusher_kernels_cuda`)
and the two accumulation kernels
(:mod:`struphy.pic.accumulation.accum_kernels_cuda`). For each kernel it
times the CPU (Pyccel) reference and the GPU (CuPy ``RawKernel``) port on
identical input data and prints a numpy-vs-cupy speedup table.

Run with:

    ARRAY_BACKEND=numpy python bench_gpu/bench_kernels.py

Options (see ``--help``): ``--n-markers``, ``--num-elements``, ``--degree``,
``--repeats``, ``--kernel`` (repeatable, to run a subset).

Why ``ARRAY_BACKEND=numpy`` for everything, GPU included
----------------------------------------------------------
The CPU kernels are Pyccel-compiled functions that need real NumPy buffers,
and markers are host-resident regardless of backend (see
``ISSUE_cupy_particles_never_pushed.md``). The GPU kernel wrappers import
``cupy`` directly inside each function body and don't consult
``cunumpy``'s active backend at all -- they just expect CuPy arrays as
arguments. So a single ``ARRAY_BACKEND=numpy`` process can build one set of
NumPy scene arrays, hand them straight to the CPU kernels, and hand
``cupy.asarray(...)`` mirrors of the *same* arrays to the GPU kernels: both
variants of every kernel run back-to-back on byte-identical input, in one
process, with no subprocess/backend-switching dance required.
"""

import argparse
import os
import time

if os.environ.get("ARRAY_BACKEND", "numpy") != "numpy":
    raise SystemExit(
        "Run this benchmark with ARRAY_BACKEND=numpy -- see the module docstring "
        "for why the GPU kernels don't need ARRAY_BACKEND=cupy to be benchmarked.",
    )

import numpy as np


def timeit(fn, repeats: int, warmup: int = 1) -> float:
    """Best-of-``repeats`` wall-clock time of ``fn()``, in seconds.

    Best-of (not mean) since the only noise on a shared cluster node is
    contention that slows a run down, never speeds one up -- the minimum is
    the closest thing to "this kernel's own cost" we can measure without a
    dedicated node. ``warmup`` calls run first and are excluded, absorbing
    the one-time CUDA context / RawKernel-compile cost of the first GPU call.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


class Scene:
    """One shared set of markers + domain + Derham + random FE coefficient
    fields, used to build every kernel case below. Field values are random
    (not physically meaningful) -- this benchmark measures raw kernel
    throughput, not physics, so only shapes/dtypes need to be realistic.
    """

    def __init__(self, n_elements, degree, n_markers_target, seed=1234):
        from struphy import domains
        from struphy.feec.mass import WeightedMassOperators
        from struphy.feec.psydac_derham import Derham
        from struphy.io.options import DerhamOptions
        from struphy.particles.parameters import LoadingParameters
        from struphy.pic.particles import Particles6D
        from struphy.topology.grids import TensorProductGrid

        # kind_map == 12 (Colella): the "general" (non-Cuboid) CUDA path,
        # i.e. the one that evaluates DF(eta) per marker instead of assuming
        # it's constant -- this is the actual new work ported this branch,
        # and the one every real (non-trivial-geometry) simulation uses.
        self.domain = domains.Colella(Lx=2.0, Ly=3.0, alpha=0.1, Lz=4.0)
        grid = TensorProductGrid(num_elements=n_elements)
        derham_opts = DerhamOptions(degree=degree)
        self.derham = Derham(grid, derham_opts, comm=None)
        self.mass_ops = WeightedMassOperators(self.derham, self.domain)

        loading_params = LoadingParameters(
            Np=n_markers_target,
            seed=seed,
            moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            spatial="uniform",
        )
        self.particles = Particles6D(loading_params=loading_params, domain=self.domain)
        self.particles.draw_markers()
        self.particles.initialize_weights()
        self.n_markers = self.particles.markers.shape[0]

        self.args_markers = self.particles.args_markers
        self.args_domain = self.domain.args_domain
        self.args_derham = self.derham.args_derham

        self._markers0 = self.particles.markers.copy()
        self._rng = np.random.default_rng(seed)

        self.pn = tuple(int(p) for p in self.args_derham.pn)
        self.starts = tuple(int(s) for s in self.args_derham.starts)
        self.kind_map = int(self.args_domain.kind_map)

        import cupy as cp

        self.params_dev = cp.asarray(np.asarray(self.args_domain.params, dtype=float), dtype=cp.float64)
        self.tn1_dev = cp.asarray(np.asarray(self.args_derham.tn1, dtype=float), dtype=cp.float64)
        self.tn2_dev = cp.asarray(np.asarray(self.args_derham.tn2, dtype=float), dtype=cp.float64)
        self.tn3_dev = cp.asarray(np.asarray(self.args_derham.tn3, dtype=float), dtype=cp.float64)

        # random FE coefficient fields, one per Derham space actually used
        # below (0-form/H1 scalar; 1-form/Hcurl, 2-form/Hdiv, vector/H1vec
        # each 3 components).
        self.fields = {
            "0": self._random_field("0"),
            "1": self._random_field("1"),
            "2": self._random_field("2"),
            "v": self._random_field("v"),
        }

    def _random_field(self, form: str):
        from feectools.linalg.block import BlockVector
        from feectools.linalg.stencil import StencilVector

        space = self.derham.coeff_spaces[form]
        if form in ("0", "3"):
            v = StencilVector(space)
            v._data[:] = self._rng.uniform(-1.0, 1.0, v._data.shape)
            return (v._data,)
        bv = BlockVector(space)
        arrs = []
        for bl in bv.blocks:
            bl._data[:] = self._rng.uniform(-1.0, 1.0, bl._data.shape)
            arrs.append(bl._data)
        return tuple(arrs)

    def dev(self, arr):
        import cupy as cp

        return cp.asarray(arr)

    def reset_markers(self):
        self.particles.markers[:] = self._markers0

    def random_f0_values(self):
        return self._rng.uniform(0.1, 2.0, size=self.n_markers).astype(np.float64)

    def random_noise(self):
        return self._rng.normal(size=(self.n_markers, 3)).astype(np.float64)


# ---------------------------------------------------------------------------
# Kernel cases: each is (name, cpu_call_factory, gpu_call_factory), where
# both factories take the Scene and return a zero-arg callable that runs one
# kernel invocation (including the host<->device marker round-trip for the
# GPU side, since that's part of the real per-step cost).
# ---------------------------------------------------------------------------


def _stage1_abc():
    """Single-stage (n_stages=1) RK Butcher arrays: makes the CPU kernels'
    internal ``dt*a[stage]``/``dt*b[stage]``/``last`` bookkeeping match the
    GPU wrappers' explicit ``dt_a=dt, dt_b=dt, last=1.0`` -- see
    push_eta_stage's body for the exact formula this mirrors."""
    return np.array([1.0]), np.array([1.0]), np.array([1.0])


def make_cases(scene: Scene, dt: float):
    import struphy.pic.accumulation.accum_kernels as accum_kernels
    import struphy.pic.pushing.pusher_kernels as pusher_kernels
    from struphy.pic.accumulation.accum_kernels_cuda import (
        charge_density_0form_gpu,
        linear_vlasov_ampere_gpu,
    )
    from struphy.pic.pushing.pusher_kernels_cuda import (
        push_bxu_H1vec_general_gpu,
        push_bxu_Hcurl_general_gpu,
        push_bxu_Hdiv_general_gpu,
        push_deterministic_diffusion_stage_general_gpu,
        push_eta_stage_general_gpu,
        push_pc_eta_stage_H1vec_general_gpu,
        push_pc_eta_stage_Hcurl_general_gpu,
        push_pc_eta_stage_Hdiv_general_gpu,
        push_pc_GXu_full_general_gpu,
        push_pc_GXu_general_gpu,
        push_random_diffusion_stage_gpu,
        push_v_with_efield_general_gpu,
        push_vxb_analytic_general_gpu,
        push_vxb_implicit_general_gpu,
        push_weights_with_efield_lin_va_general_gpu,
    )

    am, ad, ah = scene.args_markers, scene.args_domain, scene.args_derham
    a1, b1, c1 = _stage1_abc()
    n_cols = scene.particles.markers.shape[1]
    pn, tn1, tn2, tn3, starts = scene.pn, scene.tn1_dev, scene.tn2_dev, scene.tn3_dev, scene.starts
    kind_map, params_dev = scene.kind_map, scene.params_dev
    boundary_cut = 0.1
    cases = {}

    def add(name, cpu_fn, gpu_fn):
        cases[name] = (cpu_fn, gpu_fn)

    # --- push_eta_stage ---
    add(
        "push_eta_stage",
        lambda: pusher_kernels.push_eta_stage(dt, 0, am, ad, a1, b1, c1),
        lambda: push_eta_stage_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            am.first_free_idx,
            kind_map,
            params_dev,
            dt,
            dt,
            1.0,
        ),
    )

    # --- push_v_with_efield ---
    e1 = scene.fields["1"]
    e1_dev = tuple(scene.dev(a) for a in e1)
    add(
        "push_v_with_efield",
        lambda: pusher_kernels.push_v_with_efield(dt, 0, am, ad, ah, *e1, dt),
        lambda: push_v_with_efield_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *e1_dev,
            kind_map,
            params_dev,
            dt,
        ),
    )

    # --- push_vxb_analytic / push_vxb_implicit ---
    b2 = scene.fields["2"]
    b2_dev = tuple(scene.dev(a) for a in b2)
    add(
        "push_vxb_analytic",
        lambda: pusher_kernels.push_vxb_analytic(dt, 0, am, ad, ah, *b2),
        lambda: push_vxb_analytic_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *b2_dev,
            kind_map,
            params_dev,
            dt,
        ),
    )
    add(
        "push_vxb_implicit",
        lambda: pusher_kernels.push_vxb_implicit(dt, 0, am, ad, ah, *b2),
        lambda: push_vxb_implicit_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *b2_dev,
            kind_map,
            params_dev,
            dt,
        ),
    )

    # --- push_bxu_Hdiv / Hcurl / H1vec ---
    u2, u1, uv = scene.fields["2"], scene.fields["1"], scene.fields["v"]
    u2_dev = tuple(scene.dev(a) for a in u2)
    u1_dev = tuple(scene.dev(a) for a in u1)
    uv_dev = tuple(scene.dev(a) for a in uv)
    add(
        "push_bxu_Hdiv",
        lambda: pusher_kernels.push_bxu_Hdiv(dt, 0, am, ad, ah, *b2, *u2, boundary_cut),
        lambda: push_bxu_Hdiv_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *b2_dev,
            *u2_dev,
            kind_map,
            params_dev,
            boundary_cut,
            dt,
        ),
    )
    add(
        "push_bxu_Hcurl",
        lambda: pusher_kernels.push_bxu_Hcurl(dt, 0, am, ad, ah, *b2, *u1, boundary_cut),
        lambda: push_bxu_Hcurl_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *b2_dev,
            *u1_dev,
            kind_map,
            params_dev,
            boundary_cut,
            dt,
        ),
    )
    add(
        "push_bxu_H1vec",
        lambda: pusher_kernels.push_bxu_H1vec(dt, 0, am, ad, ah, *b2, *uv, boundary_cut),
        lambda: push_bxu_H1vec_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *b2_dev,
            *uv_dev,
            kind_map,
            params_dev,
            boundary_cut,
            dt,
        ),
    )

    # --- push_pc_GXu_full / push_pc_GXu (9 "G" tensor blocks, reusing the 3
    # Hcurl component shapes -- row i's 3 blocks all share component i's
    # shape, see pusher_kernels_cuda.py's push_pc_GXu_full_general docs) ---
    c1_arr, c2_arr, c3_arr = scene.fields["1"]
    rng = scene._rng
    g = {}
    for row, comp in ((1, c1_arr), (2, c2_arr), (3, c3_arr)):
        for col in (1, 2, 3):
            arr = rng.uniform(-1.0, 1.0, comp.shape)
            g[f"{row}{col}"] = arr
    g_order = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
    g_full = [g[k] for k in g_order]
    g_full_dev = [scene.dev(a) for a in g_full]
    add(
        "push_pc_GXu_full",
        lambda: pusher_kernels.push_pc_GXu_full(dt, 0, am, ad, ah, *g_full),
        lambda: push_pc_GXu_full_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *g_full_dev,
            kind_map,
            params_dev,
            dt,
        ),
    )
    add(
        "push_pc_GXu",
        lambda: pusher_kernels.push_pc_GXu(dt, 0, am, ad, ah, *g_full),
        lambda: push_pc_GXu_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *g_full_dev[:6],
            kind_map,
            params_dev,
            dt,
        ),
    )

    # --- push_pc_eta_stage_Hcurl / Hdiv / H1vec ---
    add(
        "push_pc_eta_stage_Hcurl",
        lambda: pusher_kernels.push_pc_eta_stage_Hcurl(dt, 0, am, ad, ah, *u1, False, a1, b1, c1),
        lambda: push_pc_eta_stage_Hcurl_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            am.first_free_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *u1_dev,
            False,
            kind_map,
            params_dev,
            dt,
            dt,
            1.0,
        ),
    )
    add(
        "push_pc_eta_stage_Hdiv",
        lambda: pusher_kernels.push_pc_eta_stage_Hdiv(dt, 0, am, ad, ah, *u2, False, a1, b1, c1),
        lambda: push_pc_eta_stage_Hdiv_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            am.first_free_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *u2_dev,
            False,
            kind_map,
            params_dev,
            dt,
            dt,
            1.0,
        ),
    )
    add(
        "push_pc_eta_stage_H1vec",
        lambda: pusher_kernels.push_pc_eta_stage_H1vec(dt, 0, am, ad, ah, *uv, False, a1, b1, c1),
        lambda: push_pc_eta_stage_H1vec_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            am.first_free_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *uv_dev,
            False,
            kind_map,
            params_dev,
            dt,
            dt,
            1.0,
        ),
    )

    # --- push_weights_with_efield_lin_va ---
    f0_values = scene.random_f0_values()
    f0_values_dev = scene.dev(f0_values)
    kappa, vth = 1.0, 1.0
    add(
        "push_weights_with_efield_lin_va",
        lambda: pusher_kernels.push_weights_with_efield_lin_va(dt, 0, am, ad, ah, *e1, f0_values, kappa, vth),
        lambda: push_weights_with_efield_lin_va_general_gpu(
            scene.particles.markers,
            n_cols,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *e1_dev,
            f0_values_dev,
            kappa,
            vth,
            kind_map,
            params_dev,
            dt,
        ),
    )

    # --- push_deterministic_diffusion_stage ---
    pi_u = scene.fields["0"][0]
    pi_grad = (scene.fields["0"][0], scene.fields["0"][0], scene.fields["0"][0])  # shape-only stand-ins
    pi_u_dev = scene.dev(pi_u)
    pi_grad_dev = tuple(scene.dev(a) for a in pi_grad)
    diffusion_coeff = 0.1
    add(
        "push_deterministic_diffusion_stage",
        lambda: pusher_kernels.push_deterministic_diffusion_stage(
            dt,
            0,
            am,
            ad,
            ah,
            pi_u,
            *pi_grad,
            diffusion_coeff,
            a1,
            b1,
            c1,
        ),
        lambda: push_deterministic_diffusion_stage_general_gpu(
            scene.particles.markers,
            n_cols,
            am.first_init_idx,
            am.first_free_idx,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            pi_u_dev,
            *pi_grad_dev,
            diffusion_coeff,
            kind_map,
            params_dev,
            dt,
            dt,
            1.0,
        ),
    )

    # --- push_random_diffusion_stage (domain-independent) ---
    noise = scene.random_noise()
    add(
        "push_random_diffusion_stage",
        lambda: pusher_kernels.push_random_diffusion_stage(dt, 0, am, ad, noise, diffusion_coeff, a1, b1, c1),
        lambda: push_random_diffusion_stage_gpu(scene.particles.markers, n_cols, noise, diffusion_coeff, dt),
    )

    # --- charge_density_0form (AccumulatorVector, H1) ---
    vec_shape = scene.fields["0"][0].shape
    vec_cpu = np.zeros(vec_shape, dtype=float)
    vec_gpu = scene.dev(np.zeros(vec_shape, dtype=float))
    weight_idx = scene.particles.index["weights"]
    add(
        "charge_density_0form",
        lambda: (vec_cpu.fill(0.0), accum_kernels.charge_density_0form(am, ah, ad, vec_cpu))[-1],
        lambda: (
            vec_gpu.fill(0.0),
            charge_density_0form_gpu(
                scene.particles.markers,
                weight_idx,
                pn,
                tn1,
                tn2,
                tn3,
                starts,
                vec_gpu,
            ),
        )[-1],
    )

    # --- linear_vlasov_ampere (Accumulator, symmetric V1 -> V1 matrix + vector) ---
    from feectools.linalg.block import BlockVector

    op = scene.mass_ops.create_weighted_mass("Hcurl", "Hcurl", weights="symm")
    mat_cpu, mat_gpu_ = {}, {}
    for a_ in range(3):
        for b_ in range(3):
            if b_ >= a_ and op.matrix.blocks[a_][b_] is not None:
                shape_ = op.matrix.blocks[a_][b_]._data.shape
                key = f"{a_ + 1}{b_ + 1}"
                mat_cpu[key] = np.zeros(shape_, dtype=float)
                mat_gpu_[key] = scene.dev(np.zeros(shape_, dtype=float))
    vec_space = scene.derham.coeff_spaces["1"]
    vec_bv = BlockVector(vec_space)
    vlva_vec_cpu = [np.zeros(bl._data.shape, dtype=float) for bl in vec_bv.blocks]
    vlva_vec_gpu = [scene.dev(v) for v in vlva_vec_cpu]
    lva_f0 = scene.random_f0_values()
    lva_f0_dev = scene.dev(lva_f0)
    mat_keys = ["11", "12", "13", "22", "23", "33"]

    def _lva_cpu():
        for v in mat_cpu.values():
            v.fill(0.0)
        for v in vlva_vec_cpu:
            v.fill(0.0)
        accum_kernels.linear_vlasov_ampere(
            am,
            ah,
            ad,
            *[mat_cpu[k] for k in mat_keys],
            *vlva_vec_cpu,
            lva_f0,
        )

    def _lva_gpu():
        for v in mat_gpu_.values():
            v.fill(0.0)
        for v in vlva_vec_gpu:
            v.fill(0.0)
        linear_vlasov_ampere_gpu(
            scene.particles.markers,
            kind_map,
            params_dev,
            lva_f0_dev,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
            *[mat_gpu_[k] for k in mat_keys],
            *vlva_vec_gpu,
        )

    add("linear_vlasov_ampere", _lva_cpu, _lva_gpu)

    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-markers", type=int, default=200_000, help="approximate number of markers (via ppc)")
    parser.add_argument("--num-elements", type=int, nargs=3, default=(16, 16, 8), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--degree", type=int, nargs=3, default=(3, 3, 3), metavar=("PX", "PY", "PZ"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--kernel",
        action="append",
        default=None,
        help="restrict to one kernel (repeatable); default: run all",
    )
    args = parser.parse_args()

    print(
        f"Building scene: num_elements={tuple(args.num_elements)}, degree={tuple(args.degree)}, Np~={args.n_markers} ..."
    )
    scene = Scene(tuple(args.num_elements), tuple(args.degree), args.n_markers)
    print(f"  -> {scene.n_markers} markers, domain kind_map={scene.kind_map} (Colella)")

    cases = make_cases(scene, args.dt)
    names = args.kernel if args.kernel else list(cases.keys())
    for name in names:
        if name not in cases:
            raise SystemExit(f"Unknown kernel {name!r}. Choices: {sorted(cases)}")

    rows = []
    for name in names:
        cpu_fn, gpu_fn = cases[name]

        def cpu_run(cpu_fn=cpu_fn):
            scene.reset_markers()
            cpu_fn()

        def gpu_run(gpu_fn=gpu_fn):
            scene.reset_markers()
            gpu_fn()

        cpu_t = timeit(cpu_run, args.repeats)
        gpu_t = timeit(gpu_run, args.repeats)
        rows.append((name, cpu_t, gpu_t, cpu_t / gpu_t))
        print(f"  {name}: cpu={cpu_t * 1e3:.3f} ms  gpu={gpu_t * 1e3:.3f} ms  speedup={cpu_t / gpu_t:.1f}x")

    print()
    print(f"{'kernel':<36} {'n_markers':>10} {'cpu (ms)':>12} {'gpu (ms)':>12} {'speedup':>10}")
    print("-" * 84)
    for name, cpu_t, gpu_t, speedup in rows:
        print(f"{name:<36} {scene.n_markers:>10} {cpu_t * 1e3:>12.3f} {gpu_t * 1e3:>12.3f} {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
