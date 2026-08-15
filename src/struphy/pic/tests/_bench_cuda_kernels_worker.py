"""
Worker script for :mod:`bench_cuda_kernels`.

Times one of the CUDA-RawKernel-ported operations (see
:mod:`struphy.pic.pushing.pusher_kernels_cuda` and
:mod:`struphy.pic.sph_eval_kernels_cuda`) in isolation, on a single marker
set, and prints the median wall time per call (in seconds) as the last line
on stdout. Runs in a fresh subprocess per (backend, op, Np) combination
because ``ARRAY_BACKEND`` is read once at import time (by ``cunumpy``) and
cannot be changed within a running process.

Usage::

    ARRAY_BACKEND=<numpy|cupy> python _bench_cuda_kernels_worker.py <op> <Np> <n_reps>

``op`` is one of: push_eta, push_v, eval_density_flat, eval_density_mesh, sort_boxes
"""

import statistics
import sys
import time

N_WARMUP = 1


def _timed_calls(call, n_reps: int) -> list[float]:
    """Time ``n_reps`` calls to ``call()``, synchronizing the default CUDA
    stream after each one under the CuPy backend.

    This matters specifically for :func:`_bench_eval_density`: unlike the
    pusher kernels (which end in a synchronizing ``.get(out=markers)``),
    ``box_based_evaluation_flat_gpu``/``_meshgrid_gpu`` write their result via
    ``out[:] = dev_out`` whenever the caller's ``out`` is already a CuPy
    array (as it is here, from ``xp.zeros_like`` on CuPy eval points) -- a
    device-to-device copy that's asynchronous, so an unsynchronized
    ``time.perf_counter()`` around the call would measure only kernel-launch
    overhead, not completion.
    """
    import cunumpy

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        call()
        if cunumpy.cupy_backend:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _bench_pushers(op: str, Np: int, n_reps: int) -> list[float]:
    """push_eta_stage / push_v_with_efield on a Cuboid, all-periodic marker
    set -- the configuration :func:`~struphy.pic.pushing.pusher.Pusher`'s
    device-resident fast paths require, matching ``params_PressureLessSPH.py``.
    """
    from cunumpy import PyccelKernel

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import ParticlesSPH
    from struphy.pic.pushing import pusher_kernels
    from struphy.pic.pushing.pusher import Pusher
    from struphy.topology.grids import TensorProductGrid

    domain = domains.Cuboid()
    dt = 0.01

    loading_params = LoadingParameters(Np=Np, seed=1234)
    particles = ParticlesSPH(loading_params=loading_params, domain=domain)
    particles.draw_markers(sort=False)
    particles.initialize_weights()

    if op == "push_eta":
        butcher = ButcherTableau()
        pusher = Pusher(
            particles,
            PyccelKernel(pusher_kernels.push_eta_stage),
            (butcher.a_stage, butcher.b, butcher.c),
            domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            mpi_sort="each",
        )
    elif op == "push_v":
        grid = TensorProductGrid(num_elements=(16, 16, 8))
        derham_opts = DerhamOptions()
        derham = Derham(grid, derham_opts, comm=None)
        _, e_field = create_equal_random_arrays(derham.V1fem, seed=2345, flattened=True)
        pusher = Pusher(
            particles,
            PyccelKernel(pusher_kernels.push_v_with_efield),
            (derham.args_derham, e_field[0]._data, e_field[1]._data, e_field[2]._data, 1.0),
            domain.args_domain,
            alpha_in_kernel=1.0,
        )
    else:
        raise ValueError(f"unknown op {op!r}")

    for _ in range(N_WARMUP):
        pusher(dt)

    return _timed_calls(lambda: pusher(dt), n_reps)


def _bench_eval_density(op: str, Np: int, n_reps: int) -> list[float]:
    """box_based_evaluation_flat / _meshgrid, the SPH kernel-density-estimation
    sum used by :meth:`~struphy.pic.base.Particles.eval_density`."""
    import cunumpy as xp

    from struphy import BoundaryParameters, LoadingParameters, SortingParameters, domains, perturbations
    from struphy.fields_background.equils import ConstantVelocity
    from struphy.pic.particles import ParticlesSPH

    domain = domains.Cuboid()

    # A fixed, known-safe box grid (rather than scaling boxes_per_dim with
    # Np): the periodic ghost/self-communication bookkeeping in
    # put_particles_in_boxes() needs a generous bufsize/box_bufsize margin
    # that's easiest to just fix once here (a separate, pre-existing
    # bufsize-tuning concern, unrelated to the kernels being benchmarked).
    # Pseudo-random loading (the default), not tesselation, so Np is honored
    # directly instead of being derived from ppb.
    boxes_per_dim = (8, 8, 4)
    n_boxes_per_dim = boxes_per_dim[0]

    loading_params = LoadingParameters(Np=Np, seed=1234)
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain
    pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(0.3,))}
    boundary_params = BoundaryParameters(bc_sph=("periodic", "periodic", "periodic"))
    sorting_params = SortingParameters(boxes_per_dim=boxes_per_dim, box_bufsize=10.0)

    particles = ParticlesSPH(
        loading_params=loading_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        bufsize=5.0,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
    )
    particles.draw_markers(sort=False)
    particles.initialize_weights()

    # flat: n_eval points total. mesh: n_eval**3 points (a full 3-D grid) --
    # kept an order of magnitude smaller so the *naive* NumPy/Pyccel
    # meshgrid path (no vectorization across points) stays tractable up to
    # Np=10**6; this doesn't change what's being measured, just how many
    # evaluation points are timed per call.
    n_eval = 40 if op == "eval_density_flat" else 12
    eta1 = xp.linspace(0.02, 0.98, n_eval)
    eta2 = xp.linspace(0.02, 0.98, n_eval)
    eta3 = xp.linspace(0.02, 0.98, n_eval)
    h1 = h2 = h3 = 1.0 / n_boxes_per_dim

    if op == "eval_density_flat":
        e1, e2, e3 = eta1, eta2, eta3
    elif op == "eval_density_mesh":
        e1, e2, e3 = xp.meshgrid(eta1, eta2, eta3, indexing="ij")
    else:
        raise ValueError(f"unknown op {op!r}")

    def call():
        particles.eval_density(e1, e2, e3, h1, h2, h3, kernel_type="gaussian_3d")

    for _ in range(N_WARMUP):
        call()

    return _timed_calls(call, n_reps)


def _bench_sort_boxes(Np: int, n_reps: int) -> list[float]:
    """assign_box_to_each_particle + assign_particles_to_boxes (see
    :mod:`~struphy.pic.sorting_kernels_cuda`) via
    :meth:`~struphy.pic.base.Particles.put_particles_in_boxes` -- the
    per-step box-sorting bookkeeping that both the SPH pushers
    (``Pusher._box_comm``) and every ``eval_density``/``eval_velocity`` call
    (via ``_eval_sph``) run before touching the box-based marker structure.
    Same particle setup as :func:`_bench_eval_density`, minus the evaluation
    points, since only the box bookkeeping is being timed here.

    ``sorting_boxes._communicate`` (true by default for SPH particles) is
    forced off: it makes ``put_particles_in_boxes`` additionally run
    ``_communicate_boxes()``, whose ghost-particle-destination bookkeeping
    (``_get_destinations_box``, MPI-send-buffer prep) is pure host/NumPy
    Python control flow -- unrelated to, and in single-process runs far more
    expensive than, the two CUDA-ported kernels this benchmark targets. With
    an actual MPI communicator that bookkeeping is unavoidable and would
    dominate real per-step cost regardless of backend; it is out of scope for
    this GPU-kernel benchmark specifically.

    Unlike :func:`_bench_eval_density`, ``boxes_per_dim`` is scaled with
    ``Np`` here (targeting ~30 particles/box) instead of using a fixed
    ``(8, 8, 4)`` grid: box-based SPH only makes sense with a modest,
    Np-independent number of particles per box (that's the point of the
    27-neighbour search), and a fixed grid at Np=10**6 would put ~4000
    particles in every box, inflating the ``boxes`` array (and therefore its
    host<->device transfer, which -- unlike the in-place CPU kernel -- this
    GPU port must do) by two orders of magnitude for no physical reason."""
    from struphy import BoundaryParameters, LoadingParameters, SortingParameters, domains, perturbations
    from struphy.fields_background.equils import ConstantVelocity
    from struphy.pic.particles import ParticlesSPH

    domain = domains.Cuboid()
    n_per_dim = max(2, round((Np / 30.0) ** (1.0 / 3.0)))
    boxes_per_dim = (n_per_dim, n_per_dim, n_per_dim)

    loading_params = LoadingParameters(Np=Np, seed=1234)
    background = ConstantVelocity(n=1.5, density_profile="constant")
    background.domain = domain
    pert = {"n": perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(0.3,))}
    boundary_params = BoundaryParameters(bc_sph=("periodic", "periodic", "periodic"))
    sorting_params = SortingParameters(boxes_per_dim=boxes_per_dim, box_bufsize=3.0)

    particles = ParticlesSPH(
        loading_params=loading_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        bufsize=5.0,
        domain=domain,
        background=background,
        perturbations=pert,
        n_as_volume_form=True,
    )
    particles.draw_markers(sort=False)
    particles.initialize_weights()
    particles.sorting_boxes._communicate = False

    def call():
        particles.put_particles_in_boxes()

    for _ in range(N_WARMUP):
        call()

    return _timed_calls(call, n_reps)


def main(op: str, Np: int, n_reps: int) -> float:
    if op in ("push_eta", "push_v"):
        times = _bench_pushers(op, Np, n_reps)
    elif op in ("eval_density_flat", "eval_density_mesh"):
        times = _bench_eval_density(op, Np, n_reps)
    elif op == "sort_boxes":
        times = _bench_sort_boxes(Np, n_reps)
    else:
        raise ValueError(f"unknown op {op!r}")

    return statistics.median(times)


if __name__ == "__main__":
    op = sys.argv[1]
    Np = int(sys.argv[2])
    n_reps = int(sys.argv[3])

    median_time = main(op, Np, n_reps)
    print(median_time)
