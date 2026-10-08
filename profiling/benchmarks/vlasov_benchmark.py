"""Benchmark of the Vlasov-Ampère and Vlasov-Maxwell models on standard test problems.

Runs a fixed number of time steps of :class:`~struphy.models.VlasovAmpereOneSpecies` or
:class:`~struphy.models.VlasovMaxwellOneSpecies` on the NumPy (CPU) or CuPy (GPU) backend and reports the setup
time, the time per step, the time per propagator, the cost per marker and the host/device transfers in the time loop.

Test problems (the physics follows the verification tests in ``src/struphy/models/tests/verification``):

``landau``
    Weak Landau damping (both models). Periodic box of length 4 pi (k = 0.5), Maxwellian with a density
    perturbation of 1e-3 cos(k x), control-variate weights, cubic splines in x. Damping rate -0.1533.
``weibel``
    Weibel instability (Vlasov-Maxwell only), the setup of Kraus et al., J. Plasma Phys. 83 (2017), Sec. 6.2:
    box of length 2 pi / 1.25, anisotropic Maxwellian (vth1 = vth3 = 0.02 / sqrt(2), vth2 = sqrt(12) vth1),
    seed field B3 = -1e-4 cos(k x), no density perturbation. Growth rate 0.02784.

All runs use ``alpha = 1`` and ``epsilon = -1`` (electrons, time in units of the inverse plasma frequency,
velocities in units of c), as the verification tests do.

Sizes (``--size``; each value can be overridden on the command line):

========  =================  =====  =====  ==================================================
size      cells (x, y, z)    ppc    steps  purpose
========  =================  =====  =====  ==================================================
smoke     8, 1, 1            50     3      check that the benchmark runs (seconds)
standard  32, 1, 1           1000   20     the verification tests' resolution (32k markers)
large     128, 1, 1          20000  20     GPU-sized (2.56M markers)
========  =================  =====  =====  ==================================================

Examples::

    python profiling/benchmarks/vlasov_benchmark.py --model VlasovAmpereOneSpecies --case landau --size standard
    CUNUMPY_BACKEND=cupy python profiling/benchmarks/vlasov_benchmark.py --model VlasovMaxwellOneSpecies \\
        --case weibel --size large --json weibel_gpu.json
    mpirun -n 2 python profiling/benchmarks/vlasov_benchmark.py --size standard

The backend is cunumpy's active backend (``CUNUMPY_BACKEND=numpy|cupy``, or ``--backend``). Every timed region
ends with ``xp.synchronize()``, so GPU times include the kernels and not only their launch. Per-propagator timing
adds one synchronization per propagator and step; ``--no-propagator-timing`` measures the step without them.
Times are the maximum over MPI ranks. Sorting boxes are off by default, since ``SortingBoxes`` does not work on
the CuPy backend yet (struphy-hub/struphy#726); ``--sorting-boxes N`` turns them on.

Without a GPU, ``CUNUMPY_FAKE_CUPY=1 ... --backend cupy --size smoke`` runs the CuPy code path on cunumpy's fake
CuPy with every CUDA launch emulated on the CPU: a check that the models run on the device backend, not a timing.

CPU reference (NumPy backend, ``--size standard``, 1 rank, Apple M-series laptop, 8 October 2026): VlasovAmpere
Landau 67 ms per step, VlasovMaxwell Landau 82 ms, VlasovMaxwell Weibel 86 ms (2.1 to 2.7 microseconds per marker
and step; ``VlasovAmpereCoupling`` takes 68 to 86 % of a step).
"""

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field

import cunumpy as xp
from cunumpy.kernel_testing import fake_cupy_active
from maybempi import MPI

from struphy.models.tests.gpu_e2e import device_context

SIZES = {
    "smoke": {"num_elements": (8, 1, 1), "ppc": 50, "steps": 3},
    "standard": {"num_elements": (32, 1, 1), "ppc": 1000, "steps": 20},
    "large": {"num_elements": (128, 1, 1), "ppc": 20000, "steps": 20},
}

CASES = {
    # box length, time step, expected rate (damping < 0, growth > 0) of the electric (landau) or magnetic (weibel)
    # field energy amplitude
    "landau": {
        "r1": 4 * math.pi,
        "dt": 0.05,
        "rate": -0.1533,
        "models": ("VlasovAmpereOneSpecies", "VlasovMaxwellOneSpecies"),
    },
    "weibel": {"r1": 2 * math.pi / 1.25, "dt": 0.05, "rate": 0.02784, "models": ("VlasovMaxwellOneSpecies",)},
}


@dataclass
class Result:
    """What one benchmark run measured. Times in seconds, maximum over MPI ranks."""

    model: str
    case: str
    backend: str
    mpi_ranks: int
    num_elements: tuple
    degree: tuple
    ppc: int
    n_markers: int
    steps: int
    dt: float
    setup: float = 0.0  # Simulation.allocate (FEEC operators, markers, initial Poisson solve)
    compile_cuda: float = 0.0  # Simulation.compile_cuda_kernels (0 on NumPy)
    step_times: list = field(default_factory=list)  # model.integrate, one per step
    diagnostics: float = 0.0  # scalars, saved markers, binned distribution functions (all steps)
    output: float = 0.0  # DataContainer.save_data (all steps)
    total_run: float = 0.0  # Simulation.run
    propagators: dict = field(default_factory=dict)  # class name -> total time over all steps
    transfers_in_loop: dict = field(default_factory=dict)  # cunumpy transfer counts during the time loop
    scalars_end: dict = field(default_factory=dict)  # name -> value after the last step
    host: str = field(default_factory=platform.node)

    @property
    def step_median(self) -> float:
        # the first step includes one-time work (CUDA module loads, allocations of work arrays)
        steady = self.step_times[1:] or self.step_times
        return statistics.median(steady)

    @property
    def ns_per_marker_step(self) -> float:
        return 1e9 * self.step_median / max(self.n_markers, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="VlasovAmpereOneSpecies", choices=CASES["landau"]["models"])
    parser.add_argument("--case", default="landau", choices=tuple(CASES))
    parser.add_argument("--size", default="standard", choices=tuple(SIZES))
    parser.add_argument(
        "--num-elements", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="override the size's grid"
    )
    parser.add_argument("--degree", type=int, nargs=3, default=(3, 1, 1), metavar=("PX", "PY", "PZ"))
    parser.add_argument("--ppc", type=int, help="markers per cell (overrides the size)")
    parser.add_argument("--steps", type=int, help="number of time steps (overrides the size)")
    parser.add_argument("--dt", type=float, help="time step (overrides the case)")
    parser.add_argument("--backend", choices=("numpy", "cupy"), help="cunumpy backend (default: the active one)")
    parser.add_argument("--sorting-boxes", type=int, default=0, metavar="N", help="sorting boxes in x (default: off)")
    parser.add_argument("--save-step", type=int, default=1, help="compute diagnostics and save every N steps")
    parser.add_argument("--no-propagator-timing", action="store_true", help="time only whole steps")
    parser.add_argument("--out", help="output folder (default: a temporary folder, removed afterwards)")
    parser.add_argument("--json", help="also write the result to this JSON file (rank 0)")
    args = parser.parse_args()
    if args.model not in CASES[args.case]["models"]:
        parser.error(f"case {args.case!r} is only set up for {', '.join(CASES[args.case]['models'])}")
    return args


def make_simulation(args, out_folders: str):
    """The model, markers, initial condition and options of the chosen case and size."""
    from struphy import (
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        SavingParameters,
        Simulation,
        SortingParameters,
        Time,
        WeightsParameters,
        domains,
        grids,
        maxwellians,
        perturbations,
    )
    from struphy.models import VlasovAmpereOneSpecies, VlasovMaxwellOneSpecies

    case = CASES[args.case]
    size = SIZES[args.size]
    num_elements = tuple(args.num_elements or size["num_elements"])
    ppc = args.ppc or size["ppc"]
    steps = args.steps or size["steps"]
    dt = args.dt or case["dt"]

    if args.model == "VlasovAmpereOneSpecies":
        model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    else:
        model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0)

    env = EnvironmentOptions(
        out_folders=out_folders,
        sim_folder=f"{args.model}_{args.case}",
        save_step=args.save_step,
        save_restart=False,  # restart data would dominate the run time for many markers
    )
    time_opts = Time(dt=dt, Tend=round(steps * dt, 14))

    if args.sorting_boxes:
        sorting_params = SortingParameters(boxes_per_dim=(args.sorting_boxes, 1, 1), do_sort=True)
    else:
        sorting_params = SortingParameters()

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=ppc, seed=1234),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=sorting_params,
        saving_params=SavingParameters(n_markers=0),  # benchmark the physics, not the marker output
        bufsize=0.4,
    )

    for name, prop in model.propagators.__dict__.items():
        prop.options = prop.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

    background = maxwellians.Maxwellian3D(n=(1.0, None))
    if args.case == "landau":
        model.kinetic_ions.var.add_background(background)
        perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-3,))
        model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))
    else:
        vth1 = 0.02 / math.sqrt(2.0)
        anisotropic = maxwellians.Maxwellian3D(
            n=(1.0, None), vth1=(vth1, None), vth2=(math.sqrt(12.0) * vth1, None), vth3=(vth1, None)
        )
        model.kinetic_ions.var.add_background(anisotropic)
        # B3 = -1e-4 cos(k x) in physical coordinates: kx = 2 pi l / Lx, so Lx must be the box length
        seed = perturbations.ModesCos(ls=(1,), amps=(-1e-4,), Lx=case["r1"], given_in_basis="physical", comp=2)
        model.em_fields.b_field.add_perturbation(seed)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domains.Cuboid(r1=case["r1"]),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=tuple(args.degree)),
    )
    result = Result(
        model=args.model,
        case=args.case,
        backend=xp.get_backend(),
        mpi_ranks=MPI.COMM_WORLD.Get_size(),
        num_elements=num_elements,
        degree=tuple(args.degree),
        ppc=ppc,
        n_markers=ppc * math.prod(num_elements),
        steps=steps,
        dt=dt,
    )
    return sim, result


class TimedPropagator:
    """Stands in for a propagator in ``model.prop_list`` and adds its synchronized run time to ``totals``."""

    def __init__(self, propagator, totals: dict):
        self._propagator = propagator
        self._totals = totals
        self._name = type(propagator).__name__

    def __call__(self, dt):
        xp.synchronize()
        t0 = time.perf_counter()
        self._propagator(dt)
        xp.synchronize()
        self._totals[self._name] = self._totals.get(self._name, 0.0) + time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self._propagator, name)


def instrument(sim, result: Result, timed_propagators: bool):
    """Wrap the phases of ``sim.run()`` with synchronized timers and count the transfers of the time loop."""
    from cunumpy.profiling import count_transfers

    from struphy.io.output_handling import DataContainer

    model = sim.model
    in_loop = [False]
    # the transfer counter of the time loop: count_transfers() is entered when the loop starts
    loop = {"context": count_transfers(), "counter": None}

    def timed(fn, record):
        def wrapper(*args, **kwargs):
            xp.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            xp.synchronize()
            record(time.perf_counter() - t0)
            return out

        return wrapper

    def add(name):
        def record(t):
            if in_loop[0]:
                setattr(result, name, getattr(result, name) + t)

        return record

    def after_allocate(t):
        result.setup = t
        if timed_propagators:
            model._prop_list = [TimedPropagator(p, result.propagators) for p in model.prop_list]

    sim.allocate = timed(sim.allocate, after_allocate)
    sim.compile_cuda_kernels = timed(sim.compile_cuda_kernels, lambda t: setattr(result, "compile_cuda", t))

    original_init_datasets = sim._initialize_hdf5_datasets

    def init_datasets(*args, **kwargs):
        out = original_init_datasets(*args, **kwargs)
        in_loop[0] = True  # the time loop starts after this, the last step of the setup
        loop["counter"] = loop["context"].__enter__()
        return out

    sim._initialize_hdf5_datasets = init_datasets
    model.integrate = timed(model.integrate, lambda t: result.step_times.append(t) if in_loop[0] else None)
    for name in ("update_scalar_quantities", "update_markers_to_be_saved", "update_distr_functions"):
        setattr(model, name, timed(getattr(model, name), add("diagnostics")))

    original_save_data = DataContainer.save_data
    record_output = add("output")

    def save_data(self, *args, **kwargs):
        xp.synchronize()
        t0 = time.perf_counter()
        out = original_save_data(self, *args, **kwargs)
        record_output(time.perf_counter() - t0)
        return out

    DataContainer.save_data = save_data
    return loop, lambda: setattr(DataContainer, "save_data", original_save_data)


def reduce_max(value: float) -> float:
    comm = MPI.COMM_WORLD
    return comm.allreduce(value, op=MPI.MAX) if comm.Get_size() > 1 else value


def run(args) -> Result:
    comm = MPI.COMM_WORLD
    out = args.out
    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="struphy_vlasov_benchmark_") if comm.Get_rank() == 0 else None
        out = comm.bcast(tmp.name if tmp else None, root=0)

    sim, result = make_simulation(args, out)
    loop, restore = instrument(sim, result, timed_propagators=not args.no_propagator_timing)
    try:
        # on cunumpy's fake CuPy (no GPU) the CUDA launches are emulated on the CPU: a smoke test, not a timing
        with device_context() if xp.get_backend() == "cupy" else contextlib.nullcontext():
            xp.synchronize()
            t0 = time.perf_counter()
            sim.run()
            xp.synchronize()
            result.total_run = time.perf_counter() - t0
    finally:
        if loop["counter"] is not None:
            loop["context"].__exit__(None, None, None)
        restore()

    counter = loop["counter"]
    kinds = ("to_host", "to_device", "kernel_conversion", "fallback", "device_copy", "sync")
    result.transfers_in_loop = {kind: counter.count(kind) for kind in kinds}
    result.transfers_in_loop["bytes_to_host"] = counter.bytes_to_host
    for name, scalar in sim.model.scalars.dct.items():
        value = float(xp.to_numpy(scalar.value)[0]) if hasattr(scalar, "value") else float("nan")
        result.scalars_end[name] = value

    for name in ("setup", "compile_cuda", "diagnostics", "output", "total_run"):
        setattr(result, name, reduce_max(getattr(result, name)))
    result.step_times = [reduce_max(t) for t in result.step_times]
    result.propagators = {k: reduce_max(v) for k, v in sorted(result.propagators.items())}

    comm.Barrier()
    if tmp is not None:
        tmp.cleanup()
    return result


def report(result: Result) -> str:
    warning = ["!! cunumpy's fake CuPy: CUDA launches emulated on the CPU, the times below are not GPU timings", ""]
    lines = (warning if result.backend == "cupy" and fake_cupy_active() else []) + [
        f"{result.model} | {result.case} | backend {result.backend} | {result.mpi_ranks} MPI rank(s) | host {result.host}",
        f"grid {result.num_elements}, degree {result.degree}, {result.ppc} markers per cell = {result.n_markers:,} markers, "
        f"{result.steps} steps of dt = {result.dt}",
        "",
    ]
    lines += [
        f"  setup (allocate + initial Poisson)  {result.setup:10.3f} s",
        f"  CUDA kernel compilation             {result.compile_cuda:10.3f} s",
        f"  time loop: model.integrate          {sum(result.step_times):10.3f} s",
        f"             diagnostics              {result.diagnostics:10.3f} s",
        f"             output                   {result.output:10.3f} s",
        f"  Simulation.run total                {result.total_run:10.3f} s",
        "",
        f"  step: median {1e3 * result.step_median:.3f} ms (first step {1e3 * result.step_times[0]:.3f} ms), "
        f"{result.ns_per_marker_step:.1f} ns per marker and step",
    ]
    if result.propagators:
        lines.append("  per propagator (all steps):")
        total = sum(result.propagators.values()) or 1.0
        for name, t in result.propagators.items():
            lines.append(f"    {name:<28} {t:10.3f} s  ({100 * t / total:5.1f} %)")
    lines.append(
        "  host/device transfers in the time loop (through cunumpy): "
        + ", ".join(f"{k}={v}" for k, v in result.transfers_in_loop.items())
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.backend:
        xp.set_backend(args.backend)
    result = run(args)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(report(result))
        if args.json:
            data = asdict(result)
            data.update(step_median=result.step_median, ns_per_marker_step=result.ns_per_marker_step)
            with open(args.json, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nwrote {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
