"""Benchmark of the Vlasov-Ampère and Vlasov-Maxwell models on standard test problems.

Runs a fixed number of time steps of :class:`~struphy.models.VlasovAmpereOneSpecies` or
:class:`~struphy.models.VlasovMaxwellOneSpecies` on the NumPy (CPU) or CuPy (GPU) backend and reports the setup
time, the time per step, the time per propagator and per kernel, and the cost per marker.

All timing comes from scope-profiler: the run activates the profiling session of :meth:`Simulation.run`, so the
regions that struphy already defines (``setup: allocate``, ``model.integrate``, ``prop: <name>``,
``kernel: <name>``, ...) are recorded, and the benchmark only reads them back. On the CuPy backend the session
records CUDA-event device time for every region in addition to the CPU time (``use_gpu_timing``, backend
``cupy``), and by default synchronizes the device when a region ends (``gpu_sync_on_exit``), so the CPU time of a
region includes its kernels and not only their launch; ``--no-gpu-sync`` records the enqueue-side CPU time instead.
The profile is written to an HDF5 file (``--profile``) for the full scope-profiler tool chain, and its HTML report
(call tree, charts, per-rank statistics) next to it (``--report``, ``--no-report``)::

    scope-profiler inspect vlasov_benchmark_VlasovAmpereOneSpecies_landau_cupy.h5
    open vlasov_benchmark_VlasovAmpereOneSpecies_landau_cupy.html

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

The backend is cunumpy's active backend (``CUNUMPY_BACKEND=numpy|cupy``, or ``--backend``). Times are the maximum
over MPI ranks. Sorting boxes are off by default, since ``SortingBoxes`` does not work on the CuPy backend yet
(struphy-hub/struphy#726); ``--sorting-boxes N`` turns them on.

Without a GPU, ``CUNUMPY_FAKE_CUPY=1 ... --backend cupy --size smoke`` runs the CuPy code path on cunumpy's fake
CuPy with every CUDA launch emulated on the CPU: a check that the models run on the device backend, not a timing.
The fake CuPy has no CUDA events, so GPU timing is off in that mode.

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
from dataclasses import asdict, dataclass, field

import cunumpy as xp
import numpy as np
from cunumpy.kernel_testing import fake_cupy_active
from maybempi import MPI
from scope_profiler import ProfilingResults, read_h5

from struphy import ProfilingOptions
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

# scope-profiler regions of Simulation.run that the report summarizes
SESSION_REGION = "scope_profiler.session"
SETUP_REGION = "setup: allocate"
COMPILE_REGION = "setup: compile cuda kernels"
STEP_REGION = "model.integrate"
PROPAGATOR_PREFIX = "prop: "
KERNEL_PREFIX = "kernel: "


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
    profile: str  # the scope-profiler HDF5 file of the run
    report: str | None  # the scope-profiler HTML report of the run, if written
    gpu_timing: bool  # CUDA-event device times recorded per region
    gpu_sync_on_exit: bool  # device synchronized at every region end (CPU times include the kernels)
    setup: float = 0.0  # Simulation.allocate (FEEC operators, markers, initial Poisson solve)
    compile_cuda: float = 0.0  # Simulation.compile_cuda_kernels (0 on NumPy)
    step_times: list = field(default_factory=list)  # model.integrate, one per step
    total_run: float = 0.0  # the scope-profiler session, i.e. Simulation.run
    propagators: dict = field(default_factory=dict)  # class name -> {"cpu": total, "gpu": total or None}
    kernels: dict = field(default_factory=dict)  # kernel name -> {"cpu": total, "gpu": total or None}
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
    parser.add_argument(
        "--no-gpu-sync",
        action="store_true",
        help="do not synchronize the device at region ends (CPU times are then launch times; GPU times unaffected)",
    )
    parser.add_argument(
        "--profile",
        help="scope-profiler HDF5 file of the run (default: vlasov_benchmark_<model>_<case>_<backend>.h5)",
    )
    parser.add_argument(
        "--report", help="scope-profiler HTML report of the run (default: the profile's name with .html)"
    )
    parser.add_argument("--no-report", action="store_true", help="do not write the HTML report")
    parser.add_argument("--out", help="output folder (default: a temporary folder, removed afterwards)")
    parser.add_argument("--json", help="also write the result to this JSON file (rank 0)")
    args = parser.parse_args()
    if args.model not in CASES[args.case]["models"]:
        parser.error(f"case {args.case!r} is only set up for {', '.join(CASES[args.case]['models'])}")
    return args


def profiling_options(args, profile_path: str) -> ProfilingOptions:
    """The scope-profiler settings of the run: GPU timing through CuPy's CUDA events on the CuPy backend."""
    # the fake CuPy (no GPU) has CUDA events without elapsed time, so device timing is only possible on a GPU
    gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()
    return ProfilingOptions(
        file_path=profile_path,
        label=f"{args.model} {args.case} {xp.get_backend()}",
        use_gpu_timing=gpu_timing,
        gpu_timing_backend="cupy",
        gpu_sync_on_exit=gpu_timing and not args.no_gpu_sync,
    )


def make_simulation(args, out_folders: str, profile_path: str):
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

    opts = profiling_options(args, profile_path)
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domains.Cuboid(r1=case["r1"]),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=tuple(args.degree)),
        profiling_opts=opts,
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
        profile=os.path.abspath(profile_path),
        report=None,
        gpu_timing=opts.use_gpu_timing,
        gpu_sync_on_exit=opts.gpu_sync_on_exit,
    )
    return sim, result


def max_over_ranks_per_call(region) -> list:
    """The duration of each call of a region, the maximum over the ranks (every rank makes the same calls)."""
    per_rank = [region[rank].durations for rank in region.ranks]
    n_calls = min(len(durations) for durations in per_rank)
    return np.max([durations[:n_calls] for durations in per_rank], axis=0).tolist()


def totals_over_ranks(region) -> dict:
    """The total CPU time of a region and, with GPU timing, its total device time; the maximum over the ranks."""
    cpu = max(region.total_durations().values())
    gpu = None
    if region.has_gpu_timing:
        gpu = max(region[rank].gpu_total_duration for rank in region.ranks if region[rank].has_gpu_timing)
    return {"cpu": float(cpu), "gpu": None if gpu is None else float(gpu)}


def collect(results: ProfilingResults, result: Result) -> None:
    """Fill ``result`` from the scope-profiler regions of the run."""
    if SETUP_REGION in results:
        result.setup = results[SETUP_REGION].max_duration
    if COMPILE_REGION in results:
        result.compile_cuda = results[COMPILE_REGION].max_duration
    if SESSION_REGION in results:
        result.total_run = results[SESSION_REGION].max_duration
    elif results.total_time is not None:
        result.total_run = results.total_time
    result.step_times = max_over_ranks_per_call(results[STEP_REGION])
    for region in results:
        if region.name.startswith(PROPAGATOR_PREFIX):
            result.propagators[region.name[len(PROPAGATOR_PREFIX) :]] = totals_over_ranks(region)
        elif region.name.startswith(KERNEL_PREFIX):
            result.kernels[region.name[len(KERNEL_PREFIX) :]] = totals_over_ranks(region)
    result.propagators = dict(sorted(result.propagators.items(), key=lambda item: -item[1]["cpu"]))
    result.kernels = dict(sorted(result.kernels.items(), key=lambda item: -item[1]["cpu"]))


def run(args) -> Result:
    comm = MPI.COMM_WORLD
    out = args.out
    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="struphy_vlasov_benchmark_") if comm.Get_rank() == 0 else None
        out = comm.bcast(tmp.name if tmp else None, root=0)
    profile_path = args.profile or f"vlasov_benchmark_{args.model}_{args.case}_{xp.get_backend()}.h5"

    sim, result = make_simulation(args, out, profile_path)
    # on cunumpy's fake CuPy (no GPU) the CUDA launches are emulated on the CPU: a smoke test, not a timing
    with device_context() if xp.get_backend() == "cupy" else contextlib.nullcontext():
        sim.run(profiling_activated=True)

    for name, scalar in sim.model.scalars.dct.items():
        value = float(xp.to_numpy(scalar.value)[0]) if hasattr(scalar, "value") else float("nan")
        result.scalars_end[name] = value

    # scope-profiler merges the ranks into one file at the end of the session
    if comm.Get_rank() == 0:
        results = read_h5(profile_path)
        collect(results, result)
        if not args.no_report:
            from scope_profiler import create_html_report

            report_path = args.report or os.path.splitext(profile_path)[0] + ".html"
            result.report = os.path.abspath(create_html_report(results, report_path))

    comm.Barrier()
    if tmp is not None:
        tmp.cleanup()
    return result


def _time_table(title: str, totals: dict, gpu_timing: bool) -> list:
    """Lines of a table with the CPU and (if recorded) GPU total of every entry and its share of the CPU sum."""
    if not totals:
        return []
    lines = [f"  {title}, maximum over ranks):"]
    width = max(len(name) for name in totals)
    cpu_sum = sum(entry["cpu"] for entry in totals.values()) or 1.0
    header = f"    {'':<{width}} {'CPU [s]':>10} {'share':>7}"
    if gpu_timing:
        header += f" {'GPU [s]':>10}"
    lines.append(header)
    for name, entry in totals.items():
        line = f"    {name:<{width}} {entry['cpu']:>10.3f} {100 * entry['cpu'] / cpu_sum:>6.1f}%"
        if gpu_timing:
            line += f" {entry['gpu']:>10.3f}" if entry["gpu"] is not None else f" {'-':>10}"
        lines.append(line)
    return lines


def report(result: Result) -> str:
    warning = ["!! cunumpy's fake CuPy: CUDA launches emulated on the CPU, the times below are not GPU timings", ""]
    lines = (warning if result.backend == "cupy" and fake_cupy_active() else []) + [
        f"{result.model} | {result.case} | backend {result.backend} | {result.mpi_ranks} MPI rank(s) | host {result.host}",
        f"grid {result.num_elements}, degree {result.degree}, {result.ppc} markers per cell = {result.n_markers:,} markers, "
        f"{result.steps} steps of dt = {result.dt}",
    ]
    if result.gpu_timing:
        sync = "device synchronized at region ends" if result.gpu_sync_on_exit else "no synchronization at region ends"
        lines.append(f"GPU timing: CUDA events through CuPy, {sync}")
    lines += [
        "",
        f"  setup (allocate + initial Poisson)  {result.setup:10.3f} s",
        f"  CUDA kernel compilation             {result.compile_cuda:10.3f} s",
        f"  time loop: model.integrate          {sum(result.step_times):10.3f} s",
        f"  Simulation.run total                {result.total_run:10.3f} s",
        "",
        f"  step: median {1e3 * result.step_median:.3f} ms (first step {1e3 * result.step_times[0]:.3f} ms), "
        f"{result.ns_per_marker_step:.1f} ns per marker and step",
    ]
    lines += _time_table("per propagator (all steps", result.propagators, result.gpu_timing)
    lines += _time_table("per kernel (whole run, including the setup", result.kernels, result.gpu_timing)
    lines += ["", f"  profile: {result.profile}  (scope-profiler inspect {os.path.basename(result.profile)})"]
    if result.report:
        lines.append(f"  report:  {result.report}")
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
