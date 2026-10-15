"""
Standalone benchmark (not a pytest test) of the operations on the marker buffer that depend on its memory layout
(see :mod:`struphy.pic.marker_layout` and struphy-hub/struphy#707; one column per marker, C-ordered): the row
assignments at the start of every push, two pushers, one accumulation kernel, hole detection, boundary conditions, the MPI sort, binning and the
gathers of valid markers used by the diagnostics.

Runs on the active cunumpy backend (``CUNUMPY_BACKEND=numpy`` or ``cupy``); on the GPU every timing ends with a
device synchronization. Prints the median wall time of each operation in milliseconds, and writes them to a JSON
file with ``--json``, so that two runs (e.g. before and after a layout change) can be compared with ``--compare``.

Usage::

    mpirun -n 1 python src/struphy/pic/tests/bench_marker_layout.py [--Np 1000000] [--repeats 5] [--json out.json]
    python src/struphy/pic/tests/bench_marker_layout.py --compare before.json after.json
"""

import argparse
import json
import statistics
import time


def _time(fun, repeats: int) -> float:
    """Median wall time of ``fun()`` in milliseconds, after one warm-up call."""
    import cunumpy as xp

    fun()
    xp.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fun()
        xp.synchronize()
        times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def run(
    Np: int, repeats: int, num_elements: int, degree: int, save_markers: str = None, all_rows: bool = False
) -> dict:
    import cunumpy as xp
    import numpy as np
    from maybempi import MPI

    from struphy import BoundaryParameters, LoadingParameters, WeightsParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
    from struphy.pic.pushing.kernels.push_vxb_analytic import push_vxb_analytic
    from struphy.pic.pushing.pusher import Pusher, _prepare_push_cuda
    from struphy.topology.grids import TensorProductGrid

    if all_rows:
        # every column counts as used, as before n_rows_used existed (for comparing the same build)
        from struphy.pic.base import Particles

        def _all_rows(self, n_emptied=0, n_filled=0):
            self._set_n_rows_used(self.n_rows)

        Particles._track_rows = _all_rows
        Particles.count_rows_used = _all_rows

    comm = MPI.COMM_WORLD
    domain = domains.Cuboid(r1=2.0, r2=3.0, r3=4.0)
    grid = TensorProductGrid(num_elements=[num_elements] * 3)
    derham = Derham(grid, DerhamOptions(degree=[degree] * 3), comm=comm)

    particles = Particles6D(
        comm_world=comm,
        domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
        loading_params=LoadingParameters(Np=Np, seed=1234, spatial="uniform"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(bc=("periodic", "periodic", "periodic")),
        domain=domain,
    )
    particles.draw_markers()
    particles.mpi_sort_markers()
    particles.initialize_weights()

    markers = particles.markers
    first_pusher_idx = particles.first_pusher_idx
    first_shift_idx = particles.first_shift_idx
    residual_idx = particles.residual_idx
    n_init = 3 + particles.vdim

    def preamble():
        """The row assignments at the start of Pusher._push."""
        if xp.get_backend() == "cupy":
            n_cols, n_rows = markers.shape
            _prepare_push_cuda()(
                markers,
                first_pusher_idx,
                n_init,
                n_threads=n_rows * (n_cols - 2 - first_pusher_idx),
            )
        else:
            markers[first_pusher_idx:first_shift_idx] = markers[:n_init]
            markers[first_shift_idx:residual_idx] = 0.0
            markers[residual_idx:-2] = 0.0

    butcher = ButcherTableau("rk4")
    push_eta = Pusher(
        particles,
        push_eta_stage,
        (xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages),
        domain.args_domain,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        pushes_eta=True,
        local_eval_only=True,
    )

    _, b2 = create_equal_random_arrays(derham.V2fem, seed=3456, flattened=True)
    push_vxb = Pusher(
        particles,
        push_vxb_analytic,
        (derham.args_derham, b2[0]._data, b2[1]._data, b2[2]._data),
        domain.args_domain,
        alpha_in_kernel=1.0,
        pushes_eta=False,
    )

    rho = derham.coeff_spaces["H1"].zeros()._data

    def accumulate():
        rho[:] = 0.0
        charge_density_0form(particles.args_markers, derham.args_derham, domain.args_domain, rho)

    bin_edges = (xp.linspace(0.0, 1.0, 65), xp.linspace(-5.0, 5.0, 65))
    components = (True, False, False, True, False, False)

    ops = {
        "preamble (row assignments)": preamble,
        "push_eta_stage (RK4 pusher)": lambda: push_eta(1e-3),
        "push_vxb_analytic (pusher)": lambda: push_vxb(1e-3),
        "charge_density_0form (accumulation)": accumulate,
        "update_holes": particles.update_holes,
        "apply_kinetic_bc (periodic)": particles.apply_kinetic_bc,
        "mpi_sort_markers": particles.mpi_sort_markers,
        "binning e1_v1": lambda: particles.binning(components, bin_edges),
        "gather valid markers (markers_wo_holes_and_ghost)": lambda: particles.markers_wo_holes_and_ghost,
        "gather weights of valid markers": lambda: particles.weights,
    }

    info = {
        "backend": xp.get_backend(),
        "n_rows": particles.n_rows,
        "n_cols": particles.n_cols,
        "Np": Np,
        "c_contiguous": bool(markers.flags.c_contiguous),
        "n_rows_used": int(getattr(particles, "n_rows_used", particles.n_rows)),
    }
    result = {"info": info, "ms": {name: _time(fun, repeats) for name, fun in ops.items()}}

    if save_markers is not None:
        # the valid markers after all timed calls, sorted by ID, for a bit-for-bit comparison of two runs; saved with
        # one row per marker, as by runs with the earlier (n_rows, n_cols) buffer
        valid = xp.to_numpy(particles.markers_wo_holes_and_ghost)
        np.save(save_markers, valid[:, np.argsort(valid[-1])].T)
    return result


def _print(result: dict):
    info = result["info"]
    order = "C-ordered" if info["c_contiguous"] else "not C-contiguous"
    print(
        f"backend {info['backend']}, markers {info['n_cols']} x {info['n_rows']} (one column per marker, {order}), "
        f"Np = {info['Np']}, columns used {info.get('n_rows_used', info['n_rows'])}"
    )
    width = max(len(name) for name in result["ms"])
    for name, ms in result["ms"].items():
        print(f"  {name:<{width}}  {ms:10.2f} ms")


def _compare(before: dict, after: dict):
    width = max(len(name) for name in before["ms"])
    print(f"  {'operation':<{width}}  {'before':>10}  {'after':>10}  {'speedup':>8}")
    for name, ms_before in before["ms"].items():
        ms_after = after["ms"].get(name)
        if ms_after is None:
            continue
        print(f"  {name:<{width}}  {ms_before:8.2f}ms  {ms_after:8.2f}ms  {ms_before / ms_after:7.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--Np", type=int, default=1_000_000, help="number of markers")
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per operation (median is reported)")
    parser.add_argument("--num-elements", type=int, default=8, help="cells per direction")
    parser.add_argument("--degree", type=int, default=3, help="spline degree per direction")
    parser.add_argument("--json", help="write the timings to this file")
    parser.add_argument("--save-markers", help="save the valid markers after the run (sorted by ID) to this .npy file")
    parser.add_argument(
        "--all-rows", action="store_true", help="treat every marker column as used (no n_rows_used bound)"
    )
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="compare two JSON files and exit")
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0]) as f, open(args.compare[1]) as g:
            _compare(json.load(f), json.load(g))
    else:
        result = run(args.Np, args.repeats, args.num_elements, args.degree, args.save_markers, args.all_rows)
        _print(result)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)
