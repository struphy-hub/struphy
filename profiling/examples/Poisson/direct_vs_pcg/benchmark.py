"""Run params_poisson_solver.py with both solvers and report the DirectSolver vs. PCG
speedup on total and steady-state (post-factorization) per-solve wall time.

Deliberately uses the same small, single-rank-sized case as params_poisson_solver.py's
own defaults (see its module docstring for why) -- this is not a scaling study, just a
quick correctness + speed check, meant to run in well under a minute on a laptop.

Usage: python benchmark.py [--num-elements NX NY NZ] [--repeats N]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
PARAMS = HERE / "params_poisson_solver.py"


def child_python() -> str:
    if sys.prefix != sys.base_prefix:
        return sys.executable

    repo_venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if repo_venv_python.exists():
        return str(repo_venv_python)

    return sys.executable

LINE_RE = re.compile(
    r"^\s*solve: PoissonSolve\s+\d+\s+(?P<calls>\d+)\s+(?P<total>[\d.eE+-]+)\s+"
    r"(?P<avg>[\d.eE+-]+)\s+(?P<min>[\d.eE+-]+)\s+(?P<max>[\d.eE+-]+)",
    re.MULTILINE,
)
ERR_RE = re.compile(r"max relative error in Phi after \d+ steps: ([\d.eE+-]+)")


def run(solver: str, num_elements: list[int], repeats: int) -> dict:
    sim_folder = HERE / "sim_00"  # params_poisson_solver.py names it f"sim_{id:02d}"
    shutil.rmtree(sim_folder, ignore_errors=True)
    cmd = [
        child_python(),
        str(PARAMS),
        "--solver", solver,
        "--num-elements", *map(str, num_elements),
        "--repeats", str(repeats),
        "--id", "0",
    ]
    env = os.environ.copy()
    pythonpath = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "feectools"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
    shutil.rmtree(sim_folder, ignore_errors=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out)
        raise RuntimeError(f"solver={solver} run failed (see output above)")

    m = LINE_RE.search(out)
    assert m is not None, f"could not find 'solve: PoissonSolve' timing line in output:\n{out}"
    e = ERR_RE.search(out)
    assert e is not None, f"could not find relative error line in output:\n{out}"

    calls = int(m["calls"])
    total = float(m["total"])
    first = float(m["max"])  # the one-time factorization/first-solve call is always the outlier max
    steady_state = (total - first) / (calls - 1) if calls > 1 else float("nan")

    return {"calls": calls, "total": total, "first_call": first, "steady_state": steady_state,
            "rel_err": float(e[1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-elements", type=int, nargs=3, default=[8, 8, 8])
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    results = {solver: run(solver, args.num_elements, args.repeats) for solver in ("pcg", "direct")}

    print(f"\nnum_elements={args.num_elements}, repeats={args.repeats}\n")
    print(f"{'solver':<8} {'total [s]':>12} {'first call [s]':>16} {'steady-state [s]':>18} {'rel err':>10}")
    for solver, r in results.items():
        print(f"{solver:<8} {r['total']:>12.5f} {r['first_call']:>16.5f} {r['steady_state']:>18.6f} {r['rel_err']:>10.2e}")

    pcg, direct = results["pcg"], results["direct"]
    print(f"\nDirectSolver speedup, total wall time:    {pcg['total'] / direct['total']:.2f}x")
    print(f"DirectSolver speedup, steady-state solve: {pcg['steady_state'] / direct['steady_state']:.2f}x")

    breakeven = direct["first_call"] / (pcg["steady_state"] - direct["steady_state"])
    print(f"Break-even repeat count (total wall time): ~{breakeven:.0f}")
