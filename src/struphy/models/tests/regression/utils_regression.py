"""Regression data of the model tests: collect, compare against and update the stored baselines.

After the test run with the default parameter file (before the restart in ``call_test``) the following quantities
are collected:

- ``time``: the time grid of the run,
- ``scalar/<name>``: the time series of every scalar quantity of the model,
- ``feec/<species>/<variable>``: the l2 norm of the global FE coefficient vector at the final time,
- ``markers/<species>/<variable>/{n,sum_w,sum_w_v2}``: number of markers, sum of weights and sum of
  ``w * |v|^2`` over all valid markers at the final time.

The baselines are stored as JSON in ``baselines/<case>.json``, with one entry per number of MPI ranks, since the
seeded random marker loading and the noise perturbations depend on the domain decomposition.

When a change of the numbers is intended (bug fix, new default), update the baselines with::

    pytest -m models --update-baselines src/struphy/models/tests/regression/
    mpirun -n 2 python -m pytest -p no:xdist --with-mpi -m models --update-baselines src/struphy/models/tests/regression/

and explain in the PR why the numbers changed.
"""

import datetime
import json
import logging
import subprocess
from pathlib import Path

import h5py
import numpy as np

from struphy.models.variables import FEECVariable, PICVariable, SPHVariable

logger = logging.getLogger("struphy")

BASELINE_DIR = Path(__file__).parent / "baselines"

# seed for the random marker loading and the noise perturbations of the regression runs
SEED = 1234

RTOL = 1e-6
ATOL = 1e-12

# cases whose default run is known to produce non-finite values (inf/nan); remove a case once it is fixed
_DT_TOO_LARGE = "default dt=0.01 too large: VariationalMomentumAdvection does not converge, blow-up in the first step"
KNOWN_NONFINITE = {
    "VariationalBarotropicFluid": _DT_TOO_LARGE,
    "VariationalPressurelessFluid": _DT_TOO_LARGE,
}

UPDATE_COMMANDS = (
    "pytest -m models --update-baselines src/struphy/models/tests/regression/\n"
    "mpirun -n 2 python -m pytest -p no:xdist --with-mpi -m models --update-baselines "
    "src/struphy/models/tests/regression/"
)


class RegressionMismatch(AssertionError):
    """The run differs from the stored baseline."""


def set_seeds(model, seed: int = SEED):
    """Seed the random marker loading and the noise perturbations whose seed is not set, so that runs are reproducible
    for a given number of MPI ranks."""
    for species in model.species.values():
        loading_params = getattr(species, "loading_params", None)
        if loading_params is not None and loading_params.seed is None:
            loading_params.seed = seed
        for variable in species.variables.values():
            perturbations = variable.perturbations
            if perturbations is None:
                continue
            if not isinstance(perturbations, list):
                perturbations = [perturbations]
            for perturbation in perturbations:
                if getattr(perturbation, "seed", 0) is None:
                    perturbation.seed = seed


def _allreduce(comm, value: float) -> float:
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value)
    return value


def collect(sim) -> dict:
    """Collect the regression data of a finished run. Collective: call it on all ranks."""
    data = {}

    # time series of the scalars; every rank's data file holds the global values
    with h5py.File(sim.data.file_path, "r") as file:
        data["time"] = np.asarray(file["time/value"]).ravel().tolist()
        if "scalar" in file:
            for key in sorted(file["scalar"]):
                data[f"scalar/{key}"] = np.asarray(file["scalar"][key]).ravel().tolist()

    model = sim.model

    # norms of the FE coefficients at the final time
    feec_species = model.field_species | model.fluid_species | model.diagnostic_species
    for species_name, species in feec_species.items():
        for var_name, variable in species.variables.items():
            if isinstance(variable, FEECVariable):
                vec = variable.spline.vector
                data[f"feec/{species_name}/{var_name}"] = float(np.sqrt(abs(vec.inner(vec))))

    # marker moments at the final time
    for species_name, species in model.particle_species.items():
        for var_name, variable in species.variables.items():
            if isinstance(variable, (PICVariable, SPHVariable)):
                particles = variable.particles
                comm = particles.mpi_comm
                w = np.asarray(particles.weights)
                v = np.asarray(particles.velocities)
                key = f"markers/{species_name}/{var_name}"
                data[f"{key}/n"] = int(_allreduce(comm, int(w.size)))
                data[f"{key}/sum_w"] = float(_allreduce(comm, float(np.sum(w))))
                data[f"{key}/sum_w_v2"] = float(_allreduce(comm, float(np.sum(w * np.sum(v**2, axis=0)))))

    return data


def check_finite(case: str, data: dict) -> list[str]:
    """Non-finite values fail the test, unless the case is in :data:`KNOWN_NONFINITE`.

    Returns
    -------
    list[str]
        One line per problem; empty if the data is acceptable.
    """
    nonfinite = [key for key, val in data.items() if not np.all(np.isfinite(np.asarray(val, dtype=float)))]
    if case in KNOWN_NONFINITE:
        if not nonfinite:
            return [f"all values are finite now: remove '{case}' from KNOWN_NONFINITE in {Path(__file__).name}"]
        logger.warning(f"Known non-finite values in '{case}' ({KNOWN_NONFINITE[case]}): {nonfinite}")
        return []
    return [f"{key}: non-finite values {data[key]}" for key in nonfinite]


def baseline_path(case: str) -> Path:
    return BASELINE_DIR / f"{case}.json"


def _load(case: str) -> dict:
    path = baseline_path(case)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def update(case: str, n_ranks: int, data: dict):
    """Store ``data`` as the baseline of ``case`` for ``n_ranks`` MPI ranks. Call it on one rank only."""
    baseline = _load(case)
    baseline.setdefault("case", case)
    baseline.setdefault("ranks", {})
    baseline["ranks"][str(n_ranks)] = {
        "commit": _git_commit(),
        "date": datetime.date.today().isoformat(),
        "values": data,
    }
    baseline["ranks"] = dict(sorted(baseline["ranks"].items()))
    BASELINE_DIR.mkdir(exist_ok=True)
    with open(baseline_path(case), "w") as f:
        json.dump(baseline, f, indent=1)
        f.write("\n")
    logger.warning(f"Updated regression baseline {baseline_path(case)} for {n_ranks} rank(s).")


def compare(case: str, n_ranks: int, data: dict, rtol: float = RTOL, atol: float = ATOL) -> list[str]:
    """Compare ``data`` against the baseline of ``case`` for ``n_ranks`` MPI ranks.

    Returns
    -------
    list[str]
        One line per difference; empty if the run matches the baseline.
    """
    entry = _load(case).get("ranks", {}).get(str(n_ranks))
    if entry is None:
        return [f"no baseline for {n_ranks} rank(s) in {baseline_path(case)}"]

    reference = entry["values"]
    lines = []
    for key in sorted(set(reference) | set(data)):
        if key not in data:
            lines.append(f"{key}: in the baseline, but not in the run")
        elif key not in reference:
            lines.append(f"{key}: in the run, but not in the baseline")
        else:
            ref = np.asarray(reference[key], dtype=float)
            new = np.asarray(data[key], dtype=float)
            if ref.shape != new.shape:
                lines.append(f"{key}: shape {new.shape} != baseline shape {ref.shape}")
            elif not np.allclose(new, ref, rtol=rtol, atol=atol, equal_nan=True):
                err = np.max(np.abs(new - ref) / np.maximum(np.abs(ref), atol))
                lines.append(f"{key}: max rel. difference {err:.3e}\n    run:      {new}\n    baseline: {ref}")
    return lines


def mismatch_message(case: str, n_ranks: int, lines: list[str]) -> str:
    return (
        f"Regression test '{case}' on {n_ranks} rank(s) differs from its baseline:\n"
        + "\n".join(f"  {line}" for line in lines)
        + "\nIf the change is intended (e.g. a bug fix), update the baselines with\n"
        + UPDATE_COMMANDS
        + "\nand explain in the PR why the numbers changed."
    )
