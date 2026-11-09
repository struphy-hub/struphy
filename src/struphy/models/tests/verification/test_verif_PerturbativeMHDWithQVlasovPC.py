"""Verification of PerturbativeMHDWithQVlasovPC (nonlinear MHD in q = sqrt(p) with pressure-coupled energetic ions):
reduction to PerturbativeMHDWithQ without energetic ions, and second-order time convergence of the coupled dynamics at
finite amplitude (Strang splitting)."""

import logging

from struphy import set_logging_level
from struphy.models import PerturbativeMHDWithQ, PerturbativeMHDWithQVlasovPC
from struphy.models.tests.verification.perturbative_hybrid_utils import (
    add_energetic_ions,
    global_max,
    max_diff,
    perturb_mhd,
    run,
    set_tight_options,
)

set_logging_level()
logger = logging.getLogger("struphy")

MODEL = PerturbativeMHDWithQVlasovPC


def test_reduces_to_perturbative_mhd(
    mpi_tmp_path: str,
):
    """With energetic ions of zero density (zero marker weights) the coupling steps do nothing, and the model gives
    the MHD variables of PerturbativeMHDWithQ."""
    parent = PerturbativeMHDWithQ()
    set_tight_options(parent)
    perturb_mhd(parent)
    x_parent = run(mpi_tmp_path, parent, f"{MODEL.__name__}_parent", dt=0.05, Tend=0.2)

    model = MODEL()
    set_tight_options(model)
    perturb_mhd(model)
    add_energetic_ions(model, n_h=0.0)
    x = run(mpi_tmp_path, model, f"{MODEL.__name__}_no_ions", dt=0.05, Tend=0.2)

    diff = max_diff(x, x_parent) / global_max(float(abs(x_parent).max()))
    logger.info(f"relative difference to PerturbativeMHDWithQ without energetic ions: {diff:.2e}")
    assert diff < 1e-9


def test_time_convergence(
    mpi_tmp_path: str,
):
    """Second order in time (Strang, weights of the nonlinear terms at t^{n+1/2}) with energetic ions of 10% density
    and a resolved cyclotron motion: the differences of the MHD variables at T between dt, dt/2 and dt/4 decrease
    by about 4 (smaller time steps than for the current coupling, whose errors reach the asymptotic regime earlier)."""
    xs = []
    for dt in (0.02, 0.01, 0.005):
        model = MODEL(hot_epsilon=0.2)
        set_tight_options(model)
        perturb_mhd(model)
        add_energetic_ions(model)
        xs.append(run(mpi_tmp_path, model, f"{MODEL.__name__}_conv", dt=dt, Tend=0.2))
    errs = [max_diff(xs[0], xs[1]), max_diff(xs[1], xs[2])]
    logger.info(f"differences between successive time steps {errs}, ratio {errs[0] / errs[1]:.2f}")
    assert errs[0] / errs[1] > 3.0
