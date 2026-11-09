"""Verification of PerturbativeMHDWithQDriftkineticCC (nonlinear MHD in q = sqrt(p) with current-coupled guiding-center
energetic ions): time convergence of the coupled dynamics at finite amplitude (Strang splitting)."""

import logging

from struphy import set_logging_level
from struphy.models import PerturbativeMHDWithQDriftkineticCC
from struphy.models.tests.verification.perturbative_hybrid_utils import (
    add_energetic_ions,
    max_diff,
    perturb_mhd,
    run,
    set_tight_options,
)

set_logging_level()
logger = logging.getLogger("struphy")

MODEL = PerturbativeMHDWithQDriftkineticCC


def test_time_convergence():
    """First order in time with guiding centers of 10% density (Strang splitting; the total-field operator of the
    shear-Alfven step ``shear_alf`` is evaluated at t^n within the step, so the nonlinear shear-Alfven terms are first
    order): the differences of the MHD variables at T between dt, dt/2 and dt/4 decrease by at least about 2.
    (Measured: ratio 4.2 between dt = 0.04, 0.02, 0.01, which is pre-asymptotic, and 3.0 between dt = 0.02, 0.01,
    0.005, tending to 2.)"""
    xs = []
    for dt in (0.02, 0.01, 0.005):
        model = MODEL()
        set_tight_options(model)
        perturb_mhd(model)
        add_energetic_ions(model)
        xs.append(run(model, f"{MODEL.__name__}_conv", dt=dt, Tend=0.2))
    errs = [max_diff(xs[0], xs[1]), max_diff(xs[1], xs[2])]
    logger.info(f"differences between successive time steps {errs}, ratio {errs[0] / errs[1]:.2f}")
    assert errs[0] / errs[1] > 1.7
