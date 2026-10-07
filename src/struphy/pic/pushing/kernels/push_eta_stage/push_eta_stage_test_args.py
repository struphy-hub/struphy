"""Parity-test arguments of ``push_eta_stage``: every boundary condition, Euler and RK4, every stage."""

from struphy.ode.utils import ButcherTableau
from struphy.pic.tests.kernel_test_args import BOUNDARY_CONDITIONS, butcher_arguments, marker_arguments

CASES = [
    (bc, method, stage)
    for bc in BOUNDARY_CONDITIONS
    for method in ("forward_euler", "rk4")
    for stage in range(ButcherTableau(method).n_stages)
]
RTOL = 1e-13
ATOL = 1e-14


def make_args(backend, seed):
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage

    bc, method, stage = CASES[seed]
    args = (*marker_arguments(bc), *butcher_arguments(method))
    # reach this stage independently on each backend
    for previous in range(stage):
        push_eta_stage(0.2, previous, *args)
    return (0.2, stage, *args)
