"""Parity-test arguments of ``push_vxb_analytic``: random 2-form coefficients, every boundary condition."""

from struphy.pic.tests.kernel_test_args import (
    BOUNDARY_CONDITIONS,
    derham_arguments,
    marker_arguments,
    spline_coefficients,
)

CASES = BOUNDARY_CONDITIONS
RTOL = 1e-13
ATOL = 1e-14


def make_args(backend, seed):
    return (0.2, 0, *marker_arguments(CASES[seed]), derham_arguments(), *spline_coefficients())
