"""Convergence of the polar projections under uniform mesh refinement (port of ``test/test_convergence.jl``).

With cubic splines in both directions, the 0-form projections converge with order 4 and the 1- and 2-form projections
with order 3 in the physical L2 norm, also on the perturbed disk where the spaces are not r-compatible with the
mapping.
"""

import numpy as np
import pytest

from struphy.polar.homotopy import PolarComplex2D
from struphy.polar.tests.reference_core import ClampedBSplineSpace, PeriodicBSplineSpace, perturbed_disk


def f(x, y):
    return np.exp(x) * np.sin(2 * y + 1)


def grad_f(x, y):
    return (np.exp(x) * np.sin(2 * y + 1), 2 * np.exp(x) * np.cos(2 * y + 1))


def A(x, y):
    return (np.cos(x - y), x * np.exp(y))


def B(x, y):
    return np.sin(x + 2 * y) + x**2


CELLS = (4, 8, 16)


def _complexes():
    return [
        PolarComplex2D(ClampedBSplineSpace(3, n), PeriodicBSplineSpace(3, 2 * n), perturbed_disk(), regularity=2)
        for n in CELLS
    ]


@pytest.mark.convergence
@pytest.mark.parametrize("method", ["commuting", "l2"])
def test_polar_projection_convergence(method):
    targets = (f, A, B)
    errors = [[], [], []]
    for polar in _complexes():
        for k in range(3):
            if method == "commuting":
                projection = polar[k](targets[k])
            else:
                projection = polar.l2_projection(k, targets[k])
            errors[k].append(polar.spaces[k].l2_error(projection, targets[k], additional_order=4))
    for k, rate in enumerate((4, 3, 3)):
        e = np.array(errors[k])
        assert np.all(e[1:] / e[:-1] < 1)
        assert np.log2(e[-2] / e[-1]) > rate - 0.5


@pytest.mark.convergence
def test_gradient_of_commuting_projection_converges():
    errors = []
    for polar in _complexes():
        gradient = polar.spaces[0].derivative_matrix @ polar[0](f)
        errors.append(polar.spaces[1].l2_error(gradient, grad_f, additional_order=4))
    assert np.log2(errors[-2] / errors[-1]) > 2.5
