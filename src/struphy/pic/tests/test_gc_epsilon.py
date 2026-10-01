from types import SimpleNamespace

import pytest

from struphy.pic.particles import Particles5D, Particles5Dvperp


@pytest.mark.parametrize("cls", [Particles5D, Particles5Dvperp])
def test_epsilon_from_equation_params(cls):
    """The epsilon property must return the value stored in equation_params."""
    particles = cls.__new__(cls)
    particles._equation_params = SimpleNamespace(epsilon=0.123)
    assert particles.epsilon == 0.123
