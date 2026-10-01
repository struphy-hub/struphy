from typing import get_args

import pytest

from struphy.io.options import LiteralOptions


def _stage_kernel_step(y0, dt, a_stage, b):
    """Emulate the stage loop of the ``*_stage`` particle pusher kernels for dy/dt = y."""
    y = y0
    acc = 0.0
    for stage in range(b.size):
        last = 1.0 if stage == b.size - 1 else 0.0
        k = y
        acc += dt * b[stage] * k
        y = y0 + dt * a_stage[stage] * k + last * acc
    return y


@pytest.mark.parametrize("algo", get_args(LiteralOptions.OptsButcher))
def test_a_stage(algo):
    """a_stage must either reproduce the full tableau in the pusher stage loop or raise."""

    import cunumpy as xp

    from struphy.ode.utils import ButcherTableau

    bt = ButcherTableau(algo)

    if xp.any(xp.tril(bt.a, k=-2) != 0.0):
        with pytest.raises(NotImplementedError):
            bt.a_stage
        return

    errs = []
    for dt in (0.1, 0.05):
        y = 1.0
        for _ in range(round(1.0 / dt)):
            y = _stage_kernel_step(y, dt, bt.a_stage, bt.b)
        errs.append(abs(y - xp.exp(1.0)))

    rate = xp.log2(errs[0] / errs[1])
    assert abs(rate - bt.conv_rate) < 0.1


if __name__ == "__main__":
    for algo in get_args(LiteralOptions.OptsButcher):
        test_a_stage(algo)
