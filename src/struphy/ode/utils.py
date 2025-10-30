from dataclasses import dataclass
from typing import Literal, get_args

import cunumpy as xp

OptsButcher = Literal[
    "rk4",
    "forward_euler",
    "heun2",
    "rk2",
    "heun3",
    "3/8 rule",
]


@dataclass
class ButcherTableau:
    r"""
    Butcher tableau for explicit s-stage Runge-Kutta methods.

    The Butcher tableau has the form

    .. image:: ../../pics/butcher_tableau.png
        :align: center
        :scale: 70%

    Parameters
    ----------
    algo : OptsButcher
        Name of the RK method.
    """

    algo: OptsButcher = "rk4"

    def __post_init__(self):
        # choose algorithm
        if self.algo == "forward_euler":
            a = ()
            b = (1.0,)
            c = (0.0,)
            conv_rate = 1
        elif self.algo == "heun2":
            a = ((1.0,),)
            b = (1 / 2, 1 / 2)
            c = (0.0, 1.0)
            conv_rate = 2
        elif self.algo == "rk2":
            a = ((1 / 2,),)
            b = (0.0, 1.0)
            c = (0.0, 1 / 2)
            conv_rate = 2
        elif self.algo == "heun3":
            a = ((1 / 3,), (0.0, 2 / 3))
            b = (1 / 4, 0.0, 3 / 4)
            c = (0.0, 1 / 3, 2 / 3)
            conv_rate = 3
        elif self.algo == "rk4":
            a = ((1 / 2,), (0.0, 1 / 2), (0.0, 0.0, 1.0))
            b = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
            c = (0.0, 1 / 2, 1 / 2, 1.0)
            conv_rate = 4
        elif self.algo == "3/8 rule":
            a = ((1 / 3,), (-1 / 3, 1.0), (1.0, -1.0, 1.0))
            b = (1 / 8, 3 / 8, 3 / 8, 1 / 8)
            c = (0.0, 1 / 3, 2 / 3, 1.0)
            conv_rate = 4
        else:
            raise NotImplementedError(f"Chosen algorithm {self.algo} is not implemented.")

        self._b = xp.array(b)
        self._c = xp.array(c)
        assert self._b.size == self._c.size

        self._n_stages = self._b.size
        assert len(a) == self.n_stages - 1

        self._a = xp.tri(self.n_stages, k=-1)
        for l, st in enumerate(a):
            assert len(st) == l + 1
            self._a[l + 1, : l + 1] = st

        self._conv_rate = conv_rate

    __available_methods__ = get_args(OptsButcher)

    @property
    def a(self):
        """Characteristic coefficients of the method (see tableau in class docstring)."""
        return self._a

    @property
    def b(self):
        """Characteristic coefficients of the method (see tableau in class docstring)."""
        return self._b

    @property
    def c(self):
        """Characteristic coefficients of the method (see tableau in class docstring)."""
        return self._c

    @property
    def n_stages(self):
        """Number of stages of the s-stage Runge-Kutta method."""
        return self._n_stages

    @property
    def conv_rate(self):
        """Convergence rate of the s-stage Runge-Kutta method."""
        return self._conv_rate
