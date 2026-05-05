from dataclasses import dataclass
from typing import Literal, get_args

import cunumpy as xp

from struphy.io.options import LiteralOptions


@dataclass
class ButcherTableau:
    r"""
    Butcher tableau for explicit s-stage Runge–Kutta methods.

    Encodes the coefficients of an explicit Run–Kutta method in the
    standard Butcher tableau form::

        c | a
        --+-----
          | b

    The tableau image is also included in the project documentation:

    .. image:: ../../pics/butcher_tableau.png
        :align: center
        :scale: 70%

    Parameters
    ----------
    algo : LiteralOptions.OptsButcher, optional
        Identifier of the RK method to use. Supported identifiers are
        defined by :class:`struphy.io.options.LiteralOptions.OptsButcher`.
        Defaults to ``"rk4"``.

    Attributes
    ----------
    a : cunumpy.ndarray, shape (s, s)
        Strictly lower-triangular matrix of stage coefficients (``a_ij``).
    b : cunumpy.ndarray, shape (s,)
        Weights used to combine stage derivatives into the final update.
    c : cunumpy.ndarray, shape (s,)
        Stage nodes (time fractions) corresponding to each row of ``a``.
    n_stages : int
        Number of stages ``s`` of the Run--Kutta method.
    conv_rate : int
        Formal order of convergence of the method.

    Notes
    -----
    - Arrays are stored using the project's array module ``cunumpy`` (imported
      as ``xp``) so they behave like numpy arrays but can be swapped for other
      array backends if configured.
    - Only explicit (strictly lower-triangular ``a``) Run--Kutta methods
      are supported. Passing an unsupported ``algo`` raises
      :class:`NotImplementedError`.

    Examples
    --------
    >>> bt = ButcherTableau("rk4")
    >>> bt.n_stages
    4
    >>> bt.b  # doctest: +SKIP
    array([1/6, 1/3, 1/3, 1/6])
    """

    algo: LiteralOptions.OptsButcher = "rk4"

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

            self._a[l + 1, : l + 1] = xp.array(st)

        self._conv_rate = conv_rate

    __available_methods__ = get_args(LiteralOptions.OptsButcher)

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
