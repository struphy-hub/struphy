from dataclasses import dataclass
from typing import Callable, Literal, get_args
import cunumpy as xp
from line_profiler import profile
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

class TimeDependentSource(Propagator):
    r"""Propagates a source term :math:`S(t) \in V_h^n` of the form

    .. math::

        S(t) = \sum_{ijk} c_{ijk} \Lambda^n_{ijk} * h(\omega t)\,,

    where :math:`h(\omega t)` is one of the functions in Notes.

    Notes
    -----

    * :math:`h(\omega t) = \cos(\omega t)` (default)
    * :math:`h(\omega t) = \sin(\omega t)`
    """

    class Variables:
        def __init__(self):
            self._source: FEECVariable = None

        @property
        def source(self) -> FEECVariable:
            return self._source

        @source.setter
        def source(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._source = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsTimeSource = Literal["cos", "sin"]
        # propagator options
        omega: float = 2.0 * xp.pi
        hfun: OptsTimeSource = "cos"

        def __post_init__(self):
            # checks
            check_option(self.hfun, self.OptsTimeSource)

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    @profile
    def allocate(self, verbose: bool = False):
        if self.options.hfun == "cos":

            def hfun(t):
                return xp.cos(self.options.omega * t)
        elif self.options.hfun == "sin":

            def hfun(t):
                return xp.sin(self.options.omega * t)
        else:
            raise NotImplementedError(f"{self.options.hfun =} not implemented.")

        self._hfun = hfun
        self._c0 = self.variables.source.spline.vector.copy()

    @profile
    def __call__(self, dt):
        # new coeffs
        cn1 = self._c0 * self._hfun(self.time_state[0])

        # write new coeffs into self.feec_vars
        # max_dc = self.feec_vars_update(cn1)
        self.update_feec_variables(source=cn1)
