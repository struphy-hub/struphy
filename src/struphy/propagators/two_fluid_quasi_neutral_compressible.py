import logging
from dataclasses import dataclass
from typing import Callable, get_args
from warnings import warn

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class TwoFluidQuasiNeutralCompressible(Propagator):
    r"""FEEC discretization of the compressible quasi-neutral two-fluid model."""

    class Variables:
        def __init__(self) -> None:
            self._u: FEECVariable | None = None
            self._ue: FEECVariable | None = None
            self._phi: FEECVariable | None = None
            self._n: FEECVariable | None = None

        @property
        def u(self): return self._u
        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable) and new.space == "Hdiv"
            self._u = new

        @property
        def ue(self): return self._ue
        @ue.setter
        def ue(self, new):
            assert isinstance(new, FEECVariable) and new.space == "Hdiv"
            self._ue = new

        @property
        def phi(self): return self._phi
        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable) and new.space == "L2"
            self._phi = new

        @property
        def n(self): return self._n
        @n.setter
        def n(self, new):
            assert isinstance(new, FEECVariable) and new.space == "L2"
            self._n = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        pass

    @property
    def options(self) -> Options:
        assert hasattr(self, "_options"), "Options not set."
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    def allocate(self):
        pass

    def __call__(self, dt):
        pass