"Propagator base class."

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector
from scope_profiler import ProfileManager

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.projected_equils import ProjectedFluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.io.options import OptionsBase
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class Propagator(metaclass=ABCMeta):
    """Base class for propagators used in StruphyModels.

    Note
    ----
    All Struphy propagators are subclasses of ``Propagator`` and must be added under ``struphy/propagators/``.
    """

    @abstractmethod
    class Variables:
        """Define variable names and types to be updated by the propagator."""

        def __init__(self):
            self._var1 = None

        @property
        def var1(self):
            return self._var1

        @var1.setter
        def var1(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles6D"
            self._var1 = new

    @abstractmethod
    def __init__(self):
        self.variables = self.Variables()

    @abstractmethod
    @dataclass(repr=False)
    class Options(OptionsBase):
        """Template for configuration options of a propagator.

        Subclasses should override this to define specific propagator options.
        """

        # specific literals
        OptsTemplate = Literal["implicit", "explicit"]
        # propagator options
        opt1: str = ("implicit",)

        def __post_init__(self):
            # checks
            check_option(self.opt1, self.OptsTemplate)

    @property
    @abstractmethod
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    @abstractmethod
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @abstractmethod
    def allocate(self):
        """Allocate all data/objects of the instance."""

    @abstractmethod
    def __call__(self, dt: float):
        """Update variables from t -> t + dt.
        Use ``Propagators.feec_vars_update`` to write to FEEC variables to ``Propagator.feec_vars``.

        Parameters
        ----------
        dt : float
            Time step size.
        """

    @property
    def _solve_region(self) -> str:
        """Name of the profiling region for the linear solve(s) of this propagator."""
        if not hasattr(self, "_solve_region_name"):
            self._solve_region_name = "solve: " + self.__class__.__name__
        return self._solve_region_name

    def show_options(self):
        """Print the options of the propagator."""
        logger.info(f"\nOptions for propagator '{self.__class__.__name__}':")
        for k, v in self.options.__dict__.items():
            logger.info(f"    {k + ':':<20}{v}")

    @ProfileManager.profile("update_feec_variables")
    def update_feec_variables(self, **new_coeffs):
        r"""Return max_diff = max(abs(new - old)) for each new_coeffs,
        update feec coefficients and update ghost regions.

        Returns
        -------
        diffs : dict
            max_diff for all feec variables.
        """
        diffs = {}
        for var, new in new_coeffs.items():
            assert "_" + var in self.variables.__dict__, f"{var} not in {self.variables.__dict__}."
            assert isinstance(new, (StencilVector, BlockVector))
            old_var = getattr(self.variables, var)
            assert isinstance(old_var, FEECVariable)
            old = old_var.spline.vector
            assert new.space == old.space

            # calculate maximum of difference abs(new - old)
            diffs[var] = xp.max(xp.abs(new.toarray() - old.toarray()))

            # copy new coeffs into old
            new.copy(out=old)

            # important: sync processes!
            old.update_ghost_regions()

        return diffs

    @property
    def init_kernels(self) -> tuple[KernelSetup, ...]:
        r"""Tuple of initialization kernel setups for evaluation at
        :math:`\boldsymbol \eta^n`
        in an iterative :class:`~struphy.pic.pushing.pusher.Pusher`.
        """
        return getattr(self, "_init_kernels", ())

    @property
    def eval_kernels(self) -> tuple[KernelSetup, ...]:
        r"""Tuple of evaluation kernel setups for evaluation at
        :math:`\alpha_i \eta_{i}^{n+1,k} + (1 - \alpha_i) \eta_{i}^n`
        for :math:`i=1, 2, 3` and different :math:`\alpha_i \in [0,1]`,
        in an iterative :class:`~struphy.pic.pushing.pusher.Pusher`.
        """
        return getattr(self, "_eval_kernels", ())

    @property
    def rank(self):
        """MPI rank, is 0 if no communicator."""
        return self._rank

    @property
    def derham(self) -> Derham:
        """Derham spaces and projectors."""
        assert hasattr(
            self,
            "_derham",
        ), "Derham not set. Please do obj.derham = ..."
        assert isinstance(self._derham, Derham)
        return self._derham

    @derham.setter
    def derham(self, derham: Derham):
        assert isinstance(derham, Derham)
        self._derham = derham

    @property
    def domain(self) -> Domain:
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        assert hasattr(self, "_domain"), "Domain for analytical MHD equilibrium not set. Please do obj.domain = ..."
        assert isinstance(self._domain, Domain)
        return self._domain

    @domain.setter
    def domain(self, domain: Domain):
        assert isinstance(domain, Domain)
        self._domain = domain

    @property
    def mass_ops(self) -> WeightedMassOperators:
        """Weighted mass operators."""
        assert hasattr(self, "_mass_ops"), "Weighted mass operators not set. Please do obj.mass_ops = ..."
        assert isinstance(self._mass_ops, WeightedMassOperators)
        return self._mass_ops

    @mass_ops.setter
    def mass_ops(self, mass_ops: WeightedMassOperators):
        assert isinstance(mass_ops, WeightedMassOperators)
        self._mass_ops = mass_ops

    @property
    def basis_ops(self) -> BasisProjectionOperators:
        """Basis projection operators."""
        assert hasattr(self, "_basis_ops"), "Basis projection operators not set. Please do obj.basis_ops = ..."
        assert isinstance(self._basis_ops, BasisProjectionOperators)
        return self._basis_ops

    @basis_ops.setter
    def basis_ops(self, basis_ops: BasisProjectionOperators):
        assert isinstance(basis_ops, BasisProjectionOperators)
        self._basis_ops = basis_ops

    @property
    def projected_equil(self) -> ProjectedFluidEquilibriumWithB:
        """Fluid equilibrium projected on 3d Derham sequence with commuting projectors."""
        assert hasattr(
            self,
            "_projected_equil",
        ), "Projected MHD equilibrium not set."
        assert isinstance(self._projected_equil, ProjectedFluidEquilibriumWithB)
        return self._projected_equil

    @projected_equil.setter
    def projected_equil(self, new: ProjectedFluidEquilibriumWithB):
        assert isinstance(new, ProjectedFluidEquilibriumWithB)
        self._projected_equil = new

    @property
    def time_state(self):
        """A pointer to the time variable of the dynamics ('t')."""
        return self._time_state

    def add_time_state(self, time_state):
        """Add a pointer to the time variable of the dynamics ('t').

        Parameters
        ----------
        time_state : ndarray
            Of size 1, holds the current physical time 't'.
        """
        assert time_state.size == 1
        self._time_state = time_state

    def add_init_kernel(self, setup: KernelSetup):
        """Register an evaluation at the start of each push.

        The setup names the callable, its arguments, and the exact output
        marker columns. Its evaluation weights must all be zero.
        """
        if not isinstance(setup, KernelSetup):
            raise TypeError("init kernels must be KernelSetup instances")
        if any(setup.alpha):
            raise ValueError("init kernels must evaluate the initial state (alpha=0)")
        self._init_kernels = self.init_kernels + (setup,)

    def add_eval_kernel(self, setup: KernelSetup):
        """Register an evaluation before each pusher stage/iteration.

        The setup's alpha weights determine both the evaluation state and
        the preceding MPI sort.
        """
        if not isinstance(setup, KernelSetup):
            raise TypeError("eval kernels must be KernelSetup instances")
        self._eval_kernels = self.eval_kernels + (setup,)
