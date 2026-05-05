import logging
from abc import ABCMeta, abstractmethod
from typing import Callable

import cunumpy as xp

from struphy.io.options import LiteralOptions
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class Perturbation(metaclass=ABCMeta):
    """Abstract base class for perturbation functions used as initial conditions in simulations.

    This class provides the interface and common functionality for defining perturbation fields
    in logical (eta) or physical coordinate spaces. Subclasses must implement the ``__call__``
    method to define the perturbation as a callable function of spatial coordinates.

    The class supports flexible representation bases (p-forms, vector fields, physical coordinates)
    and allows specification of which component is perturbed for vector-valued quantities.

    Attributes
    ----------
    given_in_basis : str
        Specifies the basis representation of the perturbation. Options:

        - '0', '1', '2', '3' : Differential form basis (0-form=scalar, 1-form, etc.)
        - 'v' : Vector field basis
        - 'physical' : Physical (mapped) domain coordinates
        - 'physical_at_eta' : Physical components evaluated in logical (eta) domain, u(F(eta))
        - 'norm' : Normalized co-variant basis (:math:`delta_i / |delta_i|`)

    comp : int
        Component index for vector-valued perturbations (0-2 for vector components,
        0 for scalar-valued functions). Default is 0.

    Examples
    --------
    Subclasses should override ``__call__`` to implement specific perturbation fields:

    >>> class CustomPerturbation(Perturbation):
    ...     def __init__(self):
    ...         self.given_in_basis = 'physical'
    ...     def __call__(self, eta1, eta2, eta3, flat_eval=False):
    ...         return eta1 * eta2  # Example perturbation field
    """

    @abstractmethod
    def __call__(self, eta1, eta2, eta3, flat_eval=False):
        """Evaluate the perturbation field at given coordinates.

        Parameters
        ----------
        eta1, eta2, eta3 : ndarray or float
            Coordinate values in the eta (logical) space, or physical space depending on
            the perturbation's basis representation.

        flat_eval : bool, default=False
            If True, treat inputs as flattened arrays and return flattened output.
            If False, preserve the array shapes for meshgrid-like evaluation.

        Returns
        -------
        ndarray or float
            Perturbation field values at the given coordinates, with shape matching
            the input coordinates (or flattened if flat_eval=True).
        """
        pass

    def prepare_eval_pts(self):
        # TODO: we could prepare the arguments via a method in this base class (flat_eval, sparse meshgrid, etc.).
        pass

    def __repr__(self):
        logger.info(f"    {self.__class__.__name__}:")
        for k, v in self.__dict__.items():
            logger.info(f"        {k}: {v}")
        return ""

    @property
    def given_in_basis(self) -> str:
        r"""In which basis the perturbation is represented, must be set in child class (use the setter below).

        Either
            * '0', '1', '2' or  '3' for a p-form basis
            * 'v' for a vector-field basis
            * 'physical' when defined on the physical (mapped) domain
            * 'physical_at_eta' when given the physical components evaluated on the logical domain, u(F(eta))
            * 'norm' when given in the normalized co-variant basis (:math:`\delta_i / |\delta_i|`)
        """
        return self._given_in_basis

    @given_in_basis.setter
    def given_in_basis(self, new: str):
        check_option(new, LiteralOptions.GivenInBasis)
        self._given_in_basis = new

    @property
    def comp(self) -> int:
        """Which component of vector is perturbed (=0 for scalar-valued functions).
        Can be set in child class (use the setter below)."""
        if not hasattr(self, "_comp"):
            self._comp = 0
        return self._comp

    @comp.setter
    def comp(self, new: int):
        assert new in (0, 1, 2)
        self._comp = new

    def _mask_subdomain(self, *etas, perb_domain: tuple[tuple[float] | None, ...] = (None, None, None)):
        """
        Create mask of particles within perturbation subdomain.

        :param eta: location of each partiles in space
        :param perb_domain : tuple[tuple[float]]
            Subdomain in logical space in which the pertrubation is applied to: ((eta1_min, eta1_max), (eta2_min, eta2_max), (eta3_min, eta3_max)).
            None means apply perturbation to all domain in that direction
        """

        mask = xp.ones_like(etas[0], dtype=bool)

        for i in range(len(perb_domain)):
            if perb_domain[i] is None:
                continue
            mask &= (perb_domain[i][0] <= etas[i]) & (etas[i] <= perb_domain[i][1])

        return mask


class GenericPerturbation(Perturbation):
    """Generic perturbation defined by a user-provided function.

    Parameters
    ----------
    fun: callable
        A function of three variables (x, y, z) defining the perturbation in physical space.

    given_in_basis: str
        In which basis the perturbation is represented (see base class).

    comp: int
        Which component (0, 1 or 2) of vector is perturbed (=0 for scalar-valued functions)
    """

    def __init__(
        self,
        fun: callable,
        given_in_basis: LiteralOptions.GivenInBasis = None,
        comp: int = 0,
    ):
        self.fun = fun

        # use the setters
        self.given_in_basis = given_in_basis
        self.comp = comp

    def __call__(self, x, y, z):
        return self.fun(x, y, z)
