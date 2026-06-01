import logging
import warnings
from abc import ABCMeta, abstractmethod

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.models.variables import Variable
from struphy.particles.parameters import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    SavingParameters,
    SortingParameters,
    WeightsParameters,
)
from struphy.physics.physics import ConstantsOfNature, Units

logger = logging.getLogger("struphy")


class Species(metaclass=ABCMeta):
    """
    Abstract base class representing a single plasma species in a StruphyModel.

    This class manages variables, charge/mass properties, and equation parameters for a plasma species.
    Concrete implementations include FieldSpecies, FluidSpecies, ParticleSpecies, and DiagnosticSpecies.

    Attributes
    ----------
    variables : dict
        Dictionary of Variable objects associated with this species, keyed by variable name.
        Variables are automatically discovered during initialization.
    charge_number : int
        Charge number in units of elementary charge (default = 1).
        For field species, this is set to 0.
    mass_number : int
        Mass number in units of proton mass (default = 1).
        For field species, this is set to 0.
    alpha : float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency.
        If None, computed from units and charge/mass numbers.
    epsilon : float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit).
        If None, computed from units and charge/mass numbers.
    kappa : float, optional
        Normalized plasma frequency: plasma frequency × time unit.
        If None, computed from units and charge/mass numbers.
    equation_params : EquationParameters
        Object containing normalization parameters (alpha, epsilon, kappa) for scaled equations.

    Methods
    -------
    init_variables()
        Discover and cache Variable objects from instance attributes.
    setup_equation_params(units)
        Compute equation normalization parameters from physical units.

    Notes
    -----
    All Species subclasses must implement __init__() and call init_variables() to properly
    set up the variables dictionary.
    """

    @abstractmethod
    def __init__(self):
        self.init_variables()

    def __repr__(self):
        out = ""
        for k, v in self.variables.items():
            out += f"        {k}:".ljust(20)
            out += f"{v}\n"
        return out

    def init_variables(
        self,
        charge_number: int = 1,
        mass_number: int = 1,
        alpha: float = None,
        epsilon: float = None,
        kappa: float = None,
    ):
        """Create variables dict and set physical and equation parameters for a plasma species.

        Sets the charge and mass numbers, and optionally overrides normalized equation parameters
        (alpha, epsilon, kappa) that would otherwise be computed from physical units.

        Parameters
        ----------
        charge_number : int, optional
            Charge number in units of elementary charge (default = 1).
        mass_number : int, optional
            Mass number in units of proton mass (default = 1).
        alpha : float, optional
            Dimensionless parameter: plasma frequency / cyclotron frequency.
            If None, computed from units and charge/mass numbers (default = None).
        epsilon : float, optional
            Normalized cyclotron period: 1 / (cyclotron frequency × time unit).
            If None, computed from units and charge/mass numbers (default = None).
        kappa : float, optional
            Normalized plasma frequency: plasma frequency × time unit.
            If None, computed from units and charge/mass numbers (default = None)."""

        # create variables dict
        self._variables = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Variable):
                v._name = k
                v._species = self
                self._variables[k] = v

        # set species properties
        self._charge_number = charge_number
        self._mass_number = mass_number
        self._alpha = alpha
        self._epsilon = epsilon
        self._kappa = kappa

    @property
    def variables(self) -> dict:
        return self._variables

    @property
    def charge_number(self) -> int:
        """Charge number in units of elementary charge (default = 1)."""
        if not hasattr(self, "_charge_number"):
            self._charge_number = 1
        return self._charge_number

    @property
    def mass_number(self) -> int:
        """Mass number in units of proton mass (default = 1)."""
        if not hasattr(self, "_mass_number"):
            self._mass_number = 1
        return self._mass_number

    @property
    def alpha(self) -> float:
        """The ratio of plasma frequency to cyclotron frequency, Omega_p / Omega_c (default = None)."""
        if not hasattr(self, "_alpha"):
            self._alpha = None
        return self._alpha

    @property
    def epsilon(self) -> float:
        """The normalized cyclotron period, 1/(Omega_c * time_unit), default = None."""
        if not hasattr(self, "_epsilon"):
            self._epsilon = None
        return self._epsilon

    @property
    def kappa(self) -> float:
        """The normalized plasma frequency, Omega_p * time_unit (default = None)."""
        if not hasattr(self, "_kappa"):
            self._kappa = None
        return self._kappa

    class EquationParameters:
        """Normalization parameters of one species, appearing in scaled equations."""

        def __init__(
            self,
            species,
            units: Units = None,
            alpha: float = None,
            epsilon: float = None,
            kappa: float = None,
        ):
            self.species = species

            if units is None:
                units = Units()

            Z = species.charge_number
            A = species.mass_number

            con = ConstantsOfNature()

            # relevant frequencies
            om_p = xp.sqrt(units.n * (Z * con.e) ** 2 / (con.eps0 * A * con.mH))
            om_c = Z * con.e * units.B / (A * con.mH)

            # compute equation parameters
            if alpha is None:
                self.alpha = om_p / om_c
            else:
                self.alpha = alpha
                if MPI.COMM_WORLD.Get_rank() == 0:
                    warnings.warn(f"Override equation parameter {self.alpha =}")

            if epsilon is None:
                self.epsilon = 1.0 / (om_c * units.t)
            else:
                self.epsilon = epsilon
                if MPI.COMM_WORLD.Get_rank() == 0:
                    warnings.warn(f"Override equation parameter {self.epsilon =}")

            if kappa is None:
                self.kappa = om_p * units.t
            else:
                self.kappa = kappa
                if MPI.COMM_WORLD.Get_rank() == 0:
                    warnings.warn(f"Override equation parameter {self.kappa =}")

        def show(self):
            print(f"\nEquation parameters for species {self.species.__class__.__name__}:")
            for key, val in self.__dict__.items():
                if key != "species":
                    print(f"{(key + ':').ljust(25)} {val:4.3e}")

    @property
    def equation_params(self) -> EquationParameters:
        return self._equation_params

    def setup_equation_params(self, units: Units):
        """Set the following equation parameters:

        * alpha = plasma-frequenca / cyclotron frequency
        * epsilon = 1 / (cyclotron frequency * time unit)
        * kappa = plasma frequency * time unit
        """
        self._equation_params = self.EquationParameters(
            species=self,
            units=units,
            alpha=self.alpha,
            epsilon=self.epsilon,
            kappa=self.kappa,
        )


class FieldSpecies(Species):
    """
    Represents a field species with zero mass and charge.

    Field species are used to represent electromagnetic or other non-particle fields in a plasma
    model. They have no direct physical mass or charge properties (charge_number = 0, mass_number = 0),
    but may have associated equation parameters for scaled formulations.
    """

    def init_variables(self):
        """Create variables dict.
        For field species, set charge and mass numbers to zero by default."""
        super().init_variables(charge_number=0, mass_number=0)


class FluidSpecies(Species):
    """
    Represents a single fluid species evolving in 3D configuration space.

    FluidSpecies describes macroscopic plasma dynamics using fluid or moment-based equations
    (e.g., Euler equations, MHD equations). Each fluid species has a specific charge and mass number
    and evolves according to fluid propagators.

    Examples
    --------
    >>> electrons = MyFluidSpecies(charge_number=-1, mass_number=1/1836)  # for electrons
    """


class ParticleSpecies(Species):
    """
    Represents a single kinetic species in 3D configuration space plus velocity space.

    ParticleSpecies describes plasma dynamics using kinetic theory via particles or markers
    in 3D + vdim (where vdim is 2 or 3) phase space. This class manages particle initialization,
    sorting, and diagnostic output.

    Methods
    -------
    set_markers(loading_params, weights_params, boundary_params, sorting_params, saving_params, bufsize)
        Configure marker initialization and parameters.

    Examples
    --------
    >>> electrons = MyParticleSpecies(charge_number=-1, mass_number=1/1836)
    >>> load_params = LoadingParameters(Np=100000)
    >>> electrons.set_markers(loading_params=load_params)
    """

    def set_markers(
        self,
        loading_params: LoadingParameters = None,
        weights_params: WeightsParameters = None,
        boundary_params: BoundaryParameters = None,
        sorting_params: SortingParameters = None,
        saving_params: SavingParameters = None,
        bufsize: float = 1.0,
    ):
        """Set marker parameters for loading, weight calculation, kernel density reconstruction
        and boundary conditions in parameter/launch files.

        Parameters
        ----------
        loading_params : LoadingParameters

        weights_params : WeightsParameters

        boundary_params : BoundaryParameters

        sorting_params : SortingParameters

        saving_params : SavingParameters

        bufsize : float
            Size of buffer (as multiple of total size, default=.25) in markers array.
        """

        # defaults
        if loading_params is None:
            loading_params = LoadingParameters()

        if weights_params is None:
            weights_params = WeightsParameters()

        if boundary_params is None:
            boundary_params = BoundaryParameters()

        if sorting_params is None:
            sorting_params = SortingParameters()

        if saving_params is None:
            saving_params = SavingParameters()

        self.loading_params = loading_params
        self.weights_params = weights_params
        self.boundary_params = boundary_params
        self.sorting_params = sorting_params
        self.saving_params = saving_params
        self.bufsize = bufsize

        logger.info(f"\nMarker parameters for species '{self.__class__.__name__}':")
        logger.info(self.loading_params.__repr_no_defaults__())
        logger.info(self.weights_params.__repr_no_defaults__())
        logger.info(self.boundary_params.__repr_no_defaults__())
        logger.info(self.sorting_params.__repr_no_defaults__())
        logger.info(self.saving_params.__repr_no_defaults__())
        logger.info(f"Marker array buffer size: {self.bufsize * 100:.1f}% of total size")

    # def set_sorting_boxes(
    #     self,
    #     do_sort: bool = False,
    #     sorting_frequency: int = 0,
    #     boxes_per_dim: tuple = (12, 12, 1),
    #     box_bufsize: float = 2.0,
    #     dims_mask: tuple = (True, True, True),
    # ):
    #     """Set options for sorting particles/markers in parameter/launch files.
    #     The sorting boxes are used to sort particles in memory and for SPH kernel evaluations.

    #     For SPH kernel evaluation, the box size 1.0/boxes_per_dim[i] defines the maximal
    #     kernel width in direction i.

    #     Parameters
    #     ----------
    #     do_sort: bool
    #         Whether to sort particles in memory.

    #     sorting_frequency: int
    #         The number of time steps between two sortings (=0 means no sorting is performed).

    #     boxes_per_dim: tuple
    #         Number of boxes in each direction of logical space, (n_eta1, n_eta2, n_eta3).

    #     box_bufsize : float
    #         Between 0 and 1, relative buffer size for box array (default = 0.25).
    #         A number of 1.0 means that the box array is double the size needed to hold N/n_boxes particles,
    #         where N is the total number of particles.

    #     mpi_dims_mask: tuple[bool]
    #         True if the dimension is to be used in the domain decomposition (=default for each dimension).
    #         If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
    #     """
    #     self.do_sort = do_sort
    #     self.sorting_fequency = sorting_frequency
    #     self.boxes_per_dim = boxes_per_dim
    #     self.box_bufsize = box_bufsize
    #     self.dims_mask = dims_mask

    # def set_save_data(
    #     self,
    #     n_markers: int | float = 3,
    #     binning_plots: tuple[BinningPlot] = (),
    #     kernel_density_plots: tuple[KernelDensityPlot] = (),
    # ):
    #     """Set options for saving particle/marker information in parameter/launch files.

    #     Parameters
    #     ----------
    #     n_markers: int | float
    #         Number of particles/markers for which to save trajectories.
    #         If float and <1.0, then understood as the fraction of the total number of markers.

    #     binned_data: tuple[BinningPlot]
    #         A tuple of BinningPlot objects.

    #     kernel_density_plots: tuple[KernelDensityPlot]
    #         A tuple of KernelDensityPlot objects.
    #     """
    #     self.n_markers = n_markers
    #     self.binning_plots = binning_plots
    #     self.kernel_density_plots = kernel_density_plots


class DiagnosticSpecies(Species):
    """
    Represents a diagnostic species for output and analysis of non-physical quantities.

    DiagnosticSpecies are used to track derived quantities and diagnostics that are not part
    of the primary simulation fields. They have zero mass and charge and are typically computed
    from other species data during simulation postprocessing or on-the-fly diagnostics.

    Notes
    -----
    Diagnostic species do not directly participate in the dynamics equations but are updated
    based on values from other species (field, fluid, or particle species).

    Examples
    --------
    >>> vorticity = DiagnosticSpecies()  # For tracking curl of velocity field
    >>> energy = DiagnosticSpecies()  # For tracking kinetic/thermal energy density
    """
