# for type checking (cyclic imports)
from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.fields_background.base import FluidEquilibrium
from struphy.fields_background.projected_equils import ProjectedFluidEquilibrium
from struphy.geometry.base import Domain
from struphy.initial.perturbations import Perturbation
from struphy.io.options import FieldsBackground, LiteralOptions
from struphy.kinetic_background.base import KineticBackground
from struphy.pic import particles
from struphy.pic.base import Particles
from struphy.pic.particles import ParticlesSPH
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import check_option

if TYPE_CHECKING:
    from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies, Species


class Variable(metaclass=ABCMeta):
    """
    Abstract base class for a single variable of a Species object.

    Variables represent the fundamental degrees of freedom in a plasma simulation. The complete solution
    of a StruphyModel is a collection of Variables organized within Species. Multiple Variables can be
    combined within a Species to represent different physical quantities (e.g., density, velocity, temperature).

    Attributes
    ----------
    space : str
        The function space or discretization type of the variable (e.g., 'H1' for finite elements,
        'Particles6D' for PIC, 'ParticlesSPH' for SPH). Must be implemented by subclasses.
    backgrounds : FluidEquilibrium | KineticBackground | None
        Static background(s) representing equilibrium or reference state. Multiple backgrounds can be added
        and are combined by addition. Defaults to None.
    perturbations : Perturbation | list[Perturbation] | None
        Initial perturbations added to backgrounds to define initial conditions.
        Multiple perturbations can be added and are combined by addition. Defaults to None.
    save_data : bool
        Flag indicating whether this variable's data should be saved during simulation (default=True).
    species : Species
        Reference to the parent Species object containing this variable.

    Methods
    -------
    allocate()
        Allocate memory and initialize data structures. Must be implemented by subclasses.
    add_background(background)
        Add a static background condition to the variable.
    show_backgrounds()
        Display information about defined backgrounds.
    show_perturbations()
        Display information about defined perturbations.

    Notes
    -----
    The initial condition for a variable is constructed as: background(s) + perturbation(s).
    If neither is specified, the variable initializes to zero.

    All concrete implementations (FEECVariable, PICVariable, SPHVariable) must define the `space` property
    and implement the `allocate()` method.
    """

    @property
    @abstractmethod
    def space(self):
        """The function space of the variable, e.g. 'H1' for finite element variables or 'Particles6D' for PIC variables."""
        pass

    @abstractmethod
    def allocate(self):
        """Alocate object and memory for variable."""

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.space})"

    @property
    def backgrounds(self):
        """The static background. Multiple backgrounds can be defined in a list,
        via the add_background method."""
        if not hasattr(self, "_backgrounds"):
            self._backgrounds = None
        return self._backgrounds

    @property
    def perturbations(self):
        """The perturbations. Multiple perturbations can be defined in a list,
        via the add_perturbation method defined in sub-classes."""
        if not hasattr(self, "_perturbations"):
            self._perturbations = None
        return self._perturbations

    @property
    def save_data(self):
        """Store variable data during simulation (default=True)."""
        if not hasattr(self, "_save_data"):
            self._save_data = True
        return self._save_data

    @save_data.setter
    def save_data(self, new):
        assert isinstance(new, bool)
        self._save_data = new

    @property
    def species(self) -> Species:
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    @property
    def __name__(self):
        if not hasattr(self, "_name"):
            self._name = None
        return self._name

    def add_background(self, background, verbose=True):
        """Add a static background for this variable.
        Multiple backgrounds can be added up."""
        if not hasattr(self, "_backgrounds") or self.backgrounds is None:
            self._backgrounds = background
        else:
            if not isinstance(self.backgrounds, list):
                self._backgrounds = [self.backgrounds]
            self._backgrounds += [background]

    def show_backgrounds(self):
        if self.backgrounds is not None:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - backgrounds:")
            if isinstance(self.backgrounds, list):
                for background in self.backgrounds:
                    print(background)
            else:
                print(self.backgrounds)
        else:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no background.")

    def show_perturbations(self):
        if self.perturbations is not None:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - perturbations:")
            if isinstance(self.perturbations, list):
                for perturbation in self.perturbations:
                    print(perturbation)
            else:
                print(self.perturbations)
        else:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no perturbation.")


class FEECVariable(Variable):
    """
    Grid-based finite element variable using FEEC (Finite Element Exterior Calculus) discretization.

    FEECVariable represents field quantities discretized on a computational grid using finite element
    methods. It supports arbitrary FEEC spaces (H1, Hdiv, Hcurl, L2) and is used for electromagnetic
    fields, fluid moments, or other spatially distributed quantities.

    Attributes
    ----------
    space : str
        The FEEC function space. Options: 'H1', 'HDiv', 'HCurl', 'L2'.
        Determines the continuity and smoothness properties of the discretization.
    spline : SplineFunction
        The underlying spline-based representation of the field on the computational mesh.
        Only available after calling `allocate()`.
    species : FieldSpecies | FluidSpecies
        Parent species containing this variable.
    backgrounds : FieldsBackground | None
        Equilibrium fields added to initial conditions.
    perturbations : Perturbation | list[Perturbation] | None
        Initial perturbations added to backgrounds.

    Methods
    -------
    add_background(background)
        Add an equilibrium field background.
    add_perturbation(perturbation)
        Add initial perturbations to the field.
    allocate(derham, domain, equil, verbose)
        Allocate spline function and initialize on the mesh.

    Notes
    -----
    Initial conditions are constructed by combining backgrounds and perturbations:
    initial_field = background(s) + perturbation(s)

    If neither background nor perturbation is specified, the variable initializes to zero.

    Examples
    --------
    >>> E_field = FEECVariable(space='Hdiv')  # For electric field
    >>> E_field.add_background(FieldsBackground(type='uniform', magnitude=1.0))
    """

    def __init__(self, space: LiteralOptions.OptsFEECSpace = "H1"):
        check_option(space, LiteralOptions.OptsFEECSpace)
        self._space = space

    @property
    def space(self):
        return self._space

    @property
    def spline(self) -> SplineFunction:
        if not hasattr(self, "_spline"):
            raise ValueError("Warning: spline not allocated yet. Call allocate() first.")
        return self._spline

    @property
    def species(self) -> FieldSpecies | FluidSpecies:
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    def add_background(self, background: FieldsBackground, verbose=True):
        super().add_background(background, verbose=verbose)

    def add_perturbation(self, perturbation: Perturbation, verbose=True):
        """Add an initial :class:`~struphy.initial.base.Perturbation` for this variable.
        Multiple perturbations can be added up."""
        if not hasattr(self, "_perturbations") or self.perturbations is None:
            self._perturbations = perturbation
        else:
            if not isinstance(self.perturbations, list):
                self._perturbations = [self.perturbations]
            self._perturbations += [perturbation]

    def allocate(
        self,
        derham: Derham,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        verbose: bool = False,
    ):
        self._spline = derham.create_spline_function(
            name=self.__name__,
            space_id=self.space,
            backgrounds=self.backgrounds,
            perturbations=self.perturbations,
            domain=domain,
            equil=equil,
            verbose=verbose,
        )


class PICVariable(Variable):
    """
    Kinetic variable using Particle-In-Cell (PIC) discretization in phase space.

    PICVariable represents kinetic distribution functions discretized via marker/particle methods
    in 3D configuration space plus velocity space (3D+2D or 3D+3D). Supports various phase-space
    decompositions for efficient kinetic simulations.

    Attributes
    ----------
    space : str
        Phase space decomposition. Options include, among others:
        - 'Particles6D': Full 3D config + 3D velocity space
        - 'Particles5D': 3D config + 2D gyro-velocity space (for strong magnetic fields)
    particles : Particles
        The marker/particle object managing kinetic data. Only available after `allocate()`.
    particles_class : type
        Reference to the Particles class corresponding to the space.
    species : ParticleSpecies
        Parent kinetic species containing this variable.
    backgrounds : KineticBackground
        Mandatory equilibrium distribution function (e.g., Maxwellian).
    initial_condition : KineticBackground
        The initial kinetic distribution. If not explicitly set, uses the background.
    perturbations : Perturbation | list[Perturbation] | None
        Perturbations to moments of the distribution function.
    n_as_volume_form : bool
        Whether number density is represented as a differential form (volume-weighted) or scalar.
    n_to_save : int
        Number of markers for which trajectories are saved.
    saved_markers : ndarray
        Array storing saved marker data.

    Methods
    -------
    add_background(background, n_as_volume_form)
        Set mandatory equilibrium distribution (e.g., Maxwellian).
    add_initial_condition(init)
        Set initial kinetic distribution (must be consistent with background).
    show_initial_condition()
        Display current initial condition information.
    allocate(clone_config, derham, domain, equil, projected_equil, verbose)
        Initialize particles and allocate marker arrays.

    Notes
    -----
    The background is mandatory and often used for noise reduction. The initial condition should be
    consistent with the background to avoid numerical artifacts.

    If no initial condition is specified, the background is used as the initial condition.
    If both are specified, they should be the same base distribution with perturbations on top.

    Examples
    --------
    >>> electrons = PICVariable(space='Particles6D')
    >>> maxwellian = Maxwellian3D(n=(1.0, None), vth2=(0.1, None))
    >>> electrons.add_background(maxwellian)
    >>> pert = TorusModesCos(amps=(0.1,))
    >>> electrons.add_perturbation(pert)
    """

    def __init__(self, space: LiteralOptions.OptsPICSpace = "Particles6D"):
        check_option(space, LiteralOptions.OptsPICSpace)
        self._space = space
        for name, cls in inspect.getmembers(particles):
            if inspect.isclass(cls) and cls.__module__ == particles.__name__ and name == space:
                self._particles_class = cls

    @property
    def space(self):
        return self._space

    @property
    def particles_class(self) -> Particles:
        return self._particles_class

    @property
    def particles(self) -> Particles:
        if not hasattr(self, "_particles"):
            raise ValueError("Warning: particles not allocated yet. Call allocate() first.")
        return self._particles

    @property
    def species(self) -> ParticleSpecies:
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    @property
    def n_as_volume_form(self) -> bool:
        """Whether the number density n is given as a volume form or a scalar function (=default)."""
        if not hasattr(self, "_n_as_volume_form"):
            self._n_as_volume_form = False
        return self._n_as_volume_form

    def add_background(self, background: KineticBackground, n_as_volume_form: bool = False, verbose=True):
        self._n_as_volume_form = n_as_volume_form
        super().add_background(background, verbose=verbose)

    def add_initial_condition(self, init: KineticBackground, verbose=True):
        """The initial condition must be consistent with the background."""
        self._initial_condition = init

    def show_initial_condition(self):
        if self.initial_condition is not None:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - initial condition:")
            print(self.initial_condition)
        else:
            print(
                f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no initial condition."
            )

    @property
    def initial_condition(self) -> KineticBackground:
        if not hasattr(self, "_initial_condition"):
            self._initial_condition = self.backgrounds
        return self._initial_condition

    def allocate(
        self,
        clone_config: CloneConfig = None,
        derham: Derham = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        projected_equil: ProjectedFluidEquilibrium = None,
        verbose: bool = False,
    ):
        # assert isinstance(self.species, KineticSpecies)
        assert isinstance(self.backgrounds, KineticBackground), (
            "List input not allowed, you can sum Kineticbackgrounds before passing them to add_background."
        )

        if derham is None:
            domain_decomp = None
        else:
            domain_array = derham.domain_array
            nprocs = derham.domain_decomposition.nprocs
            domain_decomp = (domain_array, nprocs)

        kinetic_class = getattr(particles, self.space)

        comm_world = MPI.COMM_WORLD
        if comm_world.Get_size() == 1:
            comm_world = None

        self._particles: Particles = kinetic_class(
            comm_world=comm_world,
            clone_config=clone_config,
            domain_decomp=domain_decomp,
            mpi_dims_mask=self.species.dims_mask,
            boxes_per_dim=self.species.boxes_per_dim,
            box_bufsize=self.species.box_bufsize,
            name=self.species.__class__.__name__,
            loading_params=self.species.loading_params,
            weights_params=self.species.weights_params,
            boundary_params=self.species.boundary_params,
            bufsize=self.species.bufsize,
            domain=domain,
            equil=equil,
            projected_equil=projected_equil,
            background=self.backgrounds,
            initial_condition=self.initial_condition,
            n_as_volume_form=self.n_as_volume_form,
            # perturbations=self.perturbations,
            equation_params=self.species.equation_params,
            verbose=verbose,
        )

        if self.species.do_sort:
            sort = True
        else:
            sort = False
        self.particles.draw_markers(sort=sort, verbose=verbose)
        self.particles.initialize_weights()

        # allocate array for saving markers if not present
        n_markers = self.species.n_markers
        if isinstance(n_markers, float):
            if n_markers > 1.0:
                self._n_to_save = int(n_markers)
            else:
                self._n_to_save = int(self.particles.n_mks_global * n_markers)
        else:
            self._n_to_save = n_markers

        assert self._n_to_save <= self.particles.Np, (
            f"The number of markers for which data should be stored (={self._n_to_save}) murst be <= than the total number of markers (={self.particles.Np})"
        )
        if self._n_to_save > 0:
            self._saved_markers = xp.zeros(
                (self._n_to_save, self.particles.markers.shape[1]),
                dtype=float,
            )

        # other data (wave-particle power exchange, etc.)
        # TODO

    @property
    def n_to_save(self) -> int:
        return self._n_to_save

    @property
    def saved_markers(self) -> xp.ndarray:
        return self._saved_markers


class SPHVariable(Variable):
    """
    Fluid variable using Smoothed Particle Hydrodynamics (SPH) discretization.

    SPHVariable represents macroscopic fluid quantities (density, velocity, temperature) using
    SPH methods, where fields are reconstructed from marker positions and properties. This is a
    particle-based Lagrangian approach to fluid dynamics.

    Attributes
    ----------
    space : str
        Always 'ParticlesSPH' for SPH method.
    particles : ParticlesSPH
        The SPH marker/particle object. Only available after `allocate()`.
    particles_class : type
        Reference to the ParticlesSPH class.
    particle_data : dict
        Dictionary storing particle-associated data fields.
    species : ParticleSpecies
        Parent SPH species containing this variable.
    backgrounds : FluidEquilibrium
        Background fluid state (density, velocity, pressure profiles).
    perturbations : dict[str, Perturbation]
        Perturbations to density ('n') and velocity components ('u1', 'u2', 'u3').
        Each component can have independent perturbations or None.
    n_as_volume_form : bool
        Always True for SPH; number density represented as volume-weighted quantity.
    n_to_save : int
        Number of SPH particles for which data is saved.
    saved_markers : ndarray
        Array storing saved particle data.

    Methods
    -------
    add_background(background)
        Set the background fluid equilibrium state.
    add_perturbation(del_n, del_u1, del_u2, del_u3)
        Add perturbations to density and/or velocity components.
    show_perturbations()
        Display detailed information about density and velocity perturbations.
    allocate(derham, domain, equil, projected_equil, verbose)
        Initialize SPH particles and allocate marker arrays.

    Notes
    -----
    Initial conditions combine background and perturbations:
    - density: background + del_n
    - velocity: background + (del_u1, del_u2, del_u3)

    If neither background nor perturbations are specified, fields initialize to zero.

    Examples
    --------
    >>> fluid = SPHVariable()
    >>> equil = HomogenSlab()
    >>> fluid.add_background(equil)
    >>> pert = TorusModesCos(amps=(0.1,))
    >>> fluid.add_perturbation(del_n=pert, del_u1=pert)
    """

    def __init__(self):
        self._space = "ParticlesSPH"
        self._n_as_volume_form = True
        self._particle_data = {}

    @property
    def space(self):
        return self._space

    @property
    def particles_class(self) -> Particles:
        return ParticlesSPH

    @property
    def particles(self) -> ParticlesSPH:
        if not hasattr(self, "_particles"):
            raise ValueError("Warning: particles not allocated yet. Call allocate() first.")
        return self._particles

    @property
    def particle_data(self):
        return self._particle_data

    @property
    def species(self) -> ParticleSpecies:
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    @property
    def n_as_volume_form(self) -> bool:
        """Whether the number density n is given as a volume form or scalar function (=default)."""
        return self._n_as_volume_form

    def add_background(self, background: FluidEquilibrium, verbose=True):
        super().add_background(background, verbose=verbose)

    def add_perturbation(
        self,
        del_n: Perturbation = None,
        del_u1: Perturbation = None,
        del_u2: Perturbation = None,
        del_u3: Perturbation = None,
        verbose=True,
    ):
        """Add an initial :class:`~struphy.initial.base.Perturbation` for the fluid density and/or velocity."""
        self._perturbations = {}
        self._perturbations["n"] = del_n
        self._perturbations["u1"] = del_u1
        self._perturbations["u2"] = del_u2
        self._perturbations["u3"] = del_u3

    def show_perturbations(self):
        if self.perturbations is not None:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - perturbations:")
            for key, perturbation in self.perturbations.items():
                if perturbation is not None:
                    print(f"    {key}: {perturbation.__class__.__name__}")
                    for k, v in perturbation.__dict__.items():
                        print(f"        {k}: {v}")
                else:
                    print(f"    {key}: None")
        else:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no perturbation.")

    @property
    def perturbations(self) -> dict[str, Perturbation]:
        if not hasattr(self, "_perturbations"):
            self._perturbations = None
        return self._perturbations

    def allocate(
        self,
        derham: Derham = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        projected_equil: ProjectedFluidEquilibrium = None,
        verbose: bool = False,
    ):
        assert isinstance(self.backgrounds, FluidEquilibrium), (
            "List input not allowed, you can sum Kineticbackgrounds before passing them to add_background."
        )

        self.backgrounds.domain = domain

        if derham is None:
            domain_decomp = None
        else:
            domain_array = derham.domain_array
            nprocs = derham.domain_decomposition.nprocs
            domain_decomp = (domain_array, nprocs)

        comm_world = MPI.COMM_WORLD
        if comm_world.Get_size() == 1:
            comm_world = None

        self._particles = ParticlesSPH(
            comm_world=comm_world,
            domain_decomp=domain_decomp,
            mpi_dims_mask=self.species.dims_mask,
            boxes_per_dim=self.species.boxes_per_dim,
            box_bufsize=self.species.box_bufsize,
            name=self.species.__class__.__name__,
            loading_params=self.species.loading_params,
            weights_params=self.species.weights_params,
            boundary_params=self.species.boundary_params,
            bufsize=self.species.bufsize,
            domain=domain,
            equil=equil,
            projected_equil=projected_equil,
            background=self.backgrounds,
            n_as_volume_form=self.n_as_volume_form,
            perturbations=self.perturbations,
            equation_params=self.species.equation_params,
            verbose=verbose,
        )

        if self.species.do_sort:
            sort = True
        else:
            sort = False
        self.particles.draw_markers(sort=sort, verbose=verbose)
        self.particles.initialize_weights()

        # allocate array for saving markers if not present
        n_markers = self.species.n_markers
        if isinstance(n_markers, float):
            if n_markers > 1.0:
                self._n_to_save = int(n_markers)
            else:
                self._n_to_save = int(self.particles.n_mks_global * n_markers)
        else:
            self._n_to_save = n_markers

        assert self._n_to_save <= self.particles.Np, (
            f"The number of markers for which data should be stored (={self._n_to_save}) murst be <= than the total number of markers (={self.particles.Np})"
        )
        if self._n_to_save > 0:
            self._saved_markers = xp.zeros(
                (self._n_to_save, self.particles.markers.shape[1]),
                dtype=float,
            )

        # other data (wave-particle power exchange, etc.)
        # TODO

    @property
    def n_to_save(self) -> int:
        return self._n_to_save

    @property
    def saved_markers(self) -> xp.ndarray:
        return self._saved_markers
