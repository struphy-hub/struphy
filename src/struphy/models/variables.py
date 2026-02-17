# for type checking (cyclic imports)
from __future__ import annotations

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
    """Single variable of a Species object.

    The solution of a model is a collection of Variables.
    Multiple Variables can be combined within a Species.
    For example, a Species 'Hydrogen' could be composed of the Variables density, velocity and temperature,
    which satisfy a PDE within a model.
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
                    print(f"  {background.__class__.__name__}:")
                    for k, v in background.__dict__.items():
                        print(f"    {k}: {v}")
            else:
                print(f"  {self.backgrounds.__class__.__name__}:")
                for k, v in self.backgrounds.__dict__.items():
                    print(f"    {k}: {v}")
        else:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no background.")

    def show_perturbations(self):
        if self.perturbations is not None:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - perturbations:")
            if isinstance(self.perturbations, list):
                for perturbation in self.perturbations:
                    print(f"  {perturbation.__class__.__name__}:")
                    for k, v in perturbation.__dict__.items():
                        print(f"    {k}: {v}")
            else:
                print(f"  {self.perturbations.__class__.__name__}:")
                for k, v in self.perturbations.__dict__.items():
                    print(f"    {k}: {v}")
        else:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - no perturbation.")


class FEECVariable(Variable):
    """Basic finite element variable for grid-based methods.

    Initial conditions for a FEECVariable consist of a background plus a perturbation, which are added up.
    If neither a background nor a perturbation is present, the Variable is initialized as zero.
    """

    def __init__(self, space: LiteralOptions.OptsFEECSpace = "H1"):
        check_option(space, LiteralOptions.OptsFEECSpace)
        self._space = space

    @property
    def space(self):
        return self._space

    @property
    def spline(self) -> SplineFunction:
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
    """Basic particle variable in PIC methods.

    A background is mandatory and can be used for noise reduction for instance.
    The initial condition is a kinetic background with optional perturbations added to it.
    If no inital condition is specified, the background is taken as inital condition.
    If both a background and an initial condition are specified, they should be consistent
    (i.e. the initial condition should be the background with perturbations on top)."""

    def __init__(self, space: LiteralOptions.OptsPICSpace = "Particles6D"):
        check_option(space, LiteralOptions.OptsPICSpace)
        self._space = space

    @property
    def space(self):
        return self._space

    @property
    def particles(self) -> Particles:
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
            print(f"  {self.initial_condition.__class__.__name__}:")
            for k, v in self.initial_condition.__dict__.items():
                print(f"    {k}: {v}")
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
    """Basic variable for SPH methods.

    Initial conditions for a SPHVariable consist of a background plus a perturbation for density and velocity, which are added up.
    If neither a background nor a perturbation is present, the Variable is initialized as zero.
    """

    def __init__(self):
        self._space = "ParticlesSPH"
        self._n_as_volume_form = True
        self._particle_data = {}

    @property
    def space(self):
        return self._space

    @property
    def particles(self) -> ParticlesSPH:
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
        print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - perturbations:")
        for key, perturbation in self.perturbations.items():
            if perturbation is not None:
                print(f"  {key}: {perturbation.__class__.__name__}")
                for k, v in perturbation.__dict__.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: None")

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
