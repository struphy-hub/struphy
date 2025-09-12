# for type checking (cyclic imports)
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.fields_background.base import FluidEquilibrium
from struphy.fields_background.projected_equils import ProjectedFluidEquilibrium
from struphy.geometry.base import Domain
from struphy.initial.perturbations import Perturbation
from struphy.io.options import (
    FieldsBackground,
    OptsFEECSpace,
    OptsPICSpace,
    check_option,
)
from struphy.kinetic_background.base import KineticBackground
from struphy.pic import particles
from struphy.pic.base import Particles
from struphy.utils.clone_config import CloneConfig

if TYPE_CHECKING:
    from struphy.models.species import FieldSpecies, FluidSpecies, KineticSpecies, Species


class Variable(metaclass=ABCMeta):
    """Single variable (unknown) of a Species."""

    @abstractmethod
    def allocate(self):
        """Alocate object and memory for variable."""

    @property
    def backgrounds(self):
        if not hasattr(self, "_backgrounds"):
            self._backgrounds = None
        return self._backgrounds

    @property
    def perturbations(self):
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
        """Type inference of added background done in sub class."""
        if not hasattr(self, "_backgrounds") or self.backgrounds is None:
            self._backgrounds = background
        else:
            if not isinstance(self.backgrounds, list):
                self._backgrounds = [self.backgrounds]
            self._backgrounds += [background]

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(
                f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - added background '{background.__class__.__name__}' with:"
            )
            for k, v in background.__dict__.items():
                print(f"  {k}: {v}")

    def add_perturbation(self, perturbation: Perturbation, verbose=True):
        if not hasattr(self, "_perturbations") or self.perturbations is None:
            self._perturbations = perturbation
        else:
            if not isinstance(self.perturbations, list):
                self._perturbations = [self.perturbations]
            self._perturbations += [perturbation]

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(
                f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - added perturbation '{perturbation.__class__.__name__}' with:"
            )
            for k, v in perturbation.__dict__.items():
                print(f"  {k}: {v}")


class FEECVariable(Variable):
    def __init__(self, space: OptsFEECSpace = "H1"):
        check_option(space, OptsFEECSpace)
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

    def allocate(
        self,
        derham: Derham,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
    ):
        self._spline = derham.create_spline_function(
            name=self.__name__,
            space_id=self.space,
            backgrounds=self.backgrounds,
            perturbations=self.perturbations,
            domain=domain,
            equil=equil,
        )


class PICVariable(Variable):
    def __init__(self, space: OptsPICSpace = "Particles6D"):
        check_option(space, OptsPICSpace)
        self._space = space
        self._kinetic_data = {}

    @property
    def space(self):
        return self._space

    @property
    def particles(self) -> Particles:
        return self._particles

    @property
    def kinetic_data(self):
        return self._kinetic_data

    @property
    def species(self) -> KineticSpecies:
        if not hasattr(self, "_species"):
            self._species = None
        return self._species

    @property
    def n_as_volume_form(self) -> bool:
        """Whether the number density n is given as a volume form or scalar function (=default)."""
        if not hasattr(self, "_n_as_volume_form"):
            self._n_as_volume_form = False
        return self._n_as_volume_form

    def add_background(self, background: KineticBackground, n_as_volume_form: bool = False, verbose=True):
        self._n_as_volume_form = n_as_volume_form
        super().add_background(background, verbose=verbose)

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
            f"List input not allowed, you can sum Kineticbackgrounds before passing them to add_background."
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

        # for storing the binned distribution function
        self.kinetic_data["bin_edges"] = {}
        self.kinetic_data["f"] = {}
        self.kinetic_data["df"] = {}
        
        for bin_plot in self.species.binning_plots:
            sli = bin_plot.slice
            n_bins = bin_plot.n_bins
            ranges = bin_plot.ranges
            
            assert ((len(sli) - 2) / 3).is_integer(), f"Binning coordinates must be separated by '_', but reads {sli}."
            assert len(sli.split("_")) == len(ranges) == len(n_bins), (
                f"Number of slices names ({len(sli.split('_'))}), number of bins ({len(n_bins)}), and number of ranges ({len(ranges)}) are inconsistent with each other!\n\n"
            )
            self.kinetic_data["bin_edges"][sli] = []
            dims = (len(sli) - 2) // 3 + 1
            for j in range(dims):
                self.kinetic_data["bin_edges"][sli] += [
                    np.linspace(
                        ranges[j][0],
                        ranges[j][1],
                        n_bins[j] + 1,
                    ),
                ]
            self.kinetic_data["f"][sli] = np.zeros(n_bins, dtype=float)
            self.kinetic_data["df"][sli] = np.zeros(n_bins, dtype=float)

        # for storing an sph evaluation of the density n
        if self.species.n_sph is not None:
            plot_pts = self.species.n_sph["plot_pts"]

            self.kinetic_data["n_sph"] = []
            self.kinetic_data["plot_pts"] = []
            for i, pts in enumerate(plot_pts):
                assert len(pts) == 3
                eta1 = np.linspace(0.0, 1.0, pts[0])
                eta2 = np.linspace(0.0, 1.0, pts[1])
                eta3 = np.linspace(0.0, 1.0, pts[2])
                ee1, ee2, ee3 = np.meshgrid(
                    eta1,
                    eta2,
                    eta3,
                    indexing="ij",
                )
                self.kinetic_data["plot_pts"] += [(ee1, ee2, ee3)]
                self.kinetic_data["n_sph"] += [np.zeros(ee1.shape, dtype=float)]

        # other data (wave-particle power exchange, etc.)
        # TODO   
    
    
class SPHVariable(Variable):
    pass
