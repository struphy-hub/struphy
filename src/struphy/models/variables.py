from abc import ABCMeta, abstractmethod
from mpi4py import MPI
import inspect

from struphy.initial.base import InitialCondition
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.io.options import FieldsBackground
from struphy.initial.perturbations import Perturbation
from struphy.geometry.base import Domain
from struphy.fields_background.base import FluidEquilibrium
from struphy.fields_background.projected_equils import ProjectedFluidEquilibrium
from struphy.pic.base import Particles
from struphy.kinetic_background.base import Maxwellian
from struphy.pic import particles
from struphy.models.species import Species, KineticSpecies
from struphy.utils.clone_config import CloneConfig


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

    def add_background(self, background, verbose=True):
        """Type inference of added background done in sub class."""
        if not hasattr(self, "_backgrounds") or self.backgrounds is None:
            self._backgrounds = background
        else:
            if not isinstance(self.backgrounds, list):
                self._backgrounds = [self.backgrounds]
            self._backgrounds += [background]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - added background '{background.__class__.__name__}' with:")
            for k, v in background.__dict__.items():
                print(f'  {k}: {v}')

    def add_perturbation(self, perturbation: Perturbation, verbose=True):
        if not hasattr(self, "_perturbations") or self.perturbations is None:
            self._perturbations = perturbation
        else:
            if not isinstance(self.perturbations, list):
                self._perturbations = [self.perturbations]
            self._perturbations += [perturbation]
        
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nVariable '{self.__name__}' of species '{self.species.__class__.__name__}' - added perturbation '{perturbation.__class__.__name__}' with:")
            for k, v in perturbation.__dict__.items():
                print(f'  {k}: {v}')

    def define_initial_condition(self):
        self._initial_condition = InitialCondition(
            background=self.backgrounds,
            perturbation=self.perturbations,
        )

    
class FEECVariable(Variable):
    def __init__(self, name: str = "a_feec_var", space: str = "H1"):
        assert space in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")
        self._name = name
        self._space = space
        
    @property
    def __name__(self):
        return self._name
        
    @property
    def space(self):
        return self._space
    
    @property
    def spline(self) -> SplineFunction:
        return self._spline
    
    def add_background(self, background: FieldsBackground, verbose=True):
        super().add_background(background, verbose=verbose)
    
    def allocate(self, derham: Derham, domain: Domain = None, equil: FluidEquilibrium = None,):
        self._spline = derham.create_spline_function(
                        name=self.__name__,
                        space_id=self.space,
                        backgrounds=self.backgrounds,
                        perturbations=self.perturbations,
                        domain=domain,
                        equil=equil,
                    )
    
    
class PICVariable(Variable):
    def __init__(self, name: str = "a_pic_var", space: str = "Particles6D"):
        assert space in ("Particles6D", "Particles5D", "Particles3D", "DeltaFParticles6D")
        self._name = name
        self._space = space
        self._kinetic_data = None
        
    @property
    def __name__(self):
        return self._name
        
    @property
    def space(self):
        return self._space
    
    @property
    def particles(self) -> Particles:
        return self._particles
    
    @property
    def kinetic_data(self):
        return self._kinetic_data
    
    def add_background(self, background: Maxwellian, verbose=True):
        super().add_background(background, verbose=verbose)
    
    def allocate(self, 
                 clone_config: CloneConfig = None,
                 derham: Derham = None, 
                 domain: Domain = None, 
                 equil: FluidEquilibrium = None,
                 projected_equil: ProjectedFluidEquilibrium = None,
                 ):
        
        assert isinstance(self.species, KineticSpecies)

        if derham is None:
            domain_decomp = None
        else:
            domain_array = derham.domain_array
            nprocs = derham.domain_decomposition.nprocs
            domain_decomp = (domain_array, nprocs)

        kinetic_class = getattr(particles, self.space)

        self._particles: Particles = kinetic_class(
            comm_world=MPI.COMM_WORLD,
            clone_config=clone_config,
            Np=self.species.Np,
            ppc=self.species.ppc,
            domain_decomp=domain_decomp,
            mpi_dims_mask=self.species.dims_mask,
            ppb=self.species.ppb,
            boxes_per_dim=self.species.boxes_per_dim,
            box_bufsize=self.species.box_bufsize,
            bc=self.species.bc,
            bc_refill=self.species.bc_refill,
            control_variate=self.species.control_variate,
            name=self.species.__class__.__name__,
            # bc_sph=self.species.bc_sph,
            loading=self.species.loading,
            loading_params=self.species.loading_params,
            weights_params=self.species.reject_weights,
            bufsize=self.species.bufsize,
            domain=domain,
            equil=equil,
            projected_equil=projected_equil,
            bckgr_params=self.backgrounds,
            pert_params=self.perturbations,
            equation_params=self.species.equation_params,
        )

        # for storing markers
        self._kinetic_data = {}

        # for storing the distribution function
        if "f" in val["params"]["save_data"]:
            slices = val["params"]["save_data"]["f"]["slices"]
            n_bins = val["params"]["save_data"]["f"]["n_bins"]
            ranges = val["params"]["save_data"]["f"]["ranges"]

            val["kinetic_data"]["f"] = {}
            val["kinetic_data"]["df"] = {}
            val["bin_edges"] = {}
            if len(slices) > 0:
                for i, sli in enumerate(slices):
                    assert ((len(sli) - 2) / 3).is_integer()
                    assert len(slices[i].split("_")) == len(ranges[i]) == len(n_bins[i]), (
                        f"Number of slices names ({len(slices[i].split('_'))}), number of bins ({len(n_bins[i])}), and number of ranges ({len(ranges[i])}) are inconsistent with each other!\n\n"
                    )
                    val["bin_edges"][sli] = []
                    dims = (len(sli) - 2) // 3 + 1
                    for j in range(dims):
                        val["bin_edges"][sli] += [
                            np.linspace(
                                ranges[i][j][0],
                                ranges[i][j][1],
                                n_bins[i][j] + 1,
                            ),
                        ]
                    val["kinetic_data"]["f"][sli] = np.zeros(
                        n_bins[i],
                        dtype=float,
                    )
                    val["kinetic_data"]["df"][sli] = np.zeros(
                        n_bins[i],
                        dtype=float,
                    )

        # for storing an sph evaluation of the density n
        if "n_sph" in val["params"]["save_data"]:
            plot_pts = val["params"]["save_data"]["n_sph"]["plot_pts"]

            val["kinetic_data"]["n_sph"] = []
            val["plot_pts"] = []
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
                val["plot_pts"] += [(ee1, ee2, ee3)]
                val["kinetic_data"]["n_sph"] += [np.zeros(ee1.shape, dtype=float)]

        # other data (wave-particle power exchange, etc.)
        # TODO
    
    
class SPHVariable(Variable):
    pass