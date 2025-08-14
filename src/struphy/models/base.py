import inspect
import operator
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Callable
import os
import yaml
import struphy

import yaml
from mpi4py import MPI

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import SplineFunction
from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB, MHDequilibrium
from struphy.fields_background.projected_equils import (
    ProjectedFluidEquilibrium,
    ProjectedFluidEquilibriumWithB,
    ProjectedMHDequilibrium,
)
from struphy.io.setup import setup_derham, descend_options_dict
from struphy.profiling.profiling import ProfileManager
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml, read_state
from struphy.models.species import Species, FieldSpecies, FluidSpecies, KineticSpecies, DiagnosticSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.io.options import Units, Time, DerhamOptions
from struphy.topology.grids import TensorProductGrid
from struphy.geometry.base import Domain
from struphy.geometry.domains import Cuboid
from struphy.fields_background.equils import HomogenSlab
from struphy.pic import particles
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.kinetic_background import maxwellians
from psydac.linalg.stencil import StencilVector
from struphy.io.output_handling import DataContainer


class StruphyModel(metaclass=ABCMeta):
    """
    Base class for all Struphy models.

    Note
    ----
    All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/``
    in one of the modules ``fluid.py``, ``kinetic.py``, ``hybrid.py`` or ``toy.py``.
    """

    ## abstract methods

    @abstractmethod
    class Propagators:
        pass
    
    @abstractmethod
    def __init__(self):
        """Light-weight init of model."""
    
    @property    
    @abstractmethod
    def bulk_species() -> Species:
        """Bulk species of the plasma. Must be an attribute of species_static()."""

    @property
    @abstractmethod
    def velocity_scale() -> str:
        """Velocity unit scale of the model.
        Must be one of "alfvén", "cyclotron", "light" or "thermal"."""

    @abstractmethod
    def allocate_helpers(self):
        """Allocate helper arrays that are needed during simulation."""

    @abstractmethod
    def update_scalar_quantities(self):
        """Specify an update rule for each item in ``scalar_quantities`` using :meth:`update_scalar`."""

    ## setup methods

    def setup_equation_params(self, units: Units, verbose=False):
        """Set euqation parameters for each fluid and kinetic species."""
        for _, species in self.fluid_species.items():
            assert isinstance(species, FluidSpecies)
            species.setup_equation_params(units=units, verbose=verbose)

        for _, species in self.kinetic_species.items():
            assert isinstance(species, KineticSpecies)
            species.setup_equation_params(units=units, verbose=verbose)
           
    def setup_domain_and_equil(self, domain: Domain, equil: FluidEquilibrium):
        """If a numerical equilibirum is used, the domain is taken from this equilibirum. """
        if equil is not None:
            self._equil = equil
            if "Numerical" in self.equil.__class__.__name__:
                self._domain = self.equil.domain
            else:
                self._domain = domain
                self._equil.domain = domain
        else:
            self._domain = domain
            self._equil = None      
            
        if MPI.COMM_WORLD.Get_rank() == 0 and self.verbose:
            print("\nDOMAIN:")
            print(f"type:".ljust(25), self.domain.__class__.__name__)
            for key, val in self.domain.params.items():
                if key not in {"cx", "cy", "cz"}:
                    print((key + ":").ljust(25), val)

            print("\nFLUID BACKGROUND:")
            if self.equil is not None:
                print("type:".ljust(25), self.equil.__class__.__name__)
                for key, val in self.equil.params.items():
                    print((key + ":").ljust(25), val)
            else:
                print("None.")
     
    ## species 
     
    @property
    def field_species(self) -> dict:
        if not hasattr(self, "_field_species"):
            self._field_species = {}
            for k, v in self.__dict__.items():
                if isinstance(v, FieldSpecies):
                    self._field_species[k] = v
        return self._field_species
    
    @property
    def fluid_species(self) -> dict:
        if not hasattr(self, "_fluid_species"):
            self._fluid_species = {}
            for k, v in self.__dict__.items():
                if isinstance(v, FluidSpecies):
                    self._fluid_species[k] = v
        return self._fluid_species
    
    @property
    def kinetic_species(self) -> dict:
        if not hasattr(self, "_kinetic_species"):
            self._kinetic_species = {}
            for k, v in self.__dict__.items():
                if isinstance(v, KineticSpecies):
                    self._kinetic_species[k] = v
        return self._kinetic_species
    
    @property
    def diagnostic_species(self) -> dict:
        if not hasattr(self, "_diagnostic_species"):
            self._diagnostic_species = {}
            for k, v in self.__dict__.items():
                if isinstance(v, DiagnosticSpecies):
                    self._diagnostic_species[k] = v
        return self._diagnostic_species
    
    @property
    def species(self):
        if not hasattr(self, "_species"):
            self._species = self.field_species | self.fluid_species | self.kinetic_species
        return self._species
    
    ## allocate methods
             
    def allocate_feec(self,
                 grid: TensorProductGrid,
                 derham_opts: DerhamOptions,
                 comm: MPI.Intracomm = None,
                 clone_config: CloneConfig = None,
                 ):

        # create discrete derham sequence
        if clone_config is None:
            derham_comm = MPI.COMM_WORLD
        else:
            derham_comm = clone_config.sub_comm

        self._derham = setup_derham(
            grid,
            derham_opts,
            comm=derham_comm,
            domain=self.domain,
            verbose=self.verbose,
        )
        
        # create weighted mass operators
        self._mass_ops = WeightedMassOperators(
            self.derham,
            self.domain,
            verbose=self.verbose,
            eq_mhd=self.equil,
        )
        
        # create projected equilibrium
        if isinstance(self.equil, MHDequilibrium):
            self._projected_equil = ProjectedMHDequilibrium(
                self.equil,
                self.derham,
            )
        elif isinstance(self.equil, FluidEquilibriumWithB):
            self._projected_equil = ProjectedFluidEquilibriumWithB(
                self.equil,
                self.derham,
            )
        elif isinstance(self.equil, FluidEquilibrium):
            self._projected_equil = ProjectedFluidEquilibrium(
                self.equil,
                self.derham,
            )  
        else:
            self._projected_equil = None
            
    def allocate_propagators(self):
        # set propagators base class attributes (then available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        if self.derham is not None:
            Propagator.mass_ops = self.mass_ops
            Propagator.basis_ops = BasisProjectionOperators(
                self.derham,
                self.domain,
                verbose=self.verbose,
                eq_mhd=self.equil,
            )
            Propagator.projected_equil = self.projected_equil
            
        assert len(self.prop_list) > 0, "No propagators in this model, check the model class."
        for prop in self.prop_list:
            assert isinstance(prop, Propagator)
            prop.allocate()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"\nAllocated propagator '{prop.__class__.__name__}'.")

    @staticmethod
    def diagnostics_dct():
        """Diagnostics dictionary.
        Model specific variables (FemField) which is going to be saved during the simulation.
        """

    ## basic properties

    @property
    def params(self):
        """Model parameters from :code:`parameters.yml`."""
        return self._params

    @property
    def pparams(self):
        """Plasma parameters for each species."""
        return self._pparams

    @property
    def equation_params(self):
        """Parameters appearing in model equation due to Struphy normalization."""
        return self._equation_params

    @property
    def clone_config(self):
        """Config in case domain clones are used."""
        return self._clone_config
    
    @clone_config.setter
    def clone_config(self, new):
        assert isinstance(new, CloneConfig) or new is None
        self._clone_config = new

    @property
    def diagnostics(self):
        """Dictionary of diagnostics."""
        return self._diagnostics

    @property
    def domain(self):
        """Domain object, see :ref:`avail_mappings`."""
        return self._domain

    @property
    def equil(self):
        """Fluid equilibrium object, see :ref:`fluid_equil`."""
        return self._equil

    @property
    def derham(self):
        """3d Derham sequence, see :ref:`derham`."""
        return self._derham

    @property
    def projected_equil(self):
        """Fluid equilibrium projected on 3d Derham sequence with commuting projectors."""
        return self._projected_equil

    @property
    def units(self) -> Units:
        """All Struphy units."""
        return self._units
    
    @units.setter
    def units(self, new) :
        assert isinstance(new, Units)
        self._units = new

    @property
    def mass_ops(self):
        """WeighteMassOperators object, see :ref:`mass_ops`."""
        return self._mass_ops
    
    @property
    def prop_list(self):
        """List of Propagator objects."""
        if not hasattr(self, "_prop_list"):
            self._prop_list = list(self.propagators.__dict__.values())
        return self._prop_list

    @property
    def prop_fields(self):
        """Module :mod:`struphy.propagators.propagators_fields`."""
        return self._prop_fields

    @property
    def prop_coupling(self):
        """Module :mod:`struphy.propagators.propagators_coupling`."""
        return self._prop_coupling

    @property
    def prop_markers(self):
        """Module :mod:`struphy.propagators.propagators_markers`."""
        return self._prop_markers

    @property
    def kwargs(self):
        """Dictionary holding the keyword arguments for each propagator specified in :attr:`~propagators_cls`.
        Keys must be the same as in :attr:`~propagators_cls`, values are dictionaries holding the keyword arguments."""
        return self._kwargs

    @property
    def scalar_quantities(self):
        """A dictionary of scalar quantities to be saved during the simulation."""
        return self._scalar_quantities

    @property
    def time_state(self):
        """A pointer to the time variable of the dynamics ('t')."""
        return self._time_state

    @property
    def verbose(self):
        """Bool: show model info on screen."""
        try:
            return self._verbose
        except:
            return False

    @verbose.setter
    def verbose(self, new):
        assert isinstance(new, bool)
        self._verbose = new

    @classmethod
    def options(cls):
        """Dictionary for available species options of the form {'em_fields': {}, 'fluid': {}, 'kinetic': {}}."""
        dct = {}

        for prop, vars in cls.propagators_dct().items():
            var = vars[0]
            if var in cls.species()["em_fields"]:
                species = "em_fields"
            elif var in cls.species()["kinetic"]:
                species = ["kinetic", var]
            else:
                spl = var.split("_")
                var_stem = spl[0]
                for el in spl[1:-1]:
                    var_stem += "_" + el
                species = ["fluid", var_stem]

            cls.add_option(
                species=species,
                option=prop,
                dct=dct,
            )

        return dct

    @classmethod
    def add_option(
        cls,
        species: str | list,
        option,
        dct: dict,
        *,
        key=None,
    ):
        """Add an option to the dictionary of parameters under [species][options].

        Test with "struphy params MODEL".

        Parameters
        ----------
        species : str or list
            path in the dict before the 'options' key

        option : any
            value which should be added in the dict

        dct : dict
            dictionary to which the value should be added at the corresponding position

        key : str or list
            path in the dict after the 'options' key
        """

        def getFromDict(dataDict, mapList):
            return reduce(operator.getitem, mapList, dataDict)

        def setInDict(dataDict, mapList, value):
            # Loop over dicitionary and creaty empty dicts where the path does not exist
            for k in range(len(mapList)):
                if not mapList[k] in getFromDict(dataDict, mapList[:k]).keys():
                    getFromDict(dataDict, mapList[:k])[mapList[k]] = {}
            getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

        # make sure that the base keys are top-level keys
        for base_key in ["em_fields", "fluid", "kinetic"]:
            if not base_key in dct.keys():
                dct[base_key] = {}

        if isinstance(species, str):
            species = [species]
        if isinstance(key, str):
            key = [key]

        if inspect.isclass(option):
            setInDict(
                dct,
                species + ["options"] + [option.__name__],
                option.options(),
            )
        else:
            assert key is not None, "Must provide key if option is not a class."
            setInDict(dct, species + ["options"] + key, option)

    def add_scalar(self, 
                   name: str, variable: PICVariable | SPHVariable = None, compute=None, summands=None):
        """
        Add a scalar to be saved during the simulation.

        Parameters
        ----------
        name : str
            Dictionary key for the scalar.
        variable : PICVariable | SPHVariable, optional
            The variable associated with the scalar. Required if compute is 'from_particles'.
        compute : str, optional
            Type of scalar, determines the compute operations.
            Options: 'from_particles' or 'from_field'. Default is None.
        summands : list, optional
            List of other scalar names whose values should be summed
            to compute the value of this scalar. Default is None.
        """

        assert isinstance(name, str), "name must be a string"
        if compute == "from_particles":
            assert isinstance(variable, (PICVariable, SPHVariable)), f"Variable is needed when {compute = }"

        if not hasattr(self, "_scalar_quantities"):
            self._scalar_quantities = {}

        self._scalar_quantities[name] = {
            "value": np.empty(1, dtype=float),
            "variable": variable,
            "compute": compute,
            "summands": summands,
        }

    def update_scalar(self, name, value=None):
        """Add a scalar that should be saved during the simulation.

        Parameters
        ----------
            name : str
                Dictionary key of the scalar.

            value : float, optional
                Value to be saved. Required if there are no summands.
        """

        # Ensure the name is a string
        assert isinstance(name, str)

        variable: PICVariable | SPHVariable = self._scalar_quantities[name]["variable"]
        summands = self._scalar_quantities[name]["summands"]
        compute = self._scalar_quantities[name]["compute"]

        if compute == "from_particles":
            compute_operations = [
                "sum_within_clone",
                "sum_between_clones",
                "divide_n_mks",
            ]
        elif compute == "from_sph":
            compute_operations = [
                "sum_world",
                "divide_n_mks",
            ]
        elif compute == "from_field":
            compute_operations = []
        else:
            compute_operations = []

        if summands is None:
            # Ensure the value is a float if there are no summands
            assert isinstance(value, float)

            # Create a numpy array to hold the scalar value
            value_array = np.array([value], dtype=np.float64)

            # Perform MPI operations based on the compute flags
            if "sum_world" in compute_operations and self.comm_world is not None:
                self.comm_world.Allreduce(
                    MPI.IN_PLACE,
                    value_array,
                    op=MPI.SUM,
                )

            if "sum_within_clone" in compute_operations and self.derham.comm is not None:
                self.derham.comm.Allreduce(
                    MPI.IN_PLACE,
                    value_array,
                    op=MPI.SUM,
                )
            if self.clone_config is None:
                num_clones = 1
            else:
                num_clones = self.clone_config.num_clones

            if "sum_between_clones" in compute_operations and num_clones > 1:
                self.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    value_array,
                    op=MPI.SUM,
                )

            if "average_between_clones" in compute_operations and num_clones > 1:
                self.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    value_array,
                    op=MPI.SUM,
                )
                value_array /= num_clones

            if "divide_n_mks" in compute_operations:
                # Initialize the total number of markers
                n_mks_tot = np.array([variable.particles.Np])
                value_array /= n_mks_tot

            # Update the scalar value
            self._scalar_quantities[name]["value"][0] = value_array[0]

        else:
            # Sum the values of the summands
            value = sum(self._scalar_quantities[summand]["value"][0] for summand in summands)
            self._scalar_quantities[name]["value"][0] = value

    def add_time_state(self, time_state):
        """Add a pointer to the time variable of the dynamics ('t')
        to the model and to all propagators of the model.

        Parameters
        ----------
        time_state : ndarray
            Of size 1, holds the current physical time 't'.
        """
        assert time_state.size == 1
        self._time_state = time_state
        for _, prop in self.propagators.__dict__.items():
            if isinstance(prop, Propagator):
                prop.add_time_state(time_state)

    def allocate_variables(self):
        """
        Allocate memory for model variables and set initial conditions.
        """
        # allocate memory for FE coeffs of electromagnetic fields/potentials
        if self.field_species:
            for species, spec in self.field_species.items():
                assert isinstance(spec, FieldSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, FEECVariable)
                    v.allocate(derham=self.derham, domain=self.domain, equil=self.equil,)

        # allocate memory for FE coeffs of fluid variables
        if self.fluid_species:
            for species, spec in self.fluid_species.items():
                assert isinstance(spec, FluidSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, FEECVariable)
                    v.allocate(derham=self.derham, domain=self.domain, equil=self.equil,)
                    
        # allocate memory for marker arrays of kinetic variables
        if self.kinetic_species:
            for species, spec in self.kinetic_species.items():
                assert isinstance(spec, KineticSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, (PICVariable, SPHVariable))
                    v.allocate(derham=self.derham, domain=self.domain, equil=self.equil,)
                
        # TODO: allocate memory for FE coeffs of diagnostics
        # if self.params.diagnostic_fields is not None:
        #     for key, val in self.diagnostics.items():
        #         if "params" in key:
        #             continue
        #         else:
        #             val["obj"] = self.derham.create_spline_function(
        #                 key,
        #                 val["space"],
        #                 bckgr_params=None,
        #                 pert_params=None,
        #             )

        #             self._pointer[key] = val["obj"].vector


    def integrate(self, dt, split_algo="LieTrotter"):
        """
        Advance the model by a time step ``dt`` by sequentially calling its Propagators.

        Parameters
        ----------
        dt : float
            Time step of time integration.

        split_algo : str
            Splitting algorithm. Currently available: "LieTrotter" and "Strang".
        """

        # first order in time
        if split_algo == "LieTrotter":
            for propagator in self.prop_list:
                prop_name = propagator.__class__.__name__

                with ProfileManager.profile_region(prop_name):
                    propagator(dt)

        # second order in time
        elif split_algo == "Strang":
            assert len(self.propagators) > 1

            for propagator in self.prop_list[:-1]:
                prop_name = type(propagator).__name__
                with ProfileManager.profile_region(prop_name):
                    propagator(dt / 2)

            propagator = self.prop_list[-1]
            prop_name = type(propagator).__name__
            with ProfileManager.profile_region(prop_name):
                propagator(dt)

            for propagator in self.prop_list[:-1][::-1]:
                prop_name = type(propagator).__name__
                with ProfileManager.profile_region(prop_name):
                    propagator(dt / 2)

        else:
            raise NotImplementedError(
                f"Splitting scheme {split_algo} not available.",
            )

    def update_markers_to_be_saved(self):
        """
        Writes markers with IDs that are supposed to be saved into corresponding array.
        """

        for name, species in self.kinetic_species.items():
            assert isinstance(species, KineticSpecies)
            assert len(species.variables) == 1, f"More than 1 variable per kinetic species is not allowed."
            for _, var in species.variables.items():
                assert isinstance(var, PICVariable | SPHVariable)
                obj = var.particles
                assert isinstance(obj, Particles)

            # allocate array for saving markers if not present
            if not hasattr(self, "_n_markers_saved"):
                n_markers = species.n_markers

                if isinstance(n_markers, float):
                    if n_markers > 1.0:
                        self._n_markers_saved = int(n_markers)
                    else:
                        self._n_markers_saved = int(obj.n_mks_global * n_markers)
                else:
                    self._n_markers_saved = n_markers

                assert self._n_markers_saved <= obj.Np, (
                    f"The number of markers for which data should be stored (={self._n_markers_saved}) murst be <= than the total number of markers (={obj.Np})"
                )
                if self._n_markers_saved > 0:
                    var.kinetic_data["markers"] = np.zeros(
                        (self._n_markers_saved, obj.markers.shape[1]),
                        dtype=float,
                    )

            if self._n_markers_saved > 0:
                markers_on_proc = np.logical_and(
                    obj.markers[:, -1] >= 0.0,
                    obj.markers[:, -1] < self._n_markers_saved,
                )
                n_markers_on_proc = np.count_nonzero(markers_on_proc)
                var.kinetic_data["markers"][:] = -1.0
                var.kinetic_data["markers"][:n_markers_on_proc] = obj.markers[markers_on_proc]

    def update_distr_functions(self):
        """
        Writes distribution functions slices that are supposed to be saved into corresponding array.
        """

        dim_to_int = {"e1": 0, "e2": 1, "e3": 2, "v1": 3, "v2": 4, "v3": 5}

        for name, species in self.kinetic_species.items():
            assert isinstance(species, KineticSpecies)
            assert len(species.variables) == 1, f"More than 1 variable per kinetic species is not allowed."
            for _, var in species.variables.items():
                assert isinstance(var, PICVariable | SPHVariable)
                obj = var.particles
                assert isinstance(obj, Particles)

            if obj.n_cols_diagnostics > 0:
                for i in range(obj.n_cols_diagnostics):
                    str_dn = f"d{i + 1}"
                    dim_to_int[str_dn] = 3 + obj.vdim + 3 + i

            if species.f_binned is not None:
                for slice_i, edges in var.kinetic_data["bin_edges"].items():
                    comps = slice_i.split("_")
                    components = [False] * (3 + obj.vdim + 3 + obj.n_cols_diagnostics)

                    for comp in comps:
                        components[dim_to_int[comp]] = True

                    f_slice, df_slice = obj.binning(components, edges)

                    var.kinetic_data["f"][slice_i][:] = f_slice
                    var.kinetic_data["df"][slice_i][:] = df_slice

            if species.n_sph is not None:
                h1 = 1 / obj.boxes_per_dim[0]
                h2 = 1 / obj.boxes_per_dim[1]
                h3 = 1 / obj.boxes_per_dim[2]

                ndim = np.count_nonzero([d > 1 for d in obj.boxes_per_dim])
                if ndim == 0:
                    kernel_type = "gaussian_3d"
                else:
                    kernel_type = "gaussian_" + str(ndim) + "d"

                for i, pts in enumerate(val["plot_pts"]):
                    n_sph = obj.eval_density(
                        *pts,
                        h1=h1,
                        h2=h2,
                        h3=h3,
                        kernel_type=kernel_type,
                        fast=True,
                    )
                    val["kinetic_data"]["n_sph"][i][:] = n_sph

    def print_scalar_quantities(self):
        """
        Check if scalar_quantities are not "nan" and print to screen.
        """
        sq_str = ""
        for key, scalar_dict in self._scalar_quantities.items():
            val = scalar_dict["value"]
            assert not np.isnan(val[0]), f"Scalar {key} is {val[0]}."
            sq_str += key + ": {:14.11f}".format(val[0]) + "   "
        print(sq_str)

    # def initialize_from_params(self):
    #     """
    #     Set initial conditions for FE coefficients (electromagnetic and fluid)
    #     and markers according to parameter file.
    #     """

    #     # initialize em fields
    #     if self.field_species:
    #         with ProfileManager.profile_region("initialize_em_fields"):
    #             for key, val in self.em_fields.items():
    #                 if "params" in key:
    #                     continue
    #                 else:
    #                     obj = val["obj"]
    #                     assert isinstance(obj, SplineFunction)

    #                     obj.initialize_coeffs(
    #                         domain=self.domain,
    #                         bckgr_obj=self.equil,
    #                     )

    #                     if self.rank_world == 0 and self.verbose:
    #                         print(f'\nEM field "{key}" was initialized with:')

    #                         _params = self.em_fields["params"]

    #                         if "background" in _params:
    #                             if key in _params["background"]:
    #                                 bckgr_types = _params["background"][key]
    #                                 if bckgr_types is None:
    #                                     pass
    #                                 else:
    #                                     print("background:")
    #                                     for _type, _bp in bckgr_types.items():
    #                                         print(" " * 4 + _type, ":")
    #                                         for _pname, _pval in _bp.items():
    #                                             print((" " * 8 + _pname + ":").ljust(25), _pval)
    #                             else:
    #                                 print("No background.")
    #                         else:
    #                             print("No background.")

    #                         if "perturbation" in _params:
    #                             if key in _params["perturbation"]:
    #                                 pert_types = _params["perturbation"][key]
    #                                 if pert_types is None:
    #                                     pass
    #                                 else:
    #                                     print("perturbation:")
    #                                     for _type, _pp in pert_types.items():
    #                                         print(" " * 4 + _type, ":")
    #                                         for _pname, _pval in _pp.items():
    #                                             print((" " * 8 + _pname + ":").ljust(25), _pval)
    #                             else:
    #                                 print("No perturbation.")
    #                         else:
    #                             print("No perturbation.")

    #     if len(self.fluid) > 0:
    #         with ProfileManager.profile_region("initialize_fluids"):
    #             for species, val in self.fluid.items():
    #                 for variable, subval in val.items():
    #                     if "params" in variable:
    #                         continue
    #                     else:
    #                         obj = subval["obj"]
    #                         assert isinstance(obj, SplineFunction)
    #                         obj.initialize_coeffs(
    #                             domain=self.domain,
    #                             bckgr_obj=self.equil,
    #                             species=species,
    #                         )

    #                 if self.rank_world == 0 and self.verbose:
    #                     print(
    #                         f'\nFluid species "{species}" was initialized with:',
    #                     )

    #                     _params = val["params"]

    #                     if "background" in _params:
    #                         for variable in val:
    #                             if "params" in variable:
    #                                 continue
    #                             if variable in _params["background"]:
    #                                 bckgr_types = _params["background"][variable]
    #                                 if bckgr_types is None:
    #                                     pass
    #                                 else:
    #                                     print(f"{variable} background:")
    #                                     for _type, _bp in bckgr_types.items():
    #                                         print(" " * 4 + _type, ":")
    #                                         for _pname, _pval in _bp.items():
    #                                             print((" " * 8 + _pname + ":").ljust(25), _pval)
    #                             else:
    #                                 print(f"{variable}: no background.")
    #                     else:
    #                         print("No background.")

    #                     if "perturbation" in _params:
    #                         for variable in val:
    #                             if "params" in variable:
    #                                 continue
    #                             if variable in _params["perturbation"]:
    #                                 pert_types = _params["perturbation"][variable]
    #                                 if pert_types is None:
    #                                     pass
    #                                 else:
    #                                     print(f"{variable} perturbation:")
    #                                     for _type, _pp in pert_types.items():
    #                                         print(" " * 4 + _type, ":")
    #                                         for _pname, _pval in _pp.items():
    #                                             print((" " * 8 + _pname + ":").ljust(25), _pval)
    #                             else:
    #                                 print(f"{variable}: no perturbation.")
    #                     else:
    #                         print("No perturbation.")

    #     # initialize particles
    #     if len(self.kinetic) > 0:
    #         with ProfileManager.profile_region("initialize_particles"):
    #             for species, val in self.kinetic.items():
    #                 obj = val["obj"]
    #                 assert isinstance(obj, Particles)

    #                 if self.rank_world == 0 and self.verbose:
    #                     _params = val["params"]
    #                     assert "background" in _params, "Kinetic species must have background."

    #                     bckgr_types = _params["background"]
    #                     print(
    #                         f'\nKinetic species "{species}" was initialized with:',
    #                     )
    #                     for _type, _bp in bckgr_types.items():
    #                         print(_type, ":")
    #                         for _pname, _pval in _bp.items():
    #                             print((" " * 4 + _pname + ":").ljust(25), _pval)

    #                     if "perturbation" in _params:
    #                         for variable, pert_types in _params["perturbation"].items():
    #                             if pert_types is None:
    #                                 pass
    #                             else:
    #                                 print(f"{variable} perturbation:")
    #                                 for _type, _pp in pert_types.items():
    #                                     print(" " * 4 + _type, ":")
    #                                     for _pname, _pval in _pp.items():
    #                                         print((" " * 8 + _pname + ":").ljust(25), _pval)
    #                     else:
    #                         print("No perturbation.")

    #                 obj.draw_markers(sort=True, verbose=self.verbose)
    #                 obj.mpi_sort_markers(do_test=True)

    #                 if not val["params"]["markers"]["loading"] == "restart":
    #                     if obj.coords == "vpara_mu":
    #                         obj.save_magnetic_moment()

    #                     if val["space"] != "ParticlesSPH" and obj.f0.coords == "constants_of_motion":
    #                         obj.save_constants_of_motion()

    #                     obj.initialize_weights(
    #                         reject_weights=obj.weights_params["reject_weights"],
    #                         threshold=obj.weights_params["threshold"],
    #                     )

    def initialize_from_restart(self, data):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers from restart group in hdf5 files.

        Parameters
        ----------
        data : struphy.io.output_handling.DataContainer
            The data object that links to the hdf5 files.
        """

        # initialize em fields
        if len(self.em_fields) > 0:
            for key, val in self.em_fields.items():
                if "params" in key:
                    continue
                else:
                    obj = val["obj"]
                    assert isinstance(obj, SplineFunction)
                    obj.initialize_coeffs_from_restart_file(data.file)

        # initialize fields
        if len(self.fluid) > 0:
            for species, val in self.fluid.items():
                for variable, subval in val.items():
                    if "params" in variable:
                        continue
                    else:
                        obj = subval["obj"]
                        assert isinstance(obj, SplineFunction)
                        obj.initialize_coeffs_from_restart_file(
                            data.file,
                            species,
                        )

        # initialize particles
        if len(self.kinetic) > 0:
            for key, val in self.kinetic.items():
                obj = val["obj"]
                assert isinstance(obj, Particles)
                obj.draw_markers(verbose=self.verbose)
                obj._markers[:, :] = data.file["restart/" + key][-1, :, :]

                # important: sets holes attribute of markers!
                if self.comm_world is not None:
                    obj.mpi_sort_markers(do_test=True)

    def initialize_data_output(self, data: DataContainer, size):
        """
        Create datasets in hdf5 files according to model unknowns and diagnostics data.

        Parameters
        ----------
        data : struphy.io.output_handling.DataContainer
            The data object that links to the hdf5 files.

        size : int
            Number of MPI processes of the model run.

        Returns
        -------
        save_keys_all : list
            Keys of datasets which are saved during the simulation.

        save_keys_end : list
            Keys of datasets which are saved at the end of a simulation to enable restarts.
        """

        # save scalar quantities in group 'scalar/'
        for key, scalar in self.scalar_quantities.items():
            val = scalar["value"]
            key_scalar = "scalar/" + key
            data.add_data({key_scalar: val})

        # store grid_info only for runs with 512 ranks or smaller
        if self._scalar_quantities and self.derham is not None:
            if size <= 512:
                data.file["scalar"].attrs["grid_info"] = self.derham.domain_array
            else:
                data.file["scalar"].attrs["grid_info"] = self.derham.domain_array[0]
        else:
            pass

        # save feec data in group 'feec/'
        feec_species = self.field_species | self.fluid_species | self.diagnostic_species
        for species, val in feec_species.items():
            assert isinstance(val, Species)
            
            species_path = os.path.join("feec", species)
            species_path_restart = os.path.join("restart", species)

            for variable, subval in val.variables.items():
                assert isinstance(subval, FEECVariable)
                spline = subval.spline

                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                spline.extract_coeffs(update_ghost_regions=False)

                # save numpy array to be updated each time step.
                if subval.save_data:
                    key_field = os.path.join(species_path, variable)

                    if isinstance(spline.vector_stencil, StencilVector):
                        data.add_data(
                            {key_field: spline.vector_stencil._data},
                        )

                    else:
                        for n in range(3):
                            key_component = os.path.join(key_field, str(n + 1))
                            data.add_data(
                                {key_component: spline.vector_stencil[n]._data},
                            )

                    # save field meta data
                    data.file[key_field].attrs["space_id"] = spline.space_id
                    data.file[key_field].attrs["starts"] = spline.starts
                    data.file[key_field].attrs["ends"] = spline.ends
                    data.file[key_field].attrs["pads"] = spline.pads

                # save numpy array to be updated only at the end of the simulation for restart.
                key_field_restart = os.path.join(species_path_restart, variable)

                if isinstance(spline.vector_stencil, StencilVector):
                    data.add_data(
                        {key_field_restart: spline.vector_stencil._data},
                    )
                else:
                    for n in range(3):
                        key_component_restart = os.path.join(key_field_restart, str(n + 1))
                        data.add_data(
                            {key_component_restart: spline.vector_stencil[n]._data},
                        )

        # save kinetic data in group 'kinetic/'
        for name, species in self.kinetic_species.items():
            assert isinstance(species, KineticSpecies)
            assert len(species.variables) == 1, f"More than 1 variable per kinetic species is not allowed."
            for _, var in species.variables.items():
                assert isinstance(var, PICVariable | SPHVariable)
                obj = var.particles
                assert isinstance(obj, Particles)

            key_spec = "kinetic/" + key
            key_spec_restart = "restart/" + key

            data.add_data({key_spec_restart: obj._markers})

            for key1, val1 in var.kinetic_data.items():
                key_dat = key_spec + "/" + key1

                # case of "f" and "df"
                if isinstance(val1, dict):
                    for key2, val2 in val1.items():
                        key_f = key_dat + "/" + key2
                        data.add_data({key_f: val2})

                        dims = (len(key2) - 2) // 3 + 1
                        for dim in range(dims):
                            data.file[key_f].attrs["bin_centers" + "_" + str(dim + 1)] = (
                                var.kinetic_data["bin_edges"][key2][dim][:-1]
                                + (var.kinetic_data["bin_edges"][key2][dim][1] - var.kinetic_data["bin_edges"][key2][dim][0]) / 2
                            )
                # case of "n_sph"
                elif isinstance(val1, list):
                    for i, v1 in enumerate(val1):
                        key_n = key_dat + "/view_" + str(i)
                        data.add_data({key_n: v1})
                        # save 1d point values, not meshgrids, because attrs size is limited
                        eta1 = var.kinetic_data["plot_pts"][i][0][:, 0, 0]
                        eta2 = var.kinetic_data["plot_pts"][i][1][0, :, 0]
                        eta3 = var.kinetic_data["plot_pts"][i][2][0, 0, :]
                        data.file[key_n].attrs["eta1"] = eta1
                        data.file[key_n].attrs["eta2"] = eta2
                        data.file[key_n].attrs["eta3"] = eta3
                else:
                    data.add_data({key_dat: val1})

        # keys to be saved at each time step and only at end (restart)
        save_keys_all = []
        save_keys_end = []

        for key in data.dset_dict:
            if "restart" in key:
                save_keys_end.append(key)
            else:
                save_keys_all.append(key)

        return save_keys_all, save_keys_end

    ###################
    # Class methods :
    ###################

    @classmethod
    def show_options(cls):
        """Print available model options to screen."""

        print(
            'Options are given under the keyword "options" for each species dict. \
Available options stand in lists as dict values.\nThe first entry of a list denotes the default value.'
        )

        tab = "    "

        print(f'\nAvailable options for model "{cls.__name__}":')
        print("\nem_fields:")
        if "options" in cls.options()["em_fields"]:
            print(tab + "options:")
            for opt_k, opt_v in cls.options()["em_fields"]["options"].items():
                if isinstance(opt_v, dict):
                    print((2 * tab + opt_k + " :").ljust(25))
                    for key, val in opt_v.items():
                        print((3 * tab + key + " :").ljust(25), val)
                else:
                    print((2 * tab + opt_k + " :").ljust(25), opt_v)
        else:
            print("None.")

        print("\nfluid:")
        if len(cls.species()["fluid"]) > 0:
            for spec_name in cls.species()["fluid"]:
                print(tab + spec_name + ":")
                print(2 * tab + "options:")
                if "options" in cls.options()["fluid"][spec_name]:
                    for opt_k, opt_v in cls.options()["fluid"][spec_name]["options"].items():
                        if isinstance(opt_v, dict):
                            print((3 * tab + opt_k + " :").ljust(25))
                            for key, val in opt_v.items():
                                print((4 * tab + key + " :").ljust(25), val)
                        else:
                            print((3 * tab + opt_k + " :").ljust(25), opt_v)
                else:
                    print("None.")
        else:
            print("None.")

        print("\nkinetic:")
        if len(cls.species()["kinetic"]) > 0:
            for spec_name in cls.species()["kinetic"]:
                print(tab + spec_name + ":")
                print(2 * tab + "options:")
                if "options" in cls.options()["kinetic"][spec_name]:
                    for opt_k, opt_v in cls.options()["kinetic"][spec_name]["options"].items():
                        if isinstance(opt_v, dict):
                            print((3 * tab + opt_k + " :").ljust(25))
                            for key, val in opt_v.items():
                                print((4 * tab + key + " :").ljust(25), val)
                        else:
                            print((3 * tab + opt_k + " :").ljust(25), opt_v)
                else:
                    print("None.")
        else:
            print("None.")

    @classmethod
    def write_parameters_to_file(cls, parameters=None, file=None, save=True, prompt=True):
        import os

        import yaml

        import struphy
        import struphy.utils.utils as utils

        # Read struphy state file
        state = utils.read_state()

        i_path = state["i_path"]
        assert os.path.exists(i_path), f"The path '{i_path}' does not exist. Set path with `struphy --set-i PATH`"

        if file is None:
            file = os.path.join(i_path, "params_" + cls.__name__ + ".yml")
        else:
            assert ".yml" in file or ".yaml" in file, "File must have a a .yml (.yaml) extension."
            file = os.path.join(i_path, file)

        if save:
            if not prompt:
                yn = "Y"
            else:
                yn = input(f"Writing to {file}, are you sure (Y/n)? ")

            if yn in ("", "Y", "y", "yes", "Yes"):
                dict_to_yaml(parameters, file)
                print(
                    f'Default parameter file for {cls.__name__} has been created; you can now launch with "struphy run {cls.__name__}".',
                )
            else:
                pass

    def generate_default_parameter_file(
        self,
        path: str = None,
        prompt: bool = True,
    ):
        """Generate a parameter file with default options for each species,
        and save it to the current input path.

        The default name is params_<model_name>.yml.

        Parameters
        ----------
        path : str
            Alternative path to getcwd()/params_MODEL.py.

        prompt : bool
            Whether to prompt for overwriting the specified .yml file.
            
        Returns
        -------
        params_path : str
            The path of the parameter file.
        """
        
        if path is None:
            path = os.path.join(os.getcwd(), f"params_{self.__class__.__name__}.py")

        # create new default file
        try:
            file =  open(path, "x")
        except FileExistsError:
            if not prompt:
                yn = "Y"
            else:
                yn = input(f"\nFile {path} exists, overwrite (Y/n)? ")
            if yn in ("", "Y", "y", "yes", "Yes"):
                file = open(path, "w")
            else:
                print("exiting ...")
                return
        except FileNotFoundError:
            folder = os.path.join("/", *path.split("/")[:-1])
            if not prompt:
                yn = "Y"
            else:
                yn = input(f"\nFolder {folder} does not exist, create (Y/n)? ")
            if yn in ("", "Y", "y", "yes", "Yes"):
                os.makedirs(folder)
                file = open(path, "x")
            else:
                print("exiting ...")
                return   
                    
        file.write("from struphy.io.options import EnvironmentOptions, Units, Time\n")
        file.write("from struphy.geometry import domains\n")
        file.write("from struphy.fields_background import equils\n")
        
        species_params = "\n# species parameters\n"
        kinetic_params = ""    
        has_plasma = False
        has_feec = False
        has_pic = False
        has_sph = False
        for sn, species in self.species.items():
            assert isinstance(species, Species)
            
            if isinstance(species, (FluidSpecies, KineticSpecies)):
                has_plasma = True
                species_params += f"model.{sn}.set_phys_params()\n"
                if isinstance(species, KineticSpecies):
                    kinetic_params += f"model.{sn}.set_markers()\n\
model.{sn}.set_sorting()\n\
model.{sn}.set_save_data()\n"
            
            for vn, var in species.variables.items():
                if isinstance(var, FEECVariable):
                    has_feec = True
                    if var.space in ("H1", "L2"):
                        init_bckgr_feec = f"model.{sn}.{vn}.add_background(FieldsBackground())\n"
                        init_pert_feec = f"model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos())\n"
                    else:
                        init_bckgr_feec = f"model.{sn}.{vn}.add_background(FieldsBackground())\n"
                        init_pert_feec = f"model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=0))\n\
model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=1))\n\
model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=2))\n"
                        
                    exclude = f"# model.{sn}.{vn}.save_data = False\n"
                elif isinstance(var, PICVariable):
                    has_pic = True
                    init_pert_pic = f"perturbation = perturbations.TorusModesCos()\n"
                    init_bckgr_pic = f"model.{sn}.{vn}.add_background(maxwellians.Maxwellian3D(perturbation=perturbation))\n"
                    
                    exclude = f"# model.....save_data = False\n"
                elif isinstance(var, SPHVariable):
                    has_sph = True
        
        file.write("from struphy.topology import grids\n") 
        file.write("from struphy.io.options import DerhamOptions\n")
        file.write("from struphy.io.options import FieldsBackground\n")
        file.write("from struphy.initial import perturbations\n")
        
        file.write("from struphy.kinetic_background import maxwellians\n")
        file.write("from struphy import main\n")
            
        file.write("\n# import model, set verbosity\n")
        file.write(f"from {self.__module__} import {self.__class__.__name__} as Model\n")
        file.write("verbose = True\n")
        
        file.write("\n# environment options\n")
        file.write("env = EnvironmentOptions()\n")
        
        file.write("\n# units\n")
        file.write("units = Units()\n")
        
        file.write("\n# time stepping\n")
        file.write("time_opts = Time()\n")
        
        file.write("\n# geometry\n")
        file.write("domain = domains.Cuboid()\n")
        
        file.write("\n# fluid equilibrium (can be used as part of initial conditions)\n")
        file.write("equil = equils.HomogenSlab()\n")
        
        if has_feec:
            grid = "grid = grids.TensorProductGrid()\n"
            derham = "derham_opts = DerhamOptions()\n"
        else:
            grid = "grid = None\n"
            derham = "derham_opts = None\n"
            
        file.write("\n# grid\n")
        file.write(grid)
            
        file.write("\n# derham options\n")
        file.write(derham)
        
        file.write("\n# light-weight model instance\n")
        file.write("model = Model()\n")
        
        if has_plasma:
            file.write(species_params)
            
        if has_pic:
            file.write(kinetic_params)
            
        file.write("\n# propagator options\n")
        for prop in self.propagators.__dict__:
            file.write(f"model.propagators.{prop}.set_options()\n")
            
        file.write("\n# initial conditions (background + perturbation)\n")
        if has_feec:
            file.write(init_bckgr_feec)
            file.write(init_pert_feec)
        if has_pic:
            file.write(init_pert_pic)
            file.write(init_bckgr_pic)
            
        file.write("\n# optional: exclude variables from saving\n")
        file.write(exclude)
        
        file.write('\nif __name__ == "__main__":\n')
        file.write("    # start run\n")
        file.write("    main.run(model, \n\
                params_path=__file__, \n\
                env=env, \n\
                units=units, \n\
                time_opts=time_opts, \n\
                domain=domain, \n\
                equil=equil, \n\
                grid=grid, \n\
                derham_opts=derham_opts, \n\
                verbose=verbose, \n\
                )")
        
        file.close()
        
        print(f"\nDefault parameter file for '{self.__class__.__name__}' has been created in {path}.\n\
You can now launch with 'struphy run {self.__class__.__name__}' or with 'struphy run -i params_{self.__class__.__name__}.py'")
        
        return path

    ###################
    # Private methods :
    ###################

    def _init_variable_dicts(self):
        """
        Initialize em-fields, fluid and kinetic dictionaries for information on the model variables.
        """

        # electromagnetic fields, fluid and/or kinetic species
        self._em_fields = {}
        self._fluid = {}
        self._kinetic = {}
        self._diagnostics = {}

        if self.rank_world == 0 and self.verbose:
            print("\nMODEL SPECIES:")

        # create dictionaries for each em-field/species and fill in space/class name and parameters
        for var_name, space in self.species()["em_fields"].items():
            assert space in {"H1", "Hcurl", "Hdiv", "L2", "H1vec"}
            assert self.params.em_fields is not None, '"em_fields" is missing in parameter file.'

            if self.rank_world == 0 and self.verbose:
                print("em_field:".ljust(25), f'"{var_name}" ({space})')

            self._em_fields[var_name] = {}

            # space
            self._em_fields[var_name]["space"] = space

            # initial conditions
            if "background" in self.params.em_fields:
                # background= self.params.em_fields["background"].get(var_name)
                self._em_fields[var_name]["background"] = self.params.em_fields["background"].get(var_name)
            # else:
            #     background = None

            if "perturbation" in self.params.em_fields:
                # perturbation = self.params.em_fields["perturbation"].get(var_name)
                self._em_fields[var_name]["perturbation"] = self.params.em_fields["perturbation"].get(var_name)
            # else:
            #     perturbation = None

            # which components to save
            if "save_data" in self.params.em_fields:
                # save_data = self.params.em_fields["save_data"]["comps"][var_name]
                self._em_fields[var_name]["save_data"] = self.params.em_fields["save_data"]["comps"][var_name]
            else:
                self._em_fields[var_name]["save_data"] = True
                # save_data = True

            # self._em_fields[var_name] = Variable(name=var_name,
            #                                      space=space,
            #                                      background=background,
            #                                      perturbation=perturbation,
            #                                      save_data=save_data,)

            # overall parameters
            # print(f'{self._em_fields = }')
            self._em_fields["params"] = self.params.em_fields

        for var_name, space in self.species()["fluid"].items():
            assert isinstance(space, dict)
            assert "fluid" in self.params, 'Top-level key "fluid" is missing in parameter file.'
            assert var_name in self.params["fluid"], f"Fluid species {var_name} is missing in parameter file."

            if self.rank_world == 0 and self.verbose:
                print("fluid:".ljust(25), f'"{var_name}" ({space})')

            self._fluid[var_name] = {}
            for sub_var_name, sub_space in space.items():
                self._fluid[var_name][sub_var_name] = {}

                # space
                self._fluid[var_name][sub_var_name]["space"] = sub_space

                # initial conditions
                if "background" in self.params["fluid"][var_name]:
                    self._fluid[var_name][sub_var_name]["background"] = self.params["fluid"][var_name][
                        "background"
                    ].get(sub_var_name)
                if "perturbation" in self.params["fluid"][var_name]:
                    self._fluid[var_name][sub_var_name]["perturbation"] = self.params["fluid"][var_name][
                        "perturbation"
                    ].get(sub_var_name)

                # which components to save
                if "save_data" in self.params["fluid"][var_name]:
                    self._fluid[var_name][sub_var_name]["save_data"] = self.params["fluid"][var_name]["save_data"][
                        "comps"
                    ][sub_var_name]

                else:
                    self._fluid[var_name][sub_var_name]["save_data"] = True

            # overall parameters
            self._fluid[var_name]["params"] = self.params["fluid"][var_name]

        for var_name, space in self.species()["kinetic"].items():
            assert "Particles" in space
            assert "kinetic" in self.params, 'Top-level key "kinetic" is missing in parameter file.'
            assert var_name in self.params["kinetic"], f"Kinetic species {var_name} is missing in parameter file."

            if self.rank_world == 0 and self.verbose:
                print("kinetic:".ljust(25), f'"{var_name}" ({space})')

            self._kinetic[var_name] = {}
            self._kinetic[var_name]["space"] = space
            self._kinetic[var_name]["params"] = self.params["kinetic"][var_name]

        if self.diagnostics_dct() is not None:
            for var_name, space in self.diagnostics_dct().items():
                assert space in {"H1", "Hcurl", "Hdiv", "L2", "H1vec"}

                if self.rank_world == 0 and self.verbose:
                    print("diagnostics:".ljust(25), f'"{var_name}" ({space})')

                self._diagnostics[var_name] = {}
                self._diagnostics[var_name]["space"] = space
                self._diagnostics["params"] = self.params["diagnostics"][var_name]

                # which components to save
                if "save_data" in self.params["diagnostics"][var_name]:
                    self._diagnostics[var_name]["save_data"] = self.params["diagnostics"][var_name]["save_data"]

                else:
                    self._diagnostics[var_name]["save_data"] = True

    def compute_plasma_params(self, verbose=True):
        """
        Compute and print volume averaged plasma parameters for each species of the model.

        Global parameters:
        - plasma volume
        - transit length
        - magnetic field

        Species dependent parameters:
        - mass
        - charge
        - density
        - pressure
        - thermal energy kBT
        - Alfvén speed v_A
        - thermal speed v_th
        - thermal frequency Omega_th
        - cyclotron frequency Omega_c
        - plasma frequency Omega_p
        - Alfvèn frequency Omega_A
        - thermal Larmor radius rho_th
        - MHD length scale v_a/Omega_c
        - rho/L
        - alpha = Omega_p/Omega_c
        - epsilon = 1/(t*Omega_c)
        """

        # units affices for printing
        units_affix = {}
        units_affix["plasma volume"] = " m³"
        units_affix["transit length"] = " m"
        units_affix["magnetic field"] = " T"
        units_affix["mass"] = " kg"
        units_affix["charge"] = " C"
        units_affix["density"] = " m⁻³"
        units_affix["pressure"] = " bar"
        units_affix["kBT"] = " keV"
        units_affix["v_A"] = " m/s"
        units_affix["v_th"] = " m/s"
        units_affix["vth1"] = " m/s"
        units_affix["vth2"] = " m/s"
        units_affix["vth3"] = " m/s"
        units_affix["Omega_th"] = " Mrad/s"
        units_affix["Omega_c"] = " Mrad/s"
        units_affix["Omega_p"] = " Mrad/s"
        units_affix["Omega_A"] = " Mrad/s"
        units_affix["rho_th"] = " m"
        units_affix["v_A/Omega_c"] = " m"
        units_affix["rho_th/L"] = ""
        units_affix["alpha"] = ""
        units_affix["epsilon"] = ""

        h = 1 / 20
        eta1 = np.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta2 = np.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta3 = np.linspace(h / 2.0, 1.0 - h / 2.0, 20)

        ##  global parameters
        
        # plasma volume (hat x^3)
        det_tmp = self.domain.jacobian_det(eta1, eta2, eta3)
        vol1 = np.mean(np.abs(det_tmp))
        # plasma volume (m⁻³)
        plasma_volume = vol1 * self.units.x ** 3
        # transit length (m)
        transit_length = plasma_volume ** (1 / 3)
        # magnetic field (T)
        if isinstance(self.equil, FluidEquilibriumWithB):
            B_tmp = self.equil.absB0(eta1, eta2, eta3)
        else:
            B_tmp = np.zeros((eta1.size, eta2.size, eta3.size))
        magnetic_field = np.mean(B_tmp * np.abs(det_tmp)) / vol1 * self.units.B
        B_max = np.max(B_tmp) * self.units.B
        B_min = np.min(B_tmp) * self.units.B

        if magnetic_field < 1e-14:
            magnetic_field = np.nan
            # print("\n+++++++ WARNING +++++++ magnetic field is zero - set to nan !!")

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("\nPLASMA PARAMETERS:")
            print(
                f"Plasma volume:".ljust(25),
                "{:4.3e}".format(plasma_volume) + units_affix["plasma volume"],
            )
            print(
                f"Transit length:".ljust(25),
                "{:4.3e}".format(transit_length) + units_affix["transit length"],
            )
            print(
                f"Avg. magnetic field:".ljust(25),
                "{:4.3e}".format(magnetic_field) + units_affix["magnetic field"],
            )
            print(
                f"Max magnetic field:".ljust(25),
                "{:4.3e}".format(B_max) + units_affix["magnetic field"],
            )
            print(
                f"Min magnetic field:".ljust(25),
                "{:4.3e}".format(B_min) + units_affix["magnetic field"],
            )

        # # species dependent parameters
        # self._pparams = {}

        # if len(self.fluid_species) > 0:
        #     for species, val in self.fluid_species.items():
        #         self._pparams[species] = {}
        #         # type
        #         self._pparams[species]["type"] = "fluid"
        #         # mass (kg)
        #         self._pparams[species]["mass"] = val["params"]["phys_params"]["A"] * m_p
        #         # charge (C)
        #         self._pparams[species]["charge"] = val["params"]["phys_params"]["Z"] * e
        #         # density (m⁻³)
        #         self._pparams[species]["density"] = (
        #             np.mean(
        #                 self.equil.n0(
        #                     eta1,
        #                     eta2,
        #                     eta3,
        #                 )
        #                 * np.abs(det_tmp),
        #             )
        #             * self.units.x ** 3
        #             / plasma_volume
        #             * self.units.n
        #         )
        #         # pressure (bar)
        #         self._pparams[species]["pressure"] = (
        #             np.mean(
        #                 self.equil.p0(
        #                     eta1,
        #                     eta2,
        #                     eta3,
        #                 )
        #                 * np.abs(det_tmp),
        #             )
        #             * self.units.x ** 3
        #             / plasma_volume
        #             * self.units.p
        #             * 1e-5
        #         )
        #         # thermal energy (keV)
        #         self._pparams[species]["kBT"] = self._pparams[species]["pressure"] * 1e5 / self._pparams[species]["density"] / e * 1e-3

        # if len(self.kinetic) > 0:
        #     eta1mg, eta2mg, eta3mg = np.meshgrid(
        #         eta1,
        #         eta2,
        #         eta3,
        #         indexing="ij",
        #     )

        #     for species, val in self.kinetic.items():
        #         self._pparams[species] = {}
        #         # type
        #         self._pparams[species]["type"] = "kinetic"
        #         # mass (kg)
        #         self._pparams[species]["mass"] = val["params"]["phys_params"]["A"] * m_p
        #         # charge (C)
        #         self._pparams[species]["charge"] = val["params"]["phys_params"]["Z"] * e

        #         # create temp kinetic object for (default) parameter extraction
        #         tmp_bckgr = val["params"]["background"]

        #         if val["space"] != "ParticlesSPH":
        #             tmp = None
        #             for fi, maxw_params in tmp_bckgr.items():
        #                 if fi[-2] == "_":
        #                     fi_type = fi[:-2]
        #                 else:
        #                     fi_type = fi

        #                 if tmp is None:
        #                     tmp = getattr(maxwellians, fi_type)(
        #                         maxw_params=maxw_params,
        #                         equil=self.equil,
        #                     )
        #                 else:
        #                     tmp = tmp + getattr(maxwellians, fi_type)(
        #                         maxw_params=maxw_params,
        #                         equil=self.equil,
        #                     )

        #         if val["space"] != "ParticlesSPH" and tmp.coords == "constants_of_motion":
        #             # call parameters
        #             a1 = self.domain.params_map["a1"]
        #             r = eta1mg * (1 - a1) + a1
        #             psi = self.equil.psi_r(r)

        #             # density (m⁻³)
        #             self._pparams[species]["density"] = (
        #                 np.mean(tmp.n(psi) * np.abs(det_tmp)) * self.units.x ** 3 / plasma_volume * self.units.n
        #             )
        #             # thermal speed (m/s)
        #             self._pparams[species]["v_th"] = (
        #                 np.mean(tmp.vth(psi) * np.abs(det_tmp)) * self.units.x ** 3 / plasma_volume * self.units.v
        #             )
        #             # thermal energy (keV)
        #             self._pparams[species]["kBT"] = self._pparams[species]["mass"] * self._pparams[species]["v_th"] ** 2 / e * 1e-3
        #             # pressure (bar)
        #             self._pparams[species]["pressure"] = (
        #                 self._pparams[species]["kBT"] * e * 1e3 * self._pparams[species]["density"] * 1e-5
        #             )

        #         else:
        #             # density (m⁻³)
        #             # self._pparams[species]['density'] = np.mean(tmp.n(
        #             #     eta1mg, eta2mg, eta3mg) * np.abs(det_tmp)) * units['x']**3 / plasma_volume * units['n']
        #             self._pparams[species]["density"] = 99.0
        #             # thermal speeds (m/s)
        #             vth = []
        #             # vths = tmp.vth(eta1mg, eta2mg, eta3mg)
        #             vths = [99.0]
        #             for k in range(len(vths)):
        #                 vth += [
        #                     vths[k] * np.abs(det_tmp) * self.units.x ** 3 / plasma_volume * self.units.v,
        #                 ]
        #             thermal_speed = 0.0
        #             for dir in range(val["obj"].vdim):
        #                 # self._pparams[species]['vth' + str(dir + 1)] = np.mean(vth[dir])
        #                 self._pparams[species]["vth" + str(dir + 1)] = 99.0
        #                 thermal_speed += self._pparams[species]["vth" + str(dir + 1)]
        #             # TODO: here it is assumed that background density parameter is called "n",
        #             # and that background thermal speeds are called "vthn"; make this a convention?
        #             # self._pparams[species]['v_th'] = thermal_speed / \
        #             #     val['obj'].vdim
        #             self._pparams[species]["v_th"] = 99.0
        #             # thermal energy (keV)
        #             # self._pparams[species]['kBT'] = self._pparams[species]['mass'] * \
        #             #     self._pparams[species]['v_th']**2 / e * 1e-3
        #             self._pparams[species]["kBT"] = 99.0
        #             # pressure (bar)
        #             # self._pparams[species]['pressure'] = self._pparams[species]['kBT'] * \
        #             #     e * 1e3 * self._pparams[species]['density'] * 1e-5
        #             self._pparams[species]["pressure"] = 99.0

        # for species in self._pparams:
        #     # alfvén speed (m/s)
        #     self._pparams[species]["v_A"] = magnetic_field / np.sqrt(
        #         mu0 * self._pparams[species]["mass"] * self._pparams[species]["density"],
        #     )
        #     # thermal speed (m/s)
        #     self._pparams[species]["v_th"] = np.sqrt(
        #         self._pparams[species]["kBT"] * 1e3 * e / self._pparams[species]["mass"],
        #     )
        #     # thermal frequency (Mrad/s)
        #     self._pparams[species]["Omega_th"] = self._pparams[species]["v_th"] / transit_length * 1e-6
        #     # cyclotron frequency (Mrad/s)
        #     self._pparams[species]["Omega_c"] = self._pparams[species]["charge"] * magnetic_field / self._pparams[species]["mass"] * 1e-6
        #     # plasma frequency (Mrad/s)
        #     self._pparams[species]["Omega_p"] = (
        #         np.sqrt(
        #             self._pparams[species]["density"] * (self._pparams[species]["charge"]) ** 2 / eps0 / self._pparams[species]["mass"],
        #         )
        #         * 1e-6
        #     )
        #     # alfvén frequency (Mrad/s)
        #     self._pparams[species]["Omega_A"] = self._pparams[species]["v_A"] / transit_length * 1e-6
        #     # Larmor radius (m)
        #     self._pparams[species]["rho_th"] = self._pparams[species]["v_th"] / (self._pparams[species]["Omega_c"] * 1e6)
        #     # MHD length scale (m)
        #     self._pparams[species]["v_A/Omega_c"] = self._pparams[species]["v_A"] / (np.abs(self._pparams[species]["Omega_c"]) * 1e6)
        #     # dim-less ratios
        #     self._pparams[species]["rho_th/L"] = self._pparams[species]["rho_th"] / transit_length

        # if verbose and self.rank_world == 0:
        #     print("\nSPECIES PARAMETERS:")
        #     for species, ch in self._pparams.items():
        #         print(f"\nname:".ljust(26), species)
        #         print(f"type:".ljust(25), ch["type"])
        #         ch.pop("type")
        #         print(f"is bulk:".ljust(25), species == self.bulk_species())
        #         for kinds, vals in ch.items():
        #             print(
        #                 kinds.ljust(25),
        #                 "{:+4.3e}".format(
        #                     vals,
        #                 ),
        #                 units_affix[kinds],
        #             )


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def ignore_aliases(self, data):
        return True
