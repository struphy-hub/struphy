import inspect
import operator
from abc import ABCMeta, abstractmethod
from functools import reduce

import yaml
from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.stencil import StencilVector

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import SplineFunction
from struphy.fields_background.base import FluidEquilibrium, FluidEquilibriumWithB, MHDequilibrium
from struphy.fields_background.projected_equils import (
    ProjectedFluidEquilibrium,
    ProjectedFluidEquilibriumWithB,
    ProjectedMHDequilibrium,
)
from struphy.io.setup import setup_derham, setup_domain_and_equil
from struphy.profiling.profiling import ProfileManager
from struphy.propagators.base import Propagator
from struphy.utils.arrays import xp as np
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml
from struphy.io.parameters import StruphyParameters


class StruphyModel(metaclass=ABCMeta):
    """
    Base class for all Struphy models.

    Parameters
    ----------
    params : StruphyParameters
        Simulation parameters.

    comm : mpi4py.MPI.Intracomm
        MPI communicator for parallel runs.

    clone_config: struphy.utils.CloneConfig
        Contains the # TODO

    Note
    ----
    All Struphy models are subclasses of ``StruphyModel`` and should be added to ``struphy/models/``
    in one of the modules ``fluid.py``, ``kinetic.py``, ``hybrid.py`` or ``toy.py``.
    """

    def __init__(
        self,
        params: StruphyParameters = None,
        comm: MPI.Intracomm = None,
        clone_config: CloneConfig = None,
    ):
        assert "em_fields" in self.species()
        assert "fluid" in self.species()
        assert "kinetic" in self.species()

        assert "em_fields" in self.options()
        assert "fluid" in self.options()
        assert "kinetic" in self.options()

        if params is None:
            params = self.generate_default_parameter_file(
                save=False,
                prompt=False,
            )

        self._comm_world = comm
        self._clone_config = clone_config

        self._params = params

        # get rank and size
        if self.comm_world is None:
            self._rank_world = 0
        else:
            self._rank_world = self.comm_world.Get_rank()

        # initialize model variable dictionaries
        self._init_variable_dicts()

        # compute model units
        self._units, self._equation_params = self.model_units(
            self.params,
            verbose=self.verbose,
            comm=self.comm_world,
        )

        # create domain, equilibrium
        self._domain, self._equil = setup_domain_and_equil(params)

        if self.rank_world == 0 and self.verbose:
            print("\nTIME:")
            print(
                f"time step:".ljust(25),
                "{0} ({1:4.2e} s)".format(
                    params.time.dt,
                    params.time.dt * self.units["t"],
                ),
            )
            print(
                f"final time:".ljust(25),
                "{0} ({1:4.2e} s)".format(
                    params.time.Tend,
                    params.time.Tend * self.units["t"],
                ),
            )
            print(f"splitting algo:".ljust(25), params.time.split_algo)

            print("\nDOMAIN:")
            print(f"type:".ljust(25), self.domain.__class__.__name__)
            for key, val in self.domain.params.items():
                if key not in {"cx", "cy", "cz"}:
                    print((key + ":").ljust(25), val)

            print("\nFLUID BACKGROUND:")
            if params.equil is not None:
                print("type:".ljust(25), self.equil.__class__.__name__)
                for key, val in self.equil.params.items():
                    print((key + ":").ljust(25), val)
            else:
                print("None.")

        # create discrete derham sequence
        if params.grid is not None:
            dims_mask = params.grid.mpi_dims_mask
            if dims_mask is None:
                dims_mask = [True] * 3

            if clone_config is None:
                derham_comm = self.comm_world
            else:
                derham_comm = clone_config.sub_comm

            self._derham = setup_derham(
                params.grid,
                params.derham,
                comm=derham_comm,
                domain=self.domain,
                mpi_dims_mask=dims_mask,
                verbose=self.verbose,
            )
        else:
            self._derham = None
            print("\nDERHAM:\nMeshless simulation - no Derham complex set up.")

        self._projected_equil = None
        self._mass_ops = None
        if self.derham is not None:
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

            # create weighted mass operators
            self._mass_ops = WeightedMassOperators(
                self.derham,
                self.domain,
                verbose=self.verbose,
                eq_mhd=self.equil,
            )

        # allocate memory for variables
        self._pointer = {}
        self._allocate_variables()

        # store plasma parameters
        if self.rank_world == 0:
            self._pparams = self._compute_plasma_params(verbose=self.verbose)
        else:
            self._pparams = self._compute_plasma_params(verbose=False)

        # if self.rank_world == 0:
        #     self._show_chosen_options()

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

        # create dummy lists/dicts to be filled by the sub-class
        self._propagators = []
        self._kwargs = {}
        self._scalar_quantities = {}

        return params

    @staticmethod
    @abstractmethod
    def species():
        """Species dictionary of the form {'em_fields': {}, 'fluid': {}, 'kinetic': {}}.

        The dynamical fields and kinetic species of the model.

        Keys of the three sub-dicts are either:

        a) the electromagnetic field/potential names (b_field, e_field)
        b) the fluid species names (e.g. mhd)
        c) the names of the kinetic species (e.g. electrons, energetic_ions)

        Corresponding values are:

        a) a space ID ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"),
        b) a dict with key=variable_name (e.g. n, U, p, ...) and value=space ID ("H1", "Hcurl", "Hdiv", "L2" or "H1vec"),
        c) the type of particles ("Particles6D", "Particles5D", ...)."""
        pass

    @staticmethod
    @abstractmethod
    def bulk_species():
        """Name of the bulk species of the plasma. Must be a key of self.fluid or self.kinetic, or None."""
        pass

    @staticmethod
    @abstractmethod
    def velocity_scale():
        """String that sets the velocity scale unit of the model.
        Must be one of "alfvén", "cyclotron" or "light"."""
        pass

    @staticmethod
    def diagnostics_dct():
        """Diagnostics dictionary.
        Model specific variables (FemField) which is going to be saved during the simulation.
        """
        pass

    @staticmethod
    @abstractmethod
    def propagators_dct(cls):
        """Dictionary holding the propagators of the model in the sequence they should be called.
        Keys are the propagator classes and values are lists holding variable names (str) updated by the propagator."""
        pass

    @abstractmethod
    def update_scalar_quantities(self):
        """Specify an update rule for each item in ``scalar_quantities`` using :meth:`update_scalar`."""
        pass

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
    def comm_world(self):
        """MPI_COMM_WORLD communicator."""
        return self._comm_world

    @property
    def rank_world(self):
        """Global rank."""
        return self._rank_world

    @property
    def clone_config(self):
        """Config in case domain clones are used."""
        return self._clone_config

    @property
    def pointer(self):
        """Dictionary pointing to the data structures of the species (Stencil/BlockVector or "Particle" class).

        The keys are the keys from the "species" property.
        In case of a fluid species, the keys are like "species_variable"."""
        return self._pointer

    @property
    def em_fields(self):
        """Dictionary of electromagnetic field/potential variables."""
        return self._em_fields

    @property
    def fluid(self):
        """Dictionary of fluid species."""
        return self._fluid

    @property
    def kinetic(self):
        """Dictionary of kinetic species."""
        return self._kinetic

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
    def units(self):
        """All Struphy units."""
        return self._units

    @property
    def mass_ops(self):
        """WeighteMassOperators object, see :ref:`mass_ops`."""
        return self._mass_ops

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
    def propagators(self):
        """A list of propagator instances for the model."""
        return self._propagators

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

    def add_scalar(self, name, species=None, compute=None, summands=None):
        """
        Add a scalar to be saved during the simulation.

        Parameters
        ----------
        name : str
            Dictionary key for the scalar.
        species : str, optional
            The species associated with the scalar. Required if compute is 'from_particles'.
        compute : str, optional
            Type of scalar, determines the compute operations.
            Options: 'from_particles' or 'from_field'. Default is None.
        summands : list, optional
            List of other scalar names whose values should be summed
            to compute the value of this scalar. Default is None.
        """

        assert isinstance(name, str), "name must be a string"
        if compute == "from_particles":
            assert isinstance(
                species,
                str,
            ), "species must be a string when compute is 'from_particles'"

        self._scalar_quantities[name] = {
            "value": np.empty(1, dtype=float),
            "species": species,
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

        species = self._scalar_quantities[name]["species"]
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
                n_mks_tot = np.array([self.pointer[species].Np])
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
        for prop in self.propagators:
            prop.add_time_state(time_state)

    def init_propagators(self):
        """Initialize the propagator objects specified in :attr:`~propagators_cls`."""
        if self.rank_world == 0 and self.verbose:
            print("\nPROPAGATORS:")
        for (prop, variables), (prop2, kwargs_i) in zip(self.propagators_dct().items(), self.kwargs.items()):
            assert prop == prop2, (
                f'Propagators {prop} from "self.propagators_dct()" and {prop2} from "self.kwargs" must be identical !!'
            )

            if kwargs_i is None:
                if self.rank_world == 0:
                    print(f'\n-> Propagator "{prop.__name__}" will not be used.')
                continue
            else:
                if self.rank_world == 0 and self.verbose:
                    print(f'\n-> Initializing propagator "{prop.__name__}"')
                    print(f"-> for variables {variables}")
                    print(f"-> with the following parameters:")
                    for k, v in kwargs_i.items():
                        if isinstance(v, StencilVector):
                            print(f"{k}: {repr(v)}")
                        else:
                            print(f"{k}: {v}")

                prop_instance = prop(
                    *[self.pointer[var] for var in variables],
                    **kwargs_i,
                )
                assert isinstance(prop_instance, Propagator)
                self._propagators += [prop_instance]

        if self.rank_world == 0 and self.verbose:
            print("\nInitialization of propagators complete.")

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
            for propagator in self.propagators:
                prop_name = type(propagator).__name__

                with ProfileManager.profile_region(prop_name):
                    propagator(dt)

        # second order in time
        elif split_algo == "Strang":
            assert len(self.propagators) > 1

            for propagator in self.propagators[:-1]:
                prop_name = type(propagator).__name__
                with ProfileManager.profile_region(prop_name):
                    propagator(dt / 2)

            propagator = self.propagators[-1]
            prop_name = type(propagator).__name__
            with ProfileManager.profile_region(prop_name):
                propagator(dt)

            for propagator in self.propagators[:-1][::-1]:
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

        from struphy.pic.base import Particles

        for val in self.kinetic.values():
            obj = val["obj"]
            assert isinstance(obj, Particles)

            # allocate array for saving markers if not present
            if not hasattr(self, "_n_markers_saved"):
                n_markers = val["params"]["save_data"].get("n_markers", 0)

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
                    val["kinetic_data"]["markers"] = np.zeros(
                        (self._n_markers_saved, obj.markers.shape[1]),
                        dtype=float,
                    )

            if self._n_markers_saved > 0:
                markers_on_proc = np.logical_and(
                    obj.markers[:, -1] >= 0.0,
                    obj.markers[:, -1] < self._n_markers_saved,
                )
                n_markers_on_proc = np.count_nonzero(markers_on_proc)
                val["kinetic_data"]["markers"][:] = -1.0
                val["kinetic_data"]["markers"][:n_markers_on_proc] = obj.markers[markers_on_proc]

    def update_distr_functions(self):
        """
        Writes distribution functions slices that are supposed to be saved into corresponding array.
        """

        from struphy.pic.base import Particles

        dim_to_int = {"e1": 0, "e2": 1, "e3": 2, "v1": 3, "v2": 4, "v3": 5}

        for val in self.kinetic.values():
            obj = val["obj"]
            assert isinstance(obj, Particles)

            if obj.n_cols_diagnostics > 0:
                for i in range(obj.n_cols_diagnostics):
                    str_dn = f"d{i + 1}"
                    dim_to_int[str_dn] = 3 + obj.vdim + 3 + i

            if "f" in val["params"]["save_data"]:
                for slice_i, edges in val["bin_edges"].items():
                    comps = slice_i.split("_")
                    components = [False] * (3 + obj.vdim + 3 + obj.n_cols_diagnostics)

                    for comp in comps:
                        components[dim_to_int[comp]] = True

                    f_slice, df_slice = obj.binning(components, edges)

                    val["kinetic_data"]["f"][slice_i][:] = f_slice
                    val["kinetic_data"]["df"][slice_i][:] = df_slice

            if "n_sph" in val["params"]["save_data"]:
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

    def initialize_from_params(self):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid)
        and markers according to parameter file.
        """

        from struphy.feec.psydac_derham import Derham
        from struphy.pic.base import Particles

        if self.rank_world == 0 and self.verbose:
            print("\nINITIAL CONDITIONS:")

        # initialize em fields
        if len(self.em_fields) > 0:
            with ProfileManager.profile_region("initialize_em_fields"):
                for key, val in self.em_fields.items():
                    if "params" in key:
                        continue
                    else:
                        obj = val["obj"]
                        assert isinstance(obj, SplineFunction)

                        obj.initialize_coeffs(
                            domain=self.domain,
                            bckgr_obj=self.equil,
                        )

                        if self.rank_world == 0 and self.verbose:
                            print(f'\nEM field "{key}" was initialized with:')

                            _params = self.em_fields["params"]

                            if "background" in _params:
                                if key in _params["background"]:
                                    bckgr_types = _params["background"][key]
                                    if bckgr_types is None:
                                        pass
                                    else:
                                        print("background:")
                                        for _type, _bp in bckgr_types.items():
                                            print(" " * 4 + _type, ":")
                                            for _pname, _pval in _bp.items():
                                                print((" " * 8 + _pname + ":").ljust(25), _pval)
                                else:
                                    print("No background.")
                            else:
                                print("No background.")

                            if "perturbation" in _params:
                                if key in _params["perturbation"]:
                                    pert_types = _params["perturbation"][key]
                                    if pert_types is None:
                                        pass
                                    else:
                                        print("perturbation:")
                                        for _type, _pp in pert_types.items():
                                            print(" " * 4 + _type, ":")
                                            for _pname, _pval in _pp.items():
                                                print((" " * 8 + _pname + ":").ljust(25), _pval)
                                else:
                                    print("No perturbation.")
                            else:
                                print("No perturbation.")

        if len(self.fluid) > 0:
            with ProfileManager.profile_region("initialize_fluids"):
                for species, val in self.fluid.items():
                    for variable, subval in val.items():
                        if "params" in variable:
                            continue
                        else:
                            obj = subval["obj"]
                            assert isinstance(obj, SplineFunction)
                            obj.initialize_coeffs(
                                domain=self.domain,
                                bckgr_obj=self.equil,
                                species=species,
                            )

                    if self.rank_world == 0 and self.verbose:
                        print(
                            f'\nFluid species "{species}" was initialized with:',
                        )

                        _params = val["params"]

                        if "background" in _params:
                            for variable in val:
                                if "params" in variable:
                                    continue
                                if variable in _params["background"]:
                                    bckgr_types = _params["background"][variable]
                                    if bckgr_types is None:
                                        pass
                                    else:
                                        print(f"{variable} background:")
                                        for _type, _bp in bckgr_types.items():
                                            print(" " * 4 + _type, ":")
                                            for _pname, _pval in _bp.items():
                                                print((" " * 8 + _pname + ":").ljust(25), _pval)
                                else:
                                    print(f"{variable}: no background.")
                        else:
                            print("No background.")

                        if "perturbation" in _params:
                            for variable in val:
                                if "params" in variable:
                                    continue
                                if variable in _params["perturbation"]:
                                    pert_types = _params["perturbation"][variable]
                                    if pert_types is None:
                                        pass
                                    else:
                                        print(f"{variable} perturbation:")
                                        for _type, _pp in pert_types.items():
                                            print(" " * 4 + _type, ":")
                                            for _pname, _pval in _pp.items():
                                                print((" " * 8 + _pname + ":").ljust(25), _pval)
                                else:
                                    print(f"{variable}: no perturbation.")
                        else:
                            print("No perturbation.")

        # initialize particles
        if len(self.kinetic) > 0:
            with ProfileManager.profile_region("initialize_particles"):
                for species, val in self.kinetic.items():
                    obj = val["obj"]
                    assert isinstance(obj, Particles)

                    if self.rank_world == 0 and self.verbose:
                        _params = val["params"]
                        assert "background" in _params, "Kinetic species must have background."

                        bckgr_types = _params["background"]
                        print(
                            f'\nKinetic species "{species}" was initialized with:',
                        )
                        for _type, _bp in bckgr_types.items():
                            print(_type, ":")
                            for _pname, _pval in _bp.items():
                                print((" " * 4 + _pname + ":").ljust(25), _pval)

                        if "perturbation" in _params:
                            for variable, pert_types in _params["perturbation"].items():
                                if pert_types is None:
                                    pass
                                else:
                                    print(f"{variable} perturbation:")
                                    for _type, _pp in pert_types.items():
                                        print(" " * 4 + _type, ":")
                                        for _pname, _pval in _pp.items():
                                            print((" " * 8 + _pname + ":").ljust(25), _pval)
                        else:
                            print("No perturbation.")

                    obj.draw_markers(sort=True, verbose=self.verbose)
                    if self.comm_world is not None:
                        obj.mpi_sort_markers(do_test=True)

                    if not val["params"]["markers"]["loading"] == "restart":
                        if obj.coords == "vpara_mu":
                            obj.save_magnetic_moment()

                        if val["space"] != "ParticlesSPH" and obj.f0.coords == "constants_of_motion":
                            obj.save_constants_of_motion()

                        obj.initialize_weights(
                            reject_weights=obj.weights_params["reject_weights"],
                            threshold=obj.weights_params["threshold"],
                        )

    def initialize_from_restart(self, data):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers from restart group in hdf5 files.

        Parameters
        ----------
        data : struphy.io.output_handling.DataContainer
            The data object that links to the hdf5 files.
        """

        from struphy.feec.psydac_derham import Derham
        from struphy.pic.base import Particles

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

    def initialize_data_output(self, data, size):
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

        from psydac.linalg.stencil import StencilVector

        from struphy.feec.psydac_derham import Derham
        from struphy.io.output_handling import DataContainer
        from struphy.pic.base import Particles

        assert isinstance(data, DataContainer)

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

        # save electromagentic fields/potentials data in group 'feec/'
        for key, val in self.em_fields.items():
            if "params" in key:
                continue
            else:
                obj = val["obj"]
                assert isinstance(obj, SplineFunction)

                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                obj.extract_coeffs(update_ghost_regions=False)

                # save numpy array to be updated each time step.
                if val["save_data"]:
                    key_field = "feec/" + key

                    if isinstance(obj.vector_stencil, StencilVector):
                        data.add_data(
                            {key_field: obj.vector_stencil._data},
                        )

                    else:
                        for n in range(3):
                            key_component = key_field + "/" + str(n + 1)
                            data.add_data(
                                {key_component: obj.vector_stencil[n]._data},
                            )

                    # save field meta data
                    data.file[key_field].attrs["space_id"] = obj.space_id
                    data.file[key_field].attrs["starts"] = obj.starts
                    data.file[key_field].attrs["ends"] = obj.ends
                    data.file[key_field].attrs["pads"] = obj.pads

                # save numpy array to be updated only at the end of the simulation for restart.
                key_field_restart = "restart/" + key

                if isinstance(obj.vector_stencil, StencilVector):
                    data.add_data(
                        {key_field_restart: obj.vector_stencil._data},
                    )
                else:
                    for n in range(3):
                        key_component_restart = key_field_restart + "/" + str(n + 1)
                        data.add_data(
                            {key_component_restart: obj.vector_stencil[n]._data},
                        )

        # save fluid data in group 'feec/'
        for species, val in self.fluid.items():
            species_path = "feec/" + species + "_"
            species_path_restart = "restart/" + species + "_"

            for variable, subval in val.items():
                if "params" in variable:
                    continue
                else:
                    obj = subval["obj"]
                    assert isinstance(obj, SplineFunction)

                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    obj.extract_coeffs(update_ghost_regions=False)

                    # save numpy array to be updated each time step.
                    if subval["save_data"]:
                        key_field = species_path + variable

                        if isinstance(obj.vector_stencil, StencilVector):
                            data.add_data(
                                {key_field: obj.vector_stencil._data},
                            )

                        else:
                            for n in range(3):
                                key_component = key_field + "/" + str(n + 1)
                                data.add_data(
                                    {key_component: obj.vector_stencil[n]._data},
                                )

                        # save field meta data
                        data.file[key_field].attrs["space_id"] = obj.space_id
                        data.file[key_field].attrs["starts"] = obj.starts
                        data.file[key_field].attrs["ends"] = obj.ends
                        data.file[key_field].attrs["pads"] = obj.pads

                    # save numpy array to be updated only at the end of the simulation for restart.
                    key_field_restart = species_path_restart + variable

                    if isinstance(obj.vector_stencil, StencilVector):
                        data.add_data(
                            {key_field_restart: obj.vector_stencil._data},
                        )
                    else:
                        for n in range(3):
                            key_component_restart = key_field_restart + "/" + str(n + 1)
                            data.add_data(
                                {key_component_restart: obj.vector_stencil[n]._data},
                            )

        # save kinetic data in group 'kinetic/'
        for key, val in self.kinetic.items():
            obj = val["obj"]
            assert isinstance(obj, Particles)

            key_spec = "kinetic/" + key
            key_spec_restart = "restart/" + key

            data.add_data({key_spec_restart: obj._markers})

            for key1, val1 in val["kinetic_data"].items():
                key_dat = key_spec + "/" + key1

                # case of "f" and "df"
                if isinstance(val1, dict):
                    for key2, val2 in val1.items():
                        key_f = key_dat + "/" + key2
                        data.add_data({key_f: val2})

                        dims = (len(key2) - 2) // 3 + 1
                        for dim in range(dims):
                            data.file[key_f].attrs["bin_centers" + "_" + str(dim + 1)] = (
                                val["bin_edges"][key2][dim][:-1]
                                + (val["bin_edges"][key2][dim][1] - val["bin_edges"][key2][dim][0]) / 2
                            )
                # case of "n_sph"
                elif isinstance(val1, list):
                    for i, v1 in enumerate(val1):
                        key_n = key_dat + "/view_" + str(i)
                        data.add_data({key_n: v1})
                        # save 1d point values, not meshgrids, because attrs size is limited
                        eta1 = val["plot_pts"][i][0][:, 0, 0]
                        eta2 = val["plot_pts"][i][1][0, :, 0]
                        eta3 = val["plot_pts"][i][2][0, 0, :]
                        data.file[key_n].attrs["eta1"] = eta1
                        data.file[key_n].attrs["eta2"] = eta2
                        data.file[key_n].attrs["eta3"] = eta3
                else:
                    data.add_data({key_dat: val1})

        # save diagnostics data in group 'feec/'
        for key, val in self.diagnostics.items():
            if "params" in key:
                continue
            else:
                obj = val["obj"]
                assert isinstance(obj, SplineFunction)

                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                obj.extract_coeffs(update_ghost_regions=False)

                # save numpy array to be updated each time step.
                if val["save_data"]:
                    key_field = "feec/" + key

                    if isinstance(obj.vector_stencil, StencilVector):
                        data.add_data(
                            {key_field: obj.vector_stencil._data},
                        )

                    else:
                        for n in range(3):
                            key_component = key_field + "/" + str(n + 1)
                            data.add_data(
                                {key_component: obj.vector_stencil[n]._data},
                            )

                    # save field meta data
                    data.file[key_field].attrs["space_id"] = obj.space_id
                    data.file[key_field].attrs["starts"] = obj.starts
                    data.file[key_field].attrs["ends"] = obj.ends
                    data.file[key_field].attrs["pads"] = obj.pads

                # save numpy array to be updated only at the end of the simulation for restart.
                key_field_restart = "restart/" + key

                if isinstance(obj.vector_stencil, StencilVector):
                    data.add_data(
                        {key_field_restart: obj.vector_stencil._data},
                    )
                else:
                    for n in range(3):
                        key_component_restart = key_field_restart + "/" + str(n + 1)
                        data.add_data(
                            {key_component_restart: obj.vector_stencil[n]._data},
                        )

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
    def model_units(cls, params: StruphyParameters, verbose: bool=False, comm: MPI.Intracomm=None,):
        """
        Return model units and print them to screen.

        Parameters
        ----------
        params : StruphyParameters
            model parameters.

        verbose : bool, optional
            print model units to screen.

        comm : obj
            MPI communicator.

        Returns
        -------
        units_basic : dict
            Basic units for time, length, mass and magnetic field.

        units_der : dict
            Derived units for velocity, pressure, mass density and particle density.
        """

        from struphy.io.setup import derive_units

        if comm is None:
            rank = 0
        else:
            rank = comm.Get_rank()

        # look for bulk species in fluid OR kinetic parameter dictionaries
        Z_bulk = None
        A_bulk = None
        if params.fluid is not None:
            if cls.bulk_species() in params["fluid"]:
                Z_bulk = params["fluid"][cls.bulk_species()]["phys_params"]["Z"]
                A_bulk = params["fluid"][cls.bulk_species()]["phys_params"]["A"]
        if params.kinetic is not None:
            if cls.bulk_species() in params["kinetic"]:
                Z_bulk = params["kinetic"][cls.bulk_species()]["phys_params"]["Z"]
                A_bulk = params["kinetic"][cls.bulk_species()]["phys_params"]["A"]

        # compute model units
        kBT = params.units.kBT

        units = derive_units(
            Z_bulk=Z_bulk,
            A_bulk=A_bulk,
            x=params.units.x,
            B=params.units.B,
            n=params.units.n,
            kBT=kBT,
            velocity_scale=cls.velocity_scale(),
        )

        # print to screen
        if verbose and rank == 0:
            print("\nUNITS:")
            print(
                f"Unit of length:".ljust(25),
                "{:4.3e}".format(units["x"]) + " m",
            )
            print(
                f"Unit of time:".ljust(25),
                "{:4.3e}".format(units["t"]) + " s",
            )
            print(
                f"Unit of velocity:".ljust(25),
                "{:4.3e}".format(units["v"]) + " m/s",
            )
            print(
                f"Unit of magnetic field:".ljust(25),
                "{:4.3e}".format(units["B"]) + " T",
            )

            if A_bulk is not None:
                print(
                    f"Unit of particle density:".ljust(25),
                    "{:4.3e}".format(units["n"]) + " m⁻³",
                )
                print(
                    f"Unit of mass density:".ljust(25),
                    "{:4.3e}".format(units["rho"]) + " kg/m³",
                )
                print(
                    f"Unit of pressure:".ljust(25),
                    "{:4.3e}".format(units["p"] * 1e-5) + " bar",
                )
                print(
                    f"Unit of current density:".ljust(25),
                    "{:4.3e}".format(units["j"]) + " A/m²",
                )

        # compute equation parameters for each species
        e = 1.602176634e-19  # elementary charge (C)
        mH = 1.67262192369e-27  # proton mass (kg)
        eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)

        equation_params = {}
        if params.fluid is not None:
            for species in params["fluid"]:
                Z = params["fluid"][species]["phys_params"]["Z"]
                A = params["fluid"][species]["phys_params"]["A"]

                # compute equation parameters
                om_p = np.sqrt(units["n"] * (Z * e) ** 2 / (eps0 * A * mH))
                om_c = Z * e * units["B"] / (A * mH)
                equation_params[species] = {}
                equation_params[species]["alpha"] = om_p / om_c
                equation_params[species]["epsilon"] = 1.0 / (om_c * units["t"])
                equation_params[species]["kappa"] = om_p * units["t"]

                if verbose and rank == 0:
                    print("\nNORMALIZATION PARAMETERS:")
                    print("- " + species + ":")
                    for key, val in equation_params[species].items():
                        print((key + ":").ljust(25), "{:4.3e}".format(val))

        if params.kinetic is not None:
            for species in params["kinetic"]:
                Z = params["kinetic"][species]["phys_params"]["Z"]
                A = params["kinetic"][species]["phys_params"]["A"]

                # compute equation parameters
                om_p = np.sqrt(units["n"] * (Z * e) ** 2 / (eps0 * A * mH))
                om_c = Z * e * units["B"] / (A * mH)
                equation_params[species] = {}
                equation_params[species]["alpha"] = om_p / om_c
                equation_params[species]["epsilon"] = 1.0 / (om_c * units["t"])
                equation_params[species]["kappa"] = om_p * units["t"]

                if verbose and rank == 0:
                    if "fluid" not in params:
                        print("\nNORMALIZATION PARAMETERS:")
                    print("- " + species + ":")
                    for key, val in equation_params[species].items():
                        print((key + ":").ljust(25), "{:4.3e}".format(val))

        return units, equation_params

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

    @classmethod
    def generate_default_parameter_file(
        cls,
        file: str = None,
        save: bool = True,
        prompt: bool = True,
    ):
        """Generate a parameter file with default options for each species,
        and save it to the current input path.

        The default name is params_<model_name>.yml.

        Parameters
        ----------
        file : str
            Alternative filename to params_<model_name>.yml.

        save : bool
            Whether to save the parameter file in the current input path.

        prompt : bool
            Whether to prompt for overwriting the specified .yml file.

        Returns
        -------
        The default parameter dictionary."""

        import os

        import yaml

        import struphy
        from struphy.io.setup import descend_options_dict

        libpath = struphy.__path__[0]

        # load a standard parameter file
        with open(os.path.join(libpath, "io/inp/parameters.yml")) as tmp:
            parameters = yaml.load(tmp, Loader=yaml.FullLoader)

        parameters["model"] = cls.__name__

        # extract default em_fields parameters
        bckgr_params_1_em = parameters["em_fields"]["background"]["var_1"]
        bckgr_params_2_em = parameters["em_fields"]["background"]["var_2"]
        parameters["em_fields"].pop("background")

        pert_params_1_em = parameters["em_fields"]["perturbation"]["var_1"]
        pert_params_2_em = parameters["em_fields"]["perturbation"]["var_2"]
        parameters["em_fields"].pop("perturbation")

        # extract default fluid parameters
        bckgr_params_1_fluid = parameters["fluid"]["species_name"]["background"]["var_1"]
        bckgr_params_2_fluid = parameters["fluid"]["species_name"]["background"]["var_2"]
        parameters["fluid"]["species_name"].pop("background")

        pert_params_1_fluid = parameters["fluid"]["species_name"]["perturbation"]["var_1"]
        pert_params_2_fluid = parameters["fluid"]["species_name"]["perturbation"]["var_2"]
        parameters["fluid"]["species_name"].pop("perturbation")

        # standard Maxwellians
        parameters["kinetic"]["species_name"].pop("background")
        maxw_name = {
            "6D": "Maxwellian3D",
            "5D": "GyroMaxwellian2D",
            "4D": "Maxwellian1D",
            "3D": "ColdPlasma",
            "PH": "ConstantVelocity",
        }

        # init options dicts
        d_opts = {"em_fields": [], "fluid": {}, "kinetic": {}}

        # set the correct names in the parameter file
        if len(cls.species()["em_fields"]) > 0:
            parameters["em_fields"]["background"] = {}
            parameters["em_fields"]["perturbation"] = {}
            for name, space in cls.species()["em_fields"].items():
                if space in {"H1", "L2"}:
                    parameters["em_fields"]["background"][name] = bckgr_params_1_em
                    parameters["em_fields"]["perturbation"][name] = pert_params_1_em
                elif space in {"Hcurl", "Hdiv", "H1vec"}:
                    parameters["em_fields"]["background"][name] = bckgr_params_2_em
                    parameters["em_fields"]["perturbation"][name] = pert_params_2_em
        else:
            parameters.pop("em_fields")

        # find out the default em_fields options of the model
        if "options" in cls.options()["em_fields"]:
            # create the default options parameters
            d_default = descend_options_dict(
                cls.options()["em_fields"]["options"],
                d_opts["em_fields"],
            )
            parameters["em_fields"]["options"] = d_default

        # fluid
        fluid_params = parameters["fluid"].pop("species_name")

        if len(cls.species()["fluid"]) > 0:
            for name, dct in cls.species()["fluid"].items():
                parameters["fluid"][name] = fluid_params
                parameters["fluid"][name]["background"] = {}
                parameters["fluid"][name]["perturbation"] = {}

                # find out the default fluid options of the model
                if name in cls.options()["fluid"]:
                    d_opts["fluid"][name] = []

                    # create the default options parameters
                    d_default = descend_options_dict(
                        cls.options()["fluid"][name]["options"],
                        d_opts["fluid"][name],
                    )

                    parameters["fluid"][name]["options"] = d_default

                # set the correct names parameter file
                for sub_name, space in dct.items():
                    if space in {"H1", "L2"}:
                        parameters["fluid"][name]["background"][sub_name] = bckgr_params_1_fluid
                        parameters["fluid"][name]["perturbation"][sub_name] = pert_params_1_fluid
                    elif space in {"Hcurl", "Hdiv", "H1vec"}:
                        parameters["fluid"][name]["background"][sub_name] = bckgr_params_2_fluid
                        parameters["fluid"][name]["perturbation"][sub_name] = pert_params_2_fluid
        else:
            parameters.pop("fluid")

        # kinetic
        kinetic_params = parameters["kinetic"].pop("species_name")

        if len(cls.species()["kinetic"]) > 0:
            parameters["kinetic"] = {}

            for name, kind in cls.species()["kinetic"].items():
                parameters["kinetic"][name] = kinetic_params

                # find out the default kinetic options of the model
                if name in cls.options()["kinetic"]:
                    d_opts["kinetic"][name] = []

                    # create the default options parameters
                    d_default = descend_options_dict(
                        cls.options()["kinetic"][name]["options"],
                        d_opts["kinetic"][name],
                    )

                    parameters["kinetic"][name]["options"] = d_default

                # set the background
                dim = kind[-2:]
                parameters["kinetic"][name]["background"] = {
                    maxw_name[dim]: {"n": 0.05},
                }
        else:
            parameters.pop("kinetic")

        # diagnostics
        if cls.diagnostics_dct() is not None:
            parameters["diagnostics"] = {}
            for name, space in cls.diagnostics_dct().items():
                parameters["diagnostics"][name] = {"save_data": True}

        cls.write_parameters_to_file(
            parameters=parameters,
            file=file,
            save=save,
            prompt=prompt,
        )

        return parameters

    ###################
    # Private methods :
    ###################

    def _init_variable_dicts(self):
        """
        Initialize em-fields, fluid and kinetic dictionaries for information on the model variables.
        """

        # from struphy.models.variables import Variable

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

    def _allocate_variables(self):
        """
        Allocate memory for model variables.
        Creates FEM fields for em-fields and fluid variables and a particle class for kinetic species.
        """

        from struphy.feec.psydac_derham import Derham
        from struphy.pic import particles
        from struphy.pic.base import Particles

        # allocate memory for FE coeffs of electromagnetic fields/potentials
        if self.params.em_fields is not None:
            for variable, dct in self.em_fields.items():
                if "params" in variable:
                    continue
                else:
                    dct["obj"] = self.derham.create_spline_function(
                        variable,
                        dct["space"],
                        bckgr_params=dct.get("background"),
                        pert_params=dct.get("perturbation"),
                    )

                    self._pointer[variable] = dct["obj"].vector

        # allocate memory for FE coeffs of fluid variables
        if self.params.fluid is not None:
            for species, dct in self.fluid.items():
                for variable, subdct in dct.items():
                    if "params" in variable:
                        continue
                    else:
                        subdct["obj"] = self.derham.create_spline_function(
                            variable,
                            subdct["space"],
                            bckgr_params=subdct.get("background"),
                            pert_params=subdct.get("perturbation"),
                        )

                        self._pointer[species + "_" + variable] = subdct["obj"].vector

        # marker arrays and plasma parameters of kinetic species
        if self.params.kinetic is not None:
            for species, val in self.kinetic.items():
                assert any([key in val["params"]["markers"] for key in ["Np", "ppc", "ppb"]])

                bckgr_params = val["params"].get("background", None)
                pert_params = val["params"].get("perturbation", None)
                boxes_per_dim = val["params"].get("boxes_per_dim", None)
                mpi_dims_mask = val["params"].get("dims_mask", None)
                weights_params = val["params"].get("weights", None)

                if self.derham is None:
                    domain_decomp = None
                else:
                    domain_array = self.derham.domain_array
                    nprocs = self.derham.domain_decomposition.nprocs
                    domain_decomp = (domain_array, nprocs)

                kinetic_class = getattr(particles, val["space"])
                # print(f"{kinetic_class = }")
                val["obj"] = kinetic_class(
                    comm_world=self.comm_world,
                    clone_config=self.clone_config,
                    **val["params"]["markers"],
                    weights_params=weights_params,
                    domain_decomp=domain_decomp,
                    mpi_dims_mask=mpi_dims_mask,
                    boxes_per_dim=boxes_per_dim,
                    name=species,
                    equation_params=self.equation_params[species],
                    domain=self.domain,
                    equil=self.equil,
                    projected_equil=self.projected_equil,
                    bckgr_params=bckgr_params,
                    pert_params=pert_params,
                )

                obj = val["obj"]
                assert isinstance(obj, Particles)

                self._pointer[species] = obj

                # for storing markers
                val["kinetic_data"] = {}

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

        # allocate memory for FE coeffs of diagnostics
        if self.params.diagnostic_fields is not None:
            for key, val in self.diagnostics.items():
                if "params" in key:
                    continue
                else:
                    val["obj"] = self.derham.create_spline_function(
                        key,
                        val["space"],
                        bckgr_params=None,
                        pert_params=None,
                    )

                    self._pointer[key] = val["obj"].vector

    def _compute_plasma_params(self, verbose=True):
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

        Returns
        -------
            pparams : dict
                Plasma parameters for each species.
        """

        from struphy.fields_background import equils
        from struphy.fields_background.base import FluidEquilibriumWithB
        from struphy.kinetic_background import maxwellians

        pparams = {}

        # physics constants
        e = 1.602176634e-19  # elementary charge (C)
        m_p = 1.67262192369e-27  # proton mass (kg)
        mu0 = 1.25663706212e-6  # magnetic constant (N*A^-2)
        eps0 = 8.8541878128e-12  # vacuum permittivity (F*m^-1)
        kB = 1.380649e-23  # Boltzmann constant (J*K^-1)

        # exit when there is not any plasma species
        if len(self.fluid) == 0 and len(self.kinetic) == 0:
            return

        # compute model units
        units, equation_params = self.model_units(
            self.params,
            verbose=False,
            comm=self.comm_world,
        )

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

        # global parameters
        # plasma volume (hat x^3)
        det_tmp = self.domain.jacobian_det(eta1, eta2, eta3)
        vol1 = np.mean(np.abs(det_tmp))
        # plasma volume (m⁻³)
        plasma_volume = vol1 * units["x"] ** 3
        # transit length (m)
        transit_length = plasma_volume ** (1 / 3)
        # magnetic field (T)
        if isinstance(self.equil, FluidEquilibriumWithB):
            B_tmp = self.equil.absB0(eta1, eta2, eta3)
        else:
            B_tmp = np.zeros((eta1.size, eta2.size, eta3.size))
        magnetic_field = np.mean(B_tmp * np.abs(det_tmp)) / vol1 * units["B"]
        B_max = np.max(B_tmp) * units["B"]
        B_min = np.min(B_tmp) * units["B"]

        if magnetic_field < 1e-14:
            magnetic_field = np.nan
            # print("\n+++++++ WARNING +++++++ magnetic field is zero - set to nan !!")

        if verbose:
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

        # species dependent parameters
        pparams = {}

        if len(self.fluid) > 0:
            for species, val in self.fluid.items():
                pparams[species] = {}
                # type
                pparams[species]["type"] = "fluid"
                # mass (kg)
                pparams[species]["mass"] = val["params"]["phys_params"]["A"] * m_p
                # charge (C)
                pparams[species]["charge"] = val["params"]["phys_params"]["Z"] * e
                # density (m⁻³)
                pparams[species]["density"] = (
                    np.mean(
                        self.equil.n0(
                            eta1,
                            eta2,
                            eta3,
                        )
                        * np.abs(det_tmp),
                    )
                    * units["x"] ** 3
                    / plasma_volume
                    * units["n"]
                )
                # pressure (bar)
                pparams[species]["pressure"] = (
                    np.mean(
                        self.equil.p0(
                            eta1,
                            eta2,
                            eta3,
                        )
                        * np.abs(det_tmp),
                    )
                    * units["x"] ** 3
                    / plasma_volume
                    * units["p"]
                    * 1e-5
                )
                # thermal energy (keV)
                pparams[species]["kBT"] = pparams[species]["pressure"] * 1e5 / pparams[species]["density"] / e * 1e-3

        if len(self.kinetic) > 0:
            eta1mg, eta2mg, eta3mg = np.meshgrid(
                eta1,
                eta2,
                eta3,
                indexing="ij",
            )

            for species, val in self.kinetic.items():
                pparams[species] = {}
                # type
                pparams[species]["type"] = "kinetic"
                # mass (kg)
                pparams[species]["mass"] = val["params"]["phys_params"]["A"] * m_p
                # charge (C)
                pparams[species]["charge"] = val["params"]["phys_params"]["Z"] * e

                # create temp kinetic object for (default) parameter extraction
                tmp_bckgr = val["params"]["background"]

                if val["space"] != "ParticlesSPH":
                    tmp = None
                    for fi, maxw_params in tmp_bckgr.items():
                        if fi[-2] == "_":
                            fi_type = fi[:-2]
                        else:
                            fi_type = fi

                        if tmp is None:
                            tmp = getattr(maxwellians, fi_type)(
                                maxw_params=maxw_params,
                                equil=self.equil,
                            )
                        else:
                            tmp = tmp + getattr(maxwellians, fi_type)(
                                maxw_params=maxw_params,
                                equil=self.equil,
                            )

                if val["space"] != "ParticlesSPH" and tmp.coords == "constants_of_motion":
                    # call parameters
                    a1 = self.domain.params["a1"]
                    r = eta1mg * (1 - a1) + a1
                    psi = self.equil.psi_r(r)

                    # density (m⁻³)
                    pparams[species]["density"] = (
                        np.mean(tmp.n(psi) * np.abs(det_tmp)) * units["x"] ** 3 / plasma_volume * units["n"]
                    )
                    # thermal speed (m/s)
                    pparams[species]["v_th"] = (
                        np.mean(tmp.vth(psi) * np.abs(det_tmp)) * units["x"] ** 3 / plasma_volume * units["v"]
                    )
                    # thermal energy (keV)
                    pparams[species]["kBT"] = pparams[species]["mass"] * pparams[species]["v_th"] ** 2 / e * 1e-3
                    # pressure (bar)
                    pparams[species]["pressure"] = (
                        pparams[species]["kBT"] * e * 1e3 * pparams[species]["density"] * 1e-5
                    )

                else:
                    # density (m⁻³)
                    # pparams[species]['density'] = np.mean(tmp.n(
                    #     eta1mg, eta2mg, eta3mg) * np.abs(det_tmp)) * units['x']**3 / plasma_volume * units['n']
                    pparams[species]["density"] = 99.0
                    # thermal speeds (m/s)
                    vth = []
                    # vths = tmp.vth(eta1mg, eta2mg, eta3mg)
                    vths = [99.0]
                    for k in range(len(vths)):
                        vth += [
                            vths[k] * np.abs(det_tmp) * units["x"] ** 3 / plasma_volume * units["v"],
                        ]
                    thermal_speed = 0.0
                    for dir in range(val["obj"].vdim):
                        # pparams[species]['vth' + str(dir + 1)] = np.mean(vth[dir])
                        pparams[species]["vth" + str(dir + 1)] = 99.0
                        thermal_speed += pparams[species]["vth" + str(dir + 1)]
                    # TODO: here it is assumed that background density parameter is called "n",
                    # and that background thermal speeds are called "vthn"; make this a convention?
                    # pparams[species]['v_th'] = thermal_speed / \
                    #     val['obj'].vdim
                    pparams[species]["v_th"] = 99.0
                    # thermal energy (keV)
                    # pparams[species]['kBT'] = pparams[species]['mass'] * \
                    #     pparams[species]['v_th']**2 / e * 1e-3
                    pparams[species]["kBT"] = 99.0
                    # pressure (bar)
                    # pparams[species]['pressure'] = pparams[species]['kBT'] * \
                    #     e * 1e3 * pparams[species]['density'] * 1e-5
                    pparams[species]["pressure"] = 99.0

        for species in pparams:
            # alfvén speed (m/s)
            pparams[species]["v_A"] = magnetic_field / np.sqrt(
                mu0 * pparams[species]["mass"] * pparams[species]["density"],
            )
            # thermal speed (m/s)
            pparams[species]["v_th"] = np.sqrt(
                pparams[species]["kBT"] * 1e3 * e / pparams[species]["mass"],
            )
            # thermal frequency (Mrad/s)
            pparams[species]["Omega_th"] = pparams[species]["v_th"] / transit_length * 1e-6
            # cyclotron frequency (Mrad/s)
            pparams[species]["Omega_c"] = pparams[species]["charge"] * magnetic_field / pparams[species]["mass"] * 1e-6
            # plasma frequency (Mrad/s)
            pparams[species]["Omega_p"] = (
                np.sqrt(
                    pparams[species]["density"] * (pparams[species]["charge"]) ** 2 / eps0 / pparams[species]["mass"],
                )
                * 1e-6
            )
            # alfvén frequency (Mrad/s)
            pparams[species]["Omega_A"] = pparams[species]["v_A"] / transit_length * 1e-6
            # Larmor radius (m)
            pparams[species]["rho_th"] = pparams[species]["v_th"] / (pparams[species]["Omega_c"] * 1e6)
            # MHD length scale (m)
            pparams[species]["v_A/Omega_c"] = pparams[species]["v_A"] / (np.abs(pparams[species]["Omega_c"]) * 1e6)
            # dim-less ratios
            pparams[species]["rho_th/L"] = pparams[species]["rho_th"] / transit_length

        if verbose:
            print("\nSPECIES PARAMETERS:")
            for species, ch in pparams.items():
                print(f"\nname:".ljust(26), species)
                print(f"type:".ljust(25), ch["type"])
                ch.pop("type")
                print(f"is bulk:".ljust(25), species == self.bulk_species())
                for kinds, vals in ch.items():
                    print(
                        kinds.ljust(25),
                        "{:+4.3e}".format(
                            vals,
                        ),
                        units_affix[kinds],
                    )

        return pparams


class MyDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def ignore_aliases(self, data):
        return True
