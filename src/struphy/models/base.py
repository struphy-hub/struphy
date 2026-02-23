import os
from abc import ABCMeta, abstractmethod
from textwrap import indent

import cunumpy as xp
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile
from scope_profiler import ProfileManager

from struphy.io.options import LiteralOptions
from struphy.models.species import DiagnosticSpecies, FieldSpecies, FluidSpecies, ParticleSpecies, Species
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.physics.physics import Units
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.utils.clone_config import CloneConfig

class StruphyModel(metaclass=ABCMeta):
    """
    Abstract base class for all Struphy models.

    This class defines the interface for plasma simulation models in Struphy. Concrete implementations
    must specify the model type (Fluid, Kinetic, Hybrid, or Toy), define propagators for time integration,
    and configure species (field, fluid, particle, and diagnostic). The class provides core functionality
    for managing species, scalar quantities, time integration, and particle diagnostics.

    Attributes
    ----------
    species : dict
        Dictionary of all species (field, fluid, and particle) in the model.
    field_species : dict
        Dictionary of field species in the model.
    fluid_species : dict
        Dictionary of fluid species in the model.
    particle_species : dict
        Dictionary of particle species in the model.
    diagnostic_species : dict
        Dictionary of diagnostic species in the model.
    scalar_quantities : dict
        Dictionary of scalar quantities to be tracked and saved during simulation.
    prop_list : list
        List of propagator objects controlling time integration.
    clone_config : CloneConfig or None
        Configuration for domain clones if used in parallelization.

    Abstract Methods (must be implemented by subclasses)
    ----------------------------------------------------
    model_type : classmethod
        Must return one of "Fluid", "Kinetic", "Hybrid", or "Toy".
    bulk_species : property
        Must specify the dominant plasma species.
    velocity_scale : property
        Must return velocity scale: "alfvén", "cyclotron", "light", or "thermal".
    allocate_helpers : method
        Must allocate helper arrays and perform initial solves.
    update_scalar_quantities : method
        Must define update rules for each scalar quantity.
    Propagators : class
        Must define the propagators used for time integration.
    __init__ : method
        Must perform a light-weight initialization of the model.

    Notes
    -----
    All Struphy models must be subclasses of ``StruphyModel`` and should be added to ``struphy/models/``
    in one of the modules: ``fluid.py``, ``kinetic.py``, ``hybrid.py``, or ``toy.py``.

    Time integration is performed by calling the ``integrate()`` method with a time step and
    splitting algorithm (Lie-Trotter or Strang). The model manages the execution of all propagators
    in sequence to advance the simulation state.

    Species management is handled automatically through property caching. Species attributes are
    discovered at runtime and categorized by type.

    Examples
    --------
    Subclasses should implement:

    .. code-block:: python

        class MyFluidModel(StruphyModel):
            @classmethod
            def model_type(cls):
                return "Fluid"

            @property
            def bulk_species(self):
                return self.electrons

            @property
            def velocity_scale(self):
                return "thermal"

            def allocate_helpers(self, verbose=False):
                # Initialize helper arrays
                pass

            def update_scalar_quantities(self):
                # Update tracked scalars
                pass

            class Propagators:
                # Define propagators
                pass
    """

    # ----------------
    # Abstract methods
    # ----------------

    @classmethod
    @abstractmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        """Model type (Fluid, Kinetic, Hybrid, or Toy)"""
        pass

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
    def allocate_helpers(self, verbose: bool = False):
        """Allocate helper arrays and perform initial solves if needed."""

    @abstractmethod
    def update_scalar_quantities(self):
        """Specify an update rule for each item in ``scalar_quantities`` using :meth:`update_scalar`."""

    # --------------
    # Common methods
    # --------------
    def __repr__(self):
        out = f"{self.__class__.__name__}\n"
        for k, v in self.species.items():
            out += f"    {k}:\n"
            out += f"{v}"
        return out

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def add_scalar(self, name: str, variable: PICVariable | SPHVariable = None, compute=None, summands=None):
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
            assert isinstance(variable, (PICVariable, SPHVariable)), f"Variable is needed when {compute =}"

        if not hasattr(self, "_scalar_quantities"):
            self._scalar_quantities = {}

        self._scalar_quantities[name] = {
            "value": xp.empty(1, dtype=float),
            "variable": variable,
            "compute": compute,
            "summands": summands,
        }

    def update_scalar(self, name, value=None):
        """Update a scalar during the simulation.

        Parameters
        ----------
            name : str
                Dictionary key of the scalar.

            value : float, optional
                Value to be saved. Required if there are no summands.
        """

        # Ensure the name is a string
        assert isinstance(name, str)

        scalars = self.scalar_quantities

        variable: PICVariable | SPHVariable = scalars[name]["variable"]
        summands = scalars[name]["summands"]
        compute = scalars[name]["compute"]

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
            value_array = xp.array([value], dtype=xp.float64)

            # Perform MPI operations based on the compute flags
            if "sum_world" in compute_operations and not isinstance(MPI, MockMPI):
                MPI.COMM_WORLD.Allreduce(
                    MPI.IN_PLACE,
                    value_array,
                    op=MPI.SUM,
                )

            if "sum_within_clone" in compute_operations and Propagator.derham.comm is not None:
                Propagator.derham.comm.Allreduce(
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
                n_mks_tot = xp.array([variable.particles.Np])
                value_array /= n_mks_tot

            # Update the scalar value
            scalars[name]["value"][0] = value_array[0]

        else:
            # Sum the values of the summands
            value = sum(scalars[summand]["value"][0] for summand in summands)
            scalars[name]["value"][0] = value

    def print_scalar_quantities(self):
        """
        Check if scalar_quantities are not "nan" and print to screen.
        """
        sq_str = ""
        for key, scalar_dict in self._scalar_quantities.items():
            val = scalar_dict["value"]
            assert not xp.isnan(val[0]), f"Scalar {key} is {val[0]}."
            sq_str += f"{key}:".ljust(25) + "{:4.2e}\n".format(val[0]).rjust(26)
        print(sq_str)

    def scalar_quantities_to_file(self, time: float, filepath: str):
        if time == 0:
            with open(filepath, "w") as f:
                header = "time\t" + "\t".join(self._scalar_quantities.keys()) + "\n"
                f.write(header)
        with open(filepath, "a") as f:
            line = (
                f"{time:.12e}\t"
                + "\t".join(f"{scalar_dict['value'][0]:.12e}" for scalar_dict in self._scalar_quantities.values())
                + "\n"
            )
            f.write(line)

    def setup_equation_params(self, units: Units, verbose=False):
        """Set euqation parameters for each fluid and kinetic species."""
        for _, species in self.fluid_species.items():
            assert isinstance(species, FluidSpecies)
            species.setup_equation_params(units=units, verbose=verbose)

        for _, species in self.particle_species.items():
            assert isinstance(species, ParticleSpecies)
            species.setup_equation_params(units=units, verbose=verbose)

    @profile
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

                with ProfileManager.profile_region("prop: " + prop_name):
                    propagator(dt)

        # second order in time
        elif split_algo == "Strang":
            assert len(self.prop_list) > 1

            for propagator in self.prop_list[:-1]:
                prop_name = type(propagator).__name__
                with ProfileManager.profile_region("prop: " + prop_name):
                    propagator(dt / 2)

            propagator = self.prop_list[-1]
            prop_name = type(propagator).__name__
            with ProfileManager.profile_region("prop: " + prop_name):
                propagator(dt)

            for propagator in self.prop_list[:-1][::-1]:
                prop_name = type(propagator).__name__
                with ProfileManager.profile_region("prop: " + prop_name):
                    propagator(dt / 2)

        else:
            raise NotImplementedError(
                f"Splitting scheme {split_algo} not available.",
            )

    @profile
    def update_markers_to_be_saved(self):
        """
        Writes markers with IDs that are supposed to be saved into corresponding array.
        """

        for name, species in self.particle_species.items():
            assert isinstance(species, ParticleSpecies)
            assert len(species.variables) == 1, "More than 1 variable per kinetic species is not allowed."
            for _, var in species.variables.items():
                assert isinstance(var, PICVariable | SPHVariable)
                obj = var.particles
                assert isinstance(obj, Particles)

            if var.n_to_save > 0:
                markers_on_proc = xp.logical_and(
                    obj.markers[:, -1] >= 0.0,
                    obj.markers[:, -1] < var.n_to_save,
                )
                n_markers_on_proc = xp.count_nonzero(markers_on_proc)
                var.saved_markers[:] = -1.0
                var.saved_markers[:n_markers_on_proc] = obj.markers[markers_on_proc]

    @profile
    def update_distr_functions(self):
        """
        Writes distribution functions slices that are supposed to be saved into corresponding array.
        """

        dim_to_int = {"e1": 0, "e2": 1, "e3": 2, "v1": 3, "v2": 4, "v3": 5}

        for name, species in self.particle_species.items():
            assert isinstance(species, ParticleSpecies)
            assert len(species.variables) == 1, "More than 1 variable per kinetic species is not allowed."
            for _, var in species.variables.items():
                assert isinstance(var, PICVariable | SPHVariable)
                obj = var.particles
                assert isinstance(obj, Particles)

                if obj.n_cols_diagnostics > 0:
                    for i in range(obj.n_cols_diagnostics):
                        str_dn = f"d{i + 1}"
                        dim_to_int[str_dn] = 3 + obj.vdim + 3 + i

                for bin_plot in species.binning_plots:
                    comps = bin_plot.slice.split("_")
                    components = [False] * (3 + obj.vdim + 3 + obj.n_cols_diagnostics)

                    for comp in comps:
                        components[dim_to_int[comp]] = True

                    edges = bin_plot.bin_edges
                    binning_quantity = bin_plot.output_quantity
                    divide_by_jac = bin_plot.divide_by_jac
                    f_slice, df_slice = obj.binning(
                        components, edges, output_quantity=binning_quantity, divide_by_jac=divide_by_jac
                    )

                    bin_plot.f[:] = f_slice
                    bin_plot.df[:] = df_slice

                for kd_plot in species.kernel_density_plots:
                    h1 = 1 / obj.boxes_per_dim[0]
                    h2 = 1 / obj.boxes_per_dim[1]
                    h3 = 1 / obj.boxes_per_dim[2]

                    ndim = xp.count_nonzero([d > 1 for d in obj.boxes_per_dim])
                    if ndim == 0:
                        kernel_type = "gaussian_3d"
                    else:
                        kernel_type = "gaussian_" + str(ndim) + "d"

                    pts = kd_plot.plot_pts
                    n_sph = obj.eval_density(
                        *pts,
                        h1=h1,
                        h2=h2,
                        h3=h3,
                        kernel_type=kernel_type,
                        fast=True,
                    )
                    kd_plot.n_sph[:] = n_sph

<<<<<<< HEAD
    @classmethod
    def write_parameters_to_file(cls, parameters=None, file=None, save=True, prompt=True):
        import os

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

=======
>>>>>>> devel
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
            file = open(path, "x")
        except FileExistsError:
            if not prompt:
                yn = "Y"
            else:
                yn = input(f"\nFile {path} exists, overwrite (Y/n)? ")
            if yn in ("", "Y", "y", "yes", "Yes"):
                file = open(path, "w")
            else:
                print("exiting ...")
                exit()
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
                exit()

        # loop over species to create parameter snippets
        species_params = ""
        variables_params = ""
        particle_params = """\n# -------------------
# Particle parameters
# -------------------\n"""
        init_bckgr_pic = "\n# Background for kinetic species\n"
        has_feec = False
        has_pic = False
        has_sph = False
        for sn, species in self.species.items():
            assert isinstance(species, Species)
            species_params += f"model.{sn}.set_species_properties()\n"

            if isinstance(species, ParticleSpecies):
                particle_params += "\nloading_params = LoadingParameters()\n"
                particle_params += "weights_params = WeightsParameters()\n"
                particle_params += "boundary_params = BoundaryParameters()\n"
                particle_params += f"model.{sn}.set_markers(loading_params=loading_params,\n"
                txt = "weights_params=weights_params,\n"
                particle_params += indent(txt, " " * len(f"model.{sn}.set_markers("))
                txt = "boundary_params=boundary_params,\n"
                particle_params += indent(txt, " " * len(f"model.{sn}.set_markers("))
                txt = ")\n"
                particle_params += indent(txt, " " * len(f"model.{sn}.set_markers("))
                particle_params += f"model.{sn}.set_sorting_boxes()\n"
                particle_params += f"model.{sn}.set_save_data()\n"

            for vn, var in species.variables.items():
                variables_params += f"model.{sn}.{vn}.save_data = True\n"

                if isinstance(var, FEECVariable):
                    has_feec = True
                    init_bckgr_feec = "\n# Background for (some) FEEC variables\n"
                    init_pert_feec = "\n# Perturbations for (some) FEEC variables\n"
                    if var.space in ("H1", "L2"):
                        init_bckgr_feec += f"model.{sn}.{vn}.add_background(FieldsBackground())\n"
                        init_pert_feec += f"model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos())\n"
                    else:
                        init_bckgr_feec += f"model.{sn}.{vn}.add_background(FieldsBackground())\n"
                        init_pert_feec += (
                            f"model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=0))\n\
model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=1))\n\
model.{sn}.{vn}.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=2))\n"
                        )

                elif isinstance(var, PICVariable):
                    has_pic = True
                    init_pert_pic = "\n# Perturbations for (some) kinetic species\n"
                    init_pert_pic += "perturbation = perturbations.TorusModesCos()\n"
                    if "6D" in var.space:
                        init_bckgr_pic += "maxwellian_1 = maxwellians.Maxwellian3D(n=(1.0, None))\n"
                        init_bckgr_pic += "maxwellian_2 = maxwellians.Maxwellian3D(n=(0.1, None))\n"
                        init_pert_pic += "maxwellian_1pt = maxwellians.Maxwellian3D(n=(1.0, perturbation))\n"
                        init_pert_pic += "init = maxwellian_1pt + maxwellian_2\n"
                        init_pert_pic += f"model.{sn}.{vn}.add_initial_condition(init)\n"
                    elif "5D" in var.space:
                        init_bckgr_pic += "maxwellian_1 = maxwellians.GyroMaxwellian2D(n=(1.0, None), equil=equil)\n"
                        init_bckgr_pic += "maxwellian_2 = maxwellians.GyroMaxwellian2D(n=(0.1, None), equil=equil)\n"
                        init_pert_pic += (
                            "maxwellian_1pt = maxwellians.GyroMaxwellian2D(n=(1.0, perturbation), equil=equil)\n"
                        )
                        init_pert_pic += "init = maxwellian_1pt + maxwellian_2\n"
                        init_pert_pic += f"model.{sn}.{vn}.add_initial_condition(init)\n"
                    if "3D" in var.space:
                        init_bckgr_pic += "maxwellian_1 = maxwellians.ColdPlasma(n=(1.0, None))\n"
                        init_bckgr_pic += "maxwellian_2 = maxwellians.ColdPlasma(n=(0.1, None))\n"
                        init_pert_pic += "maxwellian_1pt = maxwellians.ColdPlasma(n=(1.0, perturbation))\n"
                        init_pert_pic += "init = maxwellian_1pt + maxwellian_2\n"
                        init_pert_pic += f"model.{sn}.{vn}.add_initial_condition(init)\n"
                    init_bckgr_pic += "background = maxwellian_1 + maxwellian_2\n"
                    init_bckgr_pic += f"model.{sn}.{vn}.add_background(background)\n"

                elif isinstance(var, SPHVariable):
                    has_sph = True
                    init_bckgr_sph = "\n# Background for (some) sph variables\n"
                    init_pert_sph = "\n# Perturbations for (some) sph variables\n"
                    init_bckgr_sph += "background = equils.ConstantVelocity()\n"
                    init_bckgr_sph += f"model.{sn}.{vn}.add_background(background)\n"
                    init_pert_sph += "perturbation = perturbations.TorusModesCos()\n"
                    init_pert_sph += f"model.{sn}.{vn}.add_perturbation(del_n=perturbation)\n"

        file.write(f"""# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation. 
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = f\"\"\"\nThis is the default simulation for the model {self.__class__.__name__}. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.\n\"\"\"\n""")

        file.write("""\n# ------------------
# Import Struphy API
# ------------------\n""")

        file.write("""\nfrom struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)\n""")

        if has_pic or has_sph:
            file.write("""\n# For particles:\nfrom struphy import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    WeightsParameters,
    maxwellians,
)\n""")

        file.write("""\n# ---------------------
# Instance of the model
# ---------------------\n""")

        file.write(f"\nfrom struphy.models import {self.__class__.__name__}\n")
        file.write(f"model = {self.__class__.__name__}()\n")

        file.write("\n# List all species and set their physical properties (charge and mass number, etc.)\n")
        file.write(species_params)

        file.write("\n# List all variables and decide whether to save their data\n")
        file.write(variables_params)

        file.write("""\n# --------------------------
# Instance of the simulation
# --------------------------\n""")

        file.write("\n# Environment options\n")
        file.write("env = EnvironmentOptions()\n")

        file.write("\n# Units\n")
        file.write("base_units = BaseUnits()\n")

        file.write("\n# Time stepping\n")
        file.write("time_opts = Time()\n")

        file.write("\n# Geometry\n")
        file.write("domain = domains.Cuboid()\n")

        file.write("\n# Fluid equilibrium (can be used as part of initial conditions)\n")
        file.write("equil = equils.HomogenSlab()\n")

        # if has_feec:
        grid = "grid = grids.TensorProductGrid()\n"
        derham = "derham_opts = DerhamOptions()\n"
        # else:
        #     grid = "grid = None\n"
        #     derham = "derham_opts = None\n"

        file.write("\n# Grid\n")
        file.write(grid)

        file.write("\n# Derham options\n")
        file.write(derham)

        file.write("\n# Simulation object\n")
        file.write("""sim = Simulation(
    model=model,
    params_path=__file__,
    env=env,
    base_units=base_units,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
)\n""")

        if has_pic or has_sph:
            file.write(particle_params)

        file.write("""\n# ------------------
# Propagator options
# ------------------\n\n""")
        for prop in self.propagators.__dict__:
            file.write(f"model.propagators.{prop}.options = model.propagators.{prop}.Options()\n")

        file.write("""\n# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.\n""")
        if has_feec:
            file.write(init_bckgr_feec)
            file.write(init_pert_feec)
        if has_pic:
            file.write("""
# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).\n""")
            file.write(init_bckgr_pic)
            file.write(init_pert_pic)
        if has_sph:
            file.write(init_bckgr_sph)
            file.write(init_pert_sph)

        file.write('\nif __name__ == "__main__":\n')
        file.write("    sim.run(verbose=False)")

        file.close()

        print(
            f"\nDefault parameter file for '{self.__class__.__name__}' has been created in the cwd ({path}).\n\
You can now launch a simulation with 'python params_{self.__class__.__name__}.py'",
        )

        return path

    # -------------
    # Model species
    # -------------

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
    def particle_species(self) -> dict:
        if not hasattr(self, "_particle_species"):
            self._particle_species = {}
            for k, v in self.__dict__.items():
                if isinstance(v, ParticleSpecies):
                    self._particle_species[k] = v
        return self._particle_species

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
            self._species = self.field_species | self.fluid_species | self.particle_species
        return self._species

    # -----------------
    # Common properties
    # -----------------

    @property
    def clone_config(self):
        """Config in case domain clones are used."""
        return self._clone_config

    @clone_config.setter
    def clone_config(self, new):
        assert isinstance(new, CloneConfig) or new is None
        self._clone_config = new

    @property
    def prop_list(self):
        """List of Propagator objects."""
        if not hasattr(self, "_prop_list"):
            self._prop_list = list(self.propagators.__dict__.values())
        return self._prop_list

    # @property
    # def prop_fields(self):
    #     """Module :mod:`struphy.propagators.propagators_fields`."""
    #     return self._prop_fields

    # @property
    # def prop_coupling(self):
    #     """Module :mod:`struphy.propagators.propagators_coupling`."""
    #     return self._prop_coupling

    # @property
    # def prop_markers(self):
    #     """Module :mod:`struphy.propagators.propagators_markers`."""
    #     return self._prop_markers

    # @property
    # def kwargs(self):
    #     """Dictionary holding the keyword arguments for each propagator specified in :attr:`~propagators_cls`.
    #     Keys must be the same as in :attr:`~propagators_cls`, values are dictionaries holding the keyword arguments."""
    #     return self._kwargs

    @property
    def scalar_quantities(self):
        """A dictionary of scalar quantities to be saved during the simulation."""
        if not hasattr(self, "_scalar_quantities"):
            self._scalar_quantities = {}
        return self._scalar_quantities

    # @property
    # def time_state(self):
    #     """A pointer to the time variable of the dynamics ('t')."""
    #     return self._time_state
