import logging
import os
from abc import ABCMeta, abstractmethod
from textwrap import indent

import cunumpy as xp
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI

try:
    from IPython.display import HTML, Markdown, display
except ImportError:

    def HTML(data):
        return data

    def Markdown(data):
        return data

    def display(*objects, **kwargs):
        return objects[0] if objects else None


from line_profiler import profile
from scope_profiler import ProfileManager

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.scalars import Scalars
from struphy.models.species import DiagnosticSpecies, FieldSpecies, FluidSpecies, ParticleSpecies, Species
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.physics.physics import Units
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.utils.clone_config import CloneConfig
from struphy.utils.docstring_converter import rst_to_html, rst_to_markdown
from struphy.utils.utils import all_class_params_are_default, all_subclasses

logger = logging.getLogger("struphy")


class StruphyModelMeta(ABCMeta):
    def __iter__(cls):
        return iter(all_subclasses(cls))


class StruphyModel(metaclass=StruphyModelMeta):
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
    scalars : Scalars or None
        Scalar quantities to be tracked and saved during simulation.
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
        pass

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

    # --------------
    # Common methods
    # --------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr_no_defaults__(self) -> str:
        return self.__repr__()

    @property
    def is_default(self):
        return all_class_params_are_default(self)

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        for k, v in self.species.items():
            out += f"    {k}:\n"
            out += f"{v}"
        return out

    forced_heading_level = 5

    @classmethod
    def info(cls):
        summary = (
            rst_to_html(cls.__doc__).split("Parameters")[0].split("<")[0]
            if cls.__doc__
            else """Description not available for this model."""
        )
        summary = " ".join(summary.split())
        doc = f"**{summary}**\n"
        doc += rf"""To see detailed information on the model, run the following methods:
        
.. code-block:: python

    {cls.name()}.pde()
    {cls.name()}.normalization()
    {cls.name()}.scalar_quantities()
    {cls.name()}.discretization()
    {cls.name()}.long_description()
    {cls.name()}.examples()
    {cls.name()}.use_cases()
    {cls.name()}.cannot_be_used_for()
"""
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def pde(cls):
        doc_pde = getattr(cls, "doc_pde", None)
        doc = doc_pde.__doc__ if doc_pde else """PDE description not available for this model."""
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def normalization(cls):
        doc_normalization = getattr(cls, "doc_normalization", None)
        doc = "**Normalization:**\n"
        doc += (
            doc_normalization.__doc__
            if doc_normalization
            else """Description of normalization not available for this model."""
        )
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def scalar_quantities(cls):
        doc_scalar_quantities = getattr(cls, "doc_scalar_quantities", None)
        doc = (
            doc_scalar_quantities.__doc__
            if doc_scalar_quantities
            else """Description of scalar quantities not available for this model."""
        )
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def discretization(cls):
        doc_discretization = getattr(cls, "doc_discretization", None)
        doc = "**Discretization (Propagators called in sequence):**\n"
        doc += (
            doc_discretization()
            if doc_discretization
            else """Description of discretization not available for this model."""
        )
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def long_description(cls):
        doc_long_description = getattr(cls, "doc_long_description", None)
        doc = "**Long description:**\n"
        doc += (
            doc_long_description.__doc__
            if doc_long_description
            else """Long description not available for this model."""
        )
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def examples(cls):
        doc_examples = getattr(cls, "doc_examples", None)
        doc = "**Examples:**\n"
        doc += doc_examples.__doc__ if doc_examples else """Examples not available for this model."""
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def use_cases(cls):
        doc_use_cases = getattr(cls, "doc_use_cases", None)
        doc = "**Use cases:**\n"
        doc += doc_use_cases.__doc__ if doc_use_cases else """Description of use cases not available for this model."""
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def cannot_be_used_for(cls):
        doc_cannot_be_used_for = getattr(cls, "doc_cannot_be_used_for", None)
        doc = "**Cannot be used for:**\n"
        doc += (
            doc_cannot_be_used_for.__doc__
            if doc_cannot_be_used_for
            else """Information on scenarios for which the model is not suitable is not available."""
        )
        return display(HTML(rst_to_html(doc, forced_heading_level=cls.forced_heading_level)))

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def create_doc(cls) -> "Documentation":
        doc = Documentation(cls)
        return doc

    @property
    def scalars(self) -> Scalars | None:
        """Scalars to be updated and saved during the simulation."""
        return getattr(self, "_scalars", Scalars())

    @scalars.setter
    def scalars(self, value: Scalars):
        assert isinstance(value, Scalars)
        self._scalars = value

    @profile
    def update_scalar_quantities(self):
        """Update scalar quantities by calling their .update() method.
        This should be called at the end of each time step in the simulation loop."""
        if self.scalars is not None:
            self.scalars.update()

    def print_scalar_quantities(self):
        """
        Check if scalars are not "nan" and print to screen.
        """
        sq_str = ""
        for key, scalar in self.scalars.dct.items():
            val = scalar.value[0]
            assert not xp.isnan(val), f"Scalar {key} is {val}."
            sq_str += f"{key}:".ljust(25) + "{:4.2e}\n".format(val).rjust(26)
        logger.info(sq_str)

    def setup_equation_params(self, base_units: BaseUnits, verbose=False):
        """Compute units and set equation parameters for each fluid and kinetic species."""
        self.base_units = base_units
        self.units = Units(base_units)

        if self.bulk_species is None:
            A_bulk = None
            Z_bulk = None
        else:
            A_bulk = self.bulk_species.mass_number
            Z_bulk = self.bulk_species.charge_number

        self.units.derive_units(
            velocity_scale=self.velocity_scale,
            A_bulk=A_bulk,
            Z_bulk=Z_bulk,
            verbose=verbose,
        )

        for _, species in self.fluid_species.items():
            assert isinstance(species, FluidSpecies)
            species.setup_equation_params(units=self.units, verbose=verbose)

        for _, species in self.particle_species.items():
            assert isinstance(species, ParticleSpecies)
            species.setup_equation_params(units=self.units, verbose=verbose)

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
                logger.info("exiting ...")
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
                logger.info("exiting ...")
                exit()

        # loop over species to create parameter snippets
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

name = \"Default {self.__class__.__name__}\"
description = \"\"\"\nThis is the default simulation for the model {self.__class__.__name__}. 
It is meant to be a template for users to set up their own simulations with this model. 
It contains all the necessary components of a Struphy simulation, including the model, 
the environment options, the time stepping options, the geometry, the equilibrium, 
the grid, the Derham options, and the initial conditions. 
Users can modify this file to set up their own simulations with different parameters and initial conditions.\n\"\"\"\n""")

        file.write("""
import logging
from struphy import set_logging_level
set_logging_level(logging.WARNING)\n""")

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

        file.write("\n# Units\n")
        file.write("base_units = BaseUnits()\n")

        file.write("\n# Model instance\n")
        file.write(f"model = {self.__class__.__name__}(base_units=base_units)\n")

        file.write("\n# List all variables and decide whether to save their data\n")
        file.write(variables_params)

        file.write("""\n# --------------------------
# Instance of the simulation
# --------------------------\n""")

        file.write("\n# Environment options\n")
        file.write("env = EnvironmentOptions()\n")

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
    name=name,
    description=description,
    params_path=__file__,
    env=env,
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

        logger.info(
            f"\nDefault parameter file for '{self.__class__.__name__}' has been created in the cwd ({path}).\n\
You can now launch a simulation with 'python params_{self.__class__.__name__}.py'",
        )

        return path

    def to_dict(self) -> dict:
        """Serialize the model configuration to a dictionary."""
        dct = {"model": self.__class__.__name__}
        return dct

    @classmethod
    def from_dict(cls, dct) -> "StruphyModel":
        """Deserialize a model configuration from a dictionary."""
        model_name = dct["model"]
        return cls.from_name(model_name)

    @classmethod
    def from_name(cls, name: str) -> "StruphyModel":
        """Instantiate a model from its name."""
        from struphy.models.utils import get_model_by_name

        model_cls = get_model_by_name(name)
        return model_cls()

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

    @property
    def base_units(self) -> BaseUnits:
        """Base units of the model."""
        return self._base_units

    @base_units.setter
    def base_units(self, new_units):
        assert isinstance(new_units, BaseUnits)
        self._base_units = new_units

    @property
    def units(self) -> Units:
        """Units of the model."""
        return self._units

    @units.setter
    def units(self, new_units):
        assert isinstance(new_units, Units)
        self._units = new_units


class Documentation:
    def __init__(
        self,
        cls: StruphyModel,
    ):
        self.description = self.Content(cls.__doc__ if cls.__doc__ else "Description not available for this model.")
        self.pde = self.Content(cls.doc_pde.__doc__ if cls.doc_pde else "PDE description not available for this model.")
        self.normalization = self.Content(
            cls.doc_normalization.__doc__
            if cls.doc_normalization
            else "Description of normalization not available for this model."
        )
        self.scalar_quantities = self.Content(
            cls.doc_scalar_quantities.__doc__
            if cls.doc_scalar_quantities
            else "Description of scalar quantities not available for this model."
        )
        self.discretization = self.Content(
            cls.doc_discretization()
            if cls.doc_discretization
            else "Description of discretization not available for this model."
        )
        self.long_description = self.Content(
            cls.doc_long_description.__doc__
            if cls.doc_long_description
            else "Long description not available for this model."
        )
        self.examples = self.Content(
            cls.doc_examples.__doc__ if cls.doc_examples else "Examples not available for this model."
        )
        self.use_cases = self.Content(
            cls.doc_use_cases.__doc__ if cls.doc_use_cases else "Use cases not available for this model."
        )
        self.cannot_be_used_for = self.Content(
            cls.doc_cannot_be_used_for.__doc__
            if cls.doc_cannot_be_used_for
            else "Information on scenarios for which the model is not suitable is not available."
        )
        self.model_properties = self.Content(
            f"**Model type:** {cls.model_type()}\n- **Velocity scale:** {cls.velocity_scale}\n- **Bulk species:** {cls.bulk_species}"
        )

    class Content:
        def __init__(self, rst, md=None, html=None):
            self.rst = rst
            self.md = md or rst_to_markdown(rst)
            self.html = html or rst_to_html(rst)
