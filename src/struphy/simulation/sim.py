# third party imports
import glob
import os
import pickle
import shutil
import sysconfig
import time

import cunumpy as xp
import h5py
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.stencil import StencilVector
from line_profiler import profile
from pyevtk.hl import gridToVTK
from scope_profiler import ProfileManager

# api imports
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    PlottingData,
    PostProcessor,
    Time,
    domains,
    equils,
    grids,
)

# core imports
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.fields_background.base import (
    FluidEquilibrium,
    FluidEquilibriumWithB,
    MHDequilibrium,
    NumericalMHDequilibrium,
)
from struphy.fields_background.projected_equils import (
    ProjectedFluidEquilibrium,
    ProjectedFluidEquilibriumWithB,
    ProjectedMHDequilibrium,
)
from struphy.geometry.base import Domain
from struphy.io.output_handling import DataContainer
from struphy.io.setup import setup_derham
from struphy.models import Maxwell
from struphy.models.base import StruphyModel
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
    Species,
)
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.physics.physics import Units
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.simulation.base import SimulationBase
from struphy.utils.clone_config import CloneConfig
from struphy.utils.utils import dict_to_yaml


class Simulation(SimulationBase):
    """Top-level class to configure and run a Struphy simulation.

    The `Simulation` class wraps model setup, MPI configuration, output
    management, normalization (units), memory allocation and time stepping.
    It initializes the model's variables and propagators, prepares runtime
    metadata and output folders, and provides the main `run()` entry point
    to execute the simulation.

    Parameters
    ----------
    model : StruphyModel
        Physics model that provides species, propagators and variables.
    params_path : str, optional
        Path to a Python parameter file to save alongside outputs.
    env : EnvironmentOptions
        Runtime and output environment options.
    base_units : BaseUnits
        Units used for normalization.
    time_opts : Time
        Time-stepping options (dt, Tend, split algorithm, ...).
    domain : Domain
        Computational domain description.
    equil : FluidEquilibrium, optional
        Initial fluid equilibrium (may be None).
    grid : TensorProductGrid
        Spatial grid used for FEEC variables.
    derham_opts : DerhamOptions
        Options for discrete differential operators.
    verbose : bool, optional
        If True, print additional setup information.

    Attributes
    ----------
    meta : dict
        Metadata about the run (platform, python version, model name, etc.).
    units : Units
        Unit/normalization helper created from `base_units`.
    data : DataContainer
        Output container used to store simulation data.
    start_time : float
        Wall-clock time when the simulation object was created.
    """

    def __init__(
        self,
        model: StruphyModel,
        params_path: str = None,
        env: EnvironmentOptions = EnvironmentOptions(),
        base_units: BaseUnits = BaseUnits(),
        time_opts: Time = Time(),
        domain: Domain = domains.Cuboid(),
        equil: FluidEquilibrium = None,
        grid: grids.TensorProductGrid = grids.TensorProductGrid(),
        derham_opts: DerhamOptions = DerhamOptions(),
        verbose: bool = False,
    ):

        self._model = model
        self._params_path = params_path
        self._env = env
        self._base_units = base_units
        self._time_opts = time_opts
        self._setup_domain_and_equil(domain, equil, verbose=verbose)
        self._grid = grid
        self._derham_opts = derham_opts

        # setup profiling agent
        ProfileManager.setup(
            profiling_activated=env.profiling_activated,
            time_trace=env.profiling_trace,
            use_likwid=False,
            file_path=os.path.join(
                env.out_folders,
                env.sim_folder,
                "profiling_data.h5",
            ),
        )

        # mpi info
        if isinstance(MPI, MockMPI):
            self.comm = None
            self.rank = 0
            self.comm_size = 1
            self.Barrier = lambda: None
        else:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            self.Barrier = self.comm.Barrier

        if self.rank == 0:
            print("")
            if verbose:
                self.show_parameters()

        # synchronize MPI processes to set same start time of simulation for all processes
        self.Barrier()
        self.start_time = time.time()

        # check model
        assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
        self.model_name = model.__class__.__name__

        if self.rank == 0:
            print(f"Instance of simulation for model {self.model_name} ...")

        # meta-data
        path_out = env.path_out
        restart = env.restart
        max_runtime = env.max_runtime
        save_step = env.save_step
        sort_step = env.sort_step
        num_clones = env.num_clones
        use_mpi = (self.comm is not None,)

        self.meta = {}
        self.meta["platform"] = sysconfig.get_platform()
        self.meta["python version"] = sysconfig.get_python_version()
        self.meta["model name"] = self.model_name
        self.meta["parameter file"] = self.params_path
        self.meta["output folder"] = path_out
        self.meta["MPI processes"] = self.comm_size
        self.meta["use MPI.COMM_WORLD"] = use_mpi
        self.meta["number of domain clones"] = num_clones
        self.meta["restart"] = restart
        self.meta["max wall-clock [min]"] = max_runtime
        self.meta["save interval [steps]"] = save_step

        if self.rank == 0:
            print("\nMETADATA:")
            for k, v in self.meta.items():
                print(f"{k}:".ljust(25), v)

        # creating output folders
        self._setup_folders(
            verbose=verbose,
        )

        # save parameter file
        if self.rank == 0:
            # save python param file
            if self.params_path is not None:
                assert self.params_path[-3:] == ".py"
                try:
                    shutil.copy2(
                        self.params_path,
                        os.path.join(path_out, "parameters.py"),
                    )
                except shutil.SameFileError:
                    pass
            # pickle struphy objects
            else:
                with open(os.path.join(path_out, "env.bin"), "wb") as f:
                    pickle.dump(env, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "base_units.bin"), "wb") as f:
                    pickle.dump(base_units, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "time_opts.bin"), "wb") as f:
                    pickle.dump(time_opts, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "domain.bin"), "wb") as f:
                    # WORKAROUND: cannot pickle pyccelized classes at the moment
                    tmp_dct = {"name": domain.__class__.__name__, "params": domain.params}
                    pickle.dump(tmp_dct, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "equil.bin"), "wb") as f:
                    # WORKAROUND: cannot pickle pyccelized classes at the moment
                    if equil is not None:
                        tmp_dct = {"name": equil.__class__.__name__, "params": equil.params}
                    else:
                        tmp_dct = {}
                    pickle.dump(tmp_dct, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "grid.bin"), "wb") as f:
                    pickle.dump(grid, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "derham_opts.bin"), "wb") as f:
                    pickle.dump(derham_opts, f, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(path_out, "model_class.bin"), "wb") as f:
                    pickle.dump(model.__class__, f, pickle.HIGHEST_PROTOCOL)

        # config clones
        if self.comm is None:
            clone_config = None
        else:
            if num_clones == 1:
                clone_config = None
            else:
                # Setup domain cloning communicators
                # MPI.COMM_WORLD     : comm
                # within a clone:    : sub_comm
                # between the clones : inter_comm
                clone_config = CloneConfig(comm=self.comm, params=None, num_clones=num_clones)
                clone_config.print_clone_config()
                if model.particle_species:
                    clone_config.print_particle_config()

        self.clone_config = model.clone_config = clone_config
        self.Barrier()

        # units and normalization parameters
        self.units = Units(base_units)
        self.normalize_model()

        if self.rank == 0:
            print("\n... Done.")

    # ----------------
    # Abstract methods
    # ----------------

    def show_parameters(self):
        """Print the current simulation configuration to stdout.

        Only the MPI rank 0 prints to avoid clutter from multiple processes.
        """
        if self.rank == 0:
            print("SIMULATION PARAMETERS:")
            print("\nModel:")
            print(self.model)
            print("Parameter file path:")
            print(self.params_path)
            print("\nEnvironment options:")
            print(self.env)
            print("Base units:")
            print(self.base_units)
            print("Time stepping options:")
            print(self.time_opts)
            print("Domain:")
            print(self.domain)
            print("Fluid equilibrium:")
            print(self.equil)
            print("Grid:")
            print(self.grid)
            print("Derham options:")
            print(self.derham_opts)
            print("")

    def allocate(self, verbose: bool = False):
        """Allocate FEEC structures, model variables and propagators.

        This prepares FEEC operators, allocates variable storage for all
        species (fields, fluids, particles) and passes allocation info to
        propagators. Prints progress on MPI rank 0.
        """

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nAllocating simulation data ...")

        # feec
        self._allocate_feec(self.grid, self.derham_opts, verbose=verbose)

        # allocate model variables
        self._allocate_variables(verbose=verbose)

        # pass info to propagators
        self._allocate_propagators(verbose=verbose)

        # allocate helper fields and perform initial solves if needed
        self.model.allocate_helpers(verbose=verbose)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("... Done.")

    def save_geometry_and_equil_vtk(self, verbose: bool = False):
        """Write a VTK file with geometry and (projected) equilibrium fields.

        Only executed on MPI rank 0. Outputs basic diagnostic fields such as
        jacobian determinant, pressure and |B| when available.
        """
        # store geometry vtk
        if self.rank == 0:
            grids_log = [
                xp.linspace(1e-6, 1.0, 32),
                xp.linspace(0.0, 1.0, 32),
                xp.linspace(0.0, 1.0, 32),
            ]

            tmp = self.domain(*grids_log)
            grids_phy = [tmp[0], tmp[1], tmp[2]]

            pointData = {}
            det_df = self.domain.jacobian_det(*grids_log)
            pointData["det_df"] = det_df

            if self.equil is not None:
                p0 = self.equil.p0(*grids_log)
                pointData["p0"] = p0
                if isinstance(self.equil, FluidEquilibriumWithB):
                    absB0 = self.equil.absB0(*grids_log)
                    pointData["absB0"] = absB0

            gridToVTK(os.path.join(self.env.path_out, "geometry"), *grids_phy, pointData=pointData)

    def initialize_data_storage(self, verbose: bool = False):
        """Create the `DataContainer` and register time datasets.

        Initializes `time_state` arrays (normalized and physical time and
        index) and registers them with the output `DataContainer` so they
        are saved during the run (and on restart).
        """

        # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
        # use MPI.COMM_WORLD as communicator when storing the outputs

        self.data = DataContainer(self.env.path_out, comm=self.comm)

        # time quantities (current time value, value in seconds and index)
        self.time_state = {}
        self.time_state["value"] = xp.zeros(1, dtype=float)
        self.time_state["value_sec"] = xp.zeros(1, dtype=float)
        self.time_state["index"] = xp.zeros(1, dtype=int)

        # add time quantities to data object for saving
        for key, val in self.time_state.items():
            key_time = "time/" + key
            key_time_restart = "restart/time/" + key
            self.data.add_data({key_time: val})
            self.data.add_data({key_time_restart: val})

    def run(self, verbose: bool = False):
        """Main entry point to execute the simulation time loop.

        Responsibilities include allocation (when not restarting),
        initialization of output storage, handling restarts, running the
        main time-stepping loop, saving data at intervals, and finalizing
        profiling and metadata. Prints progress on MPI rank 0.

        Parameters
        ----------
        verbose : bool
            If True, print additional runtime information.
        """

        if self.rank == 0:
            print(f"\nStarting simulation run for model {self.model_name} ...")

        self._remove_existing_output_files(verbose=verbose)

        # Display propagator options and intial conditions:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nPROPAGATOR OPTIONS:")
            for prop in self.model.prop_list:
                assert isinstance(prop, Propagator)
                prop.show_options()

            print("\nINITIAL CONDITIONS:")
            for species in self.model.species.values():
                assert isinstance(species, Species)
                for variable in species.variables.values():
                    if isinstance(variable, FEECVariable) or isinstance(variable, SPHVariable):
                        variable.show_backgrounds()
                        variable.show_perturbations()
                    elif isinstance(variable, PICVariable):
                        variable.show_backgrounds()
                        variable.show_perturbations()
                        variable.show_initial_condition()

        if not self.env.restart:
            # equation paramters
            self.allocate(verbose=verbose)

            # output
            self.initialize_data_storage(verbose=verbose)

            # peek view into geometry
            self.save_geometry_and_equil_vtk(verbose=verbose)

            # plasma parameters
            self.compute_plasma_params(verbose=verbose)

        # print info on mpi procs
        if self.rank < 32:
            if self.rank == 0:
                print("")
            print(f"Rank {self.rank}: executing run() for model {self.model_name} ...")

        if self.comm_size > 32 and self.rank == 32:
            print(f"Ranks > 31: executing run() for model {self.model_name} ...")

        # retrieve time parameters
        dt = self.time_opts.dt
        Tend = self.time_opts.Tend
        split_algo = self.time_opts.split_algo

        # set initial conditions for all variables
        if self.env.restart:
            self._initialize_from_restart(self.data)

            with h5py.File(self.data.file_path, "a") as file:
                self.time_state["value"][0] = file["restart/time/value"][-1]
                self.time_state["value_sec"][0] = file["restart/time/value_sec"][-1]
                self.time_state["index"][0] = file["restart/time/index"][-1]

            total_steps = str(int(round((Tend - self.time_state["value"][0]) / dt)))
            print(f"""\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
RESTARTing from:
{self.time_state["value"][0]=}
{self.time_state["value_sec"][0]=}
{self.time_state["index"][0]=}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""")
        else:
            total_steps = str(int(round(Tend / dt)))

        # compute initial scalars and kinetic data, pass time state to all propagators
        self.model.update_scalar_quantities()
        self.model.update_markers_to_be_saved()
        self.model.update_distr_functions()
        self._add_time_state(self.time_state["value"])

        # add all variables to be saved to data object
        save_keys_all, save_keys_end = self._initialize_hdf5_datasets(self.data, self.comm_size)

        # ======================== main time loop ======================
        self.model.update_scalar_quantities()
        if self.rank == 0:
            print("\nINITIAL SCALAR QUANTITIES:")
            self.model.print_scalar_quantities()

            print(f"\nSTART TIME STEPPING WITH '{split_algo}' SPLITTING:")

        # time loop
        run_time_now = 0.0
        while True:
            self.Barrier()

            # stop time loop?
            break_cond_1 = self.time_state["value"][0] >= Tend
            break_cond_2 = run_time_now > self.env.max_runtime

            if break_cond_1 or break_cond_2:
                # save restart data (other data already saved below)
                self.data.save_data(keys=save_keys_end)
                end_time = time.time()
                if self.rank == 0:
                    print(f"\nTime steps done: {self.time_state['index'][0]}")
                    print(
                        "wall-clock time of simulation [sec]: ",
                        end_time - self.start_time,
                    )
                    print()
                break

            if self.env.sort_step and self.time_state["index"][0] % self.env.sort_step == 0:
                t0 = time.time()
                for key, val in self.model.pointer.items():
                    if isinstance(val, Particles):
                        val.do_sort()
                t1 = time.time()
                if self.rank == 0 and verbose:
                    message = "Particles sorted | wall clock [s]: {0:8.4f} | sorting duration [s]: {1:8.4f}".format(
                        run_time_now * 60,
                        t1 - t0,
                    )
                    print(message, end="\n")
                    print()

            # update time and index (round time to 10 decimals for a clean time grid!)
            self.time_state["value"][0] = round(self.time_state["value"][0] + dt, 14)
            self.time_state["value_sec"][0] = round(self.time_state["value_sec"][0] + dt * self.units.t, 14)
            self.time_state["index"][0] += 1

            # perform one time step dt
            t0 = time.time()
            with ProfileManager.profile_region("model.integrate"):
                self.model.integrate(dt, split_algo)
            t1 = time.time()

            run_time_now = (time.time() - self.start_time) / 60

            # update diagnostics data and save data
            if self.time_state["index"][0] % self.env.save_step == 0:
                # compute scalars and kinetic data
                self.model.update_scalar_quantities()
                self.model.update_markers_to_be_saved()
                self.model.update_distr_functions()

                # extract FEEC coefficients
                feec_species = self.model.field_species | self.model.fluid_species | self.model.diagnostic_species
                for species, val in feec_species.items():
                    assert isinstance(val, Species)
                    for variable, subval in val.variables.items():
                        assert isinstance(subval, FEECVariable)
                        spline = subval.spline
                        # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                        spline.extract_coeffs(update_ghost_regions=False)

                # save data (everything but restart data)
                self.data.save_data(keys=save_keys_all)

                # print current time and scalar quantities to screen
                if self.rank == 0 and verbose:
                    step = str(self.time_state["index"][0]).zfill(len(total_steps))

                    message = "time step:".ljust(25) + f"{step}/{total_steps}".rjust(25)
                    message += (
                        "\n"
                        + "normalized time:".ljust(25)
                        + "{0:4.2e} / {1:4.2e}".format(self.time_state["value"][0], Tend).rjust(25)
                    )
                    message += (
                        "\n"
                        + "physical time [s]:".ljust(25)
                        + "{0:4.2e} / {1:4.2e}".format(
                            self.time_state["value_sec"][0],
                            Tend * self.units.t,
                        ).rjust(25)
                    )
                    message += "\n" + "wall clock time [s]:".ljust(25) + "{0:8.4f}".format(run_time_now * 60).rjust(25)
                    message += "\n" + "last step duration [s]:".ljust(25) + "{0:8.4f}".format(t1 - t0).rjust(25)

                    print(message)
                    self.model.print_scalar_quantities()

        # ===================================================================

        self.meta["wall-clock time[min]"] = (end_time - self.start_time) / 60
        self.Barrier()

        if self.rank == 0:
            # save meta-data
            dict_to_yaml(self.meta, os.path.join(self.env.path_out, "meta.yml"))
            print("Struphy run finished.")

        if self.clone_config is not None:
            self.clone_config.free()

        ProfileManager.finalize()

    def pproc(
        self,
        step: int = 1,
        celldivide: int = 1,
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        time_trace: bool = False,
        verbose: bool = False,
    ):
        """Run post-processing on saved simulation data.

        Uses `PostProcessor` to generate plots, process guiding-center or
        physical field views, and optionally produce VTK outputs.
        """

        # setup post processor and plotting
        if not hasattr(self, "_post_processor") and self.rank == 0:
            self._post_processor = PostProcessor(sim=self)

        if time_trace:
            self.post_processor.plot_time_traces(verbose=verbose)

        self.post_processor.process(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
            verbose=verbose,
        )

    def load_plotting_data(self, verbose: bool = False):
        """Load plotting datasets produced by post-processing.

        Creates a `PlottingData` instance on rank 0 (if needed), loads the
        data and exposes convenient attributes such as `orbits`, `f`, and
        grid information for downstream plotting or analysis.
        """

        if not hasattr(self, "_plotting_data") and self.rank == 0:
            self._plotting_data = PlottingData(sim=self)
        self.plotting_data.load(verbose=verbose)

        # expose attributes
        self.orbits = self.plotting_data.orbits
        self.f = self.plotting_data.f
        self.spline_values = self.plotting_data.spline_values
        self.n_sph = self.plotting_data.n_sph
        self.grids_log = self.plotting_data.grids_log
        self.grids_phy = self.plotting_data.grids_phy
        self.t_grid = self.plotting_data.t_grid

    # ---------------------
    # Code specific methods
    # ---------------------
    def normalize_model(self, verbose: bool = False):
        """Compute derived units and normalization coefficients of equations.
        Must be re-run when species properties have been changed.
        """
        if self.model.bulk_species is None:
            A_bulk = None
            Z_bulk = None
        else:
            A_bulk = self.model.bulk_species.mass_number
            Z_bulk = self.model.bulk_species.charge_number
        self.units.derive_units(
            velocity_scale=self.model.velocity_scale,
            A_bulk=A_bulk,
            Z_bulk=Z_bulk,
            verbose=verbose,
        )
        self.model.setup_equation_params(units=self.units, verbose=verbose)

    def compute_plasma_params(self, verbose: bool = True):
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
        eta1 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta2 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta3 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)

        # global parameters

        # plasma volume (hat x^3)
        det_tmp = self.domain.jacobian_det(eta1, eta2, eta3)
        vol1 = xp.mean(xp.abs(det_tmp))
        # plasma volume (m⁻³)
        plasma_volume = vol1 * self.units.x**3
        # transit length (m)
        transit_length = plasma_volume ** (1 / 3)
        # magnetic field (T)
        if isinstance(self.equil, FluidEquilibriumWithB):
            B_tmp = self.equil.absB0(eta1, eta2, eta3)
        else:
            B_tmp = xp.zeros((eta1.size, eta2.size, eta3.size))
        magnetic_field = xp.mean(B_tmp * xp.abs(det_tmp)) / vol1 * self.units.B
        B_max = xp.max(B_tmp) * self.units.B
        B_min = xp.min(B_tmp) * self.units.B

        if magnetic_field < 1e-14:
            magnetic_field = xp.nan
            # print("\n+++++++ WARNING +++++++ magnetic field is zero - set to nan !!")

        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print("\nPLASMA PARAMETERS:")
            print(
                "Plasma volume:".ljust(25),
                "{:4.3e}".format(plasma_volume) + units_affix["plasma volume"],
            )
            print(
                "Transit length:".ljust(25),
                "{:4.3e}".format(transit_length) + units_affix["transit length"],
            )
            print(
                "Avg. magnetic field:".ljust(25),
                "{:4.3e}".format(magnetic_field) + units_affix["magnetic field"],
            )
            print(
                "Max magnetic field:".ljust(25),
                "{:4.3e}".format(B_max) + units_affix["magnetic field"],
            )
            print(
                "Min magnetic field:".ljust(25),
                "{:4.3e}".format(B_min) + units_affix["magnetic field"],
            )

    def spawn_sister(
        self,
        model: StruphyModel = None,
        params_path: str = None,
        env: EnvironmentOptions = None,
        base_units: BaseUnits = None,
        time_opts: Time = None,
        domain: Domain = None,
        equil: FluidEquilibrium = None,
        grid: grids.TensorProductGrid = None,
        derham_opts: DerhamOptions = None,
        verbose: bool = False,
    ):
        """Spawn a sister simulation with parameters that default to the current instance.
        This can be used to quickly generate multiple similar simulations."""
        if model is None:
            model = self.model
        if params_path is None:
            params_path = self.params_path
        if env is None:
            env = self.env
        if base_units is None:
            base_units = self.base_units
        if time_opts is None:
            time_opts = self.time_opts
        if domain is None:
            domain = self.domain
        if equil is None:
            equil = self.equil
        if grid is None:
            grid = self.grid
        if derham_opts is None:
            derham_opts = self.derham_opts

        sister = Simulation(
            model=model,
            params_path=params_path,
            env=env,
            base_units=base_units,
            time_opts=time_opts,
            domain=domain,
            equil=equil,
            grid=grid,
            derham_opts=derham_opts,
            verbose=verbose,
        )
        return sister

    # ---------------
    # Private methods
    # ---------------

    def _setup_folders(self, verbose: bool = False):
        """
        Setup output folders.
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            # create output folder if it does not exit
            if not os.path.exists(self.env.path_out):
                os.makedirs(self.env.path_out, exist_ok=True)
                if verbose:
                    print("Created folder " + self.env.path_out)

            # create data folder in output folder if it does not exist
            if not os.path.exists(os.path.join(self.env.path_out, "data/")):
                os.mkdir(os.path.join(self.env.path_out, "data/"))
                if verbose:
                    print("Created folder " + os.path.join(self.env.path_out, "data/"))

    def _remove_existing_output_files(self, verbose: bool = False):
        """Removes post_processing/, meta.txt and profile_tmp.
        If not restart, also removes existing hdf5 and png files in output folder."""
        if MPI.COMM_WORLD.Get_rank() == 0:
            # remove post_processing folder
            folder = os.path.join(self.env.path_out, "post_processing")
            if os.path.exists(folder):
                shutil.rmtree(folder)
                if verbose:
                    print("Removed existing folder " + folder)

            # remove meta file
            file = os.path.join(self.env.path_out, "meta.txt")
            if os.path.exists(file):
                os.remove(file)
                if verbose:
                    print("Removed existing file " + file)

            # remove profiling file
            file = os.path.join(self.env.path_out, "profile_tmp")
            if os.path.exists(file):
                os.remove(file)
                if verbose:
                    print("Removed existing file " + file)

            # remove hdf5 and png files (if NOT a restart)
            if not self.env.restart:
                files = glob.glob(os.path.join(self.env.path_out, "data", "*.hdf5"))
                for n, file in enumerate(files):
                    os.remove(file)
                    if verbose and n < 10:  # print only ten statements in case of many processes
                        print("Removed existing file " + file)

                files = glob.glob(os.path.join(self.env.path_out, "*.png"))
                for n, file in enumerate(files):
                    os.remove(file)
                    if verbose and n < 10:  # print only ten statements in case of many processes
                        print("Removed existing file " + file)

    def _setup_domain_and_equil(self, domain: Domain, equil: FluidEquilibrium, verbose: bool = False):
        """If a numerical equilibirum is used, the domain is taken from this equilibirum."""
        if equil is not None:
            if isinstance(equil, NumericalMHDequilibrium):
                self._domain = equil.domain
            else:
                self._domain = domain
                equil.domain = domain

            if hasattr(equil, "units"):
                assert isinstance(equil.units, Units)
                equil.units.derive_units(
                    velocity_scale=self.model.velocity_scale,
                    A_bulk=self.model.bulk_species.mass_number,
                    Z_bulk=self.model.bulk_species.charge_number,
                    verbose=verbose,
                )

        else:
            self._domain = domain

        self._equil = equil

        # if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
        #     print("\nDOMAIN:")
        #     print("type:".ljust(25), self.domain.__class__.__name__)
        #     for key, val in self.domain.params.items():
        #         if key not in {"cx", "cy", "cz"}:
        #             print((key + ":").ljust(25), val)

        #     print("\nFLUID BACKGROUND:")
        #     if self.equil is not None:
        #         print("type:".ljust(25), self.equil.__class__.__name__)
        #         for key, val in self.equil.params.items():
        #             print((key + ":").ljust(25), val)
        #     else:
        #         print("None.")

    @profile
    def _allocate_feec(self, grid: grids.TensorProductGrid, derham_opts: DerhamOptions, verbose: bool = False):
        """Create the discrete Derham sequence, mass/basis operators and projected equilibrium.

        This sets up the 3D Derham object (unless grid or derham_opts are
        None), creates weighted mass and basis projection operators, and
        constructs a projected equilibrium appropriate for the chosen
        equilibrium type.
        """

        # create discrete derham sequence
        if self.clone_config is None:
            derham_comm = MPI.COMM_WORLD
        else:
            derham_comm = self.clone_config.sub_comm

        if grid is None or derham_opts is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"\n{grid=}, {derham_opts=}: no Derham object set up.")
            self._derham = None
        else:
            self._derham = setup_derham(
                grid,
                derham_opts,
                comm=derham_comm,
                domain=self.domain,
                verbose=verbose,
            )

        # create weighted mass and basis operators
        if self.derham is None:
            self._mass_ops = None
            self._basis_ops = None
        else:
            self._mass_ops = WeightedMassOperators(self.derham, self.domain, eq_mhd=self.equil, verbose=verbose)

            self._basis_ops = BasisProjectionOperators(
                self.derham,
                self.domain,
                eq_mhd=self.equil,
                verbose=verbose,
            )

        # create projected equilibrium
        if self.derham is None:
            self._projected_equil = None
        else:
            if isinstance(self.equil, MHDequilibrium):
                self._projected_equil = ProjectedMHDequilibrium(
                    self.equil,
                    self.derham,
                    verbose=verbose,
                )
            elif isinstance(self.equil, FluidEquilibriumWithB):
                self._projected_equil = ProjectedFluidEquilibriumWithB(
                    self.equil,
                    self.derham,
                    verbose=verbose,
                )
            elif isinstance(self.equil, FluidEquilibrium):
                self._projected_equil = ProjectedFluidEquilibrium(
                    self.equil,
                    self.derham,
                    verbose=verbose,
                )
            else:
                self._projected_equil = None

    @profile
    def _allocate_variables(self, verbose: bool = False):
        """
        Allocate memory for model variables and set initial conditions.
        """
        # allocate memory for FE coeffs of electromagnetic fields/potentials
        if self.model.field_species:
            for species, spec in self.model.field_species.items():
                assert isinstance(spec, FieldSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, FEECVariable)
                    v.allocate(
                        derham=self.derham,
                        domain=self.domain,
                        equil=self.equil,
                        verbose=verbose,
                    )

        # allocate memory for FE coeffs of fluid variables
        if self.model.fluid_species:
            for species, spec in self.model.fluid_species.items():
                assert isinstance(spec, FluidSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, FEECVariable)
                    v.allocate(
                        derham=self.derham,
                        domain=self.domain,
                        equil=self.equil,
                        verbose=verbose,
                    )

        # allocate memory for marker arrays of kinetic variables
        if self.model.particle_species:
            for species, spec in self.model.particle_species.items():
                assert isinstance(spec, ParticleSpecies)
                for k, v in spec.variables.items():
                    if isinstance(v, PICVariable):
                        v.allocate(
                            clone_config=self.clone_config,
                            derham=self.derham,
                            domain=self.domain,
                            equil=self.equil,
                            projected_equil=self.projected_equil,
                            verbose=verbose,
                        )
                    if isinstance(v, SPHVariable):
                        v.allocate(
                            derham=self.derham,
                            domain=self.domain,
                            equil=self.equil,
                            projected_equil=self.projected_equil,
                            verbose=verbose,
                        )

        # allocate memory for FE coeffs of fluid variables
        if self.model.diagnostic_species:
            for species, spec in self.model.diagnostic_species.items():
                assert isinstance(spec, DiagnosticSpecies)
                for k, v in spec.variables.items():
                    assert isinstance(v, FEECVariable)
                    v.allocate(
                        derham=self.derham,
                        domain=self.domain,
                        equil=self.equil,
                        verbose=verbose,
                    )

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

    @profile
    def _allocate_propagators(self, verbose: bool = False):
        """Allocate propagators and bind shared FEEC/domain operators.

        Assigns `derham`, `domain`, `mass_ops`, `basis_ops` and
        `projected_equil` on the `Propagator` base class so individual
        propagator instances can access shared resources, then calls each
        propagator's `allocate` method.
        """

        # set propagators base class attributes (then available to all propagators)
        Propagator.derham = self.derham
        Propagator.domain = self.domain
        if self.derham is not None:
            Propagator.mass_ops = self.mass_ops
            Propagator.basis_ops = self.basis_ops
            Propagator.projected_equil = self.projected_equil

        assert len(self.model.prop_list) > 0, "No propagators in this model, check the model class."
        for prop in self.model.prop_list:
            assert isinstance(prop, Propagator)
            prop.allocate(verbose=verbose)
            if verbose and MPI.COMM_WORLD.Get_rank() == 0:
                print(f"\nAllocated propagator '{prop.__class__.__name__}'.")

    @profile
    def _initialize_hdf5_datasets(self, data: DataContainer, size, verbose: bool = False):
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
        for key, scalar in self.model.scalar_quantities.items():
            val = scalar["value"]
            key_scalar = "scalar/" + key
            data.add_data({key_scalar: val})

        with h5py.File(data.file_path, "a") as file:
            # store grid_info only for runs with 512 ranks or smaller
            if self.model.scalar_quantities and self.derham is not None:
                if size <= 512:
                    file["scalar"].attrs["grid_info"] = self.derham.domain_array
                else:
                    file["scalar"].attrs["grid_info"] = self.derham.domain_array[0]
            else:
                pass

            # save feec data in group 'feec/'
            feec_species = self.model.field_species | self.model.fluid_species | self.model.diagnostic_species
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
                        file[key_field].attrs["space_id"] = spline.space_id
                        file[key_field].attrs["starts"] = spline.starts
                        file[key_field].attrs["ends"] = spline.ends
                        file[key_field].attrs["pads"] = spline.pads

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
            for name, species in self.model.particle_species.items():
                assert isinstance(species, ParticleSpecies)
                assert len(species.variables) == 1, "More than 1 variable per kinetic species is not allowed."
                for varname, var in species.variables.items():
                    assert isinstance(var, PICVariable | SPHVariable)
                    obj = var.particles
                    assert isinstance(obj, Particles)

                key_spec = os.path.join("kinetic", name)
                key_spec_restart = os.path.join("restart", name)

                # restart data
                data.add_data({key_spec_restart: obj.markers})

                # marker data
                key_mks = os.path.join(key_spec, "markers")
                data.add_data({key_mks: var.saved_markers})

                # binning plot data
                for bin_plot in species.binning_plots:
                    # define slice name with binning quantity
                    slice, output_quantity = bin_plot.slice, bin_plot.output_quantity
                    slice = f"{slice}_{output_quantity}"

                    key_f = os.path.join(key_spec, "f", slice)
                    key_df = os.path.join(key_spec, "df", slice)

                    data.add_data({key_f: bin_plot.f})
                    data.add_data({key_df: bin_plot.df})

                    for dim, be in enumerate(bin_plot.bin_edges):
                        file[key_f].attrs["bin_centers" + "_" + str(dim + 1)] = be[:-1] + (be[1] - be[0]) / 2

                for i, kd_plot in enumerate(species.kernel_density_plots):
                    key_n = os.path.join(key_spec, "n_sph", f"view_{i}")

                    data.add_data({key_n: kd_plot.n_sph})
                    # save 1d point values, not meshgrids, because attrs size is limited
                    eta1 = kd_plot.plot_pts[0][:, 0, 0]
                    eta2 = kd_plot.plot_pts[1][0, :, 0]
                    eta3 = kd_plot.plot_pts[2][0, 0, :]
                    file[key_n].attrs["eta1"] = eta1
                    file[key_n].attrs["eta2"] = eta2
                    file[key_n].attrs["eta3"] = eta3

                # TODO: maybe add other data
                # else:
                #     data.add_data({key_dat: val1})

        # keys to be saved at each time step and only at end (restart)
        save_keys_all = []
        save_keys_end = []

        for key in data.dset_dict:
            if "restart" in key:
                save_keys_end.append(key)
            else:
                save_keys_all.append(key)

        return save_keys_all, save_keys_end

    def _add_time_state(self, time_state):
        """Add a pointer to the time variable of the dynamics ('t')
        to the model and to all propagators of the model.

        Parameters
        ----------
        time_state : ndarray
            Of size 1, holds the current physical time 't'.
        """
        assert time_state.size == 1
        self._time_state = time_state
        for _, prop in self.model.propagators.__dict__.items():
            if isinstance(prop, Propagator):
                prop.add_time_state(time_state)

    def _initialize_from_restart(self, data: DataContainer, verbose: bool = False):
        """
        Set initial conditions for FE coefficients (electromagnetic and fluid) and markers from restart group in hdf5 files.

        Parameters
        ----------
        data : struphy.io.output_handling.DataContainer
            The data object that links to the hdf5 files.
        """
        with h5py.File(data.file_path, "a") as file:
            for species, val in self.model.species.items():
                for variable, subval in val.variables.items():
                    # initialize feec variables
                    if isinstance(subval, FEECVariable):
                        key_restart = os.path.join("restart", species, variable)
                        subval.spline.initialize_coeffs_from_restart_file(
                            file,
                            key=key_restart,
                        )

                    # initialize pic variables
                    elif isinstance(subval, PICVariable):
                        key_restart = os.path.join("restart", species)
                        subval.particles._markers[:, :] = file[key_restart][-1, :, :]

                        if MPI.COMM_WORLD.Get_size() > 1:
                            subval.particles.mpi_sort_markers(do_test=True)

    # ------------------------------------------------------
    # Common properties with setters (from input parameters)
    # ------------------------------------------------------

    @property
    def model(self) -> StruphyModel:
        """StruphyModel object containing the PDE of the model."""
        return self._model

    @property
    def params_path(self):
        """Path to parameter file used for the run. Can be None if Simulation is instantiated in a notebook environment (no parameter file in this case)."""
        return self._params_path

    @property
    def env(self):
        """EnvironmentOptions object containing options related to the environment of the run."""
        return self._env

    @property
    def base_units(self):
        """BaseUnits object containing the four base units for the run."""
        return self._base_units

    @property
    def time_opts(self):
        """Time object containing time stepping parameters."""
        return self._time_opts

    @property
    def domain(self):
        """Domain object, see :ref:`avail_mappings`."""
        return self._domain

    @property
    def equil(self):
        """Fluid equilibrium object, see :ref:`fluid_equil`."""
        return self._equil

    @property
    def grid(self):
        """Grid object, see :ref:`grids`."""
        return self._grid

    @property
    def derham_opts(self):
        """DerhamOptions object containing options for the setup of the 3d Derham sequence."""
        return self._derham_opts

    # -----------------------------------------------------------------
    # Common properties (derived from the above properties, no setters)
    # -----------------------------------------------------------------

    @property
    def derham(self):
        """3d Derham sequence, see :ref:`derham`."""
        return self._derham

    @property
    def mass_ops(self):
        """WeighteMassOperators object, see :ref:`mass_ops`."""
        return self._mass_ops

    @property
    def basis_ops(self):
        """Basis projection operators."""
        return self._basis_ops

    @property
    def projected_equil(self):
        """Fluid equilibrium projected on 3d Derham sequence with commuting projectors."""
        return self._projected_equil

    @property
    def post_processor(self):
        """PostProcessor object for post-processing finished Struphy runs."""
        return self._post_processor

    @property
    def plotting_data(self):
        """PlottingData object for loading and storing data generated during post-processing."""
        return self._plotting_data

    @property
    def clone_config(self):
        """Config in case domain clones are used."""
        return self._clone_config

    @clone_config.setter
    def clone_config(self, new):
        assert isinstance(new, CloneConfig) or new is None
        self._clone_config = new
