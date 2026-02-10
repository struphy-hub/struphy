# api imports
from struphy import (EnvironmentOptions, 
                     BaseUnits, 
                     Time,  
                     domains, 
                     equils, 
                     grids,
                     DerhamOptions,
                     )

# core imports
from struphy.models.base import StruphyModel
from struphy.geometry.base import Domain
from struphy.fields_background.base import FluidEquilibrium, NumericalMHDequilibrium, FluidEquilibriumWithB
from struphy.io.setup import setup_folders
from struphy.physics.physics import Units
from struphy.utils.clone_config import CloneConfig
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
from struphy.propagators.base import Propagator
from struphy.models.species import DiagnosticSpecies, FieldSpecies, FluidSpecies, ParticleSpecies, Species
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.io.output_handling import DataContainer
from struphy.pic.base import Particles

# third party imports
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.stencil import StencilVector
from scope_profiler import ProfileManager
import os
import time
import pickle
import shutil
import sysconfig
import cunumpy as xp
import h5py
import glob
from line_profiler import profile
from pyevtk.hl import gridToVTK


class Simulation:
    def __init__(self, 
                model: StruphyModel, 
                params_path: str = None,
                env: EnvironmentOptions = EnvironmentOptions(),
                base_units: BaseUnits = BaseUnits(),
                time_opts: Time = Time(),
                domain: Domain = domains.Cuboid(),
                equil: FluidEquilibrium = equils.HomogenSlab(),
                grid: grids.TensorProductGrid = None,
                derham_opts: DerhamOptions = None,
                verbose: bool = False,
                ):
        
        self.model = model
        self.params_path = params_path
        self.env = env
        self.base_units = base_units
        self.time_opts = time_opts
        self.grid = grid
        self.derham_opts = derham_opts  
        self.verbose = verbose
        
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
            self.size = 1
            self.Barrier = lambda: None
        else:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.Barrier = self.comm.Barrier

        if self.rank == 0:
            print("")

        # synchronize MPI processes to set same start time of simulation for all processes
        self.Barrier()
        self.start_time = time.time()

        # check model
        assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
        model.verbose = verbose
        model_name = model.__class__.__name__

        if self.rank == 0:
            print(f"\n*** Starting run for model '{model_name}':")

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
        self.meta["model name"] = model_name
        self.meta["parameter file"] = params_path
        self.meta["output folder"] = path_out
        self.meta["MPI processes"] = self.size
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
            path_out=path_out,
            restart=restart,
            verbose=verbose,
        )

        # save parameter file
        if self.rank == 0:
            # save python param file
            if params_path is not None:
                assert params_path[-3:] == ".py"
                shutil.copy2(
                    params_path,
                    os.path.join(path_out, "parameters.py"),
                )
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
        units = Units(base_units)
        self.units = units
        if model.bulk_species is None:
            A_bulk = None
            Z_bulk = None
        else:
            A_bulk = model.bulk_species.mass_number
            Z_bulk = model.bulk_species.charge_number
        self.units.derive_units(
            velocity_scale=model.velocity_scale,
            A_bulk=A_bulk,
            Z_bulk=Z_bulk,
            verbose=verbose,
        )
        model.setup_equation_params(units=self.units, verbose=verbose)

        # domain and fluid background
        self._setup_domain_and_equil(domain, equil, verbose=verbose)

    def allocate(self, verbose: bool = False):
        # feec
        self._allocate_feec(self.grid, self.derham_opts)
        
        # allocate model variables
        self._allocate_variables(verbose=verbose)
        self.model.allocate_helpers()

        # pass info to propagators
        self._allocate_propagators()

    # def store_geometry(self, verbose: bool = False):
    #     # store geometry vtk
    #     if self.rank == 0:
    #         grids_log = [
    #             xp.linspace(1e-6, 1.0, 32),
    #             xp.linspace(0.0, 1.0, 32),
    #             xp.linspace(0.0, 1.0, 32),
    #         ]

    #         tmp = self.domain(*grids_log)
    #         grids_phy = [tmp[0], tmp[1], tmp[2]]

    #         pointData = {}
    #         det_df = self.domain.jacobian_det(*grids_log)
    #         pointData["det_df"] = det_df

    #         if self.equil is not None:
    #             p0 = self.equil.p0(*grids_log)
    #             pointData["p0"] = p0
    #             if isinstance(self.equil, FluidEquilibriumWithB):
    #                 absB0 = self.equil.absB0(*grids_log)
    #                 pointData["absB0"] = absB0

    #         gridToVTK(os.path.join(self.env.path_out, "geometry"), *grids_phy, pointData=pointData)

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
        eta1 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta2 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)
        eta3 = xp.linspace(h / 2.0, 1.0 - h / 2.0, 20)

        ##  global parameters

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
        
        scalars = self.model.scalar_quantities

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
                n_mks_tot = xp.array([variable.particles.Np])
                value_array /= n_mks_tot

            # Update the scalar value
            scalars[name]["value"][0] = value_array[0]

        else:
            # Sum the values of the summands
            value = sum(scalars[summand]["value"][0] for summand in summands)
            scalars[name]["value"][0] = value

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
    # def run(self, verbose: bool = False):
    #     if rank < 32:
    #         if rank == 0:
    #             print("")
    #         print(f"Rank {rank}: executing main.run() for model {model_name} ...")

    #     if size > 32 and rank == 32:
    #         print(f"Ranks > 31: executing main.run() for model {model_name} ...")

    #     # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
    #     # use MPI.COMM_WORLD as communicator when storing the outputs
    #     data = DataContainer(path_out, comm=comm)

    #     # time quantities (current time value, value in seconds and index)
    #     time_state = {}
    #     time_state["value"] = xp.zeros(1, dtype=float)
    #     time_state["value_sec"] = xp.zeros(1, dtype=float)
    #     time_state["index"] = xp.zeros(1, dtype=int)

    #     # add time quantities to data object for saving
    #     for key, val in time_state.items():
    #         key_time = "time/" + key
    #         key_time_restart = "restart/time/" + key
    #         data.add_data({key_time: val})
    #         data.add_data({key_time_restart: val})

    #     # retrieve time parameters
    #     dt = time_opts.dt
    #     Tend = time_opts.Tend
    #     split_algo = time_opts.split_algo

    #     # set initial conditions for all variables
    #     if restart:
    #         model.initialize_from_restart(data)

    #         with h5py.File(data.file_path, "a") as file:
    #             time_state["value"][0] = file["restart/time/value"][-1]
    #             time_state["value_sec"][0] = file["restart/time/value_sec"][-1]
    #             time_state["index"][0] = file["restart/time/index"][-1]

    #         total_steps = str(int(round((Tend - time_state["value"][0]) / dt)))
    #     else:
    #         total_steps = str(int(round(Tend / dt)))

    #     # compute initial scalars and kinetic data, pass time state to all propagators
    #     model.update_scalar_quantities()
    #     model.update_markers_to_be_saved()
    #     model.update_distr_functions()
    #     model.add_time_state(time_state["value"])

    #     # add all variables to be saved to data object
    #     save_keys_all, save_keys_end = model.initialize_data_output(data, size)

    #     # ======================== main time loop ======================
    #     model.update_scalar_quantities()
    #     if rank == 0:
    #         print("\nINITIAL SCALAR QUANTITIES:")
    #         model.print_scalar_quantities()

    #         print(f"\nSTART TIME STEPPING WITH '{split_algo}' SPLITTING:")

    #     # time loop
    #     run_time_now = 0.0
    #     while True:
    #         Barrier()

    #         # stop time loop?
    #         break_cond_1 = time_state["value"][0] >= Tend
    #         break_cond_2 = run_time_now > max_runtime

    #         if break_cond_1 or break_cond_2:
    #             # save restart data (other data already saved below)
    #             data.save_data(keys=save_keys_end)
    #             end_simulation = time.time()
    #             if rank == 0:
    #                 print(f"\nTime steps done: {time_state['index'][0]}")
    #                 print(
    #                     "wall-clock time of simulation [sec]: ",
    #                     end_simulation - start_simulation,
    #                 )
    #                 print()
    #             break

    #         if sort_step and time_state["index"][0] % sort_step == 0:
    #             t0 = time.time()
    #             for key, val in model.pointer.items():
    #                 if isinstance(val, Particles):
    #                     val.do_sort()
    #             t1 = time.time()
    #             if rank == 0 and verbose:
    #                 message = "Particles sorted | wall clock [s]: {0:8.4f} | sorting duration [s]: {1:8.4f}".format(
    #                     run_time_now * 60,
    #                     t1 - t0,
    #                 )
    #                 print(message, end="\n")
    #                 print()

    #         # update time and index (round time to 10 decimals for a clean time grid!)
    #         time_state["value"][0] = round(time_state["value"][0] + dt, 10)
    #         time_state["value_sec"][0] = round(time_state["value_sec"][0] + dt * model.units.t, 10)
    #         time_state["index"][0] += 1

    #         # perform one time step dt
    #         t0 = time.time()
    #         with ProfileManager.profile_region("model.integrate"):
    #             model.integrate(dt, split_algo)
    #         t1 = time.time()

    #         run_time_now = (time.time() - start_simulation) / 60

    #         # update diagnostics data and save data
    #         if time_state["index"][0] % save_step == 0:
    #             # compute scalars and kinetic data
    #             model.update_scalar_quantities()
    #             model.update_markers_to_be_saved()
    #             model.update_distr_functions()

    #             # extract FEEC coefficients
    #             feec_species = model.field_species | model.fluid_species | model.diagnostic_species
    #             for species, val in feec_species.items():
    #                 assert isinstance(val, Species)
    #                 for variable, subval in val.variables.items():
    #                     assert isinstance(subval, FEECVariable)
    #                     spline = subval.spline
    #                     # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
    #                     spline.extract_coeffs(update_ghost_regions=False)

    #             # save data (everything but restart data)
    #             data.save_data(keys=save_keys_all)

    #             # print current time and scalar quantities to screen
    #             if rank == 0 and verbose:
    #                 step = str(time_state["index"][0]).zfill(len(total_steps))

    #                 message = "time step: " + step + "/" + str(total_steps)
    #                 message += " | " + "time: {0:10.5f}/{1:10.5f}".format(time_state["value"][0], Tend)
    #                 message += " | " + "phys. time [s]: {0:12.10f}/{1:12.10f}".format(
    #                     time_state["value_sec"][0],
    #                     Tend * model.units.t,
    #                 )
    #                 message += " | " + "wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}".format(
    #                     run_time_now * 60,
    #                     t1 - t0,
    #                 )

    #                 print(message, end="\n")
    #                 model.print_scalar_quantities()
    #                 print()

    #     # ===================================================================

    #     self.meta["wall-clock time[min]"] = (end_simulation - start_simulation) / 60
    #     Barrier()

    #     if rank == 0:
    #         # save meta-data
    #         dict_to_yaml(self.meta, os.path.join(path_out, "meta.yml"))
    #         print("Struphy run finished.")

    #     if clone_config is not None:
    #         clone_config.free()

    #     ProfileManager.finalize()
    
    def _setup_folders(
        self,
        path_out: str,
        restart: bool,
        verbose: bool = False,
    ):
        """
        Setup output folders.
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            if verbose:
                print("\nPREPARATION AND CLEAN-UP:")

            # create output folder if it does not exit
            if not os.path.exists(path_out):
                os.makedirs(path_out, exist_ok=True)
                if verbose:
                    print("Created folder " + path_out)

            # create data folder in output folder if it does not exist
            if not os.path.exists(os.path.join(path_out, "data/")):
                os.mkdir(os.path.join(path_out, "data/"))
                if verbose:
                    print("Created folder " + os.path.join(path_out, "data/"))
            else:
                # remove post_processing folder
                folder = os.path.join(path_out, "post_processing")
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    if verbose:
                        print("Removed existing folder " + folder)

                # remove meta file
                file = os.path.join(path_out, "meta.txt")
                if os.path.exists(file):
                    os.remove(file)
                    if verbose:
                        print("Removed existing file " + file)

                # remove profiling file
                file = os.path.join(path_out, "profile_tmp")
                if os.path.exists(file):
                    os.remove(file)
                    if verbose:
                        print("Removed existing file " + file)

                # remove .png files (if NOT a restart)
                if not restart:
                    files = glob.glob(os.path.join(path_out, "*.png"))
                    for n, file in enumerate(files):
                        os.remove(file)
                        if verbose and n < 10:  # print only ten statements in case of many processes
                            print("Removed existing file " + file)

                    files = glob.glob(os.path.join(path_out, "data", "*.hdf5"))
                    for n, file in enumerate(files):
                        os.remove(file)
                        if verbose and n < 10:  # print only ten statements in case of many processes
                            print("Removed existing file " + file)
        
    def _setup_domain_and_equil(self, domain: Domain, equil: FluidEquilibrium, verbose: bool=False):
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
                    velocity_scale=self.velocity_scale,
                    A_bulk=self.bulk_species.mass_number,
                    Z_bulk=self.bulk_species.charge_number,
                    verbose=self.verbose,
                )

        else:
            self._domain = domain

        self._equil = equil

        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print("\nDOMAIN:")
            print("type:".ljust(25), self.domain.__class__.__name__)
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
    
    def _setup_derham(
        self,
    grid: grids.TensorProductGrid,
    options: DerhamOptions,
    comm: MPI.Intracomm = None,
    domain: Domain = None,
    verbose=False,
):
        """
        Creates the 3d derham sequence for given grid parameters.

        Parameters
        ----------
        grid : TensorProductGrid
            The FEEC grid.

        comm: Intracomm
            MPI communicator (sub_comm if clones are used).

        domain : Domain, optional
            The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

        verbose : bool
            Show info on screen.

        Returns
        -------
        derham : struphy.feec.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.
        """

        from struphy.feec.psydac_derham import Derham

        # number of grid cells
        Nel = grid.Nel
        # mpi
        mpi_dims_mask = grid.mpi_dims_mask

        # spline degrees
        p = options.p
        # spline types (clamped vs. periodic)
        spl_kind = options.spl_kind
        # boundary conditions (Homogeneous Dirichlet or None)
        dirichlet_bc = options.dirichlet_bc
        # Number of quadrature points per histopolation cell
        nq_pr = options.nq_pr
        # Number of quadrature points per grid cell for L^2
        nquads = options.nquads
        # C^k smoothness at eta_1=0 for polar domains
        polar_ck = options.polar_ck
        # local commuting projectors
        local_projectors = options.local_projectors

        derham = Derham(
            Nel,
            p,
            spl_kind,
            dirichlet_bc=dirichlet_bc,
            nquads=nquads,
            nq_pr=nq_pr,
            comm=comm,
            mpi_dims_mask=mpi_dims_mask,
            with_projectors=True,
            polar_ck=polar_ck,
            domain=domain,
            local_projectors=local_projectors,
        )

        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print("\nDERHAM:")
            print("number of elements:".ljust(25), Nel)
            print("spline degrees:".ljust(25), p)
            print("periodic bcs:".ljust(25), spl_kind)
            print("hom. Dirichlet bc:".ljust(25), dirichlet_bc)
            print("GL quad pts (L2):".ljust(25), nquads)
            print("GL quad pts (hist):".ljust(25), nq_pr)
            print(
                "MPI proc. per dir.:".ljust(25),
                derham.domain_decomposition.nprocs,
            )
            print("use polar splines:".ljust(25), derham.polar_ck == 1)
            print("domain on process 0:".ljust(25), derham.domain_array[0])

        return derham
    
    @profile            
    def _allocate_feec(self, grid: grids.TensorProductGrid, derham_opts: DerhamOptions):
        # create discrete derham sequence
        if self.clone_config is None:
            derham_comm = MPI.COMM_WORLD
        else:
            derham_comm = self.clone_config.sub_comm

        if grid is None or derham_opts is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"\n{grid =}, {derham_opts =}: no Derham object set up.")
            self._derham = None
        else:
            self._derham = self._setup_derham(
                grid,
                derham_opts,
                comm=derham_comm,
                domain=self.domain,
                verbose=self.verbose,
            )

        # create weighted mass and basis operators
        if self.derham is None:
            self._mass_ops = None
            self._basis_ops = None
        else:
            self._mass_ops = WeightedMassOperators(
                self.derham,
                self.domain,
                verbose=self.verbose,
                eq_mhd=self.equil,
            )

            self._basis_ops = BasisProjectionOperators(
                self.derham,
                self.domain,
                verbose=self.verbose,
                eq_mhd=self.equil,
            )

        # create projected equilibrium
        if self.derham is None:
            self._projected_equil = None
        else:
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
    def _allocate_propagators(self):
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
            prop.allocate()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"\nAllocated propagator '{prop.__class__.__name__}'.")
    
    @profile
    def _initialize_data_output(self, data: DataContainer, size):
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
    