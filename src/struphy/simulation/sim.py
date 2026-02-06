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
from struphy.fields_background.base import FluidEquilibrium, NumericalMHDequilibrium
from struphy.io.setup import setup_folders
from struphy.io.options import Units
from struphy.utils.clone_config import CloneConfig

# third party imports
from feectools.ddm.mpi import MockMPI
from feectools.ddm.mpi import mpi as MPI
from scope_profiler import ProfileManager
import os
import time
import pickle
import shutil
import sysconfig
import cunumpy as xp
import h5py
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
        self.domain = domain
        self.equil = equil
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
            comm = None
            rank = 0
            size = 1
            Barrier = lambda: None
        else:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            Barrier = comm.Barrier

        if rank == 0:
            print("")

        # synchronize MPI processes to set same start time of simulation for all processes
        Barrier()
        start_simulation = time.time()

        # check model
        assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
        model.verbose = verbose
        model_name = model.__class__.__name__

        if rank == 0:
            print(f"\n*** Starting run for model '{model_name}':")

        # meta-data
        path_out = env.path_out
        restart = env.restart
        max_runtime = env.max_runtime
        save_step = env.save_step
        sort_step = env.sort_step
        num_clones = env.num_clones
        use_mpi = (comm is not None,)

        meta = {}
        meta["platform"] = sysconfig.get_platform()
        meta["python version"] = sysconfig.get_python_version()
        meta["model name"] = model_name
        meta["parameter file"] = params_path
        meta["output folder"] = path_out
        meta["MPI processes"] = size
        meta["use MPI.COMM_WORLD"] = use_mpi
        meta["number of domain clones"] = num_clones
        meta["restart"] = restart
        meta["max wall-clock [min]"] = max_runtime
        meta["save interval [steps]"] = save_step

        if rank == 0:
            print("\nMETADATA:")
            for k, v in meta.items():
                print(f"{k}:".ljust(25), v)

        # creating output folders
        self._setup_folders(
            path_out=path_out,
            restart=restart,
            verbose=verbose,
        )

        # save parameter file
        if rank == 0:
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
        if comm is None:
            clone_config = None
        else:
            if num_clones == 1:
                clone_config = None
            else:
                # Setup domain cloning communicators
                # MPI.COMM_WORLD     : comm
                # within a clone:    : sub_comm
                # between the clones : inter_comm
                clone_config = CloneConfig(comm=comm, params=None, num_clones=num_clones)
                clone_config.print_clone_config()
                if model.particle_species:
                    clone_config.print_particle_config()

        self.clone_config = clone_config
        Barrier()
        
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
        model.set_normalization_params(units=self.units, verbose=verbose)

        # domain and fluid background
        self._setup_domain_and_equil(domain, equil, verbose=verbose)

    def allocate(self, verbose: bool = False):
        # feec
        self._allocate_feec(self.grid, self.derham_opts)
        
        # allocate model variables
        self.model.allocate_variables(verbose=verbose)
        self.model.allocate_helpers()

        # pass info to propagators
        self.model.allocate_propagators()

    def store_geometry(self, verbose: bool = False):
        # store geometry vtk
        if rank == 0:
            grids_log = [
                xp.linspace(1e-6, 1.0, 32),
                xp.linspace(0.0, 1.0, 32),
                xp.linspace(0.0, 1.0, 32),
            ]

            tmp = model.domain(*grids_log)
            grids_phy = [tmp[0], tmp[1], tmp[2]]

            pointData = {}
            det_df = model.domain.jacobian_det(*grids_log)
            pointData["det_df"] = det_df

            if model.equil is not None:
                p0 = model.equil.p0(*grids_log)
                pointData["p0"] = p0
                if isinstance(model.equil, FluidEquilibriumWithB):
                    absB0 = model.equil.absB0(*grids_log)
                    pointData["absB0"] = absB0

            gridToVTK(os.path.join(path_out, "geometry"), *grids_phy, pointData=pointData)

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

    def run(self, verbose: bool = False):
        if rank < 32:
            if rank == 0:
                print("")
            print(f"Rank {rank}: executing main.run() for model {model_name} ...")

        if size > 32 and rank == 32:
            print(f"Ranks > 31: executing main.run() for model {model_name} ...")

        # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
        # use MPI.COMM_WORLD as communicator when storing the outputs
        data = DataContainer(path_out, comm=comm)

        # time quantities (current time value, value in seconds and index)
        time_state = {}
        time_state["value"] = xp.zeros(1, dtype=float)
        time_state["value_sec"] = xp.zeros(1, dtype=float)
        time_state["index"] = xp.zeros(1, dtype=int)

        # add time quantities to data object for saving
        for key, val in time_state.items():
            key_time = "time/" + key
            key_time_restart = "restart/time/" + key
            data.add_data({key_time: val})
            data.add_data({key_time_restart: val})

        # retrieve time parameters
        dt = time_opts.dt
        Tend = time_opts.Tend
        split_algo = time_opts.split_algo

        # set initial conditions for all variables
        if restart:
            model.initialize_from_restart(data)

            with h5py.File(data.file_path, "a") as file:
                time_state["value"][0] = file["restart/time/value"][-1]
                time_state["value_sec"][0] = file["restart/time/value_sec"][-1]
                time_state["index"][0] = file["restart/time/index"][-1]

            total_steps = str(int(round((Tend - time_state["value"][0]) / dt)))
        else:
            total_steps = str(int(round(Tend / dt)))

        # compute initial scalars and kinetic data, pass time state to all propagators
        model.update_scalar_quantities()
        model.update_markers_to_be_saved()
        model.update_distr_functions()
        model.add_time_state(time_state["value"])

        # add all variables to be saved to data object
        save_keys_all, save_keys_end = model.initialize_data_output(data, size)

        # ======================== main time loop ======================
        model.update_scalar_quantities()
        if rank == 0:
            print("\nINITIAL SCALAR QUANTITIES:")
            model.print_scalar_quantities()

            print(f"\nSTART TIME STEPPING WITH '{split_algo}' SPLITTING:")

        # time loop
        run_time_now = 0.0
        while True:
            Barrier()

            # stop time loop?
            break_cond_1 = time_state["value"][0] >= Tend
            break_cond_2 = run_time_now > max_runtime

            if break_cond_1 or break_cond_2:
                # save restart data (other data already saved below)
                data.save_data(keys=save_keys_end)
                end_simulation = time.time()
                if rank == 0:
                    print(f"\nTime steps done: {time_state['index'][0]}")
                    print(
                        "wall-clock time of simulation [sec]: ",
                        end_simulation - start_simulation,
                    )
                    print()
                break

            if sort_step and time_state["index"][0] % sort_step == 0:
                t0 = time.time()
                for key, val in model.pointer.items():
                    if isinstance(val, Particles):
                        val.do_sort()
                t1 = time.time()
                if rank == 0 and verbose:
                    message = "Particles sorted | wall clock [s]: {0:8.4f} | sorting duration [s]: {1:8.4f}".format(
                        run_time_now * 60,
                        t1 - t0,
                    )
                    print(message, end="\n")
                    print()

            # update time and index (round time to 10 decimals for a clean time grid!)
            time_state["value"][0] = round(time_state["value"][0] + dt, 10)
            time_state["value_sec"][0] = round(time_state["value_sec"][0] + dt * model.units.t, 10)
            time_state["index"][0] += 1

            # perform one time step dt
            t0 = time.time()
            with ProfileManager.profile_region("model.integrate"):
                model.integrate(dt, split_algo)
            t1 = time.time()

            run_time_now = (time.time() - start_simulation) / 60

            # update diagnostics data and save data
            if time_state["index"][0] % save_step == 0:
                # compute scalars and kinetic data
                model.update_scalar_quantities()
                model.update_markers_to_be_saved()
                model.update_distr_functions()

                # extract FEEC coefficients
                feec_species = model.field_species | model.fluid_species | model.diagnostic_species
                for species, val in feec_species.items():
                    assert isinstance(val, Species)
                    for variable, subval in val.variables.items():
                        assert isinstance(subval, FEECVariable)
                        spline = subval.spline
                        # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                        spline.extract_coeffs(update_ghost_regions=False)

                # save data (everything but restart data)
                data.save_data(keys=save_keys_all)

                # print current time and scalar quantities to screen
                if rank == 0 and verbose:
                    step = str(time_state["index"][0]).zfill(len(total_steps))

                    message = "time step: " + step + "/" + str(total_steps)
                    message += " | " + "time: {0:10.5f}/{1:10.5f}".format(time_state["value"][0], Tend)
                    message += " | " + "phys. time [s]: {0:12.10f}/{1:12.10f}".format(
                        time_state["value_sec"][0],
                        Tend * model.units.t,
                    )
                    message += " | " + "wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}".format(
                        run_time_now * 60,
                        t1 - t0,
                    )

                    print(message, end="\n")
                    model.print_scalar_quantities()
                    print()

        # ===================================================================

        meta["wall-clock time[min]"] = (end_simulation - start_simulation) / 60
        Barrier()

        if rank == 0:
            # save meta-data
            dict_to_yaml(meta, os.path.join(path_out, "meta.yml"))
            print("Struphy run finished.")

        if clone_config is not None:
            clone_config.free()

        ProfileManager.finalize()
    
    def _setup_folders(
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
                
    def _allocate_feec(self, grid: TensorProductGrid, derham_opts: DerhamOptions):
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
            self._derham = setup_derham(
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
                