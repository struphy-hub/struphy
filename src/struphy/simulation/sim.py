# api imports
from struphy import (EnvironmentOptions,
                     BaseUnits,
                     Time,
                     domains,
                     equils,
                     grids,
                     DerhamOptions,
                     pproc,
                     load_plotting_data,
                     )

# core imports
from struphy.models.base import StruphyModel
from struphy.geometry.base import Domain
from struphy.fields_background.base import (FluidEquilibrium, NumericalMHDequilibrium, FluidEquilibriumWithB,)
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
from struphy.models.species import (DiagnosticSpecies, FieldSpecies, FluidSpecies, ParticleSpecies, Species,)
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.io.output_handling import DataContainer
from struphy.pic.base import Particles
from struphy.utils.utils import dict_to_yaml
from struphy.simulation.base import Simulation
from struphy.feec.psydac_derham import SplineFunction
from struphy.post_processing.orbits import orbits_tools
from struphy.kinetic_background.base import KineticBackground
from struphy.post_processing.post_processing_tools import SimData   

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
import yaml
from tqdm import tqdm
from line_profiler import profile
from pyevtk.hl import gridToVTK


class StruphySimulation(Simulation):

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

        # synchronize MPI processes to set same start time of simulation for all processes
        self.Barrier()
        self.start_time = time.time()

        # check model
        assert hasattr(model, "propagators"), "Attribute 'self.propagators' must be set in model __init__!"
        self.model_name = model.__class__.__name__

        if self.rank == 0:
            print(f"\n*** Starting run for model '{self.model_name}':")

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
        self.meta["parameter file"] = params_path
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

    # -----------------
    # Common properties
    # -----------------

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
    
    @property
    def clone_config(self):
        """Config in case domain clones are used."""
        return self._clone_config

    @clone_config.setter
    def clone_config(self, new):
        assert isinstance(new, CloneConfig) or new is None
        self._clone_config = new

    @property
    def path_pproc(self):
        """Path to post-processing folder."""
        return os.path.join(self.env.path_out, "post_processing")

    # ----------------
    # Abstract methods
    # ----------------

    def allocate(self, verbose: bool = False):
        # feec
        self._allocate_feec(self.grid, self.derham_opts, verbose=verbose)

        # allocate model variables
        self._allocate_variables(verbose=verbose)

        # pass info to propagators
        self._allocate_propagators(verbose=verbose)

        # allocate helper fields and perform initial solves if needed
        self.model.allocate_helpers(verbose=verbose)

    def save_geometry_and_equil_vtk(self, verbose: bool = False):
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
"""
            )
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
            self.time_state["value"][0] = round(self.time_state["value"][0] + dt, 10)
            self.time_state["value_sec"][0] = round(self.time_state["value_sec"][0] + dt * self.units.t, 10)
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

                    message = "time step: " + step + "/" + str(total_steps)
                    message += " | " + "time: {0:10.5f}/{1:10.5f}".format(self.time_state["value"][0], Tend)
                    message += " | " + "phys. time [s]: {0:12.10f}/{1:12.10f}".format(
                        self.time_state["value_sec"][0],
                        Tend * self.units.t,
                    )
                    message += " | " + "wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}".format(
                        run_time_now * 60,
                        t1 - t0,
                    )

                    print(message, end="\n")
                    self.model.print_scalar_quantities()
                    print()

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

    def pproc(self, step: int = 1,
        celldivide: int = 1,
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        time_trace: bool = False,
        verbose: bool = False,):
        pproc(sim=self, step=step, celldivide=celldivide, physical=physical, guiding_center=guiding_center, classify=classify, create_vtk=create_vtk, time_trace=time_trace, verbose=verbose,)

    def load_plotting_data(self, verbose: bool = False) -> SimData:
        return load_plotting_data(sim=self, verbose=verbose)

    # ---------------------
    # Code specific methods
    # ---------------------

    def pproc_fields(self, 
                     step: int = 1, 
                     celldivide: int = 1, 
                     physical: bool = False, 
                     create_vtk: bool = True,
                     verbose: bool = False,
                     ):
        fields, t_grid = self._create_femfields(step=step)
        point_data, grids_log, grids_phy = self._eval_femfields(fields, celldivide=[celldivide] * 3)
        if physical:
            point_data_phy, _, _ = self._eval_femfields(
                fields,
                celldivide=[celldivide] * 3,
                physical=True,
            )

        # directory for field data
        path_fields = os.path.join(self.path_pproc, "fields_data")

        try:
            os.mkdir(path_fields)
        except:
            shutil.rmtree(path_fields)
            os.mkdir(path_fields)

        # save data dicts for each field
        for species, vars in point_data.items():
            for name, val in vars.items():
                try:
                    os.mkdir(os.path.join(path_fields, species))
                except:
                    pass

                with open(os.path.join(path_fields, species, name + "_log.bin"), "wb") as handle:
                    pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if physical:
                    with open(os.path.join(path_fields, species, name + "_phy.bin"), "wb") as handle:
                        pickle.dump(point_data_phy[species][name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save grids
        with open(os.path.join(path_fields, "grids_log.bin"), "wb") as handle:
            pickle.dump(grids_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path_fields, "grids_phy.bin"), "wb") as handle:
            pickle.dump(grids_phy, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create vtk files
        if create_vtk:
            self._create_vtk(path_fields, t_grid, grids_phy, point_data)
            if physical:
                self._create_vtk(path_fields, t_grid, grids_phy, point_data_phy, physical=True)

    def pproc_particles(self, 
                        step: int = 1,
                        guiding_center: bool = False,
                        classify: bool = False,
                        verbose: bool = False,):
        # directory for kinetic data
        path_kinetics = os.path.join(self.path_pproc, "kinetic_data")

        try:
            os.mkdir(path_kinetics)
        except:
            shutil.rmtree(path_kinetics)
            os.mkdir(path_kinetics)

        # kinetic post-processing for each species
        for n, species in enumerate(self.kinetic_species):
            # directory for each species
            path_kinetics_species = os.path.join(path_kinetics, species)

            try:
                os.mkdir(path_kinetics_species)
            except:
                shutil.rmtree(path_kinetics_species)
                os.mkdir(path_kinetics_species)

            # markers
            if self.exist_particles["markers"]:
                self._post_process_markers(
                    path_kinetics_species,
                    step,
                )

                if guiding_center:
                    assert self.kinetic_kinds[n] == "Particles6D"
                    orbits_tools.post_process_orbit_guiding_center(self.env.path_out, path_kinetics_species, species)

                if classify:
                    orbits_tools.post_process_orbit_classification(path_kinetics_species, species)

            # distribution function
            if self.exist_particles["f"]:
                if self.kinetic_kinds[n] == "DeltaFParticles6D":
                    compute_bckgr = True
                else:
                    compute_bckgr = False

                self._post_process_f(
                    path_kinetics_species,
                    step,
                    compute_bckgr=compute_bckgr,
                )

            # sph density
            if self.exist_particles["n_sph"]:
                self._post_process_n_sph(
                    path_kinetics_species,
                    step,
                )

    def compute_plasma_params(self, verbose: bool=True):
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

    # ---------------
    # Private methods
    # ---------------

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
                    velocity_scale=self.velocity_scale,
                    A_bulk=self.bulk_species.mass_number,
                    Z_bulk=self.bulk_species.charge_number,
                    verbose=verbose,
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
    def _allocate_feec(self, grid: grids.TensorProductGrid, derham_opts: DerhamOptions, verbose: bool = False):
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
            self._derham = self._setup_derham(
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
            self._mass_ops = WeightedMassOperators(
                self.derham,
                self.domain,
                verbose=verbose,
                eq_mhd=self.equil,
            )

            self._basis_ops = BasisProjectionOperators(
                self.derham,
                self.domain,
                verbose=verbose,
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
    def _allocate_propagators(self, verbose: bool = False):
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
            if MPI.COMM_WORLD.Get_rank() == 0:
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

    def _create_femfields(self, step: int = 1, verbose: bool = False):
        """Creates instances of :class:`~struphy.feec.psydac_derham.SplineFunction` from distributed Struphy data.

        Parameters
        ----------
        step : int
            Whether to create FEM fields at every time step (step=1, default), every second time step (step=2), etc.

        Returns
        -------
        fields : dict
            Nested dictionary holding :class:`~struphy.feec.psydac_derham.SplineFunction`: fields[t][name] contains the Field with the name "name" in the hdf5 file at time t.

        t_grid : xp.ndarray
            Time grid.
        """
        # get fields names, space IDs and time grid from 0-th rank hdf5 file
        with h5py.File(os.path.join(self.env.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            space_ids = {}
            print("\nReading hdf5 data of following species:")
            for species, dset in file["feec"].items():
                space_ids[species] = {}
                print(f"{species}:")
                for var, ddset in dset.items():
                    space_ids[species][var] = ddset.attrs["space_id"]
                    print(f"  {var}:", ddset)

            t_grid = file["time/value"][::step].copy()

        # create one FemField for each snapshot
        fields = {}
        for t in t_grid:
            fields[t] = {}
            for species, vars in space_ids.items():
                fields[t][species] = {}
                for var, id in vars.items():
                    fields[t][species][var] = self.derham.create_spline_function(
                        var,
                        id,
                        verbose=False,
                    )

        # get hdf5 data
        print("")
        for rank in range(int(self.comm_size)):
            # open hdf5 file
            with h5py.File(os.path.join(self.env.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                for species, dset in file["feec"].items():
                    for var, ddset in tqdm(dset.items()):
                        # get global start indices, end indices and pads
                        gl_s = ddset.attrs["starts"]
                        gl_e = ddset.attrs["ends"]
                        pads = ddset.attrs["pads"]

                        assert gl_s.shape == (3,) or gl_s.shape == (3, 3)
                        assert gl_e.shape == (3,) or gl_e.shape == (3, 3)
                        assert pads.shape == (3,) or pads.shape == (3, 3)

                        # loop over time
                        for n, t in enumerate(t_grid):
                            # scalar field
                            if gl_s.shape == (3,):
                                s1, s2, s3 = gl_s
                                e1, e2, e3 = gl_e
                                p1, p2, p3 = pads

                                data = ddset[n * step, p1:-p1, p2:-p2, p3:-p3].copy()

                                fields[t][species][var].vector[
                                    s1 : e1 + 1,
                                    s2 : e2 + 1,
                                    s3 : e3 + 1,
                                ] = data
                                # update after each data addition, can be made more efficient
                                fields[t][species][var].vector.update_ghost_regions()

                            # vector-valued field
                            else:
                                for comp in range(3):
                                    s1, s2, s3 = gl_s[comp]
                                    e1, e2, e3 = gl_e[comp]
                                    p1, p2, p3 = pads[comp]

                                    data = ddset[str(comp + 1)][
                                        n * step,
                                        p1:-p1,
                                        p2:-p2,
                                        p3:-p3,
                                    ].copy()

                                    fields[t][species][var].vector[comp][
                                        s1 : e1 + 1,
                                        s2 : e2 + 1,
                                        s3 : e3 + 1,
                                    ] = data
                                # update after each data addition, can be made more efficient
                                fields[t][species][var].vector.update_ghost_regions()

        print("Creation of Struphy Fields done.")

        return fields, t_grid

    def _eval_femfields(
        self,
        fields: dict,
        *,
        celldivide: list = [1, 1, 1],
        physical: bool = False,
        verbose: bool = False,
    ):
        """Evaluate FEM fields obtained from :meth:`struphy.post_processing.post_processing_tools.create_femfields`.

        Parameters
        ----------
        params_in : ParamsIn
            Simulation parameters.

        fields : dict
            Obtained from struphy.diagnostics.post_processing.create_femfields.

        celldivide : list of ints
            Grid refinement in each eta direction.

        physical : bool
            Wether to do post-processing into push-forwarded physical (xyz) components of fields.

        Returns
        -------
        point_data : dict
            Nested dictionary holding values of FemFields on the grid as list of 3d xp.arrays:
            point_data[name][t] contains the values of the field with name "name" in fields[t].keys() at time t.

            If physical is True, physical components of fields are saved.
            Otherwise, logical components (differential n-forms) are saved.

        grids_log : 3-list
            1d logical grids in each eta-direction with Nel[i]*cell_divide[i] + 1 entries in each direction.

        grids_phy : 3-list
            Mapped (physical) grids obtained by domain(*grids_log).
        """

        # create logical and physical grids
        assert isinstance(fields, dict)
        assert isinstance(celldivide, list)
        assert len(celldivide) == 3

        Nel = self.grid.Nel

        grids_log = [xp.linspace(0.0, 1.0, Nel_i * n_i + 1) for Nel_i, n_i in zip(Nel, celldivide)]
        grids_phy = [
            self.domain(*grids_log)[0],
            self.domain(*grids_log)[1],
            self.domain(*grids_log)[2],
        ]

        # evaluate fields at evaluation grid and push-forward
        point_data = {}
        for species, vars in fields[list(fields.keys())[0]].items():
            point_data[species] = {}
            for name, field in vars.items():
                point_data[species][name] = {}

        print("\nEvaluating fields ...")
        for t in tqdm(fields):
            for species, vars in fields[t].items():
                for name, field in vars.items():
                    assert isinstance(field, SplineFunction)
                    space_id = field.space_id

                    # field evaluation
                    temp_val = field(*grids_log)

                    point_data[species][name][t] = []

                    # scalar spaces
                    if isinstance(temp_val, xp.ndarray):
                        if physical:
                            # push-forward
                            if space_id == "H1":
                                point_data[species][name][t].append(
                                    self.domain.push(
                                        temp_val,
                                        *grids_log,
                                        kind="0",
                                    ),
                                )
                            elif space_id == "L2":
                                point_data[species][name][t].append(
                                    self.domain.push(
                                        temp_val,
                                        *grids_log,
                                        kind="3",
                                    ),
                                )

                        else:
                            point_data[species][name][t].append(temp_val)

                    # vector-valued spaces
                    else:
                        for j in range(3):
                            if physical:
                                # push-forward
                                if space_id == "Hcurl":
                                    point_data[species][name][t].append(
                                        self.domain.push(
                                            temp_val,
                                            *grids_log,
                                            kind="1",
                                        )[j],
                                    )
                                elif space_id == "Hdiv":
                                    point_data[species][name][t].append(
                                        self.domain.push(
                                            temp_val,
                                            *grids_log,
                                            kind="2",
                                        )[j],
                                    )
                                elif space_id == "H1vec":
                                    point_data[species][name][t].append(
                                        self.domain.push(
                                            temp_val,
                                            *grids_log,
                                            kind="v",
                                        )[j],
                                    )

                            else:
                                point_data[species][name][t].append(temp_val[j])

        return point_data, grids_log, grids_phy

    def _create_vtk(
        self,
        path: str,
        t_grid: xp.ndarray,
        grids_phy: list,
        point_data: dict,
        *,
        physical: bool = False,
        verbose: bool = False,
    ):
        """Creates structured virtual toolkit files (.vts) for Paraview from evaluated field data.

        Parameters
        ----------
        path : str
            Absolute path of where to store the .vts files. Will then be in path/vtk/step_<step>.vts.

        t_grid : xp.ndarray
            Time grid.

        grids_phy : 3-list
            Mapped (physical) grids obtained from struphy.diagnostics.post_processing.eval_femfields.

        point_data : dict
            Field data obtained from struphy.diagnostics.post_processing.eval_femfields.

        physical : bool
            Wether to create vtk for push-forwarded physical (xyz) components of fields.
        """
        for species, vars in point_data.items():
            species_path = os.path.join(path, species, "vtk" + physical * "_phy")
            try:
                os.mkdir(species_path)
            except:
                shutil.rmtree(species_path)
                os.mkdir(species_path)

        # time loop
        nt = len(t_grid) - 1
        log_nt = int(xp.log10(nt)) + 1

        print(f"\nCreating vtk in {path} ...")
        for n, t in enumerate(tqdm(t_grid)):
            point_data_n = {}

            for species, vars in point_data.items():
                species_path = os.path.join(path, species, "vtk" + physical * "_phy")
                point_data_n[species] = {}
                for name, data in vars.items():
                    points_list = data[t]

                    # scalar
                    if len(points_list) == 1:
                        point_data_n[species][name] = points_list[0]

                    # vectorpoint_data[name]
                    else:
                        for j in range(3):
                            point_data_n[species][name + f"_{j + 1}"] = points_list[j]

                gridToVTK(
                    os.path.join(species_path, "step_{0:0{1}d}".format(n, log_nt)),
                    *grids_phy,
                    pointData=point_data_n[species],
                )

    def _post_process_markers(
        self,
        path_kinetic_species: str,
        step: int = 1,
        verbose: bool = False,
    ):
        """Computes the Cartesian (x, y, z) coordinates of saved markers during a simulation
        and writes them to a .npy files and to .txt files.
        Also saves the weights.

        * ``.npy`` files:

        * Particles6D:

            ===== ===== ============== ============= ======
            index | 0 | | 1 | 2 | 3 |  | 4 | 5 | 6 | | 7 |
            ===== ===== ============== ============= ======
            value  ID   position (xyz)  velocities   weight
            ===== ===== ============== ============= ======

        * Particles5D:

            ===== ===== ================ ========== ====== ====== ============
            index | 0 | | 1 | 2 | | 3 |      4        5    | 6 |  7
            ===== ===== ================ ========== ====== ====== ============
            value  ID   guiding_center   v_parallel v_perp weight magn. moment
            ===== ===== ================ ========== ====== ====== ============

        * Particles3D:

            ===== ===== ============== ======
            index | 0 | | 1 | 2 | 3 |  | 4 |
            ===== ===== ============== ======
            value  ID   position (xyz) weight
            ===== ===== ============== ======

        * ``.txt`` files :

        ===== ===== ============== ======
        index | 0 | | 1 | 2 | 3 |  | 4 |
        ===== ===== ============== ======
        value  ID   position (xyz) weight
        ===== ===== ============== ======

        ``.txt`` files can be imported to e.g. Paraview, see `08 - Kinetic data <file:///home/spossann/git_repos/struphy/doc/_build/html/tutorials/tutorial_08_struphy_data_pproc.html#Kinetic-data>`_ for details.

        Parameters
        ----------
        path_kinetic_species : str
            Path to kinetic data of considered species.

        step : int, optional
            Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.
        """
        
        species = path_kinetic_species.split("/")[-1]
        species_obj: ParticleSpecies = self.model.particle_species[species]
        
        # open hdf5 files and get names and number of saved markers of kinetic species
        with h5py.File(os.path.join(self.env.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            # get number of time steps and markers
            nt, n_markers, n_cols = file_0["kinetic/" + species + "/markers"].shape
        
        # get velocity dimension from one of the variables of the species
        for varname, var in species_obj.variables.items():
            assert isinstance(var, PICVariable | SPHVariable)
            obj: Particles = var.particles
            vdim = obj.vdim
            break

        log_nt = int(xp.log10(int(((nt - 1) / step)))) + 1

        # directory for .txt files and marker index which will be saved
        path_orbits = os.path.join(path_kinetic_species, "orbits")

        if vdim == 2:
            save_index = list(range(0, 6)) + [10] + [-1]
        elif vdim == 3:
            save_index = list(range(0, 7)) + [-1]
        else:
            save_index = list(range(0, 4)) + [-1]

        try:
            os.mkdir(path_orbits)
        except:
            shutil.rmtree(path_orbits)
            os.mkdir(path_orbits)

        # temporary array
        temp = xp.empty((n_markers, len(save_index)), order="C")
        lost_particles_mask = xp.empty(n_markers, dtype=bool)

        print(f"Evaluation of {n_markers} marker orbits for {species}")

        # loop over time grid
        for n in tqdm(range(int((nt - 1) / step) + 1)):
            # clear buffer
            temp[:, :] = 0.0

            # create text file for this time step and this species
            file_npy = os.path.join(
                path_orbits,
                species + "_{0:0{1}d}.npy".format(n, log_nt),
            )
            file_txt = os.path.join(
                path_orbits,
                species + "_{0:0{1}d}.txt".format(n, log_nt),
            )

            for i in range(int(self.comm_size)):
                with h5py.File(os.path.join(self.env.path_out, "data/", f"data_proc{i}.hdf5"), "r") as file:
                    markers = file["kinetic/" + species + "/markers"]
                    ids = markers[n * step, :, -1].astype("int")
                    ids = ids[ids != -1]  # exclude holes
                    temp[ids] = markers[n * step, : ids.size, save_index]

            # sorting out lost particles
            ids = temp[:, -1].astype("int")
            ids_lost_particles = xp.setdiff1d(xp.arange(n_markers), ids)
            ids_removed_particles = xp.nonzero(temp[:, 0] == -1.0)[0]
            ids_lost_particles = xp.array(list(set(ids_lost_particles) | set(ids_removed_particles)), dtype=int)
            lost_particles_mask[:] = False
            lost_particles_mask[ids_lost_particles] = True

            if len(ids_lost_particles) > 0:
                # lost markers are saved as [0, ..., 0, ids]
                temp[lost_particles_mask, -1] = ids_lost_particles
                ids = xp.unique(xp.append(ids, ids_lost_particles))

            assert xp.all(sorted(ids) == xp.arange(n_markers))

            # compute physical positions (x, y, z)
            pos_phys = self.domain(xp.array(temp[~lost_particles_mask, :3]), change_out_order=True)
            temp[~lost_particles_mask, :3] = pos_phys

            # save numpy
            xp.save(file_npy, temp)
            # move ids to first column and save txt
            temp = xp.roll(temp, 1, axis=1)
            xp.savetxt(file_txt, temp[:, (0, 1, 2, 3, -1)], fmt="%12.6f", delimiter=", ")

    def _post_process_f(
        self,
        path_kinetic_species,
        step=1,
        compute_bckgr=False,
        verbose: bool=False,
    ):
        """Computes and saves distribution functions of saved binning data during a simulation.

        Parameters
        ----------
        path_kinetic_species : str
            Path to kinetic data of considered species.

        step : int, optional
            Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

        compute_bckgr : bool
            Whether to compute the kinetic background values and add them to the binning data.
            This is used if non-standard weights are binned.
        """
        species = path_kinetic_species.split("/")[-1]
        species_obj: ParticleSpecies = self.model.particle_species[species]

        # directory for .npy files
        path_distr = os.path.join(path_kinetic_species, "distribution_function")

        try:
            os.mkdir(path_distr)
        except:
            shutil.rmtree(path_distr)
            os.mkdir(path_distr)

        print("Evaluation of distribution functions for " + str(species))

        # Create grids
        with h5py.File(os.path.join(self.env.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            for slice_name in tqdm(file_0["kinetic/" + species + "/f"]):
                # create a new folder for each slice
                path_slice = os.path.join(path_distr, slice_name)
                os.mkdir(path_slice)

                # Find out all names of slices
                slice_names = slice_name.split("_")

                # save grid
                for n_gr, (_, grid) in enumerate(file_0["kinetic/" + species + "/f/" + slice_name].attrs.items()):
                    grid_path = os.path.join(
                        path_slice,
                        "grid_" + slice_names[n_gr] + ".npy",
                    )
                    xp.save(grid_path, grid[:])

            # compute distribution function
            for slice_name in tqdm(file_0["kinetic/" + species + "/f"]):
                # path to folder of slice
                path_slice = os.path.join(path_distr, slice_name)

                # Find out all names of slices
                slice_names = slice_name.split("_")

                # load full-f data
                data = file_0["kinetic/" + species + "/f/" + slice_name][::step].copy()
                data_df = file_0["kinetic/" + species + "/df/" + slice_name][::step].copy()
                for rank in range(1, int(self.comm_size)):
                    with h5py.File(os.path.join(self.env.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                        data += file["kinetic/" + species + "/f/" + slice_name][::step]
                        data_df += file["kinetic/" + species + "/df/" + slice_name][::step]

                # save distribution functions
                xp.save(os.path.join(path_slice, "f_binned.npy"), data)
                xp.save(os.path.join(path_slice, "delta_f_binned.npy"), data_df)

                if compute_bckgr:
                    # bckgr_params = params["kinetic"][species]["background"]

                    # f_bckgr = None
                    # for fi, maxw_params in bckgr_params.items():
                    #     if fi[-2] == "_":
                    #         fi_type = fi[:-2]
                    #     else:
                    #         fi_type = fi

                    #     if f_bckgr is None:
                    #         f_bckgr = getattr(maxwellians, fi_type)(
                    #             maxw_params=maxw_params,
                    #         )
                    #     else:
                    #         f_bckgr = f_bckgr + getattr(maxwellians, fi_type)(
                    #             maxw_params=maxw_params,
                    #         )
                    
                    for _, var in species_obj.variables.items():
                        assert isinstance(var, PICVariable | SPHVariable)
                        f_bckgr: KineticBackground = var.backgrounds
                        break

                    # load all grids of the variables of f
                    grid_tot = []
                    factor = 1.0

                    # eta-grid
                    for comp in range(1, 4):
                        current_slice = "e" + str(comp)
                        filename = os.path.join(
                            path_slice,
                            "grid_" + current_slice + ".npy",
                        )

                        # check if file exists and is in slice_name
                        if os.path.exists(filename) and current_slice in slice_names:
                            grid_tot += [xp.load(filename)]

                        # otherwise evaluate at zero
                        else:
                            grid_tot += [xp.zeros(1)]

                    # v-grid
                    for comp in range(1, f_bckgr.vdim + 1):
                        current_slice = "v" + str(comp)
                        filename = os.path.join(
                            path_slice,
                            "grid_" + current_slice + ".npy",
                        )

                        # check if file exists and is in slice_name
                        if os.path.exists(filename) and current_slice in slice_names:
                            grid_tot += [xp.load(filename)]

                        # otherwise evaluate at zero
                        else:
                            grid_tot += [xp.zeros(1)]
                            # correct integrating out in v-direction, TODO: check for 5D Maxwellians
                            factor *= xp.sqrt(2 * xp.pi)

                    grid_eval = xp.meshgrid(*grid_tot, indexing="ij")

                    data_bckgr = f_bckgr(*grid_eval).squeeze()

                    # correct integrating out in v-direction
                    data_bckgr *= factor

                    # Now all data is just the data for delta_f
                    data_delta_f = data_df

                    # save distribution function
                    xp.save(os.path.join(path_slice, "delta_f_binned.npy"), data_delta_f)
                    # add extra axis for data_bckgr since data_delta_f has axis for time series
                    xp.save(
                        os.path.join(path_slice, "f_binned.npy"),
                        data_delta_f + data_bckgr[tuple([None])],
                    )

    def _post_process_n_sph(
        self,
        path_kinetic_species,
        step=1,
        verbose: bool=False,
    ):
        """Computes and saves the density n of saved sph data during a simulation.

        Parameters
        ----------
        path_kinetic_species : str
            Path to kinetic data of considered species.

        step : int, optional
            Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.
        """
        species = path_kinetic_species.split("/")[-1]

        # directory for .npy files
        path_n_sph = os.path.join(path_kinetic_species, "n_sph")

        try:
            os.mkdir(path_n_sph)
        except:
            shutil.rmtree(path_n_sph)
            os.mkdir(path_n_sph)

        print("Evaluation of sph density for " + str(species))

        with h5py.File(os.path.join(self.env.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            # Create grids
            for i, view in enumerate(file_0["kinetic/" + species + "/n_sph"]):
                # create a new folder for each view
                path_view = os.path.join(path_n_sph, view)
                os.mkdir(path_view)

                # build meshgrid and save
                eta1 = file_0["kinetic/" + species + "/n_sph/" + view].attrs["eta1"]
                eta2 = file_0["kinetic/" + species + "/n_sph/" + view].attrs["eta2"]
                eta3 = file_0["kinetic/" + species + "/n_sph/" + view].attrs["eta3"]

                ee1, ee2, ee3 = xp.meshgrid(
                    eta1,
                    eta2,
                    eta3,
                    indexing="ij",
                )

                grid_path = os.path.join(
                    path_view,
                    "grid_n_sph.npy",
                )
                xp.save(grid_path, (ee1, ee2, ee3))

                # load n_sph data
                data = file_0["kinetic/" + species + "/n_sph/" + view][::step].copy()
                for rank in range(1, int(self.comm_size)):
                    with h5py.File(os.path.join(self.env.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                        data += file["kinetic/" + species + "/n_sph/" + view][::step]

                # save distribution functions
                xp.save(os.path.join(path_view, "n_sph.npy"), data)
