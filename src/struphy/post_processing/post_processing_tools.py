import os
import pickle
import shutil
from typing import TYPE_CHECKING
import inspect

import cunumpy as xp
import h5py
import yaml
from feectools.ddm.mpi import mpi as MPI
from pyevtk.hl import gridToVTK
from tqdm import tqdm

from struphy.feec.psydac_derham import SplineFunction
from struphy.fields_background import equils
from struphy.fields_background.base import FluidEquilibrium
from struphy.geometry import domains
from struphy.geometry.base import Domain
from struphy.io.options import BaseUnits, EnvironmentOptions, Time
from struphy.io.setup import import_parameters_py, setup_derham
from struphy.kinetic_background import maxwellians
from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.base import Particles
from struphy.post_processing.likwid.plot_time_traces import plot_gantt_chart_plotly, plot_time_vs_duration
from struphy.post_processing.orbits import orbits_tools
from struphy.topology.grids import TensorProductGrid

if TYPE_CHECKING:
    from struphy.simulation.sim import StruphySimulation


class SplineValues:
    def __repr__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out

class Orbits:
    def __repr__(self):
        out = ""
        for species, orbits in self.__dict__.items():
            shp = orbits.shape
            out += f"    {species}, shape = {shp}\n"
            out += f"        Number of time points: {shp[0]}\n"
            out += f"        Number of particles:   {shp[1]}\n"
            out += f"        Number of attributes:  {shp[2]}\n"
        return out

class DistributionFunction:
    def __repr__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out

class DensitySPH:
    def __repr__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out

class SpecHolder:
    def __repr__(self):
        out = ""
        for name, val in self.__dict__.items():
            out += f"        {name}\n"
        return out
    
class Slice:
    pass
    
class DataDict:
    def __init__(self, data: dict):
        self.data = data
        
    def __repr__(self):
        out = f"{type(self.data) = }\n"
        out += f"{len(self.data) = }\n"
        for key, d in self.data.items():
            if isinstance(d, list):
                shp = [comp.shape for comp in d]
            else:
                shp = d.shape
            out += f"{key = }".ljust(25)
            out += f"shape = {shp}\n"
        return out

class ParamsIn:
    """Holds the input parameters of a Struphy simulation as attributes.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder.
    """

    def __init__(
        self,
        path: str,
    ):
        print(f"\nReading in paramters from {path} ... ")

        params_path = os.path.join(path, "parameters.py")
        bin_path = os.path.join(path, "env.bin")

        if os.path.exists(params_path):
            params_in = import_parameters_py(params_path)
            env = params_in.env
            base_units = params_in.base_units
            time_opts = params_in.time_opts
            domain = params_in.domain
            equil = params_in.equil
            grid = params_in.grid
            derham_opts = params_in.derham_opts
            model = params_in.model

        elif os.path.exists(bin_path):
            with open(os.path.join(path, "env.bin"), "rb") as f:
                env = pickle.load(f)
            with open(os.path.join(path, "base_units.bin"), "rb") as f:
                base_units = pickle.load(f)
            with open(os.path.join(path, "time_opts.bin"), "rb") as f:
                time_opts = pickle.load(f)
            with open(os.path.join(path, "domain.bin"), "rb") as f:
                # WORKAROUND: cannot pickle pyccelized classes at the moment
                domain_dct = pickle.load(f)
                domain: Domain = getattr(domains, domain_dct["name"])(**domain_dct["params"])
            with open(os.path.join(path, "equil.bin"), "rb") as f:
                # WORKAROUND: cannot pickle pyccelized classes at the moment
                equil_dct = pickle.load(f)
                if equil_dct:
                    equil: FluidEquilibrium = getattr(equils, equil_dct["name"])(**equil_dct["params"])
                else:
                    equil = None
            with open(os.path.join(path, "grid.bin"), "rb") as f:
                grid = pickle.load(f)
            with open(os.path.join(path, "derham_opts.bin"), "rb") as f:
                derham_opts = pickle.load(f)
            with open(os.path.join(path, "model_class.bin"), "rb") as f:
                model_class: StruphyModel = pickle.load(f)
                model = model_class()

        else:
            raise FileNotFoundError(f"Neither of the paths {params_path} or {bin_path} exists.")

        print("\n... Done.")

        self.env = env
        self.units = base_units
        self.time_opts = time_opts
        self.domain = domain
        self.equil = equil
        self.grid = grid
        self.derham_opts = derham_opts
        self.model = model

class PostProcessor:
    """Post-processing finished Struphy runs, eithr from Simulation object or from output path.

    Parameters
    ----------
    sim : StruphySimulation
        StruphySimulation object of finished run.

    path_out: str
        Path to Struphy output folder (in case no sim is given).
    """

    def __init__(
        self,
        sim: "StruphySimulation" = None,
        path_out: str = None,
    ):

        # create post-processing folder
        if sim is None:
            assert path_out is not None, (
                "If no sim object is provided, a path_out must be given to retrieve the parameters of the run to post-process."
            )
            params_in = ParamsIn(path=path_out)
            grid = params_in.grid
            derham_opts = params_in.derham_opts
            domain = params_in.domain
            model = params_in.model
            with open(os.path.join(path_out, "meta.yml"), "r") as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            comm_size = meta["MPI processes"]
        else:
            path_out = sim.env.path_out
            grid = sim.grid
            derham_opts = sim.derham_opts
            domain = sim.domain
            model = sim.model
            comm_size = sim.comm_size

        self.path_out = path_out
        self.path_pproc = os.path.join(path_out, "post_processing")
        if grid is None or derham_opts is None:
            self.derham = None
        else:
            self.derham = setup_derham(
                grid,
                derham_opts,
                comm=None,
                domain=domain,
            )
        self.domain = domain
        self.model = model
        self.comm_size = comm_size

        try:
            os.mkdir(self.path_pproc)
        except:
            shutil.rmtree(self.path_pproc)
            os.mkdir(self.path_pproc)

    def plot_time_traces(self):
        path_time_trace = os.path.join(self.path_out, "profiling_time_trace.pkl")
        plot_time_vs_duration(path_time_trace, output_path=self.path_pproc)
        plot_gantt_chart_plotly(path_time_trace, output_path=self.path_pproc)
        return

    def process(
        self,
        step: int = 1,
        celldivide: int = 1,
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        verbose: bool = False,
    ):
        """Do post processing of data in self.path_out.

        Parameters
        ----------
        step : int
            Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

        celldivide : int
            Grid refinement in evaluation of FEM fields. E.g. celldivide=2 evaluates two points per grid cell.

        physical : bool
            Wether to do post-processing into push-forwarded physical (xyz) components of fields.

        guiding_center : bool
            Compute guiding-center coordinates (only from Particles6D).

        classify : bool
            Classify guiding-center trajectories (passing, trapped or lost).

        create_vtk : bool
            Whether vtk files should be created.
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nPost-processing path {self.path_out}")

        # check for fields and kinetic data in hdf5 file that need post processing
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            # save time grid at which post-processing data is created
            xp.save(os.path.join(self.path_pproc, "t_grid.npy"), file["time/value"][::step].copy())

            if "feec" in file.keys():
                self.exist_fields = True
            else:
                self.exist_fields = False

            if "kinetic" in file.keys():
                self.exist_particles = {"markers": False, "f": False, "n_sph": False}
                self.kinetic_species = []
                self.kinetic_kinds = []
                for name in file["kinetic"].keys():
                    self.kinetic_species += [name]
                    self.kinetic_kinds += [next(iter(self.model.species[name].variables.values())).space]

                    # check for saved markers
                    if "markers" in file["kinetic"][name]:
                        self.exist_particles["markers"] = True
                    # check for saved distribution function
                    if "f" in file["kinetic"][name]:
                        self.exist_particles["f"] = True
                    # check for saved sph density
                    if "n_sph" in file["kinetic"][name]:
                        self.exist_particles["n_sph"] = True
            else:
                self.exist_particles = None

        # feec variables
        self.process_fields(
            step=step,
            celldivide=celldivide,
            physical=physical,
            create_vtk=create_vtk,
            verbose=verbose,
        )

        # particle variables
        self.process_particles(
            step=step,
            guiding_center=guiding_center,
            classify=classify,
            verbose=verbose,
        )

    def process_fields(
        self,
        step: int = 1,
        celldivide: int = 1,
        physical: bool = False,
        create_vtk: bool = True,
        verbose: bool = False,
    ):
        if not self.exist_fields:
            print("\nNo feec fields found in hdf5 file, skipping post-processing of fields.")
            return

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

    def process_particles(
        self,
        step: int = 1,
        guiding_center: bool = False,
        classify: bool = False,
        verbose: bool = False,
    ):

        if self.exist_particles is None:
            print("\nNo kinetic data found in hdf5 file, skipping post-processing of kinetic data.")
            return

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
                    orbits_tools.post_process_orbit_guiding_center(self.path_out, path_kinetics_species, species)

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
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
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
            with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
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

        Nel = self.derham.Nel

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
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            # get number of time steps and markers
            nt, n_markers, n_cols = file_0["kinetic/" + species + "/markers"].shape

        # get velocity dimension from one of the variables of the species
        for _, var in species_obj.variables.items():
            assert isinstance(var, PICVariable | SPHVariable)
            cls: Particles = var.particles_class
            vdim = cls.vdim
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
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{i}.hdf5"), "r") as file:
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
        verbose: bool = False,
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
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
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
                    with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
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
        verbose: bool = False,
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

        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
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
                    with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                        data += file["kinetic/" + species + "/n_sph/" + view][::step]

                # save distribution functions
                xp.save(os.path.join(path_view, "n_sph.npy"), data)

class PlottingData:
    """Holds post-processed plotting data as attributes.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.
    """

    def __init__(self, sim: "StruphySimulation" = None, path_out: str = None):

        if sim is None:
            assert path_out is not None, (
                "If no sim object is provided, a path_out must be given to retrieve the parameters of the run to post-process."
            )
        else:
            path_out = sim.env.path_out

        self.path_pproc = os.path.join(path_out, "post_processing")
        assert os.path.exists(self.path_pproc), f"Path {self.path_pproc} does not exist, run 'pproc' first?"

        # dictionaries to hold data
        self._orbits = Orbits()
        self._f = DistributionFunction()
        self._spline_values = SplineValues()
        self._n_sph = DensitySPH()
        self.grids_log: list[xp.ndarray] = None
        self.grids_phy: list[xp.ndarray] = None
        self.t_grid: xp.ndarray = None

    @property
    def orbits(self) -> Orbits:
        """Keys: species name. Values: 3d arrays indexed by (n, p, a), where 'n' is the time index, 'p' the particle index and 'a' the attribute index."""
        return self._orbits

    @property
    def f(self) -> DistributionFunction:
        """Keys: species name. Values: dicts of slice names ('e1_v1' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._f

    @property
    def spline_values(self) -> SplineValues:
        """Keys: species name. Values: dicts of variable names with values being 3d arrays on the grid."""
        return self._spline_values

    @property
    def n_sph(self) -> DensitySPH:
        """Keys: species name. Values: dicts of view names ('view_0' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._n_sph

    def load(self, verbose: bool = False):
        """Load data generated during post-processing."""
        print("\nLoading post-processed plotting data:")
        print(f"Data path: {self.path_pproc}")

        # load time grid
        self.t_grid = xp.load(os.path.join(self.path_pproc, "t_grid.npy"))

        # data paths
        path_fields = os.path.join(self.path_pproc, "fields_data")
        path_kinetic = os.path.join(self.path_pproc, "kinetic_data")

        # load point data
        if os.path.exists(path_fields):
            # grids
            with open(os.path.join(path_fields, "grids_log.bin"), "rb") as f:
                self.grids_log = pickle.load(f)
            with open(os.path.join(path_fields, "grids_phy.bin"), "rb") as f:
                self.grids_phy = pickle.load(f)

            # species folders
            species = next(os.walk(path_fields))[1]
            for spec in species:
                spec_holder = SpecHolder()
                setattr(self.spline_values, spec, spec_holder)
                # self.arrays[spec] = {}
                path_spec = os.path.join(path_fields, spec)
                wlk = os.walk(path_spec)
                files = next(wlk)[2]
                print(f"\nFiles in {path_spec}: {files}")
                for file in files:
                    if ".bin" in file:
                        var = file.split(".")[0]
                        with open(os.path.join(path_spec, file), "rb") as f:
                            # try:
                            data_dict = DataDict(pickle.load(f))
                            setattr(spec_holder, var, data_dict)
                            # self.arrays[spec][var] = pickle.load(f)

        if os.path.exists(path_kinetic):
            # species folders
            species = next(os.walk(path_kinetic))[1]
            for spec in species:
                path_spec = os.path.join(path_kinetic, spec)
                wlk = os.walk(path_spec)
                sub_folders = next(wlk)[1]
                for folder in sub_folders:
                    path_dat = os.path.join(path_spec, folder)
                    sub_wlk = os.walk(path_dat)

                    if "orbits" in folder:
                        files = next(sub_wlk)[2]
                        Nt = len(files) // 2
                        n = 0
                        for file in files:
                            # print(f"{file = }")
                            if ".npy" in file:
                                step = int(file.split(".")[0].split("_")[-1])
                                tmp = xp.load(os.path.join(path_dat, file))
                                if n == 0:
                                    arr = xp.zeros((Nt, *tmp.shape), dtype=float)
                                    setattr(self.orbits, spec, arr) 
                                arr[step] = tmp
                                n += 1

                    elif "distribution_function" in folder:
                        spec_holder = SpecHolder()
                        setattr(self.f, spec, spec_holder)
                        slices = next(sub_wlk)[1]
                        # print(f"{slices = }")
                        for sli in slices:
                            s = Slice()
                            setattr(spec_holder, sli, s)
                            # print(f"{sli = }")
                            files = next(sub_wlk)[2]
                            # print(f"{files = }")
                            for file in files:
                                name = file.split(".")[0]
                                tmp = xp.load(os.path.join(path_dat, sli, file))
                                # print(f"{name = }")
                                setattr(s, name, tmp)

                    elif "n_sph" in folder:
                        spec_holder = SpecHolder()
                        setattr(self.n_sph, spec, spec_holder)
                        slices = next(sub_wlk)[1]
                        # print(f"{slices = }")
                        for sli in slices:
                            s = Slice()
                            setattr(spec_holder, sli, s)
                            # print(f"{sli = }")
                            files = next(sub_wlk)[2]
                            # print(f"{files = }")
                            for file in files:
                                name = file.split(".")[0]
                                tmp = xp.load(os.path.join(path_dat, sli, file))
                                # print(f"{name = }")
                                setattr(s, name, tmp)

                    else:
                        print(f"{folder =}")
                        raise NotImplementedError

        print("\nThe following data has been loaded:")
        print("\ngrids:")
        print(f"{self.t_grid.shape =}")
        if self.grids_log is not None:
            print(f"{self.grids_log[0].shape =}")
            print(f"{self.grids_log[1].shape =}")
            print(f"{self.grids_log[2].shape =}")
        if self.grids_phy is not None:
            print(f"{self.grids_phy[0].shape =}")
            print(f"{self.grids_phy[1].shape =}")
            print(f"{self.grids_phy[2].shape =}")
        print("\nself.spline_values:")
        print(self.spline_values)
        print("self.orbits:")
        print(self.orbits)
        print("self.f:")
        print(self.f)
        print("self.n_sph:")
        print(self.n_sph)


