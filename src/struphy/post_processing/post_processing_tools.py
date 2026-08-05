import inspect
import json
import logging
import os
import pickle
import shutil
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING

import cunumpy as xp
import h5py
import yaml
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from pyevtk.hl import gridToVTK

from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.fields_background.base import FluidEquilibrium
from struphy.geometry.base import Domain
from struphy.io.options import BaseUnits, DerhamOptions, EnvironmentOptions, Time
from struphy.io.setup import import_parameters_py
from struphy.kinetic_background import maxwellians
from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.base import Particles
from struphy.post_processing.likwid.plot_time_traces import plot_gantt_chart_plotly, plot_time_vs_duration
from struphy.post_processing.orbits import orbits_tools
from struphy.topology.grids import TensorProductGrid
from struphy.utils.progress import tqdm

if TYPE_CHECKING:
    from struphy.simulation.sim import Simulation

logger = logging.getLogger("struphy")

# push-forward of each de Rham space to Cartesian components, see Domain.push
PUSH_KINDS = {"H1": "0", "Hcurl": "1", "Hdiv": "2", "L2": "3", "H1vec": "v"}


class SplineValues:
    def __str__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out


class Orbits:
    def __str__(self):
        out = ""
        for species, orbits in self.__dict__.items():
            shp = orbits.shape
            out += f"    {species}, shape = {shp}\n"
            out += f"        Number of time points: {shp[0]}\n"
            out += f"        Number of particles:   {shp[1]}\n"
            out += f"        Number of attributes:  {shp[2]}\n"
        return out


class DistributionFunction:
    def __str__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out


class DensitySPH:
    def __str__(self):
        out = ""
        for name, species in inspect.getmembers(self):
            if isinstance(species, SpecHolder):
                out += f"    {name}\n"
                out += f"{species}"
        return out


class SpecHolder:
    def __str__(self):
        out = ""
        for name, val in self.__dict__.items():
            out += f"        {name}\n"
        return out


class Slice:
    pass


class DataDict:
    def __init__(self, data: dict):
        self.data = data

    def __str__(self):
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
        logger.info(f"\nReading in parameters from {path} ... ")

        params_path = os.path.join(path, "parameters.py")
        json_path = os.path.join(path, "config.json")

        if os.path.exists(params_path):
            params_in = import_parameters_py(params_path)
            env = params_in.env
            time_opts = params_in.time_opts
            domain = params_in.domain
            equil = params_in.equil
            grid = params_in.grid
            derham_opts = params_in.derham_opts
            model = params_in.model
            sim = params_in.sim

        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                dct = json.load(f)
            env = EnvironmentOptions.from_dict(dct["env"])
            time_opts = Time.from_dict(dct["time_opts"])
            domain: Domain = Domain.from_dict(dct["domain"])
            equil = FluidEquilibrium.from_dict(dct.get("equil"))

            grid_dct = dct.get("grid")
            if grid_dct is not None:
                grid_dct = dict(grid_dct)
                if "num_elements" in grid_dct and grid_dct["num_elements"] is not None:
                    grid_dct["num_elements"] = tuple(grid_dct["num_elements"])
                if "mpi_dims_mask" in grid_dct and grid_dct["mpi_dims_mask"] is not None:
                    grid_dct["mpi_dims_mask"] = tuple(grid_dct["mpi_dims_mask"])
                grid = TensorProductGrid.from_dict(grid_dct)
            else:
                grid = None

            derham_dct = dct.get("derham_opts")
            if derham_dct is not None:
                derham_dct = dict(derham_dct)
                if "degree" in derham_dct and derham_dct["degree"] is not None:
                    derham_dct["degree"] = tuple(derham_dct["degree"])
                if "bcs" in derham_dct and derham_dct["bcs"] is not None:
                    derham_dct["bcs"] = tuple(None if bc is None else tuple(bc) for bc in derham_dct["bcs"])
                if "nquads" in derham_dct and derham_dct["nquads"] is not None:
                    derham_dct["nquads"] = tuple(derham_dct["nquads"])
                if "nquads_proj" in derham_dct and derham_dct["nquads_proj"] is not None:
                    derham_dct["nquads_proj"] = tuple(derham_dct["nquads_proj"])
                derham_opts = DerhamOptions.from_dict(derham_dct)
            else:
                derham_opts = None

            model: StruphyModel = StruphyModel.from_dict(dct["model"])
            sim = None

        else:
            raise FileNotFoundError(f"Neither of the paths {params_path} or {json_path} exists.")

        logger.info("... Done.")

        self.env = env
        self.time_opts = time_opts
        self.domain = domain
        self.equil = equil
        self.grid = grid
        self.derham_opts = derham_opts
        self.model = model
        self.sim = sim


class PostProcessor:
    """Post-process results from a finished Struphy simulation.

    This class collects and processes output data produced by a completed Struphy run. It can be
    constructed either from a finished :class:`Simulation` object or from a path to an output
    directory produced by a previous run.

    Parameters
    ----------
    sim : Simulation, optional
        Simulation object of a finished run. If provided, its metadata and output paths are used.
    path_out : str, optional
        Path to the Struphy output folder. Required if ``sim`` is not given.
    parallel_pproc : bool, optional
        Whether to run post-processing in parallel using MPI. Default is False (serial post-processing).

    Attributes
    ----------
    path_out : str
        Path to simulation output folder.
    path_pproc : str
        Path to the post-processing directory inside ``path_out``.
    derham : object or None
        Helper used to reconstruct FEEC spline fields.
    domain : Domain
        Computational domain used to map logical -> physical coordinates.
    model : StruphyModel
        Model instance describing species and variables.
    comm_size : int
        Number of MPI ranks used to produce the output.
    """

    def __init__(
        self,
        sim: "Simulation" = None,
        path_out: str = None,
        parallel_pproc: bool = False,
    ):

        # import simulation parameters from sim object or from path_out
        if sim is None:
            assert path_out is not None, (
                "If no sim object is provided, a path_out must be given to retrieve the parameters of the run to post-process."
            )
            params_in = ParamsIn(path=path_out)
            grid = params_in.grid
            derham_opts = params_in.derham_opts
            domain = params_in.domain
            model = params_in.model
            imported_sim = params_in.sim
        else:
            path_out = sim.env.path_out
            grid = sim.grid
            derham_opts = sim.derham_opts
            domain = sim.domain
            model = sim.model
            imported_sim = sim

        # create post-processing folder
        self.path_out = path_out
        self.path_pproc = os.path.join(path_out, "post_processing")

        # parallel post-processing (default: False)
        self.parallel_pproc = parallel_pproc

        # struphy objects needed for post-processing
        self.domain = domain
        self.model = model

        if self.parallel_pproc:
            assert imported_sim is not None, "Parallel post-processing only supported when the sim object is provided."
            self.derham = imported_sim.derham
            self.comm = self.derham.comm
            self.comm_size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.range_ranks = range(self.rank, self.rank + 1)
        else:
            if grid is None or derham_opts is None:
                self.derham = None
            else:
                self.derham = Derham(
                    grid,
                    derham_opts,
                    comm=None,
                    domain=domain,
                )
            self.comm = MockComm()
            # get number of MPI ranks used in the simulation from meta.yml
            with open(os.path.join(path_out, "meta.yml"), "r") as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            self.comm_size = meta["MPI processes"]
            self.rank = 0
            self.range_ranks = range(int(self.comm_size))

        # create or remove output paths
        if self.rank == 0:
            try:
                os.mkdir(self.path_pproc)
            except:
                shutil.rmtree(self.path_pproc)
                os.mkdir(self.path_pproc)
        self.comm.Barrier()

    def plot_time_traces(self):
        path_time_trace = os.path.join(self.path_out, "profiling_time_trace.pkl")
        plot_time_vs_duration(path_time_trace, output_path=self.path_pproc)
        plot_gantt_chart_plotly(path_time_trace, output_path=self.path_pproc)
        return

    def process(
        self,
        step: int = 1,
        celldivide: int | Sequence[int] = (1, 1, 1),
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
    ):
        """Run post-processing for fields and particle data in ``self.path_out``.

        Parameters
        ----------
        step : int
            Interval of saved time steps to post-process (1 = every step, 2 = every second step, ...).
        celldivide : int or sequence of int
            Grid refinement factor when evaluating FEM fields (e.g. ``celldivide=(2, 2, 2)`` evaluates two
            points per cell in each logical direction). A single int is applied to all three directions.
        physical : bool
            If True, also compute push-forwarded physical (x,y,z) components of fields.
        guiding_center : bool
            If True, compute guiding-center coordinates for particle orbits (requires
            Particles6D marker data).
        classify : bool
            If True, run orbit classification (passing, trapped, lost) after computing orbits.
        create_vtk : bool
            If True, create VTK files for visualisation.
        """
        logger.warning(f"\nPost-processing path {self.path_out}")

        # check for fields and kinetic data in hdf5 file that need post processing
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            if self.rank == 0:
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
        )

        # particle variables
        self.process_particles(
            step=step,
            guiding_center=guiding_center,
            classify=classify,
        )

    def process_fields(
        self,
        step: int = 1,
        celldivide: int | Sequence[int] = (1, 1, 1),
        physical: bool = False,
        create_vtk: bool = True,
    ):
        """Evaluate the FEEC fields of all saved time steps and write them to disk.

        The time steps are processed one after another: only the spline coefficients of a
        single snapshot are held in memory, and each rank evaluates only those points of the
        evaluation grid that lie in its own MPI domain. Arrays of the size of the global
        evaluation grid therefore only ever exist on rank 0, where they are needed for output.

        Parameters
        ----------
        step : int
            Interval of saved time steps to post-process (1 = every step, 2 = every second step, ...).
        celldivide : int or sequence of int
            Grid refinement factor when evaluating FEM fields. A single int is applied to all
            three directions.
        physical : bool
            If True, also compute push-forwarded physical (x,y,z) components of fields.
        create_vtk : bool
            If True, create VTK files for visualisation.
        """
        if not self.exist_fields:
            logger.warning("\nNo feec fields found in hdf5 file, skipping post-processing of fields.")
            return

        # one set of spline functions, re-used for every time step
        fields, t_grid = self._create_femfields(step=step)

        # evaluation grid; each rank only ever evaluates the points of its own domain
        grids_log, grid_slices = self._create_eval_grids(celldivide=celldivide)
        grids_log_loc = [grid[sl] for grid, sl in zip(grids_log, grid_slices[self.rank])]
        glob_shape = tuple(grid.size for grid in grids_log)

        # the physical grid is only needed for output, hence it is only built on rank 0
        if self.rank == 0:
            grids_phy = list(self.domain(*grids_log))
        else:
            grids_phy = None

        # point_data[species][var][t] stays an empty list on all ranks except rank 0
        point_data = {species: {name: {} for name in vars} for species, vars in fields.items()}
        point_data_phy = {species: {name: {} for name in vars} for species, vars in fields.items()}

        logger.warning("\nEvaluating fields ...")
        with ExitStack() as stack:
            # hdf5 files of the simulation ranks whose data is read by this rank
            files = [
                stack.enter_context(
                    h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r"),
                )
                for rank in self.range_ranks
            ]

            for n, t in enumerate(tqdm(t_grid)):
                self._load_femfields(fields, files, n, step=step)

                vals, vals_phy = self._eval_femfields(
                    fields,
                    grids_log_loc,
                    grid_slices,
                    glob_shape,
                    physical=physical,
                )

                if self.rank == 0:
                    for species, vars in vals.items():
                        for name, val in vars.items():
                            point_data[species][name][t] = val
                            point_data_phy[species][name][t] = vals_phy[species][name]

        # directory for field data
        path_fields = os.path.join(self.path_pproc, "fields_data")

        if self.rank == 0:
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
        self.comm.Barrier()

    def process_particles(
        self,
        step: int = 1,
        guiding_center: bool = False,
        classify: bool = False,
    ):

        if self.exist_particles is None:
            logger.warning("\nNo kinetic data found in hdf5 file, skipping post-processing of kinetic data.")
            return

        # directory for kinetic data
        path_kinetics = os.path.join(self.path_pproc, "kinetic_data")

        if self.rank == 0:
            try:
                os.mkdir(path_kinetics)
            except:
                shutil.rmtree(path_kinetics)
                os.mkdir(path_kinetics)
        self.comm.Barrier()

        # kinetic post-processing for each species
        for n, species in enumerate(self.kinetic_species):
            # directory for each species
            path_kinetics_species = os.path.join(path_kinetics, species)

            if self.rank == 0:
                try:
                    os.mkdir(path_kinetics_species)
                except:
                    shutil.rmtree(path_kinetics_species)
                    os.mkdir(path_kinetics_species)
            self.comm.Barrier()

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

    def _create_femfields(self, step: int = 1):
        """Allocate one FEEC spline field object per saved variable.

        Only a single set of fields is allocated, no matter how many time steps are
        post-processed; the coefficients of the individual snapshots are read into it one
        after another by :meth:`_load_femfields`.

        Parameters
        ----------
        step : int
            Time-step stride when reading saved snapshots (default 1).

        Returns
        -------
        fields : dict
            Nested dictionary mapping species -> variable -> ``SplineFunction``.
        t_grid : xp.ndarray
            Array of times at which the fields were saved.
        """
        # get fields names, space IDs and time grid from 0-th rank hdf5 file
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            space_ids = {}
            logger.warning("\nReading hdf5 data of following species:")
            for species, dset in file["feec"].items():
                space_ids[species] = {}
                logger.warning(f"{species}:")
                for var, ddset in dset.items():
                    space_ids[species][var] = ddset.attrs["space_id"]
                    logger.warning(f"  {var}: {ddset}")

            t_grid = file["time/value"][::step].copy()

        # create one FemField for each variable, re-used for all snapshots
        fields = {}
        for species, vars in space_ids.items():
            fields[species] = {}
            for var, id in vars.items():
                fields[species][var] = self.derham.create_spline_function(
                    var,
                    id,
                )

        logger.warning("Creation of Struphy Fields done.")

        return fields, t_grid

    def _load_femfields(self, fields: dict, files: list, n: int, step: int = 1):
        """Read the spline coefficients of one snapshot into ``fields`` (in-place).

        Parameters
        ----------
        fields : dict
            Nested dictionary species -> variable -> ``SplineFunction``, as returned
            by :meth:`_create_femfields`.
        files : list
            Open hdf5 files, one for each simulation rank processed by this rank.
        n : int
            Index of the snapshot in the (strided) time grid.
        step : int
            Time-step stride of the saved snapshots.
        """
        for file in files:
            for species, dset in file["feec"].items():
                for var, ddset in dset.items():
                    # get global start indices, end indices and pads
                    gl_s = ddset.attrs["starts"]
                    gl_e = ddset.attrs["ends"]
                    pads = ddset.attrs["pads"]

                    assert gl_s.shape == (3,) or gl_s.shape == (3, 3)
                    assert gl_e.shape == (3,) or gl_e.shape == (3, 3)
                    assert pads.shape == (3,) or pads.shape == (3, 3)

                    vector = fields[species][var].vector

                    # scalar field
                    if gl_s.shape == (3,):
                        s1, s2, s3 = gl_s
                        e1, e2, e3 = gl_e
                        p1, p2, p3 = pads

                        vector[
                            s1 : e1 + 1,
                            s2 : e2 + 1,
                            s3 : e3 + 1,
                        ] = ddset[n * step, p1:-p1, p2:-p2, p3:-p3]

                    # vector-valued field
                    else:
                        for comp in range(3):
                            s1, s2, s3 = gl_s[comp]
                            e1, e2, e3 = gl_e[comp]
                            p1, p2, p3 = pads[comp]

                            vector[comp][
                                s1 : e1 + 1,
                                s2 : e2 + 1,
                                s3 : e3 + 1,
                            ] = ddset[str(comp + 1)][n * step, p1:-p1, p2:-p2, p3:-p3]

                    vector.update_ghost_regions()

    def _create_eval_grids(self, celldivide: int | Sequence[int] = (1, 1, 1)):
        """Build the logical evaluation grids and distribute them over the MPI ranks.

        The grid points are split among the ranks exactly as
        :meth:`~struphy.feec.psydac_derham.SplineFunction._flag_pts_not_on_proc` does,
        such that every point is evaluated by exactly one rank. This allows each rank
        to allocate only its own part of the evaluation grid.

        Parameters
        ----------
        celldivide : int or sequence of int
            Refinement factor in each logical direction; a single int is applied to all
            three directions, a sequence must have length three.

        Returns
        -------
        grids_log : list
            The three global logical 1d grids.
        grid_slices : list
            One entry per rank, holding the three slices of ``grids_log`` owned by that rank.
            The slices of all ranks tile the global grid exactly.
        """
        if isinstance(celldivide, int):
            celldivide = (celldivide,) * 3

        assert isinstance(celldivide, Sequence)
        assert len(celldivide) == 3

        num_elements = self.derham.num_elements

        grids_log = [
            xp.linspace(0.0, 1.0, num_elements_i * n_i + 1) for num_elements_i, n_i in zip(num_elements, celldivide)
        ]

        # domain decomposition of the pproc communicator (one row per rank), see Derham.domain_array
        dom_arr = self.derham.domain_array

        grid_slices = []
        for rank in range(dom_arr.shape[0]):
            slices = []
            for n, grid in enumerate(grids_log):
                left = dom_arr[rank, 3 * n + 0]
                right = dom_arr[rank, 3 * n + 1]

                # points on an interior boundary are shifted into the process to the right of it
                shifted = grid.copy()
                if left != 0.0:
                    shifted[shifted == left] += 1e-8
                if right != 1.0:
                    shifted[shifted == right] += 1e-8

                inds = xp.nonzero(xp.logical_and(shifted >= left, shifted <= right))[0]
                assert inds.size > 0, f"Rank {rank} has no evaluation point in direction {n + 1}."
                assert inds.size == inds[-1] - inds[0] + 1, "Evaluation points of a rank must be contiguous."

                slices += [slice(int(inds[0]), int(inds[-1]) + 1)]

            grid_slices += [tuple(slices)]

        # the local grids must tile the global evaluation grid exactly
        n_points = sum(
            (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) * (sl[2].stop - sl[2].start) for sl in grid_slices
        )
        assert n_points == grids_log[0].size * grids_log[1].size * grids_log[2].size, (
            "The MPI domains do not tile the evaluation grid exactly."
        )

        return grids_log, grid_slices

    def _collect_on_root(self, loc_val: xp.ndarray, grid_slices: list, glob_shape: tuple):
        """Assemble the local parts of an evaluation-grid array on rank 0.

        Only rank 0 allocates an array of the size of the global evaluation grid;
        all other ranks just send the points they own.

        Parameters
        ----------
        loc_val : xp.ndarray
            Values on the evaluation points owned by this rank.
        grid_slices : list
            Slices of the global grid owned by each rank, see :meth:`_create_eval_grids`.
        glob_shape : tuple
            Number of points of the global evaluation grid in each direction.

        Returns
        -------
        xp.ndarray or None
            The global array on rank 0, None on all other ranks.
        """
        if not self.parallel_pproc:
            return loc_val

        if self.rank == 0:
            glob_val = xp.empty(glob_shape, dtype=loc_val.dtype)
            glob_val[grid_slices[0]] = loc_val

            # cache receive buffers to avoid repeated allocations in tight loops
            if not hasattr(self, "_collect_recv_bufs"):
                self._collect_recv_bufs = {}

            for rank in range(1, len(grid_slices)):
                sl = grid_slices[rank]
                shape = tuple(sl_i.stop - sl_i.start for sl_i in sl)
                buf = self._collect_recv_bufs.get((rank, shape, loc_val.dtype))
                if buf is None:
                    buf = xp.empty(shape, dtype=loc_val.dtype)
                    self._collect_recv_bufs[(rank, shape, loc_val.dtype)] = buf
                self.comm.Recv(buf, source=rank, tag=rank)
                glob_val[sl] = buf

            return glob_val

        else:
            self.comm.Send(xp.ascontiguousarray(loc_val), dest=0, tag=self.rank)
            return None

    def _eval_femfields(
        self,
        fields: dict,
        grids_log_loc: list,
        grid_slices: list,
        glob_shape: tuple,
        *,
        physical: bool = False,
    ):
        """Evaluate the spline fields of one snapshot on the evaluation grid.

        Each rank evaluates only the grid points of its own MPI domain, the values are
        then collected on rank 0.

        Parameters
        ----------
        fields : dict
            Nested dictionary species -> var -> ``SplineFunction`` holding the coefficients
            of one snapshot, see :meth:`_load_femfields`.
        grids_log_loc : list
            The three logical 1d grids restricted to the domain of this rank.
        grid_slices : list
            Slices of the global grid owned by each rank, see :meth:`_create_eval_grids`.
        glob_shape : tuple
            Number of points of the global evaluation grid in each direction.
        physical : bool, optional
            If True, also compute the push-forwarded physical (x,y,z) components.

        Returns
        -------
        vals, vals_phy : dict
            Nested dictionaries species -> var -> list of arrays (one entry for scalar-valued
            and three entries for vector-valued spaces). The arrays are only assembled on
            rank 0, the lists stay empty on all other ranks. ``vals_phy`` holds empty lists
            if ``physical`` is False.
        """
        vals = {}
        vals_phy = {}
        for species, vars in fields.items():
            vals[species] = {}
            vals_phy[species] = {}
            for name, field in vars.items():
                assert isinstance(field, SplineFunction)

                vals[species][name] = []
                vals_phy[species][name] = []

                # evaluate the field on the grid points of this rank only
                loc_val = field(*grids_log_loc, local=True)

                if physical:
                    # push-forward
                    loc_val_phy = self.domain.push(
                        loc_val,
                        *grids_log_loc,
                        kind=PUSH_KINDS[field.space_id],
                    )

                # scalar spaces
                if isinstance(loc_val, xp.ndarray):
                    comps = [loc_val]
                    comps_phy = [loc_val_phy] if physical else []
                # vector-valued spaces
                else:
                    comps = [loc_val[j] for j in range(3)]
                    comps_phy = [loc_val_phy[j] for j in range(3)] if physical else []

                # collect the values of all ranks on rank 0
                for comp in comps:
                    glob_val = self._collect_on_root(comp, grid_slices, glob_shape)
                    if self.rank == 0:
                        vals[species][name] += [glob_val]

                for comp in comps_phy:
                    glob_val = self._collect_on_root(comp, grid_slices, glob_shape)
                    if self.rank == 0:
                        vals_phy[species][name] += [glob_val]

        return vals, vals_phy

    def _create_vtk(
        self,
        path: str,
        t_grid: xp.ndarray,
        grids_phy: list,
        point_data: dict,
        *,
        physical: bool = False,
    ):
        """Write evaluated field arrays to VTK (.vts) files for visualization.

        Parameters
        ----------
        path : str
            Directory where species subfolders and their `vtk` folders will be created.
        t_grid : xp.ndarray
            Time grid corresponding to entries in ``point_data``.
        grids_phy : list
            Physical coordinate arrays returned by :meth:`_eval_femfields`.
        point_data : dict
            Evaluated field values as returned by :meth:`_eval_femfields`.
        physical : bool, optional
            If True, writes files for push-forwarded physical components (folder suffix "_phy").
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

        logger.warning(f"\nCreating vtk in {path} ...")
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
    ):
        """Compute Cartesian marker positions and write them to .npy and .txt files.

        For each saved time step this function collects marker datasets from all MPI ranks,
        reconstructs full marker arrays (positions, velocities, weights, ids), maps logical
        coordinates to physical coordinates via ``self.domain`` and writes per-step
        ``.npy`` (binary) and ``.txt`` (ASCII) files suitable for quick inspection or
        import into visualization tools.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory where results will be written.
        step : int, optional
            Time-step stride to process (default 1).
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

        if self.rank == 0:
            try:
                os.mkdir(path_orbits)
            except:
                shutil.rmtree(path_orbits)
                os.mkdir(path_orbits)
        self.comm.Barrier()

        # temporary array
        temp = xp.empty((n_markers, len(save_index)), order="C")
        lost_particles_mask = xp.empty(n_markers, dtype=bool)

        logger.warning(f"Evaluation of {n_markers} marker orbits for {species}")

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

            for rank in self.range_ranks:
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    markers = file["kinetic/" + species + "/markers"]
                    ids = markers[n * step, :, -1].astype("int")
                    ids = ids[ids != -1]  # exclude holes
                    temp[ids] = markers[n * step, : ids.size, save_index]

            if self.parallel_pproc:
                if self.rank == 0:
                    self.comm.Reduce(MPI.IN_PLACE, temp, op=MPI.SUM, root=0)
                else:
                    self.comm.Reduce(temp, None, op=MPI.SUM, root=0)

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

            if self.rank == 0:
                # save numpy
                xp.save(file_npy, temp)
                # move ids to first column and save txt
                temp = xp.roll(temp, 1, axis=1)
                xp.savetxt(file_txt, temp[:, (0, 1, 2, 3, -1)], fmt="%12.6f", delimiter=", ")
            self.comm.Barrier()

    def _post_process_f(
        self,
        path_kinetic_species,
        step=1,
        compute_bckgr=False,
    ):
        """Assemble and save distribution functions from per-rank binned data.

        This reads the binned full-f and delta-f arrays produced by the simulation across
        MPI ranks, sums them to global arrays, and stores the results under
        ``<path_kinetic_species>/distribution_function/<slice>``. When ``compute_bckgr`` is
        True, an analytic kinetic background is evaluated on the same grids and added.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory.
        step : int, optional
            Time-step stride to process (default 1).
        compute_bckgr : bool, optional
            If True, compute and add background contribution to the saved binned data.
        """
        print(f"{self.rank} starting post-processing of distribution functions for {path_kinetic_species} ...")

        species = path_kinetic_species.split("/")[-1]
        species_obj: ParticleSpecies = self.model.particle_species[species]

        # directory for .npy files
        path_distr = os.path.join(path_kinetic_species, "distribution_function")

        if self.rank == 0:
            try:
                os.mkdir(path_distr)
            except:
                shutil.rmtree(path_distr)
                os.mkdir(path_distr)
        self.comm.Barrier()

        logger.warning("Evaluation of distribution functions for " + str(species))

        # Create grids
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            slice_names = []
            for slice_name in tqdm(file_0["kinetic/" + species + "/f"]):
                slice_names += [slice_name]
                # create a new folder for each slice
                path_slice = os.path.join(path_distr, slice_name)
                if self.rank == 0:
                    os.mkdir(path_slice)
                self.comm.Barrier()

                # Find out all names of slices
                slice_splits = slice_name.split("_")

                # save grid
                for n_gr, (_, grid) in enumerate(file_0["kinetic/" + species + "/f/" + slice_name].attrs.items()):
                    grid_path = os.path.join(
                        path_slice,
                        "grid_" + slice_splits[n_gr] + ".npy",
                    )
                    if self.rank == 0:
                        xp.save(grid_path, grid[:])
                    self.comm.Barrier()

        # compute distribution function
        for slice_name in tqdm(slice_names):
            logger.info(f"Processing slice {slice_name} for species {species}")
            # path to folder of slice
            path_slice = os.path.join(path_distr, slice_name)

            # Find out all names of slices
            slice_splits = slice_name.split("_")

            for rank in self.range_ranks:
                print(f"{rank = } ----------------------------")
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    if self.parallel_pproc:
                        data = file["kinetic/" + species + "/f/" + slice_name][::step]
                        data_df = file["kinetic/" + species + "/df/" + slice_name][::step]
                    else:
                        if rank == 0:
                            data = file["kinetic/" + species + "/f/" + slice_name][::step].copy()
                            data_df = file["kinetic/" + species + "/df/" + slice_name][::step].copy()
                        else:
                            data += file["kinetic/" + species + "/f/" + slice_name][::step]
                            data_df += file["kinetic/" + species + "/df/" + slice_name][::step]

            print(f"{self.rank =} with {xp.sum(data) =} and {xp.sum(data_df) =}")

            if self.parallel_pproc:
                if self.rank == 0:
                    self.comm.Reduce(
                        MPI.IN_PLACE,
                        data,
                        op=MPI.SUM,
                        root=0,
                    )
                    self.comm.Reduce(
                        MPI.IN_PLACE,
                        data_df,
                        op=MPI.SUM,
                        root=0,
                    )
                else:
                    self.comm.Reduce(
                        data,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )
                    self.comm.Reduce(
                        data_df,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )

            print(f"{self.rank =} with {xp.sum(data) =} and {xp.sum(data_df) =}")

            print(f"{self.rank =} done.")
            if self.rank == 0:
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
                        if os.path.exists(filename) and current_slice in slice_splits:
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
                        if os.path.exists(filename) and current_slice in slice_splits:
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
    ):
        """Compute and save SPH density fields from per-rank outputs.

        Parameters
        ----------
        path_kinetic_species : str
            Path to the per-species kinetic output directory where results will be written.
        step : int, optional
            Time-step stride to process (default 1).
        """
        species = path_kinetic_species.split("/")[-1]

        # directory for .npy files
        path_n_sph = os.path.join(path_kinetic_species, "n_sph")

        if self.rank == 0:
            try:
                os.mkdir(path_n_sph)
            except:
                shutil.rmtree(path_n_sph)
                os.mkdir(path_n_sph)
        self.comm.Barrier()

        logger.warning("Evaluation of sph density for " + str(species))

        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            views = list(file_0["kinetic/" + species + "/n_sph"])

            # Create grids
            for view in views:
                # create a new folder for each view
                path_view = os.path.join(path_n_sph, view)
                if self.rank == 0:
                    os.mkdir(path_view)
                self.comm.Barrier()

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

                if self.rank == 0:
                    grid_path = os.path.join(
                        path_view,
                        "grid_n_sph.npy",
                    )
                    xp.save(grid_path, (ee1, ee2, ee3))

        # compute sph density
        for view in tqdm(views):
            path_view = os.path.join(path_n_sph, view)

            for rank in self.range_ranks:
                with h5py.File(os.path.join(self.path_out, "data/", f"data_proc{rank}.hdf5"), "r") as file:
                    if self.parallel_pproc:
                        data = file["kinetic/" + species + "/n_sph/" + view][::step]
                    else:
                        if rank == 0:
                            data = file["kinetic/" + species + "/n_sph/" + view][::step].copy()
                        else:
                            data += file["kinetic/" + species + "/n_sph/" + view][::step]

            if self.parallel_pproc:
                if self.rank == 0:
                    self.comm.Reduce(
                        MPI.IN_PLACE,
                        data,
                        op=MPI.SUM,
                        root=0,
                    )
                else:
                    self.comm.Reduce(
                        data,
                        None,
                        op=MPI.SUM,
                        root=0,
                    )

            if self.rank == 0:
                # save sph density
                xp.save(os.path.join(path_view, "n_sph.npy"), data)


class PlottingData:
    """Container for loading and accessing post-processed Struphy simulation data.

    This class provides convenient access to field data (spline values), particle orbits,
    distribution functions, and SPH density fields that were generated by
    :class:`PostProcessor`. Data is organized hierarchically by species and variable/view
    and is exposed via read-only properties.

    Parameters
    ----------
    sim : Simulation, optional
        Simulation object of a completed run. If provided, its output path is used.
    path_out : str, optional
        Path to the Struphy output folder. Required if ``sim`` is not given.

    Raises
    ------
    AssertionError
        If neither ``sim`` nor ``path_out`` is provided, or if the post-processing
        directory does not exist (call :meth:`PostProcessor.process` first).

    Attributes
    ----------
    path_pproc : str
        Path to the post-processing directory.
    t_grid : xp.ndarray or None
        Time grid (loaded after calling :meth:`load`).
    grids_log : list of xp.ndarray or None
        Logical coordinate grids (loaded after calling :meth:`load`).
    grids_phy : list of xp.ndarray or None
        Physical coordinate grids (loaded after calling :meth:`load`).

    Examples
    --------
    >>> pdata = PlottingData(path_out=\"/path/to/sim/output\")
    >>> pdata.load()
    >>> # Access particle orbits for species 'electrons'
    >>> orbits_e = pdata.orbits.electrons  # shape: (time, particles, attributes)
    >>> # Access field values
    >>> E_log = pdata.spline_values.electrons.E_log  # logical components
    """

    def __init__(self, sim: "Simulation" = None, path_out: str = None):

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
        """Particle orbit data by species.

        Returns
        -------
        Orbits
            Container where attributes are species names. Each species attribute holds
            a 3D array indexed by (t, p, a): t = time step, p = particle index,
            a = attribute index (id, position_xyz, velocities, weight, etc.).
        """
        return self._orbits

    @property
    def f(self) -> DistributionFunction:
        """Distribution function data by species.

        Returns
        -------
        DistributionFunction
            Container where attributes are species names. Each species holds a dict-like
            object mapping slice names (e.g., 'e1_v1', 'e2_v2') to slice containers,
            which store arrays like 'f_binned', 'delta_f_binned' for plotting.
        """
        return self._f

    @property
    def spline_values(self) -> SplineValues:
        """Field (spline) values by species.

        Returns
        -------
        SplineValues
            Container where attributes are species names. Each species holds a dict-like
            object mapping variable names (e.g., 'E_log', 'B_phy') to ``DataDict``
            objects containing evaluated field arrays on the grid.
        """
        return self._spline_values

    @property
    def n_sph(self) -> DensitySPH:
        """SPH density fields by species.

        Returns
        -------
        DensitySPH
            Container where attributes are species names. Each species holds a dict-like
            object mapping view names (e.g., 'view_0', 'view_1') to slice containers,
            which store arrays like 'n_sph' and associated grids for plotting.
        """
        return self._n_sph

    def load(self):
        """Load all post-processed data from disk into memory.

        Reads binary pickle files (``.bin``) and NumPy archives (``.npy``) from the
        post-processing directory. Populates ``self.t_grid``, ``self.grids_log``,
        ``self.grids_phy``, and all species-dependent data properties (orbits, f,
        spline_values, n_sph).

        Raises
        ------
        FileNotFoundError
            If expected post-processing files are missing.
        NotImplementedError
            If an unexpected data folder structure is encountered.
        """
        logger.warning("\nLoading post-processed plotting data:")
        logger.warning(f"Data path: {self.path_pproc}")

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
                logger.info(f"\nFiles in {path_spec}: {files}")
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
                            # logger.info(f"{file = }")
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
                        # logger.info(f"{slices = }")
                        for sli in slices:
                            s = Slice()
                            setattr(spec_holder, sli, s)
                            # logger.info(f"{sli = }")
                            files = next(sub_wlk)[2]
                            # logger.info(f"{files = }")
                            for file in files:
                                name = file.split(".")[0]
                                tmp = xp.load(os.path.join(path_dat, sli, file))
                                logger.info(f"{name = }")
                                setattr(s, name, tmp)

                    elif "n_sph" in folder:
                        spec_holder = SpecHolder()
                        setattr(self.n_sph, spec, spec_holder)
                        slices = next(sub_wlk)[1]
                        # logger.info(f"{slices = }")
                        for sli in slices:
                            s = Slice()
                            setattr(spec_holder, sli, s)
                            # logger.info(f"{sli = }")
                            files = next(sub_wlk)[2]
                            # logger.info(f"{files = }")
                            for file in files:
                                name = file.split(".")[0]
                                tmp = xp.load(os.path.join(path_dat, sli, file))
                                # logger.info(f"{name = }")
                                setattr(s, name, tmp)

                    else:
                        logger.info(f"{folder =}")
                        raise NotImplementedError

        logger.warning("\nThe following data has been loaded:")
        logger.warning("\ngrids:")
        logger.warning(f"{self.t_grid.shape =}")
        if self.grids_log is not None:
            logger.warning(f"{self.grids_log[0].shape =}")
            logger.warning(f"{self.grids_log[1].shape =}")
            logger.warning(f"{self.grids_log[2].shape =}")
        if self.grids_phy is not None:
            logger.warning(f"{self.grids_phy[0].shape =}")
            logger.warning(f"{self.grids_phy[1].shape =}")
            logger.warning(f"{self.grids_phy[2].shape =}")
        logger.warning("\nself.spline_values:")
        logger.warning(self.spline_values)
        logger.warning("self.orbits:")
        logger.warning(self.orbits)
        logger.warning("self.f:")
        logger.warning(self.f)
        logger.warning("self.n_sph:")
        logger.warning(self.n_sph)
