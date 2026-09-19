import hashlib
import json
import logging
import os
import shutil
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING

import cunumpy as xp
import h5py
import xarray as xr
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from pyevtk.hl import gridToVTK

from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable, SPHVariable
from struphy.pic.base import Particles
from struphy.post_processing import store
from struphy.post_processing.arrays import wrap_binned_data, wrap_field_data, wrap_orbits
from struphy.post_processing.orbits import orbits_tools
from struphy.utils.progress import tqdm

if TYPE_CHECKING:
    from struphy.post_processing.output import Output

logger = logging.getLogger("struphy")

# push-forward of each de Rham space to Cartesian components, see Domain.push
PUSH_KINDS = {"H1": "0", "Hcurl": "1", "Hdiv": "2", "L2": "3", "H1vec": "v"}


MANIFEST_SCHEMA_VERSION = 1


def source_fingerprint(path_out: str) -> str:
    """Fingerprint the raw run files that determine post-processing products."""
    digest = hashlib.sha256()
    for name in ("config.json", "run_metadata.json", "meta.yml", "data/data_proc0.hdf5"):
        path = os.path.join(path_out, name)
        if not os.path.exists(path):
            continue
        stat = os.stat(path)
        digest.update(name.encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        if name != "data/data_proc0.hdf5":
            with open(path, "rb") as stream:
                digest.update(stream.read())
    return digest.hexdigest()


def normalize_options(**options) -> dict:
    """JSON-comparable processing options, as stored in the manifest."""
    celldivide = options.get("celldivide")
    if celldivide is not None:
        options["celldivide"] = [int(celldivide)] * 3 if isinstance(celldivide, int) else [int(c) for c in celldivide]
    return options


def is_processed(path_out: str, options: dict | None = None) -> bool:
    """Whether ``path_out`` holds complete post-processing of its current raw output.

    With ``options``, the stored processing options must match as well, so a request for
    different products (e.g. ``physical=True``) is never answered with stale ones.
    """
    path = os.path.join(path_out, "post_processing", "manifest.json")
    try:
        with open(path) as stream:
            manifest = json.load(stream)
    except (OSError, ValueError):
        return False
    return (
        manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("source_fingerprint") == source_fingerprint(path_out)
        and (options is None or manifest.get("options") == normalize_options(**options))
    )


class PostProcessor:
    """Post-process the raw output of a finished Struphy simulation.

    Use :meth:`from_output` to reconstruct a serial processor from a saved run.
    For automatic MPI rank handling, use :meth:`struphy.Output.process`.

    Parameters
    ----------
    output : Output
        Output folder and lazily reconstructed configuration of the saved run.
    parallel_pproc : bool, optional
        Whether to run post-processing in parallel using MPI. This requires the same
        number of ranks as the saved run and a call on every rank. Default is False (serial post-processing).

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

    def __init__(self, output: "Output", parallel_pproc: bool = False):
        self.path_out = str(output.path_out)
        self.path_pproc = os.path.join(self.path_out, "post_processing")
        self.parallel_pproc = parallel_pproc
        self.domain = output.domain
        self.equil = output.equil
        self.model = output.model
        self.comm_size = output.mpi_ranks
        self.comm = output.comm if parallel_pproc else MockComm()
        self.rank = self.comm.Get_rank()
        if parallel_pproc and self.comm.Get_size() != self.comm_size:
            raise ValueError("Parallel post-processing requires the same number of MPI ranks as the saved run.")
        self.range_ranks = range(self.rank, self.rank + 1) if parallel_pproc else range(self.comm_size)
        self.derham = None
        if output.grid is not None and output.derham_opts is not None:
            self.derham = Derham(
                output.grid,
                output.derham_opts,
                comm=self.comm if parallel_pproc else None,
                domain=self.domain,
            )

        # the directory is only cleared in process(), so that constructing a
        # PostProcessor to inspect a run does not destroy its post-processed data
        if self.rank == 0:
            os.makedirs(self.path_pproc, exist_ok=True)
        self.comm.Barrier()

    @classmethod
    def from_output(cls, path_out: str | os.PathLike) -> "PostProcessor":
        """Create a serial processor from a saved output folder.

        Reads ``run_metadata.json`` (or legacy ``config.json`` when absent), without
        executing the parameter file or allocating a simulation. The folder may
        have been moved. Existing post-processing products are preserved until
        :meth:`process` is called. Under MPI, call this on one rank only.
        """
        from struphy.post_processing.output import Output

        return cls(Output(path_out))

    def _write_manifest(self, status, *, options=None, error=None):
        if self.rank != 0:
            return
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": status,
            "source_fingerprint": source_fingerprint(self.path_out),
            "options": options or {},
        }
        if error is not None:
            manifest["error"] = str(error)
        if status == "complete":
            manifest["products"] = sorted(
                os.path.relpath(os.path.join(root, name), self.path_pproc)
                for root, _, files in os.walk(self.path_pproc)
                for name in files
                if name != "manifest.json"
            )
        path = os.path.join(self.path_pproc, "manifest.json")
        temporary = path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    def _reset_pproc_dir(self):
        if self.rank == 0:
            if os.path.exists(self.path_pproc):
                shutil.rmtree(self.path_pproc)
            os.mkdir(self.path_pproc)
        self.comm.Barrier()

    def process(
        self,
        step: int = 1,
        celldivide: int | Sequence[int] = (1, 1, 1),
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        force: bool = False,
    ):
        """Output post-processing for fields and particle data in ``self.path_out``.

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
        force : bool
            Reprocess even when output already exists. Set False to reuse a previous
            run's results, so a plotting script can be re-run cheaply.

        Returns
        -------
        bool
            Whether post-processing actually ran.
        """
        options = normalize_options(
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
            create_vtk=create_vtk,
        )
        if not force and is_processed(self.path_out, options):
            logger.warning(f"\nReusing existing post-processing in {self.path_pproc}")
            return False

        self._reset_pproc_dir()
        if self.rank == 0:
            store.create(store.store_path(self.path_pproc), options=json.dumps(options))
        self.comm.Barrier()
        self._write_manifest("processing", options=options)
        logger.warning(f"\nPost-processing path {self.path_out}")

        # check for fields and kinetic data in hdf5 file that need post processing
        with h5py.File(os.path.join(self.path_out, "data/", "data_proc0.hdf5"), "r") as file:
            if self.rank == 0:
                # save time grid at which post-processing data is created
                xp.save(os.path.join(self.path_pproc, "t_grid.npy"), file["time/value"][::step].copy())
            self.t_grid = xp.asarray(file["time/value"][::step])

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
        try:
            self.process_fields(step=step, celldivide=celldivide, physical=physical, create_vtk=create_vtk)
            self.process_particles(step=step, guiding_center=guiding_center, classify=classify)
        except Exception as error:
            self._write_manifest("failed", options=options, error=error)
            raise

        self._write_manifest("complete", options=options)

        return True

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

        # directory for the vtk files
        path_fields = os.path.join(self.path_pproc, "fields_data")

        if self.rank == 0:
            # one group per species in the product store, with the mapped grids as coordinates
            for species, vars in point_data.items():
                variables = {}
                for name, val in vars.items():
                    variables[name] = wrap_field_data(val, grids_log, grids_phy=grids_phy, name=name)
                    if physical:
                        variables[name + "_xyz"] = wrap_field_data(
                            point_data_phy[species][name], grids_log, grids_phy=grids_phy, name=name + "_xyz"
                        )
                store.write_group(store.store_path(self.path_pproc), f"/{species}", xr.Dataset(variables))

            if create_vtk:
                try:
                    os.mkdir(path_fields)
                except FileExistsError:
                    shutil.rmtree(path_fields)
                    os.mkdir(path_fields)
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
                    orbits_tools.post_process_orbit_guiding_center(
                        self.domain, self.equil, path_kinetics_species, species
                    )

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
            if os.path.exists(species_path):
                shutil.rmtree(species_path)
            os.makedirs(species_path)

        # time loop
        nt = max(len(t_grid) - 1, 1)
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

        # temporary array, plus every step of it for the product store
        temp = xp.empty((n_markers, len(save_index)), order="C")
        orbits = []
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
                orbits.append(temp.copy())
                # save numpy
                xp.save(file_npy, temp)
                # move ids to first column and save txt
                temp = xp.roll(temp, 1, axis=1)
                xp.savetxt(file_txt, temp[:, (0, 1, 2, 3, -1)], fmt="%12.6f", delimiter=", ")
            self.comm.Barrier()

        if self.rank == 0:
            values = wrap_orbits(xp.stack(orbits), self.t_grid[: len(orbits)])
            store.write_group(store.store_path(self.path_pproc), f"/{species}", xr.Dataset({"orbits": values}))

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
            If True, add the background stored by the simulation to the binned delta f.
        """
        print(f"{self.rank} starting post-processing of distribution functions for {path_kinetic_species} ...")

        species = path_kinetic_species.split("/")[-1]

        logger.warning("Evaluation of distribution functions for " + str(species))

        # the bin centers of every slice, as saved by the simulation
        slice_grids = {}
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            for slice_name in tqdm(file_0["kinetic/" + species + "/f"]):
                dims = [part for part in slice_name.split("_")]
                centers = [grid[:] for _, grid in file_0["kinetic/" + species + "/f/" + slice_name].attrs.items()]
                slice_grids[slice_name] = dict(zip(dims, centers))
        slice_names = list(slice_grids)

        # compute distribution function
        for slice_name in tqdm(slice_names):
            logger.info(f"Processing slice {slice_name} for species {species}")
            grids = slice_grids[slice_name]

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
                full_f = data
                if compute_bckgr:
                    # the background of a delta-f species is stored by the simulation on the bin centers
                    key_background = f"kinetic/{species}/f_background/{slice_name}"
                    with h5py.File(os.path.join(self.path_out, "data", "data_proc0.hdf5"), "r") as file:
                        if key_background not in file:
                            raise ValueError(
                                f"{key_background} is missing from the raw output; outputs of older versions "
                                "do not store the background of delta-f species."
                            )
                        data_bckgr = file[key_background][()]

                    # add extra axis for data_bckgr since data_df has axis for time series
                    full_f = data_df + data_bckgr[None]

                store.write_group(
                    store.store_path(self.path_pproc),
                    f"/{species}/{slice_name}",
                    self._binned_dataset(grids, {"f": full_f, "delta_f": data_df}),
                )

    def _binned_dataset(self, grids: dict, variables: dict) -> xr.Dataset:
        """One binned product per variable, with time, bin centers and mapped coordinates."""
        dims = tuple(dim for dim in grids)
        coords = {"t": self.t_grid, **grids}
        coords.update(self._mapped_coords(grids))
        return xr.Dataset(
            {name: wrap_binned_data(values, dims, coords, name=name) for name, values in variables.items()}
        )

    def _mapped_coords(self, grids: dict) -> dict:
        """``X``, ``Y``, ``Z`` on the logical directions of ``grids``, when there are two or three."""
        logical = tuple(dim for dim in grids if dim in ("e1", "e2", "e3"))
        if len(logical) not in (2, 3) or self.domain is None:
            return {}
        try:
            if len(logical) == 2:
                mesh = xp.meshgrid(*(xp.asarray(grids[dim]) for dim in logical), indexing="ij")
                arguments = {"e1": 0.5, "e2": 0.0, "e3": 0.0}
                arguments.update(dict(zip(logical, mesh)))
                mapped = self.domain(arguments["e1"], arguments["e2"], arguments["e3"], squeeze_out=True)
            else:
                mapped = self.domain(*(xp.asarray(grids[dim]) for dim in logical))
        except (TypeError, ValueError):
            logger.debug("Could not map the coordinates of %s", logical, exc_info=True)
            return {}
        return {name: (logical, xp.asarray(grid)) for name, grid in zip(("X", "Y", "Z"), mapped)}

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

        logger.warning("Evaluation of sph density for " + str(species))

        # the evaluation points of every view, as saved by the simulation
        view_grids = {}
        with h5py.File(os.path.join(self.path_out, "data/data_proc0.hdf5"), "r") as file_0:
            for view in file_0["kinetic/" + species + "/n_sph"]:
                attrs = file_0["kinetic/" + species + "/n_sph/" + view].attrs
                view_grids[view] = {f"e{direction}": attrs["eta" + direction][:] for direction in ("1", "2", "3")}
        views = list(view_grids)

        # compute sph density
        for view in tqdm(views):
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
                store.write_group(
                    store.store_path(self.path_pproc),
                    f"/{species}/{view}",
                    self._binned_dataset(view_grids[view], {"n": data}),
                )
