import os
import pickle
import shutil

import cunumpy as xp
import h5py
import yaml
from tqdm import tqdm
from feectools.ddm.mpi import mpi as MPI 
from typing import TYPE_CHECKING

from struphy.feec.psydac_derham import SplineFunction
from struphy.fields_background import equils
from struphy.fields_background.base import FluidEquilibrium
from struphy.geometry import domains
from struphy.geometry.base import Domain
from struphy.io.options import BaseUnits, EnvironmentOptions, Time
from struphy.io.setup import import_parameters_py
from struphy.kinetic_background import maxwellians
from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable
from struphy.topology.grids import TensorProductGrid

if TYPE_CHECKING:
    from struphy.simulation.codes import StruphySimulation


class ParamsIn:
    """Holds the input parameters of a Struphy simulation as attributes."""

    def __init__(
        self,
        env: EnvironmentOptions = None,
        base_units: BaseUnits = None,
        time_opts: Time = None,
        domain=None,
        equil=None,
        grid: TensorProductGrid = None,
        derham_opts=None,
        model: StruphyModel = None,
    ):
        self.env = env
        self.units = base_units
        self.time_opts = time_opts
        self.domain = domain
        self.equil = equil
        self.grid = grid
        self.derham_opts = derham_opts
        self.model = model


class SimData:
    """Holds post-processed Struphy data as attributes.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.
    """

    def __init__(self, path: str):
        self.path = path
        self._orbits = {}
        self._f = {}
        self._spline_values = {}
        self._n_sph = {}
        self.grids_log: list[xp.ndarray] = None
        self.grids_phy: list[xp.ndarray] = None
        self.t_grid: xp.ndarray = None

    @property
    def orbits(self) -> dict[str, xp.ndarray]:
        """Keys: species name. Values: 3d arrays indexed by (n, p, a), where 'n' is the time index, 'p' the particle index and 'a' the attribute index."""
        return self._orbits

    @property
    def f(self) -> dict[str, dict[str, dict[str, xp.ndarray]]]:
        """Keys: species name. Values: dicts of slice names ('e1_v1' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._f

    @property
    def spline_values(self) -> dict[str, dict[str, xp.ndarray]]:
        """Keys: species name. Values: dicts of variable names with values being 3d arrays on the grid."""
        return self._spline_values

    @property
    def n_sph(self) -> dict[str, dict[str, dict[str, xp.ndarray]]]:
        """Keys: species name. Values: dicts of view names ('view_0' etc.) holding dicts of corresponding xp.arrays for plotting."""
        return self._n_sph

    @property
    def Nt(self) -> dict[str, int]:
        """Number of available time points (snap shots) for each species."""
        if not hasattr(self, "_Nt"):
            self._Nt = {}
            for spec, orbs in self.orbits.items():
                self._Nt[spec] = orbs.shape[0]
        return self._Nt

    @property
    def Np(self) -> dict[str, int]:
        """Number of particle orbits for each species."""
        if not hasattr(self, "_Np"):
            self._Np = {}
            for spec, orbs in self.orbits.items():
                self._Np[spec] = orbs.shape[1]
        return self._Np

    @property
    def Nattr(self) -> dict[str, int]:
        """Number of particle attributes for each species."""
        if not hasattr(self, "_Nattr"):
            self._Nattr = {}
            for spec, orbs in self.orbits.items():
                self._Nattr[spec] = orbs.shape[2]
        return self._Nattr


def get_params_of_run(path: str) -> ParamsIn:
    """Retrieve parameters of finished Struphy run.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder.
    """

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

    print("done.")

    return ParamsIn(
        env=env,
        base_units=base_units,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        model=model,
    )


def pproc(sim: StruphySimulation = None,
        path_out: str = None,
        step: int = 1,
        celldivide: int = 1,
        physical: bool = False,
        guiding_center: bool = False,
        classify: bool = False,
        create_vtk: bool = True,
        time_trace: bool = False,
        verbose: bool = False,
    ):
    """Post-processing finished Struphy runs.

    Parameters
    ----------
    sim : StruphySimulation
        StruphySimulation object of finished run.
    
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

    time_trace : bool
        whether to plot the time trace of each measured region
    """
    if sim is None:
        assert path_out is not None, "If no sim object is provided, a path_out must be given to retrieve the parameters of the run to post-process."
    else:
        path_out = sim.env.path_out

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n*** Start post-processing of {path_out}:")

    # create post-processing folder
    path_pproc = os.path.join(path_out, "post_processing")

    try:
        os.mkdir(path_pproc)
    except:
        shutil.rmtree(path_pproc)
        os.mkdir(path_pproc)

    if time_trace:
        from struphy.post_processing.likwid.plot_time_traces import plot_gantt_chart_plotly, plot_time_vs_duration

        path_time_trace = os.path.join(path_out, "profiling_time_trace.pkl")
        plot_time_vs_duration(path_time_trace, output_path=path_pproc)
        plot_gantt_chart_plotly(path_time_trace, output_path=path_pproc)
        return

    # check for fields and kinetic data in hdf5 file that need post processing
    with h5py.File(os.path.join(path_out, "data/", "data_proc0.hdf5"), "r") as file:
        # save time grid at which post-processing data is created
        xp.save(os.path.join(path_pproc, "t_grid.npy"), file["time/value"][::step].copy())

        if "feec" in file.keys():
            exist_fields = True
        else:
            exist_fields = False

        if "kinetic" in file.keys():
            sim.exist_particles = {"markers": False, "f": False, "n_sph": False}
            sim.kinetic_species = []
            sim.kinetic_kinds = []
            for name in file["kinetic"].keys():
                sim.kinetic_species += [name]
                sim.kinetic_kinds += [next(iter(sim.model.species[name].variables.values())).space]

                # check for saved markers
                if "markers" in file["kinetic"][name]:
                    sim.exist_particles["markers"] = True
                # check for saved distribution function
                if "f" in file["kinetic"][name]:
                    sim.exist_particles["f"] = True
                # check for saved sph density
                if "n_sph" in file["kinetic"][name]:
                    sim.exist_particles["n_sph"] = True
        else:
            sim.exist_particles = None

    # post-processing
    if exist_fields:
        sim.pproc_fields(step=step, celldivide=celldivide, physical=physical,
                            create_vtk=create_vtk, verbose=verbose,)      
    if sim.exist_particles is not None:
        sim.pproc_particles(step=step, guiding_center=guiding_center, classify=classify, verbose=verbose,)
        

def load_plotting_data(sim: StruphySimulation = None, path_out: str = None, verbose: bool = False,) -> SimData:
    """Load data generated during post-processing."""
    if sim is None:
        assert path_out is not None, "If no sim object is provided, a path_out must be given to retrieve the parameters of the run to post-process."
    else:
        path_out = sim.env.path_out

    path_pproc = os.path.join(path_out, "post_processing")
    assert os.path.exists(path_pproc), f"Path {path_pproc} does not exist, run 'pproc' first?"
    print("\n*** Loading post-processed simulation data:")
    print(f"{path_out =}")

    simdata = SimData(path_out)

    # load time grid
    simdata.t_grid = xp.load(os.path.join(path_pproc, "t_grid.npy"))

    # data paths
    path_fields = os.path.join(path_pproc, "fields_data")
    path_kinetic = os.path.join(path_pproc, "kinetic_data")

    # load point data
    if os.path.exists(path_fields):
        # grids
        with open(os.path.join(path_fields, "grids_log.bin"), "rb") as f:
            simdata.grids_log = pickle.load(f)
        with open(os.path.join(path_fields, "grids_phy.bin"), "rb") as f:
            simdata.grids_phy = pickle.load(f)

        # species folders
        species = next(os.walk(path_fields))[1]
        for spec in species:
            simdata._spline_values[spec] = {}
            # simdata.arrays[spec] = {}
            path_spec = os.path.join(path_fields, spec)
            wlk = os.walk(path_spec)
            files = next(wlk)[2]
            print(f"\nFiles in {path_spec}: {files}")
            for file in files:
                if ".bin" in file:
                    var = file.split(".")[0]
                    with open(os.path.join(path_spec, file), "rb") as f:
                        # try:
                        simdata._spline_values[spec][var] = pickle.load(f)
                        # simdata.arrays[spec][var] = pickle.load(f)

    if os.path.exists(path_kinetic):
        # species folders
        species = next(os.walk(path_kinetic))[1]
        print(f"{species =}")
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
                                simdata._orbits[spec] = xp.zeros((Nt, *tmp.shape), dtype=float)
                            simdata._orbits[spec][step] = tmp
                            n += 1

                elif "distribution_function" in folder:
                    simdata._f[spec] = {}
                    slices = next(sub_wlk)[1]
                    # print(f"{slices = }")
                    for sli in slices:
                        simdata._f[spec][sli] = {}
                        # print(f"{sli = }")
                        files = next(sub_wlk)[2]
                        # print(f"{files = }")
                        for file in files:
                            name = file.split(".")[0]
                            tmp = xp.load(os.path.join(path_dat, sli, file))
                            # print(f"{name = }")
                            simdata._f[spec][sli][name] = tmp

                elif "n_sph" in folder:
                    simdata._n_sph[spec] = {}
                    slices = next(sub_wlk)[1]
                    # print(f"{slices = }")
                    for sli in slices:
                        simdata._n_sph[spec][sli] = {}
                        # print(f"{sli = }")
                        files = next(sub_wlk)[2]
                        # print(f"{files = }")
                        for file in files:
                            name = file.split(".")[0]
                            tmp = xp.load(os.path.join(path_dat, sli, file))
                            # print(f"{name = }")
                            simdata._n_sph[spec][sli][name] = tmp

                else:
                    print(f"{folder =}")
                    raise NotImplementedError

    print("\nThe following data has been loaded:")
    print("\ngrids:")
    print(f"{simdata.t_grid.shape =}")
    if simdata.grids_log is not None:
        print(f"{simdata.grids_log[0].shape =}")
        print(f"{simdata.grids_log[1].shape =}")
        print(f"{simdata.grids_log[2].shape =}")
    if simdata.grids_phy is not None:
        print(f"{simdata.grids_phy[0].shape =}")
        print(f"{simdata.grids_phy[1].shape =}")
        print(f"{simdata.grids_phy[2].shape =}")
    print("\nsimdata.spline_values:")
    for k, v in simdata.spline_values.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
    print("\nsimdata.orbits:")
    for k, v in simdata.orbits.items():
        print(f"  {k}")
    print("\nsimdata.f:")
    for k, v in simdata.f.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
            for kkk, vvv in vv.items():
                print(f"      {kkk}")
    print("\nsimdata.n_sph:")
    for k, v in simdata.n_sph.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk}")
            for kkk, vvv in vv.items():
                print(f"      {kkk}")

    return simdata