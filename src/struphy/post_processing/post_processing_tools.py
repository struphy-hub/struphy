import os
import shutil

import h5py
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from struphy.feec.psydac_derham import Derham
from struphy.io.setup import setup_domain_and_equil
from struphy.kinetic_background import maxwellians
from struphy.models import fluid, hybrid, kinetic, toy
from struphy.utils.arrays import xp as np


def create_femfields(
    path: str,
    *,
    step: int = 1,
):
    """Creates instances of :class:`~struphy.feec.psydac_derham.SplineFunction` from distributed Struphy data.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder.

    step : int
        Whether to create FEM fields at every time step (step=1, default), every second time step (step=2), etc.

    Returns
    -------
    fields : dict
        Nested dictionary holding :class:`~struphy.feec.psydac_derham.SplineFunction`: fields[t][name] contains the Field with the name "name" in the hdf5 file at time t.

    space_ids : dict
        The space IDs of the fields (H1, Hcurl, Hdiv, L2 or H1vec). space_ids[name] contains the space ID of the field with the name "name".

    model : str
        From which model in struphy/models the data has been obtained.
    """

    # get model name and # of MPI processes from meta.txt file
    with open(os.path.join(path, "meta.txt"), "r") as f:
        lines = f.readlines()

    model = lines[3].split()[-1]
    nproc = lines[4].split()[-1]

    # create Derham sequence from grid parameters
    with open(os.path.join(path, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    derham = Derham(
        params["grid"]["Nel"],
        params["grid"]["p"],
        params["grid"]["spl_kind"],
    )

    # get fields names, space IDs and time grid from 0-th rank hdf5 file
    file = h5py.File(os.path.join(path, "data/", "data_proc0.hdf5"), "r")

    space_ids = {}

    for field_name, dset in file["feec"].items():
        space_ids[field_name] = dset.attrs["space_id"]

    t_grid = file["time/value"][::step].copy()

    file.close()

    # create one FemField for each snapshot
    fields = {}
    for t in t_grid:
        fields[t] = {}
        for field_name, ID in space_ids.items():
            fields[t][field_name] = derham.create_spline_function(field_name, ID)

    # get hdf5 data
    for rank in range(int(nproc)):
        # open hdf5 file
        file = h5py.File(
            os.path.join(
                path,
                "data/",
                "data_proc" + str(rank) + ".hdf5",
            ),
            "r",
        )

        for field_name, dset in tqdm(file["feec"].items()):
            # get global start indices, end indices and pads
            gl_s = dset.attrs["starts"]
            gl_e = dset.attrs["ends"]
            pads = dset.attrs["pads"]

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

                    data = dset[n * step, p1:-p1, p2:-p2, p3:-p3].copy()

                    fields[t][field_name].vector[
                        s1 : e1 + 1,
                        s2 : e2 + 1,
                        s3 : e3 + 1,
                    ] = data
                    # update after each data addition, can be made more efficient
                    fields[t][field_name].vector.update_ghost_regions()

                # vector-valued field
                else:
                    for comp in range(3):
                        s1, s2, s3 = gl_s[comp]
                        e1, e2, e3 = gl_e[comp]
                        p1, p2, p3 = pads[comp]

                        data = dset[str(comp + 1)][
                            n * step,
                            p1:-p1,
                            p2:-p2,
                            p3:-p3,
                        ].copy()

                        fields[t][field_name].vector[comp][
                            s1 : e1 + 1,
                            s2 : e2 + 1,
                            s3 : e3 + 1,
                        ] = data
                    # update after each data addition, can be made more efficient
                    fields[t][field_name].vector.update_ghost_regions()

        file.close()

    print("Creation of Struphy Fields done.")

    return fields, space_ids, model


def eval_femfields(
    path: str,
    fields: dict,
    space_ids: dict,
    *,
    celldivide: list = [1, 1, 1],
    physical: bool = False,
):
    """Evaluate FEM fields obtained from :meth:`struphy.post_processing.post_processing_tools.create_femfields`.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder.

    fields : dict
        Obtained from struphy.diagnostics.post_processing.create_femfields.

    space_ids : dict
        Obtained from struphy.diagnostics.post_processing.create_femfields.

    celldivide : list of ints
        Grid refinement in each eta direction.

    physical : bool
        Wether to do post-processing into push-forwarded physical (xyz) components of fields.

    Returns
    -------
    point_data : dict
        Nested dictionary holding values of FemFields on the grid as list of 3d np.arrays:
        point_data[name][t] contains the values of the field with name "name" in fields[t].keys() at time t.

        If physical is True, physical components of fields are saved.
        Otherwise, logical components (differential n-forms) are saved.

    grids_log : 3-list
        1d logical grids in each eta-direction with Nel[i]*cell_divide[i] + 1 entries in each direction.

    grids_phy : 3-list
        Mapped (physical) grids obtained by domain(*grids_log).
    """

    assert isinstance(fields, dict)
    assert isinstance(space_ids, dict)

    # domain object according to parameter file and grids
    with open(os.path.join(path, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    domain = setup_domain_and_equil(params)[0]

    # create logical and physical grids
    assert isinstance(celldivide, list)
    assert len(celldivide) == 3

    Nel = params["grid"]["Nel"]

    grids_log = [np.linspace(0.0, 1.0, Nel_i * n_i + 1) for Nel_i, n_i in zip(Nel, celldivide)]
    grids_phy = [
        domain(*grids_log)[0],
        domain(*grids_log)[1],
        domain(*grids_log)[2],
    ]

    # evaluate fields at evaluation grid and push-forward
    point_data = {}

    # one dict for each field
    for name in space_ids:
        point_data[name] = {}

    # time loop
    print("Evaluating fields ...")
    for t in tqdm(fields):
        # field loop
        for name, field in fields[t].items():
            # space ID
            space_id = space_ids[name]

            # field evaluation
            temp_val = field(*grids_log)

            point_data[name][t] = []

            # scalar spaces
            if isinstance(temp_val, np.ndarray):
                if physical:
                    # push-forward
                    if space_id == "H1":
                        point_data[name][t].append(
                            domain.push(
                                temp_val,
                                *grids_log,
                                kind="0",
                            ),
                        )
                    elif space_id == "L2":
                        point_data[name][t].append(
                            domain.push(
                                temp_val,
                                *grids_log,
                                kind="3",
                            ),
                        )

                else:
                    point_data[name][t].append(temp_val)

            # vector-valued spaces
            else:
                for j in range(3):
                    if physical:
                        # push-forward
                        if space_id == "Hcurl":
                            point_data[name][t].append(
                                domain.push(
                                    temp_val,
                                    *grids_log,
                                    kind="1",
                                )[j],
                            )
                        elif space_id == "Hdiv":
                            point_data[name][t].append(
                                domain.push(
                                    temp_val,
                                    *grids_log,
                                    kind="2",
                                )[j],
                            )
                        elif space_id == "H1vec":
                            point_data[name][t].append(
                                domain.push(
                                    temp_val,
                                    *grids_log,
                                    kind="v",
                                )[j],
                            )

                    else:
                        point_data[name][t].append(temp_val[j])

    return point_data, grids_log, grids_phy


def create_vtk(
    path: str,
    grids_phy: list,
    point_data: dict,
    *,
    physical: bool = False,
):
    """Creates structured virtual toolkit files (.vts) for Paraview from evaluated field data.

    Parameters
    ----------
    path : str
        Absolute path of where to store the .vts files. Will then be in path/vtk/step_<step>.vts.

    grids_phy : 3-list
        Mapped (physical) grids obtained from struphy.diagnostics.post_processing.eval_femfields.

    point_data : dict
        Field data obtained from struphy.diagnostics.post_processing.eval_femfields.

    physical : bool
        Wether to create vtk for push-forwarded physical (xyz) components of fields.
    """

    from pyevtk.hl import gridToVTK

    # directory for vtk files
    path_vtk = os.path.join(path, "vtk" + physical * "_phy")

    try:
        os.mkdir(path_vtk)
    except:
        shutil.rmtree(path_vtk)
        os.mkdir(path_vtk)

    # field names
    names = list(point_data.keys())

    # time loop
    tgrid = list(point_data[names[0]].keys())

    nt = len(tgrid) - 1
    log_nt = int(np.log10(nt)) + 1

    print("Creating vtk ...")
    for n, t in enumerate(tqdm(tgrid)):
        point_data_n = {}

        for name in names:
            points_list = point_data[name][t]

            # scalar
            if len(points_list) == 1:
                point_data_n[name] = points_list[0]

            # vector
            else:
                for j in range(3):
                    point_data_n[name + f"_{j + 1}"] = points_list[j]

        gridToVTK(
            os.path.join(path_vtk, "step_{0:0{1}d}".format(n, log_nt)),
            *grids_phy,
            pointData=point_data_n,
        )


def post_process_markers(path_in, path_out, species, kind, step=1):
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
    path_in : str
        Absolute path of simulation output folder.

    path_out : str
        Absolute path of where to store the .txt files. Will be in path_out/orbits.

    species : str
        Name of the species for which the post processing should be performed.

    kind : str
        Name of the kinetic kind (Particles6D, Particles5D or Particles3D).

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.
    """

    # get # of MPI processes from meta.txt file
    with open(os.path.join(path_in, "meta.txt"), "r") as f:
        lines = f.readlines()

    nproc = lines[4].split()[-1]

    with open(os.path.join(path_in, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # create domain for calculating markers' physical coordinates
    domain = setup_domain_and_equil(params)[0]

    # open hdf5 files and get names and number of saved markers of kinetic species
    files = [
        h5py.File(
            os.path.join(
                path_in,
                "data/",
                f"data_proc{i}.hdf5",
            ),
            "r",
        )
        for i in range(int(nproc))
    ]

    # get number of time steps and markers
    nt, n_markers, n_cols = files[0]["kinetic/" + species + "/markers"].shape

    log_nt = int(np.log10(int(((nt - 1) / step)))) + 1

    # directory for .txt files and marker index which will be saved
    if "5D" in kind:
        path_orbits = os.path.join(path_out, "guiding_center")
        save_index = list(range(0, 6)) + [10] + [-1]
    elif "6D" in kind or "SPH" in kind:
        path_orbits = os.path.join(path_out, "orbits")
        save_index = list(range(0, 7)) + [-1]
    else:
        path_orbits = os.path.join(path_out, "orbits")
        save_index = list(range(0, 4)) + [-1]

    try:
        os.mkdir(path_orbits)
    except:
        shutil.rmtree(path_orbits)
        os.mkdir(path_orbits)

    # temporary array
    temp = np.empty((n_markers, len(save_index)), order="C")
    lost_particles_mask = np.empty(n_markers, dtype=bool)

    print(f"Evaluation of {n_markers} marker orbits for {species}")

    # loop over time grid
    for n in tqdm(range(int((nt - 1) / step) + 1)):
        # clear buffer
        temp[:, :] = 0

        # create text file for this time step and this species
        file_npy = os.path.join(
            path_orbits,
            species + "_{0:0{1}d}.npy".format(n, log_nt),
        )
        file_txt = os.path.join(
            path_orbits,
            species + "_{0:0{1}d}.txt".format(n, log_nt),
        )

        for file in files:
            markers = file["kinetic/" + species + "/markers"]
            ids = markers[n * step, :, -1].astype("int")
            ids = ids[ids != -1]  # exclude holes
            temp[ids] = markers[n * step, : ids.size, save_index]

        # sorting out lost particles
        ids = temp[:, -1].astype("int")
        ids_lost_particles = np.setdiff1d(np.arange(n_markers), ids)
        lost_particles_mask[:] = False
        lost_particles_mask[ids_lost_particles] = True

        if len(ids_lost_particles) > 0:
            # lost markers are saved as [0, ..., 0, ids]
            temp[lost_particles_mask, -1] = ids_lost_particles
            ids = np.unique(np.append(ids, ids_lost_particles))

        assert np.all(sorted(ids) == np.arange(n_markers))

        # compute physical positions (x, y, z)
        temp[~lost_particles_mask, :3] = domain(
            np.array(temp[~lost_particles_mask, :3]),
            change_out_order=True,
        )

        # move ids to first column and save
        temp = np.roll(temp, 1, axis=1)

        np.save(file_npy, temp)
        np.savetxt(file_txt, temp[:, (0, 1, 2, 3, -1)], fmt="%12.6f", delimiter=", ")

    # close hdf5 files
    for file in files:
        file.close()


def post_process_f(path_in, path_out, species, step=1, compute_bckgr=False):
    """Computes and saves distribution functions of saved binning data during a simulation.

    Parameters
    ----------
    path_in : str
        Absolute path of simulation output folder.

    path_out : str
        Absolute path of where to store the .txt files. Will be in path_out/orbits.

    species : str
        Name of the species for which the post processing should be performed.

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    compute_bckgr : bool
        Whehter to compute the kinetic background values and add them to the binning data.
        This is used if non-standard weights are binned.
    """

    # get model name and # of MPI processes from meta.txt file
    with open(os.path.join(path_in, "meta.txt"), "r") as f:
        lines = f.readlines()

    nproc = lines[4].split()[-1]

    # load parameters
    with open(os.path.join(path_in, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # open hdf5 files
    files = [
        h5py.File(
            os.path.join(
                path_in,
                "data/",
                f"data_proc{i}.hdf5",
            ),
            "r",
        )
        for i in range(int(nproc))
    ]

    # directory for .npy files
    path_distr = os.path.join(path_out, "distribution_function")

    try:
        os.mkdir(path_distr)
    except:
        shutil.rmtree(path_distr)
        os.mkdir(path_distr)

    print("Evaluation of distribution functions for " + str(species))

    # Create grids
    for slice_name in tqdm(files[0]["kinetic/" + species + "/f"]):
        # create a new folder for each slice
        path_slice = os.path.join(path_distr, slice_name)
        os.mkdir(path_slice)

        # Find out all names of slices
        slice_names = slice_name.split("_")

        # save grid
        for n_gr, (_, grid) in enumerate(files[0]["kinetic/" + species + "/f/" + slice_name].attrs.items()):
            grid_path = os.path.join(
                path_slice,
                "grid_" + slice_names[n_gr] + ".npy",
            )
            np.save(grid_path, grid[:])

    # compute distribution function
    for slice_name in tqdm(files[0]["kinetic/" + species + "/f"]):
        # path to folder of slice
        path_slice = os.path.join(path_distr, slice_name)

        # Find out all names of slices
        slice_names = slice_name.split("_")

        # load full-f data
        data = files[0]["kinetic/" + species + "/f/" + slice_name][::step].copy()
        for rank in range(1, int(nproc)):
            data += files[rank]["kinetic/" + species + "/f/" + slice_name][::step]

        # load delta-f data
        data_df = files[0]["kinetic/" + species + "/df/" + slice_name][::step].copy()
        for rank in range(1, int(nproc)):
            data_df += files[rank]["kinetic/" + species + "/df/" + slice_name][::step]

        # save distribution functions
        np.save(os.path.join(path_slice, "f_binned.npy"), data)
        np.save(os.path.join(path_slice, "delta_f_binned.npy"), data_df)

        if compute_bckgr:
            bckgr_params = params["kinetic"][species]["background"]

            f_bckgr = None
            for fi, maxw_params in bckgr_params.items():
                if fi[-2] == "_":
                    fi_type = fi[:-2]
                else:
                    fi_type = fi

                if f_bckgr is None:
                    f_bckgr = getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                    )
                else:
                    f_bckgr = f_bckgr + getattr(maxwellians, fi_type)(
                        maxw_params=maxw_params,
                    )

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
                    grid_tot += [np.load(filename)]

                # otherwise evaluate at zero
                else:
                    grid_tot += [np.zeros(1)]

            # v-grid
            for comp in range(1, f_bckgr.vdim + 1):
                current_slice = "v" + str(comp)
                filename = os.path.join(
                    path_slice,
                    "grid_" + current_slice + ".npy",
                )

                # check if file exists and is in slice_name
                if os.path.exists(filename) and current_slice in slice_names:
                    grid_tot += [np.load(filename)]

                # otherwise evaluate at zero
                else:
                    grid_tot += [np.zeros(1)]
                    # correct integrating out in v-direction, TODO: check for 5D Maxwellians
                    factor *= np.sqrt(2 * np.pi)

            grid_eval = np.meshgrid(*grid_tot, indexing="ij")

            data_bckgr = f_bckgr(*grid_eval).squeeze()

            # correct integrating out in v-direction
            data_bckgr *= factor

            # Now all data is just the data for delta_f
            data_delta_f = data_df

            # save distribution function
            np.save(os.path.join(path_slice, "delta_f_binned.npy"), data_delta_f)
            # add extra axis for data_bckgr since data_delta_f has axis for time series
            np.save(
                os.path.join(path_slice, "f_binned.npy"),
                data_delta_f + data_bckgr[tuple([None])],
            )

    # close hdf5 files
    for file in files:
        file.close()


def post_process_n_sph(path_in, path_out, species, step=1, compute_bckgr=False):
    """Computes and saves the density n of saved sph data during a simulation.

    Parameters
    ----------
    path_in : str
        Absolute path of simulation output folder.

    path_out : str
        Absolute path of where to store the .txt files. Will be in path_out/orbits.

    species : str
        Name of the species for which the post processing should be performed.

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    compute_bckgr : bool
        Whehter to compute the kinetic background values and add them to the binning data.
        This is used if non-standard weights are binned.
    """

    # get model name and # of MPI processes from meta.txt file
    with open(os.path.join(path_in, "meta.txt"), "r") as f:
        lines = f.readlines()

    nproc = lines[4].split()[-1]

    # load parameters
    with open(os.path.join(path_in, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # open hdf5 files
    files = [
        h5py.File(
            os.path.join(
                path_in,
                "data/",
                f"data_proc{i}.hdf5",
            ),
            "r",
        )
        for i in range(int(nproc))
    ]

    # directory for .npy files
    path_n_sph = os.path.join(path_out, "n_sph")

    try:
        os.mkdir(path_n_sph)
    except:
        shutil.rmtree(path_n_sph)
        os.mkdir(path_n_sph)

    print("Evaluation of sph density for " + str(species))

    # Create grids
    for i, view in enumerate(files[0]["kinetic/" + species + "/n_sph"]):
        # create a new folder for each view
        path_view = os.path.join(path_n_sph, view)
        os.mkdir(path_view)

        # build meshgrid and save
        eta1 = files[0]["kinetic/" + species + "/n_sph/" + view].attrs["eta1"]
        eta2 = files[0]["kinetic/" + species + "/n_sph/" + view].attrs["eta2"]
        eta3 = files[0]["kinetic/" + species + "/n_sph/" + view].attrs["eta3"]

        ee1, ee2, ee3 = np.meshgrid(
            eta1,
            eta2,
            eta3,
            indexing="ij",
        )

        grid_path = os.path.join(
            path_view,
            "grid_n_sph.npy",
        )
        np.save(grid_path, (ee1, ee2, ee3))

        # load n_sph data
        data = files[0]["kinetic/" + species + "/n_sph/" + view][::step].copy()
        for rank in range(1, int(nproc)):
            data += files[rank]["kinetic/" + species + "/n_sph/" + view][::step]

        # save distribution functions
        np.save(os.path.join(path_view, "n_sph.npy"), data)
