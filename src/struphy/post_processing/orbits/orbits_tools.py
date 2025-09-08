import os
import shutil

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from struphy.post_processing.orbits.orbits_kernels import calculate_guiding_center_from_6d


def post_process_orbit_guiding_center(path_in, path_kinetics_species, species):
    """
    Computes the Cartesian guiding center from saved full-orbit marker orbits (Particles6D) and writes them to a .npy files and to .txt files.

    * ``.npy`` files :

      ===== ===== =============== ========== ====== ============
      index | 0 | | 1 | 2 | | 3 |     4        5         6
      ===== ===== =============== ========== ====== ============
      value  ID    guiding_center v_parallel v_perp magn. moment
      ===== ===== =============== ========== ====== ============

    * ``.txt`` files :

      ===== ===== =============
      index | 0 | | 1 | 2 | 3 |
      ===== ===== =============
      value  ID   guiding_center
      ===== ===== =============

    ``.txt`` file can be imported to e.g. Paraview, see `Tutorial 08 - Kinetic data <file:///home/spossann/git_repos/struphy/doc/_build/html/tutorials/tutorial_08_struphy_data_pproc.html#Kinetic-data>`_ for details..

    Parameters
    ----------
    path_in : str
        Absolute path of simulation output folder.

    path_kinetics_species : str
        Absolute path of where to store the .txt files. Will be saved in path_kinetics_species/guiding_center.

    species : str
        Name of the species for which the post processing should be performed.
    """

    with open(os.path.join(path_in, "parameters.yml"), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # create domain for calculating markers' physical coordinates
    domain, equil = setup_domain_and_equil(params)

    # path for orbit data
    path_orbits = os.path.join(path_kinetics_species, "orbits")

    # check .npy files generated from post_process_markers
    npy_files_list = [
        file
        for file in os.listdir(
            path_orbits,
        )
        if file.endswith(".npy")
    ]
    pproc_nt = len(npy_files_list)
    n_markers = np.load(os.path.join(path_orbits, npy_files_list[0])).shape[0]

    # re-ordering npy_files
    npy_files_list = sorted(npy_files_list)

    # make directorry
    path_gc = os.path.join(path_kinetics_species, "guiding_center")

    try:
        os.mkdir(path_gc)
    except:
        shutil.rmtree(path_gc)
        os.mkdir(path_gc)

    # temporary marker array
    temp = np.empty((n_markers, 7), dtype=float)
    etas = np.empty((n_markers, 3), dtype=float)
    B_cart = np.empty((n_markers, 3), dtype=float)
    lost_particles_mask = np.empty(n_markers, dtype=bool)

    print("Evaluation of guiding center for " + str(species))

    # loop over time grid
    for n in tqdm(range(pproc_nt)):
        # clear buffer
        B_cart[:, :] = 0
        etas[:, :] = 0

        # path for numpy array and text file for this time step
        file_npy = os.path.join(path_gc, npy_files_list[n])
        file_txt = os.path.join(path_gc, npy_files_list[n][:-4] + ".txt")

        # call .npy file
        temp[:, :] = np.load(os.path.join(path_orbits, npy_files_list[n]))

        # move ids to last column and save
        temp = np.roll(temp, -1, axis=1)

        # sorting out lost particles
        lost_particles_mask = np.all(temp[:, :-1] == 0, axis=1)

        # domain inverse map
        etas[~lost_particles_mask, :] = domain.inverse_map(
            *temp[~lost_particles_mask, :3].T,
            change_out_order=True,
        )

        # eval cartesian magnetic filed at marker positions
        B_cart[~lost_particles_mask, :] = equil.b_cart(
            *np.concatenate(
                (
                    etas[:, 0][:, None],
                    etas[:, 1][:, None],
                    etas[:, 2][:, None],
                ),
            ),
        )[0].T

        # calculate guiding center positions
        calculate_guiding_center_from_6d(temp, B_cart)

        # move ids to first column and save
        temp = np.roll(temp, 1, axis=1)

        np.save(file_npy, temp)
        np.savetxt(file_txt, temp[:, :4], fmt="%12.6f", delimiter=", ")


def post_process_orbit_classification(path_kinetics_species, species):
    """
    Classify guiding center orbits as "passing", "trapped" or "lost".

    Classification data (0 for "passing", 1 for "trapped" and -1 for "lost") is added at the last column(7)
    of .npy files in a directory "kinetic_data/<name_of_species>/guiding_center/".

    ``.npy`` files :

    ===== ===== =============== ========== ====== ============ ==============
    index | 0 | | 1 | 2 | | 3 |     4        5         6             7
    ===== ===== =============== ========== ====== ============ ==============
    value  ID    guiding_center v_parallel v_perp magn. moment classification
    ===== ===== =============== ========== ====== ============ ==============

    Parameters
    ----------
    path_kinetics_species : str
        Absolute path of where to store the .txt files. Will be saved in path_kinetics_species/guiding_center.

    species : str
        Name of the species for which the post processing should be performed.

    kind : str
        Name of the kinetic kind (Particles6D, Particles5D or Particles3D).
    """

    # check whether there is guiding center orbits data or not. If there is not, do the 'post_process_orbit_guiding_center'.
    path_gc = os.path.join(path_kinetics_species, "guiding_center")

    # check .npy files generated from post_process_markers
    npy_files_list = [
        file
        for file in os.listdir(
            path_gc,
        )
        if file.endswith(".npy")
    ]
    pproc_nt = len(npy_files_list)
    n_markers = np.load(os.path.join(path_gc, npy_files_list[0])).shape[0]

    # re-ordering npy_files
    npy_files_list = sorted(npy_files_list)

    # temporary marker array
    temp = np.empty((n_markers, 8), dtype=float)
    v_parallel = np.empty(n_markers, dtype=float)
    trapped_particle_mask = np.empty(n_markers, dtype=bool)
    lost_particle_mask = np.empty(n_markers, dtype=bool)

    print("Classifying guiding center orbits for " + str(species))

    # loop over time grid
    for n in tqdm(range(pproc_nt)):
        # clear buffer
        temp[:, :] = 0

        # load .npy files
        file_npy = os.path.join(path_gc, npy_files_list[n])
        temp[:, :-1] = np.load(file_npy)

        # initial time step
        if n == 0:
            v_init = temp[:, 4]
            np.save(file_npy, temp)
            continue

        # synchronizing with former time step
        temp[:, -1] = np.load(
            os.path.join(
                path_gc,
                npy_files_list[n - 1],
            ),
        )[:, -1]

        # call parallel velocity data from .npy file
        v_parallel = np.load(os.path.join(path_gc, npy_files_list[n]))[:, 4]

        # sorting out lost particles
        lost_particle_mask = np.all(temp[:, 1:-1] == 0, axis=1)

        # check reverse of parallel velocity
        trapped_particle_mask[:] = False
        v_parallel *= v_init
        trapped_particle_mask = v_parallel < 0

        # assign "1" at the last index of trapped particles
        temp[trapped_particle_mask, -1] = 1

        # assign "-1" at the last index of lost particles
        temp[lost_particle_mask, -1] = -1

        np.save(file_npy, temp)
