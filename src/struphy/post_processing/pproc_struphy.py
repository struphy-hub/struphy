import os
import pickle
import shutil

import h5py
import numpy as np
import yaml

import struphy.post_processing.orbits.orbits_tools as orbits_pproc
import struphy.post_processing.post_processing_tools as pproc
from struphy.io.setup import import_parameters_py


def main(
    path: str,
    *,
    step: int = 1,
    celldivide: int = 1,
    physical: bool = False,
    guiding_center: bool = False,
    classify: bool = False,
    no_vtk: bool = False,
    time_trace: bool = False,
):
    """Post-processing of finished Struphy runs.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.

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

    no_vtk : bool
        whether vtk files creation should be skipped

    time_trace : bool
        whether to plot the time trace of each measured region
    """
    print("")

    # create post-processing folder
    path_pproc = os.path.join(path, "post_processing")

    try:
        os.mkdir(path_pproc)
    except:
        shutil.rmtree(path_pproc)
        os.mkdir(path_pproc)

    if time_trace:
        from struphy.post_processing.likwid.plot_time_traces import plot_gantt_chart, plot_time_vs_duration

        path_time_trace = os.path.join(path, "profiling_time_trace.pkl")
        plot_time_vs_duration(path_time_trace, output_path=path_pproc)
        plot_gantt_chart(path_time_trace, output_path=path_pproc)
        return

    # check for fields and kinetic data in hdf5 file that need post processing
    file = h5py.File(os.path.join(path, "data/", "data_proc0.hdf5"), "r")

    # save time grid at which post-processing data is created
    np.save(os.path.join(path_pproc, "t_grid.npy"), file["time/value"][::step].copy())

    if "feec" in file.keys():
        exist_fields = True
    else:
        exist_fields = False

    if "kinetic" in file.keys():
        exist_kinetic = {"markers": False, "f": False, "n_sph": False}
        for name in file["kinetic"].keys():
            # check for saved markers
            if "markers" in file["kinetic"][name]:
                exist_kinetic["markers"] = True
            # check for saved distribution function
            if "f" in file["kinetic"][name]:
                exist_kinetic["f"] = True
            # check for saved sph density
            if "n_sph" in file["kinetic"][name]:
                exist_kinetic["n_sph"] = True
    else:
        exist_kinetic = None

    file.close()

    # import parameters
    params_in = import_parameters_py(os.path.join(path, "parameters.py"))

    # field post-processing
    if exist_fields:
        fields, t_grid = pproc.create_femfields(path, params_in, step=step)

        point_data, grids_log, grids_phy = pproc.eval_femfields(
            params_in, fields, celldivide=[celldivide, celldivide, celldivide]
        )

        if physical:
            point_data_phy, grids_log, grids_phy = pproc.eval_femfields(
                params_in, fields, celldivide=[celldivide, celldivide, celldivide], physical=True
            )

        # directory for field data
        path_fields = os.path.join(path_pproc, "fields_data")

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
        if not no_vtk:
            pproc.create_vtk(path_fields, t_grid, grids_phy, point_data)
            if physical:
                pproc.create_vtk(path_fields, t_grid, grids_phy, point_data_phy, physical=True)

    # kinetic post-processing
    if exist_kinetic is not None:
        # directory for kinetic data
        path_kinetics = os.path.join(path_pproc, "kinetic_data")

        try:
            os.mkdir(path_kinetics)
        except:
            shutil.rmtree(path_kinetics)
            os.mkdir(path_kinetics)

        # kinetic post-processing for each species
        for n, species in enumerate(kinetic_species):
            # directory for each species
            path_kinetics_species = os.path.join(path_kinetics, species)

            try:
                os.mkdir(path_kinetics_species)
            except:
                shutil.rmtree(path_kinetics_species)
                os.mkdir(path_kinetics_species)

            # markers
            if exist_kinetic["markers"]:
                pproc.post_process_markers(path, path_kinetics_species, species, kinetic_kinds[n], step)

                if guiding_center:
                    assert kinetic_kinds[n] == "Particles6D"
                    orbits_pproc.post_process_orbit_guiding_center(path, path_kinetics_species, species)

                if classify:
                    orbits_pproc.post_process_orbit_classification(path_kinetics_species, species)

            # distribution function
            if exist_kinetic["f"]:
                if kinetic_kinds[n] == "DeltaFParticles6D":
                    compute_bckgr = True
                else:
                    compute_bckgr = False

                pproc.post_process_f(path, path_kinetics_species, species, step, compute_bckgr=compute_bckgr)

            # sph density
            if exist_kinetic["n_sph"]:
                pproc.post_process_n_sph(path, path_kinetics_species, species, step, compute_bckgr=compute_bckgr)


if __name__ == "__main__":
    import argparse

    import struphy

    libpath = struphy.__path__[0]

    parser = argparse.ArgumentParser(
        description="Post-process data of finished Struphy runs to prepare for diagnostics."
    )

    # paths of simulation folders
    parser.add_argument("dir", type=str, metavar="DIR", help="absolute path of simulation ouput folder to post-process")

    parser.add_argument(
        "-s", "--step", type=int, metavar="N", help="do post-processing every N-th time step (default=1)", default=1
    )

    parser.add_argument(
        "--celldivide",
        type=int,
        metavar="N",
        help="divide each grid cell by N for field evaluation (default=1)",
        default=1,
    )

    parser.add_argument(
        "--physical",
        help="do post-processing into push-forwarded physical (xyz) components",
        action="store_true",
    )

    parser.add_argument(
        "--guiding-center", help="compute guiding-center coordinates (only from Particles6D)", action="store_true"
    )

    parser.add_argument(
        "--classify", help="classify guiding-center trajectories (passing, trapped or lost)", action="store_true"
    )

    parser.add_argument("--no-vtk", help="whether vtk files creation should be skipped", action="store_true")

    parser.add_argument("--time-trace", help="whether to plot the time trace", action="store_true")

    args = parser.parse_args()

    main(
        args.dir,
        step=args.step,
        celldivide=args.celldivide,
        physical=args.physical,
        guiding_center=args.guiding_center,
        classify=args.classify,
        no_vtk=args.no_vtk,
        time_trace=args.time_trace,
    )
