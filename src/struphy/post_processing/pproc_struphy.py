def main(path, step=1, celldivide=1):
    """
    Post-processing of finished Struphy runs.

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder to post-process.

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    celldivide : int, optional
        Grid refinement in evaluation of FEM fields. E.g. celldivide=2 evaluates two points per grid cell. 
    """

    import os
    import shutil
    import h5py
    import pickle
    import yaml

    import numpy as np

    import struphy.post_processing.post_processing_tools as pproc

    print('')

    # create post-processing folder
    path_pproc = os.path.join(path, 'post_processing')

    try:
        os.mkdir(path_pproc)
    except:
        shutil.rmtree(path_pproc)
        os.mkdir(path_pproc)

    # check for fields and kinetic data in hdf5 file that need post processing
    file = h5py.File(os.path.join(path, 'data/', 'data_proc0.hdf5'), 'r')

    # save time grid at which post-processing data is created
    np.save(os.path.join(path_pproc, 't_grid.npy'),
            file['time/value'][::step].copy())

    if 'feec' in file.keys():
        exist_fields = True
    else:
        exist_fields = False

    kinetic_species = []
    if 'kinetic' in file.keys():
        exist_kinetic = {'markers': False, 'f': False}

        for name in file['kinetic'].keys():
            kinetic_species += [name]

            # check for saved markers
            if 'markers' in file['kinetic'][name]:
                exist_kinetic['markers'] = True

            # check for saved distribution function
            if 'f' in file['kinetic'][name]:
                exist_kinetic['f'] = True

    else:
        exist_kinetic = None

    file.close()

    # field post-processing
    if exist_fields:

        fields, space_ids, _ = pproc.create_femfields(path, step)

        point_data_log, point_data_phy, grids_log, grids_phy = pproc.eval_femfields(
            path, fields, space_ids, [celldivide, celldivide, celldivide])

        # directory for field data
        path_fields = os.path.join(path_pproc, 'fields_data')

        try:
            os.mkdir(path_fields)
        except:
            shutil.rmtree(path_fields)
            os.mkdir(path_fields)

        # save data dicts for each field
        for name, val in point_data_log.items():
            
            aux = name.split('_')
            # is em field
            if len(aux) == 1:
                subfolder = 'em_fields'
                new_name = name
                try:
                    os.mkdir(os.path.join(path_fields, subfolder))
                except:
                    pass
                
            # is fluid species
            elif len(aux) == 2:
                subfolder = aux[0]
                new_name = aux[1]
                try:
                    os.mkdir(os.path.join(path_fields, subfolder))
                except:
                    pass
            else:
                raise ValueError(f'Naming {name} of feec unknown is not permitted (can only have one underscore).')

            with open(os.path.join(path_fields, subfolder, new_name + '_log.bin'), 'wb') as handle:
                pickle.dump(val, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(path_fields, subfolder, new_name + '_phy.bin'), 'wb') as handle:
                pickle.dump(point_data_phy[name], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # save grids
        with open(os.path.join(path_fields, 'grids_log.bin'), 'wb') as handle:
            pickle.dump(grids_log, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path_fields, 'grids_phy.bin'), 'wb') as handle:
            pickle.dump(grids_phy, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # create vtk files
        pproc.create_vtk(path_fields, grids_phy, point_data_phy)

    # kinetic post-processing
    if exist_kinetic is not None:

        # directory for kinetic data
        path_kinetics = os.path.join(path_pproc, 'kinetic_data')

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
            if exist_kinetic['markers']:
                pproc.post_process_markers(path, path_kinetics_species, species, step)

            # distribution function
            if exist_kinetic['f']:

                with open(os.path.join(path, 'parameters.yml'), 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)

                try:
                    marker_type = params['kinetic'][species]['markers']['type']
                except:
                    marker_type = 'full_f'

                pproc.post_process_f(path, path_kinetics_species,
                                    species, step, marker_type)


if __name__ == '__main__':

    import argparse
    import struphy

    libpath = struphy.__path__[0]

    parser = argparse.ArgumentParser(
        description='Post-process data of finished Struphy runs to prepare for diagnostics.')

    # paths of simulation folders
    parser.add_argument('dir',
                        type=str,
                        metavar='DIR',
                        help='absolute path of simulation ouput folder to post-process')

    parser.add_argument('-s', '--step',
                        type=int,
                        metavar='N',
                        help='do post-processing every N-th time step (default=1)',
                        default=1)

    parser.add_argument('--celldivide',
                        type=int,
                        metavar='N',
                        help='divide each grid cell by N for field evaluation (default=1)',
                        default=1)

    args = parser.parse_args()

    main(args.dir,
         args.step,
         args.celldivide)
