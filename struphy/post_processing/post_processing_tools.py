import numpy as np

import os, shutil, h5py, yaml

from struphy.geometry import domains
from struphy.fields_background.mhd_equil import equils
from struphy.psydac_api.psydac_derham import Derham
from struphy.psydac_api.fields import Field
from struphy.kinetic_background import analytical
from struphy.models.utilities import setup_domain_mhd

from tqdm import tqdm


def create_femfields(path, step=1):
    """
    Creates instances of struphy.psydac_api.fields.Field from distributed Struphy data.

    Parameters
    ----------
    path : str
        Absolute path to folder with hdf5 data files.

    step : int, optional
        Whether to create FEM fields at every time step (step=1, default), every second time step (step=2), etc. 

    Returns
    -------
    fields : dict
        Nested dictionary holding struphy.psydac_api.field.Field: fields[t][name] contains the Field with the name "name" in the hdf5 file at time t.

    space_ids : dict
        The space IDs of the fields (H1, Hcurl, Hdiv, L2 or H1vec). space_ids[name] contains the space ID of the field with the name "name".

    model : str
        From which model in struphy.models.models the data has been obtained.
    """

    # get model name and # of MPI processes from meta.txt file
    with open(os.path.join(path, 'meta.txt'), 'r') as f:
        lines = f.readlines()

    model = lines[-6].split()[-1]
    nproc = lines[-1].split()[-1]

    # create Derham sequence from grid parameters
    with open(os.path.join(path, 'parameters.yml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    derham = Derham(params['grid']['Nel'],
                    params['grid']['p'],
                    params['grid']['spl_kind'])

    # get fields names, space IDs and time grid from 0-th rank hdf5 file
    file = h5py.File(os.path.join(path, 'data_proc0.hdf5'), 'r')
    
    space_ids = {}
    
    for field_name, dset in file['feec'].items():
        space_ids[field_name] = dset.attrs['space_id']

    t_grid = file['time/value'][::step].copy()
    
    file.close()

    # create one FemField for each snapshot
    fields = {}
    for t in t_grid:
        fields[t] = {}
        for field_name, ID in space_ids.items():
            fields[t][field_name] = Field(field_name, ID, derham)

    # get hdf5 data
    for rank in range(int(nproc)):

        # open hdf5 file
        file = h5py.File(os.path.join(path, 'data_proc' + str(rank) + '.hdf5'), 'r')

        for field_name, dset in tqdm(file['feec'].items()):

            # get global start indices, end indices and pads
            gl_s = dset.attrs['starts']
            gl_e = dset.attrs['ends']
            pads = dset.attrs['pads']

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

                    data = dset[n*step, p1:-p1, p2:-p2, p3:-p3].copy()

                    fields[t][field_name].vector[s1:e1 +
                                                 1, s2:e2 + 1, s3:e3 + 1] = data
                    # update after each data addition, can be made more efficient
                    fields[t][field_name].vector.update_ghost_regions()

                # vector-valued field
                else:
                    for comp in range(3):

                        s1, s2, s3 = gl_s[comp]
                        e1, e2, e3 = gl_e[comp]
                        p1, p2, p3 = pads[comp]

                        data = dset[str(comp + 1)][n*step, p1:-p1, p2:-p2, p3:-p3].copy()

                        fields[t][field_name].vector[comp][s1:e1 +
                                                           1, s2:e2 + 1, s3:e3 + 1] = data
                    # update after each data addition, can be made more efficient
                    fields[t][field_name].vector.update_ghost_regions()

        file.close()

    print('Creation of PSYDAC FemFields done.')

    return fields, space_ids, model


def eval_femfields(path, fields, space_ids, celldivide=[1, 1, 1]):
    """
    Evaluate FEM fields obtained from create_femfields. 

    Parameters
    ----------
    path : str
        Absolute path of simulation output folder.

    fields : dict
        Obtained from struphy.diagnostics.post_processing.create_femfields.

    space_ids : dict
        Obtained from struphy.diagnostics.post_processing.create_femfields.

    celldivide : list of ints, optional
        Grid refinement in each eta direction.

    Returns
    -------
    point_data_log : dict
        Nested dictionary holding values of FemFields on the grid as list of 3d np.arrays:
        point_data_log[name][t] contains the values of the field with name "name" in fields[t].keys() at time t.

    point_data_phy : dict
        Pushed-forward point_data_log obtained by domain.push().

    grids_log : 3-list
        1d logical grids in each eta-direction with Nel[i]*cell_divide[i] + 1 entries in each direction.  

    grids_phy : 3-list
        Mapped (physical) grids obtained by domain(*grids_log).
    """

    assert isinstance(fields, dict)
    assert isinstance(space_ids, dict)

    # domain object according to parameter file and grids
    with open(os.path.join(path, 'parameters.yml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    domain = setup_domain_mhd(params)[0]

    # create logical and physical grids
    assert isinstance(celldivide, list)
    assert len(celldivide) == 3
        
    Nel = params['grid']['Nel']
    
    grids_log = [np.linspace(0., 1., Nel_i*n_i + 1) for Nel_i, n_i in zip(Nel, celldivide)]
    grids_phy = [domain(*grids_log)[0],
                 domain(*grids_log)[1],
                 domain(*grids_log)[2]]

    # evaluate fields at evaluation grid and push-forward
    point_data_log = {}
    point_data_phy = {}
    
    # one dict for each field
    for name in space_ids:
        point_data_log[name] = {}
        point_data_phy[name] = {}
    
    # time loop
    print('Evaluating fields ...')
    for t in tqdm(fields):
        
        # field loop
        for name, field in fields[t].items():
            
            # space ID
            space_id = space_ids[name]

            # field evaluation
            temp_val = field(*grids_log)
            
            point_data_log[name][t] = []
            point_data_phy[name][t] = []

            # scalar spaces
            if isinstance(temp_val, np.ndarray):

                point_data_log[name][t].append(temp_val)

                # push-forward
                if space_id == 'H1':
                    point_data_phy[name][t].append(domain.push(
                        temp_val, *grids_log, kind='0_form'))
                elif space_id == 'L2':
                    point_data_phy[name][t].append(domain.push(
                        temp_val, *grids_log, kind='3_form'))

            # vector-valued spaces
            else:
                
                for j in range(3):

                    point_data_log[name][t].append(temp_val[j])

                    # push-forward
                    if space_id == 'Hcurl':
                        point_data_phy[name][t].append(domain.push(
                            temp_val, *grids_log, kind='1_form')[j])
                    elif space_id == 'Hdiv':
                        point_data_phy[name][t].append(domain.push(
                            temp_val, *grids_log, kind='2_form')[j])
                    elif space_id == 'H1vec':
                        point_data_phy[name][t].append(domain.push(
                            temp_val, *grids_log, kind='vector')[j])

    return point_data_log, point_data_phy, grids_log, grids_phy


def create_vtk(path, grids_phy, point_data_phy):
    """
    Creates structured virtual toolkit files (.vts) for Paraview from evaluated field data.
    
    Parameters
    ----------
    path : str
        Absolute path of where to store the .vts files. Will then be in path/vtk/step_<step>.vts.
        
    grids_phy : 3-list
        Mapped (physical) grids obtained from struphy.diagnostics.post_processing.eval_femfields.
        
    point_data_phy : dict
        Pushed-forward field data obtained from struphy.diagnostics.post_processing.eval_femfields.
    """
    
    from pyevtk.hl import gridToVTK
    
    # directory for vtk files
    path_vtk = os.path.join(path, 'vtk')
    
    try:
        os.mkdir(path_vtk)
    except:
        shutil.rmtree(path_vtk)
        os.mkdir(path_vtk)
        
    # field names
    names = list(point_data_phy.keys())
        
    # time loop
    tgrid = list(point_data_phy[names[0]].keys())
    
    nt = len(tgrid) - 1
    log_nt = int(np.log10(nt)) + 1
    
    print('Creating vtk ...')
    for n, t in enumerate(tqdm(tgrid)):
        
        point_data_n = {}
        
        for name in names:
            
            points_list = point_data_phy[name][t]
            
            # scalar
            if len(points_list) == 1:
                point_data_n[name] = points_list[0]
                
            # vector
            else:
                for j in range(3):
                    point_data_n[name + f'_{j + 1}'] = points_list[j]
        
        gridToVTK(os.path.join(path_vtk, 'step_{0:0{1}d}'.format(n, log_nt)), *grids_phy, pointData=point_data_n)


def post_process_markers(path_in, path_out, species, step=1):
    """
    Computes the Cartesian (x, y, z) coordinates of saved markers during a simulation and writes them
    to text files that can be imported to e.g. Paraview (one text file for each time step saved as
    "<name_of_species>_<time_step>.txt" in a directory "kinetic_data/<name_of_species>/orbits/").

    Parameters
    ----------
    path_in : str
        Absolute path to folder with hdf5 data files.
        
    path_out : str
        Absolute path of where to store the .txt files. Will be in path_out/orbits. 

    species : str
        Name of the species for which the post processing should be performed.
        
    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc. 
    """

    # get # of MPI processes from meta.txt file
    with open(os.path.join(path_in, 'meta.txt'), 'r') as f:
        lines = f.readlines()

    nproc = lines[-1].split()[-1]

    # create domain for calculating markers' physical coordinates
    with open(os.path.join(path_in, 'parameters.yml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    domain = setup_domain_mhd(params)[0]
    
    # open hdf5 files and get names and number of saved markers of kinetic species
    files = [h5py.File(os.path.join(path_in, f'data_proc{i}.hdf5'), 'r') for i in range(int(nproc))]

    n_IDs = files[0]['kinetic/' + species + '/markers'].shape[1]

    # directory for .txt files
    path_orbits = os.path.join(path_out, 'orbits')
    
    try:
        os.mkdir(path_orbits)
    except:
        shutil.rmtree(path_orbits)
        os.mkdir(path_orbits)

    t_grid = files[0]['time/value'][::step]

    nt = len(t_grid) - 1
    log_nt = int(np.log10(nt)) + 1

    print('Evaluation of marker orbits for ' + str(species))

    # loop over time grid
    for n, t in enumerate(tqdm(t_grid)):

        # create text file for this time step and this species
        with open(os.path.join(path_orbits, species + '_{0:0{1}d}.txt'.format(n, log_nt)), 'w') as f_out:

            # find markers with right IDs by looping over all hdf5 files and all saved markers
            for ID in range(n_IDs):

                break_flag = False
                for m in range(n_IDs):
                    for file in files:
                        marker = file['kinetic/' +
                                      species + '/markers'][n*step, m, :]

                        if marker[-1] == ID:

                            # compute x, y, z coordinates and write to .txt file
                            X = domain(marker[0], marker[1], marker[2])

                            write_string = '{0:0{1}d}'.format(int(ID), 2)
                            write_string += ',' + str(X[0])
                            write_string += ',' + str(X[1])
                            write_string += ',' + str(X[2])
                                        
                            if int(ID) < n_IDs - 1:
                                write_string += '\n'
                            
                            f_out.write(write_string)
                            break_flag = True
                            break

                    if break_flag:
                        break

    # close hdf5 files
    for file in files:
        file.close()


def post_process_f(path_in, path_out, species, step=1, marker_type='full_f'):
    """
    Computes and saves distribution function of saved binning data during a simulation
    (saved as f_<slice>.npy in a directory "kinetic_data/<name_of_species>/distribution_function/").

    Parameters
    ----------
    path_in : str
        Absolute path to folder with hdf5 data files.
        
    path_out : str
        Absolute path of where to store the .txt files. Will be in path_out/orbits. 

    species : str
        Name of the species for which the post processing should be performed.
        
    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.
    
    marker_type : str
        Which type of markers were simulated.
    """

    # get model name and # of MPI processes from meta.txt file
    with open(os.path.join(path_in, 'meta.txt'), 'r') as f:
        lines = f.readlines()

    model = lines[-6].split()[-1]
    nproc = lines[-1].split()[-1]
    
    # load parameters
    with open(os.path.join(path_in, 'parameters.yml'), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # open hdf5 files
    files = [h5py.File(os.path.join(path_in, f'data_proc{i}.hdf5'), 'r') for i in range(int(nproc))]

    # directory for .npy files
    path_distr = os.path.join(path_out, 'distribution_function')
    
    try:
        os.mkdir(path_distr)
    except:
        shutil.rmtree(path_distr)
        os.mkdir(path_distr)

    print('Evaluation of distribution functions for ' + str(species))

    # loop over saved slices and sum up all ranks
    for slice_name, dset in tqdm(files[0]['kinetic/' + species + '/f'].items()):

        # save grid
        for n_gr, (_, gr) in enumerate(files[0]['kinetic/' + species + '/f/' + slice_name].attrs.items()):
            grid_path = os.path.join(path_distr, 'grid_' + slice_name + '_' + str(n_gr + 1) + '.npy')
            np.save(grid_path, gr[:])

        # load data
        data = dset[::step].copy()
        for rank in range(1, int(nproc)):
            data += files[rank]['kinetic/' + species + '/f/' + slice_name][::step]

        assert marker_type in ['full_f', 'control_variate', 'delta_f'], \
            f'Got unexpected marker type: {marker_type}'

        if marker_type == 'full_f':
            # save distribution function
            np.save(os.path.join(path_distr, 'f_' + slice_name + '.npy'), data)

        else:

            fun_name = params['kinetic'][species]['background']['type']
            bckgr_params = params['kinetic'][species]['background'][fun_name]

            # Get background function
            if fun_name in bckgr_params:
                f_bckgr = getattr(analytical, fun_name)(
                    **bckgr_params[fun_name])
            else:
                f_bckgr = getattr(analytical, fun_name)()

            # multiplier collecting all non-integrated slices in velocity space
            data_bckgr = np.ones(data.shape)

            if isinstance(f_bckgr, analytical.Maxwellian6DUniform):
                for direction in slice_name.split('_'):
                    if direction[0] == 'v':
                        for k in range(len(slice_name.split('_'))):
                            slicing = [None] * len(data.shape)
                            slicing[k + 1] = slice(None)
                            grid_v = files[0]['kinetic/' + species + '/f/' + slice_name].attrs['bin_centers_' + str(k+1)]
                            data_bckgr *= getattr(f_bckgr, 'm' + direction[1])(0, 0, 0, grid_v)[tuple(slicing)]

            else:
                raise NotImplementedError(f'Post-processing is not yet available for background of type {fun_name}')

            if marker_type == 'control_variate':
                data_delta_f = data

            elif marker_type == 'delta_f':

                # Linearized Vlasov-Maxwell system
                if model == "LinearVlasovMaxwell":
                    assert fun_name == 'Maxwellian6DUniform', \
                        'The linearized Vlasov-Maxwell is only implemented for a uniform Maxwellian background!'

                    data_delta_f = np.multiply(data, np.sqrt(data_bckgr))

                else:
                    raise NotImplementedError(f'Post-processing for the model {model} has not been implemented yet!')

            # save distribution function
            np.save(os.path.join(path_distr, 'delta_f_' + slice_name + '.npy'), data_delta_f)
            np.save(os.path.join(path_distr, 'f_' + slice_name + '.npy'), data_delta_f + data_bckgr)

    # close hdf5 files
    for file in files:
        file.close()
