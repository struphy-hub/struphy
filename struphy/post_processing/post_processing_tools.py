import numpy as np

import os, shutil
import yaml
from tqdm import tqdm
import h5py

from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
from struphy.psydac_api.fields import Field
from psydac.fem.basic import FemField

from pyevtk.hl import gridToVTK

def create_femfields(path, snapshots=None):
    '''Creates all Psydac FemFields from distributed Struphy data.

    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data files.

        snapshots : list/array of int
            Time indices at which FemFields are to be created; must be <= #time_steps. 

    Returns
    -------
        fields : dict
            Nested dictionary holding psydac FemFields: fields[n][name] contains the Femfield of the field with the name "name" in the hdf5 file at time step n.

        space_ids : list of ints
            The space IDs of the fields (H1, Hcurl, Hdiv, L2 or H1vec).

        code : str
            From which code the data has been obtained.
    '''

    # get code name and # of MPI processes
    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()
        
    code = lines[-2].split()[-1]
    nproc = int(lines[-1].split()[-1])

    with open(path + 'parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # create Derham sequence
    derham = Derham(params['grid']['Nel'], 
                    params['grid']['p'],
                    params['grid']['spl_kind'])

    # get names and discrete spaces of fields from 0-th rank hdf5 file
    file = h5py.File(path + 'data_proc0.hdf5', 'r')
    
    names = []
    space_ids = []
    spaces = []
    
    assert 'feec' in file, 'No fields saved under feec/ in .hdf5 output.' 

    for name, dset in file['feec'].items():

        names += [name]
        space_ids += [dset.attrs['space_id']]
        spaces += [derham.Vh_fem[derham.spaces_dict[space_ids[-1]]]]
    
    # create FemFields
    dt = params['time']['dt']
    nt = int(params['time']['Tend']/dt)
    
    if snapshots is None:
        snapshots = [i for i in range(nt + 1)]
    else:
        assert max(snapshots) <= nt
        assert len(snapshots) <= nt + 1

    # create one FemField for each snapshot
    fields = {}
    for n in snapshots:
        fields[n] = {}
        for name, ID in zip(names, space_ids):
            fields[n][name] = Field(name, ID, derham)

    # get hdf5 data
    for rank in range(nproc):

        # open file (0-th rank file is already open!)
        if rank > 0: file = h5py.File(path + 'data_proc' + str(rank) + '.hdf5', 'r')

        for field_name, dset in tqdm(file['feec'].items()):

            # get global start indices, end indices and pads
            gl_s = dset.attrs['starts']
            gl_e = dset.attrs['ends']
            pads = dset.attrs['pads']
            
            assert gl_s.shape == (3,) or gl_s.shape == (3, 3)
            assert gl_e.shape == (3,) or gl_e.shape == (3, 3)
            assert pads.shape == (3,) or pads.shape == (3, 3)
            
            # loop over snapshots
            for n in fields:
                
                # scalar field
                if gl_s.shape == (3,):
                    
                    s1, s2, s3 = gl_s
                    e1, e2, e3 = gl_e
                    p1, p2, p3 = pads
                    
                    data = dset[n, p1:-p1, p2:-p2, p3:-p3]
                    
                    fields[n][field_name].vector[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = data
                    fields[n][field_name].vector.update_ghost_regions() # update after each data addition, can be made more efficient
                
                # vector-valued field
                else:
                    for comp in range(3):
                        
                        s1, s2, s3 = gl_s[comp]
                        e1, e2, e3 = gl_e[comp]
                        p1, p2, p3 = pads[comp]
                        
                        data = dset[str(comp + 1)][n, p1:-p1, p2:-p2, p3:-p3]
                        
                        fields[n][field_name].vector[comp][s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = data
                    fields[n][field_name].vector.update_ghost_regions() # update after each data addition, can be made more efficient
                               
        file.close()

    print('Creation of PSYDAC FemFields done.')

    return fields, space_ids, code

def eval_femfields(path, fields, space_ids, cell_divide=None):
    '''Evaluate B-spline fields obtained from create_femfields
    at cell boundaries (cell_divide = 1) or refined grid (cell_divide > 1).
    Creates structured virtual toolkit files (.vts) for Paraview. 
    
    Parameters
    ----------
        path : str
            Path of simulation output folder.
    
        fields : dict
            Obtained from struphy.diagnostics.post_processing.create_femfields.

        space_ids : list[int]
            The space IDs of the fields (H1, Hcurl, Hdiv, L2 or H1vec).
            
        cell_divide : int or 3-tupel or 3-list
            Grid refinement in each eta direction. If int, is assumed to be the same in each direction.

    Returns
    -------
        point_data_logic : dict
            Nested dictionary holding values of B-spline FemFields on the grid as 3d np.arrays:
            values[name][t] contains the values with the name from parameters.yml in ['fields']['general']['names']
            at time step t (see fields from create_femfields). 
            
        point_data_phys :
            Pushed-forward point_data_logic obtained by domain.push().

        grids : 3-list
            1d logical grids in each eta-direction with Nel[i]*cell_divide[i] + 1 entries in each direction.  
            
        grids_mapped : 3-list
            Mapped grids obtained by domain.evaluate().
    '''

    assert isinstance(fields, dict)

    with open(path + 'parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # directory for vtk files
    try:
        os.mkdir(path + 'vtk/')
    except:
        shutil.rmtree(path + 'vtk/')
        os.mkdir(path + 'vtk/')

    # domain object 
    dom_type = params['geometry']['type']
    dom_params = params['geometry'][dom_type]
    
    domain_class = getattr(domains, dom_type)
    domain = domain_class(dom_params)

    Nel = params['grid']['Nel']

    if cell_divide is None:
        cell_divide = [1]*3
    elif isinstance(cell_divide, int) and cell_divide > 0:
        cell_divide = [cell_divide]*3
    else:
        raise ValueError('Specify cell divide correctly.')

    dt = params['time']['dt']
    nt = int(params['time']['Tend'] / dt)

    # evaluate fields at evaluation grid and push forward
    point_data_logic = {}
    point_data_phys = {}
    print('Evaluating fields and saving vtk ...')
    for n in tqdm(fields):

        snapshot_dict = fields[n]

        point_data_n = {}
        for i, (name, field) in enumerate(snapshot_dict.items()):

            space_id = space_ids[i]

            if n == 0:
                grids = []
                # create the grid from first snapshot (breaks are always part of the grid)
                for Nel_i, n_i in zip(Nel, cell_divide):
                    grids += [np.linspace(0., 1., Nel_i*n_i + 1)]

                # physical grids
                grids_mapped = [domain.evaluate(*grids, 'x'), 
                                domain.evaluate(*grids, 'y'), 
                                domain.evaluate(*grids, 'z')]

                # create point_data dicts for each name
                point_data_logic[name] = {}
                point_data_phys[name] = {}

            # field evaluation
            temp_val = field(*grids)

            # scalar spaces
            if isinstance(temp_val, np.ndarray):
                
                point_data_logic[name][n*dt] = (temp_val,)

                # point data for vtk file at time n
                if space_id == 'H1':
                    point_data_n[name] = domain.push(temp_val, *grids, '0_form')
                elif space_id == 'L2':
                    point_data_n[name] = domain.push(temp_val, *grids, '3_form')

                point_data_phys[name][n*dt] = [point_data_n[name]]

            # vector-valued spaces
            else:

                point_data_logic[name][n*dt] = temp_val

                # point data for vtk file at time n
                if space_id == 'Hcurl':
                    for j in range(3):
                        point_data_n[name + f'_{j + 1}'] = domain.push(temp_val, *grids, f'1_form_{j + 1}')
                elif space_id == 'Hdiv':
                    for j in range(3):
                        point_data_n[name + f'_{j + 1}'] = domain.push(temp_val, *grids, f'2_form_{j + 1}')
                elif space_id == 'H1vec':
                    for j in range(3):
                        point_data_n[name + f'_{j + 1}'] = domain.push(temp_val, *grids, f'vector_{j + 1}') 

                point_data_phys[name][n*dt] = [point_data_n[name + f'_{j + 1}'] for j in range(3)]
        
        log_nt = int(np.log10(nt)) + 1

        gridToVTK(path + 'vtk/step_{0:0{1}d}'.format(n, log_nt), *grids_mapped, pointData = point_data_n)

    return point_data_logic, point_data_phys, grids, grids_mapped

def post_process_markers(path, species):
    """
    Computes the Cartesian (x, y, z) coordinates of saved markers during a simulation and writes them to text files that can be imported to e.g. Paraview (one text file for each time step saved as "<name_of_species>_<time_step>.txt" in a directory "kinetic_data/<name_of_species>/orbits/").
    
    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data files.
            
        species : str
            Name of the species for which the post processing should be performed.
    """
    
    # get code name and # of MPI processes
    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()
        
    code = lines[-2].split()[-1]
    nproc = int(lines[-1].split()[-1])

    with open(path + 'parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    # domain object 
    dom_type = params['geometry']['type']
    dom_params = params['geometry'][dom_type]
    
    domain_class = getattr(domains, dom_type)
    domain = domain_class(dom_params)
    
    # open hdf5 files and get names and number of saved markers of kinetic species
    files = [h5py.File(path + f'data_proc{i}.hdf5', 'r') for i in range(nproc)]
    
    n_IDs = files[0]['kinetic/' + species + '/markers'].shape[1]

    try:
        os.mkdir(path + 'kinetic_data/' + species + '/orbits/')
    except:
        shutil.rmtree(path + 'kinetic_data/' + species + '/orbits/')
        os.mkdir(path + 'kinetic_data/' + species + '/orbits/')
            
    
    dt = params['time']['dt']
    nt = int(params['time']['Tend']/dt)
    t = np.linspace(0., params['time']['Tend'], nt + 1)
    
    log_nt = int(np.log10(nt)) + 1
    
    print('Evaluation of marker orbits for ' + str(species))
    
    # loop over time
    for n in tqdm(range(nt + 1)):
        
        # create text file for this time step and this species
        with open(path + 'kinetic_data/' + species + '/orbits/' + species + '_{0:0{1}d}.txt'.format(n, log_nt), 'w') as f_out:

            # find markers with right IDs by looping over all hdf5 files and all saved markers
            for ID in range(n_IDs):

                break_flag = False
                for m in range(n_IDs):
                    for file in files:
                        marker = file['kinetic/' + species + '/markers'][n, m, :]

                        if marker[-1] == ID:

                            # compute x, y, z coordinates and write to .txt file
                            X = domain(marker[0], marker[1], marker[2])

                            f_out.write('{0:0{1}d}'.format(int(ID), 2) 
                                        + ',' + str(X[0])
                                        + ',' + str(X[1]) 
                                        + ',' + str(X[2]) + '\n')
                            break_flag = True
                            break

                    if break_flag: break

       
    # close hdf5 files
    for file in files:
        file.close()
        
def post_process_f(path, species):
    """
    Computes and saved distribution function of saved binning data during a simulation (saved as f_<slice>.npy in a directory "kinetic_data/<name_of_species>/distribution_function/").
    
    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data files.
            
        species : str
            Name of the species for which the post processing should be performed.
    """
    
    # get # of MPI processes
    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()
        
    nproc = int(lines[-1].split()[-1])
        
    # open hdf5 files
    files = [h5py.File(path + f'data_proc{i}.hdf5', 'r') for i in range(nproc)]
    
    # create directories
    try:
        os.mkdir(path + 'kinetic_data/' + species + '/f/')
    except:
        shutil.rmtree(path + 'kinetic_data/' + species + '/f/')
        os.mkdir(path + 'kinetic_data/' + species + '/f/')
        
    print('Evaluation of distribution functions for ' + str(species))
    
    # loop over saved slices and sum up all ranks
    for slice_name, dset in tqdm(files[0]['kinetic/' + species + '/f'].items()):
        
        # save grid
        for n_gr, (gr_name, gr) in enumerate(files[0]['kinetic/' + species + '/f/' + slice_name].attrs.items()):
            np.save(path + 'kinetic_data/' + species + '/f/grid_' + slice_name + '_' + str(n_gr + 1) + '.npy', gr[:])
        
        data = dset[:]
        for rank in range(1, nproc):
            data += files[rank]['kinetic/' + species + '/f/' + slice_name][:]
            
        # save distribution function
        np.save(path + 'kinetic_data/' + species + '/f/f_' + slice_name + '.npy', data)
        
    # close hdf5 files
    for file in files:
        file.close()
    
            
if __name__ == '__main__':
    path = 'struphy/io/out/sim_1/'
    fields, space_ids, code = create_femfields(path)
    point_data_logic, point_data_phys, grids, grids_mapped = eval_femfields(path, fields, space_ids)