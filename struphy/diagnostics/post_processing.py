import numpy as np
from struphy.feec import spline_space
from struphy.geometry import domain_3d
from struphy.diagnostics import data_module

import os
import yaml
import pickle
import h5py
from tqdm import tqdm

from struphy.geometry.domain_3d import Domain
from struphy.psydac_api.psydac_derham import Derham
from psydac.fem.basic import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace
from psydac.api.postprocessing import PostProcessManager
from sympde.topology import Cube


def post_process_fields(path, snapshots=None, npts_per_cell=1):
    '''
    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data file.

        snapshots : list of int
            Time indices at which FemFields are to be created; must be <= #time_steps. If None, all time steps are processed.

        npts_per_cell : int or 3-tupel or 3-list
            Grid refinement in each eta direction, >0. If int, is assumed to be the same in each direction.

    Returns
    -------
        code : str
            From which code the data has been obtained.

        values : dict
            Nested dictionary holding values of B-spline FemFields on the grid as 3d np.arrays:
            values[name][t] contains the values of field "name" at time step t. 
            
        grids : dict
            grids[name] contains a 3-list with the logical grids corresponding to values[name]. Each entry is 1d grid in one
            eta-direction in the format (Nel[i], npts_per_cell[i]). Grids are equally spaced.
            
        grids_phys : dict
            grids_phys[name] contains a 3-list holding 3d np.arrays which are the mapping components F_i evaluated at meshgrid(*grids).
    '''


    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

    with open(path + '/MODEL_names.bin', 'rb') as handle:
        names = pickle.load(handle)[0]

    # code name
    code = lines[-2].split()[-1]
    # number of processes
    nproc = int(lines[-1].split()[-1])

    with open(path + '/parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # domain object
    dom_type = params['geometry']['type']
    dom_params = params['geometry'][dom_type]
    DOMAIN = Domain(dom_type, dom_params)
    F_psy = DOMAIN.Psydac_mapping('F', **dom_params)

    domain_log = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))

    POST = PostProcessManager(domain=domain_log,
                              space_file=path + '/FIELD_DATA_spaces.yml',
                              fields_file=path + '/FIELD_DATA_fields.h5',
                              ncells=params['grid']['Nel'])

    V0 = POST.spaces['V0']
    V1 = POST.spaces['V1']
    V2 = POST.spaces['V2']
    V3 = POST.spaces['V3']

    dt = params['time']['dt']
    nt = int(params['time']['Tend'] / dt)
    if snapshots == 'all':
        snapshots = [i for i in range(nt + 1)]
    else:
        assert len(snapshots) <= nt + 1

    values = {}
    grids = {}
    grids_phys = {}
    for n in snapshots:
        POST.load_snapshot(n, names)
        snapshot_dict = POST._snapshot_fields
        print(snapshot_dict.keys())

        for name, field in snapshot_dict.items():

            if n == snapshots[0]:
                # create the grid from first snapshot
                if isinstance(field.space, ProductFemSpace):
                    Nel = [space.ncells for space in field.space.spaces[0].spaces]
                else:
                    assert isinstance(field.space, TensorFemSpace)
                    Nel = [space.ncells for space in field.space.spaces]

                _grids = [np.linspace(0, 1, n_i*Nel_i) for Nel_i, n_i in zip(Nel, npts_per_cell)]

                # physical grids
                grids_phys[name] = [DOMAIN.evaluate(*_grids, 'x'), DOMAIN.evaluate(*_grids, 'y'), DOMAIN.evaluate(*_grids, 'z')]

                # reshape grid
                for i in range(3):
                    _grids[i] = np.reshape(_grids[i], newshape=(Nel[i], npts_per_cell[i]))

                grids[name] = _grids

                # values on grid
                values[name] = {}

            temp_val = field.space.eval_fields(_grids, field, npts_per_cell=npts_per_cell)[0]

            if isinstance(temp_val, np.ndarray):
                temp_val = [temp_val]

            values[name][n*dt] = temp_val

    mesh, all_fields = POST.export_to_vtk(path + '/view_data', 
                                          grids, 
                                          npts_per_cell=npts_per_cell, 
                                          snapshots='all', 
                                          debug=True)

    return code, values, grids, grids_phys, mesh, all_fields


def create_femfields(path, snapshots=None):
    '''Creates all Psydac FemFields from distributed Struphy data.

    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data files.

        snapshots : list of int
            Time indices at which FemFields are to be created; must be <= #time_steps. 

    Returns
    -------
        fields : dict
            Nested dictionary holding FemFields: fields[name][t] contains the Femfield 
            with the name from parameters.yml in ['fields']['general']['names'] at time step t,
            where t is an integer from snapshots.

        DOMAIN : Struphy object

        code : str
            From which code the data has been obtained.
    '''

    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

    with open(path + '/MODEL_names.bin', 'rb') as handle:
        tmp = pickle.load(handle)
        names = tmp[0]
        space_ids = tmp[1]
        del tmp

    # code name
    code = lines[-2].split()[-1]
    # number of processes
    nproc = int(lines[-1].split()[-1])

    with open(path + '/parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # =========================================================================================
    # DOMAIN object
    # =========================================================================================
    dom_type = params['geometry']['type']
    dom_params = params['geometry'][dom_type]

    DOMAIN = Domain(dom_type, dom_params)
    # create psydac mapping for mass matrices only
    F_psy = DOMAIN.Psydac_mapping('F', **dom_params)

    # =========================================================================================
    # DERHAM sequence (Psydac)
    # =========================================================================================
    # Grid parameters
    Nel = params['grid']['Nel']             # Number of grid cells
    p = params['grid']['p']               # spline degree
    spl_kind = params['grid']['spl_kind']        # Spline type

    DR = Derham(Nel, p, spl_kind, F=F_psy)

    # =========================================================================================
    # FemFields
    # =========================================================================================
    dt = params['time']['dt']
    nt = int(params['time']['Tend'] / dt)
    if snapshots == None:
        snapshots = [i for i in range(nt + 1)]
    else:
        assert len(snapshots) <= nt + 1

    fields = {}
    spaces = {}
    for name, space in zip(names, space_ids):
        
        spaces[name] = space

        if space == 'H1':
            _space = DR.V0
        elif space == 'Hcurl':
            _space = DR.V1
        elif space == 'Hdiv':
            _space = DR.V2
        elif space == 'L2':
            _space = DR.V3
        elif space == 'H1vec':
            _space = DR.V0vec
        else:
            raise ValueError('Space for field not properly defined.')

        fields[name] = {}
        # create one FemField for each snapshot
        for n in snapshots:
            time = n*dt
            fields[name][time] = FemField(_space)

    # Get hdf5 data
    for n in range(nproc):

        f_name = '/data_proc' + str(n) + '.hdf5'
        #print(f'File: {f_name}')
        f = h5py.File(path + f_name, 'r')

        for key, dset in tqdm(f.items()):

            #print(f'key, shape: {key}, {dset.shape}')
            assert max(snapshots) <= dset.shape[0] - 1

            if len(dset.shape) < 4:
                #print('Skipping time series dataset...')
                continue

            space_id = dset.attrs['space_id']
            starts = dset.attrs['starts']
            ends = dset.attrs['ends']
            pads = dset.attrs['pads']

            if space_id in ('Hcurl', 'Hdiv', 'H1vec'):
                comp = int(key[-1])
                name = key[:-2]
            else:
                comp = 0
                name = key

            assert name in fields

            #print(f'name, comp: {name}, {comp}')

            s0 = starts[comp][0]
            s1 = starts[comp][1]
            s2 = starts[comp][2]

            e0 = ends[comp][0]
            e1 = ends[comp][1]
            e2 = ends[comp][2]

            p0 = pads[comp][0]
            p1 = pads[comp][1]
            p2 = pads[comp][2]

            for t, time in enumerate(fields[name]):
                if space_id in ('Hcurl', 'Hdiv', 'H1vec'):
                    fields[name][time][comp].coeffs[s0:e0 + 1, s1:e1 + 1,
                                            s2:e2 + 1] = dset[t, p0:-p0, p1:-p1, p2:-p2]
                else:
                    fields[name][time].coeffs[s0:e0 + 1, s1:e1 + 1,
                                        s2:e2 + 1] = dset[t, p0:-p0, p1:-p1, p2:-p2]

    print('Creation of FemFields done.')

    return fields, spaces, DOMAIN, code


def eval_femfields(fields, spaces, DOMAIN, npts_per_cell=1):
    '''Evaluate B-spline fields obtained from create_femfields
    at cell boundaries (npts_per_cell=1) or refined grid (npts_per_cell>1).
    
    Parameters
    ----------
        fields : dict
            Obtained from struphy.diagnostics.post_processing.create_femfields.

        DOMAIN : Struphy object
            
        npts_per_cell : int or 3-tupel or 3-list
            Grid refinement in each eta direction, >0. If int, is assumed to be the same in each direction.

    Returns
    -------
        values : dict
            Nested dictionary holding values of B-spline FemFields on the grid as 3d np.arrays:
            values[name][t] contains the values with the name from parameters.yml in ['fields']['general']['names']
            at time step t (see fields from create_femfields). 
            
        grids : dict
            grids[name] contains a 3-list with the logical grids corresponding to values[name]. Each entry is 1d grid in one
            eta-direction in the format (Nel[i], npts_per_cell[i]). Grids are equally spaced.
            
        grids_phys : dict
            grids_phys[name] contains a 3-list holding 3d np.arrays which are the mapping components F_i evaluated at meshgrid(*grids).'''

    assert isinstance(fields, dict)

    if isinstance(npts_per_cell, int):
        npts_per_cell = (npts_per_cell,) * 3

    values = {}
    values_phys = {}
    grids = {}
    grids_phys = {}
    
    for key, field in tqdm(fields.items()):

        # create the grid from first snapshot
        snapshot = field[list(field.keys())[0]]
        if isinstance(snapshot.space, ProductFemSpace):
            Nel = [space.ncells for space in snapshot.space.spaces[0].spaces]
        else:
            assert isinstance(snapshot.space, TensorFemSpace)
            Nel = [space.ncells for space in snapshot.space.spaces]

        _grids = [np.linspace(0, 1, n_i*Nel_i + 1)[:-1] for Nel_i, n_i in zip(Nel, npts_per_cell)]

        # physical grids
        grids_phys[key] = [DOMAIN.evaluate(*_grids, 'x'), DOMAIN.evaluate(*_grids, 'y'), DOMAIN.evaluate(*_grids, 'z')]

        values[key] = {}
        values_phys[key] = {}

        print(f'Starting evaluation of snapshots for {key}:')
        for time, snapshot in tqdm(field.items()):

            temp_val = snapshot.space.eval_fields(_grids, snapshot, npts_per_cell=npts_per_cell)[0]

            if isinstance(temp_val, np.ndarray):
                temp_val = (temp_val,)

            values[key][time] = temp_val
            
            # physical fields via push-forwards
            if   spaces[key] == 'H1':
                values_phys[key][time] = (DOMAIN.push(temp_val[0], *_grids, '0_form'),)
            elif spaces[key] == 'Hcurl':
                values_phys[key][time] = (DOMAIN.push(temp_val, *_grids, '1_form_1'),
                                          DOMAIN.push(temp_val, *_grids, '1_form_2'),
                                          DOMAIN.push(temp_val, *_grids, '1_form_3'))
            elif spaces[key] == 'Hdiv':
                values_phys[key][time] = (DOMAIN.push(temp_val, *_grids, '2_form_1'),
                                          DOMAIN.push(temp_val, *_grids, '2_form_2'),
                                          DOMAIN.push(temp_val, *_grids, '2_form_3'))
            elif spaces[key] == 'L2':
                values_phys[key][time] = (DOMAIN.push(temp_val[0], *_grids, '3_form'),)
            elif spaces[key] == 'H1vec':
                values_phys[key][time] = (DOMAIN.push(temp_val, *_grids, 'vector_1'),
                                          DOMAIN.push(temp_val, *_grids, 'vector_2'),
                                          DOMAIN.push(temp_val, *_grids, 'vector_3'))
            

        # reshape grid
        for i in range(3):
            _grids[i] = np.reshape(_grids[i], newshape=(Nel[i], npts_per_cell[i]))

        grids[key] = _grids

    return values, values_phys, grids, grids_phys


if __name__ == '__main__':
    create_femfields('struphy/io/out/sim_1/')
