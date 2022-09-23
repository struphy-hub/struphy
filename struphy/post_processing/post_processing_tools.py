import numpy as np

import os, shutil
import yaml
import pickle
from tqdm import tqdm
import h5py

from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
from psydac.api.postprocessing import PostProcessManager
from psydac.fem.basic import FemField
from sympde.topology import Cube

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkHexahedron, VtkQuad


def post_process_fields(path, npts_per_cell=None):
    '''
    Parameters
    ----------
        path : str
            Absolute path to folder with hdf5 data file.

        npts_per_cell : int or 3-tupel or 3-list
            Evaluation grid refinement for in each direction, None or >1. If int, is assumed to be the same in each direction.

    Returns
    -------
        code : str
            From which code the data has been obtained.

        point_data_logic : dict
            Nested dictionary holding point_data_logic of B-spline FemFields on the grid as 3d np.arrays:
            point_data_logic[name][t] contains the point_data_logic of field "name" at time step t. 

        point_data_phys : dict
            Nested dictionary holding point_data_logic of B-spline FemFields on the grid as 3d np.arrays:
            point_data_logic[name][t] contains the point_data_logic of field "name" at time step t. 
            
        grids : dict
            grids[name] contains a 3-list with the logical grids corresponding to point_data_logic[name]. Each entry is 1d grid in one
            eta-direction in the format (Nel[i], npts_per_cell[i]). Grids are equally spaced.
            
        grids_mapped : dict
            grids_mapped[name] contains a 3-list holding 3d np.arrays which are the mapping components F_i evaluated at meshgrid(*grids).

    Notes
    -----
        vtk files for all time steps are saved under path/vtk/.
    '''

    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()

    with open(path + 'MODEL_names.bin', 'rb') as handle:
        li = pickle.load(handle)
        names = li[0]
        space_ids = li[1]

    # code name
    code = lines[-2].split()[-1]
    # number of processes
    nproc = int(lines[-1].split()[-1])

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

    # launch manager with logical domain
    domain_log = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    ppm = PostProcessManager(domain=domain_log,
                              space_file=path + 'FIELD_DATA_spaces.yml',
                              fields_file=path + 'FIELD_DATA_fields.h5')

    Nel = params['grid']['Nel']
    local_domain = ((0, 0, 0), tuple([Ni - 1 for Ni in Nel]))

    if npts_per_cell is None:
        npts_per_cell = [2]*3
    elif isinstance(npts_per_cell, int):
        npts_per_cell = [npts_per_cell + 1]*3
    else:
        raise ValueError('Specify points per cell correctly.')

    # vtk mesh info 
    conn, offsets, celltypes, cell_shape = ppm._compute_unstructured_mesh_info(local_domain, 
                                                                                npts_per_cell=npts_per_cell, 
                                                                                cell_indexes=None)

    dt = params['time']['dt']
    nt = int(params['time']['Tend'] / dt)

    # evaluate fields at evaluation grid and push forward
    point_data_logic = {}
    point_data_phys = {}
    print('Evaluating fields and saving vtk ...')
    for n in tqdm(range(nt + 1)):

        ppm.load_snapshot(n, *names)
        snapshot_dict = ppm._snapshot_fields

        point_data_n = {}
        for i, (name, field) in enumerate(snapshot_dict.items()):

            space_id = space_ids[i]

            if n == 0:
                grids = []
                # create the grid from first snapshot (breaks are always part of the grid)
                for Nel_i, n_i in zip(Nel, npts_per_cell):
                    grids += [np.zeros((Nel_i, n_i), dtype=float)]
                    dx = 1./Nel_i
                    pts = np.linspace(0., dx, n_i)
                    pts[-1] -= 1e-6
                    for ne in range(Nel_i):
                        grids[-1][ne, :] = pts + ne*dx
                    grids[-1] = grids[-1].flatten()

                # physical grids
                grids_mapped = [domain.evaluate(*grids, 'x'), domain.evaluate(*grids, 'y'), domain.evaluate(*grids, 'z')]

                # create point_data dicts for each name
                point_data_logic[name] = {}
                point_data_phys[name] = {}

            temp_val = field.space.eval_fields(grids, field, npts_per_cell=npts_per_cell)[0]

            # scalar spaces
            if isinstance(temp_val, np.ndarray):
                
                point_data_logic[name][n*dt] = [temp_val]

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

        unstructuredGridToVTK(path + 'vtk/step_{0:0{1}d}'.format(n, log_nt),
                                  *grids_mapped,
                                  connectivity=conn,
                                  offsets=offsets,
                                  cell_types=celltypes,
                                  pointData=point_data_n,
                                  cellData=None)

    return code, point_data_logic, point_data_phys, grids, grids_mapped


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
    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()
        
    code = lines[-2].split()[-1]
    nproc = int(lines[-1].split()[-1])

    with open(path + '/parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # create Derham sequence
    derham = Derham(params['grid']['Nel'], 
                    params['grid']['p'],
                    params['grid']['spl_kind'])

    # get names and discrete spaces of fields from 0-th rank hdf5 file
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    
    names = []
    space_ids = []
    spaces = []
    
    for name, dset in file['fields'].items():
        
        names += [name]
        space_ids += [dset.attrs['space_id']]
        spaces += [getattr(derham, derham.spaces_dict[space_ids[-1]])]
    

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
        for name, space in zip(names, spaces):
            fields[n][name] = FemField(space)

    # get hdf5 data
    for rank in range(nproc):

        # open file (0-th rank file is already open!)
        if rank > 0: file = h5py.File(path + '/data_proc' + str(rank) + '.hdf5', 'r')

        for field_name, dset in tqdm(file['fields'].items()):

            # get global start indices, end indices and pads
            gl_s = dset.attrs['starts']
            gl_e = dset.attrs['ends']
            pads = dset.attrs['pads']
            
            assert gl_s.shape == (1, 3) or gl_s.shape == (3, 3)
            assert gl_e.shape == (1, 3) or gl_e.shape == (3, 3)
            assert pads.shape == (1, 3) or pads.shape == (3, 3)
            
            # loop over snapshots
            for n in fields:
                
                # scalar field
                if gl_s.shape[0] == 1:
                    
                    s1, s2, s3 = gl_s[0]
                    e1, e2, e3 = gl_e[0]
                    p1, p2, p3 = pads[0]
                    
                    data = dset[n, p1:-p1, p2:-p2, p3:-p3]
                    
                    fields[n][field_name].coeffs[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = data
                
                # vector-valued field
                else:
                    for comp in range(3):
                        
                        s1, s2, s3 = gl_s[comp]
                        e1, e2, e3 = gl_e[comp]
                        p1, p2, p3 = pads[comp]
                        
                        data = dset[str(comp + 1)][n, p1:-p1, p2:-p2, p3:-p3]
                        
                        fields[n][field_name][comp].coeffs[s1:e1 + 1, s2:e2 + 1, s3:e3 + 1] = data
                               
        file.close()

    print('Creation of PSYDAC FemFields done.')

    return fields, space_ids, code


def eval_femfields(path, fields, space_ids, npts_per_cell=1):
    '''Evaluate B-spline fields obtained from create_femfields
    at cell boundaries (npts_per_cell=1) or refined grid (npts_per_cell>1).
    
    Parameters
    ----------
        fields : dict
            Obtained from struphy.diagnostics.post_processing.create_femfields.

        domain : Struphy object
            
        npts_per_cell : int or 3-tupel or 3-list
            Grid refinement in each eta direction, >0. If int, is assumed to be the same in each direction.

    Returns
    -------
        point_data_logic : dict
            Nested dictionary holding values of B-spline FemFields on the grid as 3d np.arrays:
            values[name][t] contains the values with the name from parameters.yml in ['fields']['general']['names']
            at time step t (see fields from create_femfields). 
            
        point_data_phys :
            Pushed-forward point_data_logic obtained by domain.push().

        grids : 3-list
            1d logical grids in each eta-direction with Nel[i] * npts_per_cell[i] entries. 
            All break points other than 0. and 1. appear twice; double entries can be eliminated by using masks. 
            
        grids_mapped : 3-list
            Mapped grids obtained by domain.evaluate().
            
        masks : 3-list
            Each entry is a boolean list of same size as the corresponding grids entry. 
            It is False where a double counted break point appears, and True otherwise.
            Hence grids[i][masks[i]] gives an equally spaced 1d logical grid.'''


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
    print(Nel)
    local_domain = ((0, 0, 0), tuple([Ni - 1 for Ni in Nel]))

    if npts_per_cell is None:
        npts_per_cell = [2]*3
    elif isinstance(npts_per_cell, int):
        npts_per_cell = [npts_per_cell + 1]*3
    else:
        raise ValueError('Specify points per cell correctly.')

    # vtk mesh info 
    conn, offsets, celltypes, cell_shape = compute_unstructured_mesh_info(local_domain, 
                                                                                npts_per_cell=npts_per_cell, 
                                                                                cell_indexes=None)

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
                masks = []
                # create the grid from first snapshot (breaks are always part of the grid)
                for Nel_i, n_i in zip(Nel, npts_per_cell):
                    grids += [np.zeros((Nel_i, n_i), dtype=float)]
                    dx = 1./Nel_i
                    # grid points in one cell (at least 2, the cell boundaries)
                    pts = np.linspace(0., dx, n_i)
                    for ne in range(Nel_i):
                        grids[-1][ne, :] = pts + ne*dx
                    grids[-1] = grids[-1].flatten()
                    # pts_mask is False for the rigth cell boundary, which is a double counted break point
                    pts_mask = [True]*n_i
                    pts_mask[-1] = False
                    tmp_mask = pts_mask*Nel_i
                    tmp_mask[-1] = True
                    masks += [tmp_mask]

                # physical grids
                grids_mapped = [domain.evaluate(*grids, 'x'), 
                                domain.evaluate(*grids, 'y'), 
                                domain.evaluate(*grids, 'z')]

                # create point_data dicts for each name
                point_data_logic[name] = {}
                point_data_phys[name] = {}

            temp_val = field.space.eval_fields(grids, field, npts_per_cell=npts_per_cell)[0]

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

        unstructuredGridToVTK(path + 'vtk/step_{0:0{1}d}'.format(n, log_nt),
                                  *grids_mapped,
                                  connectivity=conn,
                                  offsets=offsets,
                                  cell_types=celltypes,
                                  pointData=point_data_n,
                                  cellData=None)

    return point_data_logic, point_data_phys, grids, grids_mapped, masks


def compute_unstructured_mesh_info(mapping_local_domain, npts_per_cell=None, cell_indexes=None):
        """
        Computes the connection, offset and celltypes arrays for exportation
        as VTK unstructured grid.

        Parameters
        ----------

        mapping_local_domain : tuple of tuple

        npts_per_cell : tuple of ints or ints, optional

        cell_indexes : tuple of arrays, optional

        Return 
        ------
        connectivity : ndarray
            1D array containing the connectivity between points
        offsets : ndarray 
            1D array containing the index of the last vertex of each cell
        celltypes : ndarray
            1D array containing the type ID of each cell
        cellshape : tuple
            Number of cell in each direction.
        """
        starts, ends = mapping_local_domain
        ldim = len(starts)

        if npts_per_cell is not None:
            n_elem = tuple(ends[i] + 1 - starts[i] for i in range(ldim))
            cellshape = np.array(n_elem) * (np.array(npts_per_cell)) - 1
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim, dtype='i')
            if ldim == 2: 
                celltypes[:] = VtkQuad.tid
                cellID = 0
                for i in range(n_elem[0] * npts_per_cell[0] - 1):
                    for j  in range(n_elem[1] * npts_per_cell[1] - 1):
                        row_top = i 
                        col_left = j

                        # VTK uses Fortran ordering
                        topleft = col_left * npts_per_cell[0] * n_elem[0] + row_top
                        topright = topleft + 1
                        botleft = topleft + n_elem[0] * npts_per_cell[0] # next column
                        botright = botleft + 1

                        connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]

                        cellID += 1
    
            elif ldim == 3:
                celltypes[:] = VtkHexahedron.tid
                cellID = 0
                n_rows = n_elem[0] * npts_per_cell[0]
                n_cols = n_elem[1] * npts_per_cell[1]
                n_layers = n_elem[2] * npts_per_cell[2]
                for i in range(n_rows - 1):
                    for j in range(n_cols - 1):
                        for k in range(n_layers - 1):

                            row_top = i
                            col_left = j 
                            layer_front = k
                            
                            # VTK uses Fortran ordering
                            top_left_front = row_top + col_left * n_rows + layer_front * n_cols * n_rows
                            top_left_back = top_left_front + 1
                            top_right_front = top_left_front + n_rows # next column
                            top_right_back = top_right_front + 1

                            bot_left_front = top_left_front + n_rows * n_cols # next layer
                            bot_left_back = bot_left_front + 1
                            bot_right_front = bot_left_front + n_rows # next column
                            bot_right_back = bot_right_front + 1

                            connectivity[8 * cellID: 8 * cellID + 8] = [
                                top_left_front, top_right_front, bot_right_front, bot_left_front,
                                top_left_back, top_right_back, bot_right_back, bot_left_back
                            ]

                            cellID += 1
        
        elif cell_indexes is not None:
            i_starts = [np.searchsorted(cell_indexes[i], starts[i], side='left') for i in range(ldim)]
            i_ends = [np.searchsorted(cell_indexes[i], ends[i], side='right') for i in range(ldim)]
            n_points = tuple(i_ends[i] - i_starts[i] for i in range(ldim))

            cellshape = np.array([n_points[i] - 1 for i in range(ldim)])
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim, dtype='i')
            if ldim == 2:
                cellID = 0
                celltypes[:] = VtkQuad.tid
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    for j in range(i_ends[1] - 1 - i_starts[1]):

                        row_top = i
                        col_left = j

                        # VTK uses Fortran ordering
                        topleft = row_top + col_left * n_points[0]
                        topright = topleft + 1
                        botleft = topleft + n_points[0]
                        botright = botleft +1

                        connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]
                        cellID += 1

            elif ldim == 3:
                cellID = 0
                celltypes[:] = VtkHexahedron.tid
                n_rows = n_points[0]
                n_cols = n_points[1]
                n_layers = n_points[2]
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    for j in range(i_ends[1] - 1 - i_starts[1]):
                        for k in range(i_ends[2] - 1 - i_starts[2]):
                            row_top = i
                            col_left = j
                            layer_front = k

                            # VTK uses Fortran ordering
                            top_left_front = row_top + col_left * n_rows + layer_front * n_cols * n_rows
                            top_left_back = top_left_front + 1
                            top_right_front = top_left_front + n_rows # next column
                            top_right_back = top_right_front + 1

                            bot_left_front = top_left_front + n_rows * n_cols # next layer
                            bot_left_back = bot_left_front + 1
                            bot_right_front = bot_left_front + n_rows # next column
                            bot_right_back = bot_right_front + 1


                            connectivity[8 * cellID: 8 * cellID + 8] = [
                                top_left_front, top_right_front, bot_right_front, bot_left_front,
                                top_left_back, top_right_back, bot_right_back, bot_left_back
                            ]

                            cellID += 1

        else:
            raise NotImplementedError("Not Supported Yet")

        return connectivity, offsets, celltypes, cellshape


if __name__ == '__main__':
    path = 'struphy/io/out/sim_1/'
    fields, space_ids, code = create_femfields(path)
    point_data_logic, point_data_phys, grids, grids_mapped, masks = eval_femfields(path, fields, space_ids, npts_per_cell=1)