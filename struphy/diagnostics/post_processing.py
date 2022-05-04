import numpy as np
from struphy.feec import spline_space
from struphy.geometry import domain_3d
from struphy.diagnostics import data_module

import os
import yaml
import h5py
from tqdm import tqdm

from struphy.geometry.domain_3d import Domain
from struphy.feec.psydac_derham import Derham_build
from psydac.fem.basic import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace


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

    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()

    # number of processes
    code = lines[-2].split()[-1]
    nproc = int(lines[-1].split()[-1])

    with open(path + 'parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # =========================================================================================
    # DOMAIN object
    # =========================================================================================
    dom_type = params['geometry']['type']
    dom_params = params['geometry']['params_' + dom_type]

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

    DR = Derham_build(Nel, p, spl_kind, F=F_psy)

    # =========================================================================================
    # FemFields
    # =========================================================================================
    # TODO: restart has to be done here
    names = params['fields']['general']['names']
    spaces = params['fields']['general']['spaces']
    dt = params['time']['dt']

    nt = int(params['time']['Tend'] / dt)
    if snapshots == None:
        snapshots = [i for i in range(nt + 1)]
    else:
        assert len(snapshots) <= nt + 1

    fields = {}
    for name, space in zip(names, spaces):

        if space == 'H1':
            _space = DR.V0
        elif space == 'Hcurl':
            _space = DR.V1
        elif space == 'Hdiv':
            _space = DR.V2
        elif space == 'L2':
            _space = DR.V3
        else:
            raise ValueError('Space for field not properly defined.')

        fields[name] = {}
        # create one FemField for each snapshot
        for n in snapshots:
            time = n*dt
            fields[name][time] = FemField(_space)

    # Get hdf5 data
    for n in range(nproc):

        f_name = 'data_proc' + str(n) + '.hdf5'
        #print(f'File: {f_name}')
        f = h5py.File(path + f_name, 'r')

        for key, dset in tqdm(f.items()):

            #print(f'key, shape: {key}, {dset.shape}')
            assert max(snapshots) <= dset.shape[0] - 1

            if len(dset.shape) < 4:
                #print('Skipping time series dataset...')
                continue

            space_cont = dset.attrs['space_cont']
            starts = dset.attrs['starts']
            ends = dset.attrs['ends']
            pads = dset.attrs['pads']

            if space_cont in ('Hcurl', 'Hdiv'):
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
                if space_cont in ('Hcurl', 'Hdiv'):
                    fields[name][time][comp].coeffs[s0:e0 + 1, s1:e1 + 1,
                                            s2:e2 + 1] = dset[t, p0:-p0, p1:-p1, p2:-p2]
                else:
                    fields[name][time].coeffs[s0:e0 + 1, s1:e1 + 1,
                                        s2:e2 + 1] = dset[t, p0:-p0, p1:-p1, p2:-p2]

    print('Creation of FemFields done.')

    return fields, DOMAIN, code


def eval_femfields(fields, DOMAIN, npts_per_cell=1):
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

        _grids = [np.linspace(0, 1, n_i*Nel_i) for Nel_i, n_i in zip(Nel, npts_per_cell)]

        # physical grids
        grids_phys[key] = [DOMAIN.evaluate(*_grids, 'x'), DOMAIN.evaluate(*_grids, 'y'), DOMAIN.evaluate(*_grids, 'z')]

        values[key] = {}

        print(f'Starting evaluation of snapshots for {key}:')
        for time, snapshot in tqdm(field.items()):

            temp_val = snapshot.space.eval_fields(_grids, snapshot, npts_per_cell=npts_per_cell)[0]

            if isinstance(temp_val, np.ndarray):
                temp_val = [temp_val]

            values[key][time] = temp_val

        # reshape grid
        for i in range(3):
            _grids[i] = np.reshape(_grids[i], newshape=(Nel[i], npts_per_cell[i]))

        grids[key] = _grids

    return values, grids, grids_phys


class Post_processing:
    """
    Evaluate the data from a simulation at specific points

    Parameters
    ----------
        params : dict
            parameters from parameters.yml 

        data : hdf5 file
            data of a simulation from data.hdf5
    """

    def __init__(self, params, data):
        self.params = params
        self.data = data

    def construct_FEEC_spaces(self, pts=None):
        """
        Construct the spaces object for the evaluation

        Parameters
        ----------
            pts : ndarray(dtype=float, ndim=3) [optional]
                Points for the evaluation in each direction (default is element boundaries).

        Notes
        -----
            If pts not given the variables from the SPACES object will be used.
        """

        # mesh parameters
        Nel = self.params['grid']['Nel']
        spl_kind = self.params['grid']['spl_kind']
        p = self.params['grid']['p']
        nq_el = self.params['grid']['nq_el']
        bc = self.params['grid']['bc']

        # domain object
        domain_type = self.params['geometry']['type']
        self.DOMAIN = domain_3d.Domain(
            domain_type, self.params['geometry']['params_' + domain_type])

        # spline spaces
        spaces_FEM_1 = spline_space.Spline_space_1d(
            Nel[0], p[0], spl_kind[0], nq_el[0], bc)
        spaces_FEM_2 = spline_space.Spline_space_1d(
            Nel[1], p[1], spl_kind[1], nq_el[1])
        spaces_FEM_3 = spline_space.Spline_space_1d(
            Nel[2], p[2], spl_kind[2], nq_el[2])

        # tensor spline space
        self.SPACES = spline_space.Tensor_spline_space(
            [spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

        # get time and space variables
        self.t = self.data['time'][:].reshape(
            (self.data['time'][:].shape[0], ))
        if pts is None:
            self.eta1 = self.SPACES.el_b[0]
            self.eta2 = self.SPACES.el_b[1]
            self.eta3 = self.SPACES.el_b[2]
        else:
            self.eta1 = pts[0]
            self.eta2 = pts[1]
            self.eta3 = pts[2]

        # get time and space sizes
        self.Nt = int(self.t.shape[0])
        self.Nx = int(self.eta1.shape[0])
        self.Ny = int(self.eta2.shape[0])
        self.Nz = int(self.eta3.shape[0])
        self.sizes = np.array([self.Nt, self.Nx, self.Ny, self.Nz]).astype(int)

    def __evaluate_data(self, form, quantity):
        """
        Evaluate the data from a simulation for a given quantity and the associated form

        Parameters
        ----------
            quantity : string
                key for the data.hdf5 file where the quantity is saved

            form : string
                from of the desired quantity

        Returns
        -------
            eval_values: ndarray
                evaluated_values in the physical space at (t, eta1, eta2, eta3) for each coordinate

        Notes
        -----
            Possible choices for form:

                * '0_form', 
                * '1_form_1',
                * '1_form_2',
                * '2_form_3',
                * '2_form_1',
                * '2_form_2',
                * '2_form_3',
                * '3_form'
        """

        total_stetps = str(self.Nt-1)
        str_len = len(total_stetps)
        print('Evaluate ' + quantity + ' as ' + form)

        if form == '0_form':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs = self.SPACES.extract_0(quantity[tn])
                eval_spline = self.SPACES.evaluate_NNN(
                    self.eta1, self.eta2, self.eta3, coeffs)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    eval_spline, self.eta1, self.eta2, self.eta3, '0_form')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '1_form_1':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_1(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_DNN(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_NDN(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_NND(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_1')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '1_form_2':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_1(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_DNN(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_NDN(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_NND(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_2')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '1_form_3':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_1(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_DNN(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_NDN(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_NND(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '1_form_3')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '2_form_1':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_2(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_NDD(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_DND(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_DDN(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_1')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '2_form_2':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_2(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_NDD(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_DND(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_DDN(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_2')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '2_form_3':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs_1, coeffs_2, coeffs_3 = self.SPACES.extract_2(
                    quantity[tn])
                eval_spline_1 = self.SPACES.evaluate_NDD(
                    self.eta1, self.eta2, self.eta3, coeffs_1)
                eval_spline_2 = self.SPACES.evaluate_DND(
                    self.eta1, self.eta2, self.eta3, coeffs_2)
                eval_spline_3 = self.SPACES.evaluate_DDN(
                    self.eta1, self.eta2, self.eta3, coeffs_3)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    [eval_spline_1, eval_spline_2, eval_spline_3], self.eta1, self.eta2, self.eta3, '2_form_3')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        elif form == '3_form':

            quantity = self.data[quantity]
            eval_values = np.zeros((self.Nt, self.Nx, self.Ny, self.Nz))

            for tn in range(self.Nt):
                coeffs = self.SPACES.extract_3(quantity[tn])
                eval_spline = self.SPACES.evaluate_DDD(
                    self.eta1, self.eta2, self.eta3, coeffs)
                eval_values[tn, :, :, :] = self.DOMAIN.push(
                    eval_spline, self.eta1, self.eta2, self.eta3, '3_form')
                print('\r', 'Steps : ' + str(tn).zfill(str_len) +
                      '/' + total_stetps, end='')

            return eval_values

        else:
            print('Wrong form')
            pass

    def save_evaluated_data(self, quantities, path_out=None, data_name=None):
        """
        Evaluate the data from the simulation for all quantities in quantities dict

        Parameters
        ----------
            quantities : dict
                dict where the keys are the quantities and the values the associated forms

            path_out : string
                path string where the data will be saved

            path_out : string
                name of the evaluated data
        """

        if path_out is None:
            path_out = os.getcwd()
        if data_name is None:
            data_name = 'eval_data.hdf5'

        name = {'0_form': '',
                '1_form_1': ' x',
                '1_form_2': ' y',
                '1_form_3': ' z',
                '2_form_1': ' x',
                '2_form_2': ' y',
                '2_form_3': ' z',
                '3_form': ''}

        DATA = data_module.Data_container(
            path_out=path_out, data_name=data_name)
        print()
        for form, quantity in quantities.items():

            eval_values = self.__evaluate_data(form, quantity)
            print()
            DATA.add_data({quantity + name[form]: eval_values})

        DATA.file.close()


if __name__ == '__main__':
    create_femfields('struphy/io/out/sim_1/')
