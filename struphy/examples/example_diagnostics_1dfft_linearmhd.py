
import pickle
import sys
import yaml
import h5py

from struphy.diagnostics.diagn_tools import fourier_1d


def main():
    """
    TODO
    """
    path = sys.argv[1]

    # read in parameters for analytical dispersion relation
    with open(path + '/parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    B0x = params['mhd_equilibrium']['HomogenSlab']['B0x']
    B0y = params['mhd_equilibrium']['HomogenSlab']['B0y']
    B0z = params['mhd_equilibrium']['HomogenSlab']['B0z']

    p0 = (2*params['mhd_equilibrium']['HomogenSlab']
          ['beta']/100)/(B0x**2 + B0y**2 + B0z**2)
    n0 = params['mhd_equilibrium']['HomogenSlab']['n0']

    gamma = 5/3

    disp_params = {'B0x': B0x, 'B0y': B0y,
                   'B0z': B0z, 'p0': p0, 'n0': n0, 'gamma': 5/3}

    # code name
    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # field names
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    names = list(file['feec'].keys())
    file.close()

    # load grids
    with open(path + '/eval_fields/grids_log.bin', 'rb') as handle:
        grids = pickle.load(handle)

    with open(path + '/eval_fields/grids_phy.bin', 'rb') as handle:
        grids_mapped = pickle.load(handle)

    # load data dicts for u_field
    with open(path + '/eval_fields/' + names[3] + '_log.bin', 'rb') as handle:
        point_data_log = pickle.load(handle)

    with open(path + '/eval_fields/' + names[3] + '_phy.bin', 'rb') as handle:
        point_data_phys = pickle.load(handle)

    # fft in (t, z) of first component of u_field on physical grid
    fourier_1d(point_data_log, names[3], code, grids,
               grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
               do_plot=True, disp_name='Mhd1D', disp_params=disp_params)

    # load data dicts for pressure
    with open(path + '/eval_fields/' + names[2] + '_log.bin', 'rb') as handle:
        point_data_log = pickle.load(handle)

    with open(path + '/eval_fields/' + names[2] + '_phy.bin', 'rb') as handle:
        point_data_phys = pickle.load(handle)

    # fft in (t, z) of pressure on physical grid
    fourier_1d(point_data_log, names[2], code, grids,
               grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
               do_plot=True, disp_name='Mhd1D', disp_params=disp_params)


if __name__ == '__main__':
    main()
