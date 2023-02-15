from struphy.diagnostics.diagn_tools import fourier_1d

import sys
import pickle
import h5py


def main():
    """
    TODO
    """
    path = sys.argv[1]

    # code name
    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # field names
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    names = list(file['feec'].keys())
    file.close()

    # load data dicts for e_field
    with open(path + '/eval_fields/' + names[1] + '_log.bin', 'rb') as handle:
        point_data_log = pickle.load(handle)

    with open(path + '/eval_fields/' + names[1] + '_phy.bin', 'rb') as handle:
        point_data_phys = pickle.load(handle)

    # load grids
    with open(path + '/eval_fields/grids_log.bin', 'rb') as handle:
        grids = pickle.load(handle)

    with open(path + '/eval_fields/grids_phy.bin', 'rb') as handle:
        grids_mapped = pickle.load(handle)

    # fft in (t, z) of first component of e_field on physical grid
    fourier_1d(point_data_log, names[1], code, grids,
               grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
               do_plot=True, disp_name='Maxwell1D')


if __name__ == '__main__':
    main()
