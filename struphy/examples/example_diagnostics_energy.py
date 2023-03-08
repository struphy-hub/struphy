import yaml
import h5py
import sys
import numpy as np

import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.fields_background.mhd_equil import equils
from struphy.models.base import setup_domain_mhd

from struphy.dispersion_relations.analytic import PC_LinMHD_6d_full1D


def main():
    """
    TODO
    """
    sim_path = sys.argv[1]

    # load parameters file
    with open(sim_path + 'parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # create domain and MHD equilibrium
    domain, mhd_equil = setup_domain_mhd(params)

    params_homogenslab = params['fields']['mhd_equilibrium']['HomogenSlab']
    dispersion = PC_LinMHD_6d_full1D(params_homogenslab)

    file = h5py.File(sim_path + 'data_proc0.hdf5', 'r')

    t = file['scalar']['time'][:]
    en_U = file['scalar']['en_U'][:]

    fig = plt.figure()
    plt.plot(t[:len(en_U)], en_U, color='r')
    plt.plot(t, 1e-8*np.e**(2*t*dispersion([0.8])['shear Alfvén_R'].imag))
    plt.yscale('log')

    plt.show()


if __name__ == '__main__':
    main()
