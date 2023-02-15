import yaml
import h5py
import sys
import numpy as np

import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.fields_background.mhd_equil import equils

from struphy.dispersion_relations.analytic import PC_LinMHD_6d_full1D


def main():
    """
    TODO
    """
    sim_path = sys.argv[1]

    # load parameters and markers
    with open(sim_path + 'parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # load domain
    dom_type = params['geometry']['type']
    dom_params = params['geometry'][dom_type]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(dom_params)

    # load MHD equilibrium
    equil_params = params['fields']['mhd_equilibrium']
    mhd_equil_class = getattr(equils, equil_params['type'])
    mhd_equil = mhd_equil_class(equil_params[equil_params['type']])

    if equil_params['use_equil_domain']:
        assert mhd_equil.domain is not None
        domain = mhd_equil.domain
    else:
        mhd_equil.domain = domain

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
