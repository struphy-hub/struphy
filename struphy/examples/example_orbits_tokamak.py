import yaml
import h5py
import sys
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.fields_background.mhd_equil import equils


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
    domain = domain_class(**dom_params)

    # load MHD equilibrium
    equil_params = params['mhd_equilibrium']
    mhd_equil_class = getattr(equils, equil_params['type'])
    mhd_equil = mhd_equil_class(**equil_params[equil_params['type']])

    if equil_params['use_equil_domain']:
        assert mhd_equil.domain is not None
        domain = mhd_equil.domain
    else:
        mhd_equil.domain = domain

    file = h5py.File(sim_path + 'data_proc0.hdf5', 'r')
    grid_info = file['scalar'].attrs['grid_info']
    file.close()

    Nt = int(params['time']['Tend']/params['time']['dt'])
    Np = params['kinetic']['ions']['save_data']['n_markers']

    log_Nt = int(np.log10(Nt)) + 1

    pos = np.zeros((Nt + 1, Np, 3), dtype=float)

    # load orbits
    print('Load ion orbits ...')
    for n in tqdm(range(Nt + 1)):

        # load x, y, z coordinates
        pos[n] = np.loadtxt(sim_path + 'kinetic_data/ions/orbits/' + 'ions_{0:0{1}d}.txt'.format(n, log_Nt), delimiter=',')[:, 1:]

        # convert to R, y, z, coordinates
        pos[n, :, 0] = np.sqrt(pos[n, :, 0]**2 + pos[n, :, 1]**2)

    # plot results
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(16)

    # plot absolute value of magnetic field in poloidal plane
    plt.subplot(1, 2, 1)
    e1 = np.linspace(0., 1., 101)
    e2 = np.linspace(0., 1., 101)

    X = domain(e1, e2, 0.)

    e1[0] += 1e-5

    plt.contourf(X[0], X[2], mhd_equil.absB0(e1, e2, 0.), levels=51)

    
    for i in range(grid_info.shape[0]):

        e1 = np.linspace(grid_info[i, 0], grid_info[i, 1], int(
            grid_info[i, 2]) + 1)
        e2 = np.linspace(grid_info[i, 3], grid_info[i, 4], int(
            grid_info[i, 5]) + 1)

        X = domain(e1, e2, 0.)

        # plot xz-plane for torus mappings, xy-plane else
        if 'Torus' in domain.__class__.__name__ or domain.__class__.__name__ == 'GVECunit':
            co1, co2 = 0, 2
        else:
            co1, co2 = 0, 1

        # eta1-isolines
        for j in range(e1.size):
            if j == 0:
                if i == 0:
                    plt.plot(X[co1, j, :], X[co2, j, :], color='k', label='domain decomposition for ' + str(grid_info.shape[0]) + ' MPI processes')
                else:
                    plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            elif j == e1.size - 1:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            else:
                plt.plot(X[co1, j, :], X[co2, j, :], color='tab:blue', alpha=.25)

        # eta2-isolines
        for k in range(e2.size):
            if k == 0:
                plt.plot(X[co1, :, k], X[co2, :, k], color='k')
            elif k == e2.size - 1:
                plt.plot(X[co1, :, k], X[co2, :, k], color='k')
            else:
                plt.plot(X[co1, :, k], X[co2, :, k], color='tab:blue', alpha=.25)
    

    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.legend()
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$|\mathbf{B}|$ [T]')

    # plot domain and particle orbits
    plt.subplot(1, 2, 2)
    
    for i in range(grid_info.shape[0]):

        e1 = np.linspace(grid_info[i, 0], grid_info[i, 1], int(
            grid_info[i, 2]) + 1)
        e2 = np.linspace(grid_info[i, 3], grid_info[i, 4], int(
            grid_info[i, 5]) + 1)

        X = domain(e1, e2, 0.)

        # plot xz-plane for torus mappings, xy-plane else
        if 'Torus' in domain.__class__.__name__ or domain.__class__.__name__ == 'GVECunit':
            co1, co2 = 0, 2
        else:
            co1, co2 = 0, 1

        # eta1-isolines
        for j in range(e1.size):
            if e1[j] == 0.:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            elif e1[j] == 1.:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            else:
                plt.plot(X[co1, j, :], X[co2, j, :], color='tab:blue', alpha=.25)

        # eta2-isolines
        for k in range(e2.size):
            plt.plot(X[co1, :, k], X[co2, :, k], color='tab:blue', alpha=.25)
    
    if domain.kind_map < 10:
        plt.scatter(domain.cx[:, :, 0], domain.cy[:, :, 0], s=2, color='b')
    
    for i in range(pos.shape[1]):
        plt.scatter(pos[:, i, 0], pos[:, i, 2], s=2)
    
    plt.axis('equal')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Passing and trapped particles (co- and counter direction)')
    
    plt.show()
    
if __name__ == '__main__':
    main()
