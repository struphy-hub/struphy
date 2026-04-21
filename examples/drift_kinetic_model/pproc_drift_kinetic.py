import importlib.util
from struphy import PlottingData, PostProcessor

import os
import cunumpy as xp
from matplotlib import pyplot as plt
import h5py


# ------------------
# Post process simulation data
# ------------------
def main():
    sim_name = "sim_4"
    sim_path = os.path.join(os.getcwd(), sim_name)

    spec = importlib.util.spec_from_file_location("params", os.path.join(sim_path, "parameters.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    pp = PostProcessor(sim=params.sim)
    pp.process(physical=True)

    pdata = PlottingData(sim=params.sim)
    pdata.load()

    # path to save plots
    # save_path = os.path.join(os.getcwd(), "images", "sim")
    # os.makedirs(save_path, exist_ok=True)

    # ------------------
    # Check simulation domain
    # ------------------

    #params.domain.show()

    # ------------------
    # Determine electrical potentail growth rate
    # ------------------

    # get scalar data (post processing not needed for scalar data)
    pa_data = os.path.join(sim_path, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        en_phi = f["scalar"]["en_phi"][()]
    
    print(time)
    print(en_phi)
    plt.figure()
    plt.plot(time, xp.sqrt(en_phi))
    plt.semilogy()
    plt.show()

    plt.figure()
    plt.plot(xp.sum(pdata.f.kinetic_ions.e1_e2_density.f_binned[-1,:,:], axis=1))
    plt.semilogy()
    plt.show()

    nrows = 4
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    def plot_radial_density(bin_name, quantity, x_label = "r", y_label = "density"):

            fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
            for i in range(nrows):
                for j in range(ncols):
                    ax_maxwellian = axs[i][j]
                    time_idx = time_indices[j + i*ncols]

                    #maxwellian distribution plot
                    f = getattr(
                        getattr(pdata.f.kinetic_ions, bin_name), quantity
                        )[time_idx]

                    pcm = ax_maxwellian.plot(xp.sum(f, axis=1))

                    ax_maxwellian.set_xlabel(x_label)
                    ax_maxwellian.set_ylabel(y_label)
                    ax_maxwellian.set_title(f"t = {pdata.t_grid[time_idx]:4.2e}")
            fig.suptitle(quantity)
            plt.tight_layout()
            plt.show()

    plot_radial_density("e1_e2_density", "f_binned")

    def plot_phaseSpace(bin_name, quantity, xs, ys, x_label = "x", y_label = "y", in_physical = False):

            fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
            for i in range(nrows):
                for j in range(ncols):
                    ax_maxwellian = axs[i][j]
                    time_idx = time_indices[j + i*ncols]

                    #maxwellian distribution plot
                    color_mapped = getattr(
                        getattr(pdata.f.kinetic_ions, bin_name), quantity
                        )[time_idx].T

                    if in_physical: color_mapped = color_mapped.T

                    pcm = ax_maxwellian.pcolor(xs, ys, color_mapped)

                    ax_maxwellian.set_xlabel(x_label)
                    ax_maxwellian.set_ylabel(y_label)
                    ax_maxwellian.set_title(f"t = {pdata.t_grid[time_idx]:4.2e}")
                    fig.colorbar(pcm, ax = ax_maxwellian)
            fig.suptitle(quantity)
            plt.tight_layout()
            plt.show()
    print(f"evol : {xp.max([xp.abs(pdata.f.kinetic_ions.e1_e2_density.f_binned[i+1]-pdata.f.kinetic_ions.e1_e2_density.f_binned[i]) for i in range(ntime-1)])}")
    e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
    e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2
    phy_bin = params.domain(e1_bin, e2_bin, 0, squeeze_out=True)
    plot_phaseSpace("e1_e2_density", "f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)


    # ------------------
    # Show evolution of electric potential
    # ------------------
    nrows = 4
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,10), sharex=True, sharey=True)

    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            phi = pdata.spline_values.em_fields.phi_phy.data[pdata.t_grid[time_idx]][0][:,:,0]

            pcm = ax_maxwellian.pcolormesh(pdata.grids_phy[0][:,:,0], pdata.grids_phy[1][:,:,0], phi)

            ax_maxwellian.set_xlabel("x")
            ax_maxwellian.set_ylabel(r"y")
            ax_maxwellian.set_title(f"Electrical potential at t = {pdata.t_grid[time_idx]:4.2e}")

            fig.colorbar(pcm, ax=ax_maxwellian)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()