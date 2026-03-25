from params_diocotron import *
from struphy import PlottingData, PostProcessor

import os
import cunumpy as xp
from matplotlib import pyplot as plt
import h5py


# ------------------
# Post process simulation data
# ------------------
def main():
    sim_path = os.path.join(os.getcwd(), "simdata")

    pp = PostProcessor(path_out=sim_path)
    pp.process(physical=True)

    pdata = PlottingData(path_out=sim_path)
    pdata.load()

    # path to save plots
    save_path = os.path.join(os.getcwd(), "images", "sim")
    os.makedirs(save_path, exist_ok=True)

    # ------------------
    # Check simulation domain
    # ------------------

    domain.show(save_dir=os.path.join(save_path,"domain.png"))


    # ------------------
    # Determine electrical potentail growth rate
    # ------------------

    # get scalar data (post processing not needed for scalar data)
    pa_data = os.path.join(env.path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        en_phi = f["scalar"]["en_phi"][()]

    # determine growth rate
    exp_func = lambda x,m,b: 10**(m*x+b)

    # time interval to determine growth rate
    ti, tf = 26, 51
    xi = xp.abs(time - ti).argmin() + 1
    xf = xp.abs(time - tf).argmin() + 1

    fitting = xp.polyfit(time[xi:xf], xp.log10(en_phi[xi:xf]), deg=1)

    fig, ax = plt.subplots(1, figsize = (18, 12))

    # plot
    ax.plot(time, en_phi, label=r"$\phi$")
    ax.plot(
        pdata.t_grid, 
        exp_func(pdata.t_grid, *fitting), 
        label=f"fitted growth rate {ti=}, {tf=}, growth_rate={fitting[0]:.4e}"
    )
    ax.axvline(ti, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(tf, color="gray", linestyle="--", alpha=0.5)

    ax.set_yscale('log')
    ax.legend()

    ax.set_title(f"{time_opts.dt=}, {time_opts.split_algo=}, {grid.Nel=}, {derham_opts.p=}, {loading_params.ppc=}")
    ax.set_xlabel("time")
    ax.set_ylabel("Energy [a.u.]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "growth_rate.png"))
    plt.close()


    # ------------------
    # Show evolution of mass density distribution
    # ------------------

    nrows = 4
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

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
        plt.savefig(os.path.join(save_path, f"{bin_name}_{quantity}_phaseSpace"))
        plt.close()

    # e1_e2_density binplot in physical coordinate
    e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
    e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2

    phy_bin = domain(e1_bin, e2_bin, 0, squeeze_out=True) # convert eta to physical coordinate
    plot_phaseSpace(bin_name="e1_e2_density", quantity="f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)
    plot_phaseSpace(bin_name="e1_e2_density", quantity="delta_f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)

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
    plt.savefig(os.path.join(save_path, "potentialEvolution"))
    plt.close()


    # ------------------
    # Make video
    # ------------------

    def extract_images(bin_name, quantity, img_dir):
        """
        Extract images from each time step to be combined to video
        """
        from tqdm import tqdm
        # Save individual images

        os.makedirs(img_dir, exist_ok=True)# good compression

        e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
        e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2

        phy_bin = domain(e1_bin, e2_bin, 0, squeeze_out=True)
        Xs, Ys = phy_bin[0], phy_bin[1]

        import warnings
        warnings.filterwarnings(
            "ignore",
            message="The input coordinates to pcolor are interpreted as cell centers"
        )

        for idx in tqdm(range(len(pdata.t_grid))):
            time = pdata.t_grid[idx]

            fig, ax = plt.subplots(1, figsize=(8,6))

            #maxwellian distribution plot
            color_mapped = getattr(
                getattr(pdata.f.kinetic_ions, bin_name), quantity
                )[idx]
            pcm = ax.pcolor(Xs,Ys,color_mapped,vmin=0,vmax=2.5)

            fig.colorbar(pcm, ax=ax)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{quantity} at t = {pdata.t_grid[idx]:4.2e}")

            filename = os.path.join(img_dir, f"frame_{idx:05d}.jpg")

            plt.savefig(
                filename,
                dpi=100,              
                format="jpg",
            )
            plt.close(fig)

    extract_images("e1_e2_density", "f_binned", os.path.join(save_path, "video"))

if __name__ == "__main__":
    main()