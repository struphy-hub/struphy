import importlib.util
from struphy import PlottingData, PostProcessor

import os
import sys
import cunumpy as xp
import scipy.optimize as sc
from matplotlib import pyplot as plt
import h5py

import logging
from struphy import set_logging_level
set_logging_level(logging.INFO)


# ------------------
# Post process simulation data
# In order to compare different simulations, execute this file as `python pproc_diocotron.py sim_1 sim_2 ...` 
# where `sim_1`, `sim_2`, etc. are the names of the simulation folders to be post-processed and plotted together.
# If only one argument, the 2D plots will be shown. If multiple arguments, only the growth rate plot will be shown.
# ------------------
def main():
    if len(sys.argv)>1 and __name__=="__main__":
        sim_names = sys.argv[1:]
    else:
        sim_names = ["sim_5"]
    en_phis = []
    times = []
    sls = []
    params_opts = []
    for i, sim_name in enumerate(sim_names):
        sim_path = os.path.join(os.getcwd(), sim_name)

        spec = importlib.util.spec_from_file_location("params", os.path.join(sim_path, "parameters.py"))
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)

        if not os.path.isdir(os.path.join(sim_path, "post_processing")):
            pp = PostProcessor(sim=params.sim)
            pp.process(physical=True)

        pdata = PlottingData(sim=params.sim)
        pdata.load()

        # ------------------
        # Determine electrical potentail growth rate
        # ------------------

        # get scalar data (post processing not needed for scalar data)
        pa_data = os.path.join(sim_path, "data")
        with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
            times.append(f["time"]["value"][()])
            en_phis.append(xp.power(f["scalar"]["en_phi"][()], 1.0))

        # time interval to determine growth rate
        ti, tf = 0.0, 42.0
        if tf>times[i][-1]: tf = times[i][-1]
        if ti>tf:
            ti = tf/2
        xi = xp.abs(pdata.t_grid - ti).argmin() # index of time 100 [a.lu.] (observed end of growth rate)
        xf = xp.abs(pdata.t_grid - tf).argmin() + 1 # index of time 200 [a.lu.] (observed end of growth rate)
        if xi==0:
            xi=1 # avoid including t=0 in fit

        sls.append(tuple([slice(xi, xf)]))

        # determine growth rate
        fitting_func = lambda x,m,b,c0: xp.exp(m*x+b)+c0
        jac_func = lambda x,m,b,c0: xp.array([x*xp.exp(m*x+b), xp.exp(m*x+b), xp.ones_like(x)]).transpose()

        params_opt, _ = sc.curve_fit(fitting_func, times[i][sls[i]], en_phis[i][sls[i]], p0=(1e-3, -5, en_phis[i][1]), jac=jac_func, maxfev=10000)#3.07e2
        params_opts.append(params_opt)

        logging.info(f"Fitted growth rate for {sim_name}: {params_opt[0]:.4e}")

    fig, ax = plt.subplots(1, figsize = (18, 12))
    for i in range(len(sim_names)):
        ax.scatter(times[i][1:], en_phis[i][1:], label=r"$\phi_{"+sim_names[i][4:]+r"}$", marker='x', s=0.05)
        ax.plot(
            times[i][sls[i]], 
            fitting_func(times[i][sls[i]], *params_opts[i]), 
            label=f"fitted growth rate {ti=}, {tf=}, growth_rate={params_opts[i][0]:.4e}, b={params_opts[i][1]:.4e}, c0={params_opts[i][2]:.4e}"
        )
    ax.axvline(ti, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(tf, color="gray", linestyle="--", alpha=0.5)

    #ax.set_yscale('log')
    ax.legend()

    ax.set_title(f"{params.time_opts.dt=}, {params.time_opts.split_algo=}, {params.grid.num_elements=}, {params.derham_opts.degree=}, {params.loading_params.ppc=}")
    ax.set_xlabel("time")
    ax.set_ylabel("Energy [a.u.]")

    plt.tight_layout()
    plt.show()
    if len(sim_names)>1:
        exit()
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
        plt.show()
        # plt.savefig(os.path.join(save_path, f"{bin_name}_{quantity}_phaseSpace"))
        # plt.close()

    # e1_e2_density binplot in physical coordinate
    e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
    e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2

    phy_bin = params.domain(e1_bin, e2_bin, 0, squeeze_out=True) # convert eta to physical coordinate
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
    plt.show()
    # plt.savefig(os.path.join(save_path, "potentialEvolution"))
    # plt.close()


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

        phy_bin = params.domain(e1_bin, e2_bin, 0, squeeze_out=True)
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

    # extract_images("e1_e2_density", "f_binned", os.path.join(save_path, "video"))
    save_video_pngs = False
    if save_video_pngs:
        if not os.path.exists(sim_path+"/video"):
            os.mkdir(sim_path+"/video")
        # create .png for video
        jump = 1
        fig = plt.figure(figsize=(8, 8))
        for n in range(ntime):
            if n % jump == 0:
                color_mapped = pdata.f.kinetic_ions.e1_e2_density.f_binned[n].T
                plt.pcolor(phy_bin[0], phy_bin[1], pdata.f.kinetic_ions.e1_e2_density.f_binned[n])
                
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.title(f"t = {pdata.t_grid[n]:4.2e}")
                plt.savefig(sim_path+"/video"+f"/fig_{n:04.0f}.png", transparent=False, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()