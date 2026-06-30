import importlib.util
import pyvista as pv
from struphy import PlottingData, PostProcessor

import os
import sys
import cunumpy as xp
from matplotlib import pyplot as plt
import h5py


# ------------------
# Post process simulation data
# ------------------
def main():
    if len(sys.argv)>1 and __name__=="__main__":
        sim_name = sys.argv[1]
    else:
        sim_name = "sim_1"
    sim_path = os.path.join(os.getcwd(), sim_name)

    spec = importlib.util.spec_from_file_location("params", os.path.join(sim_path, "parameters.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    if not os.path.isdir(os.path.join(sim_path, "post_processing")):
        pp = PostProcessor(sim=params.sim)
        pp.process(physical=True)

    pdata = PlottingData(sim=params.sim)
    pdata.load()

    equil_data = pv.read(os.path.join(sim_path, "geometry.vts"))

    # path to save plots
    # save_path = os.path.join(os.getcwd(), "images", "sim")
    # os.makedirs(save_path, exist_ok=True)

    # ------------------
    # Check simulation domain
    # ------------------

    #params.domain.show()

    # get scalar data (post processing not needed for scalar data)
    pa_data = os.path.join(sim_path, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        en_phi = f["scalar"]["phi_integral"][()]#"en_phi"
    
    t0, t1 = 700, 1000
    m, b = xp.polyfit(time[t0:t1], xp.log(xp.sqrt(en_phi))[t0:t1], 1)


    plt.figure(figsize=(6,4))
    plt.plot(time, xp.sqrt(en_phi), label="simulated value")
    plt.plot(time[t0:t1], xp.exp(time[t0:t1]* m + b), label=f"fit y=c*e^(mx) with {m=:.4e} and c={xp.exp(b):.4e}")
    plt.legend()
    plt.semilogy()
    plt.xlabel("time")
    plt.ylabel("$\mathcal{E}$(t)")
    plt.show()

    equil_dim_grid = equil_data.dimensions
    equil_grid = xp.reshape(equil_data.points, equil_dim_grid + (3,))
    equil_r = xp.sqrt(equil_grid[:,:,:,0]**2 + equil_grid[:,:,:,1]**2)
    equil_p0 = xp.reshape(equil_data.point_data["p0"], equil_dim_grid)
    if "n0" in equil_data.point_data:
         equil_n0 = xp.reshape(equil_data.point_data["n0"], equil_dim_grid)
    plt.figure()
    plt.title("radial distribution")
    plt.xlabel("r")
    plt.plot(equil_r[0,0,:], equil_p0[0,0,:], label="p0")
    if "n0" in equil_data.point_data:
        plt.plot(equil_r[0,0,:], equil_n0[0,0,:], label="n0")
        plt.plot(equil_r[0,0,:], equil_p0[0,0,:]/equil_n0[0,0,:], label="T0")
    plt.legend()
    plt.show()

    nrows = 4
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    def plot_radial_density(bin_name, quantity, x_label = "r", y_label = "density"):
        time_idx=0
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
        for i in range(nrows):
            for j in range(ncols):
                ax_maxwellian = axs[i][j]
                previous = time_idx
                time_idx = time_indices[j + i*ncols]

                #maxwellian distribution plot
                f = getattr(
                    getattr(pdata.f.kinetic_ions, bin_name), quantity
                    )[time_idx]
                f2 = getattr(
                    getattr(pdata.f.kinetic_ions, bin_name), quantity
                    )[previous]

                pcm = ax_maxwellian.plot(f[:,0])# - f2[:,0])

                ax_maxwellian.set_xlabel(x_label)
                ax_maxwellian.set_ylabel(y_label)
                ax_maxwellian.set_title(f"t = {pdata.t_grid[time_idx]:4.2e}")
        fig.suptitle(quantity)
        plt.tight_layout()
        plt.show()

    plot_radial_density("e1_e2_density", "delta_f_binned")

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

                    pcm = ax_maxwellian.pcolor(xs, ys, color_mapped, vmin=-1e-7, vmax=1e-7)

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
    plot_phaseSpace("e1_e2_density", "delta_f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)

    
    # ------------------
    # Show evolution of electric potential
    # ------------------
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,10), sharex=True, sharey=True)

    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            phi = pdata.spline_values.em_fields.phi_phy.data[pdata.t_grid[time_idx]][0][:,:,0]

            pcm = ax_maxwellian.pcolormesh(pdata.grids_phy[0][:,:,0], pdata.grids_phy[1][:,:,0], phi)

            ax_maxwellian.set_xlabel("x")
            ax_maxwellian.set_ylabel("y")
            ax_maxwellian.set_title(f"Electrical potential at t = {pdata.t_grid[time_idx]:4.2e}")

            fig.colorbar(pcm, ax=ax_maxwellian)

    plt.tight_layout()
    plt.show()

    save_video_pngs = False
    if save_video_pngs:
        if not os.path.exists(sim_path+"/video"):
            os.mkdir(sim_path+"/video")
        # create .png for video
        jump = 1
        fig = plt.figure(figsize=(8, 8))
        for n in range(50):
            if n % jump == 0:
                color_mapped = pdata.f.kinetic_ions.e1_e2_density.f_binned[n].T
                plt.pcolor(phy_bin[0], phy_bin[1], pdata.f.kinetic_ions.e1_e2_density.delta_f_binned[n])
                
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.title(f"t = {pdata.t_grid[n]:4.2e}")
                plt.savefig(sim_path+"/video"+f"/fig_{n:04.0f}.png", transparent=False, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    main()