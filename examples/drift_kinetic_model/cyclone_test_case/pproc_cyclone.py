import importlib.util
import pyvista as pv
from struphy import PlottingData, PostProcessor

import os
import sys
import glob
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

    params.domain.show()

    # get scalar data (post processing not needed for scalar data)
    pa_data = os.path.join(sim_path, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        en_phi = f["scalar"]["phi_integral"][()]#"en_phi"
    
    t0, t1 = 0.0, time[-1]
    #m, b = xp.polyfit(time[t0:t1], xp.log(xp.sqrt(en_phi))[t0:t1], 1)


    def plot_markers_from_restart(max_markers=None):
        arrays = []
        for path in sorted(glob.glob(f"{sim_path}/data/data_proc*.hdf5")):
            with h5py.File(path, "r") as f:
                arr = f["restart"]["kinetic_ions"][0]
                if arr.ndim == 3:
                    arr = arr[-1]
                arrays.append(arr)

        markers = xp.concatenate(arrays, axis=0)


        active = markers[:, 33] >= -1e-5
        print(f"number of actif markers : {active.sum()}")

        markers = markers[active]

        imax = xp.argmax(markers[:,7])
        print("marker index =", imax)

        print("eta1 =", markers[imax,0])
        print("eta2 =", markers[imax,1])

        print("vpar =", markers[imax,3])
        print("mu   =", markers[imax,4])

        print("w    =", markers[imax,5])
        print("s0   =", markers[imax,6])
        print("w0   =", markers[imax,7])

        print("markers shape:", markers.shape, f"min teta:{markers[:,1].min()}, max teta:{markers[:,1].max()}")
        for i in range(34):
            print(f"{i=}, min = {markers[:,i].min()}, max = {markers[:,i].max()}")

        if max_markers is not None:
            markers = markers[:max_markers]

        # Colonnes logiques usuelles : eta1, eta2, eta3, v_parallel, mu, weight, ...
        eta1 = markers[:, 0]
        eta2 = markers[:, 1]

                # Colonnes Particles5D :
        # 0,1,2 = eta
        # 5 = poids control-variate w
        # 7 = poids total initial w0
        w_cv = markers[:, 5]
        w0 = markers[:, 7]

        print("global w_cv min/max/sum:", w_cv.min(), w_cv.max(), w_cv.sum())
        print("global w0   min/max/sum:", w0.min(), w0.max(), w0.sum())

        # Cellule suspecte : eta1 ~= 0.0664, eta2 ~= 0.9414
        target_eta1 = 0.06640625
        target_eta2 = 0.94140625
        n_bins = 128

        e1_edges = xp.linspace(0.0, 1.0, n_bins + 1)
        e2_edges = xp.linspace(0.0, 1.0, n_bins + 1)

        i0 = int(xp.searchsorted(e1_edges, target_eta1, side="right") - 1)
        j0 = int(xp.searchsorted(e2_edges, target_eta2, side="right") - 1)

        print("target bin indices:", i0, j0)
        print("target bin eta1 range:", e1_edges[i0], e1_edges[i0 + 1])
        print("target bin eta2 range:", e2_edges[j0], e2_edges[j0 + 1])

        def bin_stats(ii, jj, label):
            mask = (
                (eta1 >= e1_edges[ii]) & (eta1 < e1_edges[ii + 1])
                & (eta2 >= e2_edges[jj]) & (eta2 < e2_edges[jj + 1])
            )

            print("\n---", label, "bin", ii, jj, "---")
            print("N markers:", int(mask.sum()))

            if mask.sum() == 0:
                return

            print("eta1 min/max:", eta1[mask].min(), eta1[mask].max())
            print("eta2 min/max:", eta2[mask].min(), eta2[mask].max())

            print("w_cv sum/min/max/maxabs:",
                  w_cv[mask].sum(),
                  w_cv[mask].min(),
                  w_cv[mask].max(),
                  xp.max(xp.abs(w_cv[mask])))

            print("w0   sum/min/max/maxabs:",
                  w0[mask].sum(),
                  w0[mask].min(),
                  w0[mask].max(),
                  xp.max(xp.abs(w0[mask])))

            k_cv = xp.argmax(xp.abs(w_cv[mask]))
            k_w0 = xp.argmax(xp.abs(w0[mask]))

            local_idx = xp.where(mask)[0]

            print("dominant cv marker columns 0:8:",
                  markers[local_idx[k_cv], :8])
            print("dominant w0 marker columns 0:8:",
                  markers[local_idx[k_w0], :8])

        bin_stats(i0, j0, "target")

        # Voisinage 3x3 autour de la cellule suspecte
        for ii in range(max(0, i0 - 1), min(n_bins, i0 + 2)):
            for jj in range(max(0, j0 - 1), min(n_bins, j0 + 2)):
                if ii == i0 and jj == j0:
                    continue
                bin_stats(ii, jj, "neighbor")

        # Histogrammes comparables : w_cv et w0
        H_cv, _, _ = xp.histogram2d(
            eta1, eta2,
            bins=(n_bins, n_bins),
            range=((0.0, 1.0), (0.0, 1.0)),
            weights=w_cv,
        )

        H_w0, _, _ = xp.histogram2d(
            eta1, eta2,
            bins=(n_bins, n_bins),
            range=((0.0, 1.0), (0.0, 1.0)),
            weights=w0,
        )

        H_count, _, _ = xp.histogram2d(
            eta1, eta2,
            bins=(n_bins, n_bins),
            range=((0.0, 1.0), (0.0, 1.0)),
        )

        print("\nHistogram target values:")
        print("H_count target:", H_count[i0, j0])
        print("H_cv target:", H_cv[i0, j0])
        print("H_w0 target:", H_w0[i0, j0])
        print("H_cv maxabs/index:", xp.max(xp.abs(H_cv)), xp.unravel_index(xp.argmax(xp.abs(H_cv)), H_cv.shape))
        print("H_w0 max/index:", H_w0.max(), xp.unravel_index(xp.argmax(H_w0), H_w0.shape))

        plt.figure()
        plt.pcolormesh(e2_edges, e1_edges, H_cv, shading="auto")
        plt.scatter([target_eta2], [target_eta1], c="red", s=40)
        plt.xlabel("eta2")
        plt.ylabel("eta1")
        plt.title("manual histogram with control-variate weight w")
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.pcolormesh(e2_edges, e1_edges, H_w0, shading="auto")
        plt.scatter([target_eta2], [target_eta1], c="red", s=40)
        plt.xlabel("eta2")
        plt.ylabel("eta1")
        plt.title("manual histogram with total initial weight w0")
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.pcolormesh(e2_edges, e1_edges, H_count, shading="auto")
        plt.scatter([target_eta2], [target_eta1], c="red", s=40)
        plt.xlabel("eta2")
        plt.ylabel("eta1")
        plt.title("manual unweighted marker count")
        plt.colorbar()
        plt.show()


        theta = 2.0 * xp.pi * eta2
        rho = eta1

        x = rho * xp.cos(theta)
        y = rho * xp.sin(theta)

        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, s=0.15, alpha=0.35, linewidths=0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(r"$\eta_1 \cos(2\pi\eta_2)$")
        plt.ylabel(r"$\eta_1 \sin(2\pi\eta_2)$")
        plt.title(f"Markers in logical disk, at end of simulation")
        plt.tight_layout()
        plt.show()

    #plot_markers_from_restart()



    plt.figure()
    plt.plot(time, en_phi, label="simulated value")
    #plt.plot(time[t0:t1], xp.exp(time[t0:t1]* m + b), label=f"fit y=c*e^(mx) with {m=} and c={xp.exp(b)}")
    plt.legend()
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
                time_idx = time_indices[j + i*ncols]

                #maxwellian distribution plot
                f = getattr(
                    getattr(pdata.f.kinetic_ions, bin_name), quantity
                    )[time_idx]

                pcm = ax_maxwellian.plot(f[:,0])

                ax_maxwellian.set_xlabel(x_label)
                ax_maxwellian.set_ylabel(y_label)
                ax_maxwellian.set_title(f"t = {pdata.t_grid[time_idx]:4.2e}")
        fig.suptitle(quantity)
        plt.tight_layout()
        plt.show()

    plot_radial_density("e1_e2_density", "delta_f_binned")


    e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
    e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2
    X, Y, Z = params.domain(e1_bin, e2_bin, 0.0, squeeze_out=True)
    R = xp.sqrt(X**2 + Y**2)

    def plot_phaseSpace(bin_name, quantity, xs, ys, x_label = "x", y_label = "y", vmin=None, vmax=None, in_physical = False):

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
                    maxi_ind = xp.unravel_index(xp.argmax(color_mapped), color_mapped.shape)
                    print(color_mapped.shape, maxi_ind)
                    print(f"eta1 = {e1_bin[maxi_ind[0]]}, eta2 = {e2_bin[maxi_ind[1]]}, R = {R[maxi_ind]}, Z = {Z[maxi_ind]}")
                    pcm = ax_maxwellian.pcolor(xs, ys, color_mapped, vmin=vmin, vmax=vmax)

                    ax_maxwellian.set_xlabel(x_label)
                    ax_maxwellian.set_ylabel(y_label)
                    ax_maxwellian.set_title(f"t = {pdata.t_grid[time_idx]:4.2e}")
                    fig.colorbar(pcm, ax = ax_maxwellian)
            fig.suptitle(quantity)
            plt.tight_layout()
            plt.show()
    #print(f"evol : {xp.max([xp.abs(pdata.f.kinetic_ions.e1_e2_density.f_binned[i+1]-pdata.f.kinetic_ions.e1_e2_density.f_binned[i]) for i in range(ntime-1)])}")

    plt.figure()
    plt.imshow(pdata.f.kinetic_ions.e1_e2_density.delta_f_binned[-1], vmin=-1e-5, vmax=1e-5)
    plt.colorbar(label="R")
    plt.xlabel("eta2")
    plt.ylabel("eta1")
    plt.show()

    plot_phaseSpace(
        "e1_e2_density",
        "f_binned",
        xs=R,
        ys=Z,
        x_label="R",
        y_label="Z",
        in_physical=True
    )

    plot_phaseSpace(
        "e1_e2_density",
        "delta_f_binned",
        xs=R,
        ys=Z,
        x_label="R",
        y_label="Z",
        in_physical=True
    )   

    # ------------------
    # Show evolution of electric potential
    # ------------------
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,10), sharex=True, sharey=True)

    X = pdata.grids_phy[0][:, :, 0]
    Y = pdata.grids_phy[1][:, :, 0]
    Z = pdata.grids_phy[2][:, :, 0]
    R = xp.sqrt(X**2 + Y**2)
    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            phi = pdata.spline_values.em_fields.phi_phy.data[pdata.t_grid[time_idx]][0][:,:,0]

            pcm = ax_maxwellian.pcolormesh(R, Z, phi, shading='auto')

            ax_maxwellian.set_aspect("equal", adjustable="box")
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