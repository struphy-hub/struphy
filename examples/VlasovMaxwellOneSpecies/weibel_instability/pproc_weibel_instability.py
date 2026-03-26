import os

import cunumpy as xp
import h5py
import params_weibel_instability as params
from matplotlib import pyplot as plt

from feectools.ddm.mpi import mpi as MPI
from struphy import PlottingData, PostProcessor
from struphy.physics.physics import Units


def main():
    # post process raw data
    sim_name = params.env.sim_folder
    sim_path = os.path.join(os.getcwd(), sim_name)
    # save_path = os.path.join(os.getcwd(), "result", "noPerb", "controlVariate"+sim_name[-1])

    pp = PostProcessor(sim=params.sim)
    pp.process()

    # get sim data
    pdata = PlottingData(sim=params.sim)
    pdata.load()

    # get parameters
    dt = params.time_opts.dt
    Tend = params.time_opts.Tend
    algo = params.time_opts.split_algo
    num_elements = params.grid.num_elements
    degree = params.derham_opts.degree

    env = params.env
    ppc = params.loading_params.Np // 32 # 32 grid points

    #get units
    units = Units(params.base_units)
    model = params.model
    model.units = units
    A_bulk = model.bulk_species.mass_number
    Z_bulk = model.bulk_species.charge_number
    model.units.derive_units(
        velocity_scale = model.velocity_scale,
        A_bulk = A_bulk,
        Z_bulk = Z_bulk
    )
    unit_t = model.units.t

    control_variate = params.weights_params.control_variate
    split_algo = params.time_opts.split_algo

    # ------------------
    # progression of EM-field energy 
    # along different direction
    # ------------------

    # energy in EM-field along different directions
    phy_grid = pdata.grids_phy[0].shape

    Nt = pdata.t_grid
    unit_volume = xp.prod([1/(phy_grid[i] - 1) for i in range(len(phy_grid))])

    def field_energy(field) -> float:
        """
        Calculate totoal energy of field in space
        """

        energy_square = xp.sum(field ** 2)

        return energy_square * unit_volume / 2

    extract_field_energy_axes = lambda field: [
        xp.array([
            field_energy(getattr(pdata.spline_values.em_fields, field).data[t][i]) for t in Nt
        ]) for i in range(3)
    ]

    electric_energy = extract_field_energy_axes("e_field_log")
    magnetic_energy = extract_field_energy_axes("b_field_log")

    # plot
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,6), sharex = True)

    # plot of energy in EM-fields
    ax.plot(pdata.t_grid, electric_energy[0], label = r"|$E_1|^2$/2", color = "blue")
    ax.plot(pdata.t_grid, electric_energy[1], label = r"|$E_2|^2$/2", color = "green")
    ax.plot(pdata.t_grid, magnetic_energy[2], label = r"|$B_3|^2$/2", color = "red")

    # determine magnetic field growth rate
    exp_func = lambda x, m, b: 10**(m*x + b)

    ti = pdata.t_grid[-1]//5
    if ti == 0.0:
        tf = pdata.t_grid[-1]
    else:
        tf = 2*ti
    print(f"{ti = }, {tf = }")

    xi = xp.abs(pdata.t_grid - ti).argmin() + 1 # index of time 100 [a.lu.] (observed end of growth rate)
    xf = xp.abs(pdata.t_grid - tf).argmin() + 1 # index of time 200 [a.lu.] (observed end of growth rate)

    fitting = xp.polyfit(pdata.t_grid[xi:xf], xp.log10(magnetic_energy[2][xi:xf]), deg = 1)
    ax.plot(
        pdata.t_grid,
        exp_func(pdata.t_grid, *fitting),
        label="fitted growth rate\n" + fr"$10^{{{fitting[0]:.5f}x {fitting[1]:.0f}}}$",
        color="cyan"
    )

    ax.plot(
        pdata.t_grid,
        exp_func(pdata.t_grid, 0.02784, fitting[1]),
        label="analytical growth rate\n" + fr"$10^{{0.02784x {fitting[1]:.0f}}}$",
        color="cyan",
        ls="--",
        alpha=0.5
    )

    ax.set_title(f"Field energy: {split_algo=}, {ppc=}, {control_variate=}")
    ax.set_title("Energy in EM field")
    ax.set_ylabel("Energy [a.u.]")
    ax.set_xlabel("time")
    ax.set_ylim(1e-14,1e0)
    ax.set_xlim(0,Tend)
    ax.legend(ncol = 3)

    ax.set_yscale("log")
    ax.minorticks_on()

    fig.suptitle(f"VlasovMaxwellOneSpecies simulation:\n {control_variate=}, {ppc=}, {algo=}")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path,"E"))
    plt.show()

    # ------------------
    # Binning distribution evolution 
    # ------------------

    nrows = 5
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_v1_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    def plot_phaseSpace(bin, bin_name):
        bins = bin_name.split("_")[:-1]
        grid_1, grid_2, *_ = ["grid_" + s for s in bins]
        bins_1 = getattr(getattr(pdata.f.kinetic_ions, bin_name), grid_1)
        bins_2 = getattr(getattr(pdata.f.kinetic_ions, bin_name), grid_2)

        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
        for i in range(nrows):
            for j in range(ncols):
                ax_maxwellian = axs[i][j]
                time_idx = time_indices[j + i*ncols]

                #maxwellian distribution plot
                color_mapped = getattr(
                    getattr(pdata.f.kinetic_ions, bin_name), bin
                    )[time_idx].T
                pcm = ax_maxwellian.pcolor(bins_1, bins_2, color_mapped)

                ax_maxwellian.set_xlabel(bins[0])
                ax_maxwellian.set_ylabel(bins[1])
                ax_maxwellian.set_title(f"{bin} at t = {pdata.t_grid[time_idx]:4.2e}")
                fig.colorbar(pcm, ax = ax_maxwellian)
                
        plt.tight_layout()
        # plt.savefig(os.path.join(save_path, f"{bin_name}_{bin}_phaseSpace"))
        plt.show()
        plt.close()

    plot_phaseSpace("f_binned", bin_name="e1_v1_density")
    plot_phaseSpace("delta_f_binned", bin_name="e1_v1_density")
    plot_phaseSpace("f_binned",bin_name="v1_v2_density")
    plot_phaseSpace("delta_f_binned",bin_name="v1_v2_density")

    # ------------------
    # Plot EM-field of each time step 
    # ------------------

    def plot_EM_state(time_step: float, n_dim = 3):    
        nearest_key = pdata.t_grid[xp.abs(pdata.t_grid - time_step).argmin()]
        electric_field = pdata.spline_values.em_fields.e_field_log.data[nearest_key]
        magnetic_field = pdata.spline_values.em_fields.b_field_log.data[nearest_key]
        
        fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (8,6), sharex = True, sharey = True)

        for i in range(n_dim):
            axs[0,i].plot(pdata.grids_log[0], electric_field[i][:,0,0])
            axs[0,i].set_title(fr"$E_{i+1}$")

            axs[1,i].plot(pdata.grids_log[0], magnetic_field[i][:,0,0])
            axs[1,i].set_title(fr"$B_{i+1}$")
        
        axs[0,0].set_ylabel(r"Electric field value")
        axs[1,0].set_ylabel(r"Magnetic field value")
        axs[1,0].set_xlabel(r"$\eta_1$")
        axs[1,1].set_xlabel(r"$\eta_1$")
        axs[1,2].set_xlabel(r"$\eta_1$")

        axs[0,0].set_ylim(-5e-3, 5e-3)
        axs[1,0].set_ylim(-5e-3, 5e-3)

        fig.suptitle(f"EM-field at time step: {nearest_key:.2f}, {ppc=},{control_variate=}")

        # plt.savefig(os.path.join(save_path, "EM_state", f"{nearest_key:.2f}".replace(".", "_") + ".png"))
        plt.show()
        plt.close()

    # os.makedirs(os.path.join(save_path, "EM_state"), exist_ok=True)
    for t in xp.linspace(0, pdata.t_grid[-1], 2):
        plot_EM_state(t)

    # ------------------
    # Current density evolution
    # ------------------

    # current_density_path = os.path.join(save_path, "current_density")
    # os.makedirs(current_density_path,exist_ok=True)

    def current_1D(time: float):
        time_step = abs(pdata.t_grid - time).argmin()
        fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (9,9),sharey = True, sharex = True)

        for i in range(3):
            for j in range(3):

                e_bins = getattr(pdata.f.kinetic_ions, f"e{i+1}_current_{j+1}").f_binned[time_step]
                es = xp.linspace(0,1,e_bins.shape[0])

                ax[i,j].axhline(color = "red", alpha = 0.5)
                ax[i,j].plot(es, e_bins)

            ax[i,0].set_ylim(-0.01,0.01)

        for i in range(3): ax[i,0].set_ylabel(fr"$j_{i+1}$")
        for j in range(3): ax[2,j].set_xlabel(fr"$\eta_{ {j+1} }$")

        fig.suptitle(f"Current density at time {time:.2f}")

        plt.tight_layout()
        # plt.savefig(os.path.join(
        #     current_density_path,
        #     f"{time:.2f}".replace(".", "_") + ".png"
        # ))
        plt.show()
        plt.close()

    for t in xp.linspace(0, pdata.t_grid[-1], 2):
        current_1D(t)
        
if __name__ == "__main__":
    main()