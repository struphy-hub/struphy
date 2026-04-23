import params_two_stream as params

import os
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy.physics.physics import Units
from struphy import PostProcessor, PlottingData


def main():
    ### Electric field progression ###
    # get parameters
    dt = params.time_opts.dt
    algo = params.time_opts.split_algo
    num_elements = params.grid.num_elements
    degree = params.derham_opts.degree

    env = params.env
    ppc = params.loading_params.ppc

    # get units
    units = Units(params.base_units)
    model = params.model
    model.units = units
    A_bulk = model.bulk_species.mass_number
    Z_bulk = model.bulk_species.charge_number
    model.units.derive_units(
            velocity_scale=model.velocity_scale,
            A_bulk=A_bulk,
            Z_bulk=Z_bulk,
        )
    unit_t = model.units.t

    #analytical solution
    m, b = 0.2845/unit_t, -5.3 # 0.2845 is determined from m/c time unit
    analytical = lambda x, m=m, b=b: 10**(m*x+b)

    # get scalar data (post processing not needed for scalar data)
    if MPI.COMM_WORLD.Get_rank() == 0:
        pa_data = os.path.join(env.path_out, "data")
        with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
            time = f["time"]["value"][()]*unit_t
            E = f["scalar"]["electric_energy"][()]

        # plot
        plt.figure(figsize=(18, 12))
        plt.plot(time, E, label="numerical")
        plt.plot(time, analytical(time), label = fr"10^({m:.2e}·x {'+' if b > 0 else '-'} {abs(b):.2})", linestyle = "--", color = "black")
        plt.yscale("log")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {num_elements=}, {degree=}, {ppc=}")
        plt.xlabel("time [s]")
        plt.ylabel("electric energy $E^2/2$ [a.u.]")

        plt.show()
        
    ### Binning distribution progression ###        
    # post process raw data
    path = os.path.join(os.getcwd(), "sim_data")
    pp = PostProcessor(sim=params.sim)
    pp.process()

    # get sim data
    pdata = PlottingData(sim=params.sim)
    pdata.load()

    # plot in e1-v1
    e1_bins = pdata.f.kinetic_ions.e1_v1_density.grid_e1
    v1_bins = pdata.f.kinetic_ions.e1_v1_density.grid_v1

    nrows = 3
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_v1_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            #maxwellian distribution plot
            color_mapped = pdata.f.kinetic_ions.e1_v1_density.f_binned[time_idx].T
            pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

            ax_maxwellian.set_xlabel(r"$\eta_1$")
            ax_maxwellian.set_ylabel(r"$v_x$")
            ax_maxwellian.set_title(fr"full-$f$ at t = {pdata.t_grid[time_idx]*unit_t:4.2e} s")
            fig.colorbar(pcm, ax = ax_maxwellian)
            
    plt.tight_layout()
    plt.show()

    save_video_pngs = False
    if save_video_pngs:
        # create .png for video
        jump = 2
        fig = plt.figure(figsize=(8, 8))
        for n in range(ntime):
            if n % jump == 0:
                color_mapped = pdata.f.kinetic_ions.e1_v1_density.f_binned[n].T
                plt.pcolor(e1_bins, v1_bins, color_mapped)
                
                plt.xlabel("position [a.u.]")
                plt.ylabel("velocity [a.u.]")
                plt.title(fr"full-$f$ at t = {pdata.t_grid[n]*unit_t:4.2e} s")
                plt.savefig(f"video/fig_{n:04.0f}.png", transparent=False, bbox_inches='tight', pad_inches=0)
                
if __name__ == "__main__":
    main() 