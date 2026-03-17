import params_weak_Landau_damping as params

import os
import cunumpy as xp
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
    Nel = params.grid.Nel
    p = params.derham_opts.p

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

    def E_exact(t):
        eps = params.perturbation.amps[0]
        r = 0.3677
        omega_r = 1.4156
        omega_i = -0.1533
        phi = 0.5362
        return (4*eps*r*xp.exp(omega_i * t) * xp.cos(omega_r * t - phi))**2 * xp.pi

    # get scalar data (post processing not needed for scalar data)
    if MPI.COMM_WORLD.Get_rank() == 0:
        pa_data = os.path.join(env.path_out, "data")
        with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
            time = f["time"]["value"][()]*unit_t
            E = f["scalar"]["en_E"][()]
        logE = xp.log10(E)

        # find where time derivative of E is zero
        dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
        zeros = dEdt * xp.roll(dEdt, -1) < 0.0
        maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
        maxima = logE[1:-1][maxima_inds]
        t_maxima = time[1:-1][maxima_inds]

        # plot
        plt.figure(figsize=(18, 12))
        plt.plot(time, E, label="numerical")
        plt.plot(time, E_exact(time/unit_t), linestyle = "--", color = "black", label = "analytical")
        plt.yscale('log')
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
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

    nrows = 4
    ntime = len(pdata.f.kinetic_ions.e1_v1_density.f_binned) 
    time_indices = [int( i/(nrows-1) * (ntime - 1) ) for i in range(nrows)]

    fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (14,10), sharex=True, sharey=True)
    for index in range(nrows):
        ax_maxwellian, ax_perturbation = axs[index][0], axs[index][1]
        time_index = time_indices[index]
        ax_title = f"t = {pdata.t_grid[time_index]} ms"


        #maxwellian distribution plot
        color_mapped = pdata.f.kinetic_ions.e1_v1_density.f_binned[time_index].T
        pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

        ax_maxwellian.set_xlabel(r"$\eta_1$")
        ax_maxwellian.set_ylabel(r"$v_x$")
        ax_maxwellian.set_title(fr"full-$f$ at t = {pdata.t_grid[time_index]*unit_t:4.2e} s")
        fig.colorbar(pcm, ax = ax_maxwellian)

        #perturbation plot
        color_mapped = pdata.f.kinetic_ions.e1_v1_density.delta_f_binned[time_index].T
        pcm = ax_perturbation.pcolor(e1_bins, v1_bins, color_mapped)

        ax_perturbation.set_xlabel(r"$\eta_1$")
        ax_perturbation.set_ylabel(r"$v_x$")
        ax_perturbation.set_title(fr"$\delta f$ at t = {pdata.t_grid[time_index]*unit_t:4.2e} s")
        fig.colorbar(pcm, ax = ax_perturbation)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()