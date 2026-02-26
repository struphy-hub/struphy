import os

import cunumpy as xp
import h5py
import params_gauss_error as damping_params
from matplotlib import pyplot as plt

from feectools.ddm.mpi import mpi as MPI
from struphy import PlottingData, PostProcessor
from struphy.physics.physics import Units

### Get Parameters ###

dt = damping_params.time_opts.dt
algo = damping_params.time_opts.split_algo
Nel = damping_params.grid.Nel
p = damping_params.derham_opts.p
control_variate = damping_params.weights_params.control_variate


env = damping_params.env
ppc = damping_params.loading_params.ppc

# get scalar data (post-processing not required)
if MPI.COMM_WORLD.Get_rank() == 0:
    pa_data = os.path.join(env.path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        gauss_error = f["scalar"]["gauss_error"][()]
        E_energy = f["scalar"]["en_E"][()]
        B_energy = f["scalar"]["en_B"][()]

    # plot
    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (10,6), sharex = True)

    axs[0].plot(time, gauss_error, color = "black")
    axs[0].set_title("Gauss law violation")
    axs[0].set_ylim(1e-1,1e4)
    axs[0].set_ylabel("residual [a.u.]")

    axs[1].plot(time, E_energy, label = r"$|E|^2/2$", color = "red")
    axs[1].plot(time, B_energy, label = r"$|B|^2/2$", color = "blue")
    axs[1].set_ylim(1e-27,1e9)
    axs[1].set_xlim(0,2)
    axs[1].set_title("Energy in EM field")
    axs[1].set_ylabel("Energy [a.u.]")
    axs[1].set_xlabel("time")

    for ax in axs: ax.grid(); ax.set_yscale("log"); ax.legend(loc = "lower right")

    fig.suptitle(f"VlasovMaxwellOneSpecies simulation:\n {control_variate=}, {ppc=}, {algo=}")
    plt.tight_layout()
    plt.savefig(f"gaussError_{control_variate=}")
    plt.show()