import params_bump_on as damping_params

import os
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy import PostProcessor, PlottingData
from struphy.physics.physics import Units

# post process raw data
path = os.path.join(os.getcwd(), "sim_data")
pp = PostProcessor(path_out=path)
pp.process()

# get sim data
pdata = PlottingData(path_out=path)
pdata.load()

### Initial velocity distribution ###
v1_bins = pdata.f.kinetic_ions.v1_density.grid_v1
f_v1 = pdata.f.kinetic_ions.v1_density.f_binned

fig, ax = plt.subplots(1 ,figsize = (14,10))

ax.plot(v1_bins, f_v1[0])
ax.set_xlabel("Velocity v")
ax.set_ylabel("Distribution f(v)")
ax.set_title("Initial velocity distribution")
plt.tight_layout()
plt.show()

### Electric field progression ###
# get parameters
dt = damping_params.time_opts.dt
algo = damping_params.time_opts.split_algo
Nel = damping_params.grid.Nel
p = damping_params.derham_opts.p

env = damping_params.env
ppc = damping_params.loading_params.ppc

#get units
units = Units(damping_params.base_units)
model = damping_params.model
model.units = units
A_bulk = model.bulk_species.mass_number
Z_bulk = model.bulk_species.charge_number
model.units.derive_units(
    velocity_scale = model.velocity_scale,
    A_bulk = A_bulk,
    Z_bulk = Z_bulk
)
unit_t = model.units.t

# get scalar data (post processing not needed for scalar data)
if MPI.COMM_WORLD.Get_rank() == 0:
    pa_data = os.path.join(env.path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]*unit_t
        E = f["scalar"]["en_E"][()]

    # plot
    plt.figure(figsize=(18, 12))
    plt.plot(time, E, label="numerical")
    plt.legend()
    plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("electric energy $E^2/2$ [a.u.]")

    plt.show()      

### Binning distribution progression ###      
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