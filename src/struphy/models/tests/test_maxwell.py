from mpi4py import MPI
import os
import numpy as np

from struphy.io.options import EnvironmentOptions, Units, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy import main
from struphy.diagnostics.diagn_tools import power_spectrum_2d

test_folder = os.path.join(os.getcwd(), "verification_tests")


def test_light_wave_1d(do_plot: bool = False):
    # import model, set verbosity
    from struphy.models.toy import Maxwell as Model
    verbose = True

    # environment options
    out_folders = os.path.join(test_folder, "maxwell")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="light_wave_1d")

    # units
    units = Units()

    # time stepping
    time_opts = Time(dt=0.05, Tend=50.0)

    # geometry
    domain = domains.Cuboid(r3=20.0)

    # fluid equilibrium (can be used as part of initial conditions)
    equil = None

    # grid
    grid = grids.TensorProductGrid(Nel=(1, 1, 128))

    # derham options
    derham_opts = DerhamOptions(p=(1, 1, 3))

    # light-weight model instance
    model = Model()

    # propagator options
    model.propagators.maxwell.set_options()

    # initial conditions (background + perturbation)
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=0))
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=1))

    # start run
    main.run(model, 
            params_path=None, 
            env=env, 
            units=units, 
            time_opts=time_opts, 
            domain=domain, 
            equil=equil, 
            grid=grid, 
            derham_opts=derham_opts, 
            verbose=verbose, 
            )
    
    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        main.pproc(env.path_out)
        
    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        simdata = main.load_data(env.path_out)

        # fft in (t, z) of first component of e_field on physical grid
        Ex_of_t = simdata.arrays["em_fields"]["e_field_log"]
        _1, _2, _3, coeffs = power_spectrum_2d(Ex_of_t,
                    "e_field_log", 
                    grids=simdata.grids_log,
                    grids_mapped=simdata.grids_phy,
                    component=0,
                    slice_at=[0, 0, None],
                    do_plot=do_plot,
                    disp_name='Maxwell1D',
                    fit_branches=1,
                    noise_level=0.5,
                    extr_order=10,
                    fit_degree=(1,),
        )
        
        # test
        assert np.abs(coeffs[0][0] - 1.0) < 0.02

        
if __name__ == "__main__":
    test_light_wave_1d(do_plot=True)