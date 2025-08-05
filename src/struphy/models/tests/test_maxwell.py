from mpi4py import MPI
import os

from struphy.io.options import EnvironmentOptions, Units, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy import main
from struphy.post_processing.post_processing_tools import create_femfields

test_folder = os.path.join(os.getcwd(), "verification_tests")


def test_light_wave_1d():
    # import model, set verbosity
    from struphy.models.toy import Maxwell as Model
    verbose = True

    # environment options
    out_folders = os.path.join(test_folder, "maxwell")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="light_wave_1d")

    # units
    units = Units()

    # time stepping
    time_opts = Time(dt=0.05, Tend=1.0 - 1e-6)

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

    # optional: exclude variables from saving
    # model.em_fields.b_field.save_data = False

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
        
        
if __name__ == "__main__":
    test_light_wave_1d()