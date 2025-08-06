from struphy.io.options import EnvironmentOptions, Units, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy import main

# import model, set verbosity
from struphy.models.toy import Maxwell as Model
verbose = True

# environment options
env = EnvironmentOptions()

# units
units = Units()

# time stepping
time_opts = Time()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid()

# derham options
derham_opts = DerhamOptions()

# light-weight model instance
model = Model()

# propagator options
model.propagators.maxwell.set_options()

# initial conditions (background + perturbation)
model.em_fields.b_field.add_background(FieldsBackground())
model.em_fields.b_field.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=0))
model.em_fields.b_field.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=1))
model.em_fields.b_field.add_perturbation(perturbations.TorusModesCos(given_in_basis='v', comp=2))

# optional: exclude variables from saving
# model.em_fields.b_field.save_data = False

if __name__ == "__main__":
    # start run
    main.run(model, 
                params_path=__file__, 
                env=env, 
                units=units, 
                time_opts=time_opts, 
                domain=domain, 
                equil=equil, 
                grid=grid, 
                derham_opts=derham_opts, 
                verbose=verbose, 
                )