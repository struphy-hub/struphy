from struphy.io.options import Units, Time, MetaOptions
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.initial import perturbations
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.kinetic_background import maxwellians
from struphy import main

# import model, set verbosity
from struphy.models.toy import Maxwell as Model
verbose = False

# meta options
meta = MetaOptions()

# units
units = Units()

# time stepping
time = Time()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid()

# derham options
derham = DerhamOptions()

# light-weight model instance
model = Model()

# propagator options
model.propagators.maxwell.set_options()

# initial conditions (background + perturbation)
model.em_fields.b_field.add_background(FieldsBackground())
model.em_fields.b_field.add_perturbation(perturbations.TorusModesCos())

# optional: exclude variables from saving
# model.em_fields.b_field.save_data = False

# start run
main.main(model, 
          params_path=__file__, 
          units=units, 
          time_opts=time, 
          domain=domain, 
          equil=equil, 
          grid=grid, 
          derham=derham, 
          meta=meta, 
          verbose=verbose, 
          )